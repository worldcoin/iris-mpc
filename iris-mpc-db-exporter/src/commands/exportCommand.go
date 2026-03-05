package commands

import (
	"context"
	"fmt"
	"math"
	"strings"
	"sync"
	"sync/atomic"
	"time"

	"go.uber.org/zap"
	"gopkg.in/DataDog/dd-trace-go.v1/ddtrace/tracer"

	"github.com/worldcoin/iris-mpc-db-exporter/src/converter"
	"github.com/worldcoin/iris-mpc-db-exporter/src/iris"
	"github.com/worldcoin/iris-mpc-db-exporter/src/metrics"
	"github.com/worldcoin/iris-mpc-db-exporter/src/o11y"
	"github.com/worldcoin/iris-mpc-db-exporter/src/persistence"
)

const CompleteExport = "COMPLETE_EXPORT"
const IncrementalExport = "INCREMENTAL_EXPORT"

func runCompleteExportCommand(ctx context.Context, mode, outputFolder string, store iris.Store, converter converter.Converter, writer persistence.Writer, startIndex, endIndex, chanBufferLen int) error {
	start := time.Now()

	span, ctx := tracer.StartSpanFromContext(ctx, fmt.Sprintf("command.%s.run", strings.ToLower(mode)))
	defer span.Finish()

	span.SetTag("startIndex", startIndex)
	span.SetTag("endIndex", endIndex)

	o11y.S(ctx).Infof("Starting %s from %d to %d", mode, startIndex, endIndex)

	irisesStream, err := store.StreamStoredIrisesByRange(ctx, startIndex, endIndex, chanBufferLen)
	if err != nil {
		return err
	}

	outputChannel := make(chan []byte, chanBufferLen)
	go func() {
		defer close(outputChannel)
		for item := range irisesStream {
			convertedIrises, err := converter.ConvertSingle(item)
			if err != nil {
				o11y.S(ctx).With(zap.Error(err)).Error("Failed to convert iris")
				panic(err)
			}
			outputChannel <- convertedIrises
		}
	}()

	path := fmt.Sprintf("%s/%d.%s", outputFolder, startIndex, converter.GetExtension())

	persistStart := time.Now()
	err = writer.PersistStream(ctx, path, outputChannel)
	if err != nil {
		return err
	}
	elapsedPersist := time.Since(persistStart)
	o11y.S(ctx).Infof("Persisting irises in chunk from %d to %d took %f", startIndex, endIndex, elapsedPersist.Seconds())

	elapsed := time.Since(start)
	o11y.S(ctx).Infof("Processing chunk from %d to %d took %f", startIndex, endIndex, elapsed.Seconds())

	return nil
}

func runIncrementalExportCommand(ctx context.Context, outputFolder string, exportOlderThan int64, store iris.Store, converter converter.Converter, writer persistence.Writer, startIndex, endIndex int) error {
	start := time.Now()

	span, ctx := tracer.StartSpanFromContext(ctx, "command.incremental_export.run")
	defer span.Finish()

	span.SetTag("startIndex", startIndex)
	span.SetTag("endIndex", endIndex)

	modifiedIrises, err := store.GetStoredIrisesOlderThanByRange(ctx, exportOlderThan, startIndex, endIndex)
	if err != nil {
		return err
	}

	if len(modifiedIrises) == 0 {
		o11y.S(ctx).Info("No irises to export")
		return nil
	}

	// now that we now we should re-export the chunk, let's get all the irises
	// this is not optimal, but it's the most straightforward way to ensure data consistency
	o11y.S(ctx).Infof("Starting incremental from %d to %d", startIndex, endIndex)

	fetchStart := time.Now()
	irises, err := store.GetStoredIrisesByRange(ctx, startIndex, endIndex)
	if err != nil {
		return err
	}

	elapsedFetch := time.Since(fetchStart)
	o11y.S(ctx).Infof("Fetching irises in chunk from %d to %d took %f", startIndex, endIndex, elapsedFetch.Seconds())

	convertedIrises, err := converter.Convert(irises)
	if err != nil {
		return err
	}

	path := fmt.Sprintf("%s/%d_%d.%s", outputFolder, startIndex, endIndex, converter.GetExtension())

	persistStart := time.Now()
	err = writer.Persist(path, convertedIrises)
	if err != nil {
		return err
	}
	elapsedPersist := time.Since(persistStart)
	o11y.S(ctx).Infof("Persisting irises in chunk from %d to %d took %f", startIndex, endIndex, elapsedPersist.Seconds())

	elapsed := time.Since(start)
	o11y.S(ctx).Infof("Processing chunk from %d to %d took %f", startIndex, endIndex, elapsed.Seconds())

	return nil
}

func ExportCommand(ctx context.Context, mode, outputFolder string, store iris.Store, converter converter.Converter, writer persistence.Writer, reader persistence.Reader, batchSize, parallelism, endIndex, chanBufferLen int) {
	if mode != CompleteExport && mode != IncrementalExport {
		o11y.S(ctx).Errorf("Invalid mode: %s", mode)
		return
	}

	startTime := time.Now()

	span, ctx := tracer.StartSpanFromContext(ctx, fmt.Sprintf("command.%s.run", strings.ToLower(mode)))
	defer span.Finish()

	var wg sync.WaitGroup
	totalIrises, err := store.GetCount(ctx)
	if err != nil {
		o11y.S(ctx).With(zap.Error(err)).Fatal("Error getting count of irises")
	}

	if totalIrises == 0 {
		o11y.S(ctx).Info("No irises to export")
		return
	}

	if endIndex != 0 {
		if endIndex < totalIrises {
			totalIrises = endIndex
			o11y.S(ctx).Infof("End index has been provided %d", totalIrises)
		} else {
			o11y.S(ctx).Infof("End index provided is greater than total irises %d", totalIrises)
			return
		}
	}

	// Get current time in unix format. It's stored as BIGINT in postgres
	unixTime := time.Now().Unix()
	timestampFilePath := fmt.Sprintf("%s/%s/%d_%d_%d", outputFolder, persistence.TimestampsFolder, unixTime, batchSize, totalIrises)

	o11y.S(ctx).Infof("Total irises: %d", totalIrises)

	batchesCountFloat := float64(totalIrises) / float64(batchSize)
	batchesCount := int(math.Ceil(batchesCountFloat))

	o11y.S(ctx).Infof("Will be processed in %d batches. \n", batchesCount)

	var runningCoroutines atomic.Int32
	var successfulBatches atomic.Int32

	var exportNewerThan *int64
	if mode == IncrementalExport {
		exportNewerThan, err = reader.GetTimeOfLastExport(ctx, outputFolder)

		// if we failed to get the time of the last export, we will do a complete export
		if err != nil {
			mode = CompleteExport
		}
	}

	for i := 0; i < batchesCount; i++ {
		start := i*batchSize + 1

		// if we are on the last batch, we need to adjust the batch size
		if i == batchesCount-1 {
			batchSize = totalIrises - start + 1
		}

		for runningCoroutines.Load() >= int32(parallelism) {
			o11y.S(ctx).Debug("Waiting for coroutines to finish")
			time.Sleep(time.Second)
		}

		wg.Add(1)
		runningCoroutines.Add(1)
		go func(start, count int) {
			defer wg.Done()
			success := true

			if mode == CompleteExport {
				err = runCompleteExportCommand(ctx, mode, outputFolder, store, converter, writer, start, start+count, chanBufferLen)
				if err != nil {
					o11y.S(ctx).With(zap.Error(err)).Errorf("Failed to run export command on interval %d-%d", start, start+count)
					success = false
				}
			}

			if mode == IncrementalExport {
				err = runIncrementalExportCommand(ctx, outputFolder, *exportNewerThan, store, converter, writer, start, start+count)
				if err != nil {
					o11y.S(ctx).With(zap.Error(err)).Errorf("Failed to run export command on interval %d-%d", start, start+count)
					success = false
				}
			}

			if err != nil {
				o11y.S(ctx).With(zap.Error(err)).Errorf("Failed to run export command on interval %d-%d", start, start+count)
				success = false
			} else {
				o11y.S(ctx).Infof("Batch %d/%d completed", i+1, batchesCount)
			}

			if success {
				successfulBatches.Add(1)
			}

			runningCoroutines.Add(-1)
		}(start, batchSize-1)
	}

	wg.Wait()
	exportSuccess := successfulBatches.Load() == int32(batchesCount)
	exportDuration := time.Since(startTime)

	o11y.S(ctx).Infof("All tasks completed in %v. Successful batches: %d/%d", exportDuration, successfulBatches.Load(), batchesCount)

	metrics.MetricIncrement(ctx, "export_complete",
		[]string{
			fmt.Sprintf("successful:%t", exportSuccess),
			fmt.Sprintf("completion_time:%s", exportDuration),
		}, 1)

	if !exportSuccess {
		o11y.S(ctx).Error("Export did not fully succeed, skipping timestamp marker to avoid advancing the checkpoint")
		return
	}

	// Create the file with the date of the beginning of the export to mark the completion of the export
	err = writer.Persist(timestampFilePath, []byte{})
}
