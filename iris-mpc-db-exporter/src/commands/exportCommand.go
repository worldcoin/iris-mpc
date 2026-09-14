package commands

import (
	"context"
	"crypto/rand"
	"encoding/hex"
	"errors"
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

const generationMarkerVersion = "v2-bin"
const completionMarkerTimeout = 30 * time.Second

var readGenerationRandom = rand.Read
var exportNow = time.Now

type exportPlan struct {
	mode          string
	chunkFolder   string
	markerPath    string
	batchSize     int
	totalIrises   int
	exportNewerAt *int64
}

func newGenerationID() (string, error) {
	var id [16]byte
	n, err := readGenerationRandom(id[:])
	if err != nil {
		return "", fmt.Errorf("generate export generation id: %w", err)
	}
	if n != len(id) {
		return "", fmt.Errorf("generate export generation id: read %d random bytes, expected %d", n, len(id))
	}
	return hex.EncodeToString(id[:]), nil
}

func buildExportPlan(mode, outputFolder string, batchSize, totalIrises int, exportNewerAt *int64) (exportPlan, error) {
	now := exportNow()
	markerTimestamp := now.Unix()
	plan := exportPlan{
		mode:          mode,
		chunkFolder:   outputFolder,
		batchSize:     batchSize,
		totalIrises:   totalIrises,
		exportNewerAt: exportNewerAt,
	}

	if mode == CompleteExport {
		markerTimestamp = now.UnixNano()
		generationID, err := newGenerationID()
		if err != nil {
			return exportPlan{}, err
		}
		plan.chunkFolder = fmt.Sprintf("%s/generations/%s", outputFolder, generationID)
		plan.markerPath = fmt.Sprintf("%s/%s/%d_%d_%d_%s-%s", outputFolder, persistence.TimestampsFolder, markerTimestamp, batchSize, totalIrises, generationMarkerVersion, generationID)
		return plan, nil
	}

	plan.markerPath = fmt.Sprintf("%s/%s/%d_%d_%d", outputFolder, persistence.TimestampsFolder, markerTimestamp, batchSize, totalIrises)
	return plan, nil
}

func runCompleteExportCommand(ctx context.Context, mode, outputFolder string, store iris.Store, converter converter.Converter, writer persistence.Writer, startIndex, endIndex, chanBufferLen int) error {
	start := time.Now()

	span, ctx := tracer.StartSpanFromContext(ctx, fmt.Sprintf("command.%s.run", strings.ToLower(mode)))
	defer span.Finish()

	span.SetTag("startIndex", startIndex)
	span.SetTag("endIndex", endIndex)

	o11y.S(ctx).Infof("Starting %s from %d to %d", mode, startIndex, endIndex)

	irisesStream, streamError, err := store.StreamStoredIrisesByRange(ctx, startIndex, endIndex, chanBufferLen)
	if err != nil {
		return err
	}

	outputChannel := make(chan []byte, chanBufferLen)
	producerStatus := make(chan error, 1)
	go func() {
		defer close(outputChannel)
		defer close(producerStatus)
		var conversionError error
		rowCount := 0
		for item := range irisesStream {
			rowCount++
			convertedIrises, err := converter.ConvertSingle(item)
			if err != nil {
				conversionError = fmt.Errorf("convert iris %d: %w", item.ID, err)
				// Let the database producer finish even when conversion stops.
				for range irisesStream {
				}
				break
			}
			outputChannel <- convertedIrises
		}
		streamErr, ok := <-streamError
		if !ok {
			streamErr = errors.New("database stream status channel closed without a value")
		}
		expectedRows := endIndex - startIndex + 1
		var rowCountErr error
		if rowCount != expectedRows {
			rowCountErr = fmt.Errorf("database stream returned %d rows, expected %d for range %d-%d", rowCount, expectedRows, startIndex, endIndex)
		}
		producerStatus <- errors.Join(conversionError, streamErr, rowCountErr)
	}()

	path := fmt.Sprintf("%s/%d.%s", outputFolder, startIndex, converter.GetExtension())

	persistStart := time.Now()
	err = writer.PersistStream(ctx, path, outputChannel, producerStatus)
	// Persistence may return before consuming the stream; unblock the producer.
	for range outputChannel {
	}
	// The writer consumes producerStatus after a complete input stream. If it
	// returned early, the caller owns the remaining status after draining.
	producerErr := <-producerStatus
	if err = errors.Join(err, producerErr); err != nil {
		return fmt.Errorf("export chunk %s: %w", path, err)
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
	err = writer.Persist(ctx, path, convertedIrises)
	if err != nil {
		return err
	}
	elapsedPersist := time.Since(persistStart)
	o11y.S(ctx).Infof("Persisting irises in chunk from %d to %d took %f", startIndex, endIndex, elapsedPersist.Seconds())

	elapsed := time.Since(start)
	o11y.S(ctx).Infof("Processing chunk from %d to %d took %f", startIndex, endIndex, elapsed.Seconds())

	return nil
}

func ExportCommand(ctx context.Context, mode, outputFolder string, store iris.Store, converter converter.Converter, writer persistence.Writer, reader persistence.Reader, batchSize, parallelism, endIndex, chanBufferLen int) error {
	if mode != CompleteExport && mode != IncrementalExport {
		return fmt.Errorf("invalid mode: %s", mode)
	}
	if batchSize <= 0 {
		return fmt.Errorf("batch size must be positive: %d", batchSize)
	}
	if parallelism <= 0 {
		return fmt.Errorf("parallelism must be positive: %d", parallelism)
	}
	if chanBufferLen < 0 {
		return fmt.Errorf("channel buffer length cannot be negative: %d", chanBufferLen)
	}

	startTime := time.Now()

	span, ctx := tracer.StartSpanFromContext(ctx, fmt.Sprintf("command.%s.run", strings.ToLower(mode)))
	defer span.Finish()

	var wg sync.WaitGroup
	totalIrises, err := store.GetCount(ctx)
	if err != nil {
		return fmt.Errorf("get iris count: %w", err)
	}

	if totalIrises == 0 {
		o11y.S(ctx).Info("No irises to export")
		return nil
	}

	if endIndex != 0 {
		if endIndex < totalIrises {
			totalIrises = endIndex
			o11y.S(ctx).Infof("End index has been provided %d", totalIrises)
		} else {
			o11y.S(ctx).Infof("End index provided is greater than total irises %d", totalIrises)
			return nil
		}
	}

	var exportNewerThan *int64
	if mode == IncrementalExport {
		exportNewerThan, err = reader.GetTimeOfLastExport(ctx, outputFolder)

		// if we failed to get the time of the last export, we will do a complete export
		if err != nil {
			o11y.S(ctx).With(zap.Error(err)).Warn("Incremental export unavailable; falling back to a complete generation")
			mode = CompleteExport
			exportNewerThan = nil
		}
	}

	plan, err := buildExportPlan(mode, outputFolder, batchSize, totalIrises, exportNewerThan)
	if err != nil {
		return err
	}

	o11y.S(ctx).Infof("Total irises: %d", totalIrises)

	batchesCountFloat := float64(totalIrises) / float64(plan.batchSize)
	batchesCount := int(math.Ceil(batchesCountFloat))

	o11y.S(ctx).Infof("Will be processed in %d batches. \n", batchesCount)

	var runningCoroutines atomic.Int32
	var successfulBatches atomic.Int32
	var firstBatchError error
	var recordBatchError sync.Once

	for i := 0; i < batchesCount; i++ {
		start := i*plan.batchSize + 1
		count := plan.batchSize

		// if we are on the last batch, we need to adjust the batch size
		if i == batchesCount-1 {
			count = totalIrises - start + 1
		}

		for runningCoroutines.Load() >= int32(parallelism) {
			o11y.S(ctx).Debug("Waiting for coroutines to finish")
			time.Sleep(time.Second)
		}

		wg.Add(1)
		runningCoroutines.Add(1)
		go func(start, count int) {
			defer wg.Done()
			defer runningCoroutines.Add(-1)
			var batchErr error

			if plan.mode == CompleteExport {
				batchErr = runCompleteExportCommand(ctx, plan.mode, plan.chunkFolder, store, converter, writer, start, start+count, chanBufferLen)
			}

			if plan.mode == IncrementalExport {
				batchErr = runIncrementalExportCommand(ctx, plan.chunkFolder, *plan.exportNewerAt, store, converter, writer, start, start+count)
			}

			if batchErr != nil {
				o11y.S(ctx).With(zap.Error(batchErr)).Errorf("Failed to run export command on interval %d-%d", start, start+count)
				recordBatchError.Do(func() {
					firstBatchError = fmt.Errorf("export interval %d-%d: %w", start, start+count, batchErr)
				})
			} else {
				o11y.S(ctx).Infof("Batch %d/%d completed", i+1, batchesCount)
				successfulBatches.Add(1)
			}
		}(start, count-1)
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
		return firstBatchError
	}

	// Create the file with the date of the beginning of the export to mark the completion of the export
	markerCtx, cancelMarker := context.WithTimeout(ctx, completionMarkerTimeout)
	defer cancelMarker()
	err = writer.Persist(markerCtx, plan.markerPath, []byte{})
	if err != nil {
		return fmt.Errorf("persist completion marker %s: %w", plan.markerPath, err)
	}
	return nil
}
