package commands

import (
	"context"
	"crypto/rand"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"math"
	"sort"
	"strings"
	"sync"
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

const safeSnapshotFormat = 3

type snapshotChunk struct {
	FirstID     int      `json:"first_id"`
	LastID      int      `json:"last_id"`
	Key         string   `json:"key"`
	SHA256      string   `json:"sha256"`
	SizeBytes   int64    `json:"size_bytes"`
	RecordCount int      `json:"record_count"`
	Epochs      []uint32 `json:"epochs"`
}

type snapshotManifest struct {
	FormatVersion int             `json:"format_version"`
	StoreID       string          `json:"store_id"`
	RowCount      int             `json:"row_count"`
	RecordSize    int             `json:"record_size"`
	Epochs        []uint32        `json:"epochs"`
	Chunks        []snapshotChunk `json:"chunks"`
}

func validateRerandExportRequest(mode, storeID string, status iris.RerandStatus) error {
	if storeID != "" && mode != CompleteExport {
		return fmt.Errorf("rerandomization snapshots require an explicit %s", CompleteExport)
	}
	explicitSafeExport := mode == CompleteExport && storeID != ""
	if (status.Initialized || status.HasPositiveState) && !explicitSafeExport {
		return fmt.Errorf("rerandomization is initialized or positive; refusing legacy export publication")
	}
	return nil
}

func requiresLegacyExportRerandLock(mode, storeID string) bool {
	return mode != CompleteExport || storeID == ""
}

func newSnapshotID() (string, error) {
	var id [16]byte
	if _, err := rand.Read(id[:]); err != nil {
		return "", err
	}
	return hex.EncodeToString(id[:]), nil
}

func runCompleteExportCommand(ctx context.Context, path string, store iris.Store, converter converter.Converter, writer persistence.Writer, startIndex, endIndex, chanBufferLen int) (snapshotChunk, error) {
	start := time.Now()

	span, ctx := tracer.StartSpanFromContext(ctx, "command.complete_export.run")
	defer span.Finish()

	span.SetTag("startIndex", startIndex)
	span.SetTag("endIndex", endIndex)

	o11y.S(ctx).Infof("Starting %s from %d to %d", CompleteExport, startIndex, endIndex)

	irisesStream, err := store.StreamStoredIrisesByRange(ctx, startIndex, endIndex, chanBufferLen)
	if err != nil {
		return snapshotChunk{}, err
	}

	outputChannel := make(chan []byte, chanBufferLen)
	type conversionResult struct {
		rows   int
		size   int64
		hash   string
		err    error
		epochs map[uint32]struct{}
	}
	converted := make(chan conversionResult, 1)
	go func() {
		defer close(outputChannel)
		hasher := sha256.New()
		rows := 0
		var size int64
		epochs := make(map[uint32]struct{})
		for item := range irisesStream {
			expectedID := int64(startIndex + rows)
			if item.ID != expectedID {
				converted <- conversionResult{err: fmt.Errorf("non-contiguous export: expected iris %d, got %d", expectedID, item.ID)}
				return
			}
			convertedIrises, err := converter.ConvertSingle(item)
			if err != nil {
				converted <- conversionResult{err: err}
				return
			}
			_, _ = hasher.Write(convertedIrises)
			size += int64(len(convertedIrises))
			rows++
			epochs[uint32(item.RerandEpoch)] = struct{}{}
			outputChannel <- convertedIrises
		}
		if rows != endIndex-startIndex+1 {
			converted <- conversionResult{err: fmt.Errorf("incomplete export range %d-%d: got %d rows", startIndex, endIndex, rows)}
			return
		}
		converted <- conversionResult{rows: rows, size: size, hash: hex.EncodeToString(hasher.Sum(nil)), epochs: epochs}
	}()

	persistStart := time.Now()
	err = writer.PersistStream(ctx, path, outputChannel)
	if err != nil {
		return snapshotChunk{}, err
	}
	conversion := <-converted
	if conversion.err != nil {
		return snapshotChunk{}, conversion.err
	}
	elapsedPersist := time.Since(persistStart)
	o11y.S(ctx).Infof("Persisting irises in chunk from %d to %d took %f", startIndex, endIndex, elapsedPersist.Seconds())

	elapsed := time.Since(start)
	o11y.S(ctx).Infof("Processing chunk from %d to %d took %f", startIndex, endIndex, elapsed.Seconds())

	epochs := make([]uint32, 0, len(conversion.epochs))
	for epoch := range conversion.epochs {
		epochs = append(epochs, epoch)
	}
	sort.Slice(epochs, func(i, j int) bool { return epochs[i] < epochs[j] })
	return snapshotChunk{
		FirstID: startIndex, LastID: endIndex, Key: path,
		SHA256: conversion.hash, SizeBytes: conversion.size, RecordCount: conversion.rows,
		Epochs: epochs,
	}, nil
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

func ExportCommand(ctx context.Context, mode, outputFolder, storeID string, store iris.Store, converter converter.Converter, writer persistence.Writer, reader persistence.Reader, batchSize, parallelism, endIndex, chanBufferLen int) {
	if mode != CompleteExport && mode != IncrementalExport {
		o11y.S(ctx).Errorf("Invalid mode: %s", mode)
		return
	}
	if batchSize <= 0 || parallelism <= 0 {
		o11y.S(ctx).Error("batch size and parallelism must be positive")
		return
	}
	requestedMode := mode
	if requiresLegacyExportRerandLock(requestedMode, storeID) {
		releaseLock, err := store.TryLegacyExportRerandLock(ctx)
		if err != nil {
			o11y.S(ctx).With(zap.Error(err)).Error("Failed to fence legacy export from rerandomization")
			return
		}
		defer func() {
			releaseCtx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
			defer cancel()
			if err := releaseLock(releaseCtx); err != nil {
				o11y.S(ctx).With(zap.Error(err)).Error("Failed to release legacy-export rerandomization fence")
			}
		}()
	}
	rerandStatus, err := store.GetRerandStatus(ctx)
	if err != nil {
		o11y.S(ctx).With(zap.Error(err)).Error("Failed to determine database rerandomization state")
		return
	}
	if err := validateRerandExportRequest(requestedMode, storeID, rerandStatus); err != nil {
		o11y.S(ctx).With(zap.Error(err)).Error("Refusing unsafe export request")
		return
	}
	if requestedMode == CompleteExport && storeID != "" && endIndex != 0 {
		o11y.S(ctx).Error("Safe rerandomization snapshots must cover the complete database")
		return
	}

	startTime := time.Now()

	span, ctx := tracer.StartSpanFromContext(ctx, fmt.Sprintf("command.%s.run", strings.ToLower(mode)))
	defer span.Finish()

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

	o11y.S(ctx).Infof("Total irises: %d", totalIrises)

	batchesCountFloat := float64(totalIrises) / float64(batchSize)
	batchesCount := int(math.Ceil(batchesCountFloat))

	o11y.S(ctx).Infof("Will be processed in %d batches. \n", batchesCount)

	var exportNewerThan *int64
	if mode == IncrementalExport {
		exportNewerThan, err = reader.GetTimeOfLastExport(ctx, outputFolder)

		// if we failed to get the time of the last export, we will do a complete export
		if err != nil {
			mode = CompleteExport
		}
	}

	// Incremental exports remain in the legacy format and can never publish a
	// safe completion marker, including their no-checkpoint complete fallback.
	safeSnapshot := requestedMode == CompleteExport && mode == CompleteExport && storeID != ""
	var snapshotID string
	if safeSnapshot {
		if err = store.VerifyRerandStoreIdentity(ctx, storeID); err != nil {
			o11y.S(ctx).With(zap.Error(err)).Error("Refusing to publish snapshot for the wrong physical store")
			return
		}
		snapshotID, err = newSnapshotID()
		if err != nil {
			o11y.S(ctx).With(zap.Error(err)).Error("Failed to generate snapshot ID")
			return
		}
	}

	type batchResult struct {
		chunk snapshotChunk
		err   error
	}
	results := make(chan batchResult, batchesCount)
	semaphore := make(chan struct{}, parallelism)
	var wg sync.WaitGroup
	for i := 0; i < batchesCount; i++ {
		start := i*batchSize + 1
		last := min(start+batchSize-1, totalIrises)

		wg.Add(1)
		go func(batch, start, last int) {
			defer wg.Done()
			semaphore <- struct{}{}
			defer func() { <-semaphore }()

			var result batchResult
			if mode == CompleteExport {
				path := fmt.Sprintf("%s/%d.%s", outputFolder, start, converter.GetExtension())
				if safeSnapshot {
					path = fmt.Sprintf("%s/snapshots/data/%s/%d.%s", outputFolder, snapshotID, start, converter.GetExtension())
				}
				result.chunk, result.err = runCompleteExportCommand(ctx, path, store, converter, writer, start, last, chanBufferLen)
			}

			if mode == IncrementalExport {
				result.err = runIncrementalExportCommand(ctx, outputFolder, *exportNewerThan, store, converter, writer, start, last)
			}

			if result.err != nil {
				o11y.S(ctx).With(zap.Error(result.err)).Errorf("Failed to run export command on interval %d-%d", start, last)
			} else {
				o11y.S(ctx).Infof("Batch %d/%d completed", batch+1, batchesCount)
			}
			results <- result
		}(i, start, last)
	}

	wg.Wait()
	close(results)
	chunks := make([]snapshotChunk, 0, batchesCount)
	successfulBatches := 0
	exportSuccess := true
	for result := range results {
		if result.err != nil {
			exportSuccess = false
		} else {
			successfulBatches++
			if mode == CompleteExport {
				chunks = append(chunks, result.chunk)
			}
		}
	}
	exportDuration := time.Since(startTime)

	o11y.S(ctx).Infof("All tasks completed in %v. Successful batches: %d/%d", exportDuration, successfulBatches, batchesCount)

	metrics.MetricIncrement(ctx, "export_complete",
		[]string{
			fmt.Sprintf("successful:%t", exportSuccess),
			fmt.Sprintf("completion_time:%s", exportDuration),
		}, 1)

	if !exportSuccess {
		o11y.S(ctx).Error("Export did not fully succeed, skipping completion marker")
		return
	}

	if safeSnapshot {
		sort.Slice(chunks, func(i, j int) bool { return chunks[i].FirstID < chunks[j].FirstID })
		epochSet := make(map[uint32]struct{})
		for _, chunk := range chunks {
			for _, epoch := range chunk.Epochs {
				epochSet[epoch] = struct{}{}
			}
		}
		epochs := make([]uint32, 0, len(epochSet))
		for epoch := range epochSet {
			epochs = append(epochs, epoch)
		}
		sort.Slice(epochs, func(i, j int) bool { return epochs[i] < epochs[j] })
		manifest := snapshotManifest{
			FormatVersion: safeSnapshotFormat,
			StoreID:       storeID,
			RowCount:      totalIrises,
			RecordSize:    converter.GetRecordSize(),
			Epochs:        epochs,
			Chunks:        chunks,
		}
		manifestBytes, marshalErr := json.Marshal(manifest)
		if marshalErr != nil {
			o11y.S(ctx).With(zap.Error(marshalErr)).Error("Failed to encode snapshot manifest")
			return
		}
		digest := sha256.Sum256(manifestBytes)
		digestHex := hex.EncodeToString(digest[:])
		manifestPath := fmt.Sprintf("%s/snapshots/manifests/%s.json", outputFolder, digestHex)
		if err = writer.Persist(manifestPath, manifestBytes); err != nil {
			o11y.S(ctx).With(zap.Error(err)).Error("Failed to persist snapshot manifest")
			return
		}
		completionPath := fmt.Sprintf("%s/snapshots/complete/%d_%s_%d", outputFolder, time.Now().UnixNano(), digestHex, len(manifestBytes))
		if err = writer.Persist(completionPath, []byte{}); err != nil {
			o11y.S(ctx).With(zap.Error(err)).Error("Failed to persist safe snapshot completion marker")
		}
		return
	}

	// Legacy exports retain their timestamp checkpoint. Safe rerand snapshots
	// use a separate completion namespace and incremental runs never advance it.
	rerandStatus, err = store.GetRerandStatus(ctx)
	if err != nil {
		o11y.S(ctx).With(zap.Error(err)).Error("Failed to recheck database rerandomization state")
		return
	}
	if rerandStatus.Initialized || rerandStatus.HasPositiveState {
		o11y.S(ctx).Error("Rerandomization became initialized or positive; skipping legacy completion marker")
		return
	}
	timestampFilePath := fmt.Sprintf("%s/%s/%d_%d_%d", outputFolder, persistence.TimestampsFolder, time.Now().Unix(), batchSize, totalIrises)
	if err = writer.Persist(timestampFilePath, []byte{}); err != nil {
		o11y.S(ctx).With(zap.Error(err)).Error("Failed to persist export timestamp marker")
	}
}
