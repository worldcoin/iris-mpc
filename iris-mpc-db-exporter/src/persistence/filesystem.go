package persistence

import (
	"context"
	"errors"
	"fmt"
	"log"
	"os"
	"path/filepath"
	"strconv"
	"strings"
)

type FilesystemWriter struct{}

func (f *FilesystemWriter) Persist(path string, data []byte) error {
	// Ensure the directory exists
	dir := filepath.Dir(path)
	err := os.MkdirAll(dir, 0755)
	if err != nil {
		log.Fatalf("Failed to create directories: %v", err)
	}

	err = os.WriteFile(path, data, 0644)
	if err != nil {
		log.Fatal(err)
	}
	return nil
}

func (f *FilesystemWriter) PersistStream(ctx context.Context, path string, inputChannel <-chan []byte) error {
	// Ensure the directory exists
	dir := filepath.Dir(path)
	err := os.MkdirAll(dir, 0755)
	if err != nil {
		return fmt.Errorf("failed to create directories: %w", err)
	}

	// Open (or create) the file for writing
	file, err := os.OpenFile(path, os.O_CREATE|os.O_WRONLY|os.O_TRUNC, 0644)
	if err != nil {
		return fmt.Errorf("failed to open file for writing: %w", err)
	}
	defer file.Close()

	// Write chunks as they arrive on the channel
	for {
		select {
		case <-ctx.Done():
			return ctx.Err() // context canceled or deadline exceeded
		case item, ok := <-inputChannel:
			if !ok {
				// channel closed; we're done receiving data
				return nil
			}
			if _, writeErr := file.Write(item); writeErr != nil {
				return fmt.Errorf("failed to write to file: %w", writeErr)
			}
		}
	}
}

type FilesystemReader struct{}

func (f *FilesystemReader) GetTimeOfLastExport(ctx context.Context, exportPath string) (*int64, error) {
	lastExportTime := int64(0)
	timestampsPrefix := fmt.Sprintf("%s/%s/", exportPath, TimestampsFolder)

	err := filepath.Walk(timestampsPrefix, func(path string, info os.FileInfo, err error) error {
		if err != nil {
			return err
		}
		// Check if the file has the desired extension
		if !info.IsDir() {
			fileName := filepath.Base(path)

			timestampStr := strings.Split(fileName, "_")[0]
			unixTime, err := strconv.ParseInt(timestampStr, 10, 64)
			if err != nil {
				return fmt.Errorf("failed to parse timestamp: %w", err)
			}
			lastExportTime = max(lastExportTime, unixTime)
		}
		return nil
	})

	if err != nil {
		return nil, err
	}

	if lastExportTime == 0 {
		return nil, errors.New("no exports found")
	}

	return &lastExportTime, nil
}
