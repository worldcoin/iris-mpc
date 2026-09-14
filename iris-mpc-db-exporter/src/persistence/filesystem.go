package persistence

import (
	"context"
	"crypto/rand"
	"encoding/hex"
	"errors"
	"fmt"
	"os"
	"path/filepath"
	"strconv"
	"strings"
)

type FilesystemWriter struct{}

func createTempFile(dir, base string, mode os.FileMode) (*os.File, error) {
	for range 100 {
		var suffix [8]byte
		if _, err := rand.Read(suffix[:]); err != nil {
			return nil, fmt.Errorf("generate temporary filename: %w", err)
		}
		path := filepath.Join(dir, "."+base+".tmp-"+hex.EncodeToString(suffix[:]))
		file, err := os.OpenFile(path, os.O_CREATE|os.O_EXCL|os.O_WRONLY, mode)
		if errors.Is(err, os.ErrExist) {
			continue
		}
		return file, err
	}
	return nil, errors.New("failed to allocate a unique temporary filename")
}

func (f *FilesystemWriter) Persist(path string, data []byte) error {
	// Ensure the directory exists
	dir := filepath.Dir(path)
	err := os.MkdirAll(dir, 0755)
	if err != nil {
		return fmt.Errorf("create directories for %s: %w", path, err)
	}

	err = os.WriteFile(path, data, 0644)
	if err != nil {
		return fmt.Errorf("write file %s: %w", path, err)
	}
	return nil
}

func (f *FilesystemWriter) PersistStream(ctx context.Context, path string, inputChannel <-chan []byte, producerStatus <-chan error) (resultErr error) {
	// Ensure the directory exists
	dir := filepath.Dir(path)
	err := os.MkdirAll(dir, 0755)
	if err != nil {
		return fmt.Errorf("failed to create directories: %w", err)
	}

	mode := os.FileMode(0644)
	preserveMode := false
	if info, statErr := os.Stat(path); statErr == nil {
		mode = info.Mode().Perm()
		preserveMode = true
	} else if !errors.Is(statErr, os.ErrNotExist) {
		return fmt.Errorf("stat existing file %s: %w", path, statErr)
	}

	file, err := createTempFile(dir, filepath.Base(path), mode)
	if err != nil {
		return fmt.Errorf("create temporary file for %s: %w", path, err)
	}
	tempPath := file.Name()
	fileClosed := false
	committed := false
	defer func() {
		if !fileClosed {
			if closeErr := file.Close(); closeErr != nil {
				resultErr = errors.Join(resultErr, fmt.Errorf("close temporary file for %s: %w", path, closeErr))
			}
		}
		if !committed {
			if removeErr := os.Remove(tempPath); removeErr != nil && !errors.Is(removeErr, os.ErrNotExist) {
				resultErr = errors.Join(resultErr, fmt.Errorf("remove temporary file for %s: %w", path, removeErr))
			}
		}
	}()
	if preserveMode {
		if err := file.Chmod(mode); err != nil {
			return fmt.Errorf("preserve permissions for %s: %w", path, err)
		}
	}

	// Write chunks as they arrive on the channel
	for {
		select {
		case <-ctx.Done():
			return ctx.Err() // context canceled or deadline exceeded
		case item, ok := <-inputChannel:
			if !ok {
				if err := readProducerStatus(producerStatus); err != nil {
					return fmt.Errorf("producer failed for %s: %w", path, err)
				}
				if err := file.Close(); err != nil {
					return fmt.Errorf("close temporary file for %s: %w", path, err)
				}
				fileClosed = true
				if err := os.Rename(tempPath, path); err != nil {
					return fmt.Errorf("replace file %s: %w", path, err)
				}
				committed = true
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
