package test

import (
	"context"
	"database/sql"
	"encoding/binary"
	"errors"
	"fmt"
	"io"
	"io/fs"
	"os"
	"path/filepath"
	"strings"
	"syscall"
	"testing"

	"github.com/stretchr/testify/assert"
	"go.uber.org/zap"

	"github.com/worldcoin/iris-mpc-db-exporter/src/commands"
	"github.com/worldcoin/iris-mpc-db-exporter/src/config"
	"github.com/worldcoin/iris-mpc-db-exporter/src/converter"
	"github.com/worldcoin/iris-mpc-db-exporter/src/iris"
	"github.com/worldcoin/iris-mpc-db-exporter/src/o11y"
	"github.com/worldcoin/iris-mpc-db-exporter/src/persistence"
)

// getFilesWithExtension reads all files with a specific extension from the given directory.
func getFilesWithExtension(dir, extension string) ([]string, error) {
	var files []string

	err := filepath.WalkDir(dir, func(path string, d fs.DirEntry, err error) error {
		if err != nil {
			return err
		}
		// Check if it's a file and has the desired extension
		if !d.IsDir() && strings.HasSuffix(d.Name(), extension) {
			files = append(files, path)
		}
		return nil
	})

	return files, err
}

type BinaryReader struct {
	SingleRowTotalSize  int
	SingleCodeSize      int
	SingleMaskSize      int
	SingleIdSize        int
	SingleVersionIdSize int
}

func NewBinaryReader(codeSize, maskSize, idSize, versionSize int) *BinaryReader {
	return &BinaryReader{
		SingleCodeSize:      codeSize,
		SingleMaskSize:      maskSize,
		SingleIdSize:        idSize,
		SingleVersionIdSize: versionSize,
		SingleRowTotalSize:  (codeSize * 2) + (maskSize * 2) + idSize + versionSize,
	}
}

func (r *BinaryReader) Read(filename string) ([]iris.StoredIris, error) {
	file, err := os.Open(filename)
	if err != nil {
		return nil, fmt.Errorf("failed to open file: %w", err)
	}
	defer file.Close()

	var storedIrisList []iris.StoredIris

	for {
		buffer := make([]byte, r.SingleRowTotalSize)
		_, err := io.ReadFull(file, buffer)
		if err == io.EOF {
			break // End of file
		}
		if err != nil {
			return nil, fmt.Errorf("failed to read file: %w", err)
		}

		storedIris, err := r.parseRow(buffer)
		if err != nil {
			return nil, fmt.Errorf("failed to parse row: %w", err)
		}

		storedIrisList = append(storedIrisList, storedIris)
	}

	return storedIrisList, nil
}

func (r *BinaryReader) parseRow(encoded []byte) (iris.StoredIris, error) {
	// Basic validation: make sure the data length is what we expect
	if len(encoded) != r.SingleRowTotalSize {
		return iris.StoredIris{}, fmt.Errorf(
			"expected %d bytes, got %d",
			r.SingleRowTotalSize,
			len(encoded),
		)
	}

	var result iris.StoredIris

	// 1. Read the ID (first SingleIdSize bytes).
	//    We stored item.ID as BigEndian.Uint32, so reverse it the same way:
	result.ID = int64(binary.BigEndian.Uint32(encoded[0:r.SingleIdSize]))
	offset := r.SingleIdSize

	// 2. Undo storeAsEvenOddPairs for each “slice” we expect.
	//    You already know the lengths from your converter’s config:
	result.LeftCode = parseEvenOddPairs(
		encoded,
		offset,
		r.SingleCodeSize,
	)
	offset += r.SingleCodeSize

	result.LeftMask = parseEvenOddPairs(
		encoded,
		offset,
		r.SingleMaskSize,
	)
	offset += r.SingleMaskSize

	result.RightCode = parseEvenOddPairs(
		encoded,
		offset,
		r.SingleCodeSize,
	)
	offset += r.SingleCodeSize

	result.RightMask = parseEvenOddPairs(
		encoded,
		offset,
		r.SingleMaskSize,
	)
	offset += r.SingleMaskSize

	// 3. Read the version ID (last SingleVersionIdSize bytes).
	//    We stored item.VersionID as BigEndian.Uint16, so reverse it the same way:
	result.VersionID = int16(binary.BigEndian.Uint16(encoded[offset : offset+r.SingleVersionIdSize]))

	return result, nil
}

// parseEvenOddPairs reverses the transformation from storeAsEvenOddPairs.
// Recall storeAsEvenOddPairs does:
//
//	output = [ shiftEven0, shiftEven1, ..., shiftEvenN, shiftOdd0, shiftOdd1, ..., shiftOddN ]
//
// where shiftEven = (evenByte - 128) mod 256, etc.
//
// To reverse it, we take the first half of the block as the “shifted even” bytes,
// the second half as the “shifted odd” bytes, then add 128 back to each, and interleave.
func parseEvenOddPairs(data []byte, offset, length int) []byte {
	if length%2 != 0 {
		panic("the stored length must be even")
	}

	half := length / 2
	result := make([]byte, length)

	// The first `half` bytes are the "even" bytes (shifted by -128).
	// The second `half` bytes are the "odd" bytes (shifted by -128).
	for i := 0; i < half; i++ {
		// “shifted even” stored at data[offset + i]
		// “shifted odd”  stored at data[offset + half + i]
		e := data[offset+i]
		o := data[offset+half+i]

		// Undo the shift by +128 (Go wraps byte if it overflows, but  e+128  is safe).
		result[2*i] = e + 128
		result[2*i+1] = o + 128
	}
	return result
}

func TestDBExporting(t *testing.T) {
	cfg := config.Load()
	ctx := context.Background()

	// logging start
	o11y.ConfigureLogging()
	defer func() {
		err := zap.S().Sync()
		// Sync is not supported on os.Stderr / os.Stdout on all platforms.
		// See: https://github.com/uber-go/zap/issues/1093#issuecomment-1120667285
		if errors.Is(err, syscall.EINVAL) {
			err = nil
		}
		if err != nil {
			o11y.S(ctx).Error("error syncing zap logger", zap.Error(err))
		}
	}()

	ctx, cancelMainCtxFn := context.WithCancel(context.Background()) // mainCtx
	defer cancelMainCtxFn()

	// Connect to database
	o11y.S(ctx).Info("Connecting to the database...")
	db, err := sql.Open("postgres", cfg.PgSQLConnectionString)
	if err != nil {
		o11y.S(ctx).With(zap.Error(err)).Fatal("Error connecting to the database")
	}
	o11y.S(ctx).Info("Connected to the database.")

	store := iris.NewStore(ctx, db, cfg)

	var reader persistence.Reader
	var writer persistence.Writer

	reader = &persistence.FilesystemReader{}
	writer = &persistence.FilesystemWriter{}

	commands.ExportCommand(ctx, "COMPLETE_EXPORT", "test_output", *store, converter.NewBinaryConverter(cfg.SingleCodeSize, cfg.SingleMaskSize, 4, 2), writer, reader, 503, 2, 0, 200)

	files, err := getFilesWithExtension("test_output", ".bin")
	if err != nil {
		t.Fatalf("failed to get files in test_output with extension .bin: %v", err)
	}
	binaryReader := NewBinaryReader(cfg.SingleCodeSize, cfg.SingleMaskSize, 4, 2)

	var exportIrises []iris.StoredIris
	for _, file := range files {
		storedIrisList, err := binaryReader.Read(file)
		if err != nil {
			t.Fatalf("failed to read file: %v", err)
		}

		exportIrises = append(exportIrises, storedIrisList...)
	}

	assert.Equal(t, len(exportIrises), 2009)

	for _, exportItem := range exportIrises {
		dbItem, err := store.GetStoredIrisesByRange(ctx, int(exportItem.ID), int(exportItem.ID))
		assert.NoError(t, err)

		assert.Equal(t, dbItem[0].ID, exportItem.ID)
		assert.Equal(t, dbItem[0].LeftCode, exportItem.LeftCode)
		assert.Equal(t, dbItem[0].LeftMask, exportItem.LeftMask)
		assert.Equal(t, dbItem[0].RightCode, exportItem.RightCode)
		assert.Equal(t, dbItem[0].RightMask, exportItem.RightMask)
		assert.Equal(t, dbItem[0].VersionID, exportItem.VersionID)
	}
}
