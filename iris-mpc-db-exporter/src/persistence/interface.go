package persistence

import (
	"context"
	"errors"
	"fmt"
	"strconv"
	"strings"
)

const (
	TimestampsFolder = "timestamps"
)

type Writer interface {
	Persist(ctx context.Context, path string, data []byte) error
	// PersistStream receives exactly one buffered terminal producer status after
	// inputChannel closes. A nil status means the complete stream was produced.
	PersistStream(ctx context.Context, path string, inputChannel <-chan []byte, producerStatus <-chan error) error
}

type Reader interface {
	GetTimeOfLastExport(ctx context.Context, exportPath string) (*int64, error)
}

var (
	errMissingProducerStatus        = errors.New("producer status channel closed without a value")
	ErrIncrementalExportUnsupported = errors.New("incremental export is unsupported after a generation marker")
)

func parseExportMarker(filename string) (timestamp int64, generation bool, recognized bool, err error) {
	parts := strings.Split(filename, "_")
	if len(parts) != 3 && len(parts) != 4 {
		return 0, false, false, nil
	}

	values := make([]int64, 3)
	for i := range values {
		values[i], err = strconv.ParseInt(parts[i], 10, 64)
		if err != nil || values[i] <= 0 {
			return 0, false, true, fmt.Errorf("invalid export marker %q numeric field %d", filename, i+1)
		}
	}

	if len(parts) == 3 {
		return values[0], false, true, nil
	}

	descriptor := strings.Split(parts[3], "-")
	if len(descriptor) != 3 || descriptor[0] != "v2" || descriptor[1] != "bin" || !isLowerHexID(descriptor[2]) {
		return 0, false, true, fmt.Errorf("invalid generation export marker %q", filename)
	}
	return values[0], true, true, nil
}

func isLowerHexID(id string) bool {
	if len(id) != 32 {
		return false
	}
	for _, char := range []byte(id) {
		if (char < '0' || char > '9') && (char < 'a' || char > 'f') {
			return false
		}
	}
	return true
}

func readProducerStatus(producerStatus <-chan error) error {
	err, ok := <-producerStatus
	if !ok {
		return errMissingProducerStatus
	}
	return err
}
