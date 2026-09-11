package persistence

import (
	"context"
	"errors"
)

const (
	TimestampsFolder = "timestamps"
)

type Writer interface {
	Persist(path string, data []byte) error
	// PersistStream receives exactly one buffered terminal producer status after
	// inputChannel closes. A nil status means the complete stream was produced.
	PersistStream(ctx context.Context, path string, inputChannel <-chan []byte, producerStatus <-chan error) error
}

type Reader interface {
	GetTimeOfLastExport(ctx context.Context, exportPath string) (*int64, error)
}

var errMissingProducerStatus = errors.New("producer status channel closed without a value")

func readProducerStatus(producerStatus <-chan error) error {
	err, ok := <-producerStatus
	if !ok {
		return errMissingProducerStatus
	}
	return err
}
