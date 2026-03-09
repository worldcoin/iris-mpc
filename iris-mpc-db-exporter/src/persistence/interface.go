package persistence

import "context"

const (
	TimestampsFolder = "timestamps"
)

type Writer interface {
	Persist(path string, data []byte) error
	PersistStream(ctx context.Context, path string, inputChannel <-chan []byte) error
}

type Reader interface {
	GetTimeOfLastExport(ctx context.Context, exportPath string) (*int64, error)
}
