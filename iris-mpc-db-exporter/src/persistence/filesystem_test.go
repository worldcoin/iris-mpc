package persistence

import (
	"context"
	"os"
	"path/filepath"
	"testing"

	"github.com/stretchr/testify/require"
)

func TestFilesystemPersistenceErrors(t *testing.T) {
	writer := &FilesystemWriter{}
	file := filepath.Join(t.TempDir(), "existing-file")
	require.NoError(t, os.WriteFile(file, []byte("existing"), 0600))
	for _, path := range []string{filepath.Join(file, "child"), t.TempDir()} {
		require.Error(t, writer.Persist(path, nil))
		input := make(chan []byte)
		close(input)
		require.Error(t, writer.PersistStream(context.Background(), path, input))
	}
}
