package persistence

import (
	"context"
	"errors"
	"os"
	"path/filepath"
	"syscall"
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
		require.Error(t, writer.PersistStream(context.Background(), path, input, terminalStatus(nil)))
	}
}

func TestFilesystemPersistStreamAtomicallyReplacesFile(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, "chunk.bin")
	require.NoError(t, os.WriteFile(path, []byte("old"), 0600))

	writer := &FilesystemWriter{}
	require.NoError(t, writer.PersistStream(context.Background(), path, streamInput([]byte("new")), terminalStatus(nil)))
	contents, err := os.ReadFile(path)
	require.NoError(t, err)
	require.Equal(t, []byte("new"), contents)
	info, err := os.Stat(path)
	require.NoError(t, err)
	require.Equal(t, os.FileMode(0600), info.Mode().Perm())
	require.Empty(t, temporaryFiles(t, dir))
}

func TestFilesystemPersistStreamHonorsUmaskForNewFile(t *testing.T) {
	oldUmask := syscall.Umask(0077)
	t.Cleanup(func() { syscall.Umask(oldUmask) })

	dir := t.TempDir()
	path := filepath.Join(dir, "chunk.bin")
	require.NoError(t, (&FilesystemWriter{}).PersistStream(context.Background(), path, streamInput([]byte("new")), terminalStatus(nil)))
	info, err := os.Stat(path)
	require.NoError(t, err)
	require.Equal(t, os.FileMode(0600), info.Mode().Perm())
	require.Empty(t, temporaryFiles(t, dir))
}

func TestFilesystemPersistStreamPreservesPriorFileOnProducerFailure(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, "chunk.bin")
	require.NoError(t, os.WriteFile(path, []byte("old"), 0600))
	wantErr := errors.New("conversion failed")

	writer := &FilesystemWriter{}
	err := writer.PersistStream(context.Background(), path, streamInput([]byte("partial")), terminalStatus(wantErr))
	require.ErrorIs(t, err, wantErr)
	contents, readErr := os.ReadFile(path)
	require.NoError(t, readErr)
	require.Equal(t, []byte("old"), contents)
	info, statErr := os.Stat(path)
	require.NoError(t, statErr)
	require.Equal(t, os.FileMode(0600), info.Mode().Perm())
	require.Empty(t, temporaryFiles(t, dir))
}

func TestFilesystemPersistStreamRejectsMissingProducerStatus(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, "chunk.bin")
	missingStatus := make(chan error)
	close(missingStatus)

	err := (&FilesystemWriter{}).PersistStream(context.Background(), path, streamInput([]byte("partial")), missingStatus)
	require.ErrorIs(t, err, errMissingProducerStatus)
	require.NoFileExists(t, path)
	require.Empty(t, temporaryFiles(t, dir))
}

func temporaryFiles(t *testing.T, dir string) []string {
	t.Helper()
	files, err := filepath.Glob(filepath.Join(dir, ".*.tmp-*"))
	require.NoError(t, err)
	return files
}
