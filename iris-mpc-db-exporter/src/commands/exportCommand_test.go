package commands

import (
	"context"
	"errors"
	"fmt"
	"io"
	"strings"
	"sync"
	"testing"

	"github.com/DATA-DOG/go-sqlmock"
	"github.com/DataDog/datadog-go/v5/statsd"
	"github.com/stretchr/testify/require"

	"github.com/worldcoin/iris-mpc-db-exporter/src/config"
	"github.com/worldcoin/iris-mpc-db-exporter/src/iris"
	"github.com/worldcoin/iris-mpc-db-exporter/src/metrics"
)

type discardMetrics struct{ io.Writer }

func (discardMetrics) Close() error { return nil }

func exportStore(t *testing.T, count int) (iris.Store, sqlmock.Sqlmock) {
	t.Helper()
	db, mock, err := sqlmock.New()
	require.NoError(t, err)
	db.SetMaxIdleConns(3)
	mock.MatchExpectationsInOrder(false)
	mock.ExpectQuery(`SELECT count\(\*\)`).WillReturnRows(sqlmock.NewRows([]string{"count"}).AddRow(count)).RowsWillBeClosed()
	client, err := statsd.NewWithWriter(discardMetrics{io.Discard})
	require.NoError(t, err)
	previous := metrics.Client
	metrics.Client = client
	t.Cleanup(func() {
		require.NoError(t, mock.ExpectationsWereMet())
		for i := 0; i < db.Stats().OpenConnections; i++ {
			mock.ExpectClose()
		}
		require.NoError(t, db.Close())
		require.NoError(t, client.Close())
		metrics.Client = previous
	})
	return *iris.NewStore(context.Background(), db, config.Config{Environment: "test"}), mock
}

func irisRows(ids ...int) *sqlmock.Rows {
	rows := sqlmock.NewRows([]string{"id", "last_modified_at", "left_code", "left_mask", "right_code", "right_mask", "version_id"})
	for _, id := range ids {
		rows.AddRow(id, nil, nil, nil, nil, nil, 0)
	}
	return rows
}

type testConverter struct{ err error }

func (c testConverter) GetExtension() string { return "bin" }
func (c testConverter) ConvertSingle(item iris.StoredIris) ([]byte, error) {
	return []byte(fmt.Sprintf("%d,", item.ID)), c.err
}
func (c testConverter) Convert(items []iris.StoredIris) ([]byte, error) {
	return []byte("converted"), c.err
}

type testWriter struct {
	mu         sync.Mutex
	chunks     map[string]string
	markers    []string
	persistErr error
	streamErr  error
	markerErr  error
}

func (w *testWriter) Persist(path string, data []byte) error {
	w.mu.Lock()
	defer w.mu.Unlock()
	if strings.Contains(path, "/timestamps/") {
		w.markers = append(w.markers, path)
		return w.markerErr
	}
	if w.chunks == nil {
		w.chunks = make(map[string]string)
	}
	w.chunks[path] = string(data)
	return w.persistErr
}

func (w *testWriter) PersistStream(_ context.Context, path string, input <-chan []byte, producerStatus <-chan error) error {
	if w.streamErr != nil {
		return w.streamErr // Intentionally leaves producers blocked unless the caller drains.
	}
	var data []byte
	for item := range input {
		data = append(data, item...)
	}
	if err, ok := <-producerStatus; !ok {
		return errors.New("missing producer status")
	} else if err != nil {
		return err
	}
	return w.Persist(path, data)
}

type testReader struct{}

func (testReader) GetTimeOfLastExport(context.Context, string) (*int64, error) {
	timestamp := int64(123)
	return &timestamp, nil
}

func TestCompleteExportFailuresSuppressMarker(t *testing.T) {
	for _, failure := range []string{"query", "scan", "rows", "converter", "persistence"} {
		t.Run(failure, func(t *testing.T) {
			store, mock := exportStore(t, 3)
			wantErr := errors.New("batch failed")
			writer := &testWriter{}
			converter := testConverter{}
			query := mock.ExpectQuery("SELECT id").WithArgs(1, 3)
			rows := irisRows(1, 2, 3)
			switch failure {
			case "query":
				query.WillReturnError(wantErr)
			case "scan":
				rows = sqlmock.NewRows([]string{"id"}).AddRow("invalid")
			case "rows":
				rows.RowError(1, wantErr)
			case "converter":
				converter.err = wantErr
			case "persistence":
				writer.streamErr = wantErr
			}
			if failure != "query" {
				query.WillReturnRows(rows).RowsWillBeClosed()
			}
			err := ExportCommand(context.Background(), CompleteExport, "output", store, converter, writer, testReader{}, 3, 1, 0, 0)
			require.Error(t, err)
			if failure != "scan" {
				require.ErrorIs(t, err, wantErr)
			}
			require.Empty(t, writer.markers)
			require.Empty(t, writer.chunks)
		})
	}
}

func TestCompleteExportJoinsConversionAndDatabaseStreamErrors(t *testing.T) {
	store, mock := exportStore(t, 2)
	conversionErr := errors.New("conversion failed")
	streamErr := errors.New("database stream failed")
	rows := irisRows(1, 2)
	rows.RowError(1, streamErr)
	mock.ExpectQuery("SELECT id").WithArgs(1, 2).WillReturnRows(rows).RowsWillBeClosed()
	writer := &testWriter{}

	err := ExportCommand(context.Background(), CompleteExport, "output", store, testConverter{err: conversionErr}, writer, testReader{}, 2, 1, 0, 0)
	require.ErrorIs(t, err, conversionErr)
	require.ErrorIs(t, err, streamErr)
	require.Empty(t, writer.chunks)
	require.Empty(t, writer.markers)
}

func TestExportAllBatchesBeforeMarker(t *testing.T) {
	for _, failedBatch := range []bool{false, true} {
		t.Run(fmt.Sprintf("failed_batch_%t", failedBatch), func(t *testing.T) {
			store, mock := exportStore(t, 5)
			writer := &testWriter{}
			wantErr := errors.New("middle batch failed")
			mock.ExpectQuery("SELECT id").WithArgs(1, 2).WillReturnRows(irisRows(1, 2)).RowsWillBeClosed()
			middle := mock.ExpectQuery("SELECT id").WithArgs(3, 4)
			if failedBatch {
				middle.WillReturnError(wantErr)
			} else {
				middle.WillReturnRows(irisRows(3, 4)).RowsWillBeClosed()
			}
			mock.ExpectQuery("SELECT id").WithArgs(5, 5).WillReturnRows(irisRows(5)).RowsWillBeClosed()
			err := ExportCommand(context.Background(), CompleteExport, "output", store, testConverter{}, writer, testReader{}, 2, 3, 0, 0)
			if failedBatch {
				require.ErrorIs(t, err, wantErr)
				require.Empty(t, writer.markers)
				require.Len(t, writer.chunks, 2)
			} else {
				require.NoError(t, err)
				require.Len(t, writer.markers, 1)
				require.True(t, strings.HasSuffix(writer.markers[0], "_2_5"))
				require.Equal(t, map[string]string{"output/1.bin": "1,2,", "output/3.bin": "3,4,", "output/5.bin": "5,"}, writer.chunks)
			}
		})
	}
}

func TestMarkerFailureIsReturned(t *testing.T) {
	store, mock := exportStore(t, 1)
	mock.ExpectQuery("SELECT id").WithArgs(1, 1).WillReturnRows(irisRows(1)).RowsWillBeClosed()
	wantErr := errors.New("marker failed")
	writer := &testWriter{markerErr: wantErr}
	err := ExportCommand(context.Background(), CompleteExport, "output", store, testConverter{}, writer, testReader{}, 1, 1, 0, 0)
	require.ErrorIs(t, err, wantErr)
	require.Len(t, writer.markers, 1)
}

func TestEmptyExportDoesNotAdvanceCheckpoint(t *testing.T) {
	store, _ := exportStore(t, 0)
	writer := &testWriter{}
	require.NoError(t, ExportCommand(context.Background(), CompleteExport, "output", store, testConverter{}, writer, testReader{}, 1, 1, 0, 0))
	require.Empty(t, writer.chunks)
	require.Empty(t, writer.markers)
}

func TestCountFailureSuppressesMarker(t *testing.T) {
	db, mock, err := sqlmock.New()
	require.NoError(t, err)
	defer db.Close()
	wantErr := errors.New("count failed")
	mock.ExpectQuery(`SELECT count\(\*\)`).WillReturnError(wantErr)
	store := iris.NewStore(context.Background(), db, config.Config{Environment: "test"})
	writer := &testWriter{}
	err = ExportCommand(context.Background(), CompleteExport, "output", *store, testConverter{}, writer, testReader{}, 1, 1, 0, 0)
	require.ErrorIs(t, err, wantErr)
	require.Empty(t, writer.markers)
	require.NoError(t, mock.ExpectationsWereMet())
}

func TestIncrementalFailuresSuppressMarker(t *testing.T) {
	for _, failure := range []string{"query", "converter", "persistence"} {
		t.Run(failure, func(t *testing.T) {
			store, mock := exportStore(t, 1)
			wantErr := errors.New("incremental failed")
			writer := &testWriter{}
			converter := testConverter{}
			query := mock.ExpectQuery("SELECT id").WithArgs(1, 1, int64(123))
			if failure == "query" {
				query.WillReturnError(wantErr)
			} else {
				query.WillReturnRows(irisRows(1)).RowsWillBeClosed()
				mock.ExpectQuery("SELECT id").WithArgs(1, 1).WillReturnRows(irisRows(1)).RowsWillBeClosed()
				if failure == "converter" {
					converter.err = wantErr
				} else {
					writer.persistErr = wantErr
				}
			}
			err := ExportCommand(context.Background(), IncrementalExport, "output", store, converter, writer, testReader{}, 1, 1, 0, 0)
			require.ErrorIs(t, err, wantErr)
			require.Empty(t, writer.markers)
		})
	}
}
