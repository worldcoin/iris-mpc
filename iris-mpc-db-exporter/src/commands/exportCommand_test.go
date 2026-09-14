package commands

import (
	"context"
	"errors"
	"fmt"
	"io"
	"path/filepath"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/DATA-DOG/go-sqlmock"
	"github.com/DataDog/datadog-go/v5/statsd"
	"github.com/stretchr/testify/require"

	"github.com/worldcoin/iris-mpc-db-exporter/src/config"
	"github.com/worldcoin/iris-mpc-db-exporter/src/iris"
	"github.com/worldcoin/iris-mpc-db-exporter/src/metrics"
	"github.com/worldcoin/iris-mpc-db-exporter/src/persistence"
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
	contexts   []context.Context
	markerCtxs []context.Context
	persistErr error
	streamErr  error
	markerErr  error
}

func (w *testWriter) Persist(ctx context.Context, path string, data []byte) error {
	w.mu.Lock()
	defer w.mu.Unlock()
	w.contexts = append(w.contexts, ctx)
	if strings.Contains(path, "/timestamps/") {
		w.markers = append(w.markers, path)
		w.markerCtxs = append(w.markerCtxs, ctx)
		return w.markerErr
	}
	if w.chunks == nil {
		w.chunks = make(map[string]string)
	}
	w.chunks[path] = string(data)
	return w.persistErr
}

func (w *testWriter) PersistStream(ctx context.Context, path string, input <-chan []byte, producerStatus <-chan error) error {
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
	return w.Persist(ctx, path, data)
}

type testReader struct{ err error }

func (r testReader) GetTimeOfLastExport(context.Context, string) (*int64, error) {
	if r.err != nil {
		return nil, r.err
	}
	timestamp := int64(123)
	return &timestamp, nil
}

func generationIDFromMarker(t *testing.T, marker string) string {
	t.Helper()
	parts := strings.Split(filepath.Base(marker), "_")
	require.Len(t, parts, 4)
	version := strings.Split(parts[3], "-")
	require.Len(t, version, 3)
	require.Equal(t, []string{"v2", "bin"}, version[:2])
	require.Regexp(t, `^[0-9a-f]{32}$`, version[2])
	return version[2]
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
				for path := range writer.chunks {
					require.Regexp(t, `^output/generations/[0-9a-f]{32}/(1|5)\.bin$`, path)
				}
			} else {
				require.NoError(t, err)
				require.Len(t, writer.markers, 1)
				generationID := generationIDFromMarker(t, writer.markers[0])
				require.Contains(t, writer.markers[0], "_2_5_v2-bin-")
				require.Equal(t, map[string]string{
					fmt.Sprintf("output/generations/%s/1.bin", generationID): "1,2,",
					fmt.Sprintf("output/generations/%s/3.bin", generationID): "3,4,",
					fmt.Sprintf("output/generations/%s/5.bin", generationID): "5,",
				}, writer.chunks)
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

func TestCompleteExportRejectsShortDatabaseRangeAndSuppressesMarker(t *testing.T) {
	store, mock := exportStore(t, 3)
	mock.ExpectQuery("SELECT id").WithArgs(1, 3).WillReturnRows(irisRows(1, 2)).RowsWillBeClosed()
	writer := &testWriter{}

	err := ExportCommand(context.Background(), CompleteExport, "output", store, testConverter{}, writer, testReader{}, 3, 1, 0, 0)
	require.ErrorContains(t, err, "returned 2 rows, expected 3")
	require.Empty(t, writer.chunks)
	require.Empty(t, writer.markers)
}

func TestCompleteExportInvocationsUseDistinctGenerations(t *testing.T) {
	ids := make(map[string]struct{})
	for range 2 {
		store, mock := exportStore(t, 1)
		mock.ExpectQuery("SELECT id").WithArgs(1, 1).WillReturnRows(irisRows(1)).RowsWillBeClosed()
		writer := &testWriter{}
		require.NoError(t, ExportCommand(context.Background(), CompleteExport, "output", store, testConverter{}, writer, testReader{}, 1, 1, 0, 0))
		require.Len(t, writer.markers, 1)
		ids[generationIDFromMarker(t, writer.markers[0])] = struct{}{}
	}
	require.Len(t, ids, 2)
}

func TestGenerationMarkersUseNanosecondOrdering(t *testing.T) {
	previousNow := exportNow
	exportNow = func() time.Time { return time.Unix(123, 456) }
	t.Cleanup(func() { exportNow = previousNow })

	completePlan, err := buildExportPlan(CompleteExport, "output", 100, 958, nil)
	require.NoError(t, err)
	require.Equal(t, "123000000456", strings.Split(filepath.Base(completePlan.markerPath), "_")[0])

	incrementalPlan, err := buildExportPlan(IncrementalExport, "output", 100, 958, new(int64))
	require.NoError(t, err)
	require.Equal(t, "123", strings.Split(filepath.Base(incrementalPlan.markerPath), "_")[0])
}

func TestGenerationRandomnessFailurePrecedesPersistence(t *testing.T) {
	previousRead := readGenerationRandom
	readGenerationRandom = func([]byte) (int, error) { return 0, errors.New("randomness unavailable") }
	t.Cleanup(func() { readGenerationRandom = previousRead })
	store, _ := exportStore(t, 1)
	writer := &testWriter{}

	err := ExportCommand(context.Background(), CompleteExport, "output", store, testConverter{}, writer, testReader{}, 1, 1, 0, 0)
	require.ErrorContains(t, err, "generate export generation id")
	require.Empty(t, writer.contexts)
	require.Empty(t, writer.chunks)
	require.Empty(t, writer.markers)
}

func TestGenuineIncrementalExportKeepsLegacyLayout(t *testing.T) {
	store, mock := exportStore(t, 1)
	mock.ExpectQuery("SELECT id").WithArgs(1, 1, int64(123)).WillReturnRows(irisRows(1)).RowsWillBeClosed()
	mock.ExpectQuery("SELECT id").WithArgs(1, 1).WillReturnRows(irisRows(1)).RowsWillBeClosed()
	writer := &testWriter{}

	require.NoError(t, ExportCommand(context.Background(), IncrementalExport, "output", store, testConverter{}, writer, testReader{}, 1, 1, 0, 0))
	require.Equal(t, map[string]string{"output/1_1.bin": "converted"}, writer.chunks)
	require.Len(t, writer.markers, 1)
	require.Len(t, strings.Split(filepath.Base(writer.markers[0]), "_"), 3)
}

func TestIncrementalFallbackUsesGenerationLayout(t *testing.T) {
	store, mock := exportStore(t, 1)
	mock.ExpectQuery("SELECT id").WithArgs(1, 1).WillReturnRows(irisRows(1)).RowsWillBeClosed()
	writer := &testWriter{}

	require.NoError(t, ExportCommand(context.Background(), IncrementalExport, "output", store, testConverter{}, writer, testReader{err: errors.New("no prior export")}, 1, 1, 0, 0))
	require.Len(t, writer.markers, 1)
	generationID := generationIDFromMarker(t, writer.markers[0])
	require.Equal(t, map[string]string{fmt.Sprintf("output/generations/%s/1.bin", generationID): "1,"}, writer.chunks)
}

func TestIncrementalAfterGenerationMarkerFallsBackToCompleteGeneration(t *testing.T) {
	store, mock := exportStore(t, 1)
	mock.ExpectQuery("SELECT id").WithArgs(1, 1).WillReturnRows(irisRows(1)).RowsWillBeClosed()
	writer := &testWriter{}

	require.NoError(t, ExportCommand(context.Background(), IncrementalExport, "output", store, testConverter{}, writer, testReader{err: persistence.ErrIncrementalExportUnsupported}, 1, 1, 0, 0))
	require.Len(t, writer.markers, 1)
	generationID := generationIDFromMarker(t, writer.markers[0])
	require.Equal(t, map[string]string{fmt.Sprintf("output/generations/%s/1.bin", generationID): "1,"}, writer.chunks)
}

func TestCompletionMarkerUsesCallerContext(t *testing.T) {
	type contextKey string
	ctx := context.WithValue(context.Background(), contextKey("request"), "test-request")
	store, mock := exportStore(t, 1)
	mock.ExpectQuery("SELECT id").WithArgs(1, 1).WillReturnRows(irisRows(1)).RowsWillBeClosed()
	writer := &testWriter{}

	require.NoError(t, ExportCommand(ctx, CompleteExport, "output", store, testConverter{}, writer, testReader{}, 1, 1, 0, 0))
	require.NotEmpty(t, writer.contexts)
	for _, persistedCtx := range writer.contexts {
		require.Equal(t, "test-request", persistedCtx.Value(contextKey("request")))
	}
	require.Len(t, writer.markerCtxs, 1)
	deadline, ok := writer.markerCtxs[0].Deadline()
	require.True(t, ok)
	require.WithinDuration(t, time.Now().Add(completionMarkerTimeout), deadline, time.Second)
}
