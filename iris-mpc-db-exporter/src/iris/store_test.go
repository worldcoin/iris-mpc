package iris

import (
	"context"
	"errors"
	"testing"

	"github.com/DATA-DOG/go-sqlmock"
	"github.com/stretchr/testify/require"
)

func TestReadFailures(t *testing.T) {
	ctx := context.Background()
	for _, method := range []string{"count", "range", "older", "stream"} {
		for _, failure := range []string{"query", "scan", "rows"} {
			t.Run(method+"/"+failure, func(t *testing.T) {
				db, mock, err := sqlmock.New()
				require.NoError(t, err)
				t.Cleanup(func() { require.NoError(t, db.Close()) })
				store := &Store{db: db, schema: "test"}
				wantErr := errors.New("database read failed")
				rows := sqlmock.NewRows([]string{"id", "last_modified_at", "left_code", "left_mask", "right_code", "right_mask", "version_id"})
				if method == "count" {
					rows = sqlmock.NewRows([]string{"count"})
					if failure == "scan" {
						rows.AddRow("invalid count")
					} else {
						rows.AddRow(1)
					}
				} else if failure == "scan" {
					rows.AddRow("invalid id", nil, nil, nil, nil, nil, 0)
				} else {
					rows.AddRow(1, nil, nil, nil, nil, nil, 0)
				}
				query := mock.ExpectQuery("SELECT")
				if failure == "query" {
					query.WillReturnError(wantErr)
				} else {
					if failure == "rows" {
						rows.RowError(0, wantErr)
					}
					query.WillReturnRows(rows).RowsWillBeClosed()
				}

				switch method {
				case "count":
					_, err = store.GetCount(ctx)
				case "range":
					_, err = store.GetStoredIrisesByRange(ctx, 1, 2)
				case "older":
					_, err = store.GetStoredIrisesOlderThanByRange(ctx, 123, 1, 2)
				case "stream":
					var records <-chan StoredIris
					var streamError <-chan error
					records, streamError, err = store.StreamStoredIrisesByRange(ctx, 1, 2, 0)
					if err == nil {
						for range records {
						}
						err = <-streamError
					}
				}
				require.Error(t, err)
				if failure != "scan" {
					require.ErrorIs(t, err, wantErr)
				}
				require.NoError(t, mock.ExpectationsWereMet())
				mock.ExpectClose()
			})
		}
	}
}
