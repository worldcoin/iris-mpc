package test

import (
	"context"
	"database/sql"
	"fmt"
	"testing"
	"time"

	"github.com/stretchr/testify/require"

	"github.com/worldcoin/iris-mpc-db-exporter/src/config"
	"github.com/worldcoin/iris-mpc-db-exporter/src/iris"
)

func TestPostgresRangeReadsAreOrderedByID(t *testing.T) {
	ctx := context.Background()
	cfg := config.Load()

	db, err := sql.Open("postgres", cfg.PgSQLConnectionString)
	require.NoError(t, err)
	t.Cleanup(func() {
		require.NoError(t, db.Close())
	})
	require.NoError(t, db.PingContext(ctx))

	schema := fmt.Sprintf("exporter_order_test_%d", time.Now().UnixNano())
	cfg.Environment = "CI"
	cfg.ForceOverrideSchemaName = true
	cfg.OverriddenSchemaName = schema
	store := iris.NewStore(ctx, db, cfg)

	t.Cleanup(func() {
		_, cleanupErr := db.ExecContext(context.Background(), fmt.Sprintf(`DROP SCHEMA IF EXISTS "%s" CASCADE;`, schema))
		require.NoError(t, cleanupErr)
	})

	_, err = db.ExecContext(ctx, fmt.Sprintf(`
		CREATE TABLE "%s".irises (
			id BIGINT NOT NULL,
			last_modified_at BIGINT,
			left_code BYTEA,
			left_mask BYTEA,
			right_code BYTEA,
			right_mask BYTEA,
			version_id SMALLINT
		);`, schema))
	require.NoError(t, err)

	insertQuery := fmt.Sprintf(`
		INSERT INTO "%s".irises
			(id, last_modified_at, left_code, left_mask, right_code, right_mask, version_id)
		VALUES ($1, $2, $3, $4, $5, $6, $7);`, schema)
	for _, id := range []int64{3, 1, 4, 2} {
		payload := []byte{byte(id)}
		_, err = db.ExecContext(ctx, insertQuery, id, int64(0), payload, payload, payload, payload, int16(0))
		require.NoError(t, err)
	}

	t.Run("buffered range", func(t *testing.T) {
		records, err := store.GetStoredIrisesByRange(ctx, 1, 4)
		require.NoError(t, err)

		ids := make([]int64, 0, len(records))
		for _, record := range records {
			ids = append(ids, record.ID)
		}
		require.Equal(t, []int64{1, 2, 3, 4}, ids)
	})

	t.Run("streaming range", func(t *testing.T) {
		records, err := store.StreamStoredIrisesByRange(ctx, 1, 4, 4)
		require.NoError(t, err)

		var ids []int64
		for record := range records {
			ids = append(ids, record.ID)
		}
		require.Equal(t, []int64{1, 2, 3, 4}, ids)
	})
}
