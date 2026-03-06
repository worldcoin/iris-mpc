package test

import (
	"context"
	"database/sql"
	"errors"
	"log"
	"syscall"
	"testing"

	"github.com/stretchr/testify/assert"
	"go.uber.org/zap"

	"github.com/worldcoin/iris-mpc-db-exporter/src/commands"
	"github.com/worldcoin/iris-mpc-db-exporter/src/config"
	"github.com/worldcoin/iris-mpc-db-exporter/src/iris"
	"github.com/worldcoin/iris-mpc-db-exporter/src/o11y"
)

const TestDataSize = 2_009

func TestDBPopulating(t *testing.T) {
	cfg := config.Load()
	ctx, cancelMainCtxFn := context.WithCancel(context.Background()) // mainCtx
	defer cancelMainCtxFn()

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

	log.Printf("DB_URL: %s, length: %d", cfg.PgSQLConnectionString, len(cfg.PgSQLConnectionString))

	// Connect to database
	db, err := sql.Open("postgres", cfg.PgSQLConnectionString)
	if err != nil {
		log.Fatal(err)
	}

	err = db.Ping()
	assert.NoError(t, err)

	store := iris.NewStore(ctx, db, cfg)
	err = store.DropTables(ctx)
	assert.NoError(t, err)

	commands.PopulateDbWithMockDataCommand(ctx, *store, TestDataSize)

	count, err := store.GetCount(ctx)
	assert.NoError(t, err)
	assert.Equal(t, count, TestDataSize)
}
