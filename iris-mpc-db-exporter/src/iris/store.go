package iris

import (
	"context"
	"database/sql"
	"database/sql/driver"
	"fmt"

	"go.uber.org/zap"
	"gopkg.in/DataDog/dd-trace-go.v1/ddtrace/tracer"

	"github.com/worldcoin/iris-mpc-db-exporter/src/config"
	"github.com/worldcoin/iris-mpc-db-exporter/src/o11y"

	_ "github.com/lib/pq"
)

const (
	irisDbSchemaFormat          = "%s_%s_%d"
	rerandPassAdvisoryLockClass = int32(1381126734)
)

type StoredIris struct {
	ID             int64  `bson:"id"`               // BIGSERIAL
	LastModifiedAt *int64 `bson:"last_modified_at"` // BIGINT
	LeftCode       []byte `bson:"left_code"`        // BYTEA
	LeftMask       []byte `bson:"left_mask"`        // BYTEA
	RightCode      []byte `bson:"right_code"`       // BYTEA
	RightMask      []byte `bson:"right_mask"`       // BYTEA
	VersionID      int16  `bson:"version_id"`       // SMALLINT
	RerandEpoch    int32  `bson:"rerand_epoch"`     // INTEGER
	SemanticID     []byte `bson:"semantic_id"`      // UUID, 16 raw bytes
}

type StoredIrisStore interface {
	GetStoredIrisesByRange(startIndex, endIndex int) ([]StoredIris, error)
}

type Store struct {
	db                *sql.DB
	schema            string
	hasRerandMetadata bool
}

type RerandStatus struct {
	SchemaMigrated   bool
	Initialized      bool
	HasPositiveState bool
	StoreID          string
}

func discardSQLConn(conn *sql.Conn) {
	_ = conn.Raw(func(any) error { return driver.ErrBadConn })
	_ = conn.Close()
}

// TryLegacyExportRerandLock takes the same schema-scoped session advisory lock
// as try_rerand_pass_lock(). The returned release function owns a dedicated
// connection so the lock cannot move between pooled PostgreSQL sessions.
func (s *Store) TryLegacyExportRerandLock(ctx context.Context) (func(context.Context) error, error) {
	conn, err := s.db.Conn(ctx)
	if err != nil {
		return nil, fmt.Errorf("reserve connection for rerandomization lock: %w", err)
	}

	var schemaOID int32
	var acquired bool
	err = conn.QueryRowContext(ctx, `
		SELECT n.oid::integer,
		       pg_catalog.pg_try_advisory_lock($1, n.oid::integer)
		  FROM pg_catalog.pg_namespace AS n
		 WHERE n.nspname = $2`, rerandPassAdvisoryLockClass, s.schema).Scan(&schemaOID, &acquired)
	if err != nil {
		// The server evaluates the lock expression before Scan reports a
		// client-side conversion error, so conservatively discard the session.
		discardSQLConn(conn)
		return nil, fmt.Errorf("acquire legacy-export rerandomization lock: %w", err)
	}
	if !acquired {
		_ = conn.Close()
		return nil, fmt.Errorf("rerandomization pass or another legacy export is active")
	}

	released := false
	return func(releaseCtx context.Context) error {
		if released {
			return fmt.Errorf("legacy-export rerandomization lock already released")
		}
		released = true

		var unlocked bool
		unlockErr := conn.QueryRowContext(
			releaseCtx,
			`SELECT pg_catalog.pg_advisory_unlock($1, $2)`,
			rerandPassAdvisoryLockClass,
			schemaOID,
		).Scan(&unlocked)
		if unlockErr != nil {
			// A session lock must never be returned to the pool. Mark the
			// physical connection bad if an explicit unlock cannot be proved.
			discardSQLConn(conn)
			return fmt.Errorf("release legacy-export rerandomization lock: %w", unlockErr)
		}
		closeErr := conn.Close()
		if !unlocked {
			return fmt.Errorf("legacy-export rerandomization lock was not held")
		}
		if closeErr != nil {
			return fmt.Errorf("return legacy-export lock connection: %w", closeErr)
		}
		return nil
	}, nil
}

func NewStore(ctx context.Context, db *sql.DB, config config.Config) *Store {
	var schema string
	if config.ForceOverrideSchemaName {
		schema = config.OverriddenSchemaName
	} else {
		schema = fmt.Sprintf(irisDbSchemaFormat, config.SmpcSchemaName, config.Environment, config.NodeId)
	}

	// create schema if not exists - used for db populate in local environment
	o11y.S(ctx).Infof("ENV: %s", config.Environment)
	if config.Environment == "local" || config.Environment == "CI" {
		o11y.S(ctx).Infof("Creating schema %s if not exists in local environment", schema)
		createSchemaIfNotExistsCmd := fmt.Sprintf("CREATE SCHEMA IF NOT EXISTS \"%s\";", schema)
		_, err := db.Exec(createSchemaIfNotExistsCmd)
		if err != nil {
			o11y.S(ctx).With(zap.Error(err)).Fatalf("Failed to call create schema %s if not exists: %v", schema, err)
		}
	}

	return &Store{db: db, schema: schema}
}

func (s *Store) GetCount(ctx context.Context) (int, error) {
	span, ctx := tracer.StartSpanFromContext(ctx, "pgsql.get_count")
	defer span.Finish()

	query := fmt.Sprintf(`SELECT count(*) FROM "%s".irises;`, s.schema)
	rows, err := s.db.Query(query)

	if err != nil {
		o11y.S(ctx).With(zap.Error(err)).Error("Failed to fetch count")
		return -1, err
	}

	var count int
	for rows.Next() {
		if err := rows.Scan(&count); err != nil {
			o11y.S(ctx).With(zap.Error(err)).Error("Failed to scan count")
			return -1, err
		}
	}

	return count, nil
}

func (s *Store) VerifyRerandStoreIdentity(ctx context.Context, expected string) error {
	query := fmt.Sprintf(`SELECT store_id FROM "%s".get_rerand_store_state();`, s.schema)
	var actual sql.NullString
	if err := s.db.QueryRowContext(ctx, query).Scan(&actual); err != nil {
		return fmt.Errorf("read rerandomization store identity: %w", err)
	}
	if !actual.Valid || actual.String != expected {
		return fmt.Errorf("rerandomization store identity mismatch: expected %q, got %q", expected, actual.String)
	}
	return nil
}

// GetRerandStatus discovers rerandomization state from the database. The
// catalog probe keeps legacy exports compatible with schemas predating the
// rerandomization migration; once the control function exists, failures are
// returned so callers fail closed.
func (s *Store) GetRerandStatus(ctx context.Context) (RerandStatus, error) {
	var hasMetadata, hasControlFunction bool
	err := s.db.QueryRowContext(ctx, `
		SELECT
			EXISTS (
				SELECT 1 FROM information_schema.columns
				 WHERE table_schema = $1 AND table_name = 'irises'
				   AND column_name = 'rerand_epoch'
			),
			EXISTS (
				SELECT 1 FROM pg_catalog.pg_proc p
				JOIN pg_catalog.pg_namespace n ON n.oid = p.pronamespace
				 WHERE n.nspname = $1 AND p.proname = 'get_rerand_store_state'
				   AND p.pronargs = 0
			)`, s.schema).Scan(&hasMetadata, &hasControlFunction)
	if err != nil {
		return RerandStatus{}, fmt.Errorf("discover rerandomization schema: %w", err)
	}
	s.hasRerandMetadata = hasMetadata
	if hasControlFunction && !hasMetadata {
		return RerandStatus{}, fmt.Errorf("rerandomization control exists without row metadata")
	}
	if !hasMetadata {
		return RerandStatus{}, nil
	}

	if !hasControlFunction {
		positiveQuery := fmt.Sprintf(`SELECT EXISTS (
			SELECT 1 FROM "%s".irises WHERE rerand_epoch <> 0
		);`, s.schema)
		var hasPositiveRows bool
		if err := s.db.QueryRowContext(ctx, positiveQuery).Scan(&hasPositiveRows); err != nil {
			return RerandStatus{}, fmt.Errorf("read rerandomization row state: %w", err)
		}
		return RerandStatus{HasPositiveState: hasPositiveRows}, nil
	}

	stateQuery := fmt.Sprintf(`
		SELECT state.store_id, state.last_completed_epoch, state.active_epoch,
		       CASE WHEN state.store_id IS NULL
		                  AND state.last_completed_epoch = 0
		                  AND state.active_epoch IS NULL
		            THEN EXISTS (SELECT 1 FROM "%[1]s".irises WHERE rerand_epoch <> 0)
		            ELSE FALSE
		       END
		  FROM "%[1]s".get_rerand_store_state() AS state;`, s.schema)
	var storeID sql.NullString
	var lastCompleted int32
	var activeEpoch sql.NullInt32
	var hasPositiveRows bool
	if err := s.db.QueryRowContext(ctx, stateQuery).Scan(&storeID, &lastCompleted, &activeEpoch, &hasPositiveRows); err != nil {
		return RerandStatus{}, fmt.Errorf("read rerandomization control state: %w", err)
	}
	status := RerandStatus{SchemaMigrated: true}
	status.Initialized = storeID.Valid
	status.HasPositiveState = hasPositiveRows || lastCompleted != 0 || activeEpoch.Valid
	if storeID.Valid {
		status.StoreID = storeID.String
	}
	return status, nil
}

func (s *Store) irisSelectColumns() string {
	columns := "id, last_modified_at, left_code, left_mask, right_code, right_mask, version_id"
	if s.hasRerandMetadata {
		return columns + ", rerand_epoch, uuid_send(semantic_id)"
	}
	return columns + ", 0::integer AS rerand_epoch, NULL::bytea AS semantic_id"
}

func (s *Store) GetStoredIrisesByRange(ctx context.Context, startIndex, endIndex int) ([]StoredIris, error) {
	span, ctx := tracer.StartSpanFromContext(ctx, "pgsql.get_stored_irises_by_range")
	defer span.Finish()

	span.SetTag("startIndex", startIndex)
	span.SetTag("endIndex", endIndex)

	query := fmt.Sprintf(`SELECT %s FROM "%s".irises WHERE id >= $1 AND id <= $2 ORDER BY id;`, s.irisSelectColumns(), s.schema)
	rows, err := s.db.Query(query, startIndex, endIndex)
	if err != nil {
		o11y.S(ctx).With(zap.Error(err)).Errorf("Failed to fetch irises in range %d, %d. Error: %v", startIndex, endIndex, err)
		return nil, err
	}

	defer rows.Close()

	var irises []StoredIris

	for rows.Next() {
		var storedIris StoredIris

		if err := rows.Scan(&storedIris.ID, &storedIris.LastModifiedAt, &storedIris.LeftCode, &storedIris.LeftMask, &storedIris.RightCode, &storedIris.RightMask, &storedIris.VersionID, &storedIris.RerandEpoch, &storedIris.SemanticID); err != nil {
			o11y.S(ctx).With(zap.Error(err)).Error("Failed to populate iris")
			return nil, err
		}

		irises = append(irises, storedIris)
	}

	return irises, nil
}

func (s *Store) StreamStoredIrisesByRange(ctx context.Context, startIndex, endIndex, chanBufferLen int) (<-chan StoredIris, error) {
	span, ctx := tracer.StartSpanFromContext(ctx, "pgsql.get_stored_irises_by_range")
	defer span.Finish()

	span.SetTag("startIndex", startIndex)
	span.SetTag("endIndex", endIndex)
	outputChannel := make(chan StoredIris, chanBufferLen)

	query := fmt.Sprintf(`SELECT %s FROM "%s".irises WHERE id >= $1 AND id <= $2 ORDER BY id;`, s.irisSelectColumns(), s.schema)
	rows, err := s.db.Query(query, startIndex, endIndex)
	if err != nil {
		o11y.S(ctx).With(zap.Error(err)).Errorf("Failed to fetch irises in range %d, %d. Error: %v", startIndex, endIndex, err)
		return nil, err
	}

	go func() {
		defer rows.Close()
		defer close(outputChannel)
		for rows.Next() {
			var storedIris StoredIris
			if err := rows.Scan(&storedIris.ID, &storedIris.LastModifiedAt, &storedIris.LeftCode, &storedIris.LeftMask, &storedIris.RightCode, &storedIris.RightMask, &storedIris.VersionID, &storedIris.RerandEpoch, &storedIris.SemanticID); err != nil {
				o11y.S(ctx).With(zap.Error(err)).Error("Failed to populate iris")
				return
			}
			outputChannel <- storedIris
		}
	}()

	return outputChannel, nil
}

func (s *Store) GetStoredIrisesOlderThanByRange(ctx context.Context, lastModifiedAt int64, startIndex, endIndex int) ([]StoredIris, error) {
	span, ctx := tracer.StartSpanFromContext(ctx, "pgsql.get_stored_irises_by_range")
	defer span.Finish()

	span.SetTag("startIndex", startIndex)
	span.SetTag("endIndex", endIndex)

	query := fmt.Sprintf(`SELECT %s FROM "%s".irises WHERE (id >= $1 AND id <= $2 AND last_modified_at < $3) ORDER BY id;`, s.irisSelectColumns(), s.schema)
	rows, err := s.db.Query(query, startIndex, endIndex, lastModifiedAt)
	if err != nil {
		return nil, err
	}

	defer rows.Close()

	var irises []StoredIris

	for rows.Next() {
		var storedIris StoredIris

		if err := rows.Scan(&storedIris.ID, &storedIris.LastModifiedAt, &storedIris.LeftCode, &storedIris.LeftMask, &storedIris.RightCode, &storedIris.RightMask, &storedIris.VersionID, &storedIris.RerandEpoch, &storedIris.SemanticID); err != nil {
			o11y.S(ctx).With(zap.Error(err)).Error("Failed to populate iris")
			return nil, err
		}

		irises = append(irises, storedIris)
	}

	return irises, nil
}

func (s *Store) InsertIris(ctx context.Context, iris StoredIris) error {
	span, ctx := tracer.StartSpanFromContext(ctx, "pgsql.insert_stored_iris")
	defer span.Finish()

	// Prepare the SQL query
	query := fmt.Sprintf(`INSERT INTO "%s".irises (left_code, left_mask, right_code, right_mask, version_id) VALUES ($1, $2, $3, $4, $5);`, s.schema)

	// Execute the query
	_, err := s.db.ExecContext(ctx, query, iris.LeftCode, iris.LeftMask, iris.RightCode, iris.RightMask, iris.VersionID)
	if err != nil {
		o11y.S(ctx).With(zap.Error(err)).Error("Failed to insert iris")
		return err
	}

	return nil
}

func (s *Store) CreateTable(ctx context.Context) error {
	span, ctx := tracer.StartSpanFromContext(ctx, "pgsql.create_table")
	defer span.Finish()

	query := fmt.Sprintf(`
    CREATE TABLE IF NOT EXISTS "%s".irises (
        id BIGINT GENERATED BY DEFAULT AS IDENTITY (START WITH 1 MINVALUE 1) PRIMARY KEY,
        last_modified_at BIGINT DEFAULT EXTRACT(EPOCH FROM NOW())::BIGINT,
        left_code BYTEA,
        left_mask BYTEA,
        right_code BYTEA,
        right_mask BYTEA,
	    version_id SMALLINT DEFAULT 0 CHECK (version_id >= 0),
	    rerand_epoch INTEGER NOT NULL DEFAULT 0 CHECK (rerand_epoch >= 0),
	    semantic_id UUID
    );`, s.schema)

	_, err := s.db.ExecContext(ctx, query)
	if err != nil {
		o11y.S(ctx).With(zap.Error(err)).Fatal("Error creating irises table")
	}

	return err
}

func (s *Store) DropTables(ctx context.Context) error {
	span, ctx := tracer.StartSpanFromContext(ctx, "pgsql.drop_tables")
	defer span.Finish()

	query := fmt.Sprintf(`DROP TABLE IF EXISTS "%s".irises;`, s.schema)

	_, err := s.db.ExecContext(ctx, query)
	if err != nil {
		o11y.S(ctx).With(zap.Error(err)).Fatal("Error dropping irises table")
	}

	return err
}
