package iris

import (
	"context"
	"database/sql"
	"fmt"

	"go.uber.org/zap"
	"gopkg.in/DataDog/dd-trace-go.v1/ddtrace/tracer"

	"github.com/worldcoin/iris-mpc-db-exporter/src/config"
	"github.com/worldcoin/iris-mpc-db-exporter/src/o11y"

	_ "github.com/lib/pq"
)

const (
	irisDbSchemaFormat = "%s_%s_%d"
)

type StoredIris struct {
	ID             int64  `bson:"id"`               // BIGSERIAL
	LastModifiedAt *int64 `bson:"last_modified_at"` // BIGINT
	LeftCode       []byte `bson:"left_code"`        // BYTEA
	LeftMask       []byte `bson:"left_mask"`        // BYTEA
	RightCode      []byte `bson:"right_code"`       // BYTEA
	RightMask      []byte `bson:"right_mask"`       // BYTEA
	VersionID      int16  `bson:"version_id"`       // SMALLINT
}

type StoredIrisStore interface {
	GetStoredIrisesByRange(startIndex, endIndex int) ([]StoredIris, error)
}

type Store struct {
	db     *sql.DB
	schema string
}

func NewStore(ctx context.Context, db *sql.DB, config config.Config) *Store {
	schema := fmt.Sprintf(irisDbSchemaFormat, config.SmpcSchemaName, config.Environment, config.NodeId)

	// create schema if not exists - used for db populate in local environment
	o11y.S(ctx).Infof("ENV: %s", config.Environment)
	if config.Environment == "local" || config.Environment == "CI" {
		o11y.S(ctx).Info("Creating schema if not exists in local environment")
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

func (s *Store) GetStoredIrisesByRange(ctx context.Context, startIndex, endIndex int) ([]StoredIris, error) {
	span, ctx := tracer.StartSpanFromContext(ctx, "pgsql.get_stored_irises_by_range")
	defer span.Finish()

	span.SetTag("startIndex", startIndex)
	span.SetTag("endIndex", endIndex)

	query := fmt.Sprintf(`SELECT id, last_modified_at, left_code, left_mask, right_code, right_mask, version_id FROM "%s".irises WHERE id >= $1 AND id <= $2;`, s.schema)
	rows, err := s.db.Query(query, startIndex, endIndex)
	if err != nil {
		o11y.S(ctx).With(zap.Error(err)).Errorf("Failed to fetch irises in range %d, %d. Error: %v", startIndex, endIndex, err)
		return nil, err
	}

	defer rows.Close()

	var irises []StoredIris

	for rows.Next() {
		var storedIris StoredIris

		if err := rows.Scan(&storedIris.ID, &storedIris.LastModifiedAt, &storedIris.LeftCode, &storedIris.LeftMask, &storedIris.RightCode, &storedIris.RightMask, &storedIris.VersionID); err != nil {
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

	query := fmt.Sprintf(`SELECT id, last_modified_at, left_code, left_mask, right_code, right_mask, version_id FROM "%s".irises WHERE id >= $1 AND id <= $2;`, s.schema)
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
			if err := rows.Scan(&storedIris.ID, &storedIris.LastModifiedAt, &storedIris.LeftCode, &storedIris.LeftMask, &storedIris.RightCode, &storedIris.RightMask, &storedIris.VersionID); err != nil {
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

	query := fmt.Sprintf(`SELECT id, last_modified_at, left_code, left_mask, right_code, right_mask, version_id FROM "%s".irises WHERE (id >= $1 AND id <= $2 AND last_modified_at < $3);`, s.schema)
	rows, err := s.db.Query(query, startIndex, endIndex, lastModifiedAt)
	if err != nil {
		return nil, err
	}

	defer rows.Close()

	var irises []StoredIris

	for rows.Next() {
		var storedIris StoredIris

		if err := rows.Scan(&storedIris.ID, &storedIris.LastModifiedAt, &storedIris.LeftCode, &storedIris.LeftMask, &storedIris.RightCode, &storedIris.RightMask, &storedIris.VersionID); err != nil {
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
	    version_id SMALLINT DEFAULT 0 CHECK (version_id >= 0)
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
