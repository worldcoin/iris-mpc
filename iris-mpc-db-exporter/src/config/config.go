package config

import (
	"os"
	"strconv"
)

type Config struct {
	AWSRegion                  string
	AWSEndpoint                string
	ServiceName                string
	PgSQLConnectionString      string
	DatadogHost                string
	ExportBucketName           string
	ExportBucketRegion         string
	DatadogTraceEnabled        bool
	DatadogTraceDebugEnabled   bool
	NodeId                     int
	Environment                string
	SmpcSchemaName             string
	MaxItemsPerUploadPart      int
	SingleCodeSize             int
	SingleMaskSize             int
	SleepBeforeShutdownSeconds int
}

func Load() Config {
	cfg := Config{
		AWSRegion:                  loadStringVar("AWS_REGION", "eu-north-1"),
		AWSEndpoint:                loadStringVar("AWS_ENDPOINT", ""),
		ServiceName:                loadStringVar("SERVICE_NAME", "iris-mpc-db-exporter"),
		PgSQLConnectionString:      loadStringVar("DATABASE_URL", ""),
		DatadogHost:                loadStringVar("DD_AGENT_HOST", "localhost"),
		DatadogTraceEnabled:        loadBoolVar("DD_TRACE_ENABLED", false),
		DatadogTraceDebugEnabled:   loadBoolVar("DD_TRACE_DEBUG_ENABLED", false),
		ExportBucketName:           loadStringVar("EXPORT_S3_BUCKET_NAME", "bucket_name"),
		ExportBucketRegion:         loadStringVar("EXPORT_S3_BUCKET_REGION", "eu-north-1"),
		NodeId:                     loadIntVar("NODE_ID", 0),
		Environment:                loadStringVar("ENV", "local"),
		SmpcSchemaName:             loadStringVar("SMPC_SCHEMA_NAME", "SMPC"),
		MaxItemsPerUploadPart:      loadIntVar("MAX_ITEMS_PER_UPLOAD_PART", 200), // defines the number of items to be uploaded in a single part of the multipart upload
		SingleCodeSize:             loadIntVar("SINGLE_CODE_SIZE", 25600),
		SingleMaskSize:             loadIntVar("SINGLE_MASK_SIZE", 12800),
		SleepBeforeShutdownSeconds: loadIntVar("SLEEP_BEFORE_SHUTDOWN_SECONDS", 10),
	}

	return cfg
}

func loadBoolVar(key string, def bool) bool {
	value := os.Getenv(key)
	if value == "" {
		return def
	}
	b, err := strconv.ParseBool(value)
	if err != nil {
		panic(err)
	}
	return b
}

func loadStringVar(key string, def string) string {
	value := os.Getenv(key)
	if value == "" {
		return def
	}
	return value
}

func loadIntVar(key string, def int) int {
	value := os.Getenv(key)
	if value == "" {
		return def
	}
	i, err := strconv.Atoi(value)
	if err != nil {
		panic(err)
	}
	return i
}
