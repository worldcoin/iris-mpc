package main

import (
	"context"
	"database/sql"
	"errors"
	"log"
	"slices"
	"syscall"
	"time"

	"github.com/spf13/cobra"
	"go.uber.org/zap"
	"gopkg.in/DataDog/dd-trace-go.v1/ddtrace/tracer"

	"github.com/worldcoin/iris-mpc-db-exporter/src/commands"
	"github.com/worldcoin/iris-mpc-db-exporter/src/config"
	"github.com/worldcoin/iris-mpc-db-exporter/src/converter"
	"github.com/worldcoin/iris-mpc-db-exporter/src/iris"
	"github.com/worldcoin/iris-mpc-db-exporter/src/metrics"
	"github.com/worldcoin/iris-mpc-db-exporter/src/o11y"
	"github.com/worldcoin/iris-mpc-db-exporter/src/persistence"
)

const (
	CodeSize      = 25600
	MaskSize      = 12800
	IdSize        = 4
	VersionIdSize = 2
)

func populateCommand() *cobra.Command {
	var count int

	var populateCmd = &cobra.Command{
		Use:   "populate-db",
		Short: "Populate iris-mpc participant database with mock data",
		Run: func(cmd *cobra.Command, args []string) {
			cfg := config.Load()
			ctx, cancelMainCtxFn := context.WithCancel(context.Background()) // mainCtx
			defer cancelMainCtxFn()

			// Connect to database
			db, err := sql.Open("postgres", cfg.PgSQLConnectionString)
			if err != nil {
				log.Fatal(err)
			}

			store := iris.NewStore(ctx, db, cfg)

			commands.PopulateDbWithMockDataCommand(ctx, *store, count)
		},
	}

	populateCmd.Flags().IntVar(&count, "count", 1000, "How many mock irises to create")

	return populateCmd
}

func exportCommand() *cobra.Command {
	var exportMode string
	var exportFormat string
	var exportOutput string
	var batchSize int
	var parallelism int
	var outputFolder string
	var endIndex int

	allowedOutputs := []string{"hdd", "s3"}

	cfg := config.Load()
	var exportCmd = &cobra.Command{
		Use:   "export-db",
		Short: "Export iris-mpc participant database",
		Run: func(cmd *cobra.Command, args []string) {
			ctx, cancelMainCtxFn := context.WithCancel(context.Background()) // mainCtx
			defer cancelMainCtxFn()

			o11y.S(ctx).Infof(
				"Executing export command with options => exportMode: %s, exportFormat: %s, exportOutput: %s, batchSize: %d, parallelism: %d endIndex: %d\n",
				exportMode, exportFormat, exportOutput, batchSize, parallelism, endIndex,
			)

			// Connect to database
			o11y.S(ctx).Info("Connecting to the database...")
			db, err := sql.Open("postgres", cfg.PgSQLConnectionString)
			if err != nil {
				o11y.S(ctx).With(zap.Error(err)).Fatal("Error connecting to the database")
			}
			err = db.Ping()
			if err != nil {
				o11y.S(ctx).With(zap.Error(err)).Fatal("Error pinging the database")
			}
			o11y.S(ctx).Info("Connected to the database.")

			if !slices.Contains(allowedOutputs, exportOutput) {
				o11y.S(ctx).Fatalf("Invalid format: %s", exportOutput)
			}

			store := iris.NewStore(ctx, db, cfg)

			var reader persistence.Reader
			var writer persistence.Writer
			switch exportOutput {
			case "hdd":
				reader = &persistence.FilesystemReader{}
				writer = &persistence.FilesystemWriter{}
			case "s3":
				reader, err = persistence.NewS3Reader(ctx, cfg.ExportBucketName, cfg.ExportBucketRegion, cfg.AWSEndpoint)
				if err != nil {
					o11y.S(ctx).With(zap.Error(err)).Fatal("Error creating S3 reader")
				}
				writer, err = persistence.NewS3Writer(ctx, cfg.ExportBucketName, cfg.ExportBucketRegion, cfg.AWSEndpoint, cfg.MaxItemsPerUploadPart)
				if err != nil {
					o11y.S(ctx).With(zap.Error(err)).Fatal("Error creating S3 writer")
				}
			}

			// Avoid blocking producers by defining a chan buffer in case consumers are slower
			chanBufferLen := cfg.MaxItemsPerUploadPart * 2

			commands.ExportCommand(ctx, exportMode, outputFolder, *store, converter.NewBinaryConverter(CodeSize, MaskSize, IdSize, VersionIdSize), writer, reader, batchSize, parallelism, endIndex, chanBufferLen)
		},
	}

	exportCmd.Flags().StringVar(&exportMode, "export-mode", "INCREMENTAL_EXPORT", "Complete or incremental export")
	exportCmd.Flags().StringVar(&exportFormat, "export-format", "csv", "Choose which format to convert the data to (csv/binary)")
	exportCmd.Flags().StringVar(&exportOutput, "export-output", "hdd", "Choose which persistence to use (HDD/S3)")
	exportCmd.Flags().StringVar(&outputFolder, "output-folder", "output", "Specify the output folder chunks will be written to")

	exportCmd.Flags().IntVar(&batchSize, "batch-size", 10000, "Choose number of iris codes in each export batch")
	exportCmd.Flags().IntVar(&endIndex, "end-index", 0, "Choose the end index to export up to (0 is for a full export)")
	exportCmd.Flags().IntVar(&parallelism, "parallelism", 10, "Choose how many batches to export in parallel")

	return exportCmd
}

func main() {
	rootCmd := &cobra.Command{Use: "app"}
	cfg := config.Load()
	ctx := context.Background()

	// logging start
	o11y.ConfigureLogging()
	defer func() {
		err := o11y.S(ctx).Sync()
		// Sync is not supported on os.Stderr / os.Stdout on all platforms.
		// See: https://github.com/uber-go/zap/issues/1093#issuecomment-1120667285
		if errors.Is(err, syscall.EINVAL) {
			err = nil
		}
		if err != nil {
			o11y.S(ctx).Errorf("error syncing zap logger: %v", err)
		}
	}()

	// tracing start
	if cfg.DatadogTraceEnabled {
		rules := []tracer.SamplingRule{
			tracer.ServiceRule(cfg.ServiceName, 1.0000),
		}

		if cfg.DatadogTraceDebugEnabled {
			o11y.S(ctx).Info("Trace debug logging enabled")
			tracer.Start(tracer.WithDebugMode(true), tracer.WithSamplingRules(rules))
		} else {
			tracer.Start(tracer.WithSamplingRules(rules))
		}
		defer tracer.Stop()
	}

	// metrics start
	err := metrics.InitMetrics(ctx, cfg)
	if err != nil {
		o11y.S(ctx).Errorf("error initializing metrics client: %v", err)
	}
	defer metrics.CloseMetrics(ctx, time.Duration(cfg.SleepBeforeShutdownSeconds)*time.Second)

	rootCmd.AddCommand(exportCommand())
	rootCmd.AddCommand(populateCommand())

	err = rootCmd.Execute()
	if err != nil {
		o11y.S(ctx).With(zap.Error(err)).Error("Error executing command")
		return
	}
}
