# Iris MPC DB Exporter

This repository provides a tool to export the Iris MPC participant database to various formats using a CLI application.

## Features

- Export database records to CSV.
- Supports saving to the filesystem (HDD) or S3 bucket.
- Implements tracing with DataDog for performance monitoring and debugging.

## Prerequisites

- Go 1.23 or higher
- PostgreSQL database connection string
- DataDog (optional, for tracing)

## Installation

Clone the repository and build the executable:

```bash
git clone https://github.com/worldcoin/iris-mpc-db-exporter.git
cd iris-mpc-db-exporter
go build -o iris-mpc-db-exporter
```

## Usage

Run the application using the `export-db` command:

```bash
./iris-mpc-db-exporter export-db --export-format=<format> --export-output=<output>
```

### Command-Line Flags

| Flag              | Description                                                                         | Required | Default |
|-------------------|-------------------------------------------------------------------------------------|----------|---------|
| `--export-output` | Persistence method (`hdd` / `s3`)                                                   | No       | `hdd`   |
| `--export-format` | Persistence file type (`csv`)                                                       | No       | `csv`   |
| `--export-mode`   | Export from scratch or build on existing (`COMPLETE_EXPORT` / `INCREMENTAL_EXPORT`) | No       | `INCREMENTAL_EXPORT`   |
| `--batch-size`    | Number of iris codes in each export batch                                           | No       | `10000` |
| `--parallelism`   | Number of batches to export in parallel                                             | No       | `8`     |

### Example

Export the database to a CSV file:

```bash
./iris-mpc-db-exporter export-db --export-format=csv --export-output=hdd
```
