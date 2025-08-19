# ADR-002: Load Performance Optimisation

### Links
[Notion Page on the boot optimisation](https://www.notion.so/worldcoin/SMPCv2-Boot-time-optimisation-1458614bdf8c80c39170dc0040eb9acf)

## Context
- on every boot-up the container needs to load a couple of 100s of GBs (and the number is constantly growing, hopefully into TBs)
- the load is performed directly from the AuroraDB 
- we need to be able to boot the container below 10 minutes

## Decision
We have decided to prepare an `iris-mpc-db-exporter` service that is going to be periodically storing a dump of the database in an S3 bucket. The loading of the data from the bucket, based on our research are going to be significantly faster than from the AuroraDB.   

Aiming to boost the performance further we have decided to move the conversion of `u8` into `i8` to the exporting service. 

**What was also implemented there was a modification of the order of the bytes for the sake of faster loading into the memory of `iris-mpc` pods.**

## Rationale
- The loading of the data from the S3 bucket is significantly faster than from the AuroraDB.
- The conversion of `u8` into `i8` is a simple operation that can be done in the exporting service.
- The modification of the order of the bytes is a simple operation that can be done in the exporting service.

## Consequences
Because of the actions required to make the loading of the data more efficient (altered byte order, byte values shifted) right now we have two storages:
- AuroraDB (source of truth)
- S3 bucket (used for improving the data load time)

They both store the same data **but in different formats**, please refer to implementation (references are accurate in revision `ec14a0c022b73d2d291578502f6bf02cbc0a99d0`):
- `iris-mpc-store/src/lib.rs:41` (universal type for both data formats)
- `iris-mpc-gpu/src/server/actor.rs:464` (method for loading data from AuroraDB into memory)
- `iris-mpc-gpu/src/server/actor.rs:503` (method for loading data from S3 bucket into memory)
