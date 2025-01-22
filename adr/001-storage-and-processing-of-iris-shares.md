# ADR-001: Storage and processing of iris shares

## Context
- We have iris code shares that, in their raw form, are 12800 bytes (u8) each and stored in a PostgreSQL database.
- For secret sharing purposes, these iris codes need to be represented internally as u16.
- The cuBLAS library (used for efficient GPU-accelerated comparisons) requires the data to be in i8 format.
- Performing repeated on-the-fly conversions from u16 to i8 can degrade performance significantly due to large data volumes.

## Decision
Store and load iris code shares in a pre-processed i8 format (split into two limbs from each u16), so that runtime conversions are minimized or eliminated.

## Rationale
- **Performance**: The conversion from u16 to i8 (subtracting 128 for cuBLAS alignment) can be expensive if done repeatedly for large datasets.
- **cuBLAS Requirements**: The library operates natively on i8 data. Having the data already in the correct format avoids overhead.
- **Simplicity in Usage**: Once data is in i8 format, usage in GPU kernels is straightforward.

## Consequences
- **Pros**:
    - Streamlined GPU-based operations (no extra conversion step in the processing pipeline).
- **Cons**:
    - Requires additional pre-processing upon ingestion.

By pre-processing iris code shares into i8 and storing them in that format, we reduce runtime overhead and align with the cuBLAS requirements, resulting in more efficient iris-code comparisons.
