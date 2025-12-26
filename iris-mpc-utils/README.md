iris-mpc-utils
===============

Set of utility functions for working with IRIS-MPC CPU networks.

Why iris-mpc-utils ?
--------------------------------------

Centralised location for utility functions/programs previously scattered and/or duplicated across the monorepo.

What uses iris-mpc-utils ?
--------------------------------------

- Test functions in other crates.
- DevOps tool chains.

What is iris-mpc-utils roadmap ?
--------------------------------------

- DONE
  - Initialisation from existing crates

- QUEUED
  - Refactor genesis e2e tests to use iris-mpc-utils
  - New utils for simulated plaintext graphs
  - Refactor genesis tests to use new simulated plaintext graphs
  - Main HNSW binary e2e support
  - Refactoring as and when makes sense
