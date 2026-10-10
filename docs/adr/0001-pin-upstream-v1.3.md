# ADR-0001: Pin the upstream reference to meshoptimizer v1.3

- Status: Accepted
- Date: 2026-10-10

## Context

This project is a from-scratch Rust port of [zeux/meshoptimizer](https://github.com/zeux/meshoptimizer).
The upstream `master` branch moves continuously (survey HEAD: `b2330430`, 2026-10-09, `MESHOPTIMIZER_VERSION 1030`).
A port needs a frozen reference so that functional requirements, differential tests, and golden
fixtures all point at one exact revision. The latest released tag is `v1.3`
(`9e1f07b159d3cb777f1c67ed31fc11fd117986f4`).

The public API surface was extracted from `src/meshoptimizer.h` at both revisions and diffed:
85 stable `MESHOPTIMIZER_API` entry points plus 2 `MESHOPTIMIZER_EXPERIMENTAL` entry points are
identical between `v1.3` and the survey HEAD (the two master commits in between only change
internal behavior of `tangentspace.cpp`).

## Decision

All functional requirements (SRS), design (TDD), and test fixtures reference upstream **tag
`v1.3`, commit `9e1f07b159d3cb777f1c67ed31fc11fd117986f4`**. Behavior differences found between
this port and upstream are adjudicated against that revision only.

Re-pinning to a newer tag is a deliberate act: it requires a superseding ADR and regeneration
of all golden fixtures (see ADR-0007).

## Consequences

- Upstream changes after `v1.3` (including the two known master commits) are out of scope until re-pin.
- Fixture provenance is reproducible: `git checkout 9e1f07b1` in an upstream clone regenerates every input.
- The pinned revision ships `MESHOPTIMIZER_VERSION 1030` (1.3); the crate README/docs must state this
  so users know which upstream behavior they get.
