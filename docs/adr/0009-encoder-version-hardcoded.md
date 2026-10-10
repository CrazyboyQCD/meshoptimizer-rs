# ADR-0009: Encoders emit the current wire version (v1) only; version setters are not ported

- Status: Accepted
- Date: 2026-10-10
- Refines: [ADR-0005](0005-c-parity-api.md)

## Context

Upstream exposes `meshopt_encodeIndexVersion(int)` / `meshopt_encodeVertexVersion(int)`, which
write a process-global used by all subsequent encode calls. Measured at the pin (v1.3): both
globals default to `1`, setters reject values above the current decode version (`1`), and decode
supports versions 0 and 1 with per-buffer probes (`decodeIndexVersion`/`decodeVertexVersion`).
The setters' only real effect is downgrading *emitted* streams to v0 for consumers whose
decoders predate the v1 formats.

Process-global mutable state has no safe-Rust counterpart; the interim plan (TDD §6) was a
per-call `version` parameter. That costs every caller a parameter that would always be the
default, to serve a downgrade path with no known consumer.

## Decision

- Encode functions hardcode the current wire version (v1) — what upstream emits by default and
  what its own tooling uses.
- The two setters are not ported; SRS FR-05/FR-07 shrink accordingly (portable entry points:
  86 → 84).
- `encodeVertexBufferLevel` drops its `version` parameter and takes only the compression level.
- Decoders are unaffected in scope: they accept every version upstream produces (v0 and v1);
  NFR-5 is a decode-side guarantee and stays intact. `decodeIndexVersion`/`decodeVertexVersion`
  remain available.
- Golden encode fixtures cover v1 only; v0 *decode* fixtures are produced by driving upstream's
  setter during fixture generation (tooling-side usage, not part of the Rust API).

## Consequences

- The planned "global state → parameter" divergence disappears: every ported entry point is now
  an exact C-parity match in both name and contract.
- Emitting v0 streams is not possible from the public API. If a consumer pinned to pre-v1
  decoders materializes, add explicit-version encode functions as a purely additive change and
  supersede this ADR.
- On re-pin, if upstream introduces v2, version parameterization is designed once across the
  encode functions at that point — not retrofitted speculatively now.
