# Private Agent — Future Implementation Plan

All currently planned, non-future implementation work has been completed and
removed from this checklist. The following feature remains explicitly deferred
and is not part of the current implementation.

## Optional human-like text-to-speech output

- Add optional speech for approved user-visible planning and finalized status
  only, preferably using a local engine. Any remote provider requires explicit
  configuration and session consent.
- Build speech from typed plan/status events. Exclude source code, tool output,
  logs, hidden reasoning, credentials, and unfiltered conversation.
- Keep speech off by default and independent from text execution. Handle
  missing engines/devices and stop/cleanup failures without impairing text
  interaction. Do not persist audio or enable microphone input as a side
  effect.

**Acceptance:** prove only approved plan/status fields reach speech; test local
operation, remote consent, exclusion of sensitive/internal content, stop and
shutdown behavior, failure handling, and unchanged text-only operation.
