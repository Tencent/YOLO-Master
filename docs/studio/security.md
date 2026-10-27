# Studio security boundary

[Studio documentation](README.md)

## Trusted local deployment

Studio supports one service process on a trusted local machine, bound to
`127.0.0.1`, `localhost` or `::1`. The service checks loopback peers, literal
local Host headers and configured local Origins. Proxy headers cannot grant
trust. Wildcard/remote configured Origins and remote bind addresses are rejected.
There is no login, role-based access control, user isolation or per-job OS sandbox.
Public hosting and untrusted LAN/proxy deployments are unsupported.

Cross-origin resource sharing (CORS) permits configured local development origins
and legacy `null`/file-console
compatibility. An opaque `null` origin cannot authenticate its source.
Configuration belongs to the trusted operator; see
[service settings](service.md#trusted-configuration-and-existing-ui-entry-points).

## Paths and manifest authorization

The server owns model/data/output roots, network-host permissions and policy.
Client `allowed_paths`, regex patterns, lifecycle fields and artifact candidates
cannot extend access. Shell execution stays disabled. Model paths must be local;
network data inputs require a server-authorized host and reject URL credentials.

Containment resolves existing symlinks and permits them only when the resolved
target stays inside the trusted root. Broken links, out-of-root targets,
traversal, sibling-prefix escapes and invalid regex policy are rejected. Both
the output base and final job directory must remain inside the permitted root.

After process cleanup, the manager validates handler output paths and records
the permitted downloads in an artifact manifest. API lists and downloads
use exact job-relative IDs, including nested paths; every read checks current
containment and opened-file identity. Existing files outside that manifest are
not downloadable. Safe legacy absolute manifest entries are adapted at read
time without rewriting historical meaning.

## Logs, errors and persisted state

Structured job logs, error/traceback surfaces, snapshots and HTTP validation
errors are sanitized. Invalid service configuration uses fixed errors that omit
configuration values and parser exception chains. Keep identifiers free of secrets.

Windows workers detach from the service console and route Python stdout/stderr
and prebound console logging through sanitized inter-process communication (IPC).
The Windows capture path handles complete Python text lines; native descriptor
writes, interrupted partial lines and arbitrary subprocess output fall outside
this path. Windows Job Objects and Linux process-group/guardian/subreaper handling
govern lifecycle cleanup. Jobs run with the host user's permissions.

State is stored in local JSON snapshots. Keep state files and model/checkpoint
inputs under trusted operator control, and load only trusted pickle checkpoints.
Parent checkpoint acknowledgment checks containment and size without deserializing
the file. See [runtime recovery](runtime.md#training-shutdown-and-recovery)
for load/resume checks and [testing](testing.md) for verification procedures.
