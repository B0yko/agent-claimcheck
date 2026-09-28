# Benchmark tool API

The benchmark simulates three tool-using domains: `booking`, `crm` and
`coding`. This document specifies every tool and probe a generated trace can
call: its arguments, its success outputs (including the async-accepted form
of its primary write), its error forms, and the claim types a rule pack
checks against it. It documents the API only. What makes a run genuine or a
false success, and how traces are composed, is out of scope here.

## Conventions

Every tool result (`tool_result` step) follows one shape:

- `ok`: `true` or `false`.
- `error`: `null` on success, otherwise a string `"<code>: <message>"` (for
  example `"409 slot_taken: the requested slot is no longer free"`).
- `output`: an object. API tools include `status_code`, an HTTP-like integer
  (`200`, `201`, `202`, `404`, ...). A failing call has `error` set, `ok:
  false`, and `output` that is `null` or `{}`.

Each domain's primary write tool also has an **async-accepted** form: the
call returns HTTP `202` with a `status` of `"pending"`, `"queued"` or
`"accepted"`, and still returns the record or job id. Genuine runs receive
this form too, so a 202 receipt alone never tells a claim-checker whether
the run is real or not; only a later probe or a subsequent read settles it.

A probe (`state_probe` step, `role: environment`) reads the simulated final
state directly. A probe for a record that does not exist returns `ok:
false` with `error: "not_found: <message>"`.

## `booking`

Tools: `calendar.search_slots`, `calendar.create_event`, `email.send_invite`.
Probe: `calendar.get_event`.

### `calendar.search_slots`

Args: `{date, timezone, duration_min}`.

Success output: `{slots: [{start, end}, ...]}` — `start`/`end` are RFC 3339
timestamps with a UTC offset.

### `calendar.create_event` (primary write)

Args: `{title, start, duration_min, timezone, attendees: [email, ...]}`.
`start` is RFC 3339 with a UTC offset.

Success (`201`): `{event_id, status: "confirmed", start, end, timezone,
attendees}`.

Async-accepted (`202`): `{event_id, status: "pending", start, end, timezone,
attendees}`.

Errors: `409 slot_taken` (the slot was taken between search and create),
`422 validation` (a malformed field, e.g. an unparsable time), `429` (rate
limited), `timeout`.

### `email.send_invite`

Args: `{event_id, to: [email, ...]}`.

Success/async (`202`, always accepted, never confirmed synchronously):
`{message_id, status: "queued"}`.

Errors: `404` (unknown `event_id`), `429`, `timeout`.

### Probe: `calendar.get_event`

Args: `{event_id}`.

Output: `{event_id, status: confirmed | tentative | cancelled, start, end,
timezone, attendees, invites_sent: [email, ...]}`. Missing event:
`ok: false`, `error: "not_found: ..."`.

### Claims

| type | subject fields |
|---|---|
| `booked` | `start`, `timezone`, `attendee_email`, `duration_min` |
| `invite_sent` | `attendee_email` |

## `crm`

Tools: `crm.search_contacts`, `crm.get_contact`, `crm.update_contact`,
`crm.add_note`, `crm.update_deal_stage`. Probe: `crm.read_record`.

### `crm.search_contacts`

Args: `{query}`.

Success output: `{results: [{record_id, name, email, company}, ...]}`.
Duplicate or similarly named contacts can both appear.

### `crm.get_contact`

Args: `{record_id}`.

Success output: `{record_id, name, email, company, fields: {...}}`.

Errors: `404` (unknown `record_id`).

### `crm.update_contact` (primary write)

Args: `{record_id, fields: {...}}`.

Success (`200`): `{record_id, updated_fields, record}`.

Async-accepted (`202`): `{record_id, job_id, status: "accepted"}`.

Errors: `404`, `409 version_conflict` (the record changed since it was
read), `422 invalid_field`, `403 permission_denied`, `429`.

### `crm.add_note`

Args: `{record_id, body}`.

Success (`201`): `{note_id, record_id}`.

Errors: `404`, `429`.

### `crm.update_deal_stage`

Args: `{deal_id, stage}`.

Success (`200`): `{deal_id, stage, previous_stage}`.

Errors: `404`, `422 invalid_transition` (the stage change is not a legal
transition from the current stage).

### Probe: `crm.read_record`

Args: `{object: contact | deal, record_id}`.

Output: `{object, record_id, fields, notes: [...], stage}`. Missing record:
`ok: false`, `error: "not_found: ..."`.

### Claims

| type | subject fields |
|---|---|
| `updated` | `object`, `record_id`, `fields` |
| `note_added` | `record_id` |
| `stage_changed` | `record_id`, `stage` |

## `coding`

Tools: `fs.read_file`, `fs.write_file`, `shell.run`, `git.commit`. Probes:
`ci.run_tests`, `git.show_head`.

### `fs.read_file`

Args: `{path}`.

Success output: `{path, content}`.

Errors: `404 not_found`.

### `fs.write_file` (a primary write)

Args: `{path, content}`.

Success (`200`): `{path, bytes_written, sha256}`.

Async-accepted (`202`): `{path, job_id, status: "pending"}`.

Errors: `403 permission_denied`, `422 validation`.

### `shell.run`

Args: `{command}`.

Success output: `{exit_code, stdout, stderr, duration_ms}`; `ok = (exit_code
== 0)`. When the command is a test run, `stdout` carries a simulated pytest
summary line, e.g. `"==== 42 passed in 3.10s ===="` or `"==== 2 failed, 40
passed in 3.40s ===="`.

Errors: `timeout`.

### `git.commit` (primary write)

Args: `{message, paths}`.

Success (`200`): `{sha, branch, files_changed, status: "committed"}` — `sha`
is 7 hex characters.

Async-accepted (`202`): `{sha, branch, files_changed, status: "queued"}`.

Errors: `hook_failed` (a commit hook rejected the change),
`nothing_to_commit`.

### Probe: `ci.run_tests`

Args: `{suite}`.

Output: `{exit_code, passed, failed, total, suite}`, from a clean checkout
(so a probe run reflects what is actually on disk, independent of what the
agent's own `shell.run` reported).

### Probe: `git.show_head`

Args: `{}`.

Output: `{sha, message, files}` — the current HEAD commit.

### Claims

| type | subject fields |
|---|---|
| `tests_passed` | `suite`, `count` |
| `fixed` | `file` |
| `committed` | `sha` |
