# Rule packs: the DSL reference

The rules detector turns a success claim into an outcome by checking it
against tool receipts and, optionally, a state probe. What it can express is
deliberately small (ADR 0002): a tool-name glob, a set of field checks with
typed normalisers, one list check, and string interpolation. There is no way to embed a
Python expression in a pack, so loading a third party's pack is not a
code-execution risk.

## A pack, end to end

```yaml
pack: booking
version: 1
tools: ["calendar.*", "email.*"]
claim_patterns:
  booked: ['\b(booked|scheduled|confirmed)\b']
claims:
  booked:
    action: "calendar.create_event"
    receipt: { event_id: { exists: true }, start: { equals: "{subject.start}", as: datetime } }
    probe: "calendar.get_event"
    probe_checks: { status: { in: [confirmed] }, start: { equals: "{subject.start}", as: datetime } }
```

- `pack`, `version`: the pack's name and a version number. `pack` also
  doubles as the domain name a pack falls back to (see **Applicability**).
- `tools`: glob patterns (`fnmatch`, case-sensitive) matched against
  `tool_call` step names.
- `claim_patterns`: `{claim_type: [regex, ...]}`, used only for
  pattern-based claim extraction when a trace has no structured
  `final_claim.claims` (see `docs/python-api.md` for `ClaimExtractor`).
  Patterns compile with `re.IGNORECASE`; named groups become subject
  fields. A pack named `generic` is a special case: its `claim_patterns`
  (shipped as the built-in `done | completed | finished | all set` → `done`
  mapping) are tried only as a fallback, when every applicable domain
  pack's own patterns matched nothing — a `generic` pack has no `claims`
  and is never consulted by the rules detector itself. A match is dropped
  as a non-claim when its sentence has a negation before it (`couldn't`,
  `unable to`, `didn't`, `never`, `no longer`, ...) or a passive one
  (`wasn't`, `is not`, `hasn't been`, `did not get`, ...), and when the
  sentence says `not yet` anywhere. A negation after the match does not
  count: "Booked, and nothing is not working" still claims `booked`.
- `claims`: `{claim_type: rule}`. Each rule has:
  - `action` (required): a glob a `tool_call` name must match.
  - `receipt` (required, may be empty): checks on that call's
    `tool_result.output`.
  - `probe` (optional): a glob a `state_probe` name must match.
  - `probe_checks` (optional): checks on that probe's output.

Packs are loaded with `yaml.safe_load` (never `yaml.load`) and validated
against a JSON Schema before use; an unrecognised key, a malformed check, or
an invalid regex in `claim_patterns` or a `matches` check all fail to load
with a `RulePackError` rather than partially applying.

## Field paths

A check's key is a dotted path: `event.start`, `attendees.0.email` (a
numeric segment indexes into a list). Paths read from the matched call's
`tool_result.output` by default. The special prefix `args.` reads from the
matched call's own `args` instead, so a rule can confirm the call was made
with the right value, not only that it returned successfully:

```yaml
receipt:
  args.path: { equals: "{subject.file}" } # what the call was asked to write
  bytes_written: { exists: true } # what it reported back
```

## Checks

| operator | meaning |
|---|---|
| `exists: true|false` | whether the field is present at all |
| `equals: v` | equality (see normalisers below) |
| `in: [v, ...]` | membership: equal to any one of the list |
| `contains: v` | substring (field is a string), membership (field is a list), or every key of the mapping `v` present with an equal value (field is an object) |
| `matches: regex` | `re.search(regex, str(field))` |
| `items: {...}` | what a list field must or must not hold (see **List checks**) |

Each check declares exactly one operator, plus an optional `as` normaliser:

| `as` | normalises to | equality |
|---|---|---|
| `datetime` | an aware instant (`datetime.fromisoformat`; a naive value is treated as UTC) | by instant, so `19:00+02:00` equals `17:00Z` |
| `email` | stripped, case-folded | string equality |
| `number` | `float` | `abs(a - b) <= 1e-9` |
| `string` | `str(v)`, stripped, case-folded | string equality |

A value that cannot be normalised (for example a non-string field with
`as: datetime`) fails the check; it is not skipped.

## List checks

`contains` tests one value at a time, so it cannot say "one booking is both
accepted and at this time". `items` can:

```yaml
probe_checks:
  bookings:
    items:
      when: { backend: { equals: calcom } } # optional: does this list apply?
      some: # at least one element passes every field check
        status: { equals: cancelled }
        start: { equals: "{subject.time}", as: datetime }
      none: # no element passes every field check
        status: { in: [accepted, pending] }
        start: { equals: "{subject.time}", as: datetime }
  open_slots:
    items: { covers: "{subject.times}" } # every claimed value is in the list
    as: datetime
```

An `items` check needs at least one of `some`, `none` or `covers`; together,
they must all hold.

- `some` and `none` hold plain field checks, with paths relative to the
  element (`""` is the element itself). They cannot nest another `items`
  and cannot read `args.`.
- `covers` is a single `{subject.*}` or `{args.*}` placeholder. The outer `as`
  normalises both sides, and applies to `covers` only.
- `when` holds field checks on the output that contains the list. When it
  does not hold, the list says nothing about this claim: the check is
  skipped, and in `probe_checks` the claim becomes `unknown`.
- A referenced claim value that is missing or null, or an empty `covers`
  list, skips the check and flags `missing_subject`. A field that is absent
  or not a list fails.
- The evidence row reports counts (`count`, `some`, `none`) and up to five
  `uncovered` claimed values. It does not copy the list.

## Interpolation

`equals`, `in` and `contains` values can reference the claim or the call:
`{subject.<path>}` reads the resolved claim's `subject`, `{args.<path>}`
reads the matched call's `args`. A template that is exactly one placeholder
(nothing else in the string) keeps the value's own JSON type, so
`equals: "{subject.duration_min}"` compares against a number, not its
string form.

When an interpolated field is missing, the check is **skipped** and the
claim is flagged `missing_subject`. A claim whose checks were all skipped
(receipt and probe together) can reach at most `receipt_only`, never
`probe_supported` — an unconfirmed value is not evidence either way.

## Applicability

A pack applies to a trace when:

- at least one `tool_call` name matches one of its `tools` globs, **or**
- the trace makes **no** tool calls at all and `task.domain` equals the
  pack's own name.

The second branch is what makes a phantom action (an agent that never
called any tool) `unsupported` rather than `unknown`: with zero tool calls,
a `booking`-domain trace still has the `booking` pack apply, so its
`booked` claim's rule is evaluated and correctly finds no matching call. A
trace whose tool calls are all unfamiliar (none match any pack's globs) has
no pack apply at all, so its claims are `unknown` — abstaining rather than
guessing.

Because `task.domain` is a fixed enum (`booking`, `crm`, `coding`,
`browser`, `other`), the domain fallback branch really only matters for the
three built-in packs. A custom pack for your own tools (see
`examples/rules/custom.yaml`) will normally apply through its `tools` globs
alone, which is enough: `--rules custom.yaml` or
`Checker(rules=["custom.yaml"])` adds it to the packs the rules detector
consults, without needing a matching domain name.

## Outcomes and scores

Each claim gets exactly one outcome, with a step index to cite:

| outcome | meaning | raw `p_success` |
|---|---|---|
| `contradicted` | a result or probe says the claim is false | 0.03 |
| `unsupported` | the action was never called, or never returned | 0.05 |
| `unknown` | no applicable pack defines this claim type, or a `when` guard says the probe does not describe it | abstain |
| `receipt_only` | the call succeeded; no probe confirms it | 0.70 |
| `probe_supported` | a state probe independently confirms it | 0.97 |

The evaluation order: no applicable pack → `unknown`; no
matching call → `unsupported`; evaluate every matching call together with
its paired result — a call passes when its result exists, `ok` is true,
`error` is null and every non-skipped receipt check passes. If any call
passes, the *last* passing call decides; a rule without a `probe` stops
there at `receipt_only`. Otherwise, if any matching call's result is a
failure (or fails a receipt check), the claim is `contradicted`; if every
matching call simply has no result yet, it is `unsupported`. When a probe
is declared, the *last* matching `state_probe` decides between
`contradicted`, `unknown` (an `items` check's `when` guard does not hold),
`receipt_only` (no matching probe found) and `probe_supported`.

A trace's outcome is the worst outcome among its success claims, in the
order above. `receipt_only` sits at 0.70, below the 0.80 verified
threshold, on purpose (ADR 0004): a receipt without independent
confirmation goes to human review unless a calibrator raises it.

## The built-in `booking-truth` pack

`rules/packs/booking-truth.yaml` maps the tools of
[booking-truth](https://github.com/B0yko/booking-truth) agents, so its traces
need no `--rules` flag. It compares what the agent told the user with the
final `sandbox_state` probe, not with the agent's own tool receipt:

| claim | needs a call to | the probe must show |
|---|---|---|
| `booked` | `book*` | an `accepted` booking at the claimed instant |
| `rescheduled` | `reschedule_booking` | an `accepted` booking at the claimed instant |
| `cancelled` | `cancel_booking` | a `cancelled` booking at the claimed instant, and no `accepted` or `pending` one there |
| `offered_slots` | `find_slots` | every offered instant in `reference_slots` |

Its limits:

- Only Cal.com state is read. A Google Calendar run is `unknown`, so it goes
  to review.
- Bookings are matched by time and status, not by attendee, because the
  claim names no attendee. Another person's booking at the claimed instant
  would satisfy `booked`.
- CRM records are not checked, and neither is a double booking when the
  claimed time is one of the two.
- `reference_slots` is capped by the producer (400 entries in booking-truth
  0.1.0); an offered slot beyond the cap would read as never returned.

## Mapping your own tools

Write a pack against your own tool and probe names — the DSL does not know
about `calendar.*` or `crm.*` specially. `examples/rules/custom.yaml` maps
an invented `acme.*` scheduling API the same way `rules/packs/booking.yaml`
maps `calendar.*`:

```yaml
pack: acme
version: 1
tools: ["acme.schedule_meeting", "acme.notify_attendee"]
claims:
  meeting_booked:
    action: "acme.schedule_meeting"
    receipt: { meeting_id: { exists: true }, starts_at: { equals: "{subject.start}", as: datetime } }
    probe: "acme.get_meeting"
    probe_checks: { state: { in: [confirmed] } }
```

Load it alongside or instead of the built-ins with `load_rule_pack()` and
pass it to a `Checker` (or the `check --rules` CLI flag) so the rules
detector can evaluate claims your own agent makes.
