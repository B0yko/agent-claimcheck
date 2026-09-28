# Security policy

## Reporting a vulnerability

Please report security issues privately through GitHub's
[Security Advisories](https://docs.github.com/en/code-security/security-advisories/guidance-on-reporting-and-writing/privately-reporting-a-security-vulnerability)
for this repository ("Security" tab → "Report a vulnerability"), rather than
opening a public issue. Include what you found, how to reproduce it, and
its impact if you can. Expect an acknowledgement and, once a fix is ready, a
release plus an advisory describing the issue.

## Supported versions

Only the latest released version is supported with security fixes.

## Scope and threat model

**The library and CLI** read trace files you give them and, when a judge
detector or `bench --live` is configured, send trace content to the
OpenAI-compatible endpoint named by `--config`/environment variables. No
telemetry, no other outbound calls. Secrets (API keys) are read from the
process environment only — never from a committed file — and are never
written to the response cache, the cost ledger, or any result file.

**The local dashboard** (`agent-claimcheck serve`) is a single-user,
localhost-only tool, not a hosted service (ADR 0001):

- It binds to `127.0.0.1` by default; reaching it from another machine
  requires deliberately overriding `--host`.
- It only serves the trace/result/review files named on its own command
  line — it is not a general-purpose file server.
- Every request's `Host` header must match `127.0.0.1:<port>`,
  `localhost:<port>` or the configured `--host`, which blocks DNS
  rebinding attacks from a page open in the same browser.
- Anything that starts paid work (a judge call) requires a `POST` whose
  `Origin` matches, whose content type is `application/json`, and whose
  body is small; a bare `GET` can never trigger a paid call. The API key
  used for judge calls is held by the server process and never sent to the
  browser.
- Trace content is untrusted by construction — a trace may contain text a
  prior agent run received from a tool or a user — and the UI renders it as
  plain text, never as HTML.

If you find a way around any of the above (an `Origin`/`Host` bypass, a
route that is not on the file allowlist, a key or trace content that
reaches somewhere it should not), that is exactly what this policy wants
reported.

## Not a threat model concern

This project does not authenticate users, run as a multi-tenant service, or
store data beyond the files you point it at, so account takeover,
authorization-boundary and data-residency reports do not apply — there is
no account and no hosted instance.
