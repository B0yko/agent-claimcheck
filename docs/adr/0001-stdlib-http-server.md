# 0001: stdlib HTTP server instead of a web framework

## Status

Accepted.

## Context

The dashboard is a small, local, single-user tool: a handful of read routes,
one SSE stream and a few POST actions guarded by an `Origin` and `Host`
check. It runs on `127.0.0.1` on the user's own machine, not as a hosted
service.

## Decision

Serve the dashboard with `http.server.ThreadingHTTPServer` from the standard
library, plain ES modules and CSS, and no build step. No Flask, FastAPI,
Starlette or similar framework dependency.

## Consequences

- One fewer runtime dependency, and no framework version to track.
- Request routing, JSON handling and the SSE loop are written by hand and
  covered by tests instead of relying on framework guarantees.
- If the dashboard ever needs multi-user access, authentication or a
  database, this decision should be revisited; that is explicitly out of
  scope for v0.1.
