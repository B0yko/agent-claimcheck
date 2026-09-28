# 0002: a declarative rule DSL with no code evaluation

## Status

Accepted.

## Context

Rule packs decide whether a tool receipt and an optional state probe support
a claim. Users add their own packs for their own tools and domains. That
means rule packs can come from outside the project's own repository.

## Decision

Rule packs are YAML, loaded with `yaml.safe_load` (never `yaml.load`, never
`eval`), and validated against a JSON Schema before use. A rule can only
express: a tool-name glob, a set of field-path checks (`exists`, `equals`,
`in`, `contains`, `matches`) with typed normalisers, and simple string
interpolation of claim and call fields. There is no way to embed a Python
expression, a lambda or a callback in a pack.

## Consequences

- A rule pack cannot execute arbitrary code, so loading a third party's pack
  is not a code-execution risk the way loading a third party's Python module
  would be.
- The check vocabulary is intentionally small. Anything it cannot express
  (e.g. cross-field arithmetic) is out of scope for a rule and belongs in a
  detector written in Python instead.
- A malformed pack (bad YAML, a schema violation or an invalid regex) fails
  to load with a clear `RulePackError` rather than partially applying.
