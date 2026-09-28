# 0007: no vendor adapters

## Status

Accepted.

## Context

An earlier, unpublished version of this project's judge detector called a
single proprietary vendor's decision API directly. That coupling meant the
judge detector could not be reproduced, benchmarked or run by anyone
without that vendor's access, and it tied an open-source tool to one
company's product.

## Decision

The LLM judge detector speaks only the OpenAI-compatible `/chat/completions`
shape: base URL, API key and model name, all supplied through configuration
or environment variables. It is developed and evaluated against OpenRouter,
which fronts many providers behind that same shape, but nothing in the code
recognises OpenRouter, or any other vendor, by name. Any endpoint that
implements the same request and response shape works, including a local
Ollama or vLLM server.

## Consequences

- Anyone can reproduce the judge results with their own key against their
  own choice of model, without depending on a specific vendor.
- No vendor-specific request fields, response fields or error formats are
  read anywhere in the codebase.
- A provider whose API does not fit the OpenAI-compatible shape is out of
  scope for v0.1; it would need its own adapter, deliberately not shipped
  here.
