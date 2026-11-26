# Security policy

## Supported versions

Security fixes are applied to the latest version of the `main` branch. Older
commits, forks, experimental branches, and modified model packages are not
supported.

TinyTitan is a research project. It is not intended for production,
multi-user, or security-critical deployments.

## Local server trust model

The OpenAI/Anthropic-compatible server binds `127.0.0.1` only. It has no TLS and
no authentication; `Authorization` and `x-api-key` are accepted and ignored.
The trust boundary is the machine:

- Other local processes can call the server and are accepted. They already have
  the operator's privileges, so authentication here would not be a boundary.
- Browser pages are not trusted. CORS is not enabled (no preflight answer, no
  `Access-Control-Allow-Origin`), and requests whose `Host`, `Origin` or
  `Referer` does not name a loopback host are refused before routing. That is
  what closes DNS rebinding, where a page's own hostname resolves to
  `127.0.0.1` and the browser treats the server as same-origin.

Do not expose the port through a proxy, tunnel or reverse proxy, and do not
change the bind address to a non-loopback interface. See
[`docs/server-api.md`](docs/server-api.md#trust-boundary) for the exact header
policy and error codes.

## Reporting a vulnerability

Do not open a public issue for a suspected security vulnerability. Use
[GitHub private vulnerability reporting](https://github.com/Pummelchen/TinyTitan/security/advisories/new)
instead.

Include:

- a description of the vulnerability;
- the affected commit or version;
- reproduction steps or a minimal proof of concept;
- the expected and observed behavior;
- the potential security impact; and
- a suggested mitigation, if known.

Do not include personal data, credentials, access tokens, or copyrighted model
weights in the report.

Reports are especially useful when they involve unsafe model-package handling,
path traversal, buffer or offset safety, malformed remote data, verification
bypasses, command-line injection, credential exposure, or unexpected file
access. Model quality problems, incorrect generated text, expected high
resource use, and performance regressions are not normally security
vulnerabilities.

Please allow the issue to be investigated and a fix to be prepared before
publishing vulnerability details. Credit can be included in the eventual
advisory unless the reporter prefers to remain anonymous.
