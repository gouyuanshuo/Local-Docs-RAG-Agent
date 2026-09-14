# Documentation

Start here. This index says where each kind of fact lives, so a reader can find
it and a writer knows where to put it.

`docs/` holds documentation only. The retrieval corpus lives in
`data/corpus/sample/`, and nothing under `docs/` is indexed by default: design
notes about RRF or Qdrant payloads would otherwise become near-duplicate
distractors for the eval questions written about the same topics.

## Find what you need

| I want to… | Read |
| --- | --- |
| understand the shape of the system | [Architecture](design/architecture.md) |
| know what is planned, and why in that order | [Roadmap](planning/development-roadmap.md) |
| know what is being worked on now | [tasks.md](../tasks.md) |
| know the product requirements | [spec.md](../spec.md) |
| change code, as a contributor or an agent | [AGENTS.md](../AGENTS.md), then the module contract it routes to |
| see what past reviews found | [Reviews](#reviews) |

## Where each kind of fact lives

Every fact has one home. Other documents link to it rather than copy it, because
a copy is the first thing to go stale.

| Home | Owns | Does not hold |
| --- | --- | --- |
| Docstrings | why a piece of code is shaped the way it is | cross-module flows |
| Module `AGENTS.md` | the operating contract: interface, invariants, what may and must not change | explanation; contracts stay short because agents read them on every task |
| `docs/design/` | system shape and the decisions behind it | status |
| `docs/planning/` | phase order and phase status | architecture description |
| `tasks.md` | the short-horizon task board | long-horizon plans |
| Root `AGENTS.md` | routing, engineering rules, code style, and the one copy of the quality-gate commands | module detail |
| `README.md` | what the project is and how to start it | configuration reference, status |

## Layout

```text
docs/
  README.md      this index
  design/        system shape and design decisions
  planning/      long-horizon roadmap
  reviews/       dated review snapshots, not maintained
```

## Reviews

Reviews are dated snapshots of the repository as it was. They are not updated
afterwards; findings that still matter are tracked in [tasks.md](../tasks.md).

- [2026-08-29 code review](reviews/2026-08-29-code-review.md)
- [2026-09-13 repository audit](reviews/2026-09-13-audit-report.html) (HTML)
- [2026-09-13 measurement-seam brief](reviews/2026-09-13-modularize-prompt.md)
  — the implementation brief written from that audit

## Writing conventions

- English. Identifiers, paths, and commands appear exactly as in the code.
- Relative links, so they work both on GitHub and in an editor.
- Diagrams in Mermaid, which GitHub renders.
- One home per fact, per the table above: link instead of copying.
- A change that alters behavior described here updates the document in the same
  commit. `tests/test_docs.py` fails on a broken link, a quoted repository path
  that no longer exists, or a document the index does not reach.
