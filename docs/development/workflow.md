# Workflow

How a change goes from an idea to a commit, and what must be true before it
lands.

## Before you start

1. Read [AGENTS.md](../../AGENTS.md), then the one module contract it routes you
   to.
2. Skim the current focus in [tasks.md](../../tasks.md).
3. Open the [roadmap](../planning/development-roadmap.md) only if the work
   changes phase status.

Do not start by reading the whole repository. The contracts exist so a change
can be made correctly from its module outward.

## Branches and pushes

`master` is the main branch; work happens on a feature branch. Agents working in
this repository commit locally and do not push or open a pull request unless
asked to.

## Commits

The format is defined in [AGENTS.md](../../AGENTS.md#commit-convention):
Conventional Commits, scoped to the module.

The subject says what changed. The body says why, and that part matters more
here than usual: several commit messages in this history are the fullest record
of why the design looks the way it does. A good body names the problem the change
removes, the alternative it did not take, and what it costs.

Keep one logical change per commit. A refactor that must not change behavior
goes in its own commit, separate from any change that does, so it can be verified
on its own.

## Quality gates

The commands live in one place: [AGENTS.md](../../AGENTS.md#quality-gates). This
table says what each gate guards and where CI runs it.

| Gate | Guards | CI job and step |
| --- | --- | --- |
| Ruff lint | lint rules, import order, Google docstrings, 80 columns | Backend → Ruff |
| Ruff format | formatting | Backend → Ruff format |
| Ruff `DOC201` | a `Returns:` section on every function that returns a value; a preview rule, so it runs alone | Backend → Ruff docstring returns |
| Mypy | strict types over `backend/src`, `tests`, and `scripts`; run bare so it reads `[tool.mypy] files` | Backend → Mypy |
| Pytest | behavior, import direction, eval gold, documentation integrity | Backend → Pytest |
| Compile | the package compiles | Backend → Compile Python package |
| Frontend build | strict TypeScript (`tsc -b`) and a production Vite build | Frontend → Build frontend |

CI installs the backend with `pip install -e ".[agents,qdrant,dev]"` on Python
3.13, and the frontend with `pnpm install --frozen-lockfile` on Node 22 and
pnpm 10.18.0.

Live checks against real providers and Qdrant are not gates. They live in
`scripts/`, never run in CI, and are reported separately from the offline
results; see [troubleshooting](troubleshooting.md#live-checks).

## Closing out a change

Follow the close-out steps in [AGENTS.md](../../AGENTS.md#close-out). Then make
sure the documentation still describes the code.

## Keeping documentation true

| If you changed… | Update, in the same commit |
| --- | --- |
| a module's public symbols or invariants | that module's `AGENTS.md` |
| a configuration setting | `.env.example` and `tests/conftest.py` |
| an extension point | [extending.md](extending.md) |
| a module boundary or dependency direction | [the architecture](../design/architecture.md) |
| how tests are written or organized | [testing.md](testing.md) |
| phase status | [the roadmap](../planning/development-roadmap.md) |
| what is being worked on | [tasks.md](../../tasks.md) |
| any behavior a document describes | that document |

`tests/test_docs.py` catches a broken link, a quoted path that no longer exists,
and a document the index does not reach. It cannot catch a document that is
still reachable but no longer true, which is why the update belongs in the same
commit as the change.
