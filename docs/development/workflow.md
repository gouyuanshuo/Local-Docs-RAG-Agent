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
| Installed-wheel smoke | the built wheel imports and serves health outside the checkout without an editable install or frontend assets | Wheel build and install smoke |
| Frontend tests | visible errors, independent bootstrap state, retries, and API-origin selection through mocked HTTP | Frontend → Test frontend behavior |
| Frontend build | strict TypeScript (`tsc -b`) and a production Vite build | Frontend → Build frontend |

CI restores the backend with uv 0.12.16 from the universal `uv.lock` and runs
every gate on Python 3.11 and 3.13. The Agents SDK registration regression uses
the installed SDK from that restore, not a substitute module. CI restores the
frontend with `pnpm install --frozen-lockfile` on Node 22 and pnpm 10.18.0.
It runs the frontend behavior tests before the build. A separate package job
builds the actual wheel, installs it into a fresh environment, changes to a run
directory outside the checkout, and verifies API-only startup and health.

## Dependency changes

`pyproject.toml` declares Python dependencies and `uv.lock` records the
complete resolution for every extra. To change a dependency intentionally:

1. Edit the manifest and run `uv lock` with uv 0.12.16.
2. Review both files and run `uv sync --locked --all-extras` twice, with the
   quality gates between the restores.
3. Confirm the second restore leaves `uv.lock` unchanged.
4. Commit the manifest and lock together.

Roll back by reverting the manifest and lock pair together. Reverting only
one side makes a locked restore fail or silently preserves the wrong policy.
Isolated wheel builds do not read `uv.lock`, so their exact build requirements
stay pinned in `[build-system]` and are checked by the wheel smoke job.

Live checks against real providers and Qdrant are not gates. They live in
`scripts/`, never run in CI, and are reported separately from the offline
results; see [troubleshooting](troubleshooting.md#live-checks).

Packaging or deployment changes also follow the isolated wheel, static-build,
startup, backup, and rollback checks in the
[deployment guide](deployment.md#build-and-verify-a-release).

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
