---
name: review-bugfix
description: Review code changes, investigate regressions, debug failures, and run post-change verification in this repository. Use when a task involves bug fixing, self-review after implementation, regression checking, API/runtime breakage, retrieval behavior drift, or validating that a completed task is actually complete.
---

# Review Bugfix Skill

Use this skill after meaningful code changes or when debugging.

## Required pre-check

Before reviewing or debugging:

1. Read `AGENTS.md`
2. Read `spec.md`
3. Read `tasks.md`
4. Check `docs/development-roadmap.md`

## Review workflow

1. Identify what changed
2. Check whether the change affects:
   - API payload shapes
   - retrieval behavior
   - runtime fallback behavior
   - eval output structure
   - frontend/backend integration
3. Run the smallest verification path that can catch regressions
4. Record any follow-up items in `tasks.md`

## Bugfix workflow

1. Reproduce the issue
2. Isolate whether it belongs to:
   - config
   - provider
   - retrieval
   - runtime
   - API
   - frontend integration
3. Apply the smallest reasonable fix
4. Re-run the reproduction path
5. Review neighboring code for related breakage

## Repository-specific review points

- Do not treat fallback behavior as success
- Make sure diagnostics remain visible
- For retrieval changes, check whether compare/eval outputs still make sense
- For API changes, check schema alignment and frontend expectations

## Close-out

After review or bugfix:

1. Mark the task complete in `tasks.md` if done
2. Add follow-up tasks if something remains unresolved
3. Update the roadmap only if the phase status materially changed
