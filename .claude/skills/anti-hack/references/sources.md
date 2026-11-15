# Sources

## Published

- Martin Fowler, YAGNI: https://martinfowler.com/bliki/Yagni.html
- Martin Fowler, *Refactoring*, 2nd ed., ch. 3, dead code and speculative generality: https://martinfowler.com/books/refactoring.html

## Repo decisions

- Blender, the standard library and the installed packages first: `AGENTS.md`'s "Blender does anything it can do. Code is a liability."
- Every physical number cites a source: `AGENTS.md`'s conventions.
- An odd shape is justified only by a documented constraint, and precedent in the codebase is never a reason.
- `Limitation:` with the condition that removes it.
- The findings table, shared by every reviewer.

## Judgement

- Judge by the simplest resulting model, not the smallest diff.
- Spend a real attempt on a limitation before working around it.
- The audit steps, all seven.
