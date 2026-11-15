# Sources

## Published

- Robert C. Martin, single responsibility as "one reason to change": https://blog.cleancoder.com/uncle-bob/2014/05/08/SingleReponsibilityPrinciple.html; as "one actor": *Clean Architecture*, 2017, ch. 7.
- Robert C. Martin, open/closed: https://blog.cleancoder.com/uncle-bob/2014/05/12/TheOpenClosedPrinciple.html
- Robert C. Martin, the dependency rule: https://blog.cleancoder.com/uncle-bob/2012/08/13/the-clean-architecture.html
- Robert C. Martin, dependency inversion, Liskov and interface segregation: http://butunclebob.com/ArticleS.UncleBob.PrinciplesOfOod
- David L. Parnas, "On the Criteria To Be Used in Decomposing Systems into Modules", CACM 15(12), 1972: https://doi.org/10.1145/361598.361623
- John Ousterhout, *A Philosophy of Software Design*: deep modules, information leakage, temporal decomposition: https://web.stanford.edu/~ouster/cgi-bin/book.php
- Gary Bernhardt, functional core, imperative shell: https://www.destroyallsoftware.com/screencasts/catalog/functional-core-imperative-shell
- Martin Fowler, *Refactoring*, ch. 3 smells (divergent change, shotgun surgery, feature envy, data clumps, speculative generality, repeated switches, dead code) and the rule of three: https://martinfowler.com/books/refactoring.html
- Martin Fowler, YAGNI: https://martinfowler.com/bliki/Yagni.html
- Sandi Metz, "The Wrong Abstraction": https://sandimetz.com/blog/2016/1/20/the-wrong-abstraction
- Meilir Page-Jones, *What Every Programmer Should Know about Object-Oriented Design*, 1995: connascence, with strength, degree and locality; summarised at https://connascence.io/
- Jim Weirich, "Grand Unified Theory of Software Design", summary by thoughtbot: https://thoughtbot.com/blog/connascence-as-a-vocabulary-to-discuss-coupling
- import-linter contract types: https://import-linter.readthedocs.io/en/stable/contract_types/
- PEP 8, a leading underscore marks internal use: https://peps.python.org/pep-0008/#naming-conventions

## Repo decisions

- The layer map: `pyproject.toml`'s import-linter contract.
- A variation point arrives with the second kind: `AGENTS.md`'s "No Protocol or ABC before a second implementation".
- A fix that adds a layer, flag or wrapper is suspect, and Blender, the standard library and installed packages come first: `AGENTS.md`'s "Blender does anything it can do. Code is a liability."
- Every physical number cites a source: `AGENTS.md`'s conventions.
- An odd shape is justified only by a documented constraint; precedent in the codebase is never a reason.
- `Limitation:` with the condition that removes it.
- The findings table, shared by every reviewer.

## Judgement

- Open/closed counted as branch sites on a kind (closest published idea: Fowler's repeated switches).
- Blender as a message bus; a field or default for one caller; a literal in the shell.
- A shell branch chooses storage, never the value; entry points hold no decision; error boundaries.
- A layer violation never earns an exception; layer purity beats a shortcut.
- A caller that carries an intermediate value has met a shallow module.
- The review steps, and the package audit's steps.
- Judge by the simplest resulting model, not the smallest diff.
- Spend a real attempt on a limitation before working around it.
