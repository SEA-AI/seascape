# Questions per principle

Ask these where the `architecture` review steps found friction.

- **Single responsibility (Martin: one reason to change; one actor).** Name the change that forces an edit here. Two unrelated ones is divergent change: split. One change that forces edits in many modules is shotgun surgery: a decision scattered from its home.
- **Information hiding (Parnas).** Which design decision does this module hide? A module cut by processing step (Ousterhout's temporal decomposition) leaks the same knowledge into every step. Cut by the decision likely to change.
- **Deep modules (Ousterhout).** Is the interface much smaller than the work behind it? A function whose callers carry an intermediate value from one call to the next, or recover a result it hid, has exposed its internals. Return the finished thing.
- **Open/closed.** Adding a kind should touch the union and that kind's own code. Behaviour that varies by kind is a method on the member or one dispatch beside the union. Count the places that branch on the kind: more than one is the smell.
- **Liskov.** Ask it of union members and of any callable type that several producers return: can every caller use every member without checking which one it got?
- **Interface segregation.** Does a caller depend on a type wider than the slice it uses, so unrelated changes reach it? Pass the narrowest existing type that carries what it reads: a sub-model, or a field when it reads one. Never break a cohesive record into loose primitives.
- **Dependency inversion (Martin).** When the core must call out, to Blender, a file or a clock, the core owns the interface it needs: a `Callable` type in the core's layer, which the shell implements. Better still, pass values so no port is needed.
- **Functional core, imperative shell (Bernhardt).** Could this branch run on plain values and be tested without Blender? Then it belongs below the shell.
- **Connascence (Page-Jones, Weirich).** Name the form and rate it: strength (name is the weakest, and dynamic is worse than static), degree (how many places), locality (how far apart). A string key shared across layers is connascence of name that no tool checks: one constant or one typed field. A key order is connascence of position: named fields. A formula in two places is connascence of algorithm: one home. Strong forms belong inside one function.
- **Package interface.** Does a module import another module's `_private` name? The name is either public, or the caller is in the wrong place. import-linter does not see this.
- **Error boundaries.** The shell turns an engine failure into a domain error; an entry point decides the exit code. No core function catches a shell's error.
