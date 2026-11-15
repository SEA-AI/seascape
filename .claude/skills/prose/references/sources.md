# Sources

## Published

| Rule | Source |
|---|---|
| Delete first; omit needless words | Strunk & White (1918), rule 13 |
| No hedges | Strunk & White (1918), rule 11 |
| The specific over the general | Strunk & White (1918), rule 12 |
| Comments say why; a better name beats a comment | Google Python style guide §3.8: https://google.github.io/styleguide/pyguide.html |
| `Args:` entries | Google §3.8.3 |
| Docs describe the thing as it is; no "now" or "no longer" | Google developer style, timeless documentation: https://developers.google.com/style/timeless-documentation |
| Conventional Commits | https://www.conventionalcommits.org/en/v1.0.0/; `chore` and `ci` from `@commitlint/config-conventional` |
| Imperative subject; the body says why | git, SubmittingPatches: https://git-scm.com/docs/SubmittingPatches |
| Tidying and behaviour change apart | Google eng-practices, small CLs: https://google.github.io/eng-practices/review/developer/small-cls.html |
| A PR body describes the change against its base | Google eng-practices, CL descriptions: https://google.github.io/eng-practices/review/developer/cl-descriptions.html |

## Repo decisions

- A frame over a paragraph: `AGENTS.md`'s "Renders are cheap and settle arguments".
- A number's source beside it: `AGENTS.md`'s conventions.
- Single backticks, no Sphinx roles.
- PR sections What, Why, How, Testing.
- The findings table, shared by every reviewer.

## Judgement

- What goes stale: derived numbers, copied lists, test names, hardware.
- No restating the diff; no journey; no redundancy; a table over a paragraph.
- Keep what the code can't say; a judgement says "Judgement:".
- Docstrings match the signature; a stale backticked name.
- The tics: "X, not just Y", lists padded to three, tacked-on "-ing" clauses, em-dash habits. These are AI-prose tells.
- A rephrase is never longer than what it replaces.
- Be picky; when in doubt, delete.
