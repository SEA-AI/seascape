# Auditing a package

1. Read `AGENTS.md` and write the scope and skip list down first. Audit the tip of the branch the fixes land on.
2. One reader per package, with the same brief and compact output, running the skill's Reviewing steps on the package's public names. Read whole files, not excerpts.
3. Before calling code unused, search the whole repo, string references included.
4. Before accepting a replacement, run old and new on random inputs and compare, or render both and compare pixels against the render's own noise floor. Say "unverified" where you did not.
5. Count the helper a fix adds: a dedupe that adds as much as it removes is not a finding.
6. Regenerate committed outputs, such as doc images, and diff them; a fix that alters them is a behaviour change, so report it as one.
7. Record each keep with its reason in the report, so nobody audits it twice.
