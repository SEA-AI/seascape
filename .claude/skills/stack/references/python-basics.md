# Python basics

General practice any Python reviewer knows; the `stack` body holds the rules this repo's code breaks most.

## Python

- **Default arguments are evaluated once.** `None` plus a body check, or `field(default_factory=...)`.
- **Loop over items, not indices:** `enumerate`, `zip(strict=True)`, `.items()`. Needing an index usually means the structure is wrong.
- **A generator beats a list the caller only iterates.**
- **`@override` on overrides**, so a renamed parent method fails the checker.

## Numeric code

- **Rewrite formulas that lose precision.** `log1p` and `expm1` keep the digits `log(1 + x)` and `exp(x) - 1` drop for a small `x`; `hypot` avoids overflow in `sqrt(x*x + y*y)`; `arctan2` keeps the quadrant `arctan(y / x)` loses.

## CLI

- **A CLI command writes data to stdout**, diagnostics to stderr, and exits 0 on success.
