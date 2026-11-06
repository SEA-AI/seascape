"""A distribution in place of a value, and the seeded streams it draws from."""

import math
from typing import Annotated, Any

import numpy as np
from pydantic import AfterValidator, Field, model_validator

from seascape.model import Model


def substream(seed: int, name: str) -> np.random.Generator:
    """A named substream, so adding a component cannot perturb an existing one."""
    return np.random.default_rng([seed, *name.encode()])


def _require_ordered(bounds: tuple[float, float]) -> tuple[float, float]:
    if bounds[0] > bounds[1]:
        raise ValueError(f"bounds {bounds} are reversed")
    return bounds


_Positive = Annotated[float, Field(gt=0.0)]
_Share = Annotated[float, Field(ge=0.0, le=1.0)]
_Ordered = AfterValidator(_require_ordered)


class Uniform(Model):
    """A float drawn evenly between two bounds."""

    uniform: tuple[float, float] = Field(description="The bounds.")

    def draw(self, rng: np.random.Generator) -> float:
        """One value, from `rng`."""
        return float(rng.uniform(*self.uniform))

    def bounds(self) -> tuple[float, float]:
        """The lowest and highest value it can draw."""
        return min(self.uniform), max(self.uniform)


class LogUniform(Model):
    """A float drawn evenly in its logarithm between two bounds: as often between 0.1
    and 1 as between 1 and 10."""

    log_uniform: Annotated[tuple[_Positive, _Positive], _Ordered] = Field(
        description="The bounds, lower first."
    )

    def draw(self, rng: np.random.Generator) -> float:
        """One value, from `rng`."""
        # exp(log(x)) can land an ulp below x.
        return float(
            np.clip(np.exp(rng.uniform(*np.log(self.log_uniform))), *self.log_uniform)
        )

    def bounds(self) -> tuple[float, float]:
        """The lowest and highest value it can draw."""
        return self.log_uniform


class Triangular(Model):
    """A float drawn from a triangular density peaking at the mode, a bound or not."""

    triangular: tuple[float, float, float] = Field(
        description="The lower bound, the mode and the upper bound."
    )

    @model_validator(mode="after")
    def _in_order(self) -> "Triangular":
        low, mode, high = self.triangular
        if low == high:
            raise ValueError(f"triangular {self.triangular} has equal bounds")
        if not low <= mode <= high:
            raise ValueError(f"triangular {self.triangular} is not low <= mode <= high")
        return self

    def draw(self, rng: np.random.Generator) -> float:
        """One value, from `rng`."""
        return float(rng.triangular(*self.triangular))

    def bounds(self) -> tuple[float, float]:
        """The lowest and highest value it can draw."""
        return self.triangular[0], self.triangular[2]


class Integer(Model):
    """A whole number drawn evenly between two bounds."""

    integer: tuple[int, int] = Field(description="The bounds, both included.")

    def draw(self, rng: np.random.Generator) -> int:
        """One value, from `rng`."""
        return int(rng.integers(*self.integer, endpoint=True))

    def bounds(self) -> tuple[float, float]:
        """The lowest and highest value it can draw."""
        return min(self.integer), max(self.integer)


class Choice(Model):
    """One of several values, each as likely unless weighted."""

    choice: list[Any] = Field(min_length=1, description="The values.")
    weights: list[Annotated[float, Field(ge=0.0)]] | None = Field(
        default=None,
        description="How likely each value is, relative to the others: [8, 2] is 80 "
        "and 20 percent.",
    )

    @model_validator(mode="after")
    def _a_weight_per_value(self) -> "Choice":
        if self.weights is None:
            return self
        if len(self.weights) != len(self.choice):
            raise ValueError(
                f"{len(self.weights)} weights for {len(self.choice)} values"
            )
        total = sum(self.weights)
        if not math.isfinite(total):
            raise ValueError(f"the weights sum to {total}")
        if not total:
            raise ValueError("the weights sum to 0")
        return self

    def draw(self, rng: np.random.Generator) -> Any:
        """One value, from `rng`."""
        if self.weights is None:
            return self.choice[rng.integers(len(self.choice))]
        p = np.array(self.weights) / sum(self.weights)
        return self.choice[rng.choice(len(self.choice), p=p)]

    def possible(self) -> list[Any]:
        """The values it can draw: those of no weight never come up."""
        if self.weights is None:
            return self.choice
        return [v for v, w in zip(self.choice, self.weights, strict=True) if w > 0]


class OpticalDepth(Model):
    """An object's `range_m` as optical depth: a ray keeps exp(-depth) of its light,
    with a dark target keeping 5% of its contrast over `sky.visibility_km`."""

    optical_depth: Annotated[tuple[_Positive, _Positive], _Ordered] = Field(
        description="The bounds, lower first, drawn evenly."
    )


class Loaded(Model):
    """An object's `draught_m` as how laden it is: 0 its asset's lightest draught, 1
    its deepest, so one draw fits every asset a choice can make."""

    loaded: Annotated[tuple[_Share, _Share], _Ordered] = Field(
        description="The bounds, lower first, drawn evenly."
    )


DRAWS: dict[str, type[Uniform | LogUniform | Triangular | Integer | Choice]] = {
    "uniform": Uniform,
    "log_uniform": LogUniform,
    "triangular": Triangular,
    "integer": Integer,
    "choice": Choice,
}


def kind(node: Any) -> str | None:
    """`node`'s draw key, if it holds one and nothing but that draw's fields."""
    if not isinstance(node, dict):
        return None
    kinds = [key for key in node if key in DRAWS]
    if len(kinds) == 1 and set(node) <= set(DRAWS[kinds[0]].model_fields):
        return kinds[0]
    return None
