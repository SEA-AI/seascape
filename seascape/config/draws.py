"""A distribution in place of a value, and the seeded streams it draws from."""

import math
from typing import Annotated, Any

import numpy as np
from pydantic import (
    Field,
    model_validator,
)

from seascape.model import Model


def substream(seed: int, name: str) -> np.random.Generator:
    """A named substream, so adding a component cannot perturb an existing one."""
    return np.random.default_rng([seed, *name.encode()])


class Uniform(Model):
    """A float drawn evenly between two bounds."""

    uniform: tuple[float, float] = Field(description="The bounds.")

    def draw(self, rng: np.random.Generator) -> float:
        return float(rng.uniform(*self.uniform))


class LogUniform(Model):
    """A float drawn evenly in its logarithm between two bounds: as often between 0.1
    and 1 as between 1 and 10."""

    log_uniform: tuple[
        Annotated[float, Field(gt=0.0)], Annotated[float, Field(gt=0.0)]
    ] = Field(description="The bounds, lower first.")

    @model_validator(mode="after")
    def _lower_first(self) -> "LogUniform":
        lower, upper = self.log_uniform
        if lower > upper:
            raise ValueError(f"log_uniform bounds {self.log_uniform} are reversed")
        return self

    def draw(self, rng: np.random.Generator) -> float:
        # exp(log(x)) can land an ulp below x.
        return float(
            np.clip(np.exp(rng.uniform(*np.log(self.log_uniform))), *self.log_uniform)
        )


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
        return float(rng.triangular(*self.triangular))


class Integer(Model):
    """A whole number drawn evenly between two bounds."""

    integer: tuple[int, int] = Field(description="The bounds, both included.")

    def draw(self, rng: np.random.Generator) -> int:
        return int(rng.integers(*self.integer, endpoint=True))


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
        if self.weights is None:
            return self.choice[rng.integers(len(self.choice))]
        p = np.array(self.weights) / sum(self.weights)
        return self.choice[rng.choice(len(self.choice), p=p)]


class OpticalDepth(Model):
    """An object's `range_m` as optical depth: a ray keeps exp(-depth) of its light,
    with `sky.visibility_km` taken as Koschmieder's 2%."""

    optical_depth: tuple[
        Annotated[float, Field(gt=0.0)], Annotated[float, Field(gt=0.0)]
    ] = Field(description="The bounds, lower first, drawn evenly.")

    @model_validator(mode="after")
    def _lower_first(self) -> "OpticalDepth":
        lower, upper = self.optical_depth
        if lower > upper:
            raise ValueError(f"optical_depth bounds {self.optical_depth} are reversed")
        return self


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
