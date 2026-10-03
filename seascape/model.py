from pydantic import BaseModel, ConfigDict


class Model(BaseModel):
    """Strictness shared by everything this package parses from TOML."""

    # extra: a typo in a scenario is otherwise a silent wrong render.
    # inf_nan: tomllib parses `nan` and `inf`; a nan bearing renders a camera pointing
    # nowhere and reports no error.
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)
