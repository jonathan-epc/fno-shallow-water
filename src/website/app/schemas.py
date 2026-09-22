# src/website/app/schemas.py

from pydantic import BaseModel


class ModelInput(BaseModel):
    """Defines the structure of the data the API expects to receive."""

    # The direct problem model expects 4 scalar features: H0, Q0, n, nut
    scalar_features: list[float]
    # The direct problem model expects 1 field (Bed Geometry 'B')
    # with 11x401 = 4411 points, flattened into a single list.
    field_data_flat: list[float]


class ModelOutput(BaseModel):
    """Defines the structure of the data the API will send back."""

    # The direct problem model predicts 3 scalar outputs (empty in this case)
    scalar_predictions: list[float] | None = None
    # And 3 field outputs (H, U, V)
    field_predictions_info: str | None = None
    field_predictions_shape: list[int] | None = None
    # Flattened field data for visualization: {"H": [...], "U": [...]}
    field_predictions_flat: dict[str, list[float]] | None = None
    # Optional error details
    error: str | None = None
