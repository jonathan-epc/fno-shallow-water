# src/website/app/main.py
import math
import os
from pathlib import Path

import h5py
import numpy as np
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

# We will create a model handler to manage all 12 models
from .model_loader import ModelHandler
from .schemas import ModelInput, ModelOutput

app = FastAPI(title="FNO Surrogate Model API")
BASE_DIR = Path(__file__).resolve().parent.parent
STATIC_DIR = BASE_DIR / "static"
TEMPLATES_DIR = BASE_DIR / "templates"

# --- Initialize Model Handler ---
# This will find and load all models in the models_store on startup
model_handler = ModelHandler(model_store_path=BASE_DIR / "models_store")

# --- Mount Static Files & Templates ---
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
templates = Jinja2Templates(directory=TEMPLATES_DIR)


# --- API Endpoints ---
@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


@app.get("/models")
async def get_models():
    """Returns a list of available models and their configs."""
    return model_handler.get_available_models()


@app.post("/predict/{model_key}", response_model=ModelOutput)
async def predict(model_key: str, data: ModelInput):
    """Endpoint to run prediction for a specific model."""
    try:
        results = model_handler.predict(model_key, data.dict())
        return ModelOutput(**results)
    except ValueError as ve:
        raise HTTPException(status_code=400, detail=str(ve)) from ve
    except KeyError:
        raise HTTPException(
            status_code=404, detail=f"Model key '{model_key}' not found."
        ) from None
    except Exception as e:
        import traceback

        print(f"Error during prediction: {e}")
        traceback.print_exc()
        raise HTTPException(
            status_code=500, detail=f"An internal error occurred: {str(e)}"
        ) from e


@app.get("/health")
async def health_check():
    return {"status": "ok", "models_loaded": len(model_handler.get_available_models())}


@app.get("/datasets")
async def get_datasets():
    """Returns a list of available datasets and their number of cases."""
    dat_dir = BASE_DIR.parent.parent / "data"
    datasets = []
    if not dat_dir.exists():
        return datasets

    for file in os.listdir(dat_dir):
        if file.endswith(".hdf5"):
            file_path = dat_dir / file
            try:
                with h5py.File(file_path, "r") as f:
                    keys = [k for k in f if k != "statistics"]
                    datasets.append({"name": file, "cases_count": len(keys)})
            except Exception as e:
                print(f"Failed to read {file}: {e}")
    return datasets


@app.get("/dataset/{dataset_name}/case/{index}")
async def get_dataset_case(dataset_name: str, index: int):
    """Returns a specific case from a dataset."""
    dat_dir = BASE_DIR.parent.parent / "data"
    file_path = dat_dir / dataset_name

    if not file_path.exists() or not file_path.is_file():
        raise HTTPException(
            status_code=404, detail=f"Dataset {dataset_name} not found."
        )

    try:
        with h5py.File(file_path, "r") as f:
            keys = [k for k in f if k != "statistics"]
            if index < 0 or index >= len(keys):
                raise HTTPException(
                    status_code=404, detail=f"Case index {index} out of bounds."
                )

            case_key = keys[index]
            group = f[case_key]

            scalars = {}
            for k, v in group.attrs.items():
                try:
                    val = float(v)
                    if math.isnan(val) or math.isinf(val):
                        val = 0.0
                    scalars[k] = val
                except (ValueError, TypeError):
                    if isinstance(v, bytes):
                        scalars[k] = v.decode("utf-8")
                    else:
                        scalars[k] = str(v)

            # Additional scalars we might need
            scalars["L"] = float(
                group.attrs.get("L", 6.1)
            )  # Default channel length if missing
            scalars["W"] = float(
                group.attrs.get("W", 0.3)
            )  # Default channel width if missing

            fields = {}
            for k in group:
                data = group[k][()]
                # Convert to list and flatten, replacing NaNs and Infs with 0.0
                fields[k] = np.nan_to_num(data, nan=0.0, posinf=0.0, neginf=0.0).flatten().tolist()

            return {
                "dataset": dataset_name,
                "case_index": index,
                "case_key": case_key,
                "scalars": scalars,
                "fields": fields,
            }
    except HTTPException:
        raise
    except Exception as e:
        import traceback

        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e)) from e
