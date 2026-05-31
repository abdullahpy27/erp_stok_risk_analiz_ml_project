from fastapi import FastAPI
from fastapi import HTTPException
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
from pathlib import Path

try:
    import matlab.engine
except ModuleNotFoundError:
    matlab = None

app = FastAPI()

# Enable CORS for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR.parent / "model"
PREDICT_SCRIPT = MODEL_DIR / "erp_predict.m"
MODEL_FILE = MODEL_DIR / "best_model.mat"

eng = None


def get_model_status():
    return {
        "matlab_engine_installed": matlab is not None,
        "model_dir": str(MODEL_DIR),
        "predict_script_exists": PREDICT_SCRIPT.exists(),
        "model_file_exists": MODEL_FILE.exists(),
    }


def get_matlab_engine():
    global eng

    missing_files = [
        str(path) for path in (PREDICT_SCRIPT, MODEL_FILE) if not path.exists()
    ]
    if missing_files:
        raise HTTPException(
            status_code=500,
            detail=f"Missing MATLAB model file(s): {', '.join(missing_files)}",
        )

    if matlab is None:
        raise HTTPException(
            status_code=503,
            detail=(
                "MATLAB Python engine is not installed for this Python. "
                "Install it from your MATLAB installation, then restart the API."
            ),
        )

    if eng is None:
        eng = matlab.engine.start_matlab()
        eng.addpath(str(MODEL_DIR), nargout=0)
        eng.cd(str(MODEL_DIR), nargout=0)

    return eng

class InputData(BaseModel):
    stock_quantity: float
    weekly_sales: float
    supplier_delay_days: float
    price: float
    last_sale_days: float
    seasonality: float


@app.post("/predict")
def predict(data: InputData):
    engine = get_matlab_engine()

    try:
        result = engine.erp_predict(
            float(data.stock_quantity),
            float(data.weekly_sales),
            float(data.supplier_delay_days),
            float(data.price),
            float(data.last_sale_days),
            float(data.seasonality),
        )
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"MATLAB prediction failed: {exc}",
        ) from exc

    return {
        "risk_level": str(result),
        "used_model": "Random Forest",
        "selected_model": "Best",
        "recommendation": f"Predicted stock risk is {result}.",
    }


@app.get("/")
def root():
    return {
        "status": "ok",
        **get_model_status(),
    }


@app.get("/health")
def health():
    return {
        "status": "ok",
        **get_model_status(),
    }
