import os
import joblib
import numpy as np
import pandas as pd
from typing import Optional, Dict, Any, List
from fastapi import FastAPI, Request, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel, Field

# Base directories
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, "models", "best_model")
STATIC_DIR = os.path.join(BASE_DIR, "static")
TEMPLATES_DIR = os.path.join(BASE_DIR, "templates")

# Initialize FastAPI app
app = FastAPI(
    title="Ames Housing Price Prediction AI",
    description="State-of-the-art Ames Housing Price Valuation powered by CatBoost Gradient Boosting Regressor (R² = 0.904, MAE = $16,528)",
    version="1.0.0"
)

# Ensure directories exist
os.makedirs(STATIC_DIR, exist_ok=True)
os.makedirs(os.path.join(STATIC_DIR, "css"), exist_ok=True)
os.makedirs(os.path.join(STATIC_DIR, "js"), exist_ok=True)
os.makedirs(TEMPLATES_DIR, exist_ok=True)

app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
templates = Jinja2Templates(directory=TEMPLATES_DIR)

# Global model and preprocessor cache
MODEL = None
PREPROCESSOR_BUNDLE = None


def load_artifacts():
    global MODEL, PREPROCESSOR_BUNDLE
    model_path = os.path.join(MODELS_DIR, "CatBoost_model.joblib")
    bundle_path = os.path.join(MODELS_DIR, "preprocessor_bundle.joblib")

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"CatBoost model not found at: {model_path}")
    if not os.path.exists(bundle_path):
        raise FileNotFoundError(f"Preprocessor bundle not found at: {bundle_path}")

    MODEL = joblib.load(model_path)
    PREPROCESSOR_BUNDLE = joblib.load(bundle_path)
    print("[INFO] Model and Preprocessor bundle loaded successfully into memory.")


@app.on_event("startup")
def startup_event():
    load_artifacts()


class HouseFeatures(BaseModel):
    # Numerical specifications
    OverallQual: int = Field(7, ge=1, le=10, description="Overall material and finish quality (1-10)")
    GrLivArea: float = Field(1710, ge=300, le=6000, description="Above grade living area in sq ft")
    TotalBsmtSF: float = Field(856, ge=0, le=4000, description="Total basement area in sq ft")
    FirstFlrSF: float = Field(856, ge=300, le=4000, description="First floor square footage", alias="1stFlrSF")
    SecondFlrSF: float = Field(854, ge=0, le=3000, description="Second floor square footage", alias="2ndFlrSF")
    GarageCars: int = Field(2, ge=0, le=5, description="Size of garage in car capacity")
    FullBath: int = Field(2, ge=0, le=5, description="Full bathrooms above grade")
    HalfBath: int = Field(1, ge=0, le=4, description="Half baths above grade")
    BsmtFullBath: int = Field(0, ge=0, le=4, description="Basement full bathrooms")
    BsmtHalfBath: int = Field(0, ge=0, le=3, description="Basement half bathrooms")
    TotRmsAbvGrd: int = Field(7, ge=2, le=16, description="Total rooms above grade (excluding bathrooms)")
    YearBuilt: int = Field(2003, ge=1870, le=2026, description="Original construction year")
    YrSold: int = Field(2008, ge=2000, le=2030, description="Year property sold/evaluated")

    # Categorical specifications
    Neighborhood: str = Field("CollgCr", description="Ames physical neighborhood location")
    BsmtQual: str = Field("Gd", description="Height of the basement: None, Po, Fa, TA, Gd, Ex")
    KitchenQual: str = Field("Gd", description="Kitchen quality: None, Po, Fa, TA, Gd, Ex")
    ExterQual: str = Field("Gd", description="Exterior material quality: None, Po, Fa, TA, Gd, Ex")
    GarageFinish: str = Field("RFn", description="Interior finish of the garage: None, Unf, RFn, Fin")
    FireplaceQu: str = Field("TA", description="Fireplace quality: None, Po, Fa, TA, Gd, Ex")
    HeatingQC: str = Field("Ex", description="Heating quality and condition: Po, Fa, TA, Gd, Ex")
    BsmtFinType1: str = Field("GLQ", description="Rating of basement finished area: None, Unf, LwQ, Rec, BLQ, ALQ, GLQ")
    MSSubClass: str = Field("60", description="Type of dwelling (e.g., 20 = 1-Story, 60 = 2-Story)")
    GarageType: str = Field("Attchd", description="Garage location: Attchd, Detchd, BuiltIn, Basment, CarPort, 2Types, None")
    Foundation: str = Field("PConc", description="Foundation type: PConc, CBlock, BrkTil, Slab, Stone, Wood")
    Exterior2nd: str = Field("VinylSd", description="Exterior covering on house")
    MasVnrType: str = Field("BrkFace", description="Masonry veneer type: None, BrkFace, Stone, BrkCmn")

    class Config:
        populate_by_name = True


PRESETS: List[Dict[str, Any]] = [
    {
        "id": "luxury",
        "name": "Luxury Executive Villa",
        "tagline": "High-end Northridge Heights 2-story with premium finishes & 3-car garage",
        "icon": "🏰",
        "specs": {
            "OverallQual": 9,
            "GrLivArea": 2850,
            "1stFlrSF": 1600,
            "2ndFlrSF": 1250,
            "TotalBsmtSF": 1550,
            "GarageCars": 3,
            "FullBath": 2,
            "HalfBath": 1,
            "BsmtFullBath": 1,
            "BsmtHalfBath": 0,
            "TotRmsAbvGrd": 9,
            "YearBuilt": 2006,
            "YrSold": 2009,
            "Neighborhood": "NridgHt",
            "BsmtQual": "Ex",
            "KitchenQual": "Ex",
            "ExterQual": "Ex",
            "GarageFinish": "Fin",
            "FireplaceQu": "Ex",
            "HeatingQC": "Ex",
            "BsmtFinType1": "GLQ",
            "MSSubClass": "60",
            "GarageType": "Attchd",
            "Foundation": "PConc",
            "Exterior2nd": "VinylSd",
            "MasVnrType": "Stone"
        }
    },
    {
        "id": "family",
        "name": "Modern Suburban Family Home",
        "tagline": "Turnkey 2-story residence in desirable College Creek neighborhood",
        "icon": "🏡",
        "specs": {
            "OverallQual": 7,
            "GrLivArea": 1780,
            "1stFlrSF": 950,
            "2ndFlrSF": 830,
            "TotalBsmtSF": 950,
            "GarageCars": 2,
            "FullBath": 2,
            "HalfBath": 1,
            "BsmtFullBath": 0,
            "BsmtHalfBath": 0,
            "TotRmsAbvGrd": 7,
            "YearBuilt": 2003,
            "YrSold": 2008,
            "Neighborhood": "CollgCr",
            "BsmtQual": "Gd",
            "KitchenQual": "Gd",
            "ExterQual": "Gd",
            "GarageFinish": "RFn",
            "FireplaceQu": "TA",
            "HeatingQC": "Ex",
            "BsmtFinType1": "GLQ",
            "MSSubClass": "60",
            "GarageType": "Attchd",
            "Foundation": "PConc",
            "Exterior2nd": "VinylSd",
            "MasVnrType": "BrkFace"
        }
    },
    {
        "id": "townhouse",
        "name": "Somerset Urban Townhome",
        "tagline": "Sleek low-maintenance contemporary living near city amenities",
        "icon": "🏙️",
        "specs": {
            "OverallQual": 8,
            "GrLivArea": 1650,
            "1stFlrSF": 820,
            "2ndFlrSF": 830,
            "TotalBsmtSF": 820,
            "GarageCars": 2,
            "FullBath": 2,
            "HalfBath": 1,
            "BsmtFullBath": 0,
            "BsmtHalfBath": 0,
            "TotRmsAbvGrd": 6,
            "YearBuilt": 2005,
            "YrSold": 2009,
            "Neighborhood": "Somerst",
            "BsmtQual": "Gd",
            "KitchenQual": "Gd",
            "ExterQual": "Gd",
            "GarageFinish": "Fin",
            "FireplaceQu": "Gd",
            "HeatingQC": "Ex",
            "BsmtFinType1": "GLQ",
            "MSSubClass": "160",
            "GarageType": "Attchd",
            "Foundation": "PConc",
            "Exterior2nd": "MetalSd",
            "MasVnrType": "None"
        }
    },
    {
        "id": "ranch",
        "name": "North Ames Classic Ranch",
        "tagline": "Solid single-story brick foundation with large lot and attached garage",
        "icon": "🏠",
        "specs": {
            "OverallQual": 6,
            "GrLivArea": 1280,
            "1stFlrSF": 1280,
            "2ndFlrSF": 0,
            "TotalBsmtSF": 1100,
            "GarageCars": 2,
            "FullBath": 1,
            "HalfBath": 1,
            "BsmtFullBath": 1,
            "BsmtHalfBath": 0,
            "TotRmsAbvGrd": 6,
            "YearBuilt": 1972,
            "YrSold": 2008,
            "Neighborhood": "NAmes",
            "BsmtQual": "TA",
            "KitchenQual": "TA",
            "ExterQual": "TA",
            "GarageFinish": "Unf",
            "FireplaceQu": "TA",
            "HeatingQC": "TA",
            "BsmtFinType1": "ALQ",
            "MSSubClass": "20",
            "GarageType": "Attchd",
            "Foundation": "CBlock",
            "Exterior2nd": "HdBoard",
            "MasVnrType": "BrkFace"
        }
    },
    {
        "id": "bungalow",
        "name": "Historic Old Town Craftsman",
        "tagline": "Character-rich 1930s character home with original architectural detailing",
        "icon": "🪵",
        "specs": {
            "OverallQual": 5,
            "GrLivArea": 1100,
            "1stFlrSF": 750,
            "2ndFlrSF": 350,
            "TotalBsmtSF": 750,
            "GarageCars": 1,
            "FullBath": 1,
            "HalfBath": 0,
            "BsmtFullBath": 0,
            "BsmtHalfBath": 0,
            "TotRmsAbvGrd": 5,
            "YearBuilt": 1935,
            "YrSold": 2007,
            "Neighborhood": "OldTown",
            "BsmtQual": "TA",
            "KitchenQual": "TA",
            "ExterQual": "TA",
            "GarageFinish": "Unf",
            "FireplaceQu": "None",
            "HeatingQC": "TA",
            "BsmtFinType1": "Unf",
            "MSSubClass": "50",
            "GarageType": "Detchd",
            "Foundation": "BrkTil",
            "Exterior2nd": "Wd Sdng",
            "MasVnrType": "None"
        }
    }
]


def safe_float(val, default=0.0):
    try:
        if val is None or val == "":
            return default
        return float(val)
    except Exception:
        return default


def safe_int(val, default=0):
    try:
        if val is None or val == "":
            return default
        return int(float(val))
    except Exception:
        return default


def predict_pipeline(data: Dict[str, Any]) -> Dict[str, Any]:
    if MODEL is None or PREPROCESSOR_BUNDLE is None:
        load_artifacts()

    preprocessor = PREPROCESSOR_BUNDLE["preprocessor"]
    clean_cols = PREPROCESSOR_BUNDLE["clean_feature_names"]
    num_features = PREPROCESSOR_BUNDLE["final_numerical_features"]
    cat_features = PREPROCESSOR_BUNDLE["final_categorical_features"]

    # Compute engineered features safely
    bsmt_sf = safe_float(data.get("TotalBsmtSF"), 856.0)
    flr1_sf = safe_float(data.get("1stFlrSF"), 856.0)
    flr2_sf = safe_float(data.get("2ndFlrSF"), 854.0)
    total_sf = bsmt_sf + flr1_sf + flr2_sf

    full_bath = safe_float(data.get("FullBath"), 2.0)
    half_bath = safe_float(data.get("HalfBath"), 1.0)
    bsmt_full = safe_float(data.get("BsmtFullBath"), 0.0)
    bsmt_half = safe_float(data.get("BsmtHalfBath"), 0.0)
    total_bath = full_bath + 0.5 * half_bath + bsmt_full + 0.5 * bsmt_half

    yr_sold = safe_int(data.get("YrSold"), 2008)
    yr_built = safe_int(data.get("YearBuilt"), 2003)
    house_age = max(0, yr_sold - yr_built)

    overall_qual = safe_int(data.get("OverallQual"), 7)
    overall_qual_total_sf = overall_qual * total_sf

    row = {
        "OverallQual_TotalSF": overall_qual_total_sf,
        "OverallQual": overall_qual,
        "TotalSF": total_sf,
        "GrLivArea": safe_float(data.get("GrLivArea"), 1710.0),
        "GarageCars": safe_int(data.get("GarageCars"), 2),
        "TotalBathrooms": total_bath,
        "TotalBsmtSF": bsmt_sf,
        "1stFlrSF": flr1_sf,
        "FullBath": int(full_bath),
        "TotRmsAbvGrd": safe_int(data.get("TotRmsAbvGrd"), 7),
        "HouseAge": house_age,
        # Categorical features
        "Neighborhood": str(data.get("Neighborhood", "CollgCr")),
        "BsmtQual": str(data.get("BsmtQual", "Gd")),
        "KitchenQual": str(data.get("KitchenQual", "Gd")),
        "ExterQual": str(data.get("ExterQual", "Gd")),
        "MSSubClass": str(data.get("MSSubClass", "60")),
        "GarageFinish": str(data.get("GarageFinish", "RFn")),
        "FireplaceQu": str(data.get("FireplaceQu", "TA")),
        "GarageType": str(data.get("GarageType", "Attchd")),
        "Foundation": str(data.get("Foundation", "PConc")),
        "HeatingQC": str(data.get("HeatingQC", "Ex")),
        "Exterior2nd": str(data.get("Exterior2nd", "VinylSd")),
        "BsmtFinType1": str(data.get("BsmtFinType1", "GLQ")),
        "MasVnrType": str(data.get("MasVnrType", "BrkFace"))
    }

    df_single = pd.DataFrame([row])
    feature_columns = num_features + cat_features
    X_input = df_single[feature_columns]

    # Preprocess
    proc_matrix = preprocessor.transform(X_input)
    proc_df = pd.DataFrame(proc_matrix, columns=clean_cols)

    # Predict log scale then invert via expm1
    pred_log = float(MODEL.predict(proc_df)[0])
    predicted_price = float(np.expm1(pred_log))

    # Calculate valuation context
    test_mae = 16528.23  # Empirical holdout test MAE from CatBoost evaluation
    price_low = max(10000.0, predicted_price - test_mae)
    price_high = predicted_price + test_mae
    price_per_sqft = (predicted_price / total_sf) if total_sf > 0 else 0.0

    # Rough 30-year fixed estimated monthly payment (6.5% interest rate, 20% down)
    loan_amount = predicted_price * 0.8
    monthly_rate = 0.065 / 12
    num_payments = 360
    monthly_mortgage = (
        loan_amount * (monthly_rate * (1 + monthly_rate) ** num_payments)
        / ((1 + monthly_rate) ** num_payments - 1)
    )

    return {
        "predicted_price": round(predicted_price, 2),
        "predicted_price_formatted": f"${predicted_price:,.0f}",
        "price_low": round(price_low, 2),
        "price_high": round(price_high, 2),
        "price_range_formatted": f"${price_low:,.0f} - ${price_high:,.0f}",
        "price_per_sqft": round(price_per_sqft, 2),
        "price_per_sqft_formatted": f"${price_per_sqft:,.2f}",
        "monthly_mortgage_estimate": round(monthly_mortgage, 2),
        "monthly_mortgage_formatted": f"${monthly_mortgage:,.0f}/mo",
        "total_sf": round(total_sf, 1),
        "total_bathrooms": round(total_bath, 1),
        "house_age": house_age,
        "quality_score": overall_qual,
        "model_mae": test_mae,
        "model_r2": 0.9041
    }


@app.get("/favicon.ico")
async def favicon():
    from fastapi.responses import Response
    return Response(content=b"", media_type="image/x-icon")


@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse(
        request=request,
        name="index.html",
        context={
            "presets": PRESETS,
            "model_info": {
                "name": "CatBoost Regressor",
                "r2": "90.41%",
                "mae": "$16,528",
                "features_count": 91
            }
        }
    )


@app.post("/api/predict")
async def predict_endpoint(payload: Dict[str, Any]):
    try:
        result = predict_pipeline(payload)
        return JSONResponse(content={"status": "success", "data": result})
    except Exception as e:
        print(f"[ERROR] Prediction error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/presets")
async def get_presets():
    return JSONResponse(content={"status": "success", "presets": PRESETS})


@app.get("/api/model-info")
async def get_model_info():
    top_features = [
        {"feature": "Overall Quality × Total Area", "importance": 25.10, "desc": "Compounding effect of luxury finish and spacious square footage"},
        {"feature": "Total Square Footage (TotalSF)", "importance": 8.10, "desc": "Combined habitable living and foundation floor area"},
        {"feature": "Total Bathrooms", "importance": 6.21, "desc": "Aggregate weighted count of full & half bathrooms"},
        {"feature": "Overall Quality Rating", "importance": 5.61, "desc": "1-10 craftsman rating of structural construction and materials"},
        {"feature": "House Age (YrSold - YearBuilt)", "importance": 4.58, "desc": "Property vintage depreciation and architectural era"},
        {"feature": "Fireplace Quality", "importance": 4.47, "desc": "Presence and masonry condition of hearth fireplaces"},
        {"feature": "Above Grade Living Area (GrLivArea)", "importance": 4.47, "desc": "Heated upper floor living space"},
        {"feature": "Garage Car Capacity", "importance": 4.11, "desc": "Number of vehicles accommodated"},
        {"feature": "Finished Basement Rating", "importance": 3.61, "desc": "Quality of lower level recreation finish"},
        {"feature": "Total Basement Area", "importance": 3.32, "desc": "Underground structural foundation area"}
    ]
    return JSONResponse(content={
        "status": "success",
        "model": "CatBoost Regressor (Gradient Boosting)",
        "r2_score": 0.9041,
        "rmse": 27116.86,
        "mae": 16528.23,
        "input_features": 91,
        "training_framework": "CatBoost 1.2+ with Scikit-Learn Pipeline",
        "target_variable": "np.log1p(SalePrice)",
        "top_features": top_features
    })


@app.get("/health")
async def health_check():
    return {"status": "ok", "model_loaded": MODEL is not None}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="127.0.0.1", port=8000, reload=True)
