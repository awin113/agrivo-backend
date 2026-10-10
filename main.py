import io
import os
from typing import List, Optional, Dict, Any

from fastapi import FastAPI, File, UploadFile, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from PIL import Image
import numpy as np

# Load TFLite runtime safely
try:
    import tflite_runtime.interpreter as tflite
except ImportError:
    try:
        import tensorflow.lite as tflite
    except ImportError:
        import tensorflow as tf
        tflite = tf.lite

# Initialize FastAPI App
app = FastAPI(
    title="AGRIVO Soil Classification & Crop Recommendation API",
    description="Deep Learning API for mobile image-based soil categorization and agronomic recommendations using hybrid ResNet50-MobileNetV3.",
    version="1.0.0"
)

# Enable CORS for React Native / Web clients
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static files for crop reference images
STATIC_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "static")
if os.path.exists(STATIC_DIR):
    app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

# Load model at startup
MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "soil_classifier.tflite")
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"Model file not found at {MODEL_PATH}")

interpreter = tflite.Interpreter(model_path=MODEL_PATH)
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

CLASS_NAMES = ["Clay Loam", "Loam", "Nonetype", "Vermicast"]

# Comprehensive Agronomic Database for Crops and Soil Profiles
# Directly mirroring image-classification-app-v4.py:
# - ClayLoamScreen (lines 281-330)
# - LoamScreen (lines 332-387)
# - VermicastScreen (lines 388-435)
AGRONOMIC_DATABASE: Dict[str, Dict[str, Any]] = {
    "Clay Loam": {
        "display_name": "Clay Loam",
        "description": "Clay loam is a versatile soil blend that amalgamates the characteristics of clay and loam. It constitutes a harmonious fusion of sand, silt, clay, and organic matter. The incorporation of rice husk may enrich the soil’s physical and chemical attributes.",
        "management_tip": "These plants thrive in loam mixed with rice husk and provide nutrition, and sustainability.",
        "composition": {
            "sand": "20-35%",
            "silt": "28-50%",
            "clay": "20-35%",
            "organic_matter": "2-5%",
            "other": "Varies"
        },
        "crops": [
            {
                "id": "tomato",
                "name": "Tomato",
                "scientific_name": "Solanum lycopersicum",
                "optimal_ph": "6.0 - 7.0",
                "season": "Spring / Warm climates",
                "tip": "Ideal for cultivation in regions with warm climates. Flourishes when planted in clay loam mixed with rice husk during spring. Requires soil pH levels ranging between 6 and 7 for optimal growth.",
                "image_filename": "tomato.jpg"
            },
            {
                "id": "bell_pepper",
                "name": "Bell Peppers",
                "scientific_name": "Capsicum annuum",
                "optimal_ph": "6.0 - 7.0",
                "season": "Throughout the growing season",
                "tip": "Suitable for cultivation throughout the growing season in clay loam mixed soil. Ensure proper drainage to prevent waterlogging, a common issue in clay soils. Maintain soil moisture consistently, especially during the flowering and fruiting stages.",
                "image_filename": "bell_pepper.jpg"
            },
            {
                "id": "beans",
                "name": "Beans",
                "scientific_name": "Phaseolus vulgaris",
                "optimal_ph": "6.0 - 7.0",
                "season": "After the last frost date",
                "tip": "Beans thrive in clay loam mixed with rice husk, providing a sturdy support system for climbing varieties. Plant seeds directly into well-prepared clay loam mixed soil after the last frost date. Adequate sunlight and regular watering are essential for healthy bean production.",
                "image_filename": "beans.jpg"
            }
        ]
    },
    "Loam": {
        "display_name": "Loam Mix",
        "description": "Loam mix is a versatile soil blend that combines the properties of loam with the added benefit of rice husk. It consists of a balanced mixture of sand, silt, clay, organic matter, and rice husk particles. The inclusion of rice husk enhances the soil’s physical and chemical properties.",
        "management_tip": "These plants thrive in loam mixed with rice husk and provide nutrition, and sustainability.",
        "composition": {
            "sand": "40-50%",
            "silt": "30-40%",
            "clay": "20-30%",
            "organic_matter": "2-5%",
            "other": "Varies"
        },
        "crops": [
            {
                "id": "ampalaya",
                "name": "Ampalaya (Bitter Gourd)",
                "scientific_name": "Momordica charantia",
                "optimal_ph": "6.0 - 6.7",
                "season": "Year-round (best October to February)",
                "tip": "Suitable for year-round cultivation in the Philippines. Thrives best if grown from October to February. Plant it in soil with pH levels between 6 and 6.7.",
                "image_filename": "ampalaya.jpg"
            },
            {
                "id": "lettuce",
                "name": "Lettuce",
                "scientific_name": "Lactuca sativa",
                "optimal_ph": "6.0 - 7.0",
                "season": "All summer long",
                "tip": "Easy-to-grow garden vegetables that you can harvest all summer long. Thrives in well-draining loam soil. Keep the soil moist, especially during hot and dry weather.",
                "image_filename": "lettuce.jpg"
            },
            {
                "id": "carrot",
                "name": "Carrots",
                "scientific_name": "Daucus carota",
                "optimal_ph": "6.0 - 7.0",
                "season": "Direct seeding in loam soil",
                "tip": "Carrots love loam soil due to its excellent drainage and nutrient retention. Plant them directly in the loam-mixed soil.",
                "image_filename": "carrot.jpg"
            },
            {
                "id": "beans",
                "name": "Beans (Strawberries)",
                "scientific_name": "Fragaria × ananassa / Phaseolus",
                "optimal_ph": "6.0 - 7.0",
                "season": "Containers or raised beds",
                "tip": "Bright red and juicy strawberries thrive in loam-mixed soil. Plant them in containers or raised beds.",
                "image_filename": "beans.jpg"
            }
        ]
    },
    "Vermicast": {
        "display_name": "Peat / Vermicast",
        "description": "Peat is the surface organic layer of a soil that consists of partially decomposed organic matter, derived mostly from plant material, which has accumulated under conditions of waterlogging, oxygen deficiency, high acidity and nutrient deficiency.",
        "management_tip": "Here are some crops that thrive when grown using peat / vermicast-enriched soil.",
        "composition": {
            "organic_matter": "45-60%",
            "nitrogen_n": "1-3%",
            "phosphorus_p": "0.5-2%",
            "potassium_k": "0.5-2%",
            "calcium_ca": "5-15%",
            "magnesium_mg": "0.5-3%",
            "other": "Variable"
        },
        "crops": [
            {
                "id": "pechay",
                "name": "Pechay (Chinese Cabbage)",
                "scientific_name": "Brassica rapa",
                "optimal_ph": "6.0 - 7.5",
                "season": "Year-round in Filipino cuisine",
                "tip": "A leafy green vegetable commonly used in Filipino cuisine. Grows well in various soil types, including vermicast-enriched soil. Harvest young leaves for salads or stir-fries.",
                "image_filename": "pechay.jpg"
            },
            {
                "id": "basil",
                "name": "Basil",
                "scientific_name": "Ocimum basilicum",
                "optimal_ph": "6.0 - 7.5",
                "season": "Warm with good sunlight",
                "tip": "A fragrant herb used in cooking and pesto. Thrives in well-draining soil with good sunlight. Plant basil directly in vermicast-enriched soil.",
                "image_filename": "basil.jpg"
            },
            {
                "id": "papaya",
                "name": "Papaya",
                "scientific_name": "Carica papaya",
                "optimal_ph": "6.0 - 6.5",
                "season": "Tropical climates",
                "tip": "Papaya trees love nutrient-rich soil. Vermicast provides essential nutrients for healthy fruit production. Ideal for tropical climates.",
                "image_filename": "papaya.jpg"
            }
        ]
    }
}

# Response Schemas
class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    version: str

class CropRecommendation(BaseModel):
    id: str
    name: str
    scientific_name: str
    optimal_ph: str
    season: str
    tip: str
    image_url: str

class SoilProfile(BaseModel):
    composition: Dict[str, str]
    description: str
    management_tip: str

class PredictionInfo(BaseModel):
    class_id: int
    class_name: str
    display_name: str
    confidence: float

class PredictResponse(BaseModel):
    status: str
    is_soil: bool
    prediction: PredictionInfo
    soil_profile: Optional[SoilProfile] = None
    recommended_crops: Optional[List[CropRecommendation]] = None
    message: Optional[str] = None


@app.get("/", tags=["Root"])
def root():
    return {
        "message": "AGRIVO AI Soil Classification & Crop Recommendation API",
        "docs_url": "/docs",
        "health_url": "/health"
    }

@app.get("/health", response_model=HealthResponse, tags=["Health"])
def health_check():
    return HealthResponse(
        status="healthy",
        model_loaded=True,
        version="1.0.0"
    )

@app.post("/predict", response_model=PredictResponse, tags=["Prediction"])
async def predict_soil(request: Request, file: UploadFile = File(...)):
    # Validate MIME type
    if file.content_type and not file.content_type.startswith("image/"):
        raise HTTPException(
            status_code=400,
            detail="Uploaded file must be a valid image format (JPEG, PNG, WEBP)."
        )

    try:
        contents = await file.read()
        image = Image.open(io.BytesIO(contents))
        image = image.convert("RGB")
        image = image.resize((224, 224))
    except Exception as e:
        raise HTTPException(
            status_code=400,
            detail=f"Unable to decode image file: {str(e)}"
        )

    # Convert to array matching exact Kivy app preprocessing
    img_array = np.expand_dims(np.array(image, dtype=np.float32), axis=0)

    # Run inference
    try:
        interpreter.set_tensor(input_details[0]["index"], img_array)
        interpreter.invoke()
        output_data = interpreter.get_tensor(output_details[0]["index"])[0]
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error executing model inference: {str(e)}"
        )

    class_id = int(np.argmax(output_data))
    class_name = CLASS_NAMES[class_id]
    confidence = float(output_data[class_id])

    # Base URL for static asset serving
    base_url = str(request.base_url).rstrip("/")

    # Check for out-of-distribution / Nonetype
    if class_name == "Nonetype":
        return PredictResponse(
            status="warning",
            is_soil=False,
            prediction=PredictionInfo(
                class_id=class_id,
                class_name="Nonetype",
                display_name="Unidentified Object",
                confidence=round(confidence, 4)
            ),
            message="The captured image does not appear to be soil. Please center the camera over a clean soil sample and try again."
        )

    # Lookup agronomic metadata
    profile_data = AGRONOMIC_DATABASE.get(class_name)
    if not profile_data:
        raise HTTPException(
            status_code=500,
            detail=f"No agronomic data registered for class {class_name}"
        )

    crops_list = []
    for crop in profile_data["crops"]:
        crops_list.append(
            CropRecommendation(
                id=crop["id"],
                name=crop["name"],
                scientific_name=crop["scientific_name"],
                optimal_ph=crop["optimal_ph"],
                season=crop["season"],
                tip=crop["tip"],
                image_url=f"{base_url}/static/{crop['image_filename']}"
            )
        )

    soil_profile = SoilProfile(
        composition=profile_data["composition"],
        description=profile_data["description"],
        management_tip=profile_data["management_tip"]
    )

    return PredictResponse(
        status="success",
        is_soil=True,
        prediction=PredictionInfo(
            class_id=class_id,
            class_name=class_name,
            display_name=profile_data["display_name"],
            confidence=round(confidence, 4)
        ),
        soil_profile=soil_profile,
        recommended_crops=crops_list
    )

if __name__ == "__main__":
    import uvicorn
    # Hugging Face Spaces Docker default port is 7860
    port = int(os.environ.get("PORT", 7860))
    uvicorn.run("main:app", host="0.0.0.0", port=port, reload=False)
