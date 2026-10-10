---
title: Agrivo Soil Classification API
emoji: 🌱
colorFrom: green
colorTo: lime
sdk: docker
app_port: 7860
pinned: false
---

# AGRIVO Soil Classification & Recommendation Backend

Public inference API for the AGRIVO React Native mobile application.
- **Model**: Hybrid ResNet50–MobileNetV3 (TensorFlow Lite)
- **Framework**: FastAPI + Uvicorn
- **Port**: 7860

## API Endpoints
- `GET /`: Health check & API details
- `GET /health`: JSON status check
- `POST /predict`: Upload multipart image (`file`), returns soil classification, confidence score, chemical composition, and crop recommendations.
- `GET /static/{filename}`: Static crop icon images.
