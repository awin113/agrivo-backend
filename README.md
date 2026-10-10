---
title: Agrivo Soil Classifier
emoji: 🌱
colorFrom: green
colorTo: lime
sdk: gradio
sdk_version: 4.44.0
app_file: app.py
pinned: false
---

# AGRIVO Soil Classification & Crop Recommendation

Deep Learning AI for mobile image-based soil categorization and agronomic recommendations using hybrid ResNet50-MobileNetV3.

## Endpoints:
- **Web UI**: `/` (Gradio interactive tester)
- **REST API Predict**: `POST /predict` (multipart image upload)
- **REST API Health**: `GET /health`
- **Static Assets**: `GET /static/{filename}`
