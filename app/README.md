---
title: Ovarian Cancer Dual-Stream AI
emoji: 🔬
colorFrom: pink
colorTo: indigo
sdk: gradio
app_file: app.py
pinned: false
license: mit
---

# Ovarian Cancer Dual-Stream AI: demo

This demo has two parts:
- **Clinical:** CatBoost + SHAP on 16 blood biomarkers.
- **Histopathology:** ConvNeXt + CBAM + Grad-CAM on H&E patches.

**To deploy on Hugging Face Spaces:**
1. Create a new Space and choose the Gradio SDK.
2. Upload `app.py`, `models.py`, `requirements.txt` and this README.
3. Upload `data/Supplementary data 1.xlsx` and `weights/stream_a_convnext_single_cbam.pth`.

Research demo only. It is not a medical device.
