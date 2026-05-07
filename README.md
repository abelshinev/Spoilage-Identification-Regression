# Spoiler Alert - Detection & ML Pipeline

This directory contains the core machine learning and computer vision pipeline for the Spoiler Alert project.
It is responsible for processing sticker images, extracting color data using OpenCV, and using calibrated configurations to predict the spoilage level and remaining shelf life of tracked food items.

## Tech Stack
- **Image Processing:** OpenCV (`opencv-python`)
- **Data Manipulation & Analysis:** NumPy, Pandas
- **Scientific Computing:** SciPy
- **Machine Learning:** Scikit-learn, Joblib

## System Architecture (Abstract)
- **Image Preprocessing:** Receives raw images of food stickers, isolates the region of interest (ROI), and normalizes lighting/orientation.
- **Color Extraction Engine:** Identifies the active indicator zone on the sticker and extracts the dominant colors or hues.
- **Prediction Module:** Feeds the extracted color data into a calibrated prediction model. It utilizes curve-fitting algorithms and configuration files (`sticker_config.json`, `score_config.json`) to estimate spoilage progression and calculate remaining shelf life.
- **Integration Interface:** Exposes entry points to be directly consumed by the backend, converting raw image input into actionable spoilage metrics.
