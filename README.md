# Satellite-Imagery-Based-Property-Valuation
A Multimodal Machine Learning project combining tabular housing data + satellite images to predict property prices.

The project demonstrates data engineering, multimodal fusion, model benchmarking, and visual explainability (Grad-CAM).

Objective:
  1) Build a multimodal regression model to predict house prices.
  2) Use satellite imagery + structured real-estate data to capture environmental & neighborhood context.
  3) Perform EDA and feature analysis on structured housing attributes.
  4) Extract CNN-based visual embeddings.
  5) Fuse image + tabular features for prediction.
  6) Compare Tabular-only vs Multimodal (Fusion) performance.
  7) Provide explainability using Grad-CAM.

Dataset:
  1) train.csv – Tabular training data containing housing attributes and price.
  2) test.csv – Test dataset without price.
  3) /images/train/ – Satellite images corresponding to training samples.
  4) /images/test/ – Satellite images corresponding to test samples.
  5) Each property has:
  6) Numerical & categorical attributes
  7) Latitude / longitude derived neighborhood context
  8) Satellite image showing surroundings

Setup Instructions:
  1) Install Dependencies.
     ``` python
      pip install pandas numpy scikit-learn matplotlib seaborn torch torchvision tqdm

Running the Project:

Step 1 : Fetch Satellite Images

Run: data_fetcher.ipynb

Step 2 : Preprocessing (preprocessing.ipynb)

This notebook:
  1) Cleans missing values
  2) Normalizes features
  3) Aligns tabular rows with image availability
  4) Saves processed train + test embeddings

Make sure to Include/copy the test and train dataset before running this notebook to avoid error.

Step 3: Train Multimodal Model (model_training.ipynb)

This notebook:
  1) Trains Tabular Baseline (XGBoost)
  2) Image Embedding Extraction
  3) PCA on Image Embeddings
  4) Fusion Model (Tabular + Image)
  5) Evaluates using RMSE,R²

Step 4: Grad-CAM Explainability

This notebook:
  1) Applies Grad-CAM on ResNet50
  2) Visualizes which image regions influence predictions
  3) Highlights:
     a) Roads
     b) Vegetation
     c) Building Density
     d) Neighborhood Layout
     

Step 5: Generate Test Predictions and output is the submission file.
