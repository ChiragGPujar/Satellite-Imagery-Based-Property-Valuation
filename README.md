# Satellite-Imagery-Based-Property-Valuation
A Multimodal Machine Learning project combining tabular housing data + satellite images to predict property prices.

Objective:
  1) Build a multimodal regression model to predict house prices.
  2) Use satellite imagery + structured real-estate data to capture environmental & neighborhood        context.
  3) Perform EDA and geospatial understanding.
  4) Extract CNN-based visual embeddings.
  5) Fuse image + tabular features for prediction.
  6) Provide explainability using Grad-CAM to visualize what the model learns.

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

Step 2 : Preprocessing.
This notebook:
  1) Cleans missing values
  2) Normalizes features
  3) Aligns tabular rows with image availability
  4) Saves processed train + test embeddings

Step 3: Train Multimodal Model
This notebook :
  1) Trains baseline tabular-only RandomForest
  2) Trains CNN (ResNet50) to extract image embeddings
  3) Builds Fusion Model (Tabular + Image)
  4) Evaluates using RMSE,R²

Step 4: Generate Test Predictions and output is the submission file.
