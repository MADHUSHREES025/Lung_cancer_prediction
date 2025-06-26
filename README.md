# Lung Cancer Predictor

A web application for predicting lung cancer risk and type using clinical data and CT scan images. Built with Streamlit, XGBoost, and a VGG16-based CNN.

## Features
- **Clinical Risk Prediction:** Uses XGBoost to estimate lung cancer risk from patient data.
- **CT Scan Classification:** Uses a deep learning model to classify uploaded CT scan images into cancer types or normal.
- **User-Friendly Interface:** Simple, tabbed UI for both clinical+image and image-only workflows.

## Cancer Types Detected
- Adenocarcinoma
- Large Cell Carcinoma
- Squamous Cell Carcinoma
- Normal

## Project Structure
```
.
├── main.py                # Streamlit app
├── app.py                 # Model training script
├── Train.py               # VGG16 training script
├── test.py                # Model test script
├── requirements.txt       # Python dependencies
├── vgg16_lung_cancer_cnn.keras  # Trained CNN model
├── xgboost_lung_model_survey.pkl # Trained XGBoost model
├── Data/                  # Data folders (train, test, valid)
├── survey lung cancer.csv # Clinical data
├── cancer patient data sets.csv  # Additional data
```

## Setup Instructions
1. **Clone the repository:**
   ```bash
   git clone <repo-url>
   cd <repo-folder>
   ```
2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   pip install streamlit scikit-learn xgboost tensorflow pillow
   ```
3. **Run the app:**
   ```bash
   streamlit run main.py
   ```
4. **Upload a CT scan or enter patient details to get predictions.**

## Usage
- Use the "Clinical Risk + CT Scan" tab for a two-step workflow (risk + image).
- Use the "CT Scan Only" tab to classify a CT scan directly.

## Credits
Developed by **Madhu Shree**.

## License
See [LICENSE](LICENSE) for details. 