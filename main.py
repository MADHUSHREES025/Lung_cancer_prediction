import streamlit as st
import pandas as pd
import numpy as np
import joblib
import io
import os
import gdown
from PIL import Image
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array

# --- Google Drive model download config ---
MODEL_PATH = "vgg16_lung_cancer_cnn.keras"
GDRIVE_FILE_ID = "1mn8-vVUTlPo44xIPCi-jtsksJIVx1Oil"  # 🔁 Replace with your real file ID
GDRIVE_URL = f"https://drive.google.com/file/d/1mn8-vVUTlPo44xIPCi-jtsksJIVx1Oil/view?usp=drive_link"

# --- Download CNN model if missing ---
def download_model_if_missing():
    if not os.path.exists(MODEL_PATH):
        with st.spinner("📥 Downloading CNN model from Google Drive..."):
            gdown.download(GDRIVE_URL, MODEL_PATH, quiet=False)
            st.success("✅ CNN model downloaded successfully.")

# --- Load models ---
@st.cache_resource
def load_models():
    xgb = joblib.load("xgboost_lung_model_survey.pkl")
    download_model_if_missing()
    cnn = load_model(MODEL_PATH)
    return xgb, cnn

xgb_model, cnn_model = load_models()

# --- Class labels ---
class_labels = ['adenocarcinoma', 'large.cell.carcinoma', 'normal', 'squamous.cell.carcinoma']

# --- Gender encoder ---
gender_encoder = LabelEncoder()
gender_encoder.classes_ = np.array(['F', 'M'])

# --- Streamlit layout ---
st.set_page_config(page_title="Lung Cancer Predictor", layout="centered")
st.title("🫁 Lung Cancer Risk & Stage Detection")

# --- Tabs ---
tab1, tab2 = st.tabs(["🧬 Clinical Risk + CT Scan", "🖼️ CT Scan Only"])

# ----------------------- Tab 1: Clinical Risk + CT Scan -----------------------
with tab1:
    st.subheader("🧬 Step 1: Enter Clinical Details")

    with st.form("form"):
        gender = st.selectbox("Gender", ["M", "F"])
        age = st.slider("Age", 20, 100, 50)
        smoking = st.selectbox("Smoking (1=No, 2=Yes)", [1, 2])
        yellow_fingers = st.selectbox("Yellow Fingers", [1, 2])
        anxiety = st.selectbox("Anxiety", [1, 2])
        peer_pressure = st.selectbox("Peer Pressure", [1, 2])
        chronic_disease = st.selectbox("Chronic Disease", [1, 2])
        fatigue = st.selectbox("Fatigue", [1, 2])
        allergy = st.selectbox("Allergy", [1, 2])
        wheezing = st.selectbox("Wheezing", [1, 2])
        alcohol = st.selectbox("Alcohol Consuming", [1, 2])
        coughing = st.selectbox("Coughing", [1, 2])
        breath = st.selectbox("Shortness of Breath", [1, 2])
        swallowing = st.selectbox("Swallowing Difficulty", [1, 2])
        chest_pain = st.selectbox("Chest Pain", [1, 2])
        submit = st.form_submit_button("Predict Risk")

    if submit:
        input_data = {
            'GENDER': gender_encoder.transform([gender])[0],
            'AGE': age,
            'SMOKING': smoking,
            'YELLOW_FINGERS': yellow_fingers,
            'ANXIETY': anxiety,
            'PEER_PRESSURE': peer_pressure,
            'CHRONIC DISEASE': chronic_disease,
            'FATIGUE ': fatigue,
            'ALLERGY ': allergy,
            'WHEEZING': wheezing,
            'ALCOHOL CONSUMING': alcohol,
            'COUGHING': coughing,
            'SHORTNESS OF BREATH': breath,
            'SWALLOWING DIFFICULTY': swallowing,
            'CHEST PAIN': chest_pain
        }

        df_input = pd.DataFrame([input_data])
        xgb_pred = xgb_model.predict(df_input)[0]
        xgb_proba = xgb_model.predict_proba(df_input)[0][1]

        st.subheader("🩺 Risk Prediction (XGBoost)")
        if xgb_pred == 0:
            st.success(f"✅ Low risk of lung cancer (Probability: {xgb_proba:.2f})")
        else:
            st.error(f"⚠️ High risk detected (Probability: {xgb_proba:.2f})")
            st.markdown("### Step 2: Upload CT Scan for Type Detection")

            if st.button("Continue to CT Scan Classifier ➡️"):
                uploaded = st.file_uploader("Upload CT Scan Image", type=["jpg", "jpeg", "png"], key="ct_inline")
                if uploaded:
                    try:
                        image_stream = io.BytesIO(uploaded.getvalue())
                        st.image(image_stream, caption="CT Scan Image", use_column_width=True)
                        image_stream.seek(0)
                        img = Image.open(image_stream).convert("RGB").resize((224, 224))
                        img_array = img_to_array(img) / 255.0
                        img_array = np.expand_dims(img_array, axis=0)

                        prediction = cnn_model.predict(img_array)
                        predicted_class = int(np.argmax(prediction[0]))
                        confidence = float(prediction[0][predicted_class])

                        st.subheader("🧠 CNN Prediction")
                        st.write(f"**Cancer Type:** `{class_labels[predicted_class]}`")
                        st.write(f"**Confidence:** `{confidence:.2f}`")

                        if class_labels[predicted_class] == "normal":
                            st.success("🟢 CT scan is normal.")
                        else:
                            st.error(f"🔴 Type detected: `{class_labels[predicted_class]}`")

                    except Exception as e:
                        st.warning(f"⚠️ Error processing image: {e}")

# ----------------------- Tab 2: CT Scan Only -----------------------
with tab2:
    st.subheader("🖼️ Upload CT Scan for Direct Cancer Type Detection")
    uploaded_image = st.file_uploader("Upload CT Scan", type=["jpg", "jpeg", "png"], key="ct_only")

    if uploaded_image:
        try:
            image_stream = io.BytesIO(uploaded_image.getvalue())
            st.image(image_stream, caption="CT Scan Image", use_column_width=True)
            image_stream.seek(0)
            img = Image.open(image_stream).convert("RGB").resize((224, 224))
            img_array = img_to_array(img) / 255.0
            img_array = np.expand_dims(img_array, axis=0)

            prediction = cnn_model.predict(img_array)
            predicted_class = int(np.argmax(prediction[0]))
            confidence = float(prediction[0][predicted_class])

            st.subheader("🧠 CNN Prediction")
            st.write(f"**Cancer Type:** `{class_labels[predicted_class]}`")
            st.write(f"**Confidence:** `{confidence:.2f}`")

            if class_labels[predicted_class] == "normal":
                st.success("🟢 CT scan is normal.")
            else:
                st.error(f"🔴 Type detected: `{class_labels[predicted_class]}`")

        except Exception as e:
            st.warning(f"⚠️ Error processing image: {e}")
