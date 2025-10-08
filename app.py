import streamlit as st
import numpy as np
import cv2
import joblib
from tensorflow.keras.models import load_model


# =========================
# MEDICAL DISCLAIMER
# =========================
st.sidebar.warning("⚠️ **MEDICAL DISCLAIMER**")
st.sidebar.write("""
**This application is for EDUCATIONAL and DEMONSTRATION purposes only.**

🔬 **Important Notes:**
- Predictions are based on machine learning models
- **NOT intended for actual medical diagnosis**
- Always consult healthcare professionals for medical advice
- No real patient data is stored or processed
- Accuracy may vary based on input quality
""")

# Continue with your existing code...
st.title("🩺 Health Prediction App")
# ... rest of your existing code

# =========================
# Load Diabetes Model + Scaler
# =========================
diabetes_model = joblib.load("diabetes_model.pkl")
scaler = joblib.load("diabetes_scaler.pkl")

# =========================
# Load Skin Disease Model
# =========================
skin_model = load_model("skin_disease_model.h5")

# Full class labels
skin_labels = [
    "Atopic Dermatitis",
    "Contact Dermatitis",
    "Eczema",
    "Seborrheic Dermatitis",
    "Skin Disease (unspecified)",
    "Tinea Corporis"
]

def preprocess_skin_image(img):
    """Resize and normalize image for model input"""
    img = cv2.resize(img, (224, 224))
    img = img / 255.0
    return np.expand_dims(img, axis=0)

# =========================
# Streamlit App
# =========================
st.title("🩺 Health Prediction App")

tab1, tab2 = st.tabs(["🧪 Diabetes Risk Checker", "🌿 Skin Disease Classifier"])

# -------------------------
# Diabetes Risk Checker
# -------------------------
with tab1:
    st.header("Diabetes Risk Checker")

    preg = st.number_input("Pregnancies", 0, 20, 1)
    glucose = st.number_input("Glucose", 0, 200, 120)
    bp = st.number_input("Blood Pressure", 0, 122, 70)
    skin = st.number_input("Skin Thickness", 0, 99, 20)
    insulin = st.number_input("Insulin", 0, 846, 79)
    bmi = st.number_input("BMI", 0.0, 67.1, 32.0)
    dpf = st.number_input("Diabetes Pedigree Function", 0.0, 2.5, 0.5)
    age = st.number_input("Age", 10, 100, 33)

    if st.button("Check Risk"):
        input_data = np.array([[preg, glucose, bp, skin, insulin, bmi, dpf, age]])
        scaled = scaler.transform(input_data)
        result = diabetes_model.predict(scaled)[0]
        prob = diabetes_model.predict_proba(scaled)[0][1]

        st.success("✅ Not Diabetic" if result == 0 else "⚠️ Diabetic")
        st.write(f"Risk Probability: **{prob:.2f}**")

# -------------------------
# Skin Disease Classifier
# -------------------------
with tab2:
    st.header("Skin Disease Classifier")
    st.write("Upload a skin image or take a photo to predict the disease category.")

    # File uploader
    uploaded_file = st.file_uploader("Upload a skin image", type=["jpg", "png", "jpeg"])

    # Camera input
    camera_file = st.camera_input("Or take a picture with your camera")

    img = None
    if uploaded_file is not None:
        # Handle uploaded image
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        img = cv2.imdecode(file_bytes, 1)
        st.image(img, channels="BGR", caption="Uploaded Image", use_column_width=True)

    elif camera_file is not None:
        # Handle camera image
        file_bytes = np.asarray(bytearray(camera_file.read()), dtype=np.uint8)
        img = cv2.imdecode(file_bytes, 1)
        st.image(img, channels="BGR", caption="Camera Image", use_column_width=True)

    # Prediction only if image is available
    if img is not None:
        processed = preprocess_skin_image(img)
        prediction = skin_model.predict(processed)
        class_idx = np.argmax(prediction)
        prob = prediction[0][class_idx]

        st.success(f"**Prediction:** {skin_labels[class_idx]}")
        st.write(f"**Confidence:** {prob:.2f}")