import streamlit as st
import numpy as np
from PIL import Image
import tensorflow as tf

st.set_page_config(
    page_title="AI Synthetic Face Detector",
    page_icon="🕵️",
    layout="wide"
)

MODEL_PATH = "final_model.keras"
IMG_SIZE = (224, 224)
CLASS_NAMES = ["Fake", "Real"]  # index 0 = fake, index 1 = real (alphabetical, matches classification_report.txt)


@st.cache_resource
def load_model():
    return tf.keras.models.load_model(MODEL_PATH)


def preprocess(img: Image.Image):
    img = img.convert("RGB").resize(IMG_SIZE)
    arr = np.array(img).astype("float32")
    arr = tf.keras.applications.mobilenet_v2.preprocess_input(arr)
    return np.expand_dims(arr, axis=0)


# ---------- Sidebar ----------
st.sidebar.title("🕵️ Face Detector")
st.sidebar.markdown(
    """
This app uses a **MobileNetV2**-based transfer-learning model
trained to distinguish **real** human faces from **AI-generated
(synthetic) faces**.

**Model stats**
- Accuracy: **81.42%**
- AUC-ROC: **0.891**
"""
)
page = st.sidebar.radio("Go to", ["🔍 Try the Model", "📊 Model Performance"])

# ---------- Page 1: Live prediction ----------
if page == "🔍 Try the Model":
    st.title("AI Synthetic Face Detector")
    st.write("Upload a face image and the model will predict whether it's **real** or **AI-generated (fake)**.")

    uploaded = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

    col1, col2 = st.columns([1, 1])

    if uploaded is not None:
        image = Image.open(uploaded)
        with col1:
            st.image(image, caption="Uploaded image", use_container_width=True)

        with st.spinner("Analyzing..."):
            model = load_model()
            x = preprocess(image)
            preds = model.predict(x, verbose=0)[0]
            pred_idx = int(np.argmax(preds))
            confidence = float(preds[pred_idx]) * 100

        with col2:
            label = CLASS_NAMES[pred_idx]
            if label == "Real":
                st.success(f"### ✅ Prediction: {label}")
            else:
                st.error(f"### 🤖 Prediction: {label}")
            st.metric("Confidence", f"{confidence:.2f}%")

            st.write("**Class probabilities**")
            st.progress(float(preds[1]), text=f"Real: {preds[1]*100:.2f}%")
            st.progress(float(preds[0]), text=f"Fake: {preds[0]*100:.2f}%")
    else:
        st.info("👆 Upload a JPG or PNG face image to get started.")

# ---------- Page 2: Results dashboard ----------
else:
    st.title("📊 Model Performance")
    st.caption("Evaluation on the held-out test set")

    with open("results/classification_report.txt") as f:
        report = f.read()

    m1, m2 = st.columns(2)
    m1.metric("Accuracy", "81.42%")
    m2.metric("AUC-ROC", "0.8914")

    st.text(report)

    st.subheader("Confusion Matrix")
    st.image("results/confusion_matrix.png", use_container_width=False)

    st.subheader("ROC Curve")
    st.image("results/roc_curve.png", use_container_width=False)

    st.subheader("Training History")
    st.image("results/training_history.png", use_container_width=True)

    st.subheader("Grad-CAM — what the model is looking at")
    st.image("results/gradcam_samples.png", use_container_width=True)

    st.subheader("Sample Predictions")
    st.image("results/sample_predictions.png", use_container_width=True)
