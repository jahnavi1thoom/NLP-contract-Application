import streamlit as st
import tensorflow as tf
import pickle
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os
import re

from tensorflow.keras.preprocessing.sequence import pad_sequences

# --------------------------------------------------
# PAGE CONFIG
# --------------------------------------------------
st.set_page_config(
    page_title="AI Contract Intelligence System",
    page_icon="⚖️",
    layout="wide"
)

# --------------------------------------------------
# CUSTOM CSS
# --------------------------------------------------
st.markdown("""
<style>
.main {
    padding-top: 1rem;
}
.title {
    text-align:center;
    font-size:40px;
    font-weight:bold;
    color:#2E86C1;
}
.subtitle{
    text-align:center;
    color:gray;
    font-size:18px;
}
.pred-box{
    padding:15px;
    border-radius:10px;
    background-color:#eaf2f8;
    font-size:20px;
    font-weight:bold;
}
</style>
""", unsafe_allow_html=True)

# --------------------------------------------------
# SYSTEM PATH RESOLVER
# --------------------------------------------------
def find_file_path(filename):
    search_root = os.path.dirname(os.path.abspath(__file__))
    direct_path = os.path.join(search_root, filename)
    if os.path.exists(direct_path):
        return direct_path
    
    for root, dirs, files in os.walk(search_root):
        if filename in files:
            return os.path.join(root, filename)
    return None

# --------------------------------------------------
# LOAD ARTIFACTS (Path Safe & Independent of label_encoder)
# --------------------------------------------------
@st.cache_resource
def load_artifacts():
    model_path = find_file_path("contract_model.keras")
    tokenizer_path = find_file_path("tokenizer.pkl")

    if not model_path:
        raise FileNotFoundError("contract_model.keras could not be located anywhere in this repository workspace.")
    if not tokenizer_path:
        raise FileNotFoundError("tokenizer.pkl could not be located anywhere in this repository workspace.")

    model = tf.keras.models.load_model(model_path, compile=False)
    
    with open(tokenizer_path, "rb") as f:
        tokenizer = pickle.load(f)

    return model, tokenizer

try:
    model, tokenizer = load_artifacts()
    st.sidebar.success("✓ Deep Learning Artifacts Synchronized")
except Exception as e:
    st.error(f"System Linkage Error: {e}")
    st.info("Ensure contract_model.keras and tokenizer.pkl exist inside your deployment folder tree.")
    st.stop()

MAX_LEN = 500

# --------------------------------------------------
# CLEANING
# --------------------------------------------------
def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^a-zA-Z0-9 ]', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    return text

# --------------------------------------------------
# POSITIONAL ENCODING
# --------------------------------------------------
def positional_encoding(position, d_model):
    pe = np.zeros((position, d_model))
    for pos in range(position):
        for i in range(0, d_model, 2):
            pe[pos, i] = np.sin(pos / (10000 ** (i / d_model)))
            if i + 1 < d_model:
                pe[pos, i + 1] = np.cos(pos / (10000 ** (i / d_model)))
    return pe

# --------------------------------------------------
# HEADER
# --------------------------------------------------
st.markdown('<div class="title">⚖️ AI Contract Intelligence System</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">NLP + Self Attention + Positional Encoding</div>', unsafe_allow_html=True)
st.divider()

# --------------------------------------------------
# SIDEBAR
# --------------------------------------------------
st.sidebar.header("Project Features")
st.sidebar.success("✓ Contract Upload")
st.sidebar.success("✓ Clause Prediction")
st.sidebar.success("✓ Confidence Analysis")
st.sidebar.success("✓ Keyword Highlighting")
st.sidebar.success("✓ Positional Encoding Heatmap")

# --------------------------------------------------
# FILE UPLOAD
# --------------------------------------------------
uploaded_file = st.file_uploader("Upload Contract (.txt)", type=["txt"])
contract_text = ""

if uploaded_file:
    contract_text = uploaded_file.read().decode("utf-8")
else:
    contract_text = st.text_area("Or Paste Contract Text", height=300)

# --------------------------------------------------
# PREDICT BUTTON
# --------------------------------------------------
if st.button("Analyze Contract"):
    if len(contract_text.strip()) == 0:
        st.warning("Please upload or paste a contract.")
    else:
        cleaned = clean_text(contract_text)
        seq = tokenizer.texts_to_sequences([cleaned])
        seq = pad_sequences(seq, maxlen=MAX_LEN, padding="post")

        prediction = model.predict(seq)
        predicted_class = int(np.argmax(prediction))
        confidence = float(np.max(prediction))

        # Dynamic mapping fallbacks to prevent label_encoder dependencies
        label = f"Clause Class Index {predicted_class}"

        st.markdown(
            f"""
            <div class="pred-box">
            Predicted Clause Category Identification:
            <br><span style='color:#2E86C1;'>{label}</span>
            <br><br>
            Analysis Confidence Stability Score:
            <br><span style='color:#2E86C1;'>{confidence:.2%}</span>
            </div>
            """,
            unsafe_allow_html=True
        )
        st.divider()

        # --------------------------
        # TOP predictions table
        # --------------------------
        st.subheader("Top Mathematical Sequence Predictions")
        probs = prediction[0]
        top_idx = np.argsort(probs)[::-1][:5]

        result_df = pd.DataFrame({
            "Mathematical Class Target": [f"Class {idx}" for idx in top_idx],
            "Probability Weight Matrix": probs[top_idx]
        })
        st.dataframe(result_df, use_container_width=True)

        # --------------------------
        # KEYWORD HIGHLIGHTING
        # --------------------------
        st.subheader("Important Legal Terms Identified")
        keywords = ["payment", "termination", "agreement", "confidential", "liability", "warranty", "insurance", "license", "renewal", "distributor"]
        
        highlighted = contract_text
        for word in keywords:
            highlighted = re.sub(r'\b(' + re.escape(word) + r')\b', r'<mark style="background-color: #f7dc6f; padding: 2px; border-radius: 3px;">\1</mark>', highlighted, flags=re.IGNORECASE)

        st.markdown(f"<div style='background-color:#fafafa; padding:15px; border-radius:5px;'>{highlighted}</div>", unsafe_allow_html=True)

        # --------------------------
        # POSITIONAL ENCODING
        # --------------------------
        st.subheader("Positional Encoding Dimension Layer Heatmap")
        pe = positional_encoding(50, 128)
        fig, ax = plt.subplots(figsize=(12, 4))
        sns.heatmap(pe, ax=ax, cmap="viridis")
        ax.set_title("Positional Matrix Spatial Vectors (First 50 Tokens)")
        st.pyplot(fig)

        # --------------------------
        # CONFIDENCE BAR CHART
        # --------------------------
        st.subheader("Inference Variance Metrics")
        fig2, ax2 = plt.subplots(figsize=(8, 4))
        ax2.bar([f"Class {idx}" for idx in top_idx], probs[top_idx], color="#2e86c1")
        plt.xticks(rotation=15, ha="right")
        ax2.set_ylabel("Confidence Probability Value")
        st.pyplot(fig2)

st.divider()
st.caption("Built via CUAD Dataset Training Structures | TensorFlow Framework Synchronization | Streamlit Engine Pipeline")