import streamlit as st
import tensorflow as tf
import pickle
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
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
# LOAD MODEL
# --------------------------------------------------

@st.cache_resource
def load_artifacts():

    model = tf.keras.models.load_model(
        "contract_model.keras"
    )

    with open("tokenizer.pkl", "rb") as f:
        tokenizer = pickle.load(f)

    with open("label_encoder.pkl", "rb") as f:
        label_encoder = pickle.load(f)

    return model, tokenizer, label_encoder


model, tokenizer, label_encoder = load_artifacts()

MAX_LEN = 500

# --------------------------------------------------
# CLEANING
# --------------------------------------------------

def clean_text(text):

    text = text.lower()

    text = re.sub(
        r'[^a-zA-Z0-9 ]',
        ' ',
        text
    )

    text = re.sub(
        r'\s+',
        ' ',
        text
    )

    return text


# --------------------------------------------------
# POSITIONAL ENCODING
# --------------------------------------------------

def positional_encoding(position, d_model):

    pe = np.zeros((position, d_model))

    for pos in range(position):

        for i in range(0, d_model, 2):

            pe[pos, i] = np.sin(
                pos / (10000 ** (i / d_model))
            )

            if i + 1 < d_model:

                pe[pos, i + 1] = np.cos(
                    pos / (10000 ** (i / d_model))
                )

    return pe


# --------------------------------------------------
# HEADER
# --------------------------------------------------

st.markdown(
    '<div class="title">⚖️ AI Contract Intelligence System</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">NLP + Self Attention + Positional Encoding</div>',
    unsafe_allow_html=True
)

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

uploaded_file = st.file_uploader(
    "Upload Contract (.txt)",
    type=["txt"]
)

contract_text = ""

if uploaded_file:

    contract_text = uploaded_file.read().decode("utf-8")

else:

    contract_text = st.text_area(
        "Or Paste Contract Text",
        height=300
    )

# --------------------------------------------------
# PREDICT BUTTON
# --------------------------------------------------

if st.button("Analyze Contract"):

    if len(contract_text.strip()) == 0:

        st.warning("Please upload or paste a contract.")

    else:

        cleaned = clean_text(contract_text)

        seq = tokenizer.texts_to_sequences(
            [cleaned]
        )

        seq = pad_sequences(
            seq,
            maxlen=MAX_LEN,
            padding="post"
        )

        prediction = model.predict(seq)

        predicted_class = np.argmax(prediction)

        label = label_encoder.inverse_transform(
            [predicted_class]
        )[0]

        confidence = float(
            np.max(prediction)
        )

        st.markdown(
            f"""
            <div class="pred-box">
            Predicted Clause Category:
            <br>
            {label}
            <br><br>
            Confidence:
            {confidence:.2%}
            </div>
            """,
            unsafe_allow_html=True
        )

        st.divider()

        # --------------------------
        # TOP 5 PREDICTIONS
        # --------------------------

        st.subheader("Top Predictions")

        probs = prediction[0]

        top_idx = np.argsort(probs)[::-1][:5]

        result_df = pd.DataFrame({
            "Category":
            label_encoder.inverse_transform(top_idx),

            "Probability":
            probs[top_idx]
        })

        st.dataframe(
            result_df,
            use_container_width=True
        )

        # --------------------------
        # KEYWORD HIGHLIGHTING
        # --------------------------

        st.subheader("Important Legal Terms")

        keywords = [

            "payment",
            "termination",
            "agreement",
            "confidential",
            "liability",
            "warranty",
            "insurance",
            "license",
            "renewal",
            "distributor"
        ]

        highlighted = contract_text

        for word in keywords:

            highlighted = re.sub(
                word,
                f"<mark>{word}</mark>",
                highlighted,
                flags=re.IGNORECASE
            )

        st.markdown(
            highlighted,
            unsafe_allow_html=True
        )

        # --------------------------
        # POSITIONAL ENCODING
        # --------------------------

        st.subheader(
            "Positional Encoding Heatmap"
        )

        pe = positional_encoding(
            50,
            128
        )

        fig, ax = plt.subplots(
            figsize=(12,6)
        )

        sns.heatmap(
            pe,
            ax=ax
        )

        ax.set_title(
            "Positional Encoding"
        )

        st.pyplot(fig)

        # --------------------------
        # CONFIDENCE BAR CHART
        # --------------------------

        st.subheader(
            "Prediction Confidence"
        )

        fig2, ax2 = plt.subplots()

        ax2.bar(
            result_df["Category"],
            result_df["Probability"]
        )

        plt.xticks(
            rotation=45,
            ha="right"
        )

        st.pyplot(fig2)

st.divider()

st.caption(
    "Built using CUAD Dataset | TensorFlow | Streamlit"
)