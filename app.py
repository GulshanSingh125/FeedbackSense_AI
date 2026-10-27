from pathlib import Path
import sys
import pandas as pd
import streamlit as st
import joblib

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
from train_model import ensure_model, load_dataset, train_and_evaluate

st.set_page_config(page_title="FeedbackSense AI", page_icon="💬", layout="wide")

st.title("💬 FeedbackSense AI")
st.subheader("NLP-Based Customer Feedback Classification")
st.write(
    "Classify customer feedback into business-oriented categories using a "
    "TF-IDF + Logistic Regression model."
)

MODEL_PATH = ROOT / "models" / "feedback_classifier.joblib"
REPORT_PATH = ROOT / "reports" / "evaluation_results.txt"

with st.spinner("Checking dataset and trained model..."):
    if not MODEL_PATH.exists():
        st.info("No trained model found. Downloading the dataset and training the model for first use.")
        train_and_evaluate()

model = joblib.load(MODEL_PATH)
labels = list(model.named_steps["classifier"].classes_)

with st.sidebar:
    st.header("About the model")
    st.write("**Algorithm:** Logistic Regression")
    st.write("**Text features:** Word and character TF-IDF")
    st.write("**Categories:**")
    for label in labels:
        st.write(f"- {label}")
    st.caption(
        "Dataset provenance: the Hugging Face dataset card describes the examples as synthetic. "
        "Predictions are for demonstration and should be reviewed by a person."
    )

tab_single, tab_batch, tab_info = st.tabs(
    ["Single feedback", "Batch CSV classification", "Dataset & model notes"]
)

with tab_single:
    st.markdown("### Classify one customer message")
    message = st.text_area(
        "Enter customer feedback",
        placeholder="Example: The delivery was delayed and my package arrived damaged.",
        height=130,
    )
    if st.button("Classify feedback", type="primary"):
        if not message.strip():
            st.warning("Please enter a feedback message first.")
        else:
            prediction = model.predict([message.strip()])[0]
            scores = model.predict_proba([message.strip()])[0]
            score_df = pd.DataFrame({
                "Category": model.named_steps["classifier"].classes_,
                "Model score": scores
            }).sort_values("Model score", ascending=False)

            st.success(f"Predicted category: **{prediction}**")
            st.caption("Model scores are not necessarily calibrated probabilities.")
            st.bar_chart(score_df.set_index("Category"))
            st.dataframe(score_df.style.format({"Model score": "{:.1%}"}), use_container_width=True)

with tab_batch:
    st.markdown("### Classify many feedback messages")
    st.write("Upload a CSV containing a column named `feedback`, `review`, `text`, or `customer_feedback`.")
    uploaded = st.file_uploader("Upload CSV", type=["csv"])
    if uploaded is not None:
        try:
            batch_df = pd.read_csv(uploaded)
            text_col_lookup = {str(c).strip().lower(): c for c in batch_df.columns}
            candidates = ["feedback", "review", "text", "customer_feedback", "content"]
            text_col = next((text_col_lookup[c] for c in candidates if c in text_col_lookup), None)
            if text_col is None:
                st.error(f"Could not find a feedback text column. Available columns: {batch_df.columns.tolist()}")
            else:
                texts = batch_df[text_col].fillna("").astype(str)
                valid_mask = texts.str.strip().ne("")
                results = batch_df.loc[valid_mask].copy()
                results["predicted_category"] = model.predict(texts[valid_mask].tolist())
                st.success(f"Classified {len(results)} non-empty rows.")
                st.dataframe(results.head(100), use_container_width=True)
                st.bar_chart(results["predicted_category"].value_counts())
                csv_bytes = results.to_csv(index=False).encode("utf-8")
                st.download_button(
                    "Download classified CSV",
                    data=csv_bytes,
                    file_name="feedback_classification_results.csv",
                    mime="text/csv"
                )
        except Exception as exc:
            st.error(f"Could not process the uploaded CSV: {exc}")

with tab_info:
    st.markdown("### Dataset source")
    st.markdown(
        "[AnkitAI/product-reviews-sentiment on Hugging Face]"
        "(https://huggingface.co/datasets/AnkitAI/product-reviews-sentiment)"
    )
    st.write(
        "The training script downloads `categorized_text_reviews.csv` from the repository. "
        "It inspects the source columns and removes missing rows and exact duplicate text-label pairs."
    )
    st.warning(
        "In your executed notebook, 7,000 raw rows became 320 unique text-label pairs after exact deduplication. "
        "This suggests repeated/template-like examples. The reported 98.4% test accuracy may therefore be optimistic "
        "and must not be presented as expected real-world accuracy."
    )
    st.markdown("### Current evaluation report")
    if REPORT_PATH.exists():
        st.code(REPORT_PATH.read_text(encoding="utf-8"))
    else:
        st.info("The evaluation report will appear after training.")

st.divider()
st.caption("Prototype for educational use. Validate with representative, independently collected customer feedback before business deployment.")
