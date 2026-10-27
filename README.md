# FeedbackSense AI — Customer Feedback Classification System

An NLP project that classifies customer review text into business-oriented categories using TF-IDF features and Logistic Regression. It includes a Streamlit app for individual and batch classification.

## Dataset

- **Repository:** [AnkitAI/product-reviews-sentiment](https://huggingface.co/datasets/AnkitAI/product-reviews-sentiment)
- **Expected CSV:** `categorized_text_reviews.csv`
- **Expected source columns:** `review` and `category`; the training script checks column names and standardizes them internally.
- **Categories shown in the executed notebook:** Product Feedback, Customer Service, Fraud and Scam, Operational Issues.
- **Important provenance note:** The dataset card describes the examples as synthetic. This is an educational prototype, not a validated model for real customer data.

## Features

- Download the dataset from Hugging Face.
- Inspect, clean, and deduplicate text-label pairs.
- Train word-level and character-level TF-IDF + Logistic Regression.
- Evaluate using accuracy, precision, recall, F1-score, and a confusion matrix.
- Streamlit single-message classification.
- Batch CSV upload, predictions, category chart, and downloadable results.
- Save the model pipeline and evaluation report.

## Project structure

```text
FeedbackSense_AI/
├── app.py
├── train_model.py
├── requirements.txt
├── README.md
├── DATASET_SOURCE.md
├── .gitignore
├── data/
│   └── (downloaded customer_feedback.csv)
├── models/
│   └── (generated feedback_classifier.joblib)
└── reports/
    └── (generated evaluation_results.txt)
```

## Setup on Windows

Open Command Prompt or the VS Code terminal in this folder:

```bash
python -m venv .venv
.venv\Scripts\activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

## Train the model

```bash
python train_model.py
```

This downloads the dataset if it is not already present and generates:

- `data/customer_feedback.csv`
- `models/feedback_classifier.joblib`
- `reports/evaluation_results.txt`

## Run the Streamlit application

```bash
streamlit run app.py
```

On first launch, if no model file exists, the app trains it automatically. It then opens a local URL in your browser (usually `http://localhost:8501`).

## How to use

1. **Single feedback:** paste a message and click **Classify feedback**.
2. **Batch CSV classification:** upload a CSV containing a text column named `feedback`, `review`, `text`, `customer_feedback`, or `content`.
3. Inspect predicted categories and download the results as CSV.

## Evaluation and an important warning

The notebook you ran reported:
- Raw dataset: 7,000 rows and 2 columns (`review`, `category`)
- After dropping missing/empty rows and exact duplicate text-label pairs: 320 rows
- Held-out test accuracy: 98.4% on 64 test examples

These values are from your notebook output, not a guarantee that every run or repository version will produce the same result. The large reduction from 7,000 raw rows to 320 unique pairs suggests repeated/template-like content. The 98.4% accuracy may be optimistic and should not be represented as expected performance on real customer feedback. Inspect near-duplicates and validate on independently collected, representative data before any real deployment.

## Responsible use and limitations

- A category prediction is a suggestion for triage, not a definitive decision.
- A `Fraud and Scam` prediction is not proof of fraud.
- Model scores from `predict_proba` are not necessarily calibrated confidence probabilities.
- Avoid using personal data unless you have a lawful and appropriate basis to process it.
- The dataset is synthetic, so real-world language, misspellings, context, and class balance may differ.

## AI tools disclosure

If you used ChatGPT or another AI coding assistant to help plan, generate, debug, or document this project, disclose that assistance in the submission according to the internship instructions. Review and understand all code before submitting it.

## Assignment coverage

This project report addresses:
1. Problem statement
2. Dataset/source of data
3. Approach and methodology
4. Tools/libraries used
5. Model or AI approach
6. Results/evaluation
7. Key observations
8. Possible real-world application
