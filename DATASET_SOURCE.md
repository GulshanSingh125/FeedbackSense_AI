# Dataset Source Details

- **Dataset name:** Product Reviews Sentiment (used here for category classification)
- **Publisher/repository:** AnkitAI
- **Platform:** Hugging Face Datasets
- **Official repository:** https://huggingface.co/datasets/AnkitAI/product-reviews-sentiment
- **CSV file expected by the code:** `categorized_text_reviews.csv`
- **Expected columns in the notebook output:** `review`, `category`
- **Categories observed in the notebook:** Product Feedback, Customer Service, Fraud and Scam, Operational Issues
- **Provenance:** The dataset card describes the examples as synthetic.
- **Download method:** `train_model.py` downloads the CSV from the repository when it is not already present.
- **Local saved filename:** `data/customer_feedback.csv`

## Verification required before final submission

Check the dataset card and current repository file list for the latest license, source filename, schema, and provenance. Record the date you downloaded it. Do not describe these examples as real customer reviews. The actual number of rows after cleaning is printed during training; the notebook run provided for this project showed 7,000 raw rows and 320 unique text-label pairs after exact deduplication.
