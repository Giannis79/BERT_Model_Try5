import pandas as pd
import os
import torch
from transformers import BertTokenizer, BertForSequenceClassification

# Function to remove BOM from the first line
def remove_bom(filename):
    with open(filename, 'rb') as f:
        content = f.read()
    if content.startswith(b'\xef\xbb\xbf'):
        content = content[3:]
    with open(filename, 'wb') as f:
        f.write(content)

# Load the pre-trained model and tokenizer
model_name = 'sentiment_model'
tokenizer = BertTokenizer.from_pretrained(model_name)
model = BertForSequenceClassification.from_pretrained(model_name, num_labels=1)

# Prediction Function
def predict_influence(model, tokenizer, text):
    if not isinstance(text, str):
        text = str(text)  # Ensure the text is a string
    inputs = tokenizer(text, return_tensors='pt', max_length=128, truncation=True, padding='max_length')
    outputs = model(**inputs)
    score = outputs.logits.item()
    return score

# Create the directory if it doesn't exist
output_dir = r'Kathimerini\reformatted'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# File paths
files_and_months = [
    (r'Kathimerini\Kathimerini_articles_2022_02.csv', 'February'),
    (r'Kathimerini\Kathimerini_articles_2022_03.csv', 'March'),
    (r'Kathimerini\Kathimerini_articles_2022_04.csv', 'April'),
    (r'Kathimerini\Kathimerini_articles_2022_05.csv', 'May'),
    (r'Kathimerini\Kathimerini_articles_2022_06.csv', 'June'),
    (r'Kathimerini\Kathimerini_articles_2022_07.csv', 'July'),
    (r'Kathimerini\Kathimerini_articles_2022_08.csv', 'August')
]

# Prepare results
results = []

# Process each file
for file_name, month in files_and_months:
    try:
        # Remove BOM if present
        remove_bom(file_name)

        # Load the CSV file with potential formatting issues
        articles_df = pd.read_csv(file_name, encoding='ISO-8859-1', delimiter=',', on_bad_lines='skip')

        # Print columns to ensure 'Content' is present
        print(f"Processing {file_name}, columns: {articles_df.columns.tolist()}")

        # Process each article
        for index, row in articles_df.iterrows():
            if 'Content' not in row:
                print(f"Error: 'Content' column is missing in {file_name}")
                continue
            content = row['Content']
            score = predict_influence(model, tokenizer, content)
            scaled_score = score * 10  # Scale score to 1-10
            sentiment = 'Pro-Russian' if score < 0.5 else 'Pro-Ukrainian'

            # Append results
            results.append({
                'Sentiment': sentiment,
                'Score': scaled_score,
                'Month': month
            })

    except Exception as e:
        print(f"Error processing {file_name}: {e}")

# Create a DataFrame with the results
results_df = pd.DataFrame(results)

# Save results to a new CSV file
results_df.to_csv('Kathimerini_Evaluation.csv', index=False, encoding='utf-8-sig')

print("Evaluation completed and results saved to 'Kathimerini_Evaluation.csv'")
