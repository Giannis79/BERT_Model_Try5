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
def predict_influence(model, tokenizer, text, min_logit, max_logit):
    if not isinstance(text, str):
        text = str(text)  # Ensure the text is a string
    inputs = tokenizer(text, return_tensors='pt', max_length=128, truncation=True, padding='max_length')
    outputs = model(**inputs)
    logits = outputs.logits.item()
    # Normalize logits to (0, 1) range using min-max scaling
    normalized_score = (logits - min_logit) / (max_logit - min_logit)
    scaled_score = normalized_score * 9 + 1  # Scale to (1, 10) range
    return scaled_score

# Create the directory if it doesn't exist
output_dir = r'Independent\reformatted'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Corrected file paths
files_and_months = [
    (r'Independent\reformatted_Independent_articles_2022_02.csv', 'February'),
    (r'Independent\reformatted_Independent_articles_2022_03.csv', 'March'),
    (r'Independent\reformatted_Independent_articles_2022_04.csv', 'April'),
    (r'Independent\reformatted_Independent_articles_2022_05.csv', 'May'),
    (r'Independent\reformatted_Independent_articles_2022_06.csv', 'June'),
    (r'Independent\reformatted_Independent_articles_2022_07.csv', 'July'),
    (r'Independent\reformatted_Independent_articles_2022_08.csv', 'August')
]

# Collect logits to determine min and max
logits_list = []

for file_name, month in files_and_months:
    try:
        remove_bom(file_name)
        articles_df = pd.read_csv(file_name, encoding='ISO-8859-1', delimiter=',', on_bad_lines='skip')
        print(f"Processing {file_name}, columns: {articles_df.columns.tolist()}")
        for index, row in articles_df.iterrows():
            if 'Content' not in row:
                print(f"Error: 'Content' column is missing in {file_name}")
                continue
            content = row['Content']
            if not isinstance(content, str):
                content = str(content)
            inputs = tokenizer(content, return_tensors='pt', max_length=128, truncation=True, padding='max_length')
            outputs = model(**inputs)
            logits = outputs.logits.item()
            logits_list.append(logits)
    except Exception as e:
        print(f"Error processing {file_name}: {e}")

# Ensure we collected some logits
if not logits_list:
    print("Error: No logits collected. Please check the input CSV files.")
else:
    # Determine min and max logits
    min_logit = min(logits_list)
    max_logit = max(logits_list)
    print(f"Min Logit: {min_logit}, Max Logit: {max_logit}")

    # Prepare results
    results = []

    # Process each file again with min-max normalization
    for file_name, month in files_and_months:
        try:
            remove_bom(file_name)
            articles_df = pd.read_csv(file_name, encoding='ISO-8859-1', delimiter=',', on_bad_lines='skip')
            for index, row in articles_df.iterrows():
                if 'Content' not in row:
                    continue
                content = row['Content']
                scaled_score = predict_influence(model, tokenizer, content, min_logit, max_logit)
                sentiment = 'Pro-Russian' if scaled_score < 5 else 'Pro-Ukrainian'
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
    results_df.to_csv('Independent_Evaluation.csv', index=False, encoding='utf-8-sig')

    print("Evaluation completed and results saved to 'Independent_Evaluation.csv'")
