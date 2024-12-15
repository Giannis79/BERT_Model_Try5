import pandas as pd
import torch
from transformers import BertTokenizer, BertForSequenceClassification

# Load the pre-trained model and tokenizer
model_name = 'sentiment_model'
tokenizer = BertTokenizer.from_pretrained(model_name)
model = BertForSequenceClassification.from_pretrained(model_name, num_labels=1)

# Prediction Function
def predict_influence(model, tokenizer, text):
    inputs = tokenizer(text, return_tensors='pt', max_length=128, truncation=True, padding='max_length')
    outputs = model(**inputs)
    score = outputs.logits.item()
    return score

# Load Data with different encoding
df = pd.read_csv('Model_evaluator.csv', encoding='ISO-8859-1')  # Try using 'ISO-8859-1' encoding

# Check if the dataframe is not empty
if not df.empty:
    total_texts = len(df)
    pro_russian_count = 0
    pro_ukrainian_count = 0
    pro_russian_scores = []
    pro_ukrainian_scores = []

    # Predict sentiment for each text and classify
    for index, row in df.iterrows():
        text = row['text']
        score = predict_influence(model, tokenizer, text)
        scaled_score = score * 10  # Scale score to 1-10
        # Classify based on score
        if score < 0.5:  # Assuming a threshold for Pro-Russian
            sentiment = 'Pro-Russian'
            pro_russian_count += 1
            pro_russian_scores.append(scaled_score)
        else:  # Pro-Ukrainian
            sentiment = 'Pro-Ukrainian'
            pro_ukrainian_count += 1
            pro_ukrainian_scores.append(scaled_score)
        # Print each text with its predicted sentiment and score
        print('Text {index + 1}: {sentiment} (Score: {scaled_score:.2f})')

    # Calculate average scores
    avg_pro_russian_score = sum(pro_russian_scores) / len(pro_russian_scores) if pro_russian_scores else 0
    avg_pro_ukrainian_score = sum(pro_ukrainian_scores) / len(pro_ukrainian_scores) if pro_ukrainian_scores else 0

    # Print summary results
    print(f'\nSummary:')
    print(f'Total texts processed: {total_texts}')
    print(f'Pro-Russian texts: {pro_russian_count} (Average Score: {avg_pro_russian_score:.2f})')
    print(f'Pro-Ukrainian texts: {pro_ukrainian_count} (Average Score: {avg_pro_ukrainian_score:.2f})')
else:
    print("The dataframe is empty. Please check the input CSV file.")
