import pandas as pd
from sklearn.model_selection import train_test_split
from transformers import BertTokenizer, BertForSequenceClassification, AdamW, get_linear_schedule_with_warmup
import torch
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

# Custom Dataset Class
class InfluenceDataset(Dataset):
    def __init__(self, data, tokenizer, max_len):
        self.data = data
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        row = self.data.iloc[index]
        text = row['text']
        label = row['label']
        narrative_category = row['narrative_category']
        inputs = self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=self.max_len,
            return_tensors='pt'
        )
        return {
            'input_ids': inputs['input_ids'].squeeze(),
            'attention_mask': inputs['attention_mask'].squeeze(),
            'label': torch.tensor(label, dtype=torch.float),
            'narrative_category': narrative_category
        }

# Load Data
df = pd.read_csv('narratives.csv', encoding='utf-8-sig')

# Ensure labels are numeric
label_mapping = {
    "Pro-Russian": 0,
    "Pro-Ukrainian": 1
}
df['label'] = df['label'].map(label_mapping)

# Check for missing values and drop rows with missing labels
df = df.dropna(subset=['label'])

# Train-Test Split
train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)

# Initialize Tokenizer and Model
model_name = 'bert-base-uncased'
tokenizer = BertTokenizer.from_pretrained(model_name)
model = BertForSequenceClassification.from_pretrained(model_name, num_labels=1)

# Data Loaders
max_len = 128
batch_size = 16

train_dataset = InfluenceDataset(train_df, tokenizer, max_len)
test_dataset = InfluenceDataset(test_df, tokenizer, max_len)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

# Training Function
def train_model(model, train_loader, epochs, learning_rate):
    optimizer = AdamW(model.parameters(), lr=learning_rate)
    total_steps = len(train_loader) * epochs
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=total_steps)
    loss_fn = torch.nn.MSELoss()
    model.train()
    for epoch in range(epochs):
        total_loss = 0
        for batch in tqdm(train_loader):
            optimizer.zero_grad()
            input_ids = batch['input_ids']
            attention_mask = batch['attention_mask']
            labels = batch['label']
            outputs = model(input_ids, attention_mask=attention_mask)
            logits = outputs.logits.squeeze()
            loss = loss_fn(logits, labels)
            total_loss += loss.item()
            loss.backward()
            optimizer.step()
            scheduler.step()
        print(f'Epoch {epoch + 1}, Average Loss: {total_loss / len(train_loader):.4f}')

# Train Model
train_model(model, train_loader, epochs=3, learning_rate=2e-5)

# Save Model
model.save_pretrained('sentiment_model')
tokenizer.save_pretrained('sentiment_model')
