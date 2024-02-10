import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset
from transformers import BertForSequenceClassification, BertTokenizer, BertModel
from transformers import AdamW
from sklearn.model_selection import train_test_split


# tweets = pd.read_csv("FIFA.csv")
# origTweet = tweets[:300]["Orig_Tweet"]
# origTweet.to_csv('tweetSample.csv')

df = pd.read_csv('tweetTraining.csv')
texts = df['Orig_Tweet'].tolist()
labels = df['Security Classification']
labels += 1
labels = labels.tolist()

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
encoded_texts = tokenizer(texts, padding='max_length', truncation=True, return_tensors='pt', max_length = 64)

labels_tensor = torch.tensor(labels)
dataset = TensorDataset(encoded_texts.input_ids, encoded_texts.attention_mask, labels_tensor)
model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=3)
optimizer = AdamW(model.parameters(), lr=1e-5)
criterion = torch.nn.CrossEntropyLoss()
num_epochs = 5


# Split the dataset into train, validation, and test sets
train_texts, test_texts, train_labels, test_labels = train_test_split(texts, labels, test_size=0.2, random_state=42)
train_texts, val_texts, train_labels, val_labels = train_test_split(train_texts, train_labels, test_size=0.1, random_state=42)

# Tokenize and create datasets
train_encoded_texts = tokenizer(train_texts, padding='max_length', truncation=True, return_tensors='pt', max_length=64)
val_encoded_texts = tokenizer(val_texts, padding='max_length', truncation=True, return_tensors='pt', max_length=64)
test_encoded_texts = tokenizer(test_texts, padding='max_length', truncation=True, return_tensors='pt', max_length=64)

train_dataset = TensorDataset(train_encoded_texts.input_ids, train_encoded_texts.attention_mask, torch.tensor(train_labels))
val_dataset = TensorDataset(val_encoded_texts.input_ids, val_encoded_texts.attention_mask, torch.tensor(val_labels))
test_dataset = TensorDataset(test_encoded_texts.input_ids, test_encoded_texts.attention_mask, torch.tensor(test_labels))

# Create data loaders
batch_size = 32
train_dataloader = DataLoader(train_dataset, batch_size=batch_size)
val_dataloader = DataLoader(val_dataset, batch_size=batch_size)
test_dataloader = DataLoader(test_dataset, batch_size=batch_size)

# Training loop
num_epochs = 5
best_val_accuracy = 0.0
for epoch in range(num_epochs):
    model.train()
    total_loss = 0
    for batch in train_dataloader:
        input_ids, attention_mask, labels = batch
        optimizer.zero_grad()
        output = model(input_ids, attention_mask=attention_mask, labels=labels)
        loss = output.loss
        total_loss += loss.item()
        loss.backward()
        optimizer.step()
    average_loss = total_loss / len(train_dataloader)
    print(f'Epoch {epoch + 1}, Loss: {average_loss}')

    # Validation
    model.eval()
    val_correct = 0
    with torch.no_grad():
        for batch in val_dataloader:
            input_ids, attention_mask, labels = batch
            output = model(input_ids, attention_mask=attention_mask)
            logits = output.logits
            predicted_labels = torch.argmax(logits, dim=1)
            val_correct += (predicted_labels == labels).sum().item()
    val_accuracy = val_correct / len(val_dataset)
    print(f'Validation Accuracy: {val_accuracy * 100:.2f}%')

    # Save the best model based on validation accuracy
    if val_accuracy > best_val_accuracy:
        best_val_accuracy = val_accuracy
        torch.save(model.state_dict(), 'security_model.pth')

# Load the best model
model.load_state_dict(torch.load('security_model.pth'))

# Testing
model.eval()
test_correct = 0
with torch.no_grad():
    for batch in test_dataloader:
        input_ids, attention_mask, labels = batch
        output = model(input_ids, attention_mask=attention_mask)
        logits = output.logits
        predicted_labels = torch.argmax(logits, dim=1)
        test_correct += (predicted_labels == labels).sum().item()

test_accuracy = test_correct / len(test_dataset)
print(f'Test Accuracy: {test_accuracy * 100:.2f}%')

