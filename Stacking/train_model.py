import torch
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertForSequenceClassification, DistilBertTokenizer, DistilBertForSequenceClassification
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from imblearn.over_sampling import RandomOverSampler
import pandas as pd
import numpy as np
from tqdm import tqdm

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class TextDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length=128):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        encoding = self.tokenizer(
            self.texts[idx],
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )
        return {
            'input_ids': encoding['input_ids'].squeeze(0),
            'attention_mask': encoding['attention_mask'].squeeze(0),
            'labels': torch.tensor(self.labels[idx], dtype=torch.long)
        }

def train_model_base(model, dataloader, optimizer, epochs=2):
    model.train()
    for epoch in range(epochs):
        progress = tqdm(dataloader, desc=f"Train Epoch {epoch+1}")
        for batch in progress:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            optimizer.zero_grad()
            outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            loss.backward()
            optimizer.step()
            progress.set_postfix(loss=loss.item())

def get_predictions(model, dataloader):
    model.eval()
    preds = []

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Getting Predictions"):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)

            outputs = model(input_ids, attention_mask=attention_mask)
            probs = torch.softmax(outputs.logits, dim=1)
            preds.extend(probs[:, 1].cpu().numpy())  # P(depressivo)
    
    return np.array(preds)

def final_metrics(y_true, y_pred_binary):
    print("\n=== Métricas Finais ===")
    print(f"Accuracy:  {accuracy_score(y_true, y_pred_binary):.4f}")
    print(f"Precision: {precision_score(y_true, y_pred_binary):.4f}")
    print(f"Recall:    {recall_score(y_true, y_pred_binary):.4f}")
    print(f"F1-score:  {f1_score(y_true, y_pred_binary):.4f}")

def train_stacking_model(dataset_path, epochs=2):
    df = pd.read_csv(dataset_path)
    df["text"] = df["text"].astype(str).fillna("").str.lower()
    df = df[df["text"].str.strip() != ""].drop_duplicates(subset="text")

    texts, labels = df["text"].tolist(), df["depressive"].tolist()
    X_train, X_val, y_train, y_val = train_test_split(texts, labels, test_size=0.2, random_state=42)

    # Oversampling
    ros = RandomOverSampler(random_state=42)
    X_train, y_train = ros.fit_resample(pd.DataFrame(X_train), pd.DataFrame(y_train))
    X_train, y_train = X_train[0].tolist(), y_train[0].tolist()

    # Tokenizers e datasets
    bert_tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    distil_tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")

    bert_train_ds = TextDataset(X_train, y_train, bert_tokenizer)
    bert_val_ds = TextDataset(X_val, y_val, bert_tokenizer)

    distil_train_ds = TextDataset(X_train, y_train, distil_tokenizer)
    distil_val_ds = TextDataset(X_val, y_val, distil_tokenizer)

    train_loader_bert = DataLoader(bert_train_ds, batch_size=8, shuffle=True)
    val_loader_bert = DataLoader(bert_val_ds, batch_size=8)

    train_loader_distil = DataLoader(distil_train_ds, batch_size=8, shuffle=True)
    val_loader_distil = DataLoader(distil_val_ds, batch_size=8)

    # Modelos base
    bert_model = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=2).to(device)
    distil_model = DistilBertForSequenceClassification.from_pretrained("distilbert-base-uncased", num_labels=2).to(device)

    # Otimizadores
    optimizer_bert = torch.optim.AdamW(bert_model.parameters(), lr=2e-5)
    optimizer_distil = torch.optim.AdamW(distil_model.parameters(), lr=2e-5)

    # Treina os modelos
    train_model_base(bert_model, train_loader_bert, optimizer_bert, epochs)
    train_model_base(distil_model, train_loader_distil, optimizer_distil, epochs)

    # Obtém predições dos modelos base
    bert_val_preds = get_predictions(bert_model, val_loader_bert)
    distil_val_preds = get_predictions(distil_model, val_loader_distil)

    # Empilha predições como features
    X_meta = np.stack([bert_val_preds, distil_val_preds], axis=1)

    # Meta-modelo
    meta_model = LogisticRegression()
    meta_model.fit(X_meta, y_val)

    # Predição final
    y_pred_final = meta_model.predict(X_meta)
    final_metrics(y_val, y_pred_final)

if __name__ == "__main__":
    train_stacking_model("/scratch/gabriel.lemos/Bluesky-Depression/dataset/dataset_final_3003.csv", epochs=4)
