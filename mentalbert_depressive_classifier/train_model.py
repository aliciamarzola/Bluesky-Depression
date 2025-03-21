import torch
from transformers import BertTokenizer, BertForSequenceClassification
from torch.utils.data import DataLoader, Dataset
import pandas as pd
from sklearn.model_selection import train_test_split
from torch.optim import AdamW
from tqdm import tqdm
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

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
            padding='max_length',
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt"
        )
        return {
            'input_ids': encoding['input_ids'].squeeze(0),
            'attention_mask': encoding['attention_mask'].squeeze(0),
            'labels': torch.tensor(self.labels[idx], dtype=torch.long)
        }

def train_model(dataset_path="/scratch/gabriel.lemos/Bluesky-Depression/dataset/dataset_final_limpo_emoji.csv", 
                model_save_path="mentalbert_depressive_classifier", 
                model_name="mental/mental-bert-base-uncased", 
                epochs=3):

    df = pd.read_csv(dataset_path)
    df["text"] = df["text"].astype(str).fillna("")
    df = df[df['text'].str.strip() != '']


    texts = df["text"].tolist()
    labels = df["depressive"].tolist()

    train_texts, val_texts, train_labels, val_labels = train_test_split(texts, labels, test_size=0.2, random_state=42)

    tokenizer = BertTokenizer.from_pretrained(model_name)
    train_dataset = TextDataset(train_texts, train_labels, tokenizer)
    val_dataset = TextDataset(val_texts, val_labels, tokenizer)

    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False)

    model = BertForSequenceClassification.from_pretrained(model_name, num_labels=2)
    model.to(device)

    optimizer = AdamW(model.parameters(), lr=2e-5)

    def train(model, dataloader, optimizer, epoch):
        model.train()
        total_loss = 0
        predictions, true_labels = [], []
        progress_bar = tqdm(dataloader, desc=f"Training Epoch {epoch+1}", leave=True)

        for batch in progress_bar:
            input_ids, attention_mask, labels = batch["input_ids"].to(device), batch["attention_mask"].to(device), batch["labels"].to(device)
            optimizer.zero_grad()
            outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            progress_bar.set_postfix(loss=loss.item())

            logits = outputs.logits.detach().cpu()
            batch_predictions = torch.argmax(logits, dim=1).numpy()
            batch_labels = labels.cpu().numpy()
            predictions.extend(batch_predictions)
            true_labels.extend(batch_labels)

        accuracy = accuracy_score(true_labels, predictions)
        precision = precision_score(true_labels, predictions)
        recall = recall_score(true_labels, predictions)
        f1 = f1_score(true_labels, predictions)

        print(f"\nTrain - Epoch {epoch+1}: Loss: {total_loss:.4f}, Acc: {accuracy:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1-score: {f1:.4f}")
        return total_loss / len(dataloader)

    def evaluate(model, dataloader, epoch):
        model.eval()
        correct = 0
        total = 0
        predictions, true_labels = [], []
        progress_bar = tqdm(dataloader, desc=f"Validating Epoch {epoch+1}", leave=True)

        with torch.no_grad():
            for batch in progress_bar:
                input_ids, attention_mask, labels = batch["input_ids"].to(device), batch["attention_mask"].to(device), batch["labels"].to(device)
                outputs = model(input_ids, attention_mask=attention_mask)
                batch_predictions = torch.argmax(outputs.logits, dim=1).cpu().numpy()
                batch_labels = labels.cpu().numpy()
                predictions.extend(batch_predictions)
                true_labels.extend(batch_labels)
                correct += (batch_predictions == batch_labels).sum()
                total += batch_labels.shape[0]

        # Calcular métricas de avaliação
        accuracy = accuracy_score(true_labels, predictions)
        precision = precision_score(true_labels, predictions)
        recall = recall_score(true_labels, predictions)
        f1 = f1_score(true_labels, predictions)

        print(f"\nValidation - Epoch {epoch+1}: Acc: {accuracy:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1-score: {f1:.4f}")
        return accuracy

    for epoch in range(epochs):
        train_loss = train(model, train_loader, optimizer, epoch)
        val_accuracy = evaluate(model, val_loader, epoch)
        print(f"Epoch {epoch+1} Completed - Loss: {train_loss:.4f}, Validation Accuracy: {val_accuracy:.4f}")

    model.save_pretrained(model_save_path)
    tokenizer.save_pretrained(model_save_path)
    print(f"Modelo treinado e salvo em {model_save_path}")

if __name__ == "__main__":
    train_model()
