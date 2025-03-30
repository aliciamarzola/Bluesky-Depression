import torch
from imblearn.over_sampling import RandomOverSampler
from transformers import RobertaTokenizer, RobertaForSequenceClassification
from torch.utils.data import DataLoader, Dataset
import pandas as pd
from sklearn.model_selection import train_test_split
from torch.optim import AdamW
from tqdm import tqdm
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

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

def final_metrics(model, dataloader):
    model.eval()
    all_labels = []
    all_predictions = []

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Calculating Final Metrics", leave=True):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            outputs = model(input_ids, attention_mask=attention_mask)
            predictions = torch.argmax(outputs.logits, dim=1)

            all_labels.extend(labels.cpu().numpy())
            all_predictions.extend(predictions.cpu().numpy())

    accuracy = accuracy_score(all_labels, all_predictions)
    precision = precision_score(all_labels, all_predictions)
    recall = recall_score(all_labels, all_predictions)
    f1 = f1_score(all_labels, all_predictions)
    cm = confusion_matrix(all_labels, all_predictions)

    print("\n=== Métricas finais no conjunto de validação ===")
    print(f"Accuracy:  {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1-Score:  {f1:.4f}")
    print(f"Confusion Matrix:\n{cm}")

def train_model(dataset_path="/scratch/gabriel.lemos/Bluesky-Depression/dataset/dataset_final_3003.csv",
                model_save_path="roberta_classifier", epochs=5):
    df = pd.read_csv(dataset_path)
    df["text"] = df["text"].astype(str).fillna("")
    df = df[df['text'].str.strip() != '']
    df = df.drop_duplicates(subset=['text'])

    texts = df["text"].tolist()
    labels = df["depressive"].tolist()

    train_texts, val_texts, train_labels, val_labels = train_test_split(texts, labels, test_size=0.2, random_state=42)

    ros = RandomOverSampler(sampling_strategy='auto', random_state=42)
    train_texts, train_labels = ros.fit_resample(pd.DataFrame(train_texts), pd.DataFrame(train_labels))
    train_texts = train_texts[0].tolist()
    train_labels = train_labels[0].tolist()

    tokenizer = RobertaTokenizer.from_pretrained("roberta-base")
    train_dataset = TextDataset(train_texts, train_labels, tokenizer)
    val_dataset = TextDataset(val_texts, val_labels, tokenizer)

    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False)

    model = RobertaForSequenceClassification.from_pretrained("roberta-base", num_labels=2)
    model.to(device)

    optimizer = AdamW(model.parameters(), lr=2e-5)

    def train(model, dataloader, optimizer, epoch):
        model.train()
        total_loss = 0
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

        return total_loss / len(dataloader)

    def evaluate(model, dataloader, epoch):
        model.eval()
        correct = 0
        total = 0
        progress_bar = tqdm(dataloader, desc=f"Validating Epoch {epoch+1}", leave=True)

        with torch.no_grad():
            for batch in progress_bar:
                input_ids, attention_mask, labels = batch["input_ids"].to(device), batch["attention_mask"].to(device), batch["labels"].to(device)
                outputs = model(input_ids, attention_mask=attention_mask)
                predictions = torch.argmax(outputs.logits, dim=1)
                correct += (predictions == labels).sum().item()
                total += labels.size(0)

        return correct / total

    for epoch in range(epochs):
        train_loss = train(model, train_loader, optimizer, epoch)
        val_accuracy = evaluate(model, val_loader, epoch)
        print(f"Epoch {epoch+1} Completed - Loss: {train_loss:.4f}, Validation Accuracy: {val_accuracy:.4f}")

    model.save_pretrained(model_save_path)
    tokenizer.save_pretrained(model_save_path)
    print(f"Modelo treinado e salvo em {model_save_path}")
    final_metrics(model, val_loader)

if __name__ == "__main__":
    train_model()
