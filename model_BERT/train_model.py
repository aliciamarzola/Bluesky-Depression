import torch
from transformers import BertTokenizer, BertForSequenceClassification
from torch.utils.data import DataLoader, Dataset
import pandas as pd
from sklearn.model_selection import train_test_split
from torch.optim import AdamW
from tqdm import tqdm
from sklearn.metrics import classification_report

# Definição do dispositivo
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class TextDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length=128):
        self.texts = texts
        self.labels = labels # 0 e 1
        self.tokenizer = tokenizer # converte texto em tokens numéricos
        self.max_length = max_length # comprimento máx da sequencia de tokens

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        encoding = self.tokenizer(
            self.texts[idx],
            padding='max_length',
            truncation=True, # trunca se o texto for maior que o max_length
            max_length=self.max_length,
            return_tensors="pt" # retorna tensores no formato PyTorch
        )
        return {
            'input_ids': encoding['input_ids'].squeeze(0),
            'attention_mask': encoding['attention_mask'].squeeze(0),
            'labels': torch.tensor(self.labels[idx], dtype=torch.long)
        }

def train_model(dataset_path="/scratch/gabriel.lemos/Bluesky-Depression/dataset/dataset_final_limpo_emoji.csv", model_save_path="bert_depressive_classifier", epochs=3):
    df = pd.read_csv(dataset_path)
    df["text"] = df["text"].astype(str).fillna("")
    df = df[df['text'].str.strip() != '']

    texts = df["text"].tolist()
    labels = df["depressive"].tolist()

    # divide em treino e teste
    train_texts, val_texts, train_labels, val_labels = train_test_split(texts, labels, test_size=0.2, random_state=42)

    # converte textos em tokens que o modelo bert consegue entender
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    train_dataset = TextDataset(train_texts, train_labels, tokenizer)
    val_dataset = TextDataset(val_texts, val_labels, tokenizer)

    # carregadores de dados para treinar e validar em lotes de 8 exemplos por vez
    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False)

    # carrega modelo bert pré-treinado
    model = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=2)
    model.to(device)

    # otimizador padrão com taxa de aprendizado 2e-5
    optimizer = AdamW(model.parameters(), lr=2e-5)

    def train(model, dataloader, optimizer, epoch):
        model.train()
        total_loss = 0 # variável pra acumular loss
        progress_bar = tqdm(dataloader, desc=f"Training Epoch {epoch+1}", leave=True) # barra de progresso

        for batch in progress_bar: # a cada lote
            # extrai ids dos tokens, mascara de atenção e rótulos
            input_ids, attention_mask, labels = batch["input_ids"].to(device), batch["attention_mask"].to(device), batch["labels"].to(device)
            optimizer.zero_grad() # zera gradiente acumulado de iterações anteriores
            outputs = model(input_ids, attention_mask=attention_mask, labels=labels) # realiza previsão do modelo
            loss = outputs.loss
            loss.backward() # calcula gradientes de perda
            optimizer.step() # atualiza parametros do modelo usando o otimizador
            total_loss += loss.item() # acumula perda total
            progress_bar.set_postfix(loss=loss.item())

        return total_loss / len(dataloader)

    def evaluate(model, dataloader, epoch):
        model.eval()
        total_loss = 0
        predictions, true_labels = [], []
        progress_bar = tqdm(dataloader, desc=f"Validating Epoch {epoch+1}", leave=True)

        with torch.no_grad():
            for batch in progress_bar:
                # extrai dados
                input_ids, attention_mask, labels = batch["input_ids"].to(device), batch["attention_mask"].to(device), batch["labels"].to(device)
                outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
                loss = outputs.loss
                total_loss += loss.item()
                preds = torch.argmax(outputs.logits, dim=1)
                predictions.extend(preds.cpu().numpy())
                true_labels.extend(labels.cpu().numpy())

        avg_loss = total_loss / len(dataloader)
        report = classification_report(true_labels, predictions, digits=4)
        return avg_loss, report

    for epoch in range(epochs):
        train_loss = train(model, train_loader, optimizer, epoch)
        val_loss, val_report = evaluate(model, val_loader, epoch)
        print(f"Epoch {epoch+1} Completed - Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
        print("Validation Metrics:")
        print(val_report)

    model.save_pretrained(model_save_path)
    tokenizer.save_pretrained(model_save_path)
    print(f"Modelo treinado e salvo em {model_save_path}")

if __name__ == "__main__":
    train_model()
