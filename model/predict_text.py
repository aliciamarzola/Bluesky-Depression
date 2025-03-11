import torch
from transformers import BertTokenizer, BertForSequenceClassification

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class DepressionClassifier:
    def __init__(self, model_path="bert_depressive_classifier"):
        self.tokenizer = BertTokenizer.from_pretrained(model_path)
        self.model = BertForSequenceClassification.from_pretrained(model_path)
        self.model.to(device)
        self.model.eval()

    def predict(self, text):
        inputs = self.tokenizer(text, padding=True, truncation=True, max_length=128, return_tensors="pt").to(device)

        with torch.no_grad():
            output = self.model(**inputs)
            prediction = torch.argmax(output.logits, dim=1).cpu().item()

        return "Depressivo" if prediction == 1 else "Não Depressivo"

    def find_depressive_example(self, texts, labels):
        for text, label in zip(texts, labels):
            prediction = self.predict(text)
            if prediction == "Depressivo" and label == 1:
                print("\n===== EXEMPLO ENCONTRADO =====\n")
                print(f"Texto: {text}")
                print("Classificação Predita: Depressivo")
                print("Rótulo Real: Depressivo")
                print("-" * 80)
                break  # Para assim que encontrar um exemplo válido
