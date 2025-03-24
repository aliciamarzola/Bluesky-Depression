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

        return prediction  # Retorna 1 para "Depressivo" e 0 para "Não Depressivo"

    def find_false_positives(self, texts, labels, num_examples):
        fp_count = 0  # Contador de Falsos Positivos

        for text, label in zip(texts, labels):
            prediction = self.predict(text)

            if prediction == 1 and label == 0:
                print(f"\n===== FALSO POSITIVO {fp_count+1} =====")
                print(f"Texto: {text}")
                print("Classificação Predita: Depressivo ✅")
                print("Rótulo Real: Não Depressivo ❌")
                print("-" * 80)
                fp_count += 1

            if fp_count >= num_examples:
                break  # Para quando encontrar os exemplos desejados
            
    def count_false_positives(self, texts, labels):
        fp_count = 0

        for text, label in zip(texts, labels):
            prediction = self.predict(text)
            if prediction == 1 and label == 0:
                fp_count += 1
        
        return fp_count
    
    def get_false_positives(self, texts, labels):
        false_positives = []

        for text, label in zip(texts, labels):
            prediction = self.predict(text)
            if prediction == 1 and label == 0:
                false_positives.append({"text": text, "depressive": 1})
        
        return false_positives
    
    def get_false_negatives(self, texts, labels):
        false_negatives = []

        for text, label in zip(texts, labels):
            prediction = self.predict(text)
            if prediction == 0 and label == 1:
                false_negatives.append({"text": text, "depressive": 0})
        
        return false_negatives
    
    