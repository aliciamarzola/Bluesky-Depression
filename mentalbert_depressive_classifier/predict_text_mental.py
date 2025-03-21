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

    def find_examples(self, texts, labels):
        vp_count, fp_count, fn_count = 0, 0, 0  # Contadores para os exemplos encontrados
        max_examples = 5  # Número máximo de exemplos para cada categoria

        for text, label in zip(texts, labels):
            prediction = self.predict(text)
            
            if fp_count < max_examples and prediction == 1 and label == 0:
                print("\n===== FALSO POSITIVO (FP) =====")
                print(f"Texto: {text}")
                print("Classificação Predita: Depressivo ✅")
                print("Rótulo Real: Não Depressivo ❌")
                print("-" * 80)
                fp_count += 1

            elif fn_count < max_examples and prediction == 0 and label == 1:
                print("\n===== FALSO NEGATIVO (FN) =====")
                print(f"Texto: {text}")
                print("Classificação Predita: Não Depressivo ❌")
                print("Rótulo Real: Depressivo ✅")
                print("-" * 80)
                fn_count += 1

            elif vp_count < max_examples and prediction == 1 and label == 1:
                print(f"\n===== VERDADEIRO POSITIVO (VP) {vp_count+1} =====")
                print(f"Texto: {text}")
                print("Classificação Predita: Depressivo ✅")
                print("Rótulo Real: Depressivo ✅")
                print("-" * 80)
                vp_count += 1
            
            # Para quando encontrar 5 exemplos de cada
            if vp_count >= max_examples and fp_count >= max_examples and fn_count >= max_examples:
                break
