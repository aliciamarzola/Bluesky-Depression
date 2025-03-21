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
        vp_found, fp_found, fn_found = False, False, False  # Flags para controle
        
        for text, label in zip(texts, labels):
            prediction = self.predict(text)
            if not fp_found and prediction == 1 and label == 0:
                print("\n===== FALSO POSITIVO (FP) =====")
                print(f"Texto: {text}")
                print("Classificação Predita: Depressivo ✅")
                print("Rótulo Real: Não Depressivo ❌")
                print("-" * 80)
                fp_found = True

            elif not fn_found and prediction == 0 and label == 1:
                print("\n===== FALSO NEGATIVO (FN) =====")
                print(f"Texto: {text}")
                print("Classificação Predita: Não Depressivo ❌")
                print("Rótulo Real: Depressivo ✅")
                print("-" * 80)
                fn_found = True

            # Para quando encontrar os três exemplos
            if vp_found and fp_found and fn_found:
                break

    def find_true_positives(self, texts, labels, num_examples=5):
            vp_count = 0  # Contador de Verdadeiros Positivos

            for text, label in zip(texts, labels):
                prediction = self.predict(text)

                if prediction == 1 and label == 1:
                    print(f"\n===== VERDADEIRO POSITIVO {vp_count+1} =====")
                    print(f"Texto: {text}")
                    print("Classificação Predita: Depressivo ✅")
                    print("Rótulo Real: Depressivo ✅")
                    print("-" * 80)

                
                if vp_count >= num_examples:
                    break  # Para quando encontrar os 5 exemplos