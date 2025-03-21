from predict_text import DepressionClassifier
import pandas as pd
from sklearn.model_selection import train_test_split

# Carregar dataset
df = pd.read_csv("/scratch/gabriel.lemos/Bluesky-Depression/dataset/dataset_final_limpo.csv")
df["text"] = df["text"].astype(str).fillna("")

texts = df["text"].tolist()
labels = df["depressive"].tolist()

_, val_texts, _, val_labels = train_test_split(texts, labels, test_size=0.2, random_state=42)

# Inicializar classificador
classifier = DepressionClassifier()

# Exemplo de uma predição única
texto_exemplo = "Estou me sentindo muito triste e sem esperança."
print(f"Classificação: {classifier.predict(texto_exemplo)}")

# Encontrar um exemplo depressivo do conjunto de validação
#classifier.find_examples(val_texts, val_labels)
classifier.find_true_positives(val_texts, val_labels, 5)
