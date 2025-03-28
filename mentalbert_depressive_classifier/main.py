from predict_text_mental import DepressionClassifier
import pandas as pd
from sklearn.model_selection import train_test_split

# Carregar dataset
df = pd.read_csv("dataset/dataset_novo.csv")
df["text"] = df["text"].astype(str).fillna("")

texts = df["text"].tolist()
labels = df["depressive"].tolist()

_, val_texts, _, val_labels = train_test_split(texts, labels, test_size=0.2, random_state=42)

# Inicializar classificador
classifier = DepressionClassifier()

# Encontrar um exemplo depressivo do conjunto de validação
classifier.find_false_positives(val_texts, val_labels, 15)
#classifier.find_true_positives(val_texts, val_labels, 5)

false_positive_count = classifier.count_false_positives(val_texts, val_labels)
print(f"Número de Falsos Positivos: {false_positive_count}")


# Obter falsos positivos
false_positives = classifier.get_false_positives(val_texts, val_labels)
# Criar novo dataset
df_fp = pd.DataFrame(false_positives)
df_fp.to_csv("/scratch/gabriel.lemos/Bluesky-Depression/dataset/false_positives_dataset.csv", index=False)
print("Novo dataset com falsos positivos salvo como 'false_positives_dataset.csv'")
