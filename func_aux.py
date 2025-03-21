import pandas as pd
import emoji

# Carregar o dataset CSV
df = pd.read_csv("/scratch/gabriel.lemos/Bluesky-Depression/dataset_final.csv")

# Função para remover emojis
def remove_emojis(text):
    return emoji.replace_emoji(text, replace='') if isinstance(text, str) else text

# Aplicar a função a uma coluna específica (substitua 'coluna' pelo nome correto)
df['text'] = df['text'].apply(remove_emojis)

# Salvar o CSV limpo
df.to_csv("dataset_final_f_emoji2.csv", index=False)
