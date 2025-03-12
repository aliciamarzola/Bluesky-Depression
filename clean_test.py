import pandas as pd

# Caminho do dataset original
dataset_path = "/scratch/gabriel.lemos/Bluesky-Depression/scrapers/dataset_bluesky-posts-scraper_2025-03-07_21-09-38-269.csv"

# Tentar ler o dataset detectando automaticamente o delimitador correto
try:
    df = pd.read_csv(dataset_path, delimiter=None, engine="python", error_bad_lines=False)
except Exception as e:
    print(f"Erro ao ler o arquivo CSV: {e}")
    exit()

# Garantir que as colunas "id" e "text" existam
if "id" not in df.columns or "text" not in df.columns:
    print("Erro: O dataset não contém as colunas necessárias ('id' e 'text').")
    exit()

# Manter apenas as colunas "id" e "text"
df_cleaned = df[["id", "text"]].copy()

# Criar a nova coluna "depressive" e preencher com 0
df_cleaned["depressive"] = 0

# Caminho para salvar o novo dataset
save_path = "/scratch/gabriel.lemos/Bluesky-Depression/dataset/data_test1"

# Salvar o dataset processado
df_cleaned.to_csv(save_path, index=False)

print(f"Dataset limpo salvo em: {save_path}")
