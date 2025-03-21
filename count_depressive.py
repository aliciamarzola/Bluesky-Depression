import pandas as pd
import os

# Carregar o arquivo CSV
df = pd.read_csv('/scratch/gabriel.lemos/Bluesky-Depression/dataset_final_f_emoji2.csv')  # Substitua 'arquivo.csv' pelo nome do seu arquivo

# Contar a frequência dos valores na coluna desejada
coluna = 'depressive'  # Substitua pelo nome da coluna desejada
contagem = df[coluna].value_counts()

# Exibir o resultado
print(contagem)

# Função para unir dois datasets e remover duplicatas com base em uma coluna específica
def unir_datasets(arquivo1, arquivo2, coluna_chave, diretorio_saida='novo_diretorio'):
    df1 = pd.read_csv(arquivo1)
    df2 = pd.read_csv(arquivo2)
    
        # Remover a coluna 'ID' se existir
    if 'ID' in df2.columns:
        df2 = df2.drop(columns=['id'])
    
    # Concatenar os dois DataFrames
    df_combined = pd.concat([df1, df2], ignore_index=True)
    
    # Remover duplicatas com base na coluna chave
    #df_combined = df_combined.drop_duplicates(subset=[coluna_chave])
    
    # Criar diretório se não existir
    os.makedirs(diretorio_saida, exist_ok=True)
    
    # Salvar o novo dataset no diretório especificado
    caminho_saida = os.path.join(diretorio_saida, '/scratch/gabriel.lemos/Bluesky-Depression/dataset/dataset_final_limpo.csv')
    df_combined.to_csv(caminho_saida, index=False)
    
    return df_combined, caminho_saida

# Exemplo de uso
novo_df, caminho = unir_datasets('/scratch/gabriel.lemos/Bluesky-Depression/dataset_final_f_emoji.csv', '/scratch/gabriel.lemos/Bluesky-Depression/dataset/scrapersdataset_limpo.csv', 'text')
print(f'Dataset salvo em: {caminho}')
