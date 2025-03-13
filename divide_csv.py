import os
import pandas as pd

def split_csv(file_path, output_dir, chunk_size=500):
    """
    Divide um arquivo CSV em partes de `chunk_size` linhas e salva os arquivos resultantes
    em um diretório de destino.
    
    :param file_path: Caminho do arquivo CSV de entrada.
    :param output_dir: Diretório onde os arquivos divididos serão salvos.
    :param chunk_size: Número de linhas por arquivo (padrão: 500).
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    file_name = os.path.splitext(os.path.basename(file_path))[0]
    
    chunk_num = 1
    for chunk in pd.read_csv(file_path, chunksize=chunk_size):
        output_file = os.path.join(output_dir, f"{file_name}_part{chunk_num}.csv")
        chunk.to_csv(output_file, index=False)
        print(f"Arquivo salvo: {output_file}")
        chunk_num += 1

# Exemplo de uso:
split_csv("/scratch/gabriel.lemos/Bluesky-Depression/dataset/dataset_2 - merged.csv.csv", "/scratch/gabriel.lemos/Bluesky-Depression/dataset")
