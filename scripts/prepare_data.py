# scripts/prepare_data.py

import pandas as pd
import json
import os
import re
import random

def clean_text(text):
    """Remove caracteres especiais e converte o texto para minúsculas."""
    text = re.sub(r'[^A-Za-z0-9\s]', '', text)  # Remove caracteres especiais
    text = text.lower()  # Converte para minúsculas
    return text

def prepare_data(input_path, output_path, sample_fraction=None):
    """Carrega, limpa e prepara os dados para o fine-tuning."""
    titles = []
    contents = []

    with open(input_path, 'r', encoding='utf-8') as f:
        for line in f:
            try:
                record = json.loads(line)
                if 'title' in record and 'content' in record:
                    titles.append(record['title'])
                    contents.append(record['content'])
                else:
                    print("Registro sem 'title' ou 'content' encontrado e ignorado.")
            except json.JSONDecodeError as e:
                print(f"Erro ao decodificar JSON: {e}. Linha ignorada.")

    df = pd.DataFrame({
        'title': titles,
        'content': contents
    })

    # Limpar os textos
    df['title'] = df['title'].apply(clean_text)
    df['content'] = df['content'].apply(clean_text)

    # Amostrar os dados, se especificado
    if sample_fraction:
        df = df.sample(frac=sample_fraction, random_state=42).reset_index(drop=True)
        print(f"Utilizando amostra de {sample_fraction*100}% dos dados para acelerar o treinamento.")

    # Criar os prompts e respostas
    prompts = []
    responses = []

    for index, row in df.iterrows():
        prompt = f"Pergunta: {row['title']}\nResposta:"
        response = row['content']
        prompts.append(prompt)
        responses.append(response)

    fine_tune_df = pd.DataFrame({
        'prompt': prompts,
        'response': responses
    })

    fine_tune_df.to_csv(output_path, index=False)
    print(f"Dados preparados e salvos em {output_path}.")

if __name__ == "__main__":
    input_path = os.path.join('data', 'trn.json')
    output_path = 'fine_tune_data.csv'
    sample_fraction = 0.01

    if not os.path.exists(input_path):
        raise FileNotFoundError(f"O arquivo {input_path} não foi encontrado.")

    prepare_data(input_path, output_path, sample_fraction)
