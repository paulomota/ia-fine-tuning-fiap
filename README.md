# Projeto de Fine-Tuning de Modelo de IA

## Descrição

Este projeto realiza o fine-tuning de um modelo de linguagem (GPT-2) utilizando o dataset "The AmazonTitles-1.3MM". O modelo treinado responde a perguntas baseadas nos títulos e descrições de produtos da Amazon.


## Como Executar

### 1. Clonar o Repositório

Acessar a raiz do projeto
cd tech-challange-3

### 2. Configurar o Ambiente

```bash
pip install -r requirements.txt
```

### 3.Preparar os Dados

#### 3.1 Baixar o arquivo trn.json e colocar na pasta /data

#### 3.2 Executar o comando abaixo

```bash
python scripts/prepare_data.py
```

### 4. Realizar o Fine-Tuning

```bash
python scripts/fine_tune.py
```

### 5. Gerar Respostas

```bash
python scripts/generate_responses.py
```
