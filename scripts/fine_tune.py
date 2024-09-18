# scripts/fine_tune.py

from transformers import GPT2LMHeadModel, GPT2Tokenizer, Trainer, TrainingArguments, DataCollatorForLanguageModeling
from datasets import load_dataset
import os

def main():
    # Configurações
    model_name = 'distilgpt2'  
    train_data_path = 'fine_tune_data.csv'
    output_dir = 'fine_tuned_model'

    # Carregar tokenizer e modelo
    tokenizer = GPT2Tokenizer.from_pretrained(model_name)
    model = GPT2LMHeadModel.from_pretrained(model_name)

    # Adicionar token de padding se não existir
    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens({'pad_token': '[PAD]'})
        model.resize_token_embeddings(len(tokenizer))

    # Carregar o dataset
    dataset = load_dataset('csv', data_files={'train': train_data_path})

    # Função de tokenização
    def tokenize_function(examples):
        return tokenizer(examples['prompt'], truncation=True, padding='max_length', max_length=128)

    tokenized_datasets = dataset.map(tokenize_function, batched=True)

    # Data collator
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    # Configurar os argumentos de treinamento com ajustes para acelerar
    training_args = TrainingArguments(
        output_dir=output_dir,
        overwrite_output_dir=True,
        num_train_epochs=1,  # Reduzido para acelerar
        per_device_train_batch_size=2,  # Reduzido para usar menos memória
        save_steps=5000,  # Reduzido para salvar com mais frequência
        save_total_limit=1,
        logging_steps=100,
        learning_rate=5e-5,
        weight_decay=0.01,
        evaluation_strategy="no",  # Alterar para "epoch" se tiver dados de validação
        push_to_hub=False,
        fp16=True,  # Habilitado para acelerar com GPU compatível
    )

    # Inicializar o Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets['train'],
        data_collator=data_collator,
    )

    # Testar o modelo antes do fine-tuning
    test_prompt = "Pergunta: Como funciona o aspirador Dyson?\nResposta:"
    inputs = tokenizer.encode(test_prompt, return_tensors='pt')
    outputs = model.generate(inputs, max_length=50, num_return_sequences=1)
    print("Resposta antes do fine-tuning:")
    print(tokenizer.decode(outputs[0], skip_special_tokens=True))

    # Executar o fine-tuning
    trainer.train()

    # Salvar o modelo fine-tuned
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)
    print(f"Modelo fine-tuned salvo em {output_dir}.")

if __name__ == "__main__":
    main()
