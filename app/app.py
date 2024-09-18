# app/app.py

import streamlit as st
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import torch
import os

@st.cache_resource
def load_model(model_path):
    """Carrega o tokenizer e o modelo fine-tuned."""
    tokenizer = GPT2Tokenizer.from_pretrained(model_path)
    model = GPT2LMHeadModel.from_pretrained(model_path)
    model.eval()
    return tokenizer, model

def gerar_resposta(pergunta, tokenizer, model, max_length=150):
    """Gera uma resposta para a pergunta fornecida."""
    prompt = f"Pergunta: {pergunta}\nResposta:"
    inputs = tokenizer.encode(prompt, return_tensors='pt')
    
    with torch.no_grad():
        outputs = model.generate(
            inputs,
            max_length=max_length,
            num_return_sequences=1,
            no_repeat_ngram_size=2,
            temperature=0.7,
            top_p=0.9,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id
        )
    
    resposta = tokenizer.decode(outputs[0], skip_special_tokens=True)
    resposta = resposta.split("Resposta:")[-1].strip()
    return resposta

def main():
    st.title("Chat com Modelo Fine-Tuned")
    
    model_path = 'fine_tuned_model'
    
    # Verificar se o modelo fine-tuned existe
    if not os.path.exists(model_path):
        st.error(f"A pasta {model_path} não foi encontrada. Execute o fine-tuning primeiro.")
        return
    
    tokenizer, model = load_model(model_path)
    
    pergunta = st.text_input("Faça sua pergunta:")
    
    if st.button("Responder"):
        if pergunta:
            with st.spinner('Gerando resposta...'):
                resposta = gerar_resposta(pergunta, tokenizer, model)
            st.write("**Resposta:**")
            st.write(resposta)
        else:
            st.warning("Por favor, insira uma pergunta.")

if __name__ == "__main__":
    main()
