import os
import torch
import numpy as np
from omnivoice import OmniVoice
from Words import ListWords
from tqdm import trange
from random import randint as rand, choice, choices as choice_weights, uniform
import soundfile as sf
import json

# Suportes:
from Emotions import NON_VERBAL_TAGS, GENDER, AGE, PITCH, STYLE, ACCENTS

print("Iniciando geração de dados de áudio para wake word 'Atlas'...")
# Parâmetros:
MODEL = "k2-fsa/OmniVoice"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
POISSON_LAMBDA = 10 # Com lambda=10, a maioria dos valores cairá entre os índices 5 e 15 (Tier S e Tier A).

# Pesos:
NON_VERBAL_TAGS_WEIGHTS = [4, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1] # Ponderação para tags não-verbais (sem tag é mais comum)
STYLE_WEIGHTS = [5, 1] # Ponderação, a cada 5 "" (sem estilo) temos 1 com "whisper"
ACCENTS_WEIGHTS = [10, 2, 2, 1, 1, 1, 1, 1, 1, 1, 1] # Ponderação para sotaques (sem sotaque é mais comum)
EPOCH = 10
i = 0 # Contador
data = {}

# Definindo Diretório de Saída
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(BASE_DIR, "Atlas-Dataset")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Instanciando modelo:
Model = OmniVoice.from_pretrained(MODEL).to(DEVICE)

"""
Eu acho que o while aqui é melhor, porque dado o problema do ominivoice com trechos pequenos no pós-processamento.
Tratar o erro n me gerou bons resultados. Então é melhor descartar. Só que pra não perder epoch, então usa o while
que é mais fácil fazer esse controle de epoch. Pq ai eu garanto que só vou mudar de epoch quando houver sucesso efetivo. 
"""

print("Iniciando loop de geração de áudio...")
while i < EPOCH:
    # =========================================
    # Escolha da Prashe - Curva de Poissson
    # A distribuição de Poisson é discreta e adequada para modelar a frequência de eventos
    # =========================================
    
    # Gerando indice - Curva de Poissson
    idx = np.random.poisson(lam=POISSON_LAMBDA)
    idx = min(idx, len(ListWords) - 1)

    # Buscando frase:
    phrase = ListWords[idx]

    # =========================================
    # Escolha da Tag Não-Verbal - Aleatória
    # A escolha aleatória de uma tag não-verbal adiciona variedade emocional aos dados gerados, tornando o modelo mais robusto a diferentes expressões e tons de voz.
    # =========================================
    phrase = f"{choice_weights(NON_VERBAL_TAGS, weights=NON_VERBAL_TAGS_WEIGHTS, k=1)[0]} {phrase}".strip()
    
    # =========================================
    # Escolha dos Atributos de Voz - Aleatória
    # A escolha aleatória de atributos de voz (gênero, idade, pitch, estilo, sotaque) permite criar uma variedade de amostras de áudio que refletem diferentes características vocais. Isso é crucial para treinar um modelo de reconhecimento de voz que seja inclusivo e eficaz para uma ampla gama de usuários.
    # =========================================
    choices = [
        choice(GENDER),
        choice(AGE),
        choice(PITCH),
        choice_weights(STYLE, weights=STYLE_WEIGHTS, k=1)[0],
        choice_weights(ACCENTS, weights=ACCENTS_WEIGHTS, k=1)[0],
    ]

    # =========================================
    # Geração do Áudio
    # O áudio é gerado usando o modelo OmniVoice, onde o texto é a frase
    # =========================================
    try:
        audio = Model.generate(
            text=phrase,
            instruct=", ".join([c for c in choices if c]),
            speed = uniform(0.9, 1.1), # Uma leve variação de velocidade
            language = "Portuguese"
        )

        # Salvando audio:
        output_path = os.path.join(OUTPUT_DIR, f"sample_{i:05d}.wav")
        sf.write(output_path, audio[0], Model.sampling_rate)
        data[f"sample_{i:05d}"] = {
            "person": choices[0],
            "age": choices[1],
            "pitch": choices[2],
            "style": choices[3],
            "accent": choices[4]
        }
        print(f"Áudio gerado e salvo: {output_path} | Frase: '{phrase}' | Atributos: {', '.join([c for c in choices if c])}")
        i += 1 # Incrementa o contador de arquivos gerados apenas em caso de sucesso
    except Exception as e:
        print(f"Erro ao gerar áudio para a frase '{phrase}': {e}")

# Salvando metadados em JSON
metadata_path = os.path.join(OUTPUT_DIR, "metadata.json")
with open(metadata_path, "w") as f:
    json.dump(data, f, indent=4)
print(f"Geração de dados concluída! {EPOCH} amostras de áudio geradas em '{OUTPUT_DIR}' com metadados salvos em 'metadata.json'.")