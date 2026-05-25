# Mic Monitor & Wake Word Detection

Um sistema web em tempo real para monitoramento e captura de áudio de microfone, construído em Python. Este projeto serve como base de estudos para um sistema conversacional em tempo real usando LLMs (como o assistente de voz "Atlas").

Atualmente, o projeto suporta streaming contínuo de áudio via WebSockets, monitoramento de volume (RMS e picos), e detecção nativa de Wake Word utilizando o ecossistema `openWakeWord`.

## ✨ Funcionalidades
*   **Captura Assíncrona e Multi-thread:** O áudio é capturado via `sounddevice` em uma thread separada e transmitido eficientemente para clientes conectados usando FastAPI e WebSockets.
*   **Visualização em Tempo Real (Frontend):** 
    *   Painel moderno com grid e design responsivo.
    *   Gráficos renderizados com Canvas (Waveform e Histórico de Barras de Volume).
    *   Peak meter em tempo real.
*   **Detecção de Wake Word (Gatilho de Voz):**
    *   Integrado com a biblioteca `openwakeword`.
    *   Suporte a inferência otimizada com o framework `ONNX`.
    *   Modelo padrão de teste configurado para **"alexa"**.
    *   Feedback visual imediato e contínuo da pontuação da inteligência artificial de detecção, piscando a tela quando a palavra for falada.

## 🛠️ Tecnologias Utilizadas
*   **Backend:** Python 3, FastAPI, Uvicorn (servidor ASGI) e WebSockets.
*   **Áudio & Processamento:** `sounddevice` (PortAudio wrapper), `numpy` (matemática de sinais).
*   **Inteligência Artificial (Wake Word):** `openwakeword`, `onnxruntime`.
*   **Frontend:** HTML5, Vanilla JavaScript, CSS Puro e Canvas API.

## 🚀 Como Executar

### 1. Pré-requisitos
Certifique-se de que o Python esteja instalado e ativado em seu ambiente. Crie um ambiente virtual (recomendado):
```powershell
python -m venv .venv
.\.venv\Scripts\activate
```

### 2. Instalação de Dependências
Instale todos os pacotes requeridos através do `pip`:
```powershell
pip install -r requirements.txt
```

### 3. Rodando a Aplicação
Inicie a aplicação principal (o servidor e o pipeline de escuta em segundo plano):
```powershell
python main.py
```

### 4. Acessando a Interface
Abra o seu navegador web favorito e acesse:
```
http://127.0.0.1:8000
```
Clique em **"Iniciar Captura"** na parte inferior da interface web para que o pipeline comece a transmitir dados pelo socket. Diga "Alexa" para ativar o evento visual de wake word.

## 🔍 Expandindo o Wake Word ("Atlas")
Apesar de o projeto ter como alvo final a palavra de ativação "Atlas", o `openWakeWord` não possui esse modelo nativamente.
O treino customizado usa uma venv separada, com dependencias proprias em `ModelTraning/requirements.txt`.
Veja `ModelTraning/README.md`.
