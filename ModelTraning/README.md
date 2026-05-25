# Treino generico de wake word

Esta pasta treina um modelo `openWakeWord` usando uma pasta de WAVs positivos.
O caso normal nao precisa de JSON nem de texto da wake word: informe a pasta dos audios.

Documentacao completa em HTML: `ModelTraning/training.html`.

## Uso rapido

1. Crie uma venv separada para treino.

Use Python 3.10. A stack oficial de treino do `openWakeWord 0.6.0` usa
`torchaudio<1`, entao nao e adequada para o Python 3.13 do ambiente moderno.

```powershell
py -3.10 -m venv ModelTraning\.venv_training
.\ModelTraning\.venv_training\Scripts\activate
python -m pip install --upgrade pip
```

2. Instale as dependencias de treino:

```powershell
pip install -r ModelTraning/requirements.txt
```

O arquivo usa PyTorch/Torchaudio 1.13.1 com CUDA 11.7, uma combinacao alinhada
ao extra `full` publicado pelo `openWakeWord 0.6.0`. Para checar se a GPU esta ativa:

```powershell
python -c "import torch; print(torch.cuda.is_available(), torch.version.cuda)"
```

3. Baixe os modelos internos do openWakeWord e os assets negativos usados pelo treino:

```powershell
python ModelTraning/download_assets.py
```

Esse comando precisa ser executado com a venv de treino ativa, porque tambem baixa
`melspectrogram.onnx` e `embedding_model.onnx` para o pacote `openwakeword` instalado.

4. Prepare os WAVs positivos para o formato esperado pelo openWakeWord:

```powershell
python ModelTraning/train.py DataGeneration/Dataset-Atlas --stage prepare
```

Essa etapa cria uma pasta derivada em `ModelTraning/work/atlas/positive_16k`
com WAVs mono, 16 kHz, PCM_16. Os WAVs originais continuam em
`DataGeneration/Dataset-Atlas`.

5. Gere features e treine:

```powershell
python ModelTraning/train.py DataGeneration/Dataset-Atlas
```

O modelo final sai em:

```text
ModelTraning/models/atlas.onnx
```

Durante a execucao, o script mostra um painel de status com as etapas do pipeline,
contagens de arquivos, shapes dos `.npy`, barras de progresso e mensagens periodicas
em operacoes longas. Para esconder esses indicadores extras, use `--no-visual-progress`.

## Como pensar na arquitetura

Entrada principal:

- Uma pasta com `.wav` positivos da wake word.
- Exemplo: todos os audios onde a pessoa fala "atlas".

O pipeline faz automaticamente:

- Prepara uma copia derivada dos WAVs positivos em mono, 16 kHz, PCM_16.
- Divide os WAVs positivos em `positive_train` e `positive_test`.
- Aplica augmentation nos audios positivos.
- Extrai features no formato esperado pelo `openWakeWord`.
- Usa features negativas genericas baixadas de `davidscripka/openwakeword_features`.
- Treina um classificador binario: `1` para a wake word, `0` para qualquer outra coisa.
- Exporta o modelo `.onnx`.

## Para outra wake word

Gere ou colete os WAVs positivos em outra pasta e rode:

```powershell
python ModelTraning/train.py caminho/para/Dataset-MinhaWakeWord
```

O nome do modelo e inferido pelo nome da pasta. `Dataset-Atlas` vira `atlas.onnx`.
Se quiser outro nome de saida, use `--model-name`.

## Para que serve o JSON

O JSON e opcional. Ele nao e criado por wake word. O arquivo real de preset e
`ModelTraning/training_preset.json`, e existe para parametros avancados, como:

- Quantidade de steps.
- Batch por classe.
- Pastas de ruido de fundo.
- Pastas de RIR.
- Pesos de negativos.
- Exportar tambem para TFLite.

Exemplo com config avancado:

```powershell
python ModelTraning/train.py DataGeneration/Dataset-Atlas --config ModelTraning/training_preset.json
```

Para a rotina comum, nao use JSON. Crie outro JSON somente se quiser outro regime de treino.

## TensorFlow e TFLite

O treino padrao gera ONNX e usa PyTorch. TensorFlow nao e necessario para treinar.
As dependencias de TensorFlow foram separadas em `ModelTraning/requirements-tflite.txt`
porque servem apenas para a conversao opcional `--convert-to-tflite`.

## Opcoes uteis

```powershell
python ModelTraning/train.py DataGeneration/Dataset-Atlas --steps 50000
```

```powershell
python ModelTraning/train.py DataGeneration/Dataset-Atlas --background-dir caminho/ruidos
```

```powershell
python ModelTraning/train.py DataGeneration/Dataset-Atlas --overwrite-split --overwrite-features
```

```powershell
python ModelTraning/train.py DataGeneration/Dataset-Atlas --overwrite-audio --stage prepare
```
