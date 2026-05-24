# OmniVoice - entradas aceitas

Fonte pesquisada: `k2-fsa/OmniVoice` no commit remoto `33a8ca325d9c95df20512b36864b9041c7532b35`, principalmente `README.md`, `docs/generation-parameters.md`, `docs/voice-design.md`, `omnivoice/models/omnivoice.py`, `omnivoice/cli/infer.py`, `omnivoice/cli/infer_batch.py` e `omnivoice/cli/demo.py`.

## Modos de uso

O modelo usa a mesma API `model.generate(...)` para três modos:

| Modo | Entradas principais | Observação |
|---|---|---|
| Auto voice | `text` | Sem prompt de voz; o modelo escolhe uma voz automaticamente. |
| Voice cloning | `text` + `ref_audio`; opcional `ref_text` | Clona a voz de um áudio de referência. Se `ref_text` não for passado, usa ASR/Whisper para transcrever. |
| Voice design | `text` + `instruct` | Descreve atributos da voz sem áudio de referência. |

## API Python: `model.generate(...)`

| Entrada | Tipo aceito | Obrigatório | Descrição |
|---|---|---:|---|
| `text` | `str` ou `list[str]` | Sim | Texto a sintetizar. Aceita batch via lista. |
| `language` | `str`, `list[str]` ou `None` | Não | Nome do idioma (`"English"`) ou código (`"en"`). Se inválido ou `None`, cai para modo sem idioma fixo. Melhor especificar quando souber. |
| `ref_audio` | caminho `str`, `list[str]`, `(torch.Tensor, sample_rate)` ou lista desses | Não | Áudio de referência para voice cloning. Pode ser arquivo ou waveform em memória. |
| `ref_text` | `str`, `list[str]` ou `None` | Não | Transcrição do `ref_audio`. Se ausente, o modelo tenta transcrever automaticamente. |
| `voice_clone_prompt` | `VoiceClonePrompt`, lista ou `None` | Não | Prompt de cloning pré-computado por `create_voice_clone_prompt(...)`. Se usado junto com `ref_audio`/`ref_text`, ele tem prioridade e os outros são ignorados. |
| `instruct` | `str`, `list[str]` ou `None` | Não | Instrução de voice design, com atributos separados por vírgula. Pode ser combinado com cloning no demo/API, mas o uso principal é sem `ref_audio`. |
| `duration` | `float`, `list[float | None]` ou `None` | Não | Duração fixa de saída em segundos. Tem prioridade sobre `speed`. |
| `speed` | `float`, `list[float | None]` ou `None` | Não | Fator de velocidade: `> 1.0` fala mais rápido; `< 1.0` fala mais devagar. Ignorado quando `duration` está definido. |
| `generation_config` | `OmniVoiceGenerationConfig` ou `None` | Não | Objeto de configuração. Se fornecido, tem precedência sobre os kwargs de geração. |
| `**kwargs` | campos de `OmniVoiceGenerationConfig` | Não | Atalho para passar `num_step`, `guidance_scale`, etc. |

Em batch, entradas escalares podem ser repetidas para todos os itens, ou podem ser listas do mesmo tamanho de `text`.

## `OmniVoiceGenerationConfig`

Estes campos podem ser passados em `generation_config` ou diretamente como kwargs em `generate(...)`.

| Campo | Tipo | Default | Descrição |
|---|---|---:|---|
| `num_step` | `int` | `32` | Passos de decodificação/difusão. Menor é mais rápido; maior tende a melhorar qualidade. |
| `guidance_scale` | `float` | `2.0` | Escala de classifier-free guidance. |
| `t_shift` | `float` | `0.1` | Ajuste da agenda temporal de ruído. |
| `layer_penalty_factor` | `float` | `5.0` | Penalidade por camada/codebook para controlar ordem de unmasking. |
| `position_temperature` | `float` | `5.0` | Temperatura para seleção das posições mascaradas. `0` deixa mais determinístico. |
| `class_temperature` | `float` | `0.0` | Temperatura para amostragem de tokens. `0` usa greedy. |
| `denoise` | `bool` | `True` | Adiciona token de denoise. No código, só entra quando há referência de áudio. |
| `preprocess_prompt` | `bool` | `True` | Preprocessa o áudio/texto de referência: remove silêncios, corta áudio longo sem `ref_text`, adiciona pontuação. |
| `postprocess_output` | `bool` | `True` | Pós-processa saída: remove silêncios longos, ajusta volume e aplica fade/padding. |
| `audio_chunk_duration` | `float` | `15.0` | Duração-alvo dos chunks para textos longos. |
| `audio_chunk_threshold` | `float` | `30.0` | Ativa chunking quando a duração estimada passa desse limite. |

## `create_voice_clone_prompt(...)`

Use quando quiser reaproveitar o mesmo áudio de referência em várias gerações.

| Entrada | Tipo | Default | Descrição |
|---|---|---:|---|
| `ref_audio` | caminho `str` ou `(torch.Tensor, sample_rate)` | Obrigatório | Áudio de referência. O código converte para mono e reamostra para a taxa do modelo. |
| `ref_text` | `str` ou `None` | `None` | Transcrição da referência. Se ausente, usa ASR. |
| `preprocess_prompt` | `bool` | `True` | Remove silêncios, corta áudio muito longo quando possível e adiciona pontuação ao texto. |

Recomendação do README: referência de 3 a 10 segundos. Acima de 20 segundos o código emite warning porque pode piorar qualidade, memória e velocidade.

## `instruct`: atributos de voice design

`instruct` é uma string separada por vírgulas. Só pode haver um item por categoria.

| Categoria | Valores aceitos |
|---|---|
| Gênero | `male`, `female`; ou `男`, `女` |
| Idade | `child`, `teenager`, `young adult`, `middle-aged`, `elderly`; ou `儿童`, `少年`, `青年`, `中年`, `老年` |
| Pitch | `very low pitch`, `low pitch`, `moderate pitch`, `high pitch`, `very high pitch`; ou equivalentes em chinês |
| Estilo | `whisper`; ou `耳语` |
| Sotaque em inglês | `american accent`, `british accent`, `australian accent`, `canadian accent`, `indian accent`, `chinese accent`, `korean accent`, `japanese accent`, `portuguese accent`, `russian accent` |
| Dialeto chinês | `河南话`, `陕西话`, `四川话`, `贵州话`, `云南话`, `桂林话`, `济南话`, `石家庄话`, `甘肃话`, `宁夏话`, `青岛话`, `东北话` |

Regras importantes:

- Inglês é case-insensitive.
- O código aceita vírgula normal `,` e vírgula chinesa `，`.
- Não pode misturar sotaque inglês com dialeto chinês no mesmo `instruct`.
- Se houver conflitos na mesma categoria, por exemplo `male, female`, o modelo lança erro.

Exemplos:

```python
model.generate(text="Hello", instruct="female, young adult, high pitch, british accent")
model.generate(text="你好", instruct="女，青年，高音调，四川话")
```

## Entradas no texto

Além do texto normal, o `text` aceita controles inline:

| Tipo | Entrada |
|---|---|
| Sons não verbais | `[laughter]`, `[sigh]`, `[confirmation-en]`, `[question-en]`, `[question-ah]`, `[question-oh]`, `[question-ei]`, `[question-yi]`, `[surprise-ah]`, `[surprise-oh]`, `[surprise-wa]`, `[surprise-yo]`, `[dissatisfaction-hnn]` |
| Pronúncia chinesa | Pinyin com tom, por exemplo `ZHE2`, `SHE2`, `ZHE1` dentro do texto. |
| Pronúncia inglesa | Fonemas CMU em maiúsculas entre colchetes, por exemplo `[B EY1 S]`. |

## CLI: `omnivoice-infer`

Flags aceitas para inferência unitária:

| Flag | Obrigatório | Mapeia para |
|---|---:|---|
| `--model` | Não | checkpoint local ou repo HuggingFace; default `k2-fsa/OmniVoice` |
| `--text` | Sim | `text` |
| `--output` | Sim | caminho do WAV gerado |
| `--ref_audio` | Não | `ref_audio` |
| `--ref_text` | Não | `ref_text` |
| `--instruct` | Não | `instruct` |
| `--language` | Não | `language` |
| `--num_step` | Não | `num_step` |
| `--guidance_scale` | Não | `guidance_scale` |
| `--speed` | Não | `speed`; default CLI `1.0` |
| `--duration` | Não | `duration` |
| `--t_shift` | Não | `t_shift` |
| `--denoise` | Não | `denoise` |
| `--postprocess_output` | Não | `postprocess_output` |
| `--layer_penalty_factor` | Não | `layer_penalty_factor` |
| `--position_temperature` | Não | `position_temperature` |
| `--class_temperature` | Não | `class_temperature` |
| `--device` | Não | device de inferência; auto-detectado se ausente |

## CLI batch: `omnivoice-infer-batch`

Flags principais:

| Flag | Obrigatório | Descrição |
|---|---:|---|
| `--model` | Não | checkpoint local ou repo HuggingFace |
| `--test_list` | Sim | JSONL de amostras |
| `--res_dir` | Sim | diretório de saída |
| `--num_step`, `--guidance_scale`, `--t_shift`, `--denoise` | Não | parâmetros de geração |
| `--audio_chunk_duration`, `--audio_chunk_threshold` | Não | chunking de textos longos |
| `--preprocess_prompt`, `--postprocess_output` | Não | pré/pós-processamento |
| `--layer_penalty_factor`, `--position_temperature`, `--class_temperature` | Não | sampling/decoding |
| `--lang_id` | Não | idioma padrão quando a linha JSONL não tem `language_id` |
| `--nj_per_gpu`, `--batch_duration`, `--batch_size`, `--warmup` | Não | controle operacional de batching/processos |

Campos aceitos por linha no JSONL:

| Campo | Obrigatório | Descrição |
|---|---:|---|
| `id` | Sim | nome base do arquivo `.wav` de saída |
| `text` | Sim | texto a sintetizar |
| `ref_audio` | Não | caminho do áudio de referência |
| `ref_text` | Não | transcrição do áudio de referência |
| `instruct` | Não | voice design |
| `language_id` | Não | código de idioma, por exemplo `en` |
| `duration` | Não | duração fixa em segundos |
| `speed` | Não | fator de velocidade |

Exemplo:

```json
{"id":"sample_001","text":"Hello world","ref_audio":"/path/ref.wav","ref_text":"Reference transcript","instruct":"female, british accent","language_id":"en","duration":10.0,"speed":1.0}
```

Observação: no batch, o código separa amostras com `ref_audio` das sem `ref_audio`, porque misturar cloning e não-cloning no mesmo batch pode quebrar a preparação do prompt.

## Demo web: `omnivoice-demo`

Flags de inicialização:

| Flag | Descrição |
|---|---|
| `--model` | checkpoint local ou repo HuggingFace |
| `--device` | device de inferência |
| `--ip` | IP do servidor |
| `--port` | porta do servidor |
| `--root-path` | root path para reverse proxy |
| `--share` | cria link público do Gradio |
| `--no-asr` | não carrega ASR; sem auto-transcrição de `ref_text` |
| `--asr-model` | modelo ASR; default `openai/whisper-large-v3-turbo` |

