# Capitolo 2/8 — Avviare un Modello in Produzione (Inference Deployment)

> **Di cosa parla questo capitolo**: come far girare un LLM in modo efficiente per rispondere a molte richieste contemporaneamente. Tre strumenti principali: **TGI**, **vLLM** e **llama.cpp**.

---

## 📌 TL;DR

| Strumento | Ideale per | Hardware |
|-----------|-----------|----------|
| **TGI** (Text Generation Inference) | Server di produzione, API enterprise | GPU NVIDIA |
| **vLLM** | Alta concorrenza, molte richieste parallele | GPU NVIDIA |
| **llama.cpp** | CPU, laptop, edge devices | CPU / GPU (qualsiasi) |

---

## 🧠 Concetti chiave (leggi prima)

> 💡 **Glossario rapido**
>
> - **Inference**: il modello "pensa" e genera una risposta. Diverso dal training (dove impara).
> - **Throughput**: quante richieste al secondo il sistema riesce a gestire.
> - **Latenza**: quanto tempo passa tra "invio la domanda" e "ricevo la risposta".
> - **VRAM**: memoria della GPU. I modelli grandi richiedono molta VRAM.
> - **Quantizzazione**: tecnica per ridurre le dimensioni del modello comprimendo i numeri (es. da 32 bit a 4 bit). Si perde un po' di qualità, si guadagna moltissimo in velocità e memoria.
> - **Batching**: raggruppare più richieste insieme e processarle in un colpo solo → più efficiente.

---

## 1. Text Generation Inference (TGI)

### Cos'è
TGI è il server di HuggingFace per servire LLM in produzione. Ottimizzato per GPU NVIDIA, supporta i modelli più diffusi (Llama, Mistral, Falcon, ecc.).

### Tecnologie sotto il cofano
- **Flash Attention**: ottimizza come la GPU calcola l'attenzione → usa meno memoria, va più veloce.
- **Continuous Batching**: invece di aspettare che tutte le richieste finiscano insieme, processa i token in modo continuo → meno attese per l'utente.
- **Tensor Parallelism**: divide il modello su più GPU contemporaneamente.

### Installazione (Docker — consigliato)

```bash
# Avvia TGI con Docker
# --model-id: quale modello scaricare da HuggingFace
# --num-shard: su quante GPU dividere il modello
# -p 8080:80: espone il server sulla porta 8080

docker run --gpus all \
  --shm-size 1g \
  -p 8080:80 \
  ghcr.io/huggingface/text-generation-inference:latest \
  --model-id mistralai/Mistral-7B-Instruct-v0.1 \
  --num-shard 1
```

### Usare TGI con Python

```python
from huggingface_hub import InferenceClient

# Crea un client che parla col server TGI locale
client = InferenceClient("http://localhost:8080")

# Genera testo: il modello continua la frase
response = client.text_generation(
    "Spiega cosa è il machine learning in modo semplice:",
    max_new_tokens=200,  # max token da generare
    temperature=0.7,     # creatività (0=deterministico, 1=creativo)
    do_sample=True       # usa campionamento probabilistico
)

print(response)  # restituisce una stringa
```

### TGI con API compatibile OpenAI

```python
from openai import OpenAI

# TGI espone un endpoint compatibile con le API di OpenAI
# Così puoi usare lo stesso codice per OpenAI e per modelli locali!
client = OpenAI(
    base_url="http://localhost:8080/v1",  # punta al tuo server TGI
    api_key="not-needed"                  # non serve per uso locale
)

response = client.chat.completions.create(
    model="tgi",  # nome arbitrario, TGI usa il modello caricato
    messages=[
        {"role": "user", "content": "Cosa è il machine learning?"}
    ],
    max_tokens=200
)

# La risposta ha la stessa struttura di OpenAI
print(response.choices[0].message.content)
```

---

## 2. vLLM

### Cos'è
vLLM è un motore di inference open-source con focus sull'alto throughput. Ottimo quando hai molti utenti che fanno richieste contemporaneamente.

### Tecnologia chiave: PagedAttention
I modelli tradizionali riservano un blocco di memoria fisso per ogni richiesta → spreco. PagedAttention gestisce la memoria in **pagine dinamiche** come un sistema operativo, riciclando i blocchi liberati → più utenti in parallelo con la stessa VRAM.

### Installazione

```bash
pip install vllm
```

### Avviare il server vLLM

```bash
# Avvia un server API compatibile OpenAI
python -m vllm.entrypoints.openai.api_server \
  --model mistralai/Mistral-7B-Instruct-v0.1 \
  --host 0.0.0.0 \
  --port 8000
```

### Usare vLLM con Python

```python
from openai import OpenAI

# Stesso pattern di TGI: endpoint compatibile OpenAI
client = OpenAI(
    base_url="http://localhost:8000/v1",
    api_key="not-needed"
)

response = client.chat.completions.create(
    model="mistralai/Mistral-7B-Instruct-v0.1",
    messages=[
        {"role": "system", "content": "Sei un assistente utile."},
        {"role": "user", "content": "Spiega il deep learning."}
    ],
    max_tokens=300,
    temperature=0.8
)

print(response.choices[0].message.content)
```

### Uso diretto in Python (senza server HTTP)

```python
from vllm import LLM, SamplingParams

# Carica il modello direttamente in memoria
llm = LLM(model="mistralai/Mistral-7B-Instruct-v0.1")

# Parametri di campionamento
params = SamplingParams(
    temperature=0.8,   # creatività
    top_p=0.95,        # considera solo i token più probabili che coprono il 95% della massa
    max_tokens=200
)

# Genera per una lista di prompt in batch
prompts = [
    "Cos'è il machine learning?",
    "Spiega le reti neurali.",
]

outputs = llm.generate(prompts, params)

for output in outputs:
    print(output.outputs[0].text)
```

---

## 3. llama.cpp

### Cos'è
llama.cpp è un'implementazione in C++ di modelli LLM che funziona **senza GPU** (o con GPU consumer). Usa la quantizzazione per ridurre drasticamente la memoria richiesta.

### Quando usarlo
- Hai solo una CPU o una GPU consumer (es. RTX 3060)
- Vuoi girare il modello sul tuo laptop
- Stai prototipando senza server dedicati
- Deploy su dispositivi edge (Raspberry Pi, ecc.)

### Formato GGUF
llama.cpp usa il formato **GGUF** — file .gguf che contengono il modello già quantizzato. Si trovano su HuggingFace cercando "GGUF".

Tipi di quantizzazione più comuni:
| Tipo | Bit per peso | Qualità | Memoria |
|------|-------------|---------|---------|
| Q8_0 | 8 bit | ottima | alta |
| Q4_K_M | 4 bit | buona | media |
| Q2_K | 2 bit | sufficiente | bassa |

### Installazione

```bash
pip install llama-cpp-python
```

Per usare la GPU (consigliato se disponibile):
```bash
CMAKE_ARGS="-DLLAMA_CUDA=on" pip install llama-cpp-python
```

### Usare llama.cpp con Python

```python
from llama_cpp import Llama

# Carica il modello dal file .gguf scaricato localmente
# n_gpu_layers: quanti layer mettere in GPU (0=solo CPU, -1=tutti)
# n_ctx: dimensione della finestra di contesto (memoria della conversazione)
llm = Llama(
    model_path="./mistral-7b-instruct-v0.1.Q4_K_M.gguf",
    n_gpu_layers=-1,   # -1 = usa GPU il più possibile
    n_ctx=4096         # 4096 token di contesto
)

# Formato chat compatibile OpenAI
response = llm.create_chat_completion(
    messages=[
        {"role": "system", "content": "Sei un assistente utile in italiano."},
        {"role": "user", "content": "Cos'è il machine learning?"}
    ],
    max_tokens=300,
    temperature=0.7
)

print(response["choices"][0]["message"]["content"])
```

### Avviare come server HTTP

```bash
# llama.cpp include un server HTTP compatibile OpenAI
python -m llama_cpp.server \
  --model ./mistral-7b.Q4_K_M.gguf \
  --host 0.0.0.0 \
  --port 8000
```

---

## 4. Parametri di Generazione Avanzati

Tutti e tre i framework supportano questi parametri per controllare il comportamento del modello:

### Temperature
```python
# temperature = 0: sempre la risposta più probabile (deterministico)
# temperature = 1: distribuzione originale del modello
# temperature > 1: più casuale/creativo ma meno coerente

# Usa bassa temperature per: codice, risposte fattuali
response = client.chat.completions.create(
    model="tgi",
    messages=[{"role": "user", "content": "Calcola 2+2"}],
    temperature=0.1  # quasi deterministico
)

# Alta temperature per: creative writing, brainstorming
response = client.chat.completions.create(
    model="tgi",
    messages=[{"role": "user", "content": "Scrivi una poesia"}],
    temperature=0.9  # più creativo
)
```

### Top-p e Top-k
```python
# top_p (nucleus sampling): considera solo i token che insieme
# coprono il top_p% della probabilità totale
# top_p=0.9 → scarta il 10% di token meno probabili

# top_k: considera solo i top-k token più probabili
# top_k=50 → scarta tutti tranne i 50 più probabili

response = client.chat.completions.create(
    model="tgi",
    messages=[{"role": "user", "content": "Ciao!"}],
    top_p=0.9,   # filtra per probabilità cumulativa
    # top_k=50   # (non standard OpenAI, supportato da TGI/vLLM direttamente)
)
```

### Repetition Penalty
```python
# repetition_penalty > 1: penalizza la ripetizione di token già usati
# Utile per evitare che il modello si "impalli" ripetendo frasi

# Con HuggingFace InferenceClient (TGI)
response = client.text_generation(
    "C'era una volta...",
    repetition_penalty=1.3,  # 1.0 = nessuna penalità
    max_new_tokens=200
)
```

### Stop Sequences
```python
# Ferma la generazione quando il modello produce queste stringhe
# Utile per output strutturati (JSON, codice, ecc.)

response = client.chat.completions.create(
    model="tgi",
    messages=[{"role": "user", "content": "Dai 3 esempi di frutta:"}],
    stop=["4.", "\n\n"]  # si ferma al punto 4 o a doppia riga vuota
)
```

---

## 5. Gestione della Memoria GPU

### Con TGI
```bash
# Limita la VRAM usata (es. 80% della GPU)
docker run --gpus all \
  -p 8080:80 \
  ghcr.io/huggingface/text-generation-inference:latest \
  --model-id mistralai/Mistral-7B-Instruct-v0.1 \
  --cuda-memory-fraction 0.8    # usa solo l'80% della VRAM
```

### Con vLLM
```python
from vllm import LLM

llm = LLM(
    model="mistralai/Mistral-7B-Instruct-v0.1",
    gpu_memory_utilization=0.85,  # usa l'85% della VRAM disponibile
    max_model_len=4096            # limita la lunghezza del contesto
)
```

### Con llama.cpp
```python
llm = Llama(
    model_path="./modello.gguf",
    n_gpu_layers=20,   # metti solo 20 layer in GPU, il resto in CPU
    n_ctx=2048         # contesto più corto = meno memoria
)
```

---

## 🕐 Quando usarlo

| Scenario | Strumento consigliato |
|----------|----------------------|
| API in produzione, molti utenti, GPU server | **TGI** o **vLLM** |
| Throughput massimo, richieste parallele | **vLLM** (PagedAttention) |
| Solo CPU, laptop, prototipo rapido | **llama.cpp** |
| Vuoi API compatibile OpenAI | Tutti e tre (tutti la supportano) |
| Budget limitato, niente GPU | **llama.cpp** con Q4 |
| HuggingFace ecosystem nativo | **TGI** |

---

## ⚠️ Errori comuni

1. **Out of Memory (OOM)** — Il modello non entra in VRAM
   - Soluzione: usa llama.cpp con quantizzazione Q4, oppure riduci `gpu_memory_utilization` in vLLM

2. **Modello non supportato da TGI/vLLM**
   - Non tutti i modelli sono supportati. Controlla la lista ufficiale o usa llama.cpp (più generico)

3. **Latenza alta con molte richieste**
   - TGI e vLLM gestiscono il batching automaticamente, ma assicurati di avere GPU adeguata
   - Con CPU e llama.cpp, è normale che sia più lento

4. **Formato GGUF sbagliato per llama.cpp**
   - Scegli la quantizzazione giusta per la tua RAM: Q8 per qualità, Q4 per bilanciamento, Q2 solo se necessario

5. **`model_path` non trovato in llama.cpp**
   - Devi scaricare manualmente il file .gguf da HuggingFace prima di usarlo

6. **Port già in uso**
   - Se il server non si avvia: `lsof -i :8080` e termina il processo esistente

---

## 📚 Riepilogo

```
TGI         → Server HuggingFace, Flash Attention, produzione enterprise, GPU
vLLM        → Throughput massimo, PagedAttention, molti utenti paralleli, GPU
llama.cpp   → CPU/laptop, quantizzazione GGUF, niente GPU obbligatoria

Tutti e tre espongono API compatibili OpenAI → codice intercambiabile!

Parametri chiave:
- temperature    → creatività della risposta
- top_p / top_k  → filtro sui token candidati
- repetition_penalty → evita ripetizioni
- stop sequences → termina la generazione su pattern specifici
```
