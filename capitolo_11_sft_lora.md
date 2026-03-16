# Capitolo 11 — Supervised Fine-Tuning, LoRA e Valutazione

> **Di cosa parla questo capitolo:**
> Come trasformare un modello pre-addestrato in un assistente utile, usando Chat Templates per strutturare le conversazioni, SFT per addestrarlo a seguire istruzioni, LoRA per farlo in modo efficiente, e benchmark per misurare quanto è migliorato.

---

## 📌 TL;DR

| Tecnica | Cosa fa | Quando usarla |
|---------|---------|---------------|
| **Chat Templates** | Struttura i messaggi nel formato atteso dal modello | Sempre, con modelli instruct |
| **SFT** | Addestra il modello su coppie domanda-risposta | Quando il prompting non basta |
| **LoRA** | SFT con il 90% dei parametri in meno | Quasi sempre al posto del full fine-tuning |
| **Evaluation** | Misura le prestazioni con benchmark standard | Dopo ogni fine-tuning |

---

## 1. Chat Templates

### Cos'è un Chat Template?

Un **chat template** è il formato con cui i messaggi vengono passati al modello. Ogni modello instruct si aspetta un formato specifico — usare quello sbagliato causa risposte di bassa qualità.

### Base Model vs Instruct Model

- **Base model** (`SmolLM2-135M`): addestrato su testo grezzo, predice il token successivo
- **Instruct model** (`SmolLM2-135M-Instruct`): fine-tunato per seguire istruzioni e dialogare

Per usare un base model come un assistente, devi formattare i prompt nel modo esatto che si aspetta. I modelli instruct gestiscono anche tool use, input multimodali e function calling.

### Il formato dei messaggi

```python
# Struttura base di una conversazione
messages = [
    {"role": "system", "content": "Sei un assistente utile."},
    {"role": "user", "content": "Ciao!"},
    {"role": "assistant", "content": "Ciao! Come posso aiutarti?"},
    {"role": "user", "content": "Che tempo fa?"},
]
```

**Spiegazione riga per riga:**
- `"role": "system"` → istruzione di comportamento per il modello (opzionale ma utile)
- `"role": "user"` → messaggio dell'utente
- `"role": "assistant"` → risposta del modello (usata nella cronologia del dialogo)

### Formati diversi per modelli diversi

```
# ChatML (SmolLM2, Qwen 2):
<|im_start|>system
Sei un assistente utile.<|im_end|>
<|im_start|>user
Ciao!<|im_end|>
<|im_start|>assistant

# Mistral:
[INST] Sei un assistente utile. [/INST]
[INST] Ciao! [/INST]

# Llama 3:
<|system|>Sei un assistente utile.<|end|>
<|user|>Ciao!<|end|>
<|assistant|>
```

> ⚠️ **Non mescolare mai formati diversi nello stesso progetto!**

### `apply_chat_template()` — il modo corretto

```python
from transformers import AutoTokenizer

# Carica il tokenizer del modello che userai
tokenizer = AutoTokenizer.from_pretrained("HuggingFaceTB/SmolLM2-135M-Instruct")

messages = [
    {"role": "system", "content": "Sei un assistente utile."},
    {"role": "user", "content": "Ciao!"},
]

# Il tokenizer applica AUTOMATICAMENTE il template corretto del modello
# tokenize=False → restituisce testo leggibile (utile per debug)
testo_formattato = tokenizer.apply_chat_template(messages, tokenize=False)
print(testo_formattato)

# tokenize=True → restituisce i token pronti per il modello
input_ids = tokenizer.apply_chat_template(messages, tokenize=True, return_tensors="pt")
```

**Perché usare `apply_chat_template`:** evita errori manuali di formattazione. Il tokenizer conosce già il formato esatto del suo modello.

### Template avanzati: Tool Use

```python
messages = [
    {
        "role": "system",
        "content": "Sei un assistente con accesso a strumenti: calculator, weather_api",
    },
    {"role": "user", "content": "Quanto fa 123 * 456 e piove a Parigi?"},
    {
        "role": "assistant",
        "content": "Ti aiuto subito.",
        "tool_calls": [
            {
                "tool": "calculator",
                "parameters": {"operation": "multiply", "x": 123, "y": 456},
            },
            {"tool": "weather_api", "parameters": {"city": "Paris", "country": "France"}},
        ],
    },
    # Risultati degli strumenti
    {"role": "tool", "tool_name": "calculator", "content": "56088"},
    {"role": "tool", "tool_name": "weather_api", "content": "{'condition': 'pioggia', 'temperature': 15}"},
]
```

### Template avanzati: Input Multimodali

```python
messages = [
    {
        "role": "user",
        "content": [
            {"type": "text", "text": "Cosa c'è in questa immagine?"},
            {"type": "image", "image_url": "https://esempio.com/immagine.jpg"},
        ],
    },
]
```

---

## 🗂️ Glossario Chat Templates

| Termine | Significato |
|---------|-------------|
| **Chat template** | Schema di formattazione dei messaggi per un modello specifico |
| **ChatML** | Formato usato da SmolLM2, Qwen; usa `<\|im_start\|>` e `<\|im_end\|>` |
| **System prompt** | Messaggio di sistema che definisce il comportamento del modello |
| **apply_chat_template()** | Metodo del tokenizer che formatta automaticamente i messaggi |
| **Tool call** | Struttura dati che rappresenta la chiamata a uno strumento esterno |

---

## 2. Supervised Fine-Tuning (SFT)

### Cos'è SFT?

SFT (**Supervised Fine-Tuning**) è il processo di addestramento di un modello pre-addestrato su un dataset di coppie istruzione-risposta. Trasforma un modello "base" (che predice il testo) in un "assistente" (che segue istruzioni).

**Esempio pratico:** GPT-4, Claude, e praticamente tutti gli LLM moderni che usi nelle chat hanno subito SFT.

### Quando usare SFT?

Prima di avviare SFT, chiediti: *il problema si risolve con un buon prompt?*

✅ **Usa SFT quando:**
- Il prompting non riesce a ottenere le prestazioni necessarie
- Hai bisogno di formati di output molto specifici e costanti
- Vuoi adattare il modello a un dominio specializzato (medicina, diritto, codice)
- Il costo di usare un modello grande in produzione supera il costo del fine-tuning di uno piccolo

❌ **Non usare SFT se:**
- Un modello instruct già esistente con il giusto prompt risolve il tuo problema
- Non hai abbastanza dati di qualità
- Non hai risorse computazionali

### Dataset per SFT

Il dataset deve contenere coppie input-output. Formato tipico:

```python
# Ogni esempio deve avere:
# - input: il prompt / la domanda
# - output: la risposta desiderata
# - (opzionale) context: informazioni aggiuntive

# Esempio con HuggingFaceTB/smoltalk:
from datasets import load_dataset
dataset = load_dataset("HuggingFaceTB/smoltalk", "all")
# Dataset["train"] ha colonna "messages" con struttura ChatML
```

### Implementazione con `SFTTrainer` (TRL)

```python
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import SFTConfig, SFTTrainer, setup_chat_format
import torch

# 1. Scegli il device
device = "cuda" if torch.cuda.is_available() else "cpu"

# 2. Carica il dataset
dataset = load_dataset("HuggingFaceTB/smoltalk", "all")

# 3. Carica modello e tokenizer
model_name = "HuggingFaceTB/SmolLM2-135M"
model = AutoModelForCausalLM.from_pretrained(model_name).to(device)
tokenizer = AutoTokenizer.from_pretrained(model_name)

# 4. Configura il formato chat sul modello
# setup_chat_format aggiunge token speciali e imposta il template
model, tokenizer = setup_chat_format(model=model, tokenizer=tokenizer)

# 5. Configura i parametri di training
training_args = SFTConfig(
    output_dir="./sft_output",     # Dove salvare i checkpoint
    max_steps=1000,                # Numero massimo di step (alternativa: num_train_epochs)
    per_device_train_batch_size=4, # Esempi per GPU per step
    learning_rate=5e-5,            # Velocità di apprendimento
    logging_steps=10,              # Log ogni 10 step
    save_steps=100,                # Salva checkpoint ogni 100 step
    eval_strategy="steps",         # Valuta ogni N step
    eval_steps=50,                 # Valuta ogni 50 step
)

# 6. Crea il trainer
trainer = SFTTrainer(
    model=model,
    args=training_args,
    train_dataset=dataset["train"],
    eval_dataset=dataset["test"],
    processing_class=tokenizer,    # Nuovo nome di tokenizer in TRL recente
)

# 7. Avvia il training
trainer.train()
```

**Spiegazione parametri chiave di `SFTConfig`:**

| Parametro | Cosa fa | Valore iniziale consigliato |
|-----------|---------|----------------------------|
| `num_train_epochs` | Quante volte passare sull'intero dataset | 1–3 |
| `max_steps` | Numero fisso di step (ignora epochs se impostato) | 500–2000 |
| `per_device_train_batch_size` | Batch size per GPU | 4–8 |
| `gradient_accumulation_steps` | Accumula N step prima di aggiornare i pesi (simula batch più grande) | 2–8 |
| `learning_rate` | Quanto cambiare i pesi ad ogni step | `5e-5` |
| `warmup_ratio` | Frazione del training per aumentare gradualmente il LR | 0.03–0.1 |
| `logging_steps` | Ogni quanti step loggare le metriche | 10–50 |

> 💡 **Tip:** Se hai poca memoria, aumenta `gradient_accumulation_steps` e riduci `per_device_train_batch_size`. L'effetto è lo stesso di un batch più grande.

### Packing del Dataset

Il **packing** combina più esempi brevi in un'unica sequenza, massimizzando l'utilizzo della GPU:

```python
# Packing semplice
training_args = SFTConfig(packing=True)
trainer = SFTTrainer(model=model, train_dataset=dataset, args=training_args)

# Packing con funzione di formattazione personalizzata
def formatting_func(example):
    # Combina campi multipli in una stringa
    text = f"### Domanda: {example['question']}\n### Risposta: {example['answer']}"
    return text

training_args = SFTConfig(packing=True)
trainer = SFTTrainer(
    "facebook/opt-350m",
    train_dataset=dataset,
    args=training_args,
    formatting_func=formatting_func,
)
```

### Monitorare il Training

La loss durante il training segue tipicamente 3 fasi:

```
Alta loss
    │
    │ ↘ Calo rapido iniziale (adattamento al nuovo dataset)
    │     ↘ Stabilizzazione graduale (fine-tuning)
    │          ↘ Convergenza (training completato)
    └────────────────────────────────── step
```

**Segnali di allarme:**

| Pattern | Problema | Soluzione |
|---------|----------|-----------|
| Val loss sale, Train loss scende | **Overfitting** | Riduci step, aumenta dati |
| Loss non migliora | **Underfitting** | Aumenta LR o controlla qualità dati |
| Loss bassissima subito | **Memorizzazione** | Controlla output diversità |

---

## 🗂️ Glossario SFT

| Termine | Significato |
|---------|-------------|
| **SFT** | Supervised Fine-Tuning — addestrare su coppie domanda/risposta etichettate |
| **SFTTrainer** | Classe di TRL per SFT, costruita sopra Trainer di HF |
| **TRL** | Transformers Reinforcement Learning — libreria HF per fine-tuning |
| **Packing** | Tecnica che combina esempi brevi per riempire la sequenza e ridurre sprechi |
| **Overfitting** | Il modello memorizza i dati di training invece di imparare pattern generali |
| **gradient_accumulation** | Accumula gradienti di più step prima di aggiornare i pesi |

---

## 3. LoRA (Low-Rank Adaptation)

### Cos'è LoRA?

**LoRA** è una tecnica di fine-tuning *parameter-efficient* (PEFT). Invece di modificare tutti i miliardi di parametri del modello, aggiunge piccole matrici di aggiornamento a rank ridotto nei layer di attenzione.

**Il risultato:** riduce il numero di parametri addestrabili di **~90–99%**, permettendo di fare fine-tuning su GPU consumer.

**Esempio storico:** applicato a GPT-3 (175B), LoRA ha ridotto:
- Parametri addestrabili: 10.000x
- Memoria GPU necessaria: 3x

### Come funziona LoRA (concetto)

```
# Invece di aggiornare W (matrice grande):
W_new = W + ΔW    # ΔW ha stessa dimensione di W → costoso!

# LoRA decompone ΔW in due matrici piccole:
W_new = W + B × A
# dove: A ha dimensioni (r × d_input)
#       B ha dimensioni (d_output × r)
#       r (rank) << d_output, d_input
```

Durante l'inferenza, i pesi dell'adapter possono essere **fusi** con il modello base senza aggiungere latenza.

### Parametri di LoRA

```python
from peft import LoraConfig

peft_config = LoraConfig(
    r=6,                    # Rank: dimensione delle matrici piccole (tipico: 4–32)
                            # Più alto = più espressivo, più memoria
    lora_alpha=8,           # Fattore di scaling (tipico: 2x il rank)
                            # Controlla quanto peso dare all'adapter
    lora_dropout=0.05,      # Dropout sulle matrici LoRA (previene overfitting)
    bias="none",            # Bias da addestrare: "none", "all", "lora_only"
                            # "none" = più efficiente in memoria
    target_modules="all-linear",  # A quali layer applicare LoRA
                                   # Alternativa: ["q_proj", "v_proj"]
    task_type="CAUSAL_LM",  # Tipo di task (CAUSAL_LM per generazione testo)
)
```

**Guida alla scelta del rank `r`:**

| Rank | Parametri | Quando usarlo |
|------|-----------|---------------|
| 4–8 | Minimo | Adattamenti semplici, task simili al pre-training |
| 16–32 | Moderato | Adattamenti più complessi, nuovi domini |
| 64+ | Alto | Raramente necessario, quasi come full fine-tuning |

### SFT con LoRA (workflow completo)

```python
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import SFTConfig, SFTTrainer
from peft import LoraConfig
import torch

# 1. Carica modello (nessuna modifica rispetto a SFT normale)
model_name = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.float16,  # Usa float16 per risparmiare memoria
    device_map="auto"           # Distribuisce automaticamente su GPU disponibili
)
tokenizer = AutoTokenizer.from_pretrained(model_name)

# 2. Definisci la configurazione LoRA
peft_config = LoraConfig(
    r=6,
    lora_alpha=8,
    lora_dropout=0.05,
    bias="none",
    target_modules="all-linear",
    task_type="CAUSAL_LM",
)

# 3. Configura il training
args = SFTConfig(
    output_dir="./lora_output",
    max_steps=1000,
    per_device_train_batch_size=2,
    learning_rate=2e-4,          # LoRA tipicamente usa LR più alto del full fine-tuning
    max_seq_length=1024,
)

# 4. Crea il trainer — passa peft_config per attivare LoRA
trainer = SFTTrainer(
    model=model,
    args=args,
    train_dataset=dataset["train"],
    peft_config=peft_config,     # ← QUI si attiva LoRA!
    max_seq_length=1024,
    processing_class=tokenizer,
)

# 5. Training (solo i parametri LoRA vengono aggiornati)
trainer.train()

# 6. Salva solo i pesi dell'adapter (molto leggeri!)
trainer.model.save_pretrained("./lora_adapter")
```

### Caricare un Adapter LoRA esistente

```python
from peft import PeftModel, PeftConfig

# Carica la configurazione dell'adapter per sapere quale base model usare
config = PeftConfig.from_pretrained("ybelkada/opt-350m-lora")

# Carica il modello base
model = AutoModelForCausalLM.from_pretrained(config.base_model_name_or_path)

# Applica l'adapter al modello base
lora_model = PeftModel.from_pretrained(model, "ybelkada/opt-350m-lora")

# Per usare un adapter diverso:
lora_model.set_adapter("altro_adapter")

# Per tornare al modello base puro:
base_model = lora_model.unload()
```

### Fare il Merge: Fondere Adapter + Base Model

Per il deployment in produzione conviene fondere i pesi per eliminare la doppia carica:

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

# 1. Carica il modello base
base_model = AutoModelForCausalLM.from_pretrained(
    "nome_modello_base",
    torch_dtype=torch.float16,
    device_map="auto"
)

# 2. Carica l'adapter sopra il modello base
peft_model = PeftModel.from_pretrained(
    base_model,
    "percorso/adapter",
    torch_dtype=torch.float16
)

# 3. Fonde i pesi e rimuove le strutture LoRA
# Risultato: modello normale senza overhead di adapter
merged_model = peft_model.merge_and_unload()

# 4. Salva il modello fuso (includi anche il tokenizer!)
tokenizer = AutoTokenizer.from_pretrained("nome_modello_base")
merged_model.save_pretrained("percorso/modello_fuso")
tokenizer.save_pretrained("percorso/modello_fuso")
```

> ⚠️ **Ricorda:** serve memoria sufficiente per caricare sia il base model che l'adapter contemporaneamente durante il merge.

---

## 🗂️ Glossario LoRA

| Termine | Significato |
|---------|-------------|
| **LoRA** | Low-Rank Adaptation — fine-tuning con matrici a basso rango |
| **PEFT** | Parameter-Efficient Fine-Tuning — umbrella di tecniche come LoRA |
| **Rank (r)** | Dimensione delle matrici LoRA; più basso = meno parametri |
| **lora_alpha** | Fattore di scaling dell'adapter; tipicamente 2x il rank |
| **QLoRA** | LoRA + quantizzazione del base model (ancora meno memoria) |
| **merge_and_unload()** | Fonde i pesi dell'adapter nel modello base, rimuovendo strutture LoRA |
| **target_modules** | Layer a cui applicare LoRA (es. query e value nei transformer) |

---

## 4. Valutazione del Modello Fine-Tunato

### Perché valutare?

Dopo il fine-tuning devi misurare se il modello è migliorato. I benchmark automatici forniscono un punto di partenza oggettivo e riproducibile.

### Benchmark Principali

**Conoscenza Generale:**

| Benchmark | Cosa testa | Note |
|-----------|-----------|------|
| **MMLU** | Conoscenza su 57 materie (scienze, umanistiche, ecc.) | Standard de-facto |
| **TruthfulQA** | Tendenza a ripetere misconcezioni comuni | Utile per hallucination |

**Ragionamento:**

| Benchmark | Cosa testa | Note |
|-----------|-----------|------|
| **BBH** (Big Bench Hard) | Ragionamento logico e pianificazione | Molto difficile |
| **GSM8K** | Problem solving matematico su problemi scolastici | Ottimo per math |
| **MATH** | Problemi da competizioni matematiche, multi-step | Molto difficile |

**Coding:**

| Benchmark | Cosa testa | Note |
|-----------|-----------|------|
| **HumanEval** | 164 problemi Python da risolvere | Esegue il codice generato! |

**Chat / Instruction Following:**

| Benchmark | Cosa testa | Note |
|-----------|-----------|------|
| **Alpaca Eval** | Qualità risposte (usa GPT-4 come giudice) | LLM-as-Judge |
| **Chatbot Arena** | Preferenze umane (battaglia tra due modelli) | Crowdsourcing |

### Approcci Alternativi

**LLM-as-Judge:** usa un LLM (es. GPT-4) per valutare le risposte di un altro modello. Più flessibile dei benchmark statici, cattura sfumature qualitative.

**Custom Benchmark:** crea un dataset di valutazione specifico per il tuo caso d'uso:
- Esempi reali del dominio target
- Edge case che hai incontrato
- Scenari difficili caratteristici del tuo deployment

### Valutazione con `lighteval`

```bash
# Installa lighteval
pip install lighteval

# Valuta su benchmark MMLU in ambito medico (zero-shot)
lighteval accelerate \
    "pretrained=tuo-modello" \
    "mmlu|anatomy|0|0" \
    "mmlu|high_school_biology|0|0" \
    "mmlu|high_school_chemistry|0|0" \
    "mmlu|professional_medicine|0|0" \
    --max_samples 40 \
    --batch_size 1 \
    --output_path "./risultati" \
    --save_generations true
```

**Formato del task lighteval:**
```
{suite}|{task}|{num_few_shot}|{auto_reduce}

Esempi:
"mmlu|abstract_algebra|0|0"  → MMLU, algebra, zero-shot
"mmlu|anatomy|5|1"           → MMLU, anatomia, 5-shot con riduzione auto
```

**Output atteso:**
```
|                  Task                  |Version|Metric|Value |   |Stderr|
|----------------------------------------|------:|------|-----:|---|-----:|
|leaderboard:mmlu:anatomy:5              |      0|acc   |0.4500|±  |0.1141|
|leaderboard:mmlu:high_school_biology:5  |      0|acc   |0.1500|±  |0.0819|
```

---

## Quando usarlo

| Tecnica | Scenario ideale |
|---------|----------------|
| **Chat Templates** | Qualsiasi uso di modelli instruct — sempre necessario |
| **SFT** | Nuovo dominio specializzato, formato output specifico, insufficienza del prompting |
| **LoRA** | Quasi sempre preferibile al full fine-tuning; usa su GPU con <16GB VRAM |
| **QLoRA** | GPU con <8GB VRAM (es. notebook consumer) |
| **Evaluation** | Dopo ogni iterazione di fine-tuning, prima del deployment |

---

## Errori comuni

| Errore | Causa | Soluzione |
|--------|-------|-----------|
| Risposte strane con modello instruct | Template di chat sbagliato | Controlla `tokenizer_config.json` su HF Hub; usa `apply_chat_template()` |
| CUDA out of memory durante SFT | Batch troppo grande o modello troppo grande | Riduci `per_device_train_batch_size`, usa LoRA + `load_in_4bit=True` |
| Loss non scende | LR troppo basso o dataset di scarsa qualità | Aumenta LR, controlla la qualità dei dati |
| Overfitting rapido | Dataset troppo piccolo o troppi epoch | Riduci `num_train_epochs`, aggiungi dati |
| Adapter non si carica | Base model diverso da quello usato nel training | Usa `PeftConfig.from_pretrained()` per leggere il base model corretto |
| Modello fuso di dimensioni strane | Tokenizer non salvato insieme al modello | Salva sempre `tokenizer.save_pretrained()` nella stessa cartella |

---

## Workflow Completo Consigliato

```
1. Scegli un modello base (es. SmolLM2-135M o DeepSeek-R1-Distill-Qwen-1.5B)
       ↓
2. Prepara il dataset in formato ChatML (colonna "messages")
       ↓
3. Applica apply_chat_template() per verificare il formato
       ↓
4. Configura LoRA (r=6, alpha=8) + SFTConfig
       ↓
5. Avvia training, monitora train/val loss
       ↓
6. Valuta con lighteval su benchmark rilevanti
       ↓
7. Se soddisfatto → merge_and_unload() → deploy
```

---

## Confronto SFT Full vs LoRA

| | Full Fine-Tuning | LoRA |
|--|-----------------|------|
| **Parametri** | Tutti (~miliardi) | Solo adapter (~milioni) |
| **Memoria GPU** | Molta (A100 40GB+) | Poca (consumer 8–16GB) |
| **Velocità training** | Lento | Veloce |
| **Prestazioni** | Massime | Quasi equivalenti |
| **Portabilità** | File grandi (GB) | File piccoli (MB) |
| **Quando usarlo** | Mai quasi — preferisci LoRA | Quasi sempre |
