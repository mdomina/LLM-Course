# Capitolo 8 — Debug, Aiuto e Segnalazione Bug

> **Di cosa parla questo capitolo**: cosa fare quando qualcosa va storto. Come leggere i traceback, debuggare la pipeline di training passo per passo, chiedere aiuto sul forum, e segnalare bug in modo efficace.

---

## 📌 TL;DR

```
Traceback        → leggi dal basso verso l'alto; l'ultima riga è l'errore
Errore pipeline  → debug step-by-step: dati → dataloader → modello → ottimizzatore → valutazione
CUDA error       → sposta il modello su CPU per avere messaggi chiari
Overfit su 1 batch → test rapido per verificare che il modello "possa" imparare
Forum post       → titolo descrittivo + codice formattato + traceback completo + esempio riproducibile
GitHub issue     → minimal reproducible example + info ambiente (`transformers-cli env`)
```

---

## 1. Come Leggere un Traceback Python

> ⚠️ **Regola d'oro**: i traceback si leggono **dal basso verso l'alto**.

L'ultima riga contiene il tipo di eccezione e il messaggio d'errore. Le righe sopra mostrano la sequenza di chiamate che ha portato all'errore.

```
Traceback (most recent call last):
  File "script.py", line 10, in <module>      ← dove è partita la chiamata
    outputs = model(**inputs)
  File ".../transformers/modeling_bert.py", line 473
    input_shape = input_ids.size()             ← dove è esploso l'errore
AttributeError: 'list' object has no attribute 'size'   ← LEGGI QUESTA PRIMA
```

**Strategia di debug:**
1. Leggi l'ultima riga → tipo e messaggio dell'errore
2. Risali il traceback per trovare il punto nel tuo codice (non nelle librerie)
3. Se non capisci → copia l'errore su Google / Stack Overflow
4. Se non trovi → posta sul forum HuggingFace con tutte le informazioni

---

## 2. Debug della Pipeline (errori di inference)

### Errore comune: model ID sbagliato

```python
from transformers import pipeline

# OSError: Can't load config for 'lewtun/distillbert-...'
reader = pipeline("question-answering", model="lewtun/distillbert-base-uncased-...")
#                                                      ↑ typo! "distilLbert" con due L
```

**Come verificare un repository sull'Hub:**

```python
from huggingface_hub import list_repo_files

# Controlla tutti i file presenti nel repo
list(list_repo_files(repo_id="lewtun/distilbert-base-uncased-finetuned-squad"))
# → ['.gitattributes', 'README.md', 'pytorch_model.bin', 'tokenizer_config.json', ...]
# Se manca config.json → il modello non può essere caricato!
```

**Soluzione: recuperare e pushare il config mancante:**

```python
from transformers import AutoConfig

# Scarica il config del modello base (assumendo che non sia stato modificato)
config = AutoConfig.from_pretrained("distilbert-base-uncased")

# Pusha il config nel repo con il problema
config.push_to_hub("tuo-username/nome-repo", commit_message="Add config.json")
```

### Errore comune: tensori mancanti (return_tensors)

```python
# ❌ MALE: il tokenizer restituisce liste Python, non tensori
inputs = tokenizer(question, context, add_special_tokens=True)
outputs = model(**inputs)
# → AttributeError: 'list' object has no attribute 'size'

# ✅ BENE: specifica il framework
inputs = tokenizer(question, context, add_special_tokens=True, return_tensors="pt")
outputs = model(**inputs)
```

**Regola**: passare sempre `return_tensors="pt"` (PyTorch) o `return_tensors="tf"` (TensorFlow) quando si usa il modello direttamente (fuori dalla `pipeline`).

---

## 3. Debug della Pipeline di Training — Step by Step

Quando `trainer.train()` lancia un errore, è difficile capire la fonte perché il `Trainer` combina molte cose. La strategia è **isolare ogni passo manualmente**.

### Checklist di debug (nell'ordine)

```
1. Controlla i dati
2. Controlla il dataloader (creazione dei batch)
3. Testa il forward pass del modello (su CPU)
4. Testa il backward pass (calcolo gradients)
5. Controlla la funzione compute_metrics
```

---

### Step 1: Controlla i Dati

```python
# Ispeziona il primo esempio del training set
trainer.train_dataset[0]
# Se vedi testo grezzo (non numeri) → non hai passato il dataset tokenizzato!

# Decodifica gli input per verificare che siano sensati
tokenizer.decode(trainer.train_dataset[0]["input_ids"])
# → '[CLS] premise text [SEP] hypothesis text [SEP]' ← deve avere senso

# Controlla le chiavi disponibili
trainer.train_dataset[0].keys()
# → ['attention_mask', 'input_ids', 'label', ...]

# Controlla i nomi delle label
trainer.train_dataset.features["label"].names
# → ['entailment', 'neutral', 'contradiction']
```

**Errore classico:** passare `raw_datasets` invece di `tokenized_datasets` al Trainer.

```python
# ❌ MALE
trainer = Trainer(train_dataset=raw_datasets["train"], ...)

# ✅ BENE
trainer = Trainer(train_dataset=tokenized_datasets["train"], ...)
```

---

### Step 2: Controlla il DataLoader

```python
# Tenta di creare il primo batch manualmente
for batch in trainer.get_train_dataloader():
    break
# Se esplode qui → problema nel data collator o nel dataset

# Controlla quale collator viene usato
trainer.get_train_dataloader().collate_fn
# Se vedi default_data_collator → non hai passato il tokenizer al Trainer!

# Applica manualmente il collator su alcuni sample
data_collator = trainer.get_train_dataloader().collate_fn
actual_train_set = trainer._remove_unused_columns(trainer.train_dataset)
batch = data_collator([actual_train_set[i] for i in range(4)])
```

**Errore classico:** non passare il `tokenizer` al Trainer → usa il collator generico invece di `DataCollatorWithPadding`.

```python
# ❌ MALE: nessun tokenizer → padding non funziona
trainer = Trainer(model=model, args=args, ...)

# ✅ BENE: specifica esplicitamente tokenizer e collator
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
trainer = Trainer(
    model=model,
    args=args,
    data_collator=data_collator,
    tokenizer=tokenizer,  # ← necessario per il collator automatico
    ...
)
```

---

### Step 3: Forward Pass del Modello (su CPU)

```python
# Ottieni un batch
for batch in trainer.get_train_dataloader():
    break

# !! IMPORTANTE: testa su CPU per avere messaggi d'errore chiari !!
outputs = trainer.model.cpu()(**batch)
# Se hai CUDA error → questo ti darà un errore leggibile
```

**Errore classico:** `num_labels` sbagliato.

```python
# Il modello ha 2 label ma il dataset ne ha 3
trainer.model.config.num_labels  # → 2 ← SBAGLIATO!

# → IndexError: Target 2 is out of bounds (perché 0,1,2 ma il modello si aspetta solo 0,1)

# ✅ BENE: specifica il numero corretto di label
model = AutoModelForSequenceClassification.from_pretrained(
    model_checkpoint,
    num_labels=3  # ← corretto
)
```

Una volta verificato il CPU, sposta sulla GPU:

```python
import torch

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
batch = {k: v.to(device) for k, v in batch.items()}
outputs = trainer.model.to(device)(**batch)
```

---

### Step 4: Backward Pass e Ottimizzatore

```python
loss = outputs.loss
loss.backward()  # calcola i gradienti

trainer.create_optimizer()
trainer.optimizer.step()  # aggiorna i pesi
```

Se esplode qui → problema nell'ottimizzatore custom o nei gradienti.

---

### Step 5: Valutazione (compute_metrics)

```python
# Testa la valutazione separatamente PRIMA di lanciare trainer.train()
trainer.evaluate()
# Se esplode → problema in compute_metrics
```

**Errore classico:** passare logit grezzi invece delle predizioni a `metric.compute()`.

```python
# ❌ MALE: predictions è ancora una matrice di logit [N, num_labels]
def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    return metric.compute(predictions=predictions, references=labels)
# → TypeError: only size-1 arrays can be converted to Python scalars

# ✅ BENE: converti i logit in classi con argmax
import numpy as np

def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)  # ← questa riga mancava!
    return metric.compute(predictions=predictions, references=labels)
```

---

### Script corretto finale

```python
import numpy as np
from datasets import load_dataset
import evaluate
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    DataCollatorWithPadding,
    TrainingArguments,
    Trainer,
)

raw_datasets = load_dataset("glue", "mnli")
model_checkpoint = "distilbert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)

def preprocess_function(examples):
    return tokenizer(examples["premise"], examples["hypothesis"], truncation=True)

tokenized_datasets = raw_datasets.map(preprocess_function, batched=True)

# ✅ num_labels=3 (3 classi nel dataset MNLI)
model = AutoModelForSequenceClassification.from_pretrained(model_checkpoint, num_labels=3)

args = TrainingArguments(
    "distilbert-finetuned-mnli",
    evaluation_strategy="epoch",
    save_strategy="epoch",
    learning_rate=2e-5,
    num_train_epochs=3,
    weight_decay=0.01,
)

metric = evaluate.load("glue", "mnli")

def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)  # ✅ argmax sui logit
    return metric.compute(predictions=predictions, references=labels)

data_collator = DataCollatorWithPadding(tokenizer=tokenizer)  # ✅ collator esplicito

trainer = Trainer(
    model,
    args,
    train_dataset=tokenized_datasets["train"],          # ✅ dataset tokenizzato
    eval_dataset=tokenized_datasets["validation_matched"],
    compute_metrics=compute_metrics,
    data_collator=data_collator,                        # ✅ collator passato esplicitamente
    tokenizer=tokenizer,                                # ✅ tokenizer passato
)
trainer.train()
```

---

## 4. CUDA Errors — Come Gestirli

### Out of Memory (OOM)

```
RuntimeError: CUDA out of memory. Tried to allocate 20.00 GiB ...
```

**Soluzioni in ordine:**
1. Riduci `per_device_train_batch_size`
2. Usa `gradient_accumulation_steps` per compensare
3. Abilita `fp16=True` in TrainingArguments
4. Usa un modello più piccolo
5. Usa tecniche avanzate: LoRA, quantizzazione (capitoli successivi)

### Altri CUDA Errors

```
RuntimeError: CUDA error: CUBLAS_STATUS_ALLOC_FAILED
```

Per tutti gli altri errori CUDA: **torna sempre alla CPU** per ottenere un messaggio leggibile:

```python
# Invece di lasciare che il modello fallisca su GPU
outputs = trainer.model.cpu()(**batch)
# L'errore reale apparirà qui in modo leggibile
```

---

## 5. Debugging Silenzioso — Il Modello Non Impara

Il training finisce senza errori ma i risultati sono pessimi. Strategia:

### 5.1 Ricontrolla i dati

- I testi decodificati hanno senso?
- Le label sono corrette?
- C'è una classe molto più frequente delle altre?
- Qual è la loss/accuracy di un modello random? (baseline)

### 5.2 Overfit su un singolo batch

Il test più rapido per verificare che il modello **possa** imparare:

```python
# Prendi un singolo batch
for batch in trainer.get_train_dataloader():
    break

batch = {k: v.to(device) for k, v in batch.items()}
trainer.create_optimizer()

# Allena sullo stesso batch per 20 step
for _ in range(20):
    outputs = trainer.model(**batch)
    loss = outputs.loss
    loss.backward()
    trainer.optimizer.step()
    trainer.optimizer.zero_grad()

# Valuta sullo stesso batch
with torch.no_grad():
    outputs = trainer.model(**batch)
preds = outputs.logits
labels = batch["labels"]
compute_metrics((preds.cpu().numpy(), labels.cpu().numpy()))
# → {'accuracy': 1.0} ← se il modello può imparare, deve arrivare a 100%
```

**Se non raggiunge 100%**: problema nel modello o nel framing del task.
**Se raggiunge 100%**: il modello può imparare → il problema è altrove (LR troppo bassa, dataset sbilanciato, ecc.).

> ⚠️ Dopo questo test **ricreare modello e Trainer** — il modello è sovrafit su quel batch e non serve più.

### 5.3 Non fare tuning prima di avere una baseline

Non lanciare migliaia di esperimenti con iperparametri diversi finché non hai un modello funzionante. I default del `Trainer` spesso bastano. Prima ottieni risultati decenti, poi ottimizza.

---

## 6. Chiedere Aiuto sul Forum

### Forum HuggingFace: https://discuss.huggingface.co

**Categorie:**
- **Beginners**: domande base sulle librerie
- **Intermediate / Research**: domande avanzate
- **Course**: domande specifiche sul corso

### Come scrivere un buon post

**❌ Post scarso:**
```
Titolo: "Ho un errore"
Corpo: "Non funziona, qualcuno mi aiuta? @sgugger @lysandre"
```

**✅ Post efficace:**

**1. Titolo descrittivo** — spiega il problema in una riga:
```
"Source of IndexError in the AutoModel forward pass?"
"Why can't I import GPT3ForSequenceClassification?"
```

**2. Corpo con:**
- Cosa stai cercando di fare
- Codice formattato con ` ```python ``` `
- Traceback **completo** (non solo l'ultima riga!)
- Cosa hai già provato

**3. Esempio riproducibile** — codice minimale che riproduce l'errore:

```markdown
## Ambiente
- transformers: 4.12.0
- Python: 3.8
- PyTorch: 1.9.0

## Codice
```python
from transformers import AutoTokenizer, AutoModel
tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
model = AutoModel.from_pretrained("distilbert-base-uncased")
text = "testo lungo..."
inputs = tokenizer(text, return_tensors="pt")
logits = model(**inputs).logits  # → IndexError: index out of range
```

## Traceback completo
[incolla qui l'intero traceback]

## Comportamento atteso
Mi aspettavo di ottenere gli embedding del testo.
```

---

## 7. Segnalare un Bug su GitHub

### Quando segnalare
Solo quando sei sicuro che il bug sia nella libreria, non nel tuo codice. Se non sei sicuro → prima il forum.

### Passi per segnalare

**1. Crea un minimal reproducible example:**
- Codice autonomo (niente file esterni)
- Usa dati dummy, non i tuoi dati reali
- Il più breve possibile
- Riproducibile da chiunque

**2. Raccogli le info sull'ambiente:**

```bash
transformers-cli env
# Output:
# - transformers version: 4.12.0
# - Platform: Linux-5.10.61
# - Python version: 3.8
# - PyTorch version (GPU?): 1.9.0 (True)
# - Using GPU: True
# ...
```

**3. Compila il template GitHub:**
- Ambiente (output di `transformers-cli env`)
- Descrizione del bug
- Esempio riproducibile (formattato con ` ```python ``` `)
- Comportamento atteso vs attuale
- Traceback completo

**4. Tag persone con moderazione:**
- Max 3 persone
- Solo chi è direttamente coinvolto (cerca chi ha modificato il file rilevante con `git blame`)
- Mai taggare in modo aggressivo

**5. Sii paziente e cortese:**
- È software open source, gratuito
- Se non ricevi risposta in una settimana, lascia un gentile follow-up con più info

---

## ⚠️ Errori comuni (recap)

| Errore | Causa | Soluzione |
|--------|-------|-----------|
| `OSError: Can't load config` | Model ID sbagliato o `config.json` mancante | Controlla typo sull'Hub, pusha il config |
| `AttributeError: 'list' object has no attribute 'size'` | Manca `return_tensors="pt"` | Aggiungi `return_tensors="pt"` al tokenizer |
| `ValueError: You have to specify either input_ids or inputs_embeds` | Dataset non tokenizzato passato al Trainer | Usa `tokenized_datasets`, non `raw_datasets` |
| `ValueError: expected sequence of length X at dim 1 (got Y)` | Data collator sbagliato | Usa `DataCollatorWithPadding` e passa `tokenizer` al Trainer |
| `RuntimeError: CUDA error` | Problema nella forward pass su GPU | Sposta su CPU: `trainer.model.cpu()(**batch)` |
| `IndexError: Target X is out of bounds` | `num_labels` sbagliato nel modello | Aggiungi `num_labels=N` al `from_pretrained()` |
| `TypeError: only size-1 arrays can be converted` | `compute_metrics` riceve logit non convertiti | Aggiungi `np.argmax(predictions, axis=1)` |
| Modello non impara | Decine di cause possibili | Overfit su 1 batch, controlla dati e label |

---

## 📚 Risorse utili per il debugging

- [Checklist for debugging neural networks](https://towardsdatascience.com/checklist-for-debugging-neural-networks-d8b2a9434f21) — Cecelia Shao
- [A Recipe for Training Neural Networks](http://karpathy.github.io/2019/04/25/recipe/) — Andrej Karpathy
- [How to unit test machine learning code](https://medium.com/@keeper6928/how-to-unit-test-machine-learning-code-57cf6fd81765) — Chase Roberts
- [Forum HuggingFace](https://discuss.huggingface.co/) — per domande sulla community
- [Issues Transformers GitHub](https://github.com/huggingface/transformers/issues) — per bug confermati
