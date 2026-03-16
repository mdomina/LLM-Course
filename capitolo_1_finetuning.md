# Capitolo 1 — Fine-tuning di un modello pre-addestrato

> **Obiettivo del capitolo**: imparare a prendere un modello pre-addestrato (es. BERT) e addestrarlo sul proprio dataset per un task specifico (es. classificazione del testo).

---

## Indice

1. [Preparare i dati](#1-preparare-i-dati)
2. [Fine-tuning con la Trainer API](#2-fine-tuning-con-la-trainer-api)
3. [Training loop manuale](#3-training-loop-manuale)
4. [Distributed training con Accelerate](#4-distributed-training-con-accelerate)
5. [Leggere le learning curves](#5-leggere-le-learning-curves)
6. [TL;DR — Riepilogo finale](#6-tldr--riepilogo-finale)

---

## 1. Preparare i dati

### Concetto

Prima di fare fine-tuning, i dati testuali devono diventare **tensori numerici** comprensibili dal modello. Il flusso è:

```
Testo grezzo → Tokenizer → input_ids / attention_mask / token_type_ids → Modello
```

### Dataset usato: MRPC

Il corso usa il dataset **MRPC** (Microsoft Research Paraphrase Corpus), parte del benchmark GLUE. Contiene **5.801 coppie di frasi** con un'etichetta che dice se le due frasi sono parafrasi l'una dell'altra.

| Split      | Righe |
|------------|-------|
| Train      | 3.668 |
| Validation | 408   |
| Test       | 1.725 |

### Caricare il dataset

```python
from datasets import load_dataset

raw_datasets = load_dataset("glue", "mrpc")
```

- `load_dataset` scarica e mette in **cache** il dataset in `~/.cache/huggingface/datasets`
- Restituisce un `DatasetDict` con i 3 split (train, validation, test)
- Ogni riga ha: `sentence1`, `sentence2`, `label` (0=non parafrasi, 1=parafrasi), `idx`

```python
# Accedere a un esempio
raw_train_dataset = raw_datasets["train"]
raw_train_dataset[0]
# {'idx': 0, 'label': 1, 'sentence1': '...', 'sentence2': '...'}

# Vedere i tipi di colonne
raw_train_dataset.features
# label è ClassLabel: 0 = 'not_equivalent', 1 = 'equivalent'
```

### Tokenizzare una coppia di frasi

BERT si aspetta l'input nel formato: `[CLS] frase1 [SEP] frase2 [SEP]`

```python
from transformers import AutoTokenizer

checkpoint = "bert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(checkpoint)

inputs = tokenizer("Questa è la prima frase.", "Questa è la seconda.")
```

L'output contiene tre chiavi:

- **`input_ids`** — gli ID numerici dei token
- **`attention_mask`** — 1 dove c'è un token reale, 0 dove c'è padding
- **`token_type_ids`** — 0 per i token della frase 1, 1 per quelli della frase 2

> ⚠️ `token_type_ids` esiste solo per modelli addestrati con il task "Next Sentence Prediction" (come BERT). DistilBERT non lo usa.

### Tokenizzare l'intero dataset con `.map()`

Invece di tokenizzare una frase alla volta, si usa `.map()` per applicare una funzione a tutto il dataset in modo efficiente.

```python
def tokenize_function(example):
    # example può contenere una singola riga o un batch di righe
    return tokenizer(example["sentence1"], example["sentence2"], truncation=True)
    # truncation=True: tronca se la sequenza supera la lunghezza massima del modello
    # NON aggiungiamo padding qui — lo faremo al momento del batching (più efficiente)

tokenized_datasets = raw_datasets.map(tokenize_function, batched=True)
# batched=True: passa più esempi alla volta alla funzione → molto più veloce
# grazie al tokenizer Rust di HuggingFace
```

### Dynamic Padding (padding dinamico)

Fare padding di tutte le sequenze alla lunghezza massima del dataset è **sprecone**. È meglio fare padding **per batch**, alla lunghezza della sequenza più lunga in quel batch specifico.

```python
from transformers import DataCollatorWithPadding

data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
# Sa quale token usare per il padding e da che lato aggiungerlo
```

**Dimostrazione pratica:**

```python
# Prendiamo 8 esempi dal training set
samples = tokenized_datasets["train"][:8]
# Rimuoviamo colonne non necessarie (stringhe non convertibili in tensori)
samples = {k: v for k, v in samples.items() if k not in ["idx", "sentence1", "sentence2"]}

# Le lunghezze variano: [50, 59, 47, 67, 59, 50, 62, 32]
[len(x) for x in samples["input_ids"]]

# Il collator porta tutto a 67 (il massimo del batch)
batch = data_collator(samples)
{k: v.shape for k, v in batch.items()}
# {'attention_mask': torch.Size([8, 67]),
#  'input_ids': torch.Size([8, 67]),
#  'token_type_ids': torch.Size([8, 67]),
#  'labels': torch.Size([8])}
```

### Quando usarlo

- Ogni volta che si vuole fare fine-tuning su un dataset testuale
- `.map(batched=True)` è sempre preferibile al loop manuale per performance
- Il padding dinamico è ideale quando le sequenze hanno lunghezze molto variabili

### Errori comuni

- ❌ Fare padding al momento della tokenizzazione (`padding=True` dentro `.map()`) → spreca memoria, rallenta il training
- ❌ Dimenticare `truncation=True` → crash se una sequenza supera i 512 token di BERT
- ❌ Non rimuovere le colonne stringa prima di creare i tensori → PyTorch non sa come gestirle

---

## 2. Fine-tuning con la Trainer API

### Concetto

La classe `Trainer` di HuggingFace gestisce automaticamente l'intero loop di training: forward pass, backpropagation, aggiornamento dei pesi, valutazione. È il modo più rapido per fare fine-tuning.

### Setup completo — ricapitolo della sezione precedente

```python
from datasets import load_dataset
from transformers import AutoTokenizer, DataCollatorWithPadding

raw_datasets = load_dataset("glue", "mrpc")
checkpoint = "bert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(checkpoint)

def tokenize_function(example):
    return tokenizer(example["sentence1"], example["sentence2"], truncation=True)

tokenized_datasets = raw_datasets.map(tokenize_function, batched=True)
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
```

### Step 1 — TrainingArguments

```python
from transformers import TrainingArguments

training_args = TrainingArguments("test-trainer")
# "test-trainer" è la directory dove salvare il modello e i checkpoint
# Tutti gli altri parametri hanno valori di default ragionevoli
```

**Parametri utili:**

```python
training_args = TrainingArguments(
    "test-trainer",
    eval_strategy="epoch",          # Valuta alla fine di ogni epoch (default: "no")
    learning_rate=2e-5,             # Learning rate (default: 5e-5)
    per_device_train_batch_size=16, # Batch size per GPU in training
    per_device_eval_batch_size=16,  # Batch size per GPU in valutazione
    num_train_epochs=3,             # Numero di epoch (default: 3)
    fp16=True,                      # Mixed precision → più veloce su GPU moderne
    push_to_hub=True,               # Carica il modello sull'HF Hub dopo il training
)
```

### Step 2 — Caricare il modello

```python
from transformers import AutoModelForSequenceClassification

model = AutoModelForSequenceClassification.from_pretrained(checkpoint, num_labels=2)
# num_labels=2: classificazione binaria (parafrasi / non parafrasi)
# ATTENZIONE: vedrai un warning — è normale!
# BERT è stato pre-addestrato per masked language modeling, non classificazione.
# La "testa" originale viene scartata e sostituita con una nuova (inizializzata casualmente).
# Ecco perché dobbiamo fare fine-tuning: addestrare questa nuova testa.
```

### Step 3 — Creare il Trainer

```python
from transformers import Trainer

trainer = Trainer(
    model,                                        # Il modello da addestrare
    training_args,                                # Configurazione del training
    train_dataset=tokenized_datasets["train"],    # Dataset di training
    eval_dataset=tokenized_datasets["validation"],# Dataset di validazione
    data_collator=data_collator,                  # Per il padding dinamico
    processing_class=tokenizer,                   # Tokenizer (nuovo parametro, sostituisce tokenizer=)
)
```

### Step 4 — Avviare il training

```python
trainer.train()
# Stampa il training loss ogni 500 step
# NON stampa metriche di valutazione — bisogna aggiungerle manualmente (vedi sotto)
```

### Aggiungere metriche di valutazione

Di default il Trainer non calcola accuracy o F1. Bisogna definire una funzione `compute_metrics`.

```python
import numpy as np
import evaluate

def compute_metrics(eval_preds):
    metric = evaluate.load("glue", "mrpc")  # Carica accuracy + F1 per MRPC
    logits, labels = eval_preds
    # I modelli restituiscono logits (punteggi grezzi), non probabilità
    # argmax prende l'indice con il valore più alto → la classe predetta
    predictions = np.argmax(logits, axis=-1)
    return metric.compute(predictions=predictions, references=labels)
    # Output: {'accuracy': 0.857, 'f1': 0.899}
```

### Trainer completo con valutazione

```python
training_args = TrainingArguments("test-trainer", eval_strategy="epoch")
model = AutoModelForSequenceClassification.from_pretrained(checkpoint, num_labels=2)

trainer = Trainer(
    model,
    training_args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["validation"],
    data_collator=data_collator,
    processing_class=tokenizer,
    compute_metrics=compute_metrics,  # ← aggiunta
)

trainer.train()
# Ora stampa accuracy e F1 alla fine di ogni epoch
```

### Tecniche avanzate

> 💡 **Glossario rapido — leggi prima di continuare**
>
> - **Learning rate** (tasso di apprendimento): quanto "in fretta" il modello modifica i suoi parametri ad ogni passo di training. Pensa di dover trovare il punto più basso di una valle camminando bendato: se fai passi troppo grandi rischi di saltare oltre la valle (il modello non converge), se li fai troppo piccoli ci vuole un'eternità (training lentissimo). Il valore tipico è `2e-5` cioè `0.00002` — molto piccolo di proposito, per aggiustare i pesi con delicatezza.
>
> - **Epoch**: una passata completa su tutto il dataset di training. Se hai 3.000 frasi e alleni per 3 epoch, il modello "vede" ogni frase 3 volte in totale.
>
> - **Batch / Batch size**: invece di aggiornare il modello dopo ogni singola frase, si raggruppano N frasi insieme (un "batch") e si aggiorna una volta sola. Con `batch_size=16` il modello vede 16 frasi, calcola l'errore medio, e aggiorna i pesi. Batch più grandi danno aggiornamenti più stabili ma richiedono più memoria.
>
> - **GPU e VRAM**: la GPU (scheda grafica) è molto più veloce della CPU per i calcoli dei modelli — il training passa da ore a minuti. La VRAM è la memoria della GPU: i modelli grandi possono non entrarci, ed è il problema che le tecniche sotto risolvono.
>
> - **Loss** (perdita): un numero che misura quanto il modello sta sbagliando. Più è bassa, meglio il modello sta imparando. L'obiettivo del training è minimizzarla.

---

**1. Mixed Precision (`fp16`) — usa meno memoria senza perdere qualità**

Normalmente ogni peso del modello è salvato come numero a 32 bit. Usando 16 bit (metà dello spazio) si occupa **la metà della VRAM** e il training accelera, senza differenze apprezzabili nei risultati. È la prima ottimizzazione da provare se hai una GPU moderna.

```python
training_args = TrainingArguments(
    "test-trainer",
    eval_strategy="epoch",
    fp16=True,  # Attiva la precisione a 16 bit — quasi gratuito in termini di qualità
)
```

> ⚠️ Funziona solo su GPU NVIDIA. Su CPU o Mac con chip Apple, ignoralo.

---

**2. Gradient Accumulation — simula un batch grande anche con poca VRAM**

**Problema concreto**: vorresti processare 16 frasi alla volta per avere aggiornamenti stabili, ma la tua GPU ne regge solo 4.

**Soluzione**: processa 4 frasi per 4 volte di fila *senza* aggiornare i pesi, accumula gli errori calcolati (i "gradienti"), poi aggiorna i pesi una volta sola alla fine. Il risultato matematico è identico a usare un batch di 16, ma senza richiedere più memoria.

```python
training_args = TrainingArguments(
    "test-trainer",
    per_device_train_batch_size=4,   # Quante frasi processa la GPU per volta
    gradient_accumulation_steps=4,   # Quante volte accumula prima di aggiornare i pesi
    # Effetto finale = batch size di 4 × 4 = 16 frasi per aggiornamento
)
```

---

**3. Learning Rate Scheduler — il learning rate cambia nel tempo**

Usare sempre lo stesso learning rate non è ottimale: conviene iniziare con passi un po' più grandi (imparare in fretta) e finire con passi più piccoli (rifinire con precisione). Lo "scheduler" gestisce questa variazione in automatico.

Due modalità comuni:
- **Lineare** (default): il learning rate scende in modo uniforme da `2e-5` fino a `0` alla fine del training.
- **Coseno**: scende seguendo una curva più dolce, con una discesa graduale all'inizio e alla fine.

```python
training_args = TrainingArguments(
    "test-trainer",
    learning_rate=2e-5,           # Valore di partenza (consigliato per BERT)
    lr_scheduler_type="cosine",   # Prova "cosine" in alternativa al default "linear"
)
```

> 💡 Se non sai quale scegliere, lascia il default lineare. La differenza di risultato è spesso minima.

---

**4. Early Stopping — fermati automaticamente quando smetti di migliorare**

Se alleni il modello troppo a lungo, inizia a **memorizzare** il training set invece di imparare a generalizzare (overfitting). L'early stopping monitora la performance sul validation set e ferma il training da solo non appena non migliora per N valutazioni consecutive.

**Analogia**: immagina di studiare per un esame. All'inizio migliori velocemente. Poi arriva un punto in cui studiare ancora non aiuta — anzi ti confonde. L'early stopping riconosce quel momento e dice "basta così".

```python
from transformers import EarlyStoppingCallback

training_args = TrainingArguments(
    output_dir="./results",
    eval_strategy="steps",        # Valuta ogni N step, non solo a fine epoch
    eval_steps=100,               # Valuta ogni 100 step
    save_strategy="steps",        # Salva un checkpoint ogni 100 step
    save_steps=100,
    load_best_model_at_end=True,  # Alla fine usa il checkpoint migliore, non l'ultimo
    metric_for_best_model="eval_loss",  # "migliore" = validation loss più bassa
    greater_is_better=False,      # Per la loss: più bassa è meglio → False
    num_train_epochs=10,          # Metti un numero alto: sarà l'early stopping a fermare prima
)

trainer = Trainer(
    ...
    callbacks=[EarlyStoppingCallback(early_stopping_patience=3)],
    # patience=3 → se dopo 3 valutazioni consecutive la loss non migliora, fermati
)
```

### Quando usarlo

- Quando vuoi addestrare rapidamente senza scrivere un training loop manuale
- Quando vuoi usare feature avanzate (fp16, gradient accumulation, early stopping) con poco codice
- Come punto di partenza per qualsiasi task NLP (classificazione, NER, Q&A, ecc.)

### Errori comuni

- ❌ Dimenticare `eval_strategy="epoch"` → il Trainer non valida mai durante il training
- ❌ Non passare `compute_metrics` → vedi solo la loss, non accuracy/F1
- ❌ Riusare il vecchio modello dopo aver cambiato `TrainingArguments` → il training riparte da dove si era fermato, non da zero. Ricrea sempre il modello prima di un nuovo `trainer.train()`

---

## 3. Training loop manuale

### Concetto

Invece di usare `Trainer`, si può implementare il loop di training a mano con PyTorch puro. Utile quando serve controllo totale su ogni step (loss custom, logging personalizzato, ecc.).

### Preparare i dati per PyTorch

Prima del Trainer, le colonne extra venivano rimosse automaticamente. Qui dobbiamo farlo noi:

```python
# 1. Rimuovere le colonne non necessarie (stringhe non convertibili in tensori)
tokenized_datasets = tokenized_datasets.remove_columns(["sentence1", "sentence2", "idx"])

# 2. Rinominare 'label' in 'labels' (il modello si aspetta questo nome)
tokenized_datasets = tokenized_datasets.rename_column("label", "labels")

# 3. Dire al dataset di restituire tensori PyTorch (non liste Python)
tokenized_datasets.set_format("torch")

# Verifica: restano solo le colonne che il modello capisce
tokenized_datasets["train"].column_names
# ['attention_mask', 'input_ids', 'labels', 'token_type_ids']
```

### Creare i DataLoader

```python
from torch.utils.data import DataLoader

train_dataloader = DataLoader(
    tokenized_datasets["train"],
    shuffle=True,          # Mescola i dati ad ogni epoch (importante per il training)
    batch_size=8,
    collate_fn=data_collator  # Applica il dynamic padding ad ogni batch
)

eval_dataloader = DataLoader(
    tokenized_datasets["validation"],
    batch_size=8,
    collate_fn=data_collator  # Niente shuffle per la validazione
)
```

### Caricare il modello e configurare l'ottimizzatore

```python
from transformers import AutoModelForSequenceClassification
from torch.optim import AdamW

model = AutoModelForSequenceClassification.from_pretrained(checkpoint, num_labels=2)

optimizer = AdamW(model.parameters(), lr=5e-5)
# AdamW = Adam + Weight Decay
# Weight decay: penalizza i pesi molto grandi → aiuta a evitare overfitting
# È l'ottimizzatore standard per i modelli Transformer
```

### Learning rate scheduler

```python
from transformers import get_scheduler

num_epochs = 3
num_training_steps = num_epochs * len(train_dataloader)
# len(train_dataloader) = numero di batch in un'epoch

lr_scheduler = get_scheduler(
    "linear",               # Decay lineare del learning rate
    optimizer=optimizer,
    num_warmup_steps=0,     # Niente warmup (si potrebbe aumentare per più stabilità)
    num_training_steps=num_training_steps,  # 1377 step totali (3 epoch * ~459 batch)
)
```

### Spostare il modello su GPU

```python
import torch

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
model.to(device)
# Su CPU il training può durare ore. Su GPU pochi minuti.
```

### Il training loop completo

```python
from tqdm.auto import tqdm

progress_bar = tqdm(range(num_training_steps))  # Barra di progresso

model.train()  # Mette il modello in modalità training (attiva dropout, BatchNorm, ecc.)

for epoch in range(num_epochs):
    for batch in train_dataloader:
        # 1. Spostare il batch sulla GPU
        batch = {k: v.to(device) for k, v in batch.items()}

        # 2. Forward pass: calcola le predizioni e la loss
        outputs = model(**batch)  # batch contiene input_ids, attention_mask, labels
        loss = outputs.loss       # La loss viene calcolata automaticamente se passi labels

        # 3. Backward pass: calcola i gradienti
        loss.backward()

        # 4. Aggiornare i pesi del modello
        optimizer.step()

        # 5. Aggiornare il learning rate
        lr_scheduler.step()

        # 6. Azzerare i gradienti (IMPORTANTE: senza questo, si accumulano!)
        optimizer.zero_grad()

        progress_bar.update(1)
```

> ⚠️ **L'ordine conta**: forward → backward → `optimizer.step()` → `lr_scheduler.step()` → `optimizer.zero_grad()`

### Il loop di valutazione

```python
import evaluate

metric = evaluate.load("glue", "mrpc")
model.eval()  # Disattiva dropout e altre layer specifiche del training

for batch in eval_dataloader:
    batch = {k: v.to(device) for k, v in batch.items()}

    with torch.no_grad():  # Non calcolare i gradienti → risparmia memoria e tempo
        outputs = model(**batch)

    logits = outputs.logits
    predictions = torch.argmax(logits, dim=-1)  # Classe con punteggio più alto

    # Accumula le predizioni batch per batch
    metric.add_batch(predictions=predictions, references=batch["labels"])

# Calcola la metrica finale su tutto il validation set
metric.compute()
# {'accuracy': 0.843, 'f1': 0.890}
```

### Quando usarlo

- Quando hai bisogno di una loss function custom
- Quando vuoi logging o checkpoint con logiche non standard
- Per capire esattamente cosa succede "sotto il cofano" del Trainer
- Per debugging avanzato di problemi di training

### Errori comuni

- ❌ Dimenticare `optimizer.zero_grad()` → i gradienti si accumulano da un batch all'altro
- ❌ Non chiamare `model.eval()` durante la validazione → dropout rimane attivo → risultati non deterministici
- ❌ Dimenticare `torch.no_grad()` in valutazione → spreco di memoria (PyTorch tiene il grafo per backprop)
- ❌ Non spostare i batch sul device → crash con errore "tensors on different devices"

---

## 4. Distributed training con Accelerate

### Concetto

Il loop manuale funziona su una singola GPU. Con **🤗 Accelerate** è possibile farlo girare su più GPU, TPU o qualsiasi hardware con pochissime modifiche al codice.

### Differenze rispetto al loop standard

| Standard                        | Con Accelerate                          |
|---------------------------------|-----------------------------------------|
| `model.to(device)`              | `accelerator.prepare(...)` gestisce tutto |
| `batch = {k: v.to(device) ...}` | Non serve più (Accelerate lo fa)        |
| `loss.backward()`               | `accelerator.backward(loss)`            |

### Codice completo con Accelerate

```python
from accelerate import Accelerator
from torch.optim import AdamW
from transformers import AutoModelForSequenceClassification, get_scheduler
from tqdm.auto import tqdm

# 1. Creare l'Accelerator — rileva automaticamente l'hardware disponibile
accelerator = Accelerator()

# 2. Caricare modello e ottimizzatore (NON serve .to(device) manualmente)
model = AutoModelForSequenceClassification.from_pretrained(checkpoint, num_labels=2)
optimizer = AdamW(model.parameters(), lr=3e-5)

# 3. "Wrappare" tutto con accelerator.prepare() — questo è il cuore di Accelerate
#    Prepara i dataloader, il modello e l'ottimizzatore per il training distribuito
train_dl, eval_dl, model, optimizer = accelerator.prepare(
    train_dataloader, eval_dataloader, model, optimizer
)

# 4. Scheduler e progress bar (uguali a prima)
num_epochs = 3
num_training_steps = num_epochs * len(train_dl)
lr_scheduler = get_scheduler("linear", optimizer=optimizer,
                              num_warmup_steps=0,
                              num_training_steps=num_training_steps)
progress_bar = tqdm(range(num_training_steps))

# 5. Training loop (quasi identico, due differenze chiave)
model.train()
for epoch in range(num_epochs):
    for batch in train_dl:
        # NON serve batch = {k: v.to(device) ...} → Accelerate lo gestisce
        outputs = model(**batch)
        loss = outputs.loss
        accelerator.backward(loss)  # ← invece di loss.backward()

        optimizer.step()
        lr_scheduler.step()
        optimizer.zero_grad()
        progress_bar.update(1)
```

### Eseguire su più GPU

Salva il codice in `train.py`, poi:

```bash
# Configura l'ambiente (ti chiede: quante GPU? TPU? mixed precision?)
accelerate config

# Lancia il training distribuito
accelerate launch train.py
```

### Usare Accelerate in un Notebook (es. Google Colab)

```python
from accelerate import notebook_launcher

def training_function():
    # ... tutto il codice di training qui dentro ...
    pass

notebook_launcher(training_function)
```

> ⚠️ Se usi TPU, aggiungi `padding="max_length"` e `max_length=512` al tokenizer. Le TPU preferiscono tensori di forma fissa.

### Quando usarlo

- Quando hai più GPU disponibili (locale o cloud)
- Quando vuoi il controllo del loop manuale ma con scaling automatico
- Per passare da sviluppo (1 GPU) a produzione (multi-GPU) senza riscrivere il codice

### Errori comuni

- ❌ Non passare tutti gli oggetti a `accelerator.prepare()` → training non distribuito correttamente
- ❌ Usare `loss.backward()` invece di `accelerator.backward(loss)` → problemi con mixed precision
- ❌ Non usare `padding="max_length"` con le TPU → crash per tensori di forma variabile

---

## 5. Leggere le learning curves

### Concetto

Le **learning curves** sono grafici che mostrano come loss e accuracy cambiano durante il training. Sono fondamentali per diagnosticare problemi.

### Come monitorarle — Weights & Biases (W&B)

```python
import wandb

wandb.init(project="llm-finetuning", name="bert-mrpc")

training_args = TrainingArguments(
    output_dir="./results",
    eval_strategy="steps",
    eval_steps=50,
    logging_steps=10,       # Logga ogni 10 step
    report_to="wandb",      # Invia i log a W&B
    num_train_epochs=3,
    per_device_train_batch_size=16,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["validation"],
    data_collator=data_collator,
    processing_class=tokenizer,
    compute_metrics=compute_metrics,
)

trainer.train()
```

### Pattern 1 — Training sano ✅

- Loss in training **scende progressivamente**
- Loss in validation **segue quella di training**, con un piccolo gap
- Accuracy sale, con dei "gradini" (normale — vedi sotto)

> **Perché l'accuracy ha i "gradini"?**
> La loss migliora anche quando la predizione si avvicina al target senza ancora superare la soglia (es. da 0.3 a 0.4 per una classe positiva). L'accuracy invece cambia solo quando la predizione supera 0.5 e "cambia classe". Quindi la loss è liscia, l'accuracy è a scalini.

### Pattern 2 — Overfitting ❌

**Sintomi:**
- La training loss continua a scendere
- La validation loss risale (o si ferma)
- Grande gap tra accuracy di training e di validation

**Soluzioni:**

```python
# 1. Early stopping
from transformers import EarlyStoppingCallback

trainer = Trainer(
    ...
    callbacks=[EarlyStoppingCallback(early_stopping_patience=3)],
)

# 2. Aumentare il weight decay
training_args = TrainingArguments(..., weight_decay=0.01)

# 3. Aggiungere dropout al modello (dipende dall'architettura)
```

### Pattern 3 — Underfitting ❌

**Sintomi:**
- Sia training loss che validation loss rimangono alte
- L'accuracy si ferma a valori bassi già nelle prime epoch

**Soluzioni:**

```python
# 1. Aumentare le epoch
training_args = TrainingArguments(..., num_train_epochs=10)

# 2. Alzare il learning rate
training_args = TrainingArguments(..., learning_rate=5e-5)

# 3. Usare un modello più grande (es. bert-large invece di bert-base)
```

### Pattern 4 — Curve erratiche ❌

**Sintomi:**
- Loss e accuracy oscillano molto senza un trend chiaro
- Le curve sembrano "rumorose"

**Soluzioni:**

```python
# 1. Abbassare il learning rate
training_args = TrainingArguments(..., learning_rate=1e-5)

# 2. Aumentare il batch size (gradienti più stabili)
training_args = TrainingArguments(..., per_device_train_batch_size=32)

# 3. Aggiungere gradient clipping (nel loop manuale)
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
# Metti questa riga PRIMA di optimizer.step()
```

### Quando usarlo

- Sempre, durante e dopo il training
- Confronta le curve di training e validation per diagnosi rapida
- Usa W&B o TensorBoard per visualizzazioni interattive

### Errori comuni

- ❌ Guardare solo la training loss → non vedi overfitting
- ❌ Interpretare i "gradini" dell'accuracy come anomalie → è comportamento normale
- ❌ Aspettare la fine del training per accorgersi dell'overfitting → monitora in real-time

---

## 6. TL;DR — Riepilogo finale

**Flusso completo per fare fine-tuning:**

1. **Carica il dataset** con `load_dataset("glue", "mrpc")`
2. **Tokenizza** con `.map(tokenize_function, batched=True)` — senza padding
3. **Configura il padding dinamico** con `DataCollatorWithPadding`
4. **Carica il modello** con `AutoModelForSequenceClassification.from_pretrained(..., num_labels=N)`
5. **Scegli il metodo di training:**
   - `Trainer` API → veloce, pochi parametri, ottimo punto di partenza
   - Loop manuale → controllo totale, più codice
   - Accelerate → loop manuale + multi-GPU/TPU automatico
6. **Monitora** le learning curves con W&B
7. **Diagnostica** overfitting/underfitting/curve erratiche e intervieni

**Valori di default buoni da ricordare:**

| Parametro          | Valore consigliato |
|--------------------|--------------------|
| Learning rate      | 2e-5 – 5e-5        |
| Epoch              | 3                  |
| Batch size         | 16–32              |
| Ottimizzatore      | AdamW              |
| LR scheduler       | linear             |
| Weight decay       | 0.01               |

**Risultati attesi su MRPC con BERT-base:**
- Accuracy: ~85–86%
- F1: ~89–90%
