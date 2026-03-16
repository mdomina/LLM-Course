# Capitolo 7 — Task NLP Principali con Transformers

> **Di cosa parla questo capitolo**: come applicare i modelli Transformer ai task NLP più comuni — NER, MLM, traduzione, riassunto, generazione di codice e question answering. Ogni sezione è indipendente: puoi leggere solo quella che ti interessa.

---

## 📌 TL;DR

```
NER (Token Classification) → AutoModelForTokenClassification + DataCollatorForTokenClassification + seqeval
MLM (Domain Adaptation)    → AutoModelForMaskedLM + DataCollatorForLanguageModeling(mlm=True)
Traduzione                 → AutoModelForSeq2SeqLM + Seq2SeqTrainer + SacreBLEU
Riassunto                  → AutoModelForSeq2SeqLM (mT5) + ROUGE
Generazione Codice (CLM)   → GPT2LMHeadModel da zero + DataCollatorForLanguageModeling(mlm=False)
Question Answering         → AutoModelForQuestionAnswering + offset mapping per span extraction

Architetture:
  Encoder-only  (BERT)     → classificazione, NER, QA estrattiva
  Decoder-only  (GPT-2)    → generazione testo, completamento codice
  Encoder-Decoder (T5, mT5)→ traduzione, riassunto, QA generativa
```

---

## 🧠 Glossario rapido

> 💡 **Leggi prima di continuare**
>
> - **Fine-tuning**: adattare un modello pre-addestrato a un task specifico aggiornando i pesi.
> - **Pretraining**: addestrare un modello da zero su grandi quantità di testo non etichettato.
> - **Token classification**: assegnare un'etichetta a ogni token (es. NER, POS tagging).
> - **Seq2Seq**: il modello prende una sequenza in input e genera una sequenza in output (traduzione, riassunto).
> - **Causal LM**: generazione di testo autoregressiva — ogni token è previsto dai precedenti.
> - **Masked LM**: predizione di token mascherati nel mezzo della frase (BERT-style).
> - **BLEU / SacreBLEU**: metriche per valutare la qualità della traduzione.
> - **ROUGE**: metrica per valutare la qualità dei riassunti (sovrapposizione di n-grammi).
> - **seqeval**: libreria per valutare il riconoscimento di entità (NER).

---

## 1. Token Classification — NER

### Cos'è
Assegnare un'etichetta a ogni token della frase. Task principali: NER (entità), POS (parti del discorso), chunking.

**Schema etichette IOB2:**
```
B-XXX = Beginning: inizio di un'entità di tipo XXX
I-XXX = Inside: dentro un'entità di tipo XXX
O     = Outside: non appartiene a nessuna entità
```

Esempio: `"Hugging Face"` → `[B-ORG, I-ORG]`

### Dataset: CoNLL-2003

```python
from datasets import load_dataset

raw_datasets = load_dataset("conll2003")
# → train: 14k frasi, validation: 3.2k, test: 3.5k
# Colonne: tokens, ner_tags, pos_tags, chunk_tags

# Nomi delle etichette NER
label_names = raw_datasets["train"].features["ner_tags"].feature.names
# → ['O', 'B-PER', 'I-PER', 'B-ORG', 'I-ORG', 'B-LOC', 'I-LOC', 'B-MISC', 'I-MISC']
```

### Problema: allineamento token ↔ etichette

Il tokenizer subword spezza le parole: `"lamb"` → `["la", "##mb"]`, ma nel dataset c'è un solo label per parola. Bisogna espandere i label!

```python
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")

# is_split_into_words=True: input già diviso in parole
inputs = tokenizer(raw_datasets["train"][0]["tokens"], is_split_into_words=True)
inputs.word_ids()
# → [None, 0, 1, 2, 3, 4, 5, 6, 7, 7, 8, None]
# None = token speciali ([CLS],[SEP])
# 7,7 = "lamb" è diventato 2 token ma appartengono entrambi alla parola 7

def align_labels_with_tokens(labels, word_ids):
    new_labels = []
    current_word = None
    for word_id in word_ids:
        if word_id is None:
            new_labels.append(-100)          # token speciali → ignora nella loss
        elif word_id != current_word:
            current_word = word_id
            new_labels.append(labels[word_id])  # primo subtoken → label originale
        else:
            label = labels[word_id]
            # B-XXX (dispari) → I-XXX (pari+1) per i subtoken interni
            if label % 2 == 1:
                label += 1
            new_labels.append(label)
    return new_labels
```

### Preprocessing completo

```python
def tokenize_and_align_labels(examples):
    tokenized_inputs = tokenizer(
        examples["tokens"],
        truncation=True,
        is_split_into_words=True   # IMPORTANTE: input già tokenizzato in parole
    )
    new_labels = []
    for i, labels in enumerate(examples["ner_tags"]):
        word_ids = tokenized_inputs.word_ids(i)  # i = indice nell'esempio batch
        new_labels.append(align_labels_with_tokens(labels, word_ids))

    tokenized_inputs["labels"] = new_labels
    return tokenized_inputs

tokenized_datasets = raw_datasets.map(
    tokenize_and_align_labels,
    batched=True,
    remove_columns=raw_datasets["train"].column_names,
)
```

### Data Collator per Token Classification

```python
from transformers import DataCollatorForTokenClassification

# Padda sia gli input che i label (con -100 per i label di padding)
data_collator = DataCollatorForTokenClassification(tokenizer=tokenizer)
```

### Metrica: seqeval

```python
import evaluate

metric = evaluate.load("seqeval")
# seqeval vuole stringhe, non interi!

import numpy as np

def compute_metrics(eval_preds):
    logits, labels = eval_preds
    predictions = np.argmax(logits, axis=-1)

    # Rimuovi i -100 e converti in stringhe
    true_labels = [
        [label_names[l] for l in label if l != -100]
        for label in labels
    ]
    true_predictions = [
        [label_names[p] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]
    results = metric.compute(predictions=true_predictions, references=true_labels)
    return {
        "precision": results["overall_precision"],
        "recall":    results["overall_recall"],
        "f1":        results["overall_f1"],
        "accuracy":  results["overall_accuracy"],
    }
```

### Training

```python
from transformers import AutoModelForTokenClassification, TrainingArguments, Trainer

id2label = {i: name for i, name in enumerate(label_names)}
label2id = {v: k for k, v in id2label.items()}

model = AutoModelForTokenClassification.from_pretrained(
    "bert-base-cased",
    id2label=id2label,   # per la widget di HuggingFace Hub
    label2id=label2id,
)
# Controlla che model.config.num_labels == 9 ← errore comune!

args = TrainingArguments(
    "bert-finetuned-ner",
    evaluation_strategy="epoch",
    save_strategy="epoch",
    learning_rate=2e-5,
    num_train_epochs=3,
    weight_decay=0.01,
    push_to_hub=True,
)

trainer = Trainer(
    model=model,
    args=args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["validation"],
    data_collator=data_collator,
    compute_metrics=compute_metrics,
    processing_class=tokenizer,
)
trainer.train()
trainer.push_to_hub(commit_message="Training complete")
```

### Usare il modello fine-tuned

```python
from transformers import pipeline

token_classifier = pipeline(
    "token-classification",
    model="huggingface-course/bert-finetuned-ner",
    aggregation_strategy="simple"  # raggruppa token della stessa entità
)
token_classifier("My name is Sylvain and I work at Hugging Face in Brooklyn.")
# → [{'entity_group': 'PER', 'word': 'Sylvain', 'start': 11, 'end': 18}, ...]
```

---

## 2. Masked Language Modeling — Domain Adaptation

### Cos'è
Addestrare ulteriormente un modello BERT sul proprio corpus specifico prima del fine-tuning su un task. Utile quando il dominio è molto diverso (legge, medicina, codice).

**Domain adaptation**: il modello impara il vocabolario e i pattern del nuovo dominio → migliorano i task downstream.

### Dataset: IMDb (esempio di dominio)

```python
from datasets import load_dataset

imdb_dataset = load_dataset("imdb")

# Tieni solo la colonna "text", unisci train e test
def preprocess_function(examples):
    return tokenizer(
        examples["text"],
        truncation=True,
        max_length=128,
        return_special_tokens_mask=True,  # necessario per il data collator MLM
    )

tokenized_dataset = imdb_dataset.map(
    preprocess_function, batched=True, remove_columns=["text", "label"]
)
```

### Data Collator per MLM

```python
from transformers import DataCollatorForLanguageModeling

# mlm=True: maschera casualmente il 15% dei token (BERT-style)
data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=True,
    mlm_probability=0.15  # default
)
```

### Training

```python
from transformers import AutoModelForMaskedLM, TrainingArguments, Trainer

model = AutoModelForMaskedLM.from_pretrained("distilbert-base-uncased")

args = TrainingArguments(
    "distilbert-finetuned-imdb",
    evaluation_strategy="epoch",
    learning_rate=2e-5,
    num_train_epochs=3,
    push_to_hub=True,
)

trainer = Trainer(
    model=model,
    args=args,
    train_dataset=tokenized_dataset["train"],
    eval_dataset=tokenized_dataset["test"],
    data_collator=data_collator,
)
trainer.train()
```

**Perplexity**: metrica per MLM — quanto il modello è "sorpreso" dal testo. Più bassa = migliore.

---

## 3. Traduzione (Sequence-to-Sequence)

### Cos'è
Trasformare una sequenza in un'altra lingua. Usa architetture **encoder-decoder** (Marian, mBART, mT5).

### Dataset e modello

```python
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

raw_datasets = load_dataset("kde4", lang1="en", lang2="fr")
# 210k coppie di frasi EN→FR

model_checkpoint = "Helsinki-NLP/opus-mt-en-fr"
tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
```

### Preprocessing — tokenizzare input E output

```python
max_length = 128

def preprocess_function(examples):
    inputs = [ex["en"] for ex in examples["translation"]]
    targets = [ex["fr"] for ex in examples["translation"]]

    # IMPORTANTE: text_target= per tokenizzare le traduzioni
    # Senza questo, le frasi francesi vengono tokenizzate come inglesi → errore!
    model_inputs = tokenizer(
        inputs,
        text_target=targets,   # ← la chiave è questa
        max_length=max_length,
        truncation=True,
    )
    return model_inputs

tokenized_datasets = split_datasets.map(
    preprocess_function, batched=True,
    remove_columns=split_datasets["train"].column_names
)
```

### Data Collator Seq2Seq

```python
from transformers import DataCollatorForSeq2Seq

# Padda sia input che label con -100
# Crea anche i decoder_input_ids (label shiftati di 1)
data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)
```

### Metrica: SacreBLEU

```python
import evaluate

metric = evaluate.load("sacrebleu")
# sacrebleu: standardizza la tokenizzazione per confronto equo tra modelli

def compute_metrics(eval_preds):
    preds, labels = eval_preds
    if isinstance(preds, tuple):
        preds = preds[0]

    decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

    decoded_preds = [p.strip() for p in decoded_preds]
    decoded_labels = [[l.strip()] for l in decoded_labels]  # lista di liste!

    result = metric.compute(predictions=decoded_preds, references=decoded_labels)
    return {"bleu": result["score"]}
```

### Training con Seq2SeqTrainer

```python
from transformers import Seq2SeqTrainingArguments, Seq2SeqTrainer

args = Seq2SeqTrainingArguments(
    "marian-finetuned-kde4-en-to-fr",
    evaluation_strategy="no",
    save_strategy="epoch",
    learning_rate=2e-5,
    num_train_epochs=3,
    predict_with_generate=True,  # usa generate() per la valutazione (non logits)
    fp16=True,
    push_to_hub=True,
)

trainer = Seq2SeqTrainer(
    model, args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["validation"],
    data_collator=data_collator,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics,
)
trainer.train()
# Risultato: BLEU 39 (pre-training) → 52 (dopo fine-tuning) ✓
```

### Usare il modello

```python
from transformers import pipeline

translator = pipeline("translation", model="huggingface-course/marian-finetuned-kde4-en-to-fr")
translator("Default to expanded threads")
# → [{'translation_text': 'Par défaut, développer les fils de discussion'}]
# Il modello impara "threads" → "fils de discussion" (non lascia in inglese)
```

---

## 4. Riassunto (Summarization)

### Cos'è
Condensare un testo lungo in uno breve. Task Seq2Seq con modelli come mT5, BART, Pegasus.

### Dataset e preprocessing

```python
from datasets import load_dataset
from transformers import AutoTokenizer

# Dataset bilingue EN+ES di recensioni Amazon
raw_datasets = load_dataset("amazon_reviews_multi", "all_languages")

model_checkpoint = "google/mt5-small"
tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)

max_input_length = 512
max_target_length = 30

def preprocess_function(examples):
    model_inputs = tokenizer(
        examples["review_body"],
        max_length=max_input_length,
        truncation=True,
    )
    # text_target: tokenizza le label (i titoli delle recensioni come riassunti)
    labels = tokenizer(
        examples["review_title"],
        max_length=max_target_length,
        truncation=True,
    )
    model_inputs["labels"] = labels["input_ids"]
    return model_inputs
```

### Metrica: ROUGE

```python
import evaluate

rouge_score = evaluate.load("rouge")
# ROUGE-1: overlap di unigram
# ROUGE-2: overlap di bigrammi
# ROUGE-L: sottosequenza comune più lunga

def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

    result = rouge_score.compute(
        predictions=decoded_preds,
        references=decoded_labels,
        use_stemmer=True
    )
    return {k: round(v, 4) for k, v in result.items()}
```

### Training

```python
from transformers import AutoModelForSeq2SeqLM, Seq2SeqTrainingArguments, Seq2SeqTrainer

model = AutoModelForSeq2SeqLM.from_pretrained(model_checkpoint)

args = Seq2SeqTrainingArguments(
    "mt5-small-finetuned-amazon",
    evaluation_strategy="epoch",
    learning_rate=5.6e-5,
    per_device_train_batch_size=8,
    num_train_epochs=8,
    predict_with_generate=True,
    push_to_hub=True,
)
```

---

## 5. Causal Language Modeling — Generazione da Zero

### Cos'è
Addestrare un modello **da zero** (non fine-tuning) su un corpus specifico. Utile per: linguaggi di programmazione, testi in domini rarissimi, lingue poco rappresentate.

**Causal LM**: il modello prevede il token successivo dati tutti i precedenti (GPT-style).

### Dataset: Codeparrot (Python data science)

```python
from datasets import load_dataset, DatasetDict

ds_train = load_dataset("huggingface-course/codeparrot-ds-train", split="train")
ds_valid = load_dataset("huggingface-course/codeparrot-ds-valid", split="validation")

raw_datasets = DatasetDict({"train": ds_train, "valid": ds_valid})
# 606k script Python contenenti pandas, sklearn, matplotlib, seaborn
```

### Tokenizzazione con chunking

```python
from transformers import AutoTokenizer

context_length = 128  # GPT-2 usa 1024, ma per prototipo va bene 128
tokenizer = AutoTokenizer.from_pretrained("huggingface-course/code-search-net-tokenizer")

def tokenize(element):
    outputs = tokenizer(
        element["content"],
        truncation=True,
        max_length=context_length,
        return_overflowing_tokens=True,  # divide i testi lunghi in chunk da 128 token
        return_length=True,
    )
    # Tieni solo i chunk completi (scarta l'ultimo chunk più corto)
    input_batch = [
        ids for length, ids in zip(outputs["length"], outputs["input_ids"])
        if length == context_length
    ]
    return {"input_ids": input_batch}

tokenized_datasets = raw_datasets.map(
    tokenize, batched=True,
    remove_columns=raw_datasets["train"].column_names
)
# → 16.7M esempi da 128 token ciascuno ≈ 2.1B token totali
```

### Inizializzare il modello da zero

```python
from transformers import AutoConfig, GPT2LMHeadModel

# Usa la configurazione di GPT-2 small come template
config = AutoConfig.from_pretrained(
    "gpt2",
    vocab_size=len(tokenizer),      # vocabolario del nostro tokenizer custom
    n_ctx=context_length,            # finestra di contesto
    bos_token_id=tokenizer.bos_token_id,
    eos_token_id=tokenizer.eos_token_id,
)

# IMPORTANTE: GPT2LMHeadModel(config) NON from_pretrained → pesi random!
model = GPT2LMHeadModel(config)
model_size = sum(t.numel() for t in model.parameters())
print(f"GPT-2 size: {model_size/1e6:.1f}M parameters")  # → 124.2M
```

### Data Collator per CLM

```python
from transformers import DataCollatorForLanguageModeling

tokenizer.pad_token = tokenizer.eos_token  # GPT-2 non ha pad token

# mlm=False → modalità CLM: i label sono gli input shiftati di 1
data_collator = DataCollatorForLanguageModeling(tokenizer, mlm=False)
```

### Training

```python
from transformers import Trainer, TrainingArguments

args = TrainingArguments(
    output_dir="codeparrot-ds",
    per_device_train_batch_size=32,
    evaluation_strategy="steps",
    eval_steps=5_000,
    gradient_accumulation_steps=8,   # batch effettivo = 32 * 8 = 256
    num_train_epochs=1,
    weight_decay=0.1,
    warmup_steps=1_000,
    lr_scheduler_type="cosine",      # learning rate cosine decay
    learning_rate=5e-4,
    fp16=True,
    push_to_hub=True,
)

trainer = Trainer(
    model=model,
    tokenizer=tokenizer,
    args=args,
    data_collator=data_collator,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["valid"],
)
trainer.train()
```

### Generazione di codice

```python
from transformers import pipeline
import torch

pipe = pipeline(
    "text-generation",
    model="huggingface-course/codeparrot-ds",
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
)

# Autocompletamento scatter plot
txt = """
# create scatter plot with x, y
plt."""
print(pipe(txt, num_return_sequences=1)[0]["generated_text"])
# → plt.scatter(x, y) ✓

# Autocompletamento pandas
txt = "# create dataframe from x and y\n"
print(pipe(txt, num_return_sequences=1)[0]["generated_text"])
# → df = pd.DataFrame({'x': x, 'y': y}) ✓
```

---

## 6. Question Answering Estrattiva

### Cos'è
Dato un contesto (paragrafo) e una domanda, trovare nel testo il **span** (inizio-fine) che risponde alla domanda. BERT-like (encoder-only).

Diverso dalla QA generativa (T5/BART che genera la risposta da zero).

### Dataset: SQuAD

```python
from datasets import load_dataset

raw_datasets = load_dataset("squad")
# Ogni esempio: context, question, answers (con start position)
```

### Problema: contesti lunghi e offset mapping

I contesti possono essere più lunghi della finestra del modello (512 token per BERT). Soluzione: sliding window con overlap.

```python
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")

max_length = 384
stride = 128       # overlap tra finestre consecutive

def preprocess_training_examples(examples):
    inputs = tokenizer(
        examples["question"],
        examples["context"],
        max_length=max_length,
        truncation="only_second",      # tronca solo il contesto, non la domanda
        stride=stride,                  # overlap per non perdere risposte ai bordi
        return_overflowing_tokens=True, # genera più chunk per contesti lunghi
        return_offsets_mapping=True,    # mappa token → posizione nel testo originale
        padding="max_length",
    )
    # Per ogni chunk, determina le posizioni start/end della risposta in termini di token
    # Se la risposta non è in questo chunk → label = (0, 0) oppure (-100, -100)
    ...
    return inputs
```

### Training

```python
from transformers import AutoModelForQuestionAnswering, TrainingArguments, Trainer

model = AutoModelForQuestionAnswering.from_pretrained("bert-base-cased")

args = TrainingArguments(
    "bert-finetuned-squad",
    evaluation_strategy="no",
    save_strategy="epoch",
    learning_rate=2e-5,
    num_train_epochs=3,
    fp16=True,
    push_to_hub=True,
)
```

### Post-processing: da logit a risposta testuale

Il modello restituisce `start_logits` e `end_logits` per ogni token. Bisogna trovare la coppia (start, end) con score massima che sia valida (start ≤ end, entrambi nel contesto).

```python
import numpy as np

def postprocess_qa_predictions(examples, features, raw_predictions, n_best=20):
    start_logits, end_logits = raw_predictions
    # Per ogni esempio, considera tutte le finestre che lo contengono
    # Trova la coppia (start, end) con score = start_logit + end_logit massima
    # Converti da posizione token a posizione carattere usando offset_mapping
    # Restituisci il testo del contesto corrispondente
    ...
```

### Usare il modello

```python
from transformers import pipeline

qa_pipeline = pipeline("question-answering", model="huggingface-course/bert-finetuned-squad")
context = "🤗 Transformers is backed by Jax, PyTorch and TensorFlow."
question = "Which deep learning libraries back 🤗 Transformers?"

qa_pipeline(question=question, context=context)
# → {'answer': 'Jax, PyTorch and TensorFlow', 'score': 0.97, 'start': 23, 'end': 56}
```

---

## 7. Scegliere l'Architettura Giusta

| Task | Architettura | Modello esempio | Metrica |
|------|-------------|-----------------|---------|
| NER, POS, Chunking | Encoder-only | BERT, RoBERTa | F1 (seqeval) |
| Domain adaptation MLM | Encoder-only | DistilBERT, BERT | Perplexity |
| Traduzione | Encoder-Decoder | Marian, mBART, M2M100 | SacreBLEU |
| Riassunto | Encoder-Decoder | mT5, BART, Pegasus | ROUGE |
| Generazione testo/codice | Decoder-only | GPT-2, Llama | Perplexity |
| QA estrattiva | Encoder-only | BERT, DeBERTa | F1 su span |
| QA generativa | Encoder-Decoder | T5, BART | ROUGE / BLEU |

---

## 8. Data Collator — Schema Riepilogativo

```
Task                    → Data Collator
────────────────────────────────────────────────────
Token classification    → DataCollatorForTokenClassification(tokenizer)
MLM                     → DataCollatorForLanguageModeling(tokenizer, mlm=True)
CLM (generazione)       → DataCollatorForLanguageModeling(tokenizer, mlm=False)
Traduzione / Riassunto  → DataCollatorForSeq2Seq(tokenizer, model=model)
```

---

## 🕐 Quando usarlo

| Scenario | Approccio |
|----------|-----------|
| Identificare entità in testo | NER con BERT + seqeval |
| Corpus molto diverso dai dati di pretraining | Domain adaptation MLM prima del fine-tuning |
| Traduzione in lingua specifica | Fine-tune Marian/mT5 su coppie di frasi |
| Riassumere documenti | Fine-tune mT5/BART + metrica ROUGE |
| Completamento codice / testo | GPT-2 addestrato da zero o fine-tuned |
| Trovare risposta in un documento | QA estrattiva + BERT |
| Risposta generativa aperta | QA generativa + T5/BART |

---

## ⚠️ Errori comuni

1. **`num_labels` sbagliato** — Verificare sempre `model.config.num_labels` prima del training: errore difficile da debuggare (`CUDA error: device-side assert triggered`).

2. **Non usare `is_split_into_words=True`** — Per NER il testo è già pre-tokenizzato in parole; senza questo flag il tokenizer lo tratterà come una stringa unica.

3. **Non passare `text_target=` al tokenizer per Seq2Seq** — Le frasi target vengono tokenizzate come input (vocabolario sbagliato) → performance molto peggiori.

4. **`predict_with_generate=False`** — Per traduzione e riassunto va usato `predict_with_generate=True` in `Seq2SeqTrainingArguments`, altrimenti la valutazione usa i logit invece di generare testo.

5. **DataCollator sbagliato** — Usare `DataCollatorWithPadding` per NER non padda i label → errore o risultati inconsistenti. Usa sempre il collator specifico del task.

6. **Chunk senza overlap per QA** — Se il contesto viene spezzato senza `stride`, le risposte ai bordi vengono perse. Usa `return_overflowing_tokens=True` e `stride=128`.

7. **MLM su testo non mascherato** — Se usi `DataCollatorForLanguageModeling` senza `return_special_tokens_mask=True`, i token speciali potrebbero essere mascherati → training instabile.
