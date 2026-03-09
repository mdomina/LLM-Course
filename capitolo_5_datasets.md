# Capitolo 5 — La libreria 🤗 Datasets in profondità

> **Obiettivo del capitolo**: padroneggiare la libreria `datasets` per caricare dati da qualsiasi fonte, manipolarli in modo efficiente, gestire dataset enormi senza esaurire la RAM, e creare/pubblicare dataset personalizzati sull'Hub.

---

## Indice

1. [Caricare dataset non presenti sull'Hub](#1-caricare-dataset-non-presenti-sullhub)
2. [Slice & dice — manipolare i dati](#2-slice--dice--manipolare-i-dati)
3. [Dataset enormi: memory mapping e streaming](#3-dataset-enormi-memory-mapping-e-streaming)
4. [Creare un dataset personalizzato](#4-creare-un-dataset-personalizzato)
5. [Semantic search con FAISS](#5-semantic-search-con-faiss)
6. [TL;DR — Riepilogo finale](#6-tldr--riepilogo-finale)

---

## 1. Caricare dataset non presenti sull'Hub

### Formati supportati

`load_dataset()` non funziona solo con l'Hub: accetta file locali e URL remoti in diversi formati.

| Formato | Script | Esempio |
|---|---|---|
| CSV / TSV | `"csv"` | `load_dataset("csv", data_files="file.csv")` |
| Testo | `"text"` | `load_dataset("text", data_files="file.txt")` |
| JSON / JSONL | `"json"` | `load_dataset("json", data_files="file.jsonl")` |
| DataFrame Pandas | `"pandas"` | `load_dataset("pandas", data_files="df.pkl")` |

### Caricare file locali

```python
from datasets import load_dataset

# File JSON con struttura annidata (campo "data" contiene i dati)
squad_it_dataset = load_dataset("json", data_files="SQuAD_it-train.json", field="data")
# Risultato: DatasetDict con solo split "train"

# Per avere più split in un colpo solo, usa un dizionario
data_files = {"train": "SQuAD_it-train.json", "test": "SQuAD_it-test.json"}
squad_it_dataset = load_dataset("json", data_files=data_files, field="data")
# Risultato: DatasetDict con split "train" e "test"
```

> ✅ `load_dataset` supporta anche **decompressione automatica** di `.gz`, `.zip`, `.tar` — non serve decomprimere manualmente.

```python
# Funziona direttamente con file compressi
data_files = {"train": "SQuAD_it-train.json.gz", "test": "SQuAD_it-test.json.gz"}
squad_it_dataset = load_dataset("json", data_files=data_files, field="data")
```

### Caricare file remoti (da URL)

Identico al caso locale: basta passare URL invece di percorsi.

```python
url = "https://github.com/crux82/squad-it/raw/master/"
data_files = {
    "train": url + "SQuAD_it-train.json.gz",
    "test": url + "SQuAD_it-test.json.gz",
}
squad_it_dataset = load_dataset("json", data_files=data_files, field="data")
# Scarica, decomprime e carica automaticamente
```

### Caricare TSV con separatore custom

```python
data_files = {"train": "drugsComTrain_raw.tsv", "test": "drugsComTest_raw.tsv"}
drug_dataset = load_dataset("csv", data_files=data_files, delimiter="\t")
# \t = carattere tab (TSV = Tab-Separated Values)
```

### Quando usarlo

- Dataset aziendali su server interno → usa URL diretti
- File locali prodotti da scraping o esportazioni → specifica il formato e il percorso
- Glob pattern per caricare più file come unico split: `data_files="data/*.json"`

### Errori comuni

- ❌ Non specificare `field` per JSON annidati → `load_dataset` legge la struttura sbagliata
- ❌ Usare `load_dataset` su un `.json.gz` senza sapere che funziona direttamente → inutile decomprimere a mano
- ❌ Dimenticare `delimiter="\t"` per i TSV → le colonne vengono parsate male

---

## 2. Slice & dice — manipolare i dati

### Campione casuale rapido

Prima di elaborare tutto il dataset, guarda un campione per capire cosa stai trattando.

```python
# shuffle + select = campione casuale riproducibile
drug_sample = drug_dataset["train"].shuffle(seed=42).select(range(1000))
drug_sample[:3]  # Guarda i primi 3 elementi del campione
```

### Rinominare colonne

```python
# Rinomina su tutti gli split in un colpo
drug_dataset = drug_dataset.rename_column(
    original_column_name="Unnamed: 0",
    new_column_name="patient_id"
)
```

### Filtrare righe con `.filter()`

```python
# Con lambda (per filtri semplici)
drug_dataset = drug_dataset.filter(lambda x: x["condition"] is not None)

# Con funzione esplicita (per logiche complesse)
def filter_short_reviews(x):
    return x["review_length"] > 30

drug_dataset = drug_dataset.filter(filter_short_reviews)
```

> ⚠️ Se provi a fare `.map()` su una colonna con valori `None`, ottieni `AttributeError`. Filtra sempre prima i `None`.

### Creare nuove colonne con `.map()`

```python
# La funzione restituisce una chiave NON esistente → crea una nuova colonna
def compute_review_length(example):
    return {"review_length": len(example["review"].split())}

drug_dataset = drug_dataset.map(compute_review_length)
# Ora il dataset ha una colonna "review_length" in più
```

### Ordinare con `.sort()`

```python
# Ordina per review_length (crescente)
drug_dataset["train"].sort("review_length")[:3]
```

### Pulire HTML con `.map()` + lambda

```python
import html

# Converti caratteri HTML: "I&#039;m" → "I'm"
drug_dataset = drug_dataset.map(lambda x: {"review": html.unescape(x["review"])})
```

### Accelerare con `batched=True`

Usando `batched=True`, la funzione riceve un **dizionario di liste** anziché un dizionario di valori singoli. Molto più veloce.

```python
# Senza batched: ~59 secondi con tokenizer veloce
drug_dataset.map(tokenize_function)

# Con batched=True: ~10 secondi (6x più veloce)
drug_dataset.map(tokenize_function, batched=True)
```

**Confronto prestazioni tokenizzazione:**

| Configurazione | Fast tokenizer | Slow tokenizer |
|---|---|---|
| `batched=False` | 59s | 5min 3s |
| `batched=True` | **10.8s** | 4min 41s |
| `batched=True`, `num_proc=8` | 6.5s | **41s** |

> Il fast tokenizer è scritto in **Rust** → parallelizza automaticamente → `batched=True` + fast tokenizer = combinazione ideale.
> Per slow tokenizer: usa `num_proc=8` (multiprocessing Python).

### Espandere esempi con `return_overflowing_tokens`

Quando tokenizzi con `max_length=128`, le review lunghe vengono spezzate in più chunk. Questo aumenta il numero di righe nel dataset.

```python
def tokenize_and_split(examples):
    result = tokenizer(
        examples["review"],
        truncation=True,
        max_length=128,
        return_overflowing_tokens=True,  # Restituisce tutti i chunk, non solo il primo
    )
    # overflow_to_sample_mapping mappa ogni nuovo chunk all'esempio originale
    sample_map = result.pop("overflow_to_sample_mapping")
    # Propaga i valori delle colonne originali sui nuovi chunk
    for key, values in examples.items():
        result[key] = [values[i] for i in sample_map]
    return result

tokenized_dataset = drug_dataset.map(tokenize_and_split, batched=True)
# 138.514 esempi → 206.772 chunk (ogni review lunga diventa più righe)
```

> ⚠️ Se non propaghi le colonne originali con `overflow_to_sample_mapping`, ottieni `ArrowInvalid: Column expected length X but got Y`.

### Interoperabilità con Pandas

```python
# Dataset → DataFrame Pandas
drug_dataset.set_format("pandas")
train_df = drug_dataset["train"][:]  # Slice per ottenere un vero DataFrame

# Analisi con Pandas (groupby, value_counts, ecc.)
frequencies = train_df["condition"].value_counts().to_frame()

# DataFrame → Dataset (dopo analisi)
from datasets import Dataset
freq_dataset = Dataset.from_pandas(frequencies)

# Torna al formato Arrow (indispensabile prima di usare .map() di nuovo)
drug_dataset.reset_format()
```

### Creare uno split di validazione

```python
# Divide il training set in train (80%) e validation (20%)
drug_dataset_clean = drug_dataset["train"].train_test_split(train_size=0.8, seed=42)

# Per default il secondo split si chiama "test" → rinominalo "validation"
drug_dataset_clean["validation"] = drug_dataset_clean.pop("test")

# Aggiungi il vero test set
drug_dataset_clean["test"] = drug_dataset["test"]
```

### Salvare e ricaricare dataset

```python
# Formato Arrow (più efficiente, consigliato)
drug_dataset_clean.save_to_disk("drug-reviews")
from datasets import load_from_disk
drug_dataset_reloaded = load_from_disk("drug-reviews")

# Formato JSON Lines (più portabile)
for split, dataset in drug_dataset_clean.items():
    dataset.to_json(f"drug-reviews-{split}.jsonl")

# Formato CSV
for split, dataset in drug_dataset_clean.items():
    dataset.to_csv(f"drug-reviews-{split}.csv", index=False)
```

### Quando usarlo

- `.filter()` → rimuovere righe rumorose o incomplete
- `.map()` → trasformare, pulire, aggiungere colonne
- `.set_format("pandas")` → analisi esplorative, statistiche, visualizzazioni
- `.train_test_split()` → creare validation set quando non esiste

### Errori comuni

- ❌ Non chiamare `reset_format()` dopo `.set_format("pandas")` → le operazioni successive si rompono
- ❌ Usare `.map()` senza `batched=True` su dataset grandi → molto lento
- ❌ Non gestire i `None` prima di operazioni sulle stringhe → `AttributeError`

---

## 3. Dataset enormi: memory mapping e streaming

### Il problema

Dataset come GPT-2 WebText (40 GB) o The Pile (825 GB) non entrano nella RAM. Come fare?

### Soluzione 1: Memory Mapping (default)

🤗 Datasets **non carica mai l'intero dataset in RAM**. Usa file **Apache Arrow** mappati in memoria: il sistema operativo carica in RAM solo le parti che servono in quel momento.

```python
from datasets import load_dataset
import psutil

# Dataset da 19.5 GB su disco
pubmed_dataset = load_dataset("json", data_files="PUBMED_abstracts.jsonl.zst", split="train")
# 15.5 milioni di righe!

# Quanta RAM usa?
print(f"RAM usata: {psutil.Process().memory_info().rss / (1024 * 1024):.2f} MB")
# Output: ~5.678 MB — nonostante il dataset sia 19.5 GB!

print(f"Dimensione dataset: {pubmed_dataset.dataset_size / (1024**3):.2f} GB")
# Output: 19.54 GB
```

**Come funziona il memory mapping:**
- Il dataset viene salvato come file Arrow su disco
- La libreria crea una "mappa" tra RAM e disco
- Quando accedi a un elemento, solo quella parte viene caricata
- Diversi processi condividono la stessa mappa → `.map()` parallelizzato senza copiare dati

**Velocità di lettura:** ~0.3 GB/s — sufficiente per la maggior parte dei casi d'uso.

### Soluzione 2: Streaming (per dataset troppo grandi anche per il disco)

Con `streaming=True`, gli esempi vengono scaricati e processati **uno alla volta**, senza mai salvare nulla su disco.

```python
# Carica in modalità streaming
pubmed_dataset_streamed = load_dataset(
    "json", data_files=data_files, split="train", streaming=True
)
# Restituisce un IterableDataset, non un Dataset

# Accedere agli elementi: usa next(iter(...)), non l'indicizzazione
primo_esempio = next(iter(pubmed_dataset_streamed))

# Tokenizzare on-the-fly
from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")

tokenized_dataset = pubmed_dataset_streamed.map(lambda x: tokenizer(x["text"]))
primo_tokenizzato = next(iter(tokenized_dataset))
```

**Operazioni disponibili su `IterableDataset`:**

```python
# Shuffle con buffer (non shuffla tutto, solo un buffer di N esempi)
shuffled = pubmed_dataset_streamed.shuffle(buffer_size=10_000, seed=42)

# Prendi i primi N esempi
prime_5 = list(pubmed_dataset_streamed.take(5))

# Salta i primi N esempi
resto = pubmed_dataset_streamed.skip(1000)

# Crea train/validation split da streaming
train_ds = shuffled_dataset.skip(1000)
val_ds = shuffled_dataset.take(1000)
```

### Combinare più dataset in streaming

```python
from datasets import interleave_datasets
from itertools import islice

# Carica due dataset enormi in streaming
pubmed_streamed = load_dataset("json", data_files=pubmed_url, split="train", streaming=True)
law_streamed = load_dataset("json", data_files=law_url, split="train", streaming=True)

# Intercala: prende un esempio da ciascuno a turno
combined = interleave_datasets([pubmed_streamed, law_streamed])

# Guarda i primi 2 esempi (uno da PubMed, uno da FreeLaw)
list(islice(combined, 2))
```

### Quando usarlo

| Situazione | Soluzione |
|---|---|
| Dataset < RAM disponibile | Memory mapping (default) — nessuna modifica necessaria |
| Dataset > RAM ma < spazio su disco | Memory mapping (default) — funziona lo stesso |
| Dataset > spazio su disco | Streaming (`streaming=True`) |
| Pre-training su corpus enorme | Streaming + `interleave_datasets` |

### Errori comuni

- ❌ Provare a indicizzare un `IterableDataset` con `dataset[0]` → errore: usa `next(iter(dataset))`
- ❌ Fare `.shuffle()` su streaming senza `buffer_size` → shuffle poco efficace su dataset enormi
- ❌ Non sapere che il memory mapping è il comportamento **default** → non serve fare nulla di speciale per dataset "normali"

---

## 4. Creare un dataset personalizzato

### Esempio: raccogliere issue da GitHub via API REST

```python
import requests

GITHUB_TOKEN = "hf_xxx"  # Token personale GitHub
headers = {"Authorization": f"token {GITHUB_TOKEN}"}

# Singola richiesta: prima issue della prima pagina
url = "https://api.github.com/repos/huggingface/datasets/issues?page=1&per_page=1"
response = requests.get(url, headers=headers)
print(response.status_code)  # 200 = OK
data = response.json()       # Lista di dizionari JSON
```

> ⚠️ **Rate limit**: senza token → 60 req/ora. Con token → 5.000 req/ora.
> ⚠️ **Sicurezza**: non committare mai il token nel codice. Usa un file `.env` + `python-dotenv`.

### Funzione per scaricare tutte le issue

```python
import time
import math
from pathlib import Path
import pandas as pd
from tqdm.notebook import tqdm

def fetch_issues(owner="huggingface", repo="datasets", num_issues=10_000,
                 rate_limit=5_000, issues_path=Path(".")):
    issues_path.mkdir(exist_ok=True)
    batch, all_issues = [], []
    per_page = 100
    num_pages = math.ceil(num_issues / per_page)
    base_url = "https://api.github.com/repos"

    for page in tqdm(range(num_pages)):
        query = f"issues?page={page}&per_page={per_page}&state=all"
        issues = requests.get(f"{base_url}/{owner}/{repo}/{query}", headers=headers)
        batch.extend(issues.json())

        if len(batch) > rate_limit and len(all_issues) < num_issues:
            # Salva batch e aspetta per non superare il rate limit
            all_issues.extend(batch)
            batch = []
            time.sleep(3600)  # Aspetta un'ora

    all_issues.extend(batch)
    return pd.DataFrame.from_records(all_issues)
```

### Pulizia: distinguere issue da pull request

```python
from datasets import Dataset

issues_dataset = Dataset.from_pandas(issues_df)

# L'API GitHub restituisce sia issue che PR — le PR hanno il campo "pull_request" != None
issues_dataset = issues_dataset.map(
    lambda x: {"is_pull_request": x["pull_request"] is not None}
)
```

### Arricchire il dataset con i commenti

```python
def get_comments(issue_number):
    url = f"https://api.github.com/repos/huggingface/datasets/issues/{issue_number}/comments"
    response = requests.get(url, headers=headers)
    return [r["body"] for r in response.json()]

# Aggiungi colonna "comments" — può richiedere qualche minuto
issues_with_comments = issues_dataset.map(
    lambda x: {"comments": get_comments(x["number"])}
)
```

### Caricare il dataset sull'Hub

```python
from huggingface_hub import notebook_login
notebook_login()  # O: huggingface-cli login da terminale

# Push sull'Hub
issues_with_comments.push_to_hub("github-issues")

# Chiunque può ora scaricarlo con:
from datasets import load_dataset
remote_dataset = load_dataset("tuo-username/github-issues", split="train")
```

### Dataset card

Ogni dataset sull'Hub deve avere un `README.md` con metadati YAML:

```yaml
---
language:
  - en
license: apache-2.0
task_categories:
  - text-classification
  - question-answering
---

# Nome del Dataset

Descrizione: cosa contiene, come è stato raccolto, a cosa serve.

## Struttura dei dati
Colonne: `title`, `body`, `comments`, `is_pull_request`, ...

## Come è stato creato
GitHub REST API, repository huggingface/datasets, raccolta il...

## Utilizzi previsti
...

## Limitazioni e bias
...
```

### Quando usarlo

- Dati proprietari o verticali (medici, legali, aziendali) non disponibili sull'Hub
- Scraping di dati da API web (GitHub, Reddit, Stack Overflow, ecc.)
- Combinazione di più sorgenti in un unico dataset pulito

### Errori comuni

- ❌ Hardcodare il token GitHub nel notebook → usa `.env`
- ❌ Non distinguere issue da PR → dati eterogenei nel dataset
- ❌ Non aggiungere dataset card → il dataset non viene trovato nelle ricerche Hub

---

## 5. Semantic search con FAISS

### Cos'è la ricerca semantica

La **ricerca per parola chiave** (keyword search) cerca corrispondenze esatte di termini. La **ricerca semantica** capisce il _significato_ della query e trova documenti simili concettualmente, anche se non usano le stesse parole.

Esempio: la query "come caricare un dataset offline?" trova documenti che parlano di `load_from_disk`, `save_to_disk`, `HF_DATASETS_OFFLINE`, anche se non contengono la parola "offline".

### Pipeline completa: da testo a motore di ricerca

**Step 1 — Preparare il dataset**

```python
from datasets import load_dataset

issues_dataset = load_dataset("lewtun/github-issues", split="train")

# Filtra: tieni solo issue (non PR) con almeno un commento
issues_dataset = issues_dataset.filter(
    lambda x: (x["is_pull_request"] == False and len(x["comments"]) > 0)
)
# 2.855 → 771 righe

# Tieni solo le colonne utili
issues_dataset = issues_dataset.remove_columns(
    [col for col in issues_dataset.column_names
     if col not in ["title", "body", "html_url", "comments"]]
)
```

**Step 2 — Esplodere i commenti (un commento per riga)**

```python
# Ogni issue ha una lista di commenti → crea una riga per ogni commento
issues_dataset.set_format("pandas")
df = issues_dataset[:]
comments_df = df.explode("comments", ignore_index=True)
# 771 issue → 2.842 righe (una per commento)

from datasets import Dataset
comments_dataset = Dataset.from_pandas(comments_df)
comments_dataset.reset_format()

# Filtra commenti troppo corti (es. "grazie!", "cc @lewtun")
comments_dataset = comments_dataset.map(
    lambda x: {"comment_length": len(x["comments"].split())}
)
comments_dataset = comments_dataset.filter(lambda x: x["comment_length"] > 15)
# 2.842 → 2.098 righe
```

**Step 3 — Creare il campo "text" per l'embedding**

```python
def concatenate_text(examples):
    return {
        "text": examples["title"] + " \n " + examples["body"] + " \n " + examples["comments"]
    }

comments_dataset = comments_dataset.map(concatenate_text)
# Ogni riga ha ora un campo "text" = titolo + corpo issue + commento
```

**Step 4 — Generare gli embedding**

```python
from transformers import AutoTokenizer, AutoModel
import torch

model_ckpt = "sentence-transformers/multi-qa-mpnet-base-dot-v1"
# Modello ottimizzato per la ricerca semantica asimmetrica
# (query corta → documento lungo)

tokenizer = AutoTokenizer.from_pretrained(model_ckpt)
model = AutoModel.from_pretrained(model_ckpt)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

def cls_pooling(model_output):
    # CLS pooling: usa il vettore del token [CLS] come rappresentazione della frase
    # È il primo token (indice 0) dell'ultimo hidden state
    return model_output.last_hidden_state[:, 0]

def get_embeddings(text_list):
    encoded_input = tokenizer(
        text_list, padding=True, truncation=True, return_tensors="pt"
    )
    encoded_input = {k: v.to(device) for k, v in encoded_input.items()}
    model_output = model(**encoded_input)
    return cls_pooling(model_output)

# Genera embedding per tutto il dataset
embeddings_dataset = comments_dataset.map(
    lambda x: {"embeddings": get_embeddings(x["text"]).detach().cpu().numpy()[0]}
)
# Ogni riga ha ora un vettore di 768 dimensioni
```

**Step 5 — Creare l'indice FAISS**

```python
# FAISS (Facebook AI Similarity Search) = libreria per ricerca efficiente di vettori simili
embeddings_dataset.add_faiss_index(column="embeddings")
# Crea una struttura dati ottimizzata per trovare vettori vicini rapidamente
```

**Step 6 — Fare una query**

```python
# 1. Converti la domanda in un vettore
question = "How can I load a dataset offline?"
question_embedding = get_embeddings([question]).cpu().detach().numpy()

# 2. Cerca i 5 documenti più simili
scores, samples = embeddings_dataset.get_nearest_examples(
    "embeddings",       # Colonna dell'indice
    question_embedding, # Vettore della query
    k=5                 # Numero di risultati
)

# 3. Mostra i risultati ordinati per score
import pandas as pd
samples_df = pd.DataFrame.from_dict(samples)
samples_df["scores"] = scores
samples_df.sort_values("scores", ascending=False, inplace=True)

for _, row in samples_df.iterrows():
    print(f"SCORE: {row.scores:.2f}")
    print(f"TITOLO: {row.title}")
    print(f"URL: {row.html_url}")
    print(f"COMMENTO: {row.comments[:200]}...")
    print("=" * 50)
```

### Ricerca simmetrica vs asimmetrica

| Tipo | Query | Documento | Modello consigliato |
|---|---|---|---|
| **Simmetrica** | Stessa lunghezza del documento | Simile alla query | `paraphrase-mpnet-base-v2` |
| **Asimmetrica** | Corta (domanda) | Lungo (risposta/articolo) | `multi-qa-mpnet-base-dot-v1` |

### Quando usarlo

- Motore di ricerca su documentazione interna
- FAQ bot: trova la risposta più simile a una domanda utente
- Deduplicazione di testi semanticamente simili
- Raccomandazione di contenuti correlati

### Errori comuni

- ❌ Usare keyword search quando le query sono parafrastiche → usa semantic search
- ❌ Non filtrare i commenti corti → rumore nel dataset abbassa la qualità dei risultati
- ❌ Usare `cls_pooling` senza considerare alternative (mean pooling) → per alcuni modelli mean pooling funziona meglio
- ❌ Non spostare modello e tensori sulla GPU → embedding lentissimi su CPU per dataset grandi

---

## 6. TL;DR — Riepilogo finale

**Le 5 cose fondamentali del capitolo:**

1. **Caricare da qualsiasi fonte** — `load_dataset("csv/json/text", data_files=...)` per locale e remoto, con decompressione automatica

2. **Manipolare i dati** — pipeline tipica:
   ```python
   dataset.filter(lambda x: ...)      # Rimuovi righe indesiderate
   dataset.map(func, batched=True)     # Trasforma/aggiungi colonne
   dataset.rename_column(...)          # Rinomina
   dataset.train_test_split(0.8)       # Crea validation set
   dataset.save_to_disk("path")        # Salva in Arrow
   ```

3. **Dataset enormi** — memory mapping è il default (nessuna modifica), streaming con `streaming=True` per dataset che non stanno nemmeno su disco

4. **Creare dataset custom** — raccogli dati via API → pulisci → `.push_to_hub()` → aggiungi dataset card

5. **Semantic search** — pipeline: filtra → esplodi → concatena testi → embedding con sentence-transformers → indice FAISS → query

**Cheat sheet delle operazioni più usate:**

```python
# Caricare
load_dataset("glue", "mrpc")                         # dall'Hub
load_dataset("json", data_files="f.jsonl")           # locale
load_dataset("json", data_files=url, streaming=True) # streaming

# Manipolare
ds.filter(lambda x: x["col"] is not None)
ds.map(func, batched=True, num_proc=4)
ds.rename_column("vecchio", "nuovo")
ds.remove_columns(["col1", "col2"])
ds.shuffle(seed=42).select(range(1000))
ds.sort("colonna")
ds.unique("colonna")

# Formato
ds.set_format("pandas")     # → DataFrame
ds.reset_format()           # → Arrow (default)
Dataset.from_pandas(df)     # DataFrame → Dataset

# Salvare
ds.save_to_disk("path")
load_from_disk("path")
ds.to_json("file.jsonl")
ds.push_to_hub("username/nome-dataset")
```
