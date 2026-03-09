# Capitolo 4 — L'HuggingFace Hub: condividere e usare modelli

> **Obiettivo del capitolo**: imparare a usare l'HuggingFace Hub per trovare modelli pre-addestrati, caricare i propri modelli e documentarli correttamente con una model card.

---

## Indice

1. [Cos'è l'HuggingFace Hub](#1-cosè-lhuggingface-hub)
2. [Usare modelli pre-addestrati dall'Hub](#2-usare-modelli-pre-addestrati-dallhub)
3. [Caricare un modello sull'Hub](#3-caricare-un-modello-sullhub)
4. [Scrivere una Model Card](#4-scrivere-una-model-card)
5. [TL;DR — Riepilogo finale](#5-tldr--riepilogo-finale)

---

## 1. Cos'è l'HuggingFace Hub

L'[HuggingFace Hub](https://huggingface.co/) è una piattaforma centralizzata dove chiunque può:

- **scoprire** modelli e dataset open-source
- **usare** modelli direttamente nel codice in poche righe
- **condividere** i propri modelli con la community

Ogni modello è ospitato come un **repository Git**, il che garantisce versionamento e riproducibilità. I modelli non si limitano all'NLP: ci sono modelli per speech, computer vision, audio, ecc.

Un vantaggio chiave: appena carichi un modello, l'Hub **deploy automaticamente un'Inference API** — chiunque può testarla direttamente dalla pagina del modello, senza installare nulla.

I modelli pubblici sono **completamente gratuiti**. Esistono piani a pagamento per repository privati.

---

## 2. Usare modelli pre-addestrati dall'Hub

### Il modo più rapido: `pipeline()`

Basta il nome del checkpoint — l'Hub scarica automaticamente modello e tokenizer.

```python
from transformers import pipeline

# Esempio: modello francese per il fill-mask
camembert_fill_mask = pipeline("fill-mask", model="camembert-base")
results = camembert_fill_mask("Le camembert est  :)")
# Output: lista di 5 completamenti con punteggio di confidenza
# [{'sequence': 'Le camembert est délicieux :)', 'score': 0.49, ...}, ...]
```

> ⚠️ **Attenzione al task**: il checkpoint scelto deve essere compatibile con il task della pipeline. `camembert-base` va bene per `fill-mask` perché è stato addestrato con masked language modeling. Caricarlo in `text-classification` darebbe risultati senza senso perché la "testa" del modello è sbagliata.
>
> **Come scegliere?** Usa il filtro per task sulla pagina dell'Hub prima di scegliere un checkpoint.

### Il modo flessibile: `Auto*` classes

```python
from transformers import AutoTokenizer, AutoModelForMaskedLM

# ❌ Approccio specifico — lega il codice a un'architettura precisa
from transformers import CamembertTokenizer, CamembertForMaskedLM
tokenizer = CamembertTokenizer.from_pretrained("camembert-base")
model = CamembertForMaskedLM.from_pretrained("camembert-base")

# ✅ Approccio consigliato — Auto* funziona con qualsiasi checkpoint
tokenizer = AutoTokenizer.from_pretrained("camembert-base")
model = AutoModelForMaskedLM.from_pretrained("camembert-base")
# Cambiare checkpoint in futuro richiede di modificare solo il nome del checkpoint,
# non le classi importate
```

Le classi `Auto*` sono **agnostiche all'architettura**: leggono il file `config.json` del checkpoint e istanziano automaticamente la classe giusta. Questo è il modo corretto per scrivere codice riutilizzabile.

### Leggere la model card prima di usare un modello

Prima di mettere un modello in produzione, leggi sempre la **model card** (la pagina del modello sull'Hub). Deve contenere:
- Su quali dati è stato addestrato
- Per quali lingue/domini è adatto
- I suoi limiti e bias noti

### Quando usarlo

- Ogni volta che stai iniziando un nuovo progetto NLP: cerca prima sull'Hub un modello pre-addestrato vicino al tuo task
- Per prototipare rapidamente con `pipeline()` prima di scendere a livello più basso
- Quando vuoi un modello in una lingua specifica (es. italiano, francese, cinese)

### Errori comuni

- ❌ Usare classi specifiche (`BertTokenizer`, `GPT2Model`) invece delle `Auto*` → codice fragile, difficile da cambiare checkpoint
- ❌ Non leggere la model card → rischi di usare un modello con bias o limitazioni non dichiarate
- ❌ Caricare un checkpoint in una pipeline incompatibile → risultati non sensati senza errori espliciti

---

## 3. Caricare un modello sull'Hub

Ci sono **tre metodi** per caricare un modello. Eccoli in ordine di semplicità.

### Prerequisito: autenticarsi

```bash
# Da terminale
huggingface-cli login
# Ti chiederà username e password (o token API)
```

```python
# Da notebook (es. Google Colab)
from huggingface_hub import notebook_login
notebook_login()
```

Il token viene salvato in cache (`~/.cache/huggingface/token`) e riusato automaticamente.

---

### Metodo 1 — `push_to_hub=True` nel Trainer ⭐ (più semplice)

Il modo più comodo se stai già usando la `Trainer` API. Basta aggiungere un parametro:

```python
from transformers import TrainingArguments

training_args = TrainingArguments(
    "bert-finetuned-mrpc",   # Nome del repo che verrà creato sull'Hub
    save_strategy="epoch",   # Salva (e carica) il modello alla fine di ogni epoch
    push_to_hub=True,        # Carica automaticamente ogni checkpoint sull'Hub
    hub_model_id="mio-nome/bert-finetuned-mrpc",  # Opzionale: nome personalizzato
)
```

Al termine del training, esegui un push finale che include anche la model card con i metadati:

```python
trainer.push_to_hub()
# Carica l'ultima versione del modello + genera automaticamente la model card
# con hyperparametri, metriche di valutazione, ecc.
```

---

### Metodo 2 — `.push_to_hub()` direttamente su modello/tokenizer

Utile quando non usi il `Trainer` o vuoi caricare un modello già addestrato.

```python
from transformers import AutoModelForMaskedLM, AutoTokenizer

checkpoint = "camembert-base"
model = AutoModelForMaskedLM.from_pretrained(checkpoint)
tokenizer = AutoTokenizer.from_pretrained(checkpoint)

# ... fai fine-tuning o modifiche al modello ...

# Caricare il modello sull'Hub
model.push_to_hub("dummy-model")
# Crea il repo "tuo-username/dummy-model" e carica tutti i file del modello

# Caricare anche il tokenizer (indispensabile!)
tokenizer.push_to_hub("dummy-model")

# Per caricare in un'organizzazione
tokenizer.push_to_hub("dummy-model", organization="mia-org")

# Con un token specifico (se non usi quello in cache)
tokenizer.push_to_hub("dummy-model", use_auth_token="hf_xxxx")
```

Il repo sarà raggiungibile all'indirizzo:
`https://huggingface.co/tuo-username/dummy-model`

---

### Metodo 3 — `huggingface_hub` Python library (controllo totale)

Per gestire repository in modo programmatico (creare, eliminare, listare file, ecc.).

```python
from huggingface_hub import (
    login, logout, whoami,          # Gestione utente
    create_repo, delete_repo,        # Gestione repository
    list_models, list_datasets,      # Esplorazione Hub
    upload_file, delete_file,        # Gestione file
)

# Creare un nuovo repository
create_repo("dummy-model")
create_repo("dummy-model", organization="huggingface")  # In un'organizzazione
create_repo("dummy-model", private=True)                 # Repository privato
create_repo("my-dataset", repo_type="dataset")           # Repository di dataset

# Caricare un singolo file (funziona per file < 5GB)
upload_file(
    "percorso/locale/config.json",   # File locale da caricare
    path_in_repo="config.json",       # Percorso nel repository
    repo_id="tuo-username/dummy-model",
)
```

#### Sottometodo — Repository class (interfaccia git-like)

Per gestire il repository come faresti con `git`:

```python
from huggingface_hub import Repository

# Clona il repository in locale
repo = Repository("cartella-locale", clone_from="tuo-username/dummy-model")

# Comandi git disponibili
repo.git_pull()    # Scarica gli ultimi aggiornamenti
repo.git_add()     # Aggiunge i file all'area di staging
repo.git_commit()  # Crea un commit
repo.git_push()    # Carica sul remote
repo.git_tag()     # Crea un tag

# Workflow tipico dopo aver salvato modello e tokenizer
model.save_pretrained("cartella-locale")
tokenizer.save_pretrained("cartella-locale")

repo.git_pull()
repo.git_add()
repo.git_commit("Add model and tokenizer files")
repo.git_push()
```

---

### Metodo 4 — Git diretto dalla riga di comando

Il metodo più "puro" ma anche il più manuale. Utile per capire cosa succede sotto.

```bash
# 1. Inizializzare git-lfs (per i file grandi come i pesi del modello)
git lfs install

# 2. Clonare il repository dall'Hub
git clone https://huggingface.co/tuo-username/dummy-model

# 3. Salvare modello e tokenizer nella cartella clonata (da Python)
# model.save_pretrained("dummy")
# tokenizer.save_pretrained("dummy")

# 4. Controllare lo stato
cd dummy
git status
git lfs status  # Mostra quali file vengono gestiti da LFS

# 5. Commit e push
git add .
git commit -m "First model version"
git push
```

**Cosa viene gestito da git-lfs?**

I file con estensione `.bin` (pesi PyTorch, >400MB) e `.h5` (pesi TensorFlow) vengono automaticamente tracciati da git-lfs. I file di configurazione (`.json`) restano in git normale.

```
config.json              → Git (piccolo)
pytorch_model.bin        → LFS (>400 MB)
sentencepiece.bpe.model  → LFS (grande)
tokenizer.json           → Git (piccolo)
```

### Quando usare quale metodo

| Situazione | Metodo consigliato |
|---|---|
| Sto usando `Trainer` e voglio caricare alla fine | `push_to_hub=True` in `TrainingArguments` |
| Ho già un modello salvato e voglio caricarlo | `.push_to_hub("nome-repo")` su modello e tokenizer |
| Voglio gestire il repo programmaticamente | `huggingface_hub` library |
| Preferisco la riga di comando / workflow git | Git + git-lfs |

### Errori comuni

- ❌ Caricare solo il modello senza il tokenizer → chi scarica il modello non può usarlo
- ❌ Non fare `trainer.push_to_hub()` finale → manca l'ultima versione e la model card
- ❌ Usare `upload_file` per file >5GB → usa git-lfs o la classe `Repository`
- ❌ Non inizializzare `git lfs install` prima di fare push → i file grandi non vengono caricati correttamente

---

## 4. Scrivere una Model Card

### Cos'è e perché è importante

La **model card** è il file `README.md` nella root del repository. È importante quanto il modello stesso: senza una buona documentazione, nessuno saprà come usarlo correttamente (o se è adatto al loro caso d'uso).

Il concetto nasce da un paper Google del 2018: *"Model Cards for Model Reporting"* di Margaret Mitchell et al.

### Struttura standard di una model card

```markdown
---
language: it
license: apache-2.0
datasets:
  - oscar
tags:
  - text-classification
  - bert
metrics:
  - accuracy
  - f1
---

# Nome del modello

Breve descrizione in 1-2 righe di cosa fa il modello.

## Descrizione del modello
Architettura, versione, paper di riferimento, autori.

## Usi previsti e limitazioni
Per cosa è stato pensato? Cosa NON sa fare? In quali lingue funziona?

## Come usarlo
```python
from transformers import pipeline
model = pipeline("text-classification", model="tuo-username/tuo-modello")
result = model("Testo di esempio")
```

## Dati di training
Su quali dataset è stato addestrato? Breve descrizione.

## Procedura di training
Epoch, batch size, learning rate, hardware usato, durata.

## Metriche di valutazione
Quali metriche sono state usate? Su quale split del dataset?

## Risultati di valutazione
| Dataset | Accuracy | F1 |
|---------|----------|----|
| MRPC    | 85.7%    | 89.9% |
```

### I metadati nell'intestazione (YAML front matter)

La sezione tra `---` viene letta dall'Hub per **categorizzare il modello** nei filtri di ricerca:

```yaml
---
language: it            # Lingua: it, en, fr, de, zh, ecc.
license: apache-2.0     # Licenza: mit, apache-2.0, cc-by-4.0, ecc.
datasets:               # Dataset usati per il training
  - oscar
  - wikipedia
tags:                   # Tag liberi per la ricerca
  - text-classification
  - sentiment-analysis
metrics:                # Metriche usate nella valutazione
  - accuracy
  - f1
---
```

**Esempio reale** dalla model card di `camembert-base`:

```yaml
---
language: fr
license: mit
datasets:
- oscar
---
```

Grazie a questi metadati, il modello appare nei risultati quando si filtra per "Francese" o "MIT license" sull'Hub.

### Sezioni della model card — dettaglio

| Sezione | Cosa mettere |
|---|---|
| **Model description** | Architettura, chi l'ha creato, paper di riferimento |
| **Intended uses & limitations** | Task supportati, lingue, domini; cosa NON funziona |
| **How to use** | Snippet di codice minimo per usare il modello |
| **Training data** | Nome dei dataset, breve descrizione |
| **Training procedure** | Epoch, LR, batch size, hardware |
| **Evaluation metrics** | Quali metriche, su quale dataset e split |
| **Evaluation results** | Tabella con i valori ottenuti |

### Esempi di model card ben fatte

- [`bert-base-cased`](https://huggingface.co/bert-base-cased) — modello iconico con documentazione completa
- [`gpt2`](https://huggingface.co/gpt2) — include bias noti e limitazioni
- [`distilbert-base-uncased`](https://huggingface.co/distilbert-base-uncased) — compatta ma esaustiva

### Quando usarlo

- Ogni volta che carichi un modello sull'Hub, anche per uso personale
- La model card è il biglietto da visita del modello: più è dettagliata, più sarà usato (e citato)

### Errori comuni

- ❌ Lasciare il `README.md` vuoto o con solo il nome del modello
- ❌ Non dichiarare i bias conosciuti → gli utenti potrebbero usarlo in contesti inappropriati
- ❌ Non indicare la licenza → per default il modello non può essere riusato legalmente
- ❌ Non inserire snippet di codice → chi scarica il modello non sa da dove cominciare

---

## 5. TL;DR — Riepilogo finale

**L'HuggingFace Hub in tre punti:**

1. **Trovare modelli** → usa `pipeline("task", model="nome-checkpoint")` o le classi `Auto*`. Leggi sempre la model card prima di usare in produzione.

2. **Caricare modelli** → scegli il metodo in base al tuo workflow:
   - `Trainer` → `push_to_hub=True` + `trainer.push_to_hub()`
   - Modello/tokenizer già pronti → `.push_to_hub("nome-repo")`
   - Controllo granulare → `huggingface_hub` library o git+git-lfs

3. **Documentare** → scrivi una model card nel `README.md` con metadati YAML (lingua, licenza, dataset), descrizione, snippet di codice e risultati di valutazione.

**Comandi essenziali da ricordare:**

```bash
# Login
huggingface-cli login

# Upload da terminale (git-lfs)
git lfs install
git clone https://huggingface.co/USERNAME/REPO
git add . && git commit -m "msg" && git push
```

```python
# Upload da Python (il più comodo)
training_args = TrainingArguments("nome-repo", push_to_hub=True)
trainer.push_to_hub()  # Alla fine del training

# Oppure direttamente
model.push_to_hub("nome-repo")
tokenizer.push_to_hub("nome-repo")
```
