# Capitolo 6 — Tokenizer: Costruzione e Algoritmi

> **Di cosa parla questo capitolo**: come funzionano i tokenizer "sotto il cofano", come addestrarne uno nuovo su un corpus custom, e come costruirne uno da zero con la libreria 🤗 Tokenizers.

---

## 📌 TL;DR

```
Addestrare nuovo tokenizer  → AutoTokenizer.train_new_from_iterator()
Fast tokenizer              → offsets, word_ids, char_to_token (velocità + mapping)
Pipeline token-classification → offset mapping per raggruppare entità
3 algoritmi principali:
  BPE       → merge dal basso (GPT-2, RoBERTa)
  WordPiece → merge con score (BERT, DistilBERT)
  Unigram   → potatura dall'alto (T5, XLNet via SentencePiece)
Costruire da zero → libreria tokenizers: Normalizer → PreTokenizer → Model → Trainer → PostProcessor → Decoder
```

---

## 🧠 Glossario rapido

> 💡 **Leggi prima di continuare**
>
> - **Token**: unità minima di testo che il modello "vede". Può essere una parola, parte di parola, o un carattere.
> - **Vocabolario**: insieme di tutti i token che il tokenizer conosce.
> - **Subword**: frammento di parola (es. `##ing`, `Ġthe`). Permette di gestire parole rare senza perdere significato.
> - **Tokenizer fast vs slow**: i fast sono scritti in Rust (molto più veloci su grandi dataset), i slow in Python puro.
> - **Offset mapping**: tracciamento della posizione originale di ogni token nel testo di partenza.
> - **Normalizzazione**: pulizia del testo prima della tokenizzazione (lowercase, rimozione accenti, ecc.).
> - **Pre-tokenizzazione**: suddivisione in parole prima di applicare l'algoritmo subword.

---

## 1. Addestrare un Nuovo Tokenizer

### Quando serve
Quando il modello che vuoi addestrare opera su un dominio molto diverso dall'inglese generico (es. codice Python, giapponese, dominio medico). Il tokenizer esistente sarebbe inefficiente.

> ⚠️ **Addestrare un tokenizer ≠ addestrare un modello!**
> Il training del tokenizer è un processo statistico (deterministico). Il training del modello usa gradient descent (stocastico).

### Assemblare un corpus

```python
from datasets import load_dataset

# Carica il dataset CodeSearchNet (codice Python da GitHub)
raw_datasets = load_dataset("code_search_net", "python")

# Vedi la struttura
raw_datasets["train"]
# → Dataset con 412.178 funzioni Python
```

### Creare un generatore (soluzione memory-efficient)

```python
# ❌ MALE: carica tutto in memoria
# training_corpus = [raw_datasets["train"][i: i + 1000]["whole_func_string"]
#                    for i in range(0, len(raw_datasets["train"]), 1000)]

# ✅ BENE: generatore → carica 1000 testi alla volta, libera la memoria
def get_training_corpus():
    dataset = raw_datasets["train"]
    for start_idx in range(0, len(dataset), 1000):
        samples = dataset[start_idx : start_idx + 1000]
        yield samples["whole_func_string"]
        # yield: restituisce un batch e "mette in pausa" la funzione
        # alla prossima iterazione riparte da dove si era fermata

training_corpus = get_training_corpus()
```

**Perché il generatore?** Con `yield` Python non carica mai l'intero dataset in RAM: produce un batch, lo usa, lo libera. Essenziale per dataset da GB.

### Addestrare il tokenizer

```python
from transformers import AutoTokenizer

# Carica il tokenizer vecchio come "template" (struttura, token speciali, algoritmo)
old_tokenizer = AutoTokenizer.from_pretrained("gpt2")

# Addestra un nuovo tokenizer con STESSO algoritmo ma vocabolario diverso
# 52000 = dimensione del vocabolario finale
tokenizer = old_tokenizer.train_new_from_iterator(training_corpus, 52000)
```

**Confronto tokenizzazione su codice Python:**

```python
example = '''def add_numbers(a, b):
    """Add the two numbers `a` and `b`."""
    return a + b'''

# GPT-2 originale (addestrato su inglese generico)
old_tokenizer.tokenize(example)
# → 36 token, spazi separati, underscore spezzato

# Nuovo tokenizer (addestrato su Python)
tokenizer.tokenize(example)
# → 27 token, indentazione come token unico (ĊĠĠĠ), underscore gestito meglio
```

Il nuovo tokenizer "impara" i pattern del Python: indentazione, `"""`, `_` nei nomi.

### Salvare e caricare

```python
# Salva in locale
tokenizer.save_pretrained("code-search-net-tokenizer")

# Carica da locale
tokenizer = AutoTokenizer.from_pretrained("code-search-net-tokenizer")

# Pubblica sull'Hub
tokenizer.push_to_hub("code-search-net-tokenizer")

# Carica dall'Hub (chiunque può usarlo)
tokenizer = AutoTokenizer.from_pretrained("tuo-username/code-search-net-tokenizer")
```

---

## 2. Poteri Speciali dei Fast Tokenizer

### Fast vs Slow

| | Fast tokenizer | Slow tokenizer |
|---|---|---|
| Libreria | 🤗 Tokenizers (Rust) | Python puro |
| `batched=True` | ~10s | ~5 minuti |
| Offset mapping | ✅ | ❌ |

### BatchEncoding — non è un semplice dict

```python
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")
example = "My name is Sylvain and I work at Hugging Face in Brooklyn."

encoding = tokenizer(example)
# encoding è un oggetto BatchEncoding con metodi aggiuntivi

tokenizer.is_fast   # → True
encoding.is_fast    # → True
```

### Offset Mapping — mappa token ↔ posizioni originali

```python
# Metodi disponibili sui fast tokenizer:

# Lista dei token (senza convertire da ID)
encoding.tokens()
# → ['[CLS]', 'My', 'name', 'is', 'S', '##yl', '##va', '##in', ...]

# A quale parola appartiene ogni token?
encoding.word_ids()
# → [None, 0, 1, 2, 3, 3, 3, 3, 4, 5, 6, 7, 8, 8, 9, 10, 11, 12, None]
# None = token speciali ([CLS], [SEP])
# 3,3,3,3 = i 4 token di "Sylvain" appartengono tutti alla parola 3

# Da word ID a caratteri nel testo originale
start, end = encoding.word_to_chars(3)
example[start:end]  # → 'Sylvain'

# Mappa anche al contrario: carattere → token, token → caratteri
encoding.char_to_token(11)  # → indice del token alla posizione 11
encoding.token_to_chars(5)  # → (start, end) del token 5 nel testo originale
```

**Perché è utile?**
- NER: sapere dove inizia/finisce un'entità nel testo originale
- QA: trovare la risposta esatta nel testo
- Raggruppare token dello stesso subword senza dipendere dal prefisso `##`

### Pipeline Token-Classification (NER) — sotto il cofano

```python
from transformers import pipeline

# Uso diretto della pipeline
token_classifier = pipeline("token-classification")
token_classifier("My name is Sylvain and I work at Hugging Face in Brooklyn.")
# → lista di token con entity, score, start, end

# Raggruppamento entità multi-token
token_classifier = pipeline("token-classification", aggregation_strategy="simple")
# → [{'entity_group': 'PER', 'word': 'Sylvain', 'start': 11, 'end': 18}, ...]
```

**Come funziona sotto il cofano:**

```python
from transformers import AutoTokenizer, AutoModelForTokenClassification
import torch

model_checkpoint = "dbmdz/bert-large-cased-finetuned-conll03-english"
tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
model = AutoModelForTokenClassification.from_pretrained(model_checkpoint)

example = "My name is Sylvain and I work at Hugging Face in Brooklyn."

# 1. Tokenizza con offset mapping
inputs_with_offsets = tokenizer(example, return_tensors="pt", return_offsets_mapping=True)
# return_offsets_mapping=True → aggiunge lista di (start, end) per ogni token

# 2. Passa al modello
outputs = model(**{k: v for k, v in inputs_with_offsets.items() if k != "offset_mapping"})
# outputs.logits ha shape [1, 19, 9] → 1 frase, 19 token, 9 classi

# 3. Converti logit in probabilità
probabilities = torch.nn.functional.softmax(outputs.logits, dim=-1)[0].tolist()
predictions = outputs.logits.argmax(dim=-1)[0].tolist()

# 4. Raggruppa entità usando gli offset
offsets = inputs_with_offsets["offset_mapping"][0]
tokens = inputs_with_offsets.tokens()

results = []
for idx, pred in enumerate(predictions):
    label = model.config.id2label[pred]
    if label != "O":  # "O" = Outside = non è un'entità
        start, end = offsets[idx]
        results.append({
            "entity": label,
            "score": probabilities[idx][pred],
            "word": tokens[idx],
            "start": start,
            "end": end,
        })
```

---

## 3. Normalizzazione e Pre-tokenizzazione

### Pipeline di tokenizzazione (4 fasi)

```
Testo grezzo
    ↓ Normalizzazione   (pulizia: lowercase, rimozione accenti, unicode)
    ↓ Pre-tokenizzazione (suddivisione in parole)
    ↓ Modello           (BPE / WordPiece / Unigram → subtoken)
    ↓ Post-processing   (aggiunta token speciali: [CLS], [SEP])
```

### Esplorare la normalizzazione

```python
from transformers import AutoTokenizer

# bert-base-uncased: lowercase + rimozione accenti
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
tokenizer.backend_tokenizer.normalizer.normalize_str("Héllò hôw are ü?")
# → 'hello how are u?'
```

### Esplorare la pre-tokenizzazione

```python
# BERT: divide su spazi e punteggiatura
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
tokenizer.backend_tokenizer.pre_tokenizer.pre_tokenize_str("Hello, how are  you?")
# → [('Hello', (0,5)), (',', (5,6)), ('how', (7,10)), ('are', (11,14)), ('you', (16,19)), ('?', (19,20))]
# Nota: il doppio spazio viene ignorato ma l'offset salta correttamente

# GPT-2: mantiene gli spazi come simbolo Ġ
tokenizer = AutoTokenizer.from_pretrained("gpt2")
tokenizer.backend_tokenizer.pre_tokenizer.pre_tokenize_str("Hello, how are  you?")
# → [('Hello', (0,5)), (',', (5,6)), ('Ġhow', (6,10)), ...('Ġ', (14,15)), ('Ġyou', (15,19))]
# Ġ = spazio prefisso, permette ricostruzione perfetta del testo

# T5 (SentencePiece): usa _ al posto di spazi, non divide su punteggiatura
tokenizer = AutoTokenizer.from_pretrained("t5-small")
tokenizer.backend_tokenizer.pre_tokenizer.pre_tokenize_str("Hello, how are  you?")
# → [('▁Hello,', (0,6)), ('▁how', (7,10)), ('▁are', (11,14)), ('▁you?', (16,20))]
```

### SentencePiece
Algoritmo usato da T5, mBART, XLNet. Tratta il testo come sequenza di caratteri Unicode (senza pre-tokenizzazione obbligatoria), sostituisce gli spazi con `▁`. Utile per lingue senza spazi (cinese, giapponese). La tokenizzazione è **reversibile**: basta concatenare i token e sostituire `▁` con spazio.

---

## 4. Algoritmi di Tokenizzazione Subword

### Confronto rapido

| | BPE | WordPiece | Unigram |
|---|---|---|---|
| Usato da | GPT-2, RoBERTa, BART | BERT, DistilBERT | T5, XLNet, ALBERT |
| Training | Parte piccolo, cresce | Parte piccolo, cresce | Parte grande, pota |
| Criterio merge | Coppia più frequente | Coppia con score migliore | Rimuove chi aumenta meno la loss |
| Impara | Regole di merge + vocabolario | Solo vocabolario | Vocabolario con score |

---

## 5. BPE (Byte-Pair Encoding)

### Idea
Parte dai caratteri singoli e unisce iterativamente le coppie più frequenti fino a raggiungere la dimensione di vocabolario desiderata.

### Esempio manuale

```
Corpus: "hug"×10, "pug"×5, "pun"×12, "bun"×4, "hugs"×5
Vocabolario iniziale: [b, g, h, n, p, s, u]

Step 1: coppia più frequente → ("u","g") appare 20 volte → merge → "ug"
Step 2: ("u","n") appare 16 volte → merge → "un"
Step 3: ("h","ug") appare 15 volte → merge → "hug"
```

### Implementazione BPE da zero

```python
from collections import defaultdict
from transformers import AutoTokenizer

corpus = [
    "This is the Hugging Face Course.",
    "This chapter is about tokenization.",
    "This section shows several tokenizer algorithms.",
    "Hopefully, you will be able to understand how they are trained and generate tokens.",
]

# 1. Pre-tokenizza e conta frequenze parole
tokenizer = AutoTokenizer.from_pretrained("gpt2")
word_freqs = defaultdict(int)
for text in corpus:
    words_with_offsets = tokenizer.backend_tokenizer.pre_tokenizer.pre_tokenize_str(text)
    for word, offset in words_with_offsets:
        word_freqs[word] += 1
# → {'This': 3, 'Ġis': 2, '.': 4, ...}

# 2. Alfabeto iniziale (tutti i caratteri unici)
alphabet = sorted(set(c for word in word_freqs for c in word))
vocab = ["<|endoftext|>"] + alphabet.copy()

# 3. Suddividi ogni parola in caratteri
splits = {word: list(word) for word in word_freqs}

# 4. Funzione per calcolare frequenza delle coppie
def compute_pair_freqs(splits):
    pair_freqs = defaultdict(int)
    for word, freq in word_freqs.items():
        split = splits[word]
        for i in range(len(split) - 1):
            pair_freqs[(split[i], split[i+1])] += freq
    return pair_freqs

# 5. Funzione per applicare un merge
def merge_pair(a, b, splits):
    for word in word_freqs:
        split = splits[word]
        i = 0
        while i < len(split) - 1:
            if split[i] == a and split[i+1] == b:
                split = split[:i] + [a+b] + split[i+2:]
            else:
                i += 1
        splits[word] = split
    return splits

# 6. Training loop: ripeti finché non raggiungi vocab_size
merges = {}
vocab_size = 50

while len(vocab) < vocab_size:
    pair_freqs = compute_pair_freqs(splits)
    best_pair = max(pair_freqs, key=pair_freqs.get)
    splits = merge_pair(*best_pair, splits)
    merges[best_pair] = best_pair[0] + best_pair[1]
    vocab.append(best_pair[0] + best_pair[1])

# 7. Tokenizza nuovo testo
def tokenize_bpe(text):
    pre_tokenized = [w for w, _ in tokenizer.backend_tokenizer.pre_tokenizer.pre_tokenize_str(text)]
    splits_text = [list(word) for word in pre_tokenized]
    for pair, merge in merges.items():
        for idx, split in enumerate(splits_text):
            i = 0
            while i < len(split) - 1:
                if split[i] == pair[0] and split[i+1] == pair[1]:
                    split = split[:i] + [merge] + split[i+2:]
                else:
                    i += 1
            splits_text[idx] = split
    return sum(splits_text, [])
```

---

## 6. WordPiece

### Idea
Come BPE ma il criterio di merge è diverso: invece della coppia più frequente, usa uno **score** che premia la fusione di token raramente separati.

```
score = freq(coppia AB) / (freq(A) × freq(B))
```

Così preferisce fondere `("hu", "##gging")` (entrambi rari) rispetto a `("un", "##able")` (entrambi molto comuni).

### Differenza chiave dalla tokenizzazione
- **BPE**: applica le regole di merge in ordine
- **WordPiece**: salva solo il vocabolario finale e tokenizza trovando il **subword più lungo** che inizia da sinistra

```python
# Parola "bugs" con WordPiece:
# "b" → più lungo iniziale in vocab
# "##ugs" → "##u" è in vocab
# "##ugs" → "##gs" è in vocab
# → ['b', '##u', '##gs']

# Se nessun subword trovato → l'intera parola diventa [UNK]
```

### Implementazione WordPiece (schema)

```python
# Differenza principale nel compute_pair_scores:
def compute_pair_scores(splits):
    letter_freqs = defaultdict(int)
    pair_freqs = defaultdict(int)
    for word, freq in word_freqs.items():
        split = splits[word]
        for i in range(len(split) - 1):
            pair = (split[i], split[i+1])
            letter_freqs[split[i]] += freq
            pair_freqs[pair] += freq
        letter_freqs[split[-1]] += freq

    scores = {
        pair: freq / (letter_freqs[pair[0]] * letter_freqs[pair[1]])
        for pair, freq in pair_freqs.items()
    }
    return scores

# Tokenizzazione: trova il subword più lungo a sinistra
def encode_word_wp(word):
    tokens = []
    while len(word) > 0:
        i = len(word)
        while i > 0 and word[:i] not in vocab:
            i -= 1
        if i == 0:
            return ["[UNK]"]  # nessun subword trovato → tutto [UNK]
        tokens.append(word[:i])
        word = word[i:]
        if len(word) > 0:
            word = f"##{word}"  # i subword interni hanno prefisso ##
    return tokens
```

---

## 7. Unigram

### Idea
Parte da un **vocabolario grande** e **rimuove** iterativamente i token meno importanti finché non raggiunge la dimensione desiderata. "Meno importante" = la sua rimozione aumenta meno la loss del modello.

### Tokenizzazione con algoritmo di Viterbi
Trova la suddivisione con **probabilità massima** (prodotto delle probabilità di ogni token).

```
"pug" possibili suddivisioni:
- ["p","u","g"]  → prob 0.000389
- ["p","ug"]     → prob 0.002268  ← SCELTA
- ["pu","g"]     → prob 0.002268
```

L'algoritmo di Viterbi costruisce un grafo e trova il percorso ottimale in O(n²).

### Implementazione Unigram (schema)

```python
from math import log

# Probabilità = -log(frequenza/totale) per stabilità numerica
total_sum = sum(freq for _, freq in token_freqs.items())
model = {token: -log(freq / total_sum) for token, freq in token_freqs.items()}

# Viterbi: segmentazione ottimale di una parola
def encode_word_unigram(word, model):
    # best_segmentations[i] = miglior segmentazione fino alla posizione i
    best_segmentations = [{"start": 0, "score": 1}] + [
        {"start": None, "score": None} for _ in range(len(word))
    ]
    for start_idx in range(len(word)):
        best_score_at_start = best_segmentations[start_idx]["score"]
        for end_idx in range(start_idx + 1, len(word) + 1):
            token = word[start_idx:end_idx]
            if token in model and best_score_at_start is not None:
                score = model[token] + best_score_at_start
                if (best_segmentations[end_idx]["score"] is None or
                    best_segmentations[end_idx]["score"] > score):
                    best_segmentations[end_idx] = {"start": start_idx, "score": score}
    # ricostruisci il percorso
    ...

# Training: rimuovi il 10% dei token con loss più bassa ad ogni step
percent_to_remove = 0.1
while len(model) > target_vocab_size:
    scores = compute_scores(model)
    sorted_scores = sorted(scores.items(), key=lambda x: x[1])
    for i in range(int(len(model) * percent_to_remove)):
        token_freqs.pop(sorted_scores[i][0])
    # ricalcola probabilità
    total_sum = sum(freq for _, freq in token_freqs.items())
    model = {token: -log(freq / total_sum) for token, freq in token_freqs.items()}
```

---

## 8. Costruire un Tokenizer da Zero (🤗 Tokenizers)

La libreria fornisce blocchi modulari per assembolare qualsiasi tokenizer.

### Struttura generale

```python
from tokenizers import (
    decoders, models, normalizers,
    pre_tokenizers, processors, trainers, Tokenizer
)

# 1. Scegli il modello (WordPiece / BPE / Unigram)
tokenizer = Tokenizer(models.WordPiece(unk_token="[UNK]"))

# 2. Normalizzazione
tokenizer.normalizer = normalizers.BertNormalizer(lowercase=True)
# oppure componi manualmente:
tokenizer.normalizer = normalizers.Sequence([
    normalizers.NFD(),        # Unicode NFD
    normalizers.Lowercase(),  # tutto minuscolo
    normalizers.StripAccents() # rimuovi accenti
])

# 3. Pre-tokenizzazione
tokenizer.pre_tokenizer = pre_tokenizers.BertPreTokenizer()
# oppure:
tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()  # solo spazi/punteggiatura
# oppure combina:
tokenizer.pre_tokenizer = pre_tokenizers.Sequence([
    pre_tokenizers.WhitespaceSplit(),
    pre_tokenizers.Punctuation()
])

# 4. Trainer
special_tokens = ["[UNK]", "[PAD]", "[CLS]", "[SEP]", "[MASK]"]
trainer = trainers.WordPieceTrainer(vocab_size=25000, special_tokens=special_tokens)

# 5. Addestra
tokenizer.train_from_iterator(get_training_corpus(), trainer=trainer)
# oppure da file:
# tokenizer.train(["corpus.txt"], trainer=trainer)

# 6. Post-processing (aggiunta token speciali)
cls_token_id = tokenizer.token_to_id("[CLS]")  # → 2
sep_token_id = tokenizer.token_to_id("[SEP]")  # → 3

tokenizer.post_processor = processors.TemplateProcessing(
    single="[CLS]:0 $A:0 [SEP]:0",          # frase singola
    pair="[CLS]:0 $A:0 [SEP]:0 $B:1 [SEP]:1", # coppia di frasi
    special_tokens=[("[CLS]", cls_token_id), ("[SEP]", sep_token_id)]
)
# $A = prima frase, $B = seconda frase
# :0 / :1 = token type ID

# 7. Decoder
tokenizer.decoder = decoders.WordPiece(prefix="##")

# Test
encoding = tokenizer.encode("Let's test this tokenizer.")
encoding.tokens  # → ['[CLS]', 'let', "'", 's', 'test', 'this', 'tok', '##eni', '##zer', '.', '[SEP]']

# Decode
tokenizer.decode(encoding.ids)  # → "let's test this tokenizer."
```

### Salvare e usare con 🤗 Transformers

```python
# Salva come JSON
tokenizer.save("tokenizer.json")
new_tokenizer = Tokenizer.from_file("tokenizer.json")

# Wrappa per usarlo con Transformers
from transformers import PreTrainedTokenizerFast, BertTokenizerFast

# Modo generico (per tokenizer custom)
wrapped = PreTrainedTokenizerFast(
    tokenizer_object=tokenizer,
    unk_token="[UNK]",
    pad_token="[PAD]",
    cls_token="[CLS]",
    sep_token="[SEP]",
    mask_token="[MASK]",
)

# Modo specifico (se il tuo tokenizer corrisponde a un modello esistente)
wrapped = BertTokenizerFast(tokenizer_object=tokenizer)
```

### BPE tokenizer (GPT-2 style)

```python
tokenizer = Tokenizer(models.BPE())  # niente unk_token: usa byte-level BPE

# GPT-2 non normalizza
tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
# ByteLevel: converte ogni carattere in byte → niente [UNK] mai

trainer = trainers.BpeTrainer(vocab_size=25000, special_tokens=["<|endoftext|>"])
tokenizer.train_from_iterator(get_training_corpus(), trainer=trainer)

tokenizer.post_processor = processors.ByteLevel(trim_offsets=False)
# trim_offsets=False: lo spazio prima della parola fa parte del token (come in GPT-2)

tokenizer.decoder = decoders.ByteLevel()

# Wrappa
from transformers import GPT2TokenizerFast
wrapped = GPT2TokenizerFast(tokenizer_object=tokenizer)
```

### Unigram tokenizer (XLNet style)

```python
from tokenizers import Regex

tokenizer = Tokenizer(models.Unigram())

# Normalizzazione XLNet (via SentencePiece)
tokenizer.normalizer = normalizers.Sequence([
    normalizers.Replace("``", '"'),
    normalizers.Replace("''", '"'),
    normalizers.NFKD(),
    normalizers.StripAccents(),
    normalizers.Replace(Regex(" {2,}"), " "),  # spazi multipli → uno solo
])

# Pre-tokenizer SentencePiece: usa ▁ per gli spazi
tokenizer.pre_tokenizer = pre_tokenizers.Metaspace()

special_tokens = ["<pad>", "<unk>", "<cls>", "<sep>", "<mask>", "<s>", "</s>"]
trainer = trainers.UnigramTrainer(
    vocab_size=25000,
    special_tokens=special_tokens,
    unk_token="<unk>"
)
tokenizer.train_from_iterator(get_training_corpus(), trainer=trainer)

# XLNet mette <cls> alla FINE con type_id=2 (padding a sinistra!)
tokenizer.post_processor = processors.TemplateProcessing(
    single="$A:0 <sep>:0 <cls>:2",
    pair="$A:0 <sep>:0 $B:1 <sep>:1 <cls>:2",
    special_tokens=[("<sep>", sep_token_id), ("<cls>", cls_token_id)]
)

tokenizer.decoder = decoders.Metaspace()  # converte ▁ → spazio

# Wrappa
from transformers import XLNetTokenizerFast
wrapped = XLNetTokenizerFast(tokenizer_object=tokenizer)
```

---

## 🕐 Quando usarlo

| Scenario | Azione |
|----------|--------|
| Dominio diverso dall'inglese (codice, bio, altro) | `train_new_from_iterator()` su tokenizer esistente |
| Linguaggio senza spazi (cinese, giapponese) | SentencePiece / Unigram |
| Modello BERT-like | WordPiece |
| Modello GPT-like | BPE (byte-level) |
| Vuoi NER o QA preciso a livello carattere | Fast tokenizer + offset mapping |
| Vuoi tokenizer completamente custom | 🤗 Tokenizers da zero |

---

## ⚠️ Errori comuni

1. **"Il tokenizer esistente è abbastanza"** — Non sempre! Su codice Python, il tokenizer GPT-2 generico usa 36 token per una funzione breve; uno addestrato su Python ne usa 27.

2. **Usare lista invece di generatore** — Per corpus grandi esaurisci la RAM. Usa sempre `yield` o `(... for ...)` con `def`.

3. **`train_new_from_iterator()` fallisce** — Funziona solo con fast tokenizer. Controlla `tokenizer.is_fast`.

4. **Dimenticare i special tokens nel Trainer** — Se non li passi a `WordPieceTrainer` / `BpeTrainer`, non vengono aggiunti al vocabolario.

5. **Post-processor sbagliato** — BERT vuole `[CLS]...[SEP]`, XLNet vuole `...[SEP][CLS]` con padding a sinistra. Configura il `TemplateProcessing` correttamente.

6. **WordPiece vs BPE: differenza nella tokenizzazione** — WordPiece usa il subword più lungo da sinistra; BPE applica le regole di merge in ordine. Lo stesso vocabolario produce output diversi!

7. **Carattere non in vocabolario** — In WordPiece l'intera parola diventa `[UNK]`. In BPE byte-level non succede mai (256 byte coprono tutto).
