# Capitolo 12 — Reinforcement Learning, GRPO e DeepSeek R1

> **Di cosa parla questo capitolo:**
> Come insegnare a un LLM a *ragionare* — non solo a generare testo fluente. Vedremo cos'è il Reinforcement Learning, perché è stato usato per creare modelli come ChatGPT e DeepSeek R1, e come implementarlo in pratica con l'algoritmo GRPO.

---

## 📌 TL;DR

| Concetto | In parole semplici |
|----------|-------------------|
| **Reinforcement Learning (RL)** | Addestrare un modello tramite premi e punizioni, come addestrare un cane |
| **RLHF** | RL dove i premi vengono da preferenze umane |
| **GRPO** | Algoritmo RL che genera più soluzioni, le confronta tra loro e impara da quelle migliori |
| **DeepSeek R1** | Modello che ha dimostrato che il puro RL può far emergere il ragionamento |
| **Reward function** | La funzione che decide se una risposta è "buona" o "cattiva" |
| **GRPOTrainer** | Classe TRL per fare GRPO, analoga a SFTTrainer |

---

## 1. Cos'è il Reinforcement Learning (RL)?

### L'intuizione di base: addestrare un cane

Immagina di voler insegnare al tuo cane a sedersi. Come funziona?

```
Tu dici: "Siediti!"
  ↓
Il cane prova diverse azioni:
  - Si siede ✅ → gli dai un premio 🦴 (reward positivo)
  - Non fa niente ❌ → non succede nulla (reward 0)
  - Si alza ❌ → dici "no" (reward negativo)
  ↓
Col tempo il cane impara: "sedersi → biscotto!"
```

**Il Reinforcement Learning funziona esattamente così**, ma al posto del cane c'è un modello linguistico.

### I 5 concetti chiave di RL

#### 🤖 Agente (Agent)
Chi impara e prende decisioni. Nel RL per LLM, **l'agente è il modello** stesso.

*Analogia: il cane nell'esempio sopra.*

#### 🌍 Ambiente (Environment)
Il contesto in cui l'agente opera e che gli fornisce feedback. Per un LLM, l'ambiente è astratto: può essere un utente, un sistema di valutazione automatica, o un problema da risolvere.

*Analogia: tu e la casa nell'esempio del cane.*

#### ⚡ Azione (Action)
Le scelte che l'agente può fare. Per un LLM, le azioni sono le parole (token) che genera, o la risposta che produce.

*Analogia: "sedersi", "abbaiare", "saltare" per il cane.*

#### 🏆 Reward (Ricompensa)
Un numero che dice all'agente quanto è stata buona la sua azione. Può essere positivo (bravissimo!) o negativo (sbagliato!).

Per un LLM, il reward può essere:
- `+1` se la risposta è corretta
- `-1` se la risposta contiene contenuti dannosi
- `0` se la risposta è neutra

*Analogia: il biscotto (reward positivo) o il "no!" (reward negativo).*

#### 📋 Policy (Strategia)
Le regole interne che l'agente usa per decidere cosa fare. È ciò che stiamo cercando di migliorare con il training. All'inizio è casuale, alla fine è ottimizzata.

*Analogia: la "comprensione" del cane di cosa fare quando sente "siediti".*

### Il ciclo di apprendimento RL

```
┌─────────────────────────────────────────────────────┐
│                                                     │
│  Stato attuale → AGENTE → sceglie un'AZIONE         │
│       ↑                         ↓                   │
│  nuovo stato ← AMBIENTE ← riceve REWARD             │
│                                                     │
│  L'agente aggiorna la sua POLICY basandosi          │
│  sul reward ricevuto, poi ricomincia.               │
└─────────────────────────────────────────────────────┘
```

**Esempio concreto con un LLM:**

```
Passo 1 — Osservazione: il modello riceve la domanda "Qual è la capitale della Francia?"
Passo 2 — Azione: il modello genera "La capitale della Francia è Parigi."
Passo 3 — Reward: un sistema automatico controlla → "Parigi" è giusto → reward = +1
Passo 4 — Apprendimento: il modello capisce che questo tipo di risposta è buona
Passo 5 — Iterazione: si ripete con la domanda successiva
```

### Perché RL per gli LLM?

Un LLM addestrato solo con pre-training (predire la parola successiva) impara a scrivere testo fluente, ma non necessariamente:
- **Utile**: risponde alla domanda vera?
- **Onesto**: è accurato o inventa?
- **Sicuro**: evita contenuti dannosi?

Il RL ci permette di guidare il modello verso questi obiettivi che sono difficili da catturare con il semplice pre-training.

> 💡 **Metafora:** Il pre-training insegna al modello a "parlare bene l'italiano". Il RL gli insegna a essere un buon assistente, onesto e utile. Sono due skill diverse!

---

## 2. RLHF — Reinforcement Learning from Human Feedback

### Il problema: come costruire un reward per "risposta buona"?

Definire matematicamente cosa significa "risposta buona" è difficilissimo. Non puoi scrivere una regola che funzioni per ogni domanda.

**RLHF risolve il problema così:**

```
Fase 1 — Raccolta preferenze umane:
  - Mostri a persone reali 2 risposte dello stesso modello
  - Chiedono: "Quale preferisci?"
  - Raccogli migliaia di questi confronti

Fase 2 — Addestri un Reward Model:
  - Un modello separato impara a prevedere le preferenze umane
  - Input: una risposta → Output: un punteggio (quanto piace agli umani)

Fase 3 — Fine-tuning con RL:
  - Il tuo LLM genera risposte
  - Il Reward Model le valuta (dà un punteggio)
  - Il LLM impara a generare risposte con punteggio alto
```

**RLHF è stato usato per creare:** GPT-4 (OpenAI), Gemini (Google), Claude (Anthropic), e molti altri.

---

## 3. I 3 algoritmi principali di RLHF

### PPO (Proximal Policy Optimization)
Il primo algoritmo molto efficace per RLHF. Aggiorna la policy passo per passo, limitando quanto può cambiare ad ogni step per evitare aggiornamenti destabilizzanti. **Richiede un reward model separato.**

### DPO (Direct Preference Optimization)
Approccio più semplice: invece di usare RL vero e proprio, trasforma il problema in classificazione tra risposta "scelta" e risposta "rifiutata". **Elimina il reward model**, ma usa dati di preferenza (coppie scelta/rifiutata).

### GRPO (Group Relative Policy Optimization) ← il focus di questo capitolo
Il più recente e quello che ha reso famoso DeepSeek R1. **Non richiede necessariamente un reward model**: può usare qualsiasi funzione che valuti la qualità di una risposta (es. un solver matematico, un checker di formato, ecc.).

| | PPO | DPO | GRPO |
|--|-----|-----|------|
| Reward model separato? | Sì | No | No (opzionale) |
| Tipo di dati | Qualsiasi | Coppie preferenza | Qualsiasi prompt |
| Reward | Da reward model | Implicito nei dati | Da qualsiasi funzione |
| Stabilità | Media | Alta | Alta |
| Usato in | GPT-3.5 | Llama 2 Chat | DeepSeek R1 |

---

## 4. GRPO — Come funziona nel dettaglio

> **Idea centrale:** invece di valutare una risposta in isolamento, genera MOLTE risposte per lo stesso prompt, confrontale tra loro, e impara da quelle relativamente migliori.

### Passo 1 — Group Formation (Formazione del Gruppo)

Per ogni prompt di training, il modello genera **N risposte diverse** (tipicamente 4, 8 o 16):

```
Prompt: "Se ho 3 mele e 2 arance, quanti frutti ho in totale?"

Risposta 1: "Ho bisogno di sommare 3 + 2. Il totale è 5."     → reward = 1.0
Risposta 2: "Devo fare 3 + 2 = 5 frutti."                     → reward = 1.0
Risposta 3: "La risposta è 6."                                 → reward = 0.0
Risposta 4: "Non sono sicuro, forse 5?"                        → reward = 0.5
```

Queste N risposte formano un **gruppo**.

*Analogia: come avere 4 studenti che risolvono lo stesso problema. Confronti le loro soluzioni.*

### Passo 2 — Preference Learning (Calcolo del Vantaggio Relativo)

Invece di usare i reward assoluti, GRPO **normalizza** i reward all'interno del gruppo:

```python
# Formula del vantaggio normalizzato:
advantage = (reward - mean(group_rewards)) / std(group_rewards)

# Esempio con i reward dell'esempio sopra:
rewards = [1.0, 1.0, 0.0, 0.5]
mean = 0.625
std  = 0.43

# Vantaggi calcolati:
advantage_1 = (1.0 - 0.625) / 0.43 = +0.87  ← sopra la media del gruppo
advantage_2 = (1.0 - 0.625) / 0.43 = +0.87  ← sopra la media del gruppo
advantage_3 = (0.0 - 0.625) / 0.43 = -1.45  ← molto sotto la media
advantage_4 = (0.5 - 0.625) / 0.43 = -0.29  ← leggermente sotto la media
```

**Perché normalizzare?** Perché il modello non deve imparare solo "questa risposta è buona" in assoluto, ma "questa risposta è **meglio delle altre** nel gruppo". È come dare voti relativi invece di assoluti.

*Analogia: come la valutazione "in curva" all'università — il 7 ha valore diverso se la media della classe è 6 o se è 8.*

### Passo 3 — Optimization (Aggiornamento dei Pesi)

Il modello aggiorna i suoi pesi per:
1. **Aumentare** la probabilità di generare risposte con vantaggio positivo
2. **Diminuire** la probabilità di generare risposte con vantaggio negativo
3. **Non cambiare troppo** alla volta (penalty KL divergence — vedi sotto)

```
# Pseudocodice GRPO semplificato:

Per ogni iterazione di training:
  1. Salva una copia del modello attuale (reference policy)
  2. Per ogni prompt nel batch:
     a. Genera N risposte con il modello attuale
     b. Calcola il reward per ciascuna risposta
     c. Normalizza i reward → ottieni i "vantaggi"
     d. Aggiorna il modello per favorire le risposte con vantaggio alto
     e. Applica un "freno" (KL penalty) per non cambiare troppo
```

### La KL Divergence Penalty — il freno di sicurezza

La KL divergence misura quanto il modello aggiornato si è allontanato dal modello originale. È un **freno di sicurezza**:

- Senza freno: il modello potrebbe cambiare troppo e "dimenticare" tutto quello che sapeva
- Con freno troppo forte: il modello non impara nulla di nuovo
- Con freno bilanciato: il modello migliora gradualmente senza perdere le sue capacità

> 💡 **Metafora:** È come imparare una nuova lingua. Se studi troppo intensamente e smetti di usare la lingua madre, potresti dimenticarla (no freno). Se hai paura di dimenticare e non studi mai la nuova lingua, non impari (freno troppo forte). Il giusto equilibrio è studiare la nuova lingua mantenendo anche l'altra.

---

## 5. DeepSeek R1 — La svolta

### Il problema che volevano risolvere

Prima di DeepSeek R1, tutti pensavano che per creare un LLM che ragiona bene servisse:
1. Un enorme dataset di esempi di ragionamento (SFT)
2. Poi RL sopra

DeepSeek ha provato a rispondere alla domanda: **si può far emergere il ragionamento con RL puro, senza SFT?**

### L'"Aha Moment" — il momento eureka

Durante il training di **R1-Zero** (il modello addestrato con solo RL, senza SFT), è emerso qualcosa di straordinario: il modello ha sviluppato autonomamente la capacità di **auto-correggersi** durante la soluzione di un problema.

```
Senza essere programmato esplicitamente, il modello ha imparato a:

1. Fare un tentativo iniziale: "Questo pezzo del puzzle dovrebbe andare qui per il colore"
2. Riconoscere l'errore: "Aspetta, però la forma non combacia..."
3. Correggersi: "Ah, in realtà appartiene là"
4. Spiegare: "Perché sia il colore che la forma combaciano in quella posizione"
```

Questo comportamento — chiamato **"Aha Moment"** — è emerso **spontaneamente** dal training RL, non era stato insegnato esplicitamente. È come se il modello avesse imparato a "pensare ad alta voce".

### Il processo di training in 4 fasi

```
FASE 1: Cold Start (Avvio a Freddo)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Modello base: DeepSeek-V3-Base
↓
Fine-tuning su POCHI esempi di alta qualità da R1-Zero
Obiettivo: dare al modello una base di leggibilità e qualità

FASE 2: Reasoning RL Phase (RL per il Ragionamento)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
RL su task VERIFICABILI: matematica, coding, logica
Reward: reward = 1 se risposta corretta, 0 altrimenti
(nessun reward model — si verifica direttamente!)

FASE 3: Rejection Sampling (Controllo Qualità)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Il modello genera molti esempi
DeepSeek-V3 li valuta e filtra i migliori
I migliori vengono usati per SFT (supervised fine-tuning)

FASE 4: Diverse RL Phase (RL Diversificato)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
RL su task diversi: reasoning + conversazione + sicurezza
Reward ibrido: rule-based per task oggettivi + LLM judge per task soggettivi
```

### R1-Zero vs R1 — le differenze

| | DeepSeek-R1-Zero | DeepSeek-R1 |
|--|-----------------|-------------|
| Training | Solo RL puro | Fasi multiple (SFT + RL) |
| SFT | Nessuno | Sì (fasi 1 e 3) |
| Ragionamento | Emerge spontaneamente | Potenziato e raffinato |
| Leggibilità | Scarsa (a volte cambia lingua) | Buona |
| AIME 2024 | 71.0% | 79.8% |
| **Insegnamento** | RL puro può far emergere il ragionamento | SFT + RL = risultati migliori |

### Risultati di DeepSeek R1

| Dominio | Risultato | Confronto |
|---------|-----------|-----------|
| **Matematica (AIME 2024)** | 79.8% | Comparabile a o1 di OpenAI |
| **Matematica (MATH-500)** | 97.3% | Stato dell'arte |
| **Coding (LiveCodeBench)** | 65.9% | Competitivo |
| **Conoscenza (MMLU)** | 90.8% | Eccellente |
| **Chat (AlpacaEval 2.0)** | 87.6% win rate | Ottimo |

---

## 6. Implementare GRPO con TRL

### Struttura base

Per fare GRPO hai bisogno di 4 cose:

```
1. Dataset di prompt
2. Reward function (la funzione che valuta le risposte)
3. GRPOConfig (i parametri di training)
4. GRPOTrainer (il trainer)
```

### Esempio minimo funzionante

```python
from trl import GRPOTrainer, GRPOConfig
from datasets import load_dataset

# 1. Carica il dataset (deve avere una colonna "prompt" o "messages")
dataset = load_dataset("il_tuo_dataset", split="train")

# 2. Definisci la reward function
# Input: lista di completion (risposte generate dal modello)
# Output: lista di float (reward per ciascuna risposta)
def reward_func(completions, **kwargs):
    """
    Reward semplice: premia risposte più lunghe.
    completions: lista di stringhe (le risposte generate dal modello)
    Ritorna: lista di float (un reward per ogni completion)
    """
    return [float(len(completion)) for completion in completions]
    # Nota: questa è una reward function didattica, non utile in pratica!

# 3. Configura il training
training_args = GRPOConfig(
    output_dir="output",           # Dove salvare i checkpoint
    num_train_epochs=3,            # Quante epoche
    per_device_train_batch_size=4, # Batch size per GPU
    gradient_accumulation_steps=2, # Accumula gradiente su 2 step
    logging_steps=10,              # Log ogni 10 step
    num_generations=8,             # CHIAVE: quante risposte generare per prompt (il "gruppo")
)

# 4. Crea e avvia il trainer
trainer = GRPOTrainer(
    model="Qwen/Qwen2-0.5B-Instruct",  # Modello da fine-tunare
    args=training_args,
    train_dataset=dataset,
    reward_funcs=reward_func,           # La nostra reward function
)
trainer.train()
```

### Il parametro più importante: `num_generations`

`num_generations` definisce la **dimensione del gruppo** — quante risposte genera il modello per ogni prompt prima di confrontarle:

```
num_generations = 4  → 4 risposte per prompt → gruppo piccolo
                          Pro: veloce, poca memoria
                          Contro: meno diversità per confrontare

num_generations = 8  → 8 risposte per prompt → buon equilibrio ✓

num_generations = 16 → 16 risposte per prompt → gruppo grande
                          Pro: più diversità, confronti migliori
                          Contro: lento, molta memoria
```

---

## 7. Reward Function Design — il cuore di GRPO

La reward function è **la parte più importante** di GRPO. Una reward function mal progettata porta a comportamenti indesiderati ("reward hacking" — il modello ottimizza la metrica sbagliata).

### Tipo 1: Reward basato sulla lunghezza

```python
# Premia risposte vicine alla lunghezza ideale
ideal_length = 50

def reward_len(completions, **kwargs):
    """
    Ritorna reward negativo proporzionale alla distanza dalla lunghezza ideale.
    Es: lunghezza 50 → reward 0 (perfetto)
        lunghezza 45 → reward -5
        lunghezza 80 → reward -30
    """
    return [-abs(ideal_length - len(completion)) for completion in completions]
```

### Tipo 2: Reward basato sulla correttezza (per task verificabili)

```python
def problem_reward(completions, answers, **kwargs):
    """
    Reward binario per problemi matematici.

    completions: risposte generate dal modello
    answers: risposte corrette dal dataset

    Ritorna 1.0 se la risposta estratta è corretta, 0.0 altrimenti.
    """
    rewards = []
    for completion, correct_answer in zip(completions, answers):
        try:
            answer = extract_final_answer(completion)  # estrai la risposta dalla testo
            reward = 1.0 if answer == correct_answer else 0.0
            rewards.append(reward)
        except:
            rewards.append(0.0)  # se non riesci a parsare la risposta, reward 0
    return rewards
```

> ⚠️ **Questo funziona solo per task verificabili** (matematica, codice, domande a risposta chiusa). Per task soggettivi (qualità della scrittura, utilità) serve un LLM come giudice.

### Tipo 3: Reward basato sul formato

```python
import re

def format_reward(completions, **kwargs):
    """
    Premia risposte che seguono il formato <think>...</think><answer>...</answer>
    Come in DeepSeek R1!

    Reward 1.0 se il formato è rispettato e il contenuto è sostanziale.
    Reward 0.0 se il formato è sbagliato.
    """
    # Il pattern cerca: <think> contenuto </think> <answer> contenuto </answer>
    pattern = r"<think>(.*?)</think>\s*<answer>(.*?)</answer>"

    rewards = []
    for completion in completions:
        match = re.search(pattern, completion, re.DOTALL)
        if match:
            think_content = match.group(1).strip()   # il ragionamento
            answer_content = match.group(2).strip()  # la risposta finale

            # Premiamo solo se c'è ragionamento sostanziale (> 20 caratteri)
            if len(think_content) > 20 and len(answer_content) > 0:
                rewards.append(1.0)   # formato ok + contenuto ok
            else:
                rewards.append(0.5)   # formato ok ma contenuto scarso
        else:
            rewards.append(0.0)   # formato sbagliato → nessun reward

    return rewards
```

### Tipo 4: Combinare più reward (come DeepSeek R1)

In produzione si usano più reward function insieme:

```python
trainer = GRPOTrainer(
    model=model,
    reward_funcs=[
        reward_format,          # verifica il formato
        reward_correctness,     # verifica la correttezza della risposta
        reward_len,             # verifica la lunghezza
    ],
    # ...
)
# I reward vengono sommati per ogni completion
```

---

## 8. Esercizio Pratico con TRL

Questo è l'esercizio completo di fine-tuning di un modello con GRPO. L'obiettivo: insegnare al modello a generare risposte di circa 50 token.

### Setup

```bash
pip install datasets transformers trl peft accelerate bitsandbytes
```

```python
import torch
from datasets import load_dataset
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import GRPOConfig, GRPOTrainer
```

### Carica modello e LoRA

```python
# Modello piccolo (135M parametri) — ideale per imparare
model_id = "HuggingFaceTB/SmolLM-135M-Instruct"

model = AutoModelForCausalLM.from_pretrained(
    model_id,
    torch_dtype="auto",     # usa float16 su GPU se disponibile
    device_map="auto",      # assegna automaticamente al device corretto
)
tokenizer = AutoTokenizer.from_pretrained(model_id)

# Aggiungi LoRA per ridurre la memoria necessaria
lora_config = LoraConfig(
    task_type="CAUSAL_LM",
    r=16,                           # rank LoRA
    lora_alpha=32,                  # scaling factor (2x rank)
    target_modules="all-linear",    # applica a tutti i layer lineari
)
model = get_peft_model(model, lora_config)
model.print_trainable_parameters()
# Output: "trainable params: ~2M || all params: 135M || trainable%: ~1.5%"
```

### Dataset e Reward Function

```python
# Dataset di testi (useremo smoltldr — un dataset di documenti)
dataset = load_dataset("mlabonne/smoltldr")

# Reward function: premiamo risposte vicine a 50 token
ideal_length = 50

def reward_len(completions, **kwargs):
    """
    Premio = quanto siamo lontani dalla lunghezza ideale (negativo).
    Reward 0   = perfetto (esattamente 50 token)
    Reward -10 = risposta lunga/corta di 10 token rispetto all'ideale
    """
    return [-abs(ideal_length - len(completion)) for completion in completions]
```

### Configura e Avvia il Training

```python
training_args = GRPOConfig(
    output_dir="GRPO",
    learning_rate=2e-5,                    # piccolo LR per stabilità
    per_device_train_batch_size=8,         # batch size
    gradient_accumulation_steps=2,         # accumula su 2 step
    max_prompt_length=512,                 # max lunghezza del prompt in input
    max_completion_length=96,             # max lunghezza della risposta generata
    num_generations=8,                     # 8 risposte per prompt (il "gruppo")
    optim="adamw_8bit",                    # ottimizzatore quantizzato per memoria
    num_train_epochs=1,                    # 1 epoca
    bf16=True,                             # usa bfloat16 per velocità
    logging_steps=1,                       # log ogni step
    remove_unused_columns=False,           # mantieni tutte le colonne
)

trainer = GRPOTrainer(
    model=model,
    reward_funcs=[reward_len],             # usa la nostra reward function
    args=training_args,
    train_dataset=dataset["train"],
)

trainer.train()
# Il training richiede ~1 ora su una GPU A10G
```

### Come leggere i risultati del training

```
Metrica          | Cosa significa              | Andamento desiderato
-----------------|-----------------------------|------------------------
reward           | Reward medio delle risposte | Deve aumentare (→ 0)
reward_std       | Variazione dei reward       | Alta variazione = buon confronto
kl               | Divergenza dal modello orig | Aumenta gradualmente (normale!)
loss             | Loss GRPO                   | Aumenta! (vedi nota sotto)
```

> ⚠️ **La loss in GRPO sale — è normale!** A differenza del supervised learning, in GRPO la loss misura la KL divergence (quanto il modello si è allontanato dall'originale). Più impara, più sale. Guarda il **reward**, non la loss, per capire se sta imparando.

### Salva e pubblica il modello

```python
# Fonde i pesi LoRA nel modello base e lo pubblica su HuggingFace Hub
merged_model = trainer.model.merge_and_unload()
merged_model.push_to_hub(
    "SmolGRPO-135M",                       # nome del repository
    private=False,                          # pubblica (True = privato)
    tags=["GRPO", "Reasoning-Course"]       # tag per trovarlo facilmente
)
```

### Testa il modello

```python
from transformers import pipeline

# Crea una pipeline di generazione testo
generator = pipeline("text-generation", model="SmolGRPO-135M")

# Documento lungo di test
prompt = """
# Un lungo documento sui gatti
Il gatto domestico (Felis catus) è un piccolo mammifero carnivoro addomesticato.
È l'unica specie addomesticata della famiglia Felidae. Le ultime ricerche
mostrano che la domesticazione del gatto è avvenuta nel Vicino Oriente circa
7500 a.C. È comunemente tenuto come animale domestico...
[testo molto lungo...]
"""

messages = [{"role": "user", "content": prompt}]

generated = generator(
    messages,
    max_new_tokens=256,
    do_sample=True,
    temperature=0.5,
    min_p=0.1,
)
print(generated[0]["generated_text"])
# Il modello ora dovrebbe generare riassunti di circa 50 token!
```

---

## 9. Esercizio Avanzato: GRPO con Unsloth (per hardware limitato)

**Unsloth** è una libreria che accelera il fine-tuning con LoRA, permettendo di lavorare su GPU con poca memoria (anche Google Colab gratuito!).

```bash
pip install unsloth vllm
```

### Carica il modello con Unsloth

```python
from unsloth import FastLanguageModel
import torch

max_seq_length = 1024  # lunghezza massima sequenza
lora_rank = 32         # rank LoRA

# Carica Gemma 3 1B con ottimizzazioni Unsloth
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="google/gemma-3-1b-it",
    max_seq_length=max_seq_length,
    load_in_4bit=True,       # quantizzazione 4-bit → dimezza la memoria!
    fast_inference=True,     # abilita vLLM per generazione veloce
    max_lora_rank=lora_rank,
    gpu_memory_utilization=0.6,  # usa il 60% della GPU memory
)

# Applica LoRA con Unsloth
model = FastLanguageModel.get_peft_model(
    model,
    r=lora_rank,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                    "gate_proj", "up_proj", "down_proj"],
    lora_alpha=lora_rank,
    use_gradient_checkpointing="unsloth",  # riduce ulteriormente la memoria
)
```

### Dataset e Formato per Ragionamento

```python
from datasets import load_dataset, Dataset

# Sistema prompt che chiede al modello di usare un formato specifico
SYSTEM_PROMPT = """
Rispondi nel seguente formato:
<think>
[ragionamento passo per passo]
</think>
<answer>
[risposta finale]
</answer>
"""

# Carica GSM8K (problemi matematici delle scuole medie)
def get_gsm8k_questions(split="train"):
    data = load_dataset("openai/gsm8k", "main")[split]
    data = data.map(
        lambda x: {
            # Formatta ogni esempio come conversazione ChatML
            "prompt": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": x["question"]},
            ],
            # Estrai solo il numero finale come risposta corretta
            # (GSM8K usa il formato "#### 42" per indicare la risposta)
            "answer": x["answer"].split("####")[1].strip(),
        }
    )
    return data

dataset = get_gsm8k_questions()
```

### Reward Functions per Ragionamento Matematico

```python
import re

# Funzione helper: estrae la risposta dai tag <answer>...</answer>
def extract_xml_answer(text: str) -> str:
    answer = text.split("<answer>")[-1]  # prende tutto dopo <answer>
    answer = answer.split("</answer>")[0]  # prende solo fino a </answer>
    return answer.strip()

# REWARD 1: Correttezza — la più importante!
# Reward 2.0 se la risposta è corretta, 0.0 se sbagliata
def correctness_reward_func(prompts, completions, answer, **kwargs):
    # Estrai il contenuto testuale di ogni completion
    responses = [completion[0]["content"] for completion in completions]
    # Estrai la risposta finale da ogni risposta
    extracted_responses = [extract_xml_answer(r) for r in responses]
    # Confronta con la risposta corretta dal dataset
    return [2.0 if r == a else 0.0 for r, a in zip(extracted_responses, answer)]

# REWARD 2: Formato numerico — la risposta è un numero intero?
# Reward 0.5 se la risposta è un numero, 0.0 altrimenti
def int_reward_func(completions, **kwargs):
    responses = [completion[0]["content"] for completion in completions]
    extracted_responses = [extract_xml_answer(r) for r in responses]
    return [0.5 if r.isdigit() else 0.0 for r in extracted_responses]

# REWARD 3: Formato XML — la risposta segue il formato richiesto?
# Reward 0.5 se il formato è corretto, 0.0 altrimenti
def strict_format_reward_func(completions, **kwargs):
    pattern = r"^\n<think>\n.*?\n</think>\n<answer>\n.*?\n</answer>\n$"
    responses = [completion[0]["content"] for completion in completions]
    matches = [re.match(pattern, r, re.DOTALL) for r in responses]
    return [0.5 if match else 0.0 for match in matches]
```

> 💡 **Nota:** La somma dei reward possibili è: 2.0 (correttezza) + 0.5 (intero) + 0.5 (formato) = **3.0 massimo**. Questo incentiva il modello a essere corretto (peso maggiore) E a seguire il formato.

### Avvia il Training

```python
from trl import GRPOConfig, GRPOTrainer

training_args = GRPOConfig(
    learning_rate=5e-6,
    per_device_train_batch_size=1,  # piccolo per GPU limitata
    gradient_accumulation_steps=1,
    num_generations=6,              # 6 risposte per prompt
    max_prompt_length=256,
    max_completion_length=768,      # risposte lunghe per il ragionamento
    max_steps=250,                  # training breve
    output_dir="outputs",
)

trainer = GRPOTrainer(
    model=model,
    processing_class=tokenizer,
    reward_funcs=[
        strict_format_reward_func,  # prima verifica il formato
        int_reward_func,            # poi se è un numero
        correctness_reward_func,    # poi se è corretto
    ],
    args=training_args,
    train_dataset=dataset,
)

trainer.train()
# Nota: i reward potrebbero non migliorare prima dei primi 150-200 step — sii paziente!
```

### Salva e Testa il Modello

```python
# Salva i pesi LoRA
model.save_lora("grpo_saved_lora")

# Testa il modello su un problema di matematica
from vllm import SamplingParams

text = tokenizer.apply_chat_template(
    [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": "Se Marco ha 15 mele e ne dà 7 a Sara, quante mele rimangono a Marco?"},
    ],
    tokenize=False,
    add_generation_prompt=True,
)

sampling_params = SamplingParams(temperature=0.8, top_p=0.95, max_tokens=512)

output = model.fast_generate(
    text,
    sampling_params=sampling_params,
    lora_request=model.load_lora("grpo_saved_lora"),
)[0].outputs[0].text

print(output)
# Output atteso:
# <think>
# Marco inizia con 15 mele. Ne dà 7 a Sara.
# 15 - 7 = 8 mele rimangono a Marco.
# </think>
# <answer>
# 8
# </answer>
```

---

## Quando usarlo

| Tecnica | Scenario |
|---------|---------|
| **RL / GRPO** | Vuoi insegnare un comportamento che non si definisce facilmente con esempi supervisati (es. ragionamento, sicurezza, preferenze umane) |
| **GRPO invece di PPO** | Non vuoi gestire un reward model separato; hai funzioni di valutazione automatiche |
| **GRPO invece di DPO** | Non hai dati di preferenza (coppie scelta/rifiutata); hai solo prompts e puoi definire un reward |
| **Unsloth + GRPO** | GPU con poca VRAM (<16GB), vuoi velocità massima, lavori su Google Colab |
| **Reward basato su correttezza** | Task verificabili: matematica, codice, domande a risposta chiusa |
| **Reward basato su formato** | Vuoi che il modello rispetti sempre un certo schema di output |

---

## Errori comuni

| Errore | Causa | Soluzione |
|--------|-------|-----------|
| La loss sale durante il training | Comportamento normale in GRPO | Non preoccuparti — guarda il **reward**, non la loss |
| Il reward non migliora nei primi step | GRPO è lento a partire | Aspetta 150-200 step prima di preoccuparti |
| CUDA out of memory | Troppe generazioni o batch troppo grande | Riduci `num_generations` (da 8 a 4) e `per_device_train_batch_size` |
| Il modello ottimizza la metrica sbagliata (reward hacking) | Reward function troppo semplice | Usa reward function combinate (correttezza + formato + ecc.) |
| `num_generations` deve essere divisibile per `per_device_train_batch_size` | Vincolo di TRL | Usa valori compatibili, es. num_gen=8, batch=4 |
| Il modello "dimentica" le sue capacità | KL penalty troppo bassa | Aumenta `kl_coeff` in GRPOConfig |

---

## 🗂️ Glossario Completo RL/GRPO

| Termine | Significato |
|---------|-------------|
| **RL** | Reinforcement Learning — apprendimento per rinforzo |
| **Agente** | Il modello che impara (in RL per LLM: il modello linguistico) |
| **Ambiente** | Il contesto che fornisce feedback all'agente |
| **Azione** | Scelta dell'agente (per LLM: i token/parole generati) |
| **Reward** | Numero che valuta quanto è buona un'azione |
| **Policy** | Strategia dell'agente: dato uno stato, quale azione scegliere |
| **RLHF** | RL from Human Feedback — i reward vengono da preferenze umane |
| **Reward Model** | Modello addestrato a prevedere le preferenze umane |
| **PPO** | Proximal Policy Optimization — algoritmo RL che richiede reward model |
| **DPO** | Direct Preference Optimization — usa coppie scelta/rifiutata senza RL esplicito |
| **GRPO** | Group Relative Policy Optimization — genera gruppi di risposte e le confronta |
| **KL Divergence** | Misura quanto il modello si è allontanato dalla sua versione originale |
| **num_generations** | Dimensione del "gruppo" in GRPO (quante risposte per prompt) |
| **Reward hacking** | Il modello ottimizza la metrica in modo letterale ma sbagliato |
| **Aha Moment** | Fenomeno emergente in R1-Zero: il modello impara ad auto-correggersi |
| **Unsloth** | Libreria che ottimizza il fine-tuning LoRA, riducendo tempi e memoria |
| **GRPOTrainer** | Classe TRL per il training con GRPO |
| **GRPOConfig** | Configurazione del training GRPO (estende TrainingArguments) |

---

## Riepilogo visivo del flusso GRPO

```
                    ┌─────────────────────┐
                    │  Dataset di prompt  │
                    └─────────┬───────────┘
                              │
                              ▼
              ┌───────────────────────────────┐
              │  Per ogni prompt, genera N    │
              │  risposte diverse (gruppo)    │
              │  es. N = 8                    │
              └───────────────┬───────────────┘
                              │
                              ▼
              ┌───────────────────────────────┐
              │  Valuta ogni risposta con     │
              │  la reward function           │
              │  reward = [0.8, 0.2, 1.0, ...│
              └───────────────┬───────────────┘
                              │
                              ▼
              ┌───────────────────────────────┐
              │  Normalizza i reward nel      │
              │  gruppo (calcola advantage)   │
              │  advantage = (r - mean) / std │
              └───────────────┬───────────────┘
                              │
                              ▼
              ┌───────────────────────────────┐
              │  Aggiorna il modello:         │
              │  + favorisce risposte buone   │
              │  - penalizza risposte cattive │
              │  + KL penalty (freno sicurez.)│
              └───────────────┬───────────────┘
                              │
                              ▼
                    ┌─────────────────────┐
                    │  Modello migliorato │
                    │  → prossimo batch   │
                    └─────────────────────┘
```