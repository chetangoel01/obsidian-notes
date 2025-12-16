### Executive Summary

This lecture introduces **Natural Language Processing (NLP)** with a focus on how text is turned into numbers, how those numbers are embedded into vectors, and how **language models** predict the next token in a sequence. You also saw the **course project** architecture based on **RAG/GraphRAG**, where embeddings and retrieval over a knowledge graph are used to build an AI tutor. By the end of this guide, you should understand the main NLP tasks (POS tagging, dependency parsing, NER, coreference), how **tokenization** and **Byte-Pair Encoding (BPE)** build a vocabulary, how **word embeddings** and **softmax** define a probability distribution over words, how **simple language models and RNNs** model sequences, and how these pieces plug into a **GraphRAG** system for personalized tutoring. 

---

## 1. Concept: Natural Language Processing & Core NLP Tasks

### 1. High-Level Intuition

**Problem it solves (why):** NLP gives machines a way to understand and manipulate human language—turning raw text into structured information and predictions.

**Analogy:** Think of NLP as teaching a foreigner the grammar of English and then giving them tools to highlight names, draw arrows between related words, and resolve pronouns—like scribbling notes and arrows all over a printed paragraph.

---

### 2. Conceptual Deep Dive

Key terms:

- **Corpus**: A large collection of text documents used for training models.
    
- **Token**: A basic unit of text (word or subword) after tokenization.
    
- **Part-of-Speech (POS) Tagging**: Assigning grammatical labels (noun, verb, adjective) to each token.
    
- **Dependency Parsing**: Building a **tree** where nodes are words and edges represent grammatical relations (subject-of, object-of, modifier-of, etc.).
    
- **Named Entity Recognition (NER)**: Marking spans of text as entities like **PERSON**, **ORG**, **LOCATION**, **DATE**, etc.
    
- **Coreference Resolution**: Linking pronouns and referring expressions (“he”, “she”, “it”, “the company”) back to the correct entities mentioned earlier. 
    

Typical pipeline:

1. You start with raw text:  
    `“London is the capital and largest city of England.”`
    
2. **Segmentation / tokenization** splits it into tokens, e.g. `["London", "is", "the", "capital", ...]`. 
    
3. **POS tagging** might label:
    
    - `London/NNP`, `is/VBZ`, `capital/NN`, `city/NN`, `England/NNP`, ...
        
4. **Dependency parsing** identifies grammatical structure:
    
    - `London` is **subject** of `is`.
        
    - `capital` is a complement or attribute of `London`. 
        
5. **NER** highlights entities: `"London"` → **LOCATION**, `"England"` → **LOCATION**.
    
6. **Coreference** (in longer text) links pronouns like `it` back to `"London"` or `"the city"`.
    

These tasks are often used as preprocessing or auxiliary signals to help downstream applications (question answering, summarization), or as standalone tasks.

---

### 3. Mathematical Formulation

We can model many NLP tasks as predicting a **label sequence** for a token sequence.

Let:

- Sentence tokens: $\mathbf{w} = (w_1, w_2, \dots, w_T)$
    
- Labels (e.g., POS tags): $\mathbf{y} = (y_1, y_2, \dots, y_T)$
    

A sequence model defines:

$$  
P(\mathbf{y} \mid \mathbf{w}) = \prod_{t=1}^{T} P(y_t \mid w_{1:T}, y_{1:t-1})  
$$

**Annotations:**

- $T$: number of tokens in the sentence.
    
- $w_{1:T}$: the entire token sequence.
    
- $y_t$: label for token $w_t$ (e.g., **NOUN**, **VERB**).
    
- $P(y_t \mid \cdot)$: conditional probability of the tag at position $t$ given the whole sentence and previous tags.
    

In simpler models (e.g., independent taggers):

$$ 
P(\mathbf{y} \mid \mathbf{w}) \approx \prod_{t=1}^{T} P(y_t \mid w_t)  
$$

Here, each token is tagged independently based only on its own features.

---

### 4. Worked Toy Example

Sentence:

> "London is beautiful"

1. Tokens: $w_1 = \text{"London"}, w_2 = \text{"is"}, w_3 = \text{"beautiful"}$.
    
2. Suppose a simple classifier gives these POS distributions:
    

- For **"London"**:
    
    - $P(\text{NOUN} \mid \text{"London"}) = 0.1$
        
    - $P(\text{PROPN} \mid \text{"London"}) = 0.9$
        
- For **"is"**:
    
    - $P(\text{VERB} \mid \text{"is"}) = 0.95$
        
- For **"beautiful"**:
    
    - $P(\text{ADJ} \mid \text{"beautiful"}) = 0.97$
        

3. The most likely tag sequence is:
    

- `London/PROPN`, `is/VERB`, `beautiful/ADJ`.
    

4. Joint probability (assuming independence):
    

$$ 
P(\mathbf{y} \mid \mathbf{w}) = 0.9 \times 0.95 \times 0.97 \approx 0.829  
$$

So our model says there’s ~82.9% probability this particular tagging is correct (under the naïve independence assumption).

---

### 5. Connections & Prerequisites

**Prerequisite Refresher – Probability Basics:**  
To follow these models, you should be comfortable with **conditional probability** and **product rules**: the probability of a sequence of events is often the product of the probability of each step given the previous ones. This is the same idea used later for **language modeling**, where we predict the next word given previous words.

---

## 2. Concept: Tokenization & Vocabularies

### 1. High-Level Intuition

**Problem it solves:** Tokenization converts messy strings of characters into **discrete IDs** so that neural networks can operate on them.

**Analogy:** Imagine a library where every unique word or subword has a unique index in a giant catalog. Tokenization is how you look up and assign these catalog numbers to every piece of text you see.

---

### 2. Conceptual Deep Dive

Key terms:

- **Tokenizer**: A function converting raw text into a sequence of **token IDs**.
    
- **Vocabulary** (V): The set of all symbols (words or subwords) the model knows; each has a unique integer ID. 
    
- **Word-level tokenization**: Treat every distinct word as a token (“New”, “York”, “New-York” are separate).
    
- **Subword-level tokenization**: Break words into frequently occurring pieces (“New”, “York”; or “play”, “ing”), which helps handle rare words and misspellings.
    
- **OOV (Out-of-Vocabulary)**: Tokens not present in the vocabulary; subword tokenizers reduce OOV cases.
    

The tokenizer takes input text and:

1. Possibly normalizes it (lowercasing, stripping accents, handling punctuation).
    
2. Splits into tokens (word or subword units).
    
3. Maps each token to an integer ID via the vocabulary.
    

The **vocabulary size** (|V|) strongly affects the **complexity** of the model: the output layer of a language model must predict a probability distribution over all (|V|) tokens. Larger vocabularies give finer-grained tokens but are more computationally expensive. 

---

### 3. Mathematical Formulation

Define:

- Let $\mathcal{T}$ be the tokenizer.
    
- Let $V = \{v_1, \dots, v_{|V|}\}$ be the vocabulary.
    
- Let `id` be a mapping $\text{id}: V \to \{1, \dots, |V|\}$.
    

Then for a raw string $s$:

$$  
\mathcal{T}(s) = (i_1, i_2, \dots, i_T)  
$$

**Annotations:**

- $s$: raw input text (string).
    
- $\mathcal{T}(s)$: tokenized sequence (list of token IDs).
    
- $T$: number of tokens produced.
    
- Each $i_t \in \{1,\dots,|V|\}$ identifies a vocabulary entry.
    

---

### 4. Worked Toy Example

Suppose we define a tiny word-level vocabulary:

$$  
V = \{\texttt{[PAD]}, \texttt{[UNK]}, \texttt{London}, \texttt{is}, \texttt{beautiful}\}  
$$

Assign IDs:

- `[PAD]` → 0
    
- `[UNK]` → 1
    
- `London` → 2
    
- `is` → 3
    
- `beautiful` → 4
    

Text: `"London is beautiful"`

1. The tokenizer splits on spaces → `["London", "is", "beautiful"]`.
    
2. Map each token to its ID using the dictionary:
    

$$  
\mathcal{T}("London is beautiful") = (2, 3, 4)  
$$

If we had `"London is amazing"`, and `"amazing"` is not in `V`, we would get:

$$  
\mathcal{T}("London is amazing") = (2, 3, 1)  
$$

where `1` is `[UNK]`.

---

### 5. Connections & Prerequisites

**Prerequisite Refresher – Sets & Functions:**  
We treat the vocabulary as a **set** $V$, and the tokenizer is a **function** mapping strings to sequences of indices in that set. Understanding basic set notation and how functions map inputs to outputs will help a lot in the rest of the lecture (e.g., embedding matrices map token IDs to vectors).

---

## 3. Concept: Subword Tokenization via Byte-Pair Encoding (BPE)

### 1. High-Level Intuition

**Problem it solves:** BPE balances having a **small vocabulary** (for efficiency) with the ability to represent **virtually any word**, including rare or misspelled ones, using subword chunks.

**Analogy:** Imagine compressing text by repeatedly replacing the most frequent pairs of letters with a new symbol, like inventing shorthand. Over time, common patterns like “th”, “ing”, “tion” become single symbols.

---

### 2. Conceptual Deep Dive

Key terms:

- **Byte-Pair Encoding (BPE)**: An algorithm originally from **compression** that repeatedly merges the most frequent pair of symbols into a new symbol. 
    
- **Base vocabulary**: Start with all individual characters (or bytes).
    
- **Merge rule**: A rule like “S” + “H” → “SH” that becomes a new symbol in the vocabulary.
    

Algorithm (high-level) as presented:

1. Start with text represented as a sequence of characters (or bytes).
    
2. Count how often each **pair of adjacent symbols** appears.
    
3. Find the most frequent pair (e.g., “A A” or “S H”).
    
4. Add a new symbol representing this pair (e.g., “AA” or “SH”) to the vocabulary.
    
5. Replace every occurrence of that pair with the new symbol.
    
6. Repeat steps 2–5 for a fixed number of merges or until no pair is frequent enough. 
    

The key idea: by merging frequent pairs, we:

- Reduce the **length** of the token sequence.
    
- Build up a vocabulary of useful **subwords** automatically.
    
- Keep the total number of symbols manageable.
    

---

### 3. Mathematical Formulation

Let:

- Initial alphabet: $A = \{a_1, a_2, \dots, a_k\}$.
    
- Current vocabulary after $m$ merges: $V^{(m)}$.
    
- Training corpus represented as a sequence of symbols from $V^{(m)}$.
    

At each iteration $m$:

1. Compute bigram counts:
    

$$  
c^{(m)}(x, y) = \text{number of times pair } (x, y) \text{ appears consecutively in corpus}  
$$

for all $(x, y) \in V^{(m)} \times V^{(m)}$.

2. Select the most frequent pair:
    

$$  
(x^*, y^*) = \arg\max_{(x, y)} c^{(m)}(x, y)  
$$

3. Add a new symbol $z = x^*y^*$ to vocabulary:
    

$$  
V^{(m+1)} = V^{(m)} \cup \{z\}  
$$

4. Replace every occurrence of $(x^*, y^*)$ in the corpus with the new symbol $z$.
    

**Annotations:**

- $V^{(m)}$: vocabulary after $m$ merges.
    
- $c^{(m)}(x, y)$: frequency of adjacent pair at step $m$.
    
- $z$: new merged symbol corresponding to pair $(x^*, y^*)$.
    

---

### 4. Worked Toy Example (Similar to the Lecture’s Example)

Start with a toy text (characters spaced for clarity):

`A A A B A C`

Initial vocabulary: $V^{(0)} = \{A, B, C, \text{space}\}$.

**Step 1 – Count pairs**

Pairs in the sequence:

- (A, space), (space, A), (A, space), (space, A), (A, space), (space, B), (B, space), (space, A), (A, space), (space, C)
    

Instead, consider just within words: `AAA`, `B`, `AC`.

Pairs:

- Inside `AAA`: (A, A), (A, A) → 2 times
    
- Inside `AC`: (A, C) → 1 time
    

So:

- $c(A, A) = 2$
    
- $c(A, C) = 1$
    

The most frequent pair is (A, A).

**Step 2 – Merge A A → Z**

Add new symbol $Z$:

- Replace all `A A` occurrences:
    

`A A A B A C` → `Z A B A C` (since the first two As become Z).

**Step 3 – Re-count pairs**

Now inside `Z A` and `A C`:

- Pairs: (Z, A), (A, C)
    

Suppose (Z, A) is more frequent. Merge Z A → Y.

Text becomes: `Y B A C`.

Add Y to vocabulary.

**Step 4 – Continue**

Now pairs: (Y, B), (B, A), (A, C).

You keep merging until you’ve done a set number of merges. At the end:

- You’ve added new symbols like `Z = "AA"`, `Y = "ZA"`, etc.
    
- The text is shorter in terms of number of tokens.
    
- Your vocabulary now includes those multi-character subwords. 
    

---

### 5. Connections & Prerequisites

**Prerequisite Refresher – Tokenization:**  
BPE is just a **smarter tokenizer**: instead of splitting at spaces, it **learns subword pieces** from data by looking at frequently co-occurring character pairs. You need to understand the idea of a vocabulary and token sequence from the previous concept to see how BPE modifies and grows that vocabulary.

---

## 4. Concept: Tokenization in Practice & Token Counts

### 1. High-Level Intuition

**Problem it solves:** In real LLMs, we must know **how many tokens** a prompt uses (for cost & context-length), and tokenization must match the model’s training.

**Analogy:** Think of your context window as a fixed-size tray with a limited number of slots. Different tokenizers break the same text into a different number of Lego pieces, so you can fit more or fewer words into the tray.

---

### 2. Conceptual Deep Dive

In practice:

- Every model uses a **specific tokenizer** (often a BPE-style tokenizer).
    
- Tools like the “tiktokenizer” web app let you paste text and see how many tokens it becomes for a given model. 
    
- The **context window** (e.g., 4k, 8k, 128k tokens) limits how much text the model sees at once.
    
- Different tokenizers produce different token counts for the same text; this affects:
    
    - How much text fits before truncation.
        
    - The computational cost (more tokens → more compute).
        

The lecture illustrates copying a block of text and showing how many tokens it becomes under a GPT-2-style tokenizer vs another tokenizer. 

---

### 3. Mathematical Formulation

Let:

- $s$ = input text.
    
- $\mathcal{T}$ = tokenizer for a specific model.
    
- $C$ = maximum context window (in tokens).
    

Then:

$$  
\mathcal{T}(s) = (i_1, \dots, i_T), \quad T = |\mathcal{T}(s)|  
$$

We require:

$$  
T \leq C  
$$

Otherwise, the text must be truncated or split into multiple chunks.

If we have cost roughly proportional to number of tokens:

$$  
\text{ComputeCost} \propto T  
$$

Or, in transformer models, something closer to:

$$  
\text{ComputeCost} \propto T^2 \cdot d  
$$

where:

- $T$: number of tokens.
    
- $d$: model hidden dimension.
    

---

### 4. Worked Toy Example

Suppose a model has:

- Context window $C = 8$ tokens.
    

Text 1: `"Hello, world!"`

- Tokenizer outputs 3 tokens: `["Hello", ",", "world!"]` → 3 tokens.
    
- $T_1 = 3 \leq 8$. Fits easily.
    

Text 2: `"This is a slightly longer example sentence."`

- Suppose tokenizer outputs 9 tokens: `["This", "is", "a", "slightly", "long", "er", "example", "sentence", "."]`
    
- $T_2 = 9 > 8$. It does **not** fit.
    

We must either:

- Truncate to first 8 tokens, or
    
- Split into two segments: first 8 tokens, then the 9th token continues in another request.
    

---

### 5. Connections & Prerequisites

**Prerequisite Refresher – BPE & Vocabulary:**  
The number of tokens depends on **how your tokenizer splits text** into subwords (from BPE) and what’s in the **vocabulary**. Different merge rules → different token counts for the _same_ text. That’s why the model’s tokenizer and vocabulary are tightly coupled to the model itself.

---

## 5. Concept: Word Embeddings & the Embedding Matrix

### 1. High-Level Intuition

**Problem it solves:** Embeddings turn discrete token IDs into dense numerical vectors that capture **semantic similarity**(e.g., “king” and “queen” are close).

**Analogy:** Imagine placing every word as a point in a high-dimensional “meaning space” where similar words live near each other—like cities on a map, but in hundreds of dimensions.

---

### 2. Conceptual Deep Dive

Key terms:

- **One-hot vector**: A vector with all zeros except a 1 in the position corresponding to a token’s ID.
    
- **Embedding matrix**: A trainable matrix that maps one-hot vectors (or token IDs) to dense vectors (embeddings).
    
- **Embedding dimension** (d): The size of the vector representing each token.
    
- **Output projection**: Another matrix that maps embeddings back to a vector of logits over the vocabulary, then a **softmax** converts these to probabilities. 
    

In the lecture, the instructor describes:

- A hidden or embedding vector $z$ (size $d$).
    
- A matrix that **lifts** $z$ back into a $|V|$-dimensional space, where $|V|$ is vocabulary size (e.g. 100,000). This output is then passed through **softmax** to give a probability distribution over all words. 
    

Intuitively:

1. Start with a token ID $j$.
    
2. Look up its embedding $z_j \in \mathbb{R}^d$.
    
3. Use $z_j$ (alone, or combined with context) to predict probabilities for the next word by projecting and using softmax.
    

---

### 3. Mathematical Formulation

Let:

- Vocabulary size: $|V|$.
    
- Embedding dimension: $d$.
    

**Embedding lookup:**

- Embedding matrix $E \in \mathbb{R}^{|V| \times d}$.
    
- One-hot vector for token $j$: $e_j \in \mathbb{R}^{|V|}$ (1 at position $j$, 0 elsewhere).
    

Then the embedding for token $j$ is:

$$  
z_j = E^\top e_j \in \mathbb{R}^d  
$$

or, equivalently, the $j$-th row of $E$.

**Output projection and softmax:**

- Output matrix $U \in \mathbb{R}^{|V| \times d}$.
    
- Logits $\ell \in \mathbb{R}^{|V|}$ computed as:
    

$$  
\ell = U z_j  
$$

Softmax over logits:

$$  
P(w = k \mid \text{context}) = \frac{\exp(\ell_k)}{\sum_{k'=1}^{|V|} \exp(\ell_{k'})}  
$$

**Annotations:**

- $z_j$: embedding of the current (or center) token.
    
- $E$: embedding matrix (trainable).
    
- $U$: output projection matrix (trainable).
    
- $\ell_k$: logit (unnormalized score) for vocabulary token $k$.
    
- $P(w=k \mid \text{context})$: predicted probability that token $k$ is the next word.
    

---

### 4. Worked Toy Example

Tiny vocab:

$$  
V = \{\texttt{I}, \texttt{love}, \texttt{pizza}\}, \quad |V| = 3  
$$

Let embedding dimension $d = 2$.

Embedding matrix:

$$  
E =  
\begin{bmatrix}  
1 & 0 \\  
0 & 1 \\  
1 & 1  
\end{bmatrix}  
$$

Rows correspond to `I`, `love`, `pizza`.

Suppose we want the embedding for `love`:

- `love` is token 2 → one-hot $e_2 = (0, 1, 0)^\top$.
    
- $z_{\text{love}} = E^\top e_2 = [0, 1]^\top$.
    

Let output matrix:

$$  
U =  
\begin{bmatrix}  
1 & 0 \\  
0 & 1 \\  
-1 & 1  
\end{bmatrix}  
$$

Logits:
$$  
\ell = U z_{\text{love}} =  
\begin{bmatrix}  
1 & 0 \\  
0 & 1 \\  
-1 & 1  
\end{bmatrix}  
\begin{bmatrix}  
0 \\  
1  
\end{bmatrix}
=
\begin{bmatrix}  
0 \\  
1 \\  
1  
\end{bmatrix}  
$$

Softmax:

$$  
P(w=k \mid \text{“love”}) = \frac{e^{\ell_k}}{e^0 + e^1 + e^1}  
$$

Compute denominator:

- $e^0 = 1$
    
- $e^1 \approx 2.718$
    

Denominator $Z = 1 + 2.718 + 2.718 \approx 6.436$.

So:

- $P(\texttt{I}) \approx 1 / 6.436 \approx 0.155$
    
- $P(\texttt{love}) \approx 2.718 / 6.436 \approx 0.422$
    
- $P(\texttt{pizza}) \approx 2.718 / 6.436 \approx 0.422$
    

So the model thinks “love” and “pizza” are about equally likely next words given this center embedding.

---

### 5. Connections & Prerequisites

**Prerequisite Refresher – Linear Algebra & Softmax:**  
You should recall how matrix–vector multiplication works and what a **softmax** is: it converts arbitrary real-valued scores (logits) into a **probability distribution** that sums to 1. This is the same mechanism used in earlier classification networks; here the output dimension is $|V|$, often very large (e.g., 100k words). 

---

## 6. Concept: Language Models & Sequence Probabilities

### 1. High-Level Intuition

**Problem it solves:** A language model assigns **probabilities to sequences of tokens** and can predict the **next token** given previous ones.

**Analogy:** It’s like an autocomplete that has learned the “statistics” of language: after “New York”, “City” is more likely than “Banana”.

---

### 2. Conceptual Deep Dive

Key terms:

- **Language model (LM)**: A model that defines $P(w_t \mid w_{1:t-1})$, the probability of the next token given the previous ones. 
    
- **Context window**: The number of previous tokens the model can condition on (e.g., last 3 tokens vs full sequence).
    
- **Maximum-likelihood sequence** (MLSC in the transcript): Instead of picking the best next word greedily at each step, you choose the sequence of tokens that maximizes the **overall sequence probability** (e.g., via Viterbi decoding in some models). 
    

The lecture frames a basic LM as:

$$  
P_{\text{model}}\big(w_t \mid w_{t-1}, w_{t-2}, \dots, w_{t-N}\big)  
$$

where $N$ is the **context size** (could be large in modern LMs). 

---

### 3. Mathematical Formulation

For a sequence of tokens $\mathbf{w} = (w_1, \dots, w_T)$:

By the **chain rule of probability**:

$$  
P(\mathbf{w}) = P(w_1, w_2, \dots, w_T) = \prod_{t=1}^{T} P(w_t \mid w_{1:t-1})  
$$

For an **N-gram model** (finite context):

$$  
P(w_t \mid w_{1:t-1}) \approx P(w_t \mid w_{t-N:t-1})  
$$

For a **neural LM**, each conditional probability is represented via:

$$  
P(w_t = k \mid w_{1:t-1}) = \text{softmax}_k\big( f_\theta(w_{1:t-1}) \big)  
$$

**Annotations:**

- $f_\theta(\cdot)$: neural network with parameters $\theta$ that maps context to logits.
    
- $\text{softmax}_k(\cdot)$: softmax output for vocabulary entry $k$.
    
- $N$: context length (can be full history for RNNs/transformers).
    

Training objective is typically **maximum likelihood**:

$$  
\theta^* = \arg\max_\theta \sum_{\text{sequences}} \sum_{t} \log P_\theta(w_t \mid w_{1:t-1})  
$$

Equivalent to minimizing **cross-entropy loss**.

---

### 4. Worked Toy Example

Vocabulary $V = \{\texttt{I}, \texttt{love}, \texttt{pizza}\}$.

We define a simple bigram (1-step) language model. Suppose we have conditional probabilities:

- $P(\texttt{love} \mid \texttt{I}) = 0.8$, $P(\texttt{pizza} \mid \texttt{I}) = 0.1$, $P(\texttt{I} \mid \texttt{I}) = 0.1$.
    
- $P(\texttt{pizza} \mid \texttt{love}) = 0.9$, $P(\texttt{I} \mid \texttt{love}) = 0.05$, $P(\texttt{love} \mid \texttt{love}) = 0.05$.
    
- Start token probability: $P(\texttt{I}) = 1.0$ (sentence always starts with "I" in this toy example).
    

Sequence: `"I love pizza"` → $w_1 = \texttt{I}, w_2 = \texttt{love}, w_3 = \texttt{pizza}$.

$$  
P(\text{"I love pizza"}) = P(w_1), P(w_2 \mid w_1), P(w_3 \mid w_2)  
$$  
$$  
= 1.0 \times 0.8 \times 0.9 = 0.72  
$$

So the model thinks this sequence is quite likely.

Compare to `"I pizza love"`:

$$  
P(\text{"I pizza love"}) = P(w_1), P(\texttt{pizza} \mid \texttt{I}), P(\texttt{love} \mid \texttt{pizza})  
$$

We don't have the last conditional; suppose $P(\texttt{love} \mid \texttt{pizza}) = 0.05$. Then:

$$  
= 1.0 \times 0.1 \times 0.05 = 0.005  
$$

Much smaller, so `"I love pizza"` is preferred.

---

### 5. Connections & Prerequisites

**Prerequisite Refresher – Embeddings & Softmax:**  
Language models use the **embedding** of tokens (and context) to produce logits over the vocabulary via a projection and **softmax**, exactly as in the previous concept. Here, we're just applying that machinery at every time step $t$ to produce $P(w_t \mid w_{1:t-1})$.

---

## 7. Concept: Recurrent Neural Networks (RNNs) for Sequence Modeling

### 1. High-Level Intuition

**Problem it solves:** RNNs incorporate **history** into predictions by maintaining a hidden **state** that is updated as tokens arrive.

**Analogy:** Think of an RNN as a person reading a sentence word-by-word, updating their mental state after each word, and using that evolving state to guess the next word.

---

### 2. Conceptual Deep Dive

Key terms:

- **Hidden state** $h_t$: A vector representing everything the RNN remembers up to time $t$.
    
- **Input** $x_t$: The embedding of the token at time $t$.
    
- **Parameters** $(U, W, b)$: Weights and bias controlling how new input and previous state combine.
    
- **Activation function**: Non-linear function like $\tanh$ applied to the linear combination.
    

In the lecture, a simple scalar version of an RNN is given (for conceptual clarity): $h_t$ is a scalar, and the recurrence is:

$$  
h_t = \tanh\big( U^\top x_t + W h_{t-1} + b \big)  
$$

with:

- $h_{t-1}$ as the stored **memory** (previous hidden state).
    
- $x_t$ as the current input.
    
- $(U, W, b)$ as trainable parameters. 
    

This structure can model **time series** and, by extension, **language sequences** by treating each token as part of a sequence. 

---

### 3. Mathematical Formulation

Full vector form:

- Inputs $x_t \in \mathbb{R}^d$
    
- Hidden state $h_t \in \mathbb{R}^h$
    

RNN update:

$$  
h_t = \tanh\left( U x_t + W h_{t-1} + b \right)  
$$

Output logits (for LM):

$$  
\ell_t = V h_t  
$$

Softmax:

$$  
P(w_t = k \mid w_{1:t-1}) = \frac{\exp(\ell_{t, k})}{\sum_{k'=1}^{|V|} \exp(\ell_{t, k'})}  
$$

**Annotations:**

- $U \in \mathbb{R}^{h \times d}$: weights from input to hidden.
    
- $W \in \mathbb{R}^{h \times h}$: recurrent weights (hidden to hidden).
    
- $b \in \mathbb{R}^{h}$: hidden bias.
    
- $V \in \mathbb{R}^{|V| \times h}$: hidden-to-output weights.
    
- $\tanh$: applied elementwise, keeps values in $(-1, 1)$.
    
- $\ell_{t,k}$: logit for token $k$ at time $t$.
    

---

### 4. Worked Toy Example (Scalar RNN)

Consider a **scalar** RNN (hidden dimension $h=1$) for intuition:

Parameters:

- $U = 1.0$
    
- $W = 0.5$
    
- $b = 0.0$
    

Inputs (also scalars, say simple features of tokens):

- $x_1 = 1.0$
    
- $x_2 = 0.5$
    

Initial hidden state: $h_0 = 0$.

**Step 1 – t=1**

$$  
h_1 = \tanh(U x_1 + W h_0 + b) = \tanh(1.0 \cdot 1.0 + 0.5 \cdot 0 + 0) = \tanh(1.0) \approx 0.7616  
$$

**Step 2 – t=2**

$$  
h_2 = \tanh(U x_2 + W h_1 + b) = \tanh(1.0 \cdot 0.5 + 0.5 \cdot 0.7616) = \tanh(0.5 + 0.3808) = \tanh(0.8808)  
$$

$\tanh(0.8808) \approx 0.707$.

So:

- After the first token: hidden state $h_1 \approx 0.76$.
    
- After the second token: hidden state $h_2 \approx 0.71$.
    

This hidden state is then used to predict the next token’s distribution via a linear layer + softmax (not computed here).

---

### 5. Connections & Prerequisites

**Prerequisite Refresher – Feedforward Neurons:**  
The RNN cell is like a **standard neuron** (linear combination + non-linearity) but with an extra input: the **previous hidden state**. You should recall how a single neuron computes $\tanh(w^\top x + b)$; RNNs simply add recurrence and apply this across time. RNNs were presented as a stepping stone to **transformers**, which use attention instead of recurrence for handling sequences. 

---

## 8. Concept: Retrieval-Augmented Generation (RAG) & GraphRAG for the AI Tutor Project

### 1. High-Level Intuition

**Problem it solves:** RAG+GraphRAG let an LLM **use external knowledge** (like course materials) and **structure it as a graph** so that it can provide _learning paths_ and references, not just answers.

**Analogy:** Think of the LLM as a smart teacher who has a personal bookshelf (vector database) and a concept map on the wall (knowledge graph). When you ask a question, the teacher pulls the most relevant pages and then follows the concept map to give you a structured explanation.

---

### 2. Conceptual Deep Dive

Key terms:

- **RAG (Retrieval-Augmented Generation)**: Pipeline where an LLM retrieves relevant text chunks from a knowledge base (using embeddings) and then uses them to answer queries. 
    
- **GraphRAG**: An extension where we build a **knowledge graph** of concepts and their relationships, and retrieval returns **subgraphs** (sets of nodes/edges) rather than just flat text chunks. 
    
- **Knowledge graph**: Graph where nodes are **concepts**, edges represent semantic relations (e.g., prerequisite-of, part-of).
    
- **Vector database**: Database that stores embedding vectors and supports nearest-neighbor search (e.g., using PostgreSQL + vector extensions, as mentioned). 
    

Project architecture (as described):

1. **Ingestion**:
    
    - Collect course content: website pages, PDFs, recording transcripts. 
        
    - Parse into **chunks**, store raw data in a NoSQL or relational DB (e.g., MongoDB or PostgreSQL).
        
2. **Embedding & Vector Store**:
    
    - For each chunk, compute an **embedding vector** using an embedding model.
        
    - Store these vectors in a **vector database**.
        
3. **Graph Construction** (GraphRAG-specific):
    
    - Use an LLM to extract **entities** and **relationships** from each chunk.
        
    - Build a **knowledge graph** where:
        
        - Nodes = concepts.
            
        - Edges = relations such as “requires understanding of” or “is part of”. 
            
4. **Querying**:
    
    - A student question is embedded into a vector.
        
    - Retrieve nearby chunks **and** identify relevant nodes in the graph.
        
    - Return a **subgraph** and relevant references (pages, timestamps in videos).
        
    - LLM generates a **learning path answer**, from simple to more complex concepts, with explicit references. 
        

The project’s goal: implement such a system for an AI tutor named **Erica**, using only **textual interactions** plus some visual components, and _without using pre-built RAG systems’ extra features_. 

---

### 3. Mathematical Formulation

Represent the knowledge graph as:

$$  
G = (C, E)  
$$

where:

- $C = \{c_1, \dots, c_n\}$: set of **concept nodes**.
    
- $E \subseteq C \times C$: set of directed **edges**, e.g., $(c_i, c_j)$ means "$c_i$ is a prerequisite of $c_j$".
    

Let:

- $\phi: \text{chunks} \to \mathbb{R}^d$: embedding function for text.
    
- $\psi: \text{queries} \to \mathbb{R}^d$: embedding function for questions.
    

For a query $q$:

1. Compute query embedding: $z_q = \psi(q)$.
    
2. Retrieve top-k chunks $\{x_{(1)}, \dots, x_{(k)}\}$ whose embeddings are nearest to $z_q$ in cosine or Euclidean distance.
    
3. From these chunks, identify a set of relevant concepts $C_q \subseteq C$.
    
4. Extract a **subgraph**:
    

$$  
G_q = (C_q, E_q), \quad E_q = \{(u, v) \in E \mid u, v \in C_q\}  
$$

This subgraph $G_q$ plus text references is given to the LLM to generate the answer.

---

### 4. Worked Toy Example

Mini knowledge graph for probability concepts:

- Nodes:
    
    - $c_1 = \text{"Probability basics"}$
        
    - $c_2 = \text{"Jensen's inequality"}$
        
    - $c_3 = \text{"ELBO (Evidence Lower Bound)"}$
        

Edges (prerequisite relations):

- $(c_1, c_2)$: "Probability basics" → "Jensen's inequality"
    
- $(c_2, c_3)$: "Jensen's inequality" → "ELBO"
    

Student query: "What is ELBO?"

1. Retrieve chunks mentioning ELBO → they contain concept $c_3$.
    
2. GraphRAG expands to include neighbors needed for understanding $c_3$:
    
    - Add $c_2$ (since ELBO depends on Jensen's inequality).
        
    - Add $c_1$ (since Jensen's inequality depends on probability basics). 
        
3. Subgraph $G_q = (\{c_1, c_2, c_3\}, \{(c_1, c_2), (c_2, c_3)\})$.
    

The AI tutor then answers by:

- First explaining “Probability basics”,
    
- Then “Jensen’s inequality”,
    
- Finally “ELBO”,
    
- While citing specific pages/timestamps where each concept appears.
    

---

### 5. Connections & Prerequisites

**Prerequisite Refresher – Embeddings & Retrieval:**  
GraphRAG builds directly on:

- **Embeddings** (to represent chunks and queries as vectors in $\mathbb{R}^d$).
    
- **Vector search** (to find nearest neighbors).
    
- **Graph concepts** (nodes, edges, subgraphs).
    

Your understanding of tokenization and embeddings is essential, because the quality of retrieval and graph construction depends on good chunking and good semantic embeddings.

---

### Key Takeaways & Formulas

- **NLP tasks** like POS tagging, dependency parsing, NER, and coreference resolution provide structure and meaning on top of raw text, and many are modeled as sequence labeling problems:  
    $$  
    P(\mathbf{y} \mid \mathbf{w}) = \prod_t P(y_t \mid w_{1:T}, y_{1:t-1})  
    $$
    
- **Tokenization & vocabularies** convert raw text into token IDs; vocabulary size $|V|$ directly affects model complexity (e.g., output layer size and softmax cost).
    
- **BPE (Byte-Pair Encoding)** builds a subword vocabulary by repeatedly merging the most frequent pair of symbols, trading longer subwords for shorter sequences and reducing out-of-vocabulary issues.
    
- **Embeddings** map token IDs into dense vectors:  
    $$  
    z_j = E^\top e_j, \quad P(w=k \mid \text{context}) = \text{softmax}_k(U z_j)  
    $$  
    where $E$ is the embedding matrix and $U$ projects back to vocabulary logits.
    
- **Language models** assign probabilities to sequences via the chain rule:  
    $$  
    P(w_1, \dots, w_T) = \prod_{t=1}^{T} P(w_t \mid w_{1:t-1})  
    $$  
    and are often implemented with neural networks (RNNs, transformers).
    
- **RNNs** maintain a hidden state over time:  
    $$  
    h_t = \tanh(U x_t + W h_{t-1} + b)  
    $$  
    enabling sequence modeling by combining current input with past context.
    
- **RAG & GraphRAG** for the AI tutor project combine embeddings, vector search, and a knowledge graph $G = (C, E)$ so that queries return not just text chunks but **subgraphs** of concepts and references, supporting structured explanations and learning paths.