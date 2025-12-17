## Part 3: Natural Language Processing Fundamentals

### NLP Task Overview

The professor introduced several core NLP tasks that help extract meaning from text:

**1. Tokenization** Converting words into numbers. The simplest form treats each word as a token, but this isn't optimal (discussed in detail below).

**2. Part-of-Speech (POS) Tagging** Classifying words grammatically (noun, verb, adjective, etc.). Example: "London" → noun, "is" → verb

**3. Lemmatization** Reducing words to their base form (e.g., "running" → "run")

**4. Dependency Parsing** Creating a tree structure from text where:

- A root node is identified
- Words connect to other words showing relationships
- Example: "London is the capital" → "London" is the subject, "capital" is an attribute

**5. Named Entity Recognition (NER)** Identifying and classifying entities in text:

- Countries, dates, organizations, names
- Libraries like spaCy provide this in a single line of code
- Errors occur (e.g., "Dr. Maryland" misidentified as the state)

**6. Coreference Resolution** Linking pronouns/references to their antecedents across potentially long distances in text. Example: Connecting "it" back to "loan" mentioned sentences earlier. This capability is crucial for understanding context over long passages.

---

## Part 4: Tokenization — Byte Pair Encoding (BPE)

### Word-Level vs. Subword-Level Tokenization

**Word-level:** Each word becomes a token

- Problem: Vocabulary becomes extremely large
- Unknown words create issues

**Subword-level (BPE):** Most modern LLMs use this approach

- Originated from 1970s-80s compression research (floppy disk era)
- Goal: Minimize total bits needed to represent text

### How BPE Works

**Simple Example:** Starting text: `AAABCAAABAC`

1. Find frequent byte pairs (consecutive characters)
2. Replace with new symbol
3. Repeat

Steps:

- Replace `AA` → `Z`: Results in `ZABDZABAC`
- Replace `AB` → `Y`: Results in `ZYDZYAC`
- Replace `ZY` → `X`: Results in `XDXAC`

The final representation plus the substitution mappings requires fewer bytes than the original.

**More Realistic Example: "She sells seashells by the seashore"**

1. Count character frequencies:
    
    - Space: 5, S: 5, E: 5, H: 3, A: 2, L: 2, etc.
2. Greedily merge frequent pairs:
    
    - Merge `SH` → `Y` (appears multiple times)
    - New text: `Ye_sells_seaYells_by_the_seaYore`
3. Continue merging:
    
    - Merge `YE` → `Z`
    - And so on...
4. Final result: A **vocabulary V** containing all merged subwords
    

### Why Tokenizer Choice Matters

The professor demonstrated using the **TikTokenizer** web application:

- Same text with GPT-2 tokenizer → 186 tokens
- Same text with GPT-4 tokenizer → fewer tokens (better compression)

**Implications:**

1. **Context window:** Better compression = longer effective context
2. **Performance:** GPT-2 was poor at coding because its tokenizer created many tokens for code; newer tokenizers include coding patterns
3. **Unknown tokens:** BPE greatly reduces unknown subwords compared to word-level tokenization
4. **Spelling mistakes:** Even misspelled words break into subwords, maintaining some meaning (though correction mechanisms are separate)

### Tokenizer-Model Pairing

The tokenizer and model must be matched. A vocabulary created for one model may not work well with another.

---

## Part 5: Word Embeddings — Word2Vec

### The Problem with One-Hot Encoding

With vocabulary size V (e.g., 100,000 words):

- One-hot encoding: Each word is a V-dimensional vector with a single 1

**Problem:** Words with similar meanings have zero dot product:

- `hotel` · `motel` = 0 (they're orthogonal)
- This doesn't capture semantic similarity

### The Goal

Create a vector space where **semantically similar words are close together**.

The professor demonstrated the TensorFlow Embedding Projector:

- 10,000 vocabulary entries mapped to 200-dimensional space
- Visualized in 3D using PCA
- Query "car" → nearest neighbors: "driver," "cars," "automobile," "race," "vehicle"

### Word2Vec Architecture

**Distributional Hypothesis (Firth, 1957):**

> "A word's meaning is given by words that frequently appear close by."

In finance documents, "bank" appears near "money," "institution" In National Geographic, "bank" appears near "river," "water"

**Skip-gram Model Setup:**

- Given a center word W_t, predict nearby context words (window size C)
- Example: Window of ±2 words around the center

**Mathematical Formulation:**

The probability model:

```
P(context | center word, θ) = ∏ P(w_{t+j} | w_t, θ)
```

for j from -C to +C (excluding 0)

**Optimization:** Maximum Likelihood Estimation

```
L(θ) = E_{w_t ~ P_data}[log P_model(context | w_t, θ)]
```

### Network Architecture

```
Input (V-dim)    →    Embedding (D-dim)    →    Output (V-dim)
  one-hot              Z = W^T · X              softmax → P(word)
```

**Key Components:**

1. **Input:** One-hot encoded center word (V-dimensional)
2. **Embedding matrix W:** V × D dimensions — projects to D-dimensional space
3. **Hidden representation Z:** The actual embedding (what we want!)
4. **Lifting matrix W':** D × V — lifts back to vocabulary space
5. **Softmax:** Produces probability distribution over all V words

**Training Process:**

- For each center word, predict multiple context words (4 branches if window = ±2)
- Joint optimization: Sum of cross-entropy losses for all context predictions
- Both W and W' matrices are trained simultaneously

**Final Output:** After training, freeze the **W matrix** (V × D). This is the embedding matrix that gets shared/uploaded for others to use.

### Context-Free Nature

These Word2Vec embeddings are **context-free**: the vector for "bank" is the same whether it appears near "river" or "money." Contextual embeddings (like BERT) came later with Transformers.

---

## Part 6: Language Models & Recurrent Neural Networks

### From Embeddings to Language Models

A **language model** predicts the next token given previous tokens:

```
P_model(w_t | w_{t-1}, w_{t-2}, ..., w_{t-n}, θ)
```

The context (previous tokens) can be very large in modern systems (100,000+ tokens).

### Why RNNs?

Need architectures that:

- Handle sequential data naturally
- Maintain a notion of **state** that evolves with each new input
- Process variable-length sequences

### From Kalman Filters to RNNs

The professor drew a parallel to the Kalman filter's state concept:

- State S_t (e.g., position, velocity of a vehicle)
- State depends on previous state S_{t-1} and actions A_t

For RNNs:

- State becomes **hidden state H_t**
- "Action" is the arrival of the next token X_t
- The function relating states doesn't change over time (unlike fully dynamic models)

### The Recurrent Neuron

**Sigmoidal Neuron (Review):**

```
output = σ(W^T · X + b)
```

**Recurrent Neuron:**

```
H_t = tanh(U^T · X_t + W · H_{t-1} + b)
```

**Key differences:**

- Uses **tanh** instead of sigmoid (outputs range [-1, 1] instead of [0, 1])
- **U:** weights for current input
- **W:** weights for previous hidden state (the recurrence!)
- **H_{t-1}:** stored in memory, retrieved for each computation

**Trainable Parameters:** U (vector), W (scalar for single neuron), b (bias)

### Unrolling the RNN

A feedback loop can be "unrolled" into a feedforward-like structure:

- At time 1: Input X_1 → Hidden state H_1
- At time 2: Input X_2 + H_1 → Hidden state H_2
- ...
- At time 50: Input X_50 + H_49 → Final hidden state H_50 = ŷ

This reveals the RNN as a very deep network with **shared weights** across time steps.

**Sequential Processing:** H_2 needs H_1, H_3 needs H_2, etc. — inherently sequential (limitation addressed by Transformers later).

### Time Series Prediction Example

**Problem:** Given 50 commodity prices, predict the 51st.

**Challenge:** A single hidden state (scalar) cannot capture all factors affecting price:

- Macroeconomic indicators
- Company-specific factors (revenue, cash flow)
- Market sentiment

**Solution:** The hidden state must be a **vector**, not a scalar.

### From Single Neuron to RNN Layer

Just as we went from a single sigmoidal neuron to dense layers, we go from a single recurrent neuron to **RNN layers**:

1. **Make hidden state a vector H_t**
2. **Decouple hidden state dimensionality from output dimensionality**
    - Hidden state: Many dimensions (captures latent factors)
    - Output: May be 1 dimension (the predicted price)

### Determining Hidden State Dimension

Interview question approach: "How many dimensions do I really need?"

This connects to:

- **Covariance matrices** and dimensionality reduction
- **Principal Component Analysis (PCA)**
- If your data matrix is N × M, how many principal components capture the variance?

The answer depends on the intrinsic complexity of the problem—there's no universal formula, but techniques exist to estimate it.

---

## Key Takeaways

1. **Tokenization matters:** BPE enables efficient representation with smaller context windows and better handling of novel words
    
2. **Word2Vec creates semantic spaces:** Similar words cluster together through unsupervised learning on co-occurrence patterns
    
3. **RNNs process sequences:** By maintaining hidden state, they can handle variable-length inputs and capture temporal dependencies
    
4. **Hidden state dimensionality:** Must be chosen based on the complexity of latent factors in your problem
    
5. **The progression:** Tokenization → Embedding → Sequence Modeling (RNN) → (next: Transformers)
    

---

## Looking Ahead

The professor noted that while RNNs aren't used in modern large language models, they're instructive for understanding **Transformers**, which will be covered next. Transformers address key RNN limitations including sequential processing (no parallelization) and gradient flow through long sequences.