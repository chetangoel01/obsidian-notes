# Lecture 7 Professor Notes
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


# Lecture 8 Professor Notes
## 2. Review: Tokenization and Embeddings

### Tokenization Recap

The professor reviewed tokenization—the process of taking words and breaking them into integer representations via sub-word encoding (byte-pair encoding). Key clarifications:

- The selection of which character to merge next (e.g., "SH" after "S") is determined by what characters most frequently follow in the corpus
- The stopping criterion is simple: you stop when you reach your desired vocabulary size (10,000, 100,000, etc.)

**Information Theory Connection:** The professor mentioned that tokenization connects deeply to **Shannon's source coding theorem**, which governs video compression, audio compression, and waveform representation through rate-distortion theory. Byte-pair encoding is actually a special case of entropy-based methods. A related technique called **Minimum Description Length** is entirely entropy-based. While byte-pair encoding was chosen for language models because it's simple and efficient, video tokenization requires deeper information theory background.

### Embeddings Recap

The embedding process maps tokens into a D-dimensional vector space (R^D). The key insight reviewed:

- **The meaning of a word depends primarily on what shows up next to it**
- A training window is defined around each word
- The center word is used to predict surrounding tokens
- This creates a **P_data distribution** (conditional probability distribution of context words given center word)

The architecture involves:

1. **Embedding layer**: Maps center word to lower-dimensional vector Z
2. **Lifting layer**: Uses a matrix to produce posterior probability distribution over vocabulary

Through maximum likelihood training:

- The P_model (neural network) must satisfy multiple predictions jointly
- Joint optimization of all trainable parameters occurs simultaneously
- The embedding matrix W* is the final artifact, sent to hubs with qualifiers (e.g., "trained on Wikipedia English")

**Critical point:** A single token like "bank" will have ONE embedding that's effectively a mixture of all its meanings (financial institution, river bank, etc.) from the training data. This is a limitation that contextual embeddings (via attention) will later address.

---

## 3. Language Modeling with Recurrent Neural Networks

### Why RNNs?

While many dismiss RNNs, the professor emphasized they remain relevant for:

1. **Time series prediction** beyond NLP
2. **Connection to state space models** (Hidden Markov Models, Kalman filters)
3. **Competitive architectures** to transformers that are still being researched

### The State and Memory Concept

The RNN models a function F that depends on:

- Current input X(t) (the arrival of a token)
- Previous hidden state H(t-1)

Since H(t) depends on H(t-1), which depends on H(t-2), and so on, RNNs have **memory**—the question is how long this memory survives (which connects to gradient flow).

### Basic RNN Architecture

Starting from a simple sigmoidal neuron, a feedback stage is introduced:

**Why tanh instead of sigmoid?**

- Tanh has output range [-1, 1] (positive and negative)
- Sigmoid has output range [0, 1] (many zeros in feedback loop)
- We prefer to avoid zeros propagating through feedback

### The RNN Layer Equation

For a layer with multiple neurons:

**H(t) = tanh(W · [X(t); H(t-1)] + b)**

Where:

- W is the combined weight matrix (concatenation of U transpose and W' transpose)
- U is N_neurons × N_input (processes current input)
- W' is N_neurons × N_neurons (processes previous hidden state)
- [X(t); H(t-1)] is vertical concatenation of input and previous hidden state
- b is the bias vector
- tanh is applied element-wise

**Dimensionality guidance:** The number of neurons (dimension of H) can be estimated empirically by:

1. Running SVD on the data matrix
2. Looking at the spectrum of singular values
3. Keeping dimensions that capture significant variance

### Unrolling the RNN

The RNN is conceptualized as an "unrolled" network through time:

- Each column represents one time step
- The same weights W are shared across all time steps
- This creates a very deep network when considering dependencies across many steps

### The Prediction Head

After the RNN produces H(t), predictions are made via:

**O(t) = V · H(t) + c**

Then:

- **For classification**: Y_hat(t) = softmax(O(t)) — produces K-dimensional posterior over classes
- **For regression**: Y_hat(t) = E · O(t) — simple dot product for scalar prediction

---

## 4. The Gradient Flow Problem

### The Central Question: Can Simple RNNs Capture Long-Term Dependencies?

The professor drew a two-stage unrolled RNN to analyze gradient flow during backpropagation through time (BPTT).

**Key insight:** The contribution of a previous stage to the output depends on how much gradient reaches it:

- No gradient → No parameter updates → Hidden state not improved
- The network essentially ignores that time step's information

### The Culprit: The W' Matrix

The W' matrix (N_neurons × N_neurons) that processes the feedback is the critical factor. Its **eigenvalue decomposition** determines gradient behavior:

**If eigenvalues of W' < 1:** Gradient **diminishes** (vanishing gradient)

**If eigenvalues of W' > 1:** Gradient **explodes** (exploding gradient)

Both scenarios are problematic:

- **Vanishing gradient:** Early time steps don't receive meaningful updates; their information is effectively lost
- **Exploding gradient:** Training becomes unstable due to dynamic range issues and clipping

### Simple Scalar Intuition

For a scalar W:

- If W = -0.9, H(t) diminishes over time (envelope decreases)
- If |W| > 1, H(t) explodes over time

---

## 5. Long Short-Term Memory (LSTM) Architecture

The LSTM was developed to solve the gradient flow problem by introducing controlled gates and a **highway** for gradient propagation.

### Key Innovation: Two Hidden States

1. **S(t)**: Long-term memory state (lives on the highway)
2. **H(t)**: Short-term memory state

### The Gate Mechanism

LSTMs introduce three gates, all using **sigmoidal activation** (output between 0 and 1):

**1. Input Gate:**

- Controls what new information enters the cell
- Multiplies incoming information with [0,1] values
- Softly selects what to keep vs. discard from new inputs

**2. Forget Gate:**

- Controls what to remember/forget from long-term memory
- Modulates the S pathway (the only thing affecting the highway)
- Determines how much of S(t-1) to propagate

**3. Output Gate:**

- Controls how much of H(t) propagates to the next step
- Determines what's revealed to the rest of the network

### The Highway Concept

The S(t) state travels on a "highway"—a direct path with minimal interference:

- Only the forget gate modulates it
- Gradient can flow through this highway relatively unimpeded
- This is similar to **skip connections in ResNets** (another highway network)

### Intuitive Example: Summarization

For the input: "The bank, faced with political pressure over interest rates, introduced annual savings account."

The desired summary: "The bank introduced savings account."

The input gate learns to **not allow** the middle portion ("faced with political pressure over interest rates") to affect the long-term or short-term memory, since it doesn't carry essential meaning for the summary.

**Note:** The professor mentioned LSTM likely won't be on the final exam since the coverage was superficial—just the conceptual diagram without detailed equations.

---

## 6. Language Modeling Task with RNNs

The simple RNN language model architecture:

1. **Input**: Token W(t-n) (from context)
2. **Embedding Layer**: Maps to X(t-n)
3. **RNN/LSTM Layer**: Processes with hidden state from previous step
4. **Classification Head**: Produces Y_hat (posterior over vocabulary V)

This unrolled architecture produces predictions at each step, allowing us to train by comparing Y_hat with ground truth tokens.

**Character-level RNN:** The professor referenced Andrei Karpathy's famous code (from his Stanford days) that demonstrates character-level language modeling—starting from garbage predictions and gradually learning to predict the next character correctly.

---

## 7. Neural Machine Translation (Sequence-to-Sequence)

### Encoder-Decoder Architecture

Neural machine translation exemplifies **sequence-to-sequence** models with two components:

**Encoder:**

- Processes source language tokens (X1, X2, X3)
- Produces a final hidden state H3 called the **thought vector (φ)**
- The thought vector captures the entire meaning of the input sequence

**Decoder:**

- Receives the thought vector as initial state
- Generates target language tokens (X'1, X'2, X'3) one at a time
- Uses special tokens: **SOS (Start of Sentence)** and **EOS (End of Sentence)**

### Teacher Forcing

**During Training:**

- Feed the **ground truth** tokens to decoder inputs
- We have supervision at every step

**During Inference:**

- Feed the **previous predicted token** to the next step
- No ground truth available; blue arrows connect predictions to next inputs

### The Reverse Order Trick

Tokens are often fed in **reverse order** (X3, X2, X1) to the encoder. Why?

**Grammar alignment:** Many languages follow Subject-Verb-Object order. The first word of the source is often the first word of the target. By feeding the first word last:

- It's the most "recent" information when forming the thought vector
- Gradient flow to this information is strongest
- Better footing for generating the first output word

### Bidirectionality

**Why bidirectional?** The meaning of a word can depend on what comes **before** AND **after** it.

**Example:**

- "George Washington crossed the Delaware" → George Washington = person (determined by what follows)
- "The George Washington Bridge is closed to traffic" → George Washington = structure (determined by what follows)

Bidirectional RNNs compute:

- **φ_f**: Thought vector from forward direction
- **φ_r**: Thought vector from reverse direction
- **φ**: Concatenation of both, capturing both preceding and following context

### Limitation of Single Thought Vector

The entire decoding process depends on **one vector** (φ). This was a major limitation that attention mechanisms (covered next) would address.

---

## 8. Beam Search (Maximum Likelihood Sequence Estimation)

### The Problem with Greedy Decoding

**Greedy approach:** At each step, select the token with maximum posterior probability.

This doesn't guarantee the best **sequence** of predictions. The professor showed an example:

- Greedy path: "the ship has docked" (likelihood = 0.33)
- Alternative path: (likelihood = 0.36)

The alternative with lower individual step probabilities actually has higher **sequence likelihood**.

### Beam Search Solution

Instead of tracking only the best prediction:

- Maintain **log probabilities** of **K branches** (K is a hyperparameter)
- Track multiple candidate sequences simultaneously
- At the end (EOS token), select the sequence with highest total likelihood

**Tradeoff:** Beam search is computationally expensive—branches grow quickly. K is typically kept small (e.g., 2-5).

This is also known as:

- **MLSE** (Maximum Likelihood Sequence Estimation)
- **Viterbi algorithm** (named after the inventor)

---

## 9. Evaluation: BLEU Metric

**BLEU (Bilingual Evaluation Understudy)** measures translation quality using **n-gram overlaps** with human reference translations.

### How BLEU Works

Compare machine translation (C) with human translations:

**Example:**

- Machine: "the plane blew in from Athens"
- Human 1: "the plane took off from Athens"
- Human 2: "the plane departed from Athens"

Count bigram matches:

- "the plane" → True positive (appears in both)
- "plane blew" → False positive
- "blew in" → False positive
- "in from" → False positive
- "from Athens" → True positive

**Precision-based formula:** True Positives / (True Positives + False Positives)

### Gaming Prevention

Machines could game simple n-gram counting (e.g., repeating "the plane the plane"). The full BLEU formula includes:

- Penalties for overly short translations
- Penalties for repetition
- More sophisticated n-gram counting

---

## 10. Introduction to Transformers

The professor introduced transformers as the "final destination" architecture with three key innovations:

### 1. Eliminating Recurrent Connections

No more H(t) depending on H(t-1):

- Serial architecture → **Parallel architecture**
- Enables processing all tokens simultaneously
- Dramatic speedup in training and inference

### 2. Positional Encoding

Since transformers process all tokens in parallel:

- They are **permutation invariant** by default
- Must explicitly inject position information
- Position encodings tell the model where each token is in the sequence

### 3. Contextual Embeddings via Attention

The most important innovation: **Attention mechanism** creates context-dependent embeddings.

**The Problem:** In Word2Vec, "bears" has ONE embedding regardless of context:

- "I love bears" (animal)
- "He bears the pain" (tolerance/endurance)
- "Bears won the game" (sports team - Chicago Bears)

**The Solution:** Attention mechanism pushes tokens to **different positions** in D-dimensional space based on surrounding context.

### Simple Attention Mechanism (Preview)

Given tokens arranged in matrix X (D × T, where T = context size):

**Step 1: Compute Score Matrix** S = X · X^T (T × T matrix)

This dot product measures how similar/related each pair of tokens is.

**Step 2: Apply Softmax (row-wise)** A = softmax(S)

Creates **attention weights** with properties:

- A_ij ≥ 0
- Sum over j of A_ij = 1 (each row sums to 1)

**Step 3: Create Contextual Embeddings** X_hat = A · X

Or equivalently: X_hat = softmax(X · X^T) · X

### Key Interpretation

**What attention accomplishes:**

- Started at one location in D-dimensional space (context-free embedding)
- Ended at **another location** in D-dimensional space (contextual embedding)
- The attention mechanism **pushed** the token based on weights from other tokens

**Current limitation of simple attention:** No learnable parameters—the mapping is deterministic based on context-free embeddings. Next lecture will introduce **Query, Key, Value projections** (learnable matrices) that provide control over this mapping.

---

## Key Takeaways

1. **RNNs** model sequences but suffer from gradient flow problems (vanishing/exploding)
2. **LSTMs** solve this with gates and a highway for long-term memory
3. **Encoder-decoder** architectures enable sequence-to-sequence tasks like translation
4. **Teacher forcing** uses ground truth during training, predictions during inference
5. **Beam search** finds better sequences than greedy decoding
6. **BLEU** evaluates translations via n-gram overlap with human references
7. **Transformers** eliminate recurrence, add positional encoding, and use attention for contextual embeddings
8. **Attention** is fundamentally about moving tokens in embedding space based on context

---

_Next lecture: Enhanced attention mechanism with Query, Key, Value projections (Q, K, V) and multi-head self-attention._

# Lecture 9 Professor Notes
## Part 2: Transformer Self-Attention Mechanism

### Motivation: Context-Dependent Meaning

The professor introduced transformers with an example: the word "bears" appears in three sentences but has completely different meanings:

1. "I love bears" → Animal
2. "Bears pain" → Verb (to tolerate/endure)
3. "Bears won the game" → Sports team

The goal of attention is to **move each token from its initial context-free position** in the D-dimensional embedding space **to a context-appropriate position** that reflects its meaning in that specific sentence.

### Simple Self-Attention (without Q, K, V)

In the simplest form:

- Stack all context tokens into matrix **X** (dimensions T × D, where T = context size, D = embedding dimension)
- Compute **X · X^T** to get a score matrix (similarity between all pairs of tokens)
- Apply **softmax** row-wise to normalize scores into attention weights (each row sums to 1, values between 0 and 1)
- This creates competition: only tokens that are truly helpful contribute significantly
- Compute **X̂ = softmax(X · X^T) · X** — the weighted sum that moves each token to its new position

This is called **self-attention** because each token both provides help to other tokens and receives help from other tokens (including itself).

**Critical Limitation**: Up to this point, the position of X̂_i is determined entirely by context-free embedding vectors—we need parameterization to learn appropriate transformations.

---

## Part 3: Query, Key, Value (Q, K, V) Parameterization

### Intuition Through Grammar Roles

Think of each token as needing to communicate three pieces of information:

1. **Key (K)**: "What am I?" — Conveys the token's identity/role (e.g., "I am an object," "I am a verb")
2. **Query (Q)**: "What am I looking for?" — What kind of token would help me understand my meaning (e.g., "I'm looking for a verb")
3. **Value (V)**: "Where do I start?" — The starting point for the transformation, which may already have some notion of meaning

### Example Walkthrough

In "I love bears" where bears = animal:

- Bears is grammatically an **object**
- The **key** for bears points along the "objectness" axis
- The **query** for bears asks "I'm looking for a verb" (since verbs help determine if an object is animate/inanimate)
- The token "love" emits a key saying "I'm a verb"
- Because query and key are aligned (verb-seeker meets verb), the dot product is **large** → high attention weight

The professor used the analogy: "It's almost like a romantic advertisement—I'm a male, 23, looking for a female, age 20..."

### Linear Projections for Q, K, V

Each is computed via separate trainable matrices:

- **Q = X · W_Q** (Query matrix)
- **K = X · W_K** (Key matrix)
- **V = X · W_V** (Value matrix)

All three matrices have dimensions T × D. The W matrices are learned during training.

Why linear projections work: Earlier in the course, the class saw how linear projections with matrices W and W' could map vectors to neighborhoods that carry specific meanings (like mapping word embeddings near semantically related concepts).

### Attention Computation with Q, K, V

1. Compute **score matrix S = Q · K^T** (dimensions T × T)
2. **Scale**: Divide by √D (explained below)
3. Apply **softmax** row-wise to get **attention weights A** (T × T)
4. Compute **V̂ = A · V** (dimensions T × D)

### Why Scale by √D?

Without scaling, softmax outputs become very sparse (only 1-2 tokens attend to each other). With scaling:

- The attention distribution is smoother
- Many tokens can contribute to positioning

**Analytical justification**: If inputs have variance 1, the dot product increases variance by factor D. Dividing by √D restores variance to 1.

**Numerical demonstration**: The professor showed that multiplying inputs by 100 (simulating no scaling) results in extremely sparse softmax outputs where essentially only one token helps. With proper scaling, the distribution is more balanced.

---

## Part 4: Masking for Decoder Architectures

In **decoder-only** (autoregressive) models:

- During training: All tokens are available
- During inference: Only past tokens are available

To avoid "cheating" during training, we implement **masking**:

- When at position 8, only receive attention from positions 1-7
- When at position 7, only from positions 1-6, etc.

**Implementation**: Set attention weights for future tokens to 0 (or equivalently, set the input to softmax to a very large negative number like -∞, which softmax converts to 0).

This is why decoder attention is called **Masked Self-Attention**.

---

## Part 5: Single-Head to Multi-Head Self-Attention

### Single-Head Self-Attention

The complete block diagram of single-head attention:

1. Input X (T × D)
2. Project to Q, K, V using W_Q, W_K, W_V
3. Compute S = Q · K^T (T × T)
4. Scale: S / √D
5. Apply softmax → A (T × T)
6. Compute V̂ = A · V (T × D)

### Why Multiple Heads?

Just as CNNs need multiple filters to extract different spatial patterns, transformers need multiple attention heads to extract different **temporal/sequential correlation patterns**.

Each head might focus on different grammatical or semantic relationships (though the network learns these without explicit labels).

### Multi-Head Self-Attention (MHSA)

- Run H single-head attention mechanisms **in parallel**
- Each head h has its own projection matrices: W_Q^(h), W_K^(h), W_V^(h)
- **Concatenate** all head outputs
- Pass through a **mixing matrix** (D × D) to produce final V̂ (T × D)

The number of heads H varies by architecture (early transformers used 8; modern ones may use more).

**Key Properties**:

1. Heads work in parallel to extract representations across different patterns
2. Final output is a linear combination of all single-head outputs
3. Typical architectures have 8-32+ heads

---

## Part 6: Layer Normalization and the Complete Transformer Block

### Block A: Multi-Head Self-Attention with Residuals

The MHSA block is decorated with:

1. **Layer Normalization** at the input
2. **Skip connection** (residual connection) around the MHSA

```
X → LayerNorm → MHSA ─┬→ Z
        ↑              │
        └──────────────┘ (skip connection)
```

### Batch Normalization (Review)

To understand layer normalization, the professor reviewed batch normalization:

**Key insight from backpropagation**: The gradient of a neuron's output with respect to weights W is **proportional to the inputs X**. This means:

- If inputs have very small values → small gradients → slow learning
- If inputs have very different scales → uneven learning

**Batch normalization** standardizes activations across a batch:

1. Compute mean μ and variance σ² across the batch dimension
2. Standardize: X̂ = (X - μ) / √(σ² + ε)
3. Scale and shift with **trainable parameters** γ and β: Output = γ · X̂ + β

The γ and β let the network learn the optimal distribution for each layer.

### Layer Normalization

Problem with batch normalization in transformers:

- Context sizes are huge → limited batch sizes due to VRAM constraints
- During inference, requests may come one at a time
- Can't extract reliable statistics from tiny batches

**Solution**: Layer normalization computes mean and standard deviation **across the feature dimension** (D) rather than the batch dimension.

This works even with batch size = 1.

**Modern variant**: RMS Normalization (similar function)

### Block B: MLP with Nonlinearity

Up to this point, all operations were **linear**. We need nonlinearity.

Block B:

1. **Layer Normalization** on Z
2. Pass through **MLP** (Multi-Layer Perceptron) with nonlinear activation (like GELU or ReLU)
3. **Skip connection** around the MLP

```
Z → LayerNorm → MLP ─┬→ V̂
        ↑            │
        └────────────┘ (skip connection)
```

### Complete Transformer Layer

One **Masked Transformer Layer (MTL)** = Block A + Block B in series.

A complete transformer **body** stacks multiple MTLs:

- MTL₁ → MTL₂ → ... → MTL_L
- Typical architectures use ~32 layers

### The Transformer Head

After the body builds representations, we attach a **head** for the specific task:

1. Linear layer with matrix W
2. **Softmax** across vocabulary size

This produces:

- **ŷ**: Posterior probability distribution over all tokens in vocabulary
- Select next token via greedy decoding or beam search

---

## Part 7: Positional Embeddings

### The Problem

Without positional information, transformers are **permutation invariant**—shuffling input tokens gives the same output. Word order matters in language!

### Solution: Add Positional Encodings

For token i with context-free embedding X_i:

```
X̃_i = X_i + R_i
```

Where R_i is the **positional encoding vector** for position i.

### Why Addition Instead of Concatenation?

1. Concatenation would increase dimension D (more computation)
2. Addition works if R_i doesn't drastically disrupt X_i (values bounded between -1 and 1)
3. Linear operations preserve the sum: W(X + R) = WX + WR

### The Fourier Method (Sinusoidal Positional Encoding)

The formula uses sinusoids of decreasing frequency:

- Element n of position i's encoding:
    - If n is even: sin(i / L^(n/D))
    - If n is odd: cos(i / L^((n-1)/D))

Where L relates to context size T.

**Visualization**:

- First dimension: Highest frequency (alternates rapidly)
- Second dimension: Half the frequency
- Third dimension: Quarter the frequency
- And so on...

**Intuition via Binary Encoding**: If you encode positions 1, 2, 3, 4... in binary:

```
Position 1: 0001
Position 2: 0010
Position 3: 0011
Position 4: 0100
...
```

The **rightmost bit** alternates every position (highest frequency). The next bit alternates every 2 positions (half frequency), etc.

The Fourier method is the **analog/continuous version** of binary position encoding. If you threshold the sinusoids, you'd get binary encoding.

**Advantage**: Values stay bounded in [-1, 1], providing smooth gradations rather than discrete jumps.

### Alternative: Learnable Positional Embeddings

Instead of fixed sinusoids, learn the positional vectors as parameters. Both approaches work; the Fourier method has the advantage of potentially generalizing to longer sequences than seen during training.

---

## Part 8: Mixture of Experts (MoE)

### Connection to Ensemble Methods

In residual networks (ResNets), the class studied **ensemble methods** with weak predictors that, in aggregate, produce predictions much better than individual ones.

**Best case**: Each ensemble member makes **uncorrelated mistakes** **Worst case**: All members make the **same mistake**

### What is Mixture of Experts?

MoE is a **conditional ensemble** where different "experts" (specialized sub-networks) handle different parts of the data distribution.

### Comparison with Standard Ensembles

- **Standard ensemble**: All weak predictors try to model the entire P_data distribution
- **Mixture of Experts**: Each expert specializes in a **partition** of P_data

### Connection to Mixture of Gaussians

A mixture of Gaussians models complex distributions as:

```
P(x) = π₁ · N(x | μ₁, σ₁²) + π₂ · N(x | μ₂, σ₂²) + ...
```

Where π_i are mixing coefficients (sum to 1).

Each Gaussian component has "soft responsibility" for different regions of the data.

### MoE Formulation

```
ŷ = Σᵢ₌₁ᴷ Gᵢ(x) · Fᵢ(x)
```

Where:

- K = number of experts
- **Fᵢ(x)** = prediction from expert i
- **Gᵢ(x)** = gating function (like attention weights)
- Gᵢ(x) ≥ 0 and Σᵢ Gᵢ(x) = 1

**Key difference from ensembles**: The gating network G is **trainable**—it learns which experts should handle which inputs.

### Why MoE Matters Now

MoE provides significant efficiency benefits:

- Not all experts need to be activated for every input (sparse activation)
- Reduces memory requirements on expensive accelerators
- Popularized by **DeepSeek** and other recent architectures

The professor noted that while the block diagram perspective is straightforward, the theoretical analysis of MoE with **cross-entropy loss** (rather than mean squared error) is an open area worth studying.

---

## Part 9: Looking Ahead

The professor outlined remaining topics:

1. **Vision Transformers (ViT)**: Very similar architecture to language transformers, but interpretation differs (can't use subject/verb/object analogies for images); natural interpretation of multiple heads exists
2. **Symbolic Reasoning**: Classical logical reasoning
3. **Neurosymbolic Reasoning**: Combining neural networks with symbolic reasoning to reduce hallucinations
4. **Planning**: Planning without interactions
5. **Reinforcement Learning**: Planning with interactions

---

## Key Equations Summary

**Self-Attention (with Q, K, V)**:

```
Q = X · W_Q
K = X · W_K
V = X · W_V
A = softmax(Q · K^T / √D)
V̂ = A · V
```

**Layer Normalization**:

```
μ = mean across feature dimension
σ² = variance across feature dimension
X̂ = (X - μ) / √(σ² + ε)
Output = γ · X̂ + β
```

**Sinusoidal Positional Encoding**:

```
R_i[n] = sin(i / L^(n/D))     if n is even
R_i[n] = cos(i / L^((n-1)/D)) if n is odd
```

**Mixture of Experts**:

```
ŷ = Σᵢ Gᵢ(x) · Fᵢ(x)
where Σᵢ Gᵢ(x) = 1, Gᵢ(x) ≥ 0
```

# Lecture 10 Professor Notes
## Part 2: Review of Transformers and Context-Free Representations

The professor provided a review connecting to material from previous lectures:

### Context-Free vs. Context-Rich Embeddings

- The journey started from **context-free representations** (like Word2Vec embeddings)
- When input enters a transformer's embedding layer, the output is initially **context-free** – it's simply an embedding
- The transformer architecture then creates **context-rich** representations through the attention mechanism

### The Purpose of Linear Projections (WQ, WK, WV)

The key insight is understanding _why_ we project input embeddings into Query (Q), Key (K), and Value (V) spaces:

**Analogy to PCA**: Just like Principal Component Analysis finds a new coordinate system that best represents data variance, the W matrices (WQ, WK, WV) learn coordinate systems that capture semantic meaning.

**Word2Vec properties to remember:**

- Words with similar meanings end up in the same neighborhood
- Analogies manifest as vector operations (e.g., "king" - "man" + "woman" ≈ "queen")

**The professor's analogy**: Imagine the axes in the new coordinate space represent different semantic dimensions – perhaps an "objectness" axis, a "verbness" axis, etc. The learned projections create coordinates where similar meanings are nearby.

### The Cafeteria Analogy for Attention

The professor offered an intuitive explanation of the attention mechanism:

1. You (representing a query) walk into a cafeteria full of professors
2. You raise your hand asking for help with a math problem
3. Professors who can help (keys that match your query) come over
4. Before receiving help, your knowledge was X
5. After receiving their assistance, your knowledge has been enriched (V̂)

This illustrates how attention allows tokens to gather relevant information from other tokens.

### Multi-Head Attention

Different attention heads can specialize in different patterns:

- One head might focus on grammatical patterns
- Another might focus on different types of relationships
- **Analogy to CNNs**: Just as convolutional filters detect different spatial patterns, attention heads detect different temporal/sequential patterns

**Code analogy**: One head might extract comments in Python code, while another focuses on Python keywords. This isn't strictly true (due to mixture of experts effects), but provides useful intuition.

---

## Part 3: Mixture of Experts (MoE)

The professor briefly covered the Mixture of Experts architecture, relating it to ensemble methods:

### Ensemble Error Analysis

When combining K experts/predictors, the mean squared error of the ensemble depends on:

- **V**: the variance of individual predictors
- **C**: the correlation of their mistakes

**Best case (C = 0)**: When experts make **uncorrelated mistakes**, the ensemble error is V/K – error decreases linearly with the number of experts

**Worst case (C = V)**: When experts make the same mistakes (perfectly correlated), having K experts is no better than having 1

### Key Insight

The benefit of multiple experts comes from **diversity** – they need to make different mistakes. In practice, perfect decorrelation is never achieved because:

- Training data is finite
- Similar tokens may be routed to different experts
- Different routing policies (sparse vs. dense) affect the correlation

**Reference**: The professor recommended the **DeepSeek MoE** paper for detailed block diagrams and equations on what changes in the baseline transformer architecture when using MoE.

---

## Part 4: Vision Transformers (ViTs)

The professor covered how transformers adapt from NLP to computer vision:

### Tokenization in Vision

Unlike NLP where tokens are words/subwords, in ViTs:

- Images are divided into a **grid of patches** (typically 16×16 pixels per patch)
- Each patch becomes a token
- For a 224×224 image with 16×16 patches, you get 196 tokens

**Interesting finding**: Whether you arrange the patch tokens in a specific spatial order or randomly shuffle them makes little difference to performance – the positional embeddings handle the spatial information.

### Attention Visualization in ViTs

The attention matrix A can be visualized to understand what each query attends to:

- White = high attention (close to 1)
- Black = low attention (close to 0)

**Example from the lecture**: For an image of an antelope, a query vector at the antelope's body will show high attention to other patches of the antelope (similar color, belonging to the same object).

### Multi-Head Intuition is More Natural in Vision

The interpretation of multiple heads is clearer in computer vision:

- One head might focus on **color patterns**
- Another on **texture patterns**
- Another on **shape patterns**

**Reference**: Chapter 26 of the free "Foundations of Computer Vision" book (MIT professors). Note: The book doesn't include Python implementations – the professor invited students to contribute implementations during winter break.

---

## Part 5: Introduction to Logical Reasoning

### Course Roadmap

The professor outlined the remaining topics:

1. **Logical/Symbolic Reasoning** (current lecture)
2. **Classical Planning** (using PDDL language)
3. **Planning with Interactions** (Markov Decision Processes)
4. **Reinforcement Learning**

Note: **Neurosymbolic reasoning** (combining neural and symbolic approaches to reduce hallucinations) is an important modern topic but won't be covered due to time constraints.

### Modern Applications of Logical Reasoning

Despite being developed in the 1970s, logical reasoning has experienced a **comeback**:

**AWS Example**: Amazon uses logical reasoning (called "Zelkova") to answer cybersecurity queries like "Is this Docker container connected to the internet?" When managing millions of containers/servers, automated logical reasoning over infrastructure configurations is essential.

---

## Part 6: Propositional Logic Fundamentals

### Syntax vs. Semantics

**Syntax**: How to represent logical sentences

- Propositional symbols (evaluate to true/false)
- Operators: negation (¬), conjunction (∧), disjunction (∨), implication (→), biconditional/double implication (↔)

**Semantics**: How to determine if a sentence is true

- Based on **models** (assignments of truth values to all symbols)
- A model **satisfies** a sentence when it evaluates to true

### Models

A **model** is a specific assignment of truth values to propositional symbols.

**Example**:

- Symbols: R (rain), W (wet)
- One model: {R = true, W = true}
- This model satisfies the sentence "R → W" (if it rains, it's wet)

**Multiple models**: A sentence can be satisfied by multiple models. The set of all models satisfying a sentence is denoted M(sentence).

### Truth Table for Implication (Critical!)

The implication (P → Q) truth table often confuses students:

|P|Q|P → Q|
|---|---|---|
|F|F|T|
|F|T|T|
|T|F|F|
|T|T|T|

**Intuitive explanation using rain/clouds**:

- "If it's not raining, it may or may not be cloudy" → True in both cases
- "If it is raining and it's not cloudy" → False (contradiction)
- "If it is raining and it is cloudy" → True

The only way implication is false is when the antecedent (P) is true and the consequent (Q) is false.

**Biconditional (↔)**: P ↔ Q means (P → Q) ∧ (Q → P). True when both sides have the same truth value.

---

## Part 7: The Wumpus World – A Logical Reasoning Demonstration

### The Game Setup

A classic AI problem from the 1970s demonstrating logical reasoning:

**Environment**: 4×4 grid world containing:

- **Agent**: Starts at (1,1), can move, turn, and sense
- **Wumpus (monster)**: Located in one cell, is smelly
- **Pits**: Fatal if entered, cause breeze in adjacent cells
- **Gold**: The goal – grab it and exit

**Perception**:

- **Stench (S)**: Sensed in cells adjacent to the Wumpus
- **Breeze (B)**: Sensed in cells adjacent to pits
- **Glitter**: Sensed when gold is in the current cell

**Constraint**: The agent is conservative – it cannot move unless it is **certain** the next cell is safe.

### Building the Knowledge Base

The Knowledge Base (KB) stores rules and inferences as logical sentences:

**Initial Rules (before any movement):**

- R0: ¬W₁₁ (no Wumpus at start)
- R1: ¬P₁₁ (no pit at start)
- R2: B₁₁ ↔ (P₁₂ ∨ P₂₁) (breeze at 1,1 iff pit at adjacent cell)
- R3: B₂₁ ↔ (P₁₁ ∨ P₂₂ ∨ P₃₁) (similar rule for cell 2,1)
- Similar rules for stench and other cells...

### Agent Reasoning Walkthrough

**State 1**: Location L₁₁, facing East

- Percept: No breeze, no stench
- Inference: No pit in adjacent cells, safe to move
- Action: Move forward to (2,1)

**State 2**: Location L₂₁, facing East

- Percept: **Breeze detected** (B₂₁ = true)
- Inference: There's a pit in P₃₁ OR P₂₂ (can't determine which)
- Action: **Go back** (can't move forward safely)

**State 3**: Back at L₁₁, now facing West

- Percept: No breeze (reconfirmed)
- Inference: No pit at P₁₂ (since no breeze at origin)
- Action: Turn and move up to (1,2)

**State 4**: Location L₁₂, facing North

- Percept: **Stench detected** (S₁₂ = true)
- Inference: Wumpus is adjacent → could be at (1,3) or (2,2)
- **BUT** we were at (2,1) earlier and detected no stench
- **Therefore**: Wumpus must be at (1,3), NOT (2,2)
- Also: No breeze → No pit at (2,2)
- Action: Turn and move toward (2,2)

The agent continues navigating, eventually reaching the gold at (2,3).

**Key insight**: Rules act as **constraints**. More rules in the KB make it harder to find satisfying models, but they enable more precise inference.

---

## Part 8: Knowledge Base Operations

### The Two Operations: Tell and Ask

**Tell(KB, sentence)**: Store a sentence in the knowledge base

Three possible responses:

1. **"I knew that"** (Entailment) – The sentence was already implied by KB
2. **"I didn't know that, updating"** (Contingency) – New information added
3. **"I don't believe that"** (Contradiction) – Sentence conflicts with KB

**Ask(KB, sentence)**: Query the knowledge base

Three possible responses:

1. **True** – KB entails the sentence
2. **False** – KB entails the negation
3. **"I don't know"** – KB entails neither the sentence nor its negation

### Venn Diagram Visualization

The professor used Venn diagrams to visualize KB operations:

**M(KB)**: The set of all models that satisfy the knowledge base

**Entailment (KB ⊨ α)**:

- M(α) completely contains M(KB)
- Adding α doesn't shrink the satisfying models
- The KB already "knew" this

Formally: M(KB) ∩ M(α) = M(KB)

**Contingency**:

- M(α) partially overlaps M(KB)
- Adding α shrinks the satisfying models (adds new constraints)
- The KB learns something new

Formally: M(KB) ∩ M(α) ⊂ M(KB) (strict subset, but non-empty)

**Contradiction**:

- M(α) doesn't overlap M(KB)
- No models satisfy both KB and α
- The KB rejects this information

Formally: M(KB) ∩ M(α) = ∅

### Ask Operation Formalization

- **True**: KB ⊨ α (knowledge base entails α)
- **False**: KB ⊨ ¬α (knowledge base entails negation of α)
- **Unknown**: KB ⊭ α AND KB ⊭ ¬α (neither is entailed)

---

## Part 9: Introduction to Model Checking

### The Problem

Given a knowledge base KB at some state (e.g., timestamp t=4 in Wumpus World), determine the response to a query like "¬P₁₂" (there is no pit at location 1,2).

### Model Checking Algorithm

**Step 1**: List all propositional symbols involved in the rules

- Example symbols: B₁₁, B₂₁, P₁₂, P₂₁, P₂₂, P₃₁ (7 symbols)

**Step 2**: Enumerate all possible models

- With 7 binary symbols: 2⁷ = 128 possible models
- Each row represents a different truth assignment

**Step 3**: For each model, evaluate whether it satisfies the KB

- The KB is satisfied when ALL rules (R1 ∧ R2 ∧ R3...) evaluate to true
- Rules are conjunctions – each adds a constraint

**Step 4**: Among models that satisfy the KB, check the query

- If ¬P₁₂ is true in all satisfying models → Response is **True**
- If ¬P₁₂ is false in all satisfying models → Response is **False**
- If mixed → Response is **Unknown**

### Example Rule Evaluation

For rule R2: B₁₁ ↔ (P₁₂ ∨ P₂₁)

Consider a row where: P₁₂ = false, P₂₁ = false, B₁₁ = false

- P₁₂ ∨ P₂₁ = false
- B₁₁ ↔ false = false ↔ false = **true**
- R2 evaluates to true for this model

The process repeats for all 128 models and all rules. Only models where ALL rules are true are considered.

**Reference**: Table 7.9 in AIMA (AI: A Modern Approach) shows this enumeration for the Wumpus World.
![[Pasted image 20251217152445.png]]

---

## Part 10: Important Logistics and Coming Topics

### Table of Logical Equivalences (Table 7-11 in AIMA)

The professor emphasized that students should **print Table 7-11** from the textbook, which contains logical equivalences needed for:

- The syntactic approach to logical reasoning (theorem proving)
- The final exam

### Next Lecture Preview

The professor indicated the next lecture will cover:

- **Syntactic logical reasoning** (theorem proving without model enumeration)
- This is more difficult but doesn't require enumerating all possible models

### Key Takeaway

The more rules/constraints in the knowledge base:

- The **fewer** models satisfy it
- The **more precise** the inferences become
- But the **harder** it is to satisfy (more rows to check in model checking)

---

## Summary of Key Concepts

| Topic               | Key Points                                                                                  |
| ------------------- | ------------------------------------------------------------------------------------------- |
| Transformers Review | WQ, WK, WV project to meaningful coordinate systems; multi-head captures different patterns |
| MoE                 | Ensemble benefits come from uncorrelated mistakes; C=0 gives linear error reduction         |
| Vision Transformers | Image patches = tokens; attention heads can focus on color/texture/shape                    |
| Propositional Logic | Symbols + operators; models are truth assignments                                           |
| Wumpus World        | Demonstrates rule-based reasoning; conservative navigation                                  |
| KB Operations       | Tell (entailment/contingency/contradiction), Ask (true/false/unknown)                       |
| Model Checking      | Enumerate all models, check which satisfy KB, evaluate query                                |
![[Pasted image 20251217152404.png]]

# Lecture 11 Professor Notes
## Part I: Logical Reasoning — Proof-Based (Syntactic) Method

### Overview and Context

The professor covered two main methods for logical reasoning:

1. **Model Checking** (covered in previous lecture) — Enumerates all possible models and checks truth values semantically
2. **Theorem Proving / Proof-Based Method** (covered this lecture) — Works entirely on syntax using inference rules

**Key distinction**: Model checking works by examining truth tables and possible worlds; proof-based methods work by applying formal rules of inference without considering the actual truth values of propositions.

**Real-world relevance**: AWS uses theorem proving (specifically first-order logic) for their cybersecurity systems. When a customer asks "Is this Docker container connected to the internet?", the system uses logical reasoning to respond. This is called **Automated Theorem Proving**.

---

### Inference Rules

### 1. Modus Ponens

**Notation format**: The numerator is the premise, the denominator is the conclusion (not a fraction, just notation convention).

**Rule**:

```
α → β, α
──────────
    β
```

**Interpretation**: If we know that α implies β, and we know that α is true, then we can conclude that β is true.

**Example**:

- If "it is raining implies the ground is wet" (α → β)
- And "it is raining" (α)
- Then we conclude "the ground is wet" (β)

### 2. AND Elimination

**Rule**:

```
α ∧ β
──────
  α
```

(Also applies symmetrically: α ∧ β ⊢ β)

**Interpretation**: From a conjunction, we can extract either conjunct as a standalone true statement.

---

### Table 7-11: Logical Equivalences

The professor referenced "Table 7-11" from the AIMA textbook — a crucial reference table of logical equivalences. Important rules from this table include:

### Biconditional Elimination

```
α ↔ β  ≡  (α → β) ∧ (β → α)
```

A biconditional can be rewritten as two implications conjoined.

### Contraposition

```
α → β  ≡  ¬β → ¬α
```

An implication is equivalent to its contrapositive.

### De Morgan's Laws

```
¬(α ∨ β)  ≡  ¬α ∧ ¬β
¬(α ∧ β)  ≡  ¬α ∨ ¬β
```

These laws allow transformation between negated disjunctions and conjunctions.

---

### Worked Example: Wumpus World Proof

**Query**: Prove that ¬P₁,₂ ∧ ¬P₂,₁ is TRUE (there are no pits at positions (1,2) and (2,1))

**Knowledge Base Rules** (from Wumpus World):

- **R2**: B₁,₁ ↔ P₁,₂ ∨ P₂,₁ (Breeze at (1,1) iff pit at (1,2) or (2,1))
- **R4**: ¬B₁,₁ (No breeze at position (1,1))

### Proof Steps:

**Step 1**: Start with R2 from knowledge base

```
R2: B₁,₁ ↔ P₁,₂ ∨ P₂,₁
```

**Step 2**: Apply Biconditional Elimination to get R8

```
R8: (B₁,₁ → P₁,₂ ∨ P₂,₁) ∧ (P₁,₂ ∨ P₂,₁ → B₁,₁)
```

**Step 3**: Apply AND Elimination to R8 to extract R9

```
R9: P₁,₂ ∨ P₂,₁ → B₁,₁
```

**Step 4**: Apply Contraposition to R9 to get R10

```
R10: ¬B₁,₁ → ¬(P₁,₂ ∨ P₂,₁)
```

**Step 5**: Apply Modus Ponens using R10 and R4

- Premise: ¬B₁,₁ → ¬(P₁,₂ ∨ P₂,₁) [R10]
- Premise: ¬B₁,₁ [R4 from knowledge base]
- Conclusion: ¬(P₁,₂ ∨ P₂,₁) [R11]

**Step 6**: Apply De Morgan's Law to R11

```
¬(P₁,₂ ∨ P₂,₁)  ≡  ¬P₁,₂ ∧ ¬P₂,₁
```

**Conclusion**: We have proven that ¬P₁,₂ ∧ ¬P₂,₁ is TRUE — there are no pits at either location.

---

### Transition to First-Order Logic

The professor noted:

- Propositional logic (what we covered) uses symbols without variables
- **First-order logic** extends this with predicates, quantifiers, and variables
- AWS's cybersecurity reasoning uses first-order logic
- **Neurosymbolic reasoning** combines symbolic representations with neural networks to reduce hallucinations in LLMs

---

## Part II: Classical Planning

## What is Classical Planning?

**Definition**: Planning without interactions with the environment. The environment is **deterministic** (non-stochastic).

**Goal**: Given an initial state and a goal state, find a sequence of actions that transforms the initial state into the goal state.

**Key characteristics**:

- Deterministic environment
- Complete observability
- No uncertainty in action outcomes

---

## Real-World Applications

1. **Manufacturing/Robotics**: Assembly lines, robotic arms moving objects predictably
2. **Verification systems**: Verifying AI model responses
3. **OpenRouter example**: Load balancing requests to LLMs (modeling the domain of request routing)
4. **Logistics**: Optimal shipping and delivery routes

**Connection to Modern AI**:

- PDDL solvers can be used as **tools** to ground rules and reduce hallucinations in LLMs
- Chain-of-thought reasoning in LLMs can be modulated by rules expressed in planning languages
- Researchers at MIT and NVIDIA (Chris Paxton) are converting natural language rules into PDDL expressions

---

## PDDL: Planning Domain Definition Language

PDDL is a **domain-specific language (DSL)** for specifying planning problems.

### Two Required Files:

1. **Domain File** (domain.pddl) — Specifies:
    
    - Types (classes)
    - Predicates (state descriptions)
    - Action schemas (operators with preconditions and effects)
2. **Problem File** (problem.pddl) — Specifies:
    
    - Objects (instances of types)
    - Initial state
    - Goal state

### Core Concepts:

**State**: A conjunction of ground atomic fluents — a specific arrangement of objects with predicates having constant parameters.

**Example state description** for blocks on a table:

```
on(A, table) ∧ on(B, table) ∧ on(C, A)
```

---

## Blocks World Example

**Domain**: A surface with blocks and a robotic arm that can pick up and move blocks.

### Types:

- Block

### Predicates:

- `on(block, location)` — Block is on a location
- `clear(block)` — Nothing is on top of the block
- `holding(block)` — Robotic arm is holding the block
- `hand-empty` — Robotic arm is not holding anything
- `on-table(block)` — Block is directly on the table

### Action Schema Example: Move(B, X, Y)

**Parameters**:

- B: The block being moved
- X: Current location (where B is now)
- Y: Destination (where B will go)

**Preconditions** (must all be true to execute action):

```
on(B, X) ∧ clear(B) ∧ clear(Y) ∧ block(Y) ∧ (B ≠ X) ∧ (X ≠ Y) ∧ (B ≠ Y)
```

**Effects** (what becomes true/false after action):

```
on(B, Y) ∧ clear(X) ∧ ¬on(B, X) ∧ ¬clear(Y)
```

**Interpretation**: To move block B from X to Y:

- B must be on X
- B must be clear (nothing on top)
- Y must be clear (space available)
- After moving: B is on Y, X becomes clear, B is no longer on X, Y is no longer clear

---

## PDDL Solvers and Tools

- **VS Code Plugin**: Accepts domain.pddl and problem.pddl files, calls a solver, returns sequence of actions
- **Unified Planning Library (Python)**: Express planning problems programmatically
- **PlanSys (ROS2)**: Planning system for robotics applications
- Most solvers use **forward search algorithms**

**Extensions to PDDL**:

- Temporal PDDL: Actions have duration/latency
- Used in systems where timing matters (e.g., manufacturing with deadlines)

---

## Part III: Markov Decision Processes (MDPs) and Introduction to Reinforcement Learning

## Why Reinforcement Learning?

The professor emphasized: "Reinforcement learning is a very long path... there are a lot of bodies around the trajectory" (Indiana Jones analogy). The key to understanding RL is mastering **Markov Decision Processes (MDPs)**.

**Importance**: Most modern LLMs are optimized with reinforcement learning after initial next-token prediction training. This is how ChatGPT and similar models are fine-tuned for desired behaviors.

---

## The Agent-Environment Interaction Loop

```
        ┌─────────────┐
        │             │
   ─────►   AGENT     ├─────► Action (Aₜ)
        │             │         │
        └─────────────┘         │
              ▲                 │
              │                 ▼
        Reward (Rₜ₊₁)    ┌─────────────┐
        State (Sₜ₊₁)     │ ENVIRONMENT │
              ◄──────────┤             │
                         └─────────────┘
```

### Sequence of Events:

1. Agent is in state Sₜ
2. Agent takes action Aₜ
3. Environment transitions to new state Sₜ₊₁
4. Environment provides reward Rₜ₊₁
5. Process repeats

**Experience tuple**: (S, A, R, S', A', R', ...)

---

## Key Terminology

|Term|Symbol|Definition|
|---|---|---|
|State|S (random variable), s (value)|Current situation of the agent|
|Action|A (random variable), a (value)|Decision made by the agent|
|Reward|R|Scalar feedback signal from environment|
|Episode|—|Complete interaction from start to termination|
|Trajectory|τ|Sequence of experiences over an episode|
|Terminal state|—|State where episode ends|
|Time horizon|T|When interaction terminates|

**Important notation convention**:

- Capital letters (S, A, R) = random variables
- Lowercase letters (s, a, r) = specific values taken by those variables

---

## Formal MDP Definition

An MDP is defined as a 5-tuple: **M = (S, P, R, A, γ)**

|Component|Description|
|---|---|
|**S** (script)|Set of all possible states|
|**P**|Transition model (probability of moving between states)|
|**R**|Reward function|
|**A**|Set of all possible actions|
|**γ** (gamma)|Discount factor (0 ≤ γ ≤ 1)|

---

## MDP Dynamics

The **MDP dynamics** is a probability distribution:

$$P(S', R | S, A)$$

This represents: "Given I'm in state S and take action A, what's the probability of ending up in state S' and receiving reward R?"

### Deriving Component Models:

**Transition Model** (from marginalization): $$P(S' | S, A) = \sum_R P(S', R | S, A)$$

**Reward Model**: $$P(R | S, A) = \sum_{S'} P(S', R | S, A)$$

---

## The Grid World Example

The professor used a simple grid world to illustrate concepts:

**Environment characteristics**:

- Grid of cells (states)
- Agent can move: Up, Down, Left, Right
- **Stochastic transitions**:
    - 80% probability: Move in intended direction
    - 10% probability: Move left of intended direction
    - 10% probability: Move right of intended direction
- Hitting a wall means staying in current state
- Two **terminal states**: +1 (goal) and -1 (pit)

**Why negative rewards for non-terminal states?**

The professor explained: If all states had even slightly positive rewards, the agent would wander forever collecting rewards instead of reaching the goal. Small negative rewards (e.g., -0.04) motivate the agent to reach the terminal state efficiently.

---

## Transition Model as a Table/Tensor

For each action, we need a table showing:

```
P(S' | S, A=action)
```

**Structure**:

- Rows: Current states (S)
- Columns: Next states (S')
- Values: Transition probabilities

Since there are 4 actions, we have 4 such tables — forming a 3D tensor.

**Example** for A=Up in state S₁₁:

- P(S₁₂ | S₁₁, Up) = 0.8 (moved up successfully)
- P(S₁₁ | S₁₁, Up) = 0.1 (hit wall, stayed)
- P(other | S₁₁, Up) = 0.1 (moved sideways)

---

## Reward Function

### Two-Parameter Version:

$$R(s, a) = \mathbb{E}[R_t | S_{t-1}=s, A_{t-1}=a]$$

"Expected reward received after executing action a from state s"

**Expansion**: $$R(s, a) = \sum_r \sum_{s'} r \cdot P(s', r | s, a)$$

### Three-Parameter Version:

$$R(s, a, s') = \mathbb{E}[R_t | S_{t-1}=s, A_{t-1}=a, S_t=s']$$

This version includes dependency on which state we end up in.

---

## Return (Cumulative Discounted Reward)

The **return** Gₜ is what the agent tries to maximize:

$$G_t = R_{t+1} + \gamma R_{t+2} + \gamma^2 R_{t+3} + ... = \sum_{k=0}^{\infty} \gamma^k R_{t+1+k}$$

### The Discount Factor γ:

|γ value|Agent behavior|
|---|---|
|γ ≈ 0|Myopic — focuses on immediate rewards only|
|γ ≈ 1|Far-sighted — considers long-term consequences|

**Intuition**: Like net present value of money — "$1 million today is worth more than $1 million next year." Future rewards are uncertain, so we discount them.

### Why Discounting?

1. Mathematical convenience (ensures finite sums)
2. Reflects uncertainty about the future
3. Models preference for immediate rewards
4. Avoids infinite returns in continuing tasks

---

## Policy

A **policy** π is a mapping from states to actions, expressed as a probability distribution:

$$\pi(a|s) = P(A_t = a | S_t = s)$$

### Types of Policies:

**Stochastic policy**: Probability distribution over actions

- Example: π(up|s) = 0.25, π(down|s) = 0.25, π(left|s) = 0.25, π(right|s) = 0.25

**Deterministic policy**: Always takes the same action in a given state

- P(aᵢ|s) = 1 for some specific action aᵢ

**Important note**: A stochastic policy does NOT make the MDP stochastic. The stochasticity of the MDP comes from the transition model itself.

---

## Value Functions

### State-Value Function Vπ(s)

"The expected return starting from state s and following policy π thereafter"

$$V^\pi(s) = \mathbb{E}_\pi[G_t | S_t = s]$$

**Expanded**: $$V^\pi(s) = \mathbb{E}_\pi[R_{t+1} + \gamma R_{t+2} + \gamma^2 R_{t+3} + ... | S_t = s]$$

**Interpretation**: For every state in the environment, we assign a number (value) representing how good it is to be in that state under policy π.

**Example**: In a room where the goal is to exit through a door:

- States near the door have high value
- States far from the door have lower value

---

## The Bellman Expectation Equation

The fundamental recursive relationship for value functions:

$$V^\pi(s) = \mathbb{E}_\pi[R_{t+1} + \gamma V^\pi(S_{t+1}) | S_t = s]$$

**In words**: The value of a state equals the expected immediate reward plus the discounted value of the next state.

### Backup Tree Visualization:

```
           s (current state)
           │
    ┌──────┼──────┐
    │      │      │
   a₁     a₂     a₃    ← Actions (chosen by policy π)
    │      │      │
   ─┼─    ─┼─    ─┼─   ← State transitions (governed by P(s'|s,a))
  ╱ │ ╲  ╱ │ ╲  ╱ │ ╲
 s' s' s' s' s' s' s'  ← Possible next states
```

**Key insight**: The Bellman equation connects the value of the current state to the values of possible successor states.

### Derivation Note:

The professor mentioned he has a 2-page derivation of the Bellman expectation equation that he would post to Discord. The derivation is in Richard Sutton's book but not completely explicit there.

---

## Prediction vs. Control

### Prediction Problem:

Given a policy π, evaluate V^π(s) for all states.

"What is the value of each state if I follow this policy?"

### Control Problem:

Find the optimal policy π* that maximizes value for all states.

"What is the best policy to follow?"

**Process**:

1. Start with some policy
2. Evaluate its value function (prediction)
3. Improve the policy based on the value function (control)
4. Repeat until convergence to optimal policy π*

---

## Coming Up (Preview)

The professor mentioned these topics will be covered in subsequent lectures:

1. **Bellman Optimality Equations**: For finding optimal policies
2. **Value Iteration**: Algorithm for computing optimal value function
3. **Policy Iteration**: Algorithm that alternates between evaluation and improvement
4. **Connection to LLMs**: How reinforcement learning fine-tunes language models

---

## Summary of Key Concepts

## Logical Reasoning

- Two methods: Model Checking (semantic) vs. Theorem Proving (syntactic)
- Inference rules: Modus Ponens, AND Elimination
- Table 7-11 equivalences: Biconditional, Contraposition, De Morgan's

## Classical Planning (PDDL)

- Deterministic environment
- Domain file: Types, Predicates, Action Schemas
- Problem file: Objects, Initial State, Goal State
- Solvers use forward search

## MDPs

- 5-tuple: (S, P, R, A, γ)
- Agent-Environment interaction loop
- Stochastic transitions
- Reward function incentivizes desired behavior

## Reinforcement Learning Foundations

- Policy π(a|s): Mapping from states to actions
- Return Gₜ: Discounted cumulative reward
- Value Function Vπ(s): Expected return from state s
- Bellman Equation: Recursive relationship connecting state values
- Prediction (evaluate policy) vs. Control (find optimal policy)

---

## Recommended Resources

1. **Richard Sutton's book**: "Reinforcement Learning: An Introduction" (free online)
2. **David Silver's lectures**: 12-14 lectures on RL (linked on course site)
3. **Deep Reinforcement Learning book**: By Google engineers (O'Reilly Library)
4. **AIMA textbook**: Chapters 3, 4 (problem solving), Chapter 11, Chapters 16 & 17

# Lecture 12 Professor Notes
## 1. Markov Decision Processes (MDPs) - Foundations

### 1.1 What is an MDP?

A Markov Decision Process is a mathematical framework for modeling sequential decision-making problems where outcomes are partly random and partly under the control of a decision-maker (agent).

**The MDP Tuple**: An MDP is defined by five components:

$$\mathcal{M} = (\mathcal{S}, \mathcal{P}, \mathcal{R}, \mathcal{A}, \gamma)$$

Where:

- **𝒮 (State Space)**: The set of all possible states the environment can be in
- **𝒫 (Transition Model)**: Probability of transitioning between states given actions
- **ℛ (Reward Function)**: The set of rewards or reward function
- **𝒜 (Action Space)**: The set of all possible actions the agent can take
- **γ (Discount Factor)**: A value between 0 and 1 that discounts future rewards

### 1.2 The Agent-Environment Interaction Loop

The professor emphasized this core interaction cycle:

1. Agent is in state **S**
2. Agent decides to act with action **A** (based on policy π)
3. Environment transitions to new state **S'**
4. Agent receives reward **R**
5. Process repeats

This interaction generates a sequence of **experiences** that are encoded into probability distributions.

### 1.3 The Transition Model

The transition model P(s'|s, a) describes the probability of ending up in state s' given that you were in state s and took action a.

**Key Properties**:

- The environment is **stochastic** - same action from same state may lead to different outcomes
- The Markov property: The future depends only on the current state, not the history

**Example (Grid World)**: If you're in a grid cell and choose to go "up":

- With probability 0.8, you actually go up
- With probability 0.1, you slip left
- With probability 0.1, you slip right

This is captured in transition probability tables - one for each action.

### 1.4 The Reward Function

The reward function can take two forms:

**Two-parameter version**: R(s, a)

- Expected reward received after executing action a from state s

**Three-parameter version**: R(s, a, s')

- Expected reward received after executing action a from state s and landing in state s'

**Mathematical Definition**: $$R(s, a) = \mathbb{E}[R_t | S_t = s, A_t = a]$$

### 1.5 The Policy

A **policy π** is the agent's strategy - it defines which action to take in each state.

**Stochastic Policy**: π(a|s) = P(A_t = a | S_t = s)

- Gives a probability distribution over actions for each state

**Deterministic Policy**: Directly maps states to actions

- π(s) = a

**Important Clarification from Lecture**: A stochastic policy does NOT make the MDP stochastic. The stochasticity of the MDP comes from the transition model itself.

### 1.6 Episodes and Trajectories

**Episode**: A complete interaction sequence from start to termination

- From t = 0 to t = T-1

**Trajectory (τ)**: The sequence of experiences over an episode $$\tau = (S_0, A_0, R_1, S_1, A_1, R_2, ..., S_T)$$

**Termination** occurs when:

- Agent reaches a terminal state
- Agent decides to terminate
- Running out of time (finite horizon problems)

### 1.7 The Return

The **return G_t** is the cumulative discounted reward from time t onwards:

$$G_t = R_{t+1} + \gamma R_{t+2} + \gamma^2 R_{t+3} + ... = \sum_{k=0}^{\infty} \gamma^k R_{t+k+1}$$

**Why discount?**

- Mathematical convenience (ensures finite returns for infinite horizons)
- Represents uncertainty about the future
- Models preference for immediate rewards
- Common values: γ = 0.9 or γ = 0.99

---

## 2. Value Functions

Value functions are the **objective functions** we optimize in MDPs. They estimate "how good" it is to be in a given state or to take a given action.

### 2.1 State Value Function V(s)

The **state value function** V^π(s) answers: "What is the expected return starting from state s and following policy π?"

$$V^\pi(s) = \mathbb{E}_\pi[G_t | S_t = s]$$

**Expanded form**: $$V^\pi(s) = \mathbb{E}_\pi[R_{t+1} + \gamma R_{t+2} + \gamma^2 R_{t+3} + ... | S_t = s]$$

**Intuition**: Think of a grid world where you need to exit a room. States closer to the door have higher value because you expect to receive rewards sooner.

**Visual Representation**: In a grid world, the value function is a matrix where every state/cell has a value number.

### 2.2 State-Action Value Function Q(s, a)

The **state-action value function** (Q-function) Q^π(s, a) answers: "What is the expected return starting from state s, taking action a, and then following policy π?"

$$Q^\pi(s, a) = \mathbb{E}_\pi[G_t | S_t = s, A_t = a]$$

**Why Q is "better" than V for acting optimally**: The Q function directly tells us the value of each action, making it easier to select optimal actions. With V alone, you need to know the transition model to compare action values.

---

## 3. Bellman Equations

The Bellman equations are fundamental recursive relationships that connect the value of a state to the values of successor states. They are named after Richard Bellman, a pioneer in dynamic programming and optimal control.

### 3.1 Bellman Expectation Equation for V

This equation connects the value of the current state to immediate reward plus discounted value of next states:

$$V^\pi(s) = \mathbb{E}_\pi[R_{t+1} + \gamma V^\pi(S_{t+1}) | S_t = s]$$

**Expanded form with explicit sums**: $$V^\pi(s) = \sum_a \pi(a|s) \sum_{s'} P(s'|s,a)[R(s,a,s') + \gamma V^\pi(s')]$$

**Interpretation**: The value of a state equals the expected immediate reward plus the discounted expected value of the next state.

### 3.2 The Backup Tree Concept

The professor emphasized understanding backup trees:

```
        (s)           ← Starting state
       / | \
      a1 a2 a3        ← Possible actions (determined by π)
     /|\ |  |\ 
   s' s'' s'''        ← Possible next states (determined by P)
```

- **Solid lines** (from state to action): Determined by policy π(a|s)
- **Dashed lines** (from action to next state): Determined by transition model P(s'|s,a)
- Each next state has its own value V(s')
- We "back up" values from successor states to compute current state value

### 3.3 Bellman Expectation Equation for Q

$$Q^\pi(s,a) = \mathbb{E}_\pi[R_{t+1} + \gamma Q^\pi(S_{t+1}, A_{t+1}) | S_t = s, A_t = a]$$

**Expanded**: $$Q^\pi(s,a) = \sum_{s'} P(s'|s,a)[R(s,a,s') + \gamma \sum_{a'} \pi(a'|s') Q^\pi(s',a')]$$

### 3.4 Bellman Optimality Equations (Nonlinear!)

When we want to find the **optimal** value function (V* or Q*), we introduce the **max** operator, which makes the equations **nonlinear**.

**For V***:
$$
V^*(s) = \max_{a} \sum_{s'} P(s' \mid s,a) \bigl[ R(s,a,s') + \gamma V^*(s') \bigr]
$$
**For Q***: 
$$
Q^*(s,a) = \sum_{s'} P(s' \mid s,a) \bigl[ R(s,a,s') + \gamma \max_{a'} Q^*(s',a') \bigr]
$$
**Why nonlinear?** The max operator is nonlinear - we can't solve these with simple matrix inversion like the expectation equations. We need iterative methods like **policy iteration** or **value iteration**.
### 3.5 Solving Bellman Equations

**Linear Case (Bellman Expectation)**:

- Can be written as: **V = R + γPV**
- Rearranges to: **(I - γP)V = R**
- Solution: **V = (I - γP)^(-1) R**

**However**, the professor strongly warned against matrix inversion in practice:

> "Every time you see matrix inversion, don't implement it - it will more likely blow up."

Instead, use **iterative methods** based on the fact that the Bellman operator is a **contraction**.

### 3.6 The Bellman Operator as a Contraction

A **contraction** mapping has the property that applying it repeatedly converges to a fixed point, regardless of starting point.

**Simple scalar example**: $$x_{k+1} = \gamma x_k + c$$

For γ < 1, this converges to x* = c/(1-γ)

**The Bellman operator is a contraction** with factor γ: $$V_{k+1}(s) = \sum_a \pi(a|s) \sum_{s'} P(s'|s,a)[R(s,a,s') + \gamma V_k(s')]$$

This iterative method is called **Iterative Policy Evaluation**.

---

## 4. Prediction vs Control Problems

### 4.1 Prediction Problem (Policy Evaluation)

**Goal**: Given a policy π, calculate V^π(s) for all states

**Question**: "How good is this policy?"

**Method**: Use Bellman Expectation equations iteratively

**Example**: Given a uniform random policy (equal probability for all actions), what is the value of each state?

### 4.2 Control Problem

**Goal**: Find the optimal policy π* that maximizes value

**Question**: "What is the best way to act?"

**Involves two sub-problems**:

1. Policy Evaluation: Evaluate current policy
2. Policy Improvement: Use greedy action selection to improve policy

**Methods**: Policy Iteration, Value Iteration

---

## 5. Policy Iteration

Policy Iteration is the key algorithm for solving MDPs (finding π*). It alternates between two steps until convergence.

### 5.1 The Algorithm

```
1. Initialize π arbitrarily (e.g., uniform random)
2. Repeat until convergence:
   a. Policy Evaluation: Compute V^π for current policy
   b. Policy Improvement: For each state, update policy to be greedy w.r.t. V^π
      π'(s) = argmax_a Σ_{s'} P(s'|s,a)[R(s,a,s') + γV^π(s')]
3. Return π* (optimal policy)
```

### 5.2 Policy Evaluation (Step 2a)

Use iterative policy evaluation:

$$V_{k+1}(s) = \sum_a \pi(a|s) \sum_{s'} P(s'|s,a)[R(s,a,s') + \gamma V_k(s')]$$

Repeat until V converges (change is below threshold).

### 5.3 Policy Improvement (Step 2b)

**Greedy Policy Improvement**: For each state, select the action that maximizes expected value:

$$\pi'(s) = \arg\max_a \sum_{s'} P(s'|s,a)[R(s,a,s') + \gamma V^\pi(s')]$$

This **eliminates suboptimal actions** from the policy.

### 5.4 Convergence Guarantee

The professor emphasized a key insight:

> "The optimal policy may converge much sooner than the value function. Despite the fact that the value function may change from one iteration to another, the relative benefits or relative values between adjacent states that drive the decision-making do not change, and therefore the optimal policy may have converged much sooner."

### 5.5 Grid World Example

**Setup**: 4x4 grid, two terminal states (corners), reward of -1 for each step

**Iteration 0**:

- Initialize V = 0 for all states
- Policy = uniform random (0.25 probability for each direction)

**Iteration 1**:

- Evaluate policy → Get new V values
- Act greedily → Arrows now point toward terminal states

**Iteration 2+**:

- Continue until policy no longer changes
- Final result: Optimal arrows in each cell pointing toward shortest path to terminal

---

## 6. Reinforcement Learning Fundamentals

### 6.1 The Key Difference: Model-Free Learning

In **MDP** (what we covered earlier):

- We **know** the transition model P(s'|s,a)
- We **know** the reward function R(s,a,s')
- We can compute value functions exactly

In **Reinforcement Learning**:

- We **do NOT know** the transition model
- We **do NOT know** the reward function
- We must **learn** from experience (interactions with environment)

### 6.2 The Backup Tree Comparison

The professor drew a clear distinction using backup trees:

**Dynamic Programming (Full Backup)**:

```
        (s)
      /  |  \
    a1  a2  a3      ← Know all actions
   /|\  |   /|\
 s' s'' s'''        ← Know all possible next states and their probabilities
```

- **Breadth-first expansion**
- Know exactly all states we can transition to
- Know exactly all rewards
- Full backup from all successor states

**Monte Carlo / Reinforcement Learning (Sample Backup)**:

```
   (s)
    |
    a1                ← Take ONE action
    |
   s'                 ← Observe ONE next state
    |
    a2
    |
   s''
    ...
    |
 Terminal            ← Complete trajectory
```

- **Depth-first sampling**
- Environment is a **black box**
- We just take actions and observe what happens
- Generate complete trajectories (episodes)

### 6.3 The Learning Paradigm

In RL, we:

1. Interact with the environment
2. Collect experiences (s, a, r, s')
3. Learn value functions from these experiences
4. Use estimated values to improve policy

---

## 7. Monte Carlo Methods

### 7.1 Core Idea

Monte Carlo (MC) methods learn from **complete episodes**. We:

1. Generate many trajectories by interacting with the environment
2. Observe the actual returns (rewards) received
3. Estimate value as the **sample mean** of observed returns

### 7.2 The Monte Carlo Value Update

**Basic idea**: Visit a state multiple times, track returns, compute average

For each state s visited in an episode:

1. Increment visit counter: N(s) = N(s) + 1
2. Update total return: G_total(s) = G_total(s) + G_t
3. Estimate value: V(s) = G_total(s) / N(s)

**Incremental Sample Mean Form** (connects to Kalman filter lecture!):

$$V(s) \leftarrow V(s) + \alpha [G_t - V(s)]$$

Where:

- α = 1/N(s) (or a constant learning rate)
- G_t = actual return observed from state s in this episode
- [G_t - V(s)] = "error" or "surprise" - difference between observed and expected

### 7.3 The Monte Carlo Equation

$$V^\pi(S_t) = V^\pi(S_t) + \alpha [G_t - V^\pi(S_t)]$$

**Interpretation**:

- New estimate = Old estimate + learning_rate × (target - old estimate)
- Target is the **actual return G_t** observed
- This is exactly the incremental sample mean we studied in Kalman filters!

### 7.4 Advantages of Monte Carlo

- Simple and intuitive
- No bias (uses actual returns, not estimates)
- Works well with episodic tasks
- Does not require knowledge of environment dynamics

### 7.5 Major Limitation of Monte Carlo

**You must wait until the episode terminates to learn!**

The professor used a driving analogy:

> "It's okay to wait until the car is crashing, and then go back and say 'okay, I now have an estimate of the value of not going 100 miles per hour in a corner.' In reality, I prefer schemes that allow me to act as I go."

This limitation leads us to **Temporal Difference methods**.

---

## 8. Temporal Difference Learning

### 8.1 TD(0) - One-Step Temporal Difference

**Core Innovation**: Don't wait for episode end - bootstrap from estimated values!

**TD(0) Update Equation**: $$V(S_t) \leftarrow V(S_t) + \alpha [R_{t+1} + \gamma V(S_{t+1}) - V(S_t)]$$

**The TD Target**: $$\text{TD Target} = R_{t+1} + \gamma V(S_{t+1})$$

**The TD Error (δ)**: $$\delta_t = R_{t+1} + \gamma V(S_{t+1}) - V(S_t)$$

### 8.2 Comparing MC and TD(0)

|Aspect|Monte Carlo|TD(0)|
|---|---|---|
|**Target**|Actual return G_t|R_{t+1} + γV(S_{t+1})|
|**Must wait for**|Episode end|Next time step|
|**Bias**|Unbiased|Biased (uses estimates)|
|**Variance**|High|Lower|
|**Bootstrapping**|No|Yes|

### 8.3 The Secret of TD(0)

The professor emphasized this key insight:

> "TD(0) combines the **bootstrapping** of dynamic programming with the **sampling** of Monte Carlo."

- **From DP**: We use estimated values of successor states (bootstrapping)
- **From MC**: We sample trajectories through the environment

### 8.4 The Brooklyn-to-Jersey Analogy

The professor gave this memorable analogy:

> "If I am in Brooklyn and want to go back to Jersey, I have two options - Brooklyn Bridge or Manhattan Bridge."
> 
> "**TD(0) says**: The moment you cross the Manhattan Bridge, go ahead and update the value of Brooklyn."
> 
> "**TD(λ) says**: Wait until you exit the Holland Tunnel and reach New Jersey before you update Brooklyn, because you never know what is waiting for you in Canal Street."

### 8.5 N-Step TD Returns

Instead of bootstrapping after 1 step, we can wait n steps:

**n-step Return**: $$G_t^{(n)} = R_{t+1} + \gamma R_{t+2} + ... + \gamma^{n-1} R_{t+n} + \gamma^n V(S_{t+n})$$

**n-step TD Update**: $$V(S_t) \leftarrow V(S_t) + \alpha [G_t^{(n)} - V(S_t)]$$

**Special Cases**:

- n = 1: TD(0)
- n = ∞: Monte Carlo (full return)

### 8.6 TD(λ) - Combining All N-Step Returns

TD(λ) takes a weighted average of ALL n-step returns:

**The λ-return**: $$G_t^\lambda = (1-\lambda) \sum_{n=1}^{\infty} \lambda^{n-1} G_t^{(n)}$$

**Properties**:

- **λ = 0**: Reduces to TD(0) (one-step bootstrap)
- **λ = 1**: Reduces to Monte Carlo (full return)
- **0 < λ < 1**: Weighted combination

**Exponential Weighting**: The weights decay exponentially:

- G_t^(1) gets weight (1-λ)
- G_t^(2) gets weight (1-λ)λ
- G_t^(3) gets weight (1-λ)λ²
- And so on...

**TD(λ) Update**: $$V_{t+n}(S_t) = V_{t+n-1}(S_t) + \alpha [G_t^\lambda - V_{t+n-1}(S_t)]$$

**Key Benefit**: TD(λ) allows **decoupling** the time that actions are taken from when value function updates are done.

---

## 9. Key Equations Summary

### MDP Definitions

$$G_t = \sum_{k=0}^{\infty} \gamma^k R_{t+k+1}$$

$$V^\pi(s) = \mathbb{E}_\pi[G_t | S_t = s]$$

$$Q^\pi(s,a) = \mathbb{E}_\pi[G_t | S_t = s, A_t = a]$$

### Bellman Expectation Equations

$$V^\pi(s) = \sum_a \pi(a|s) \sum_{s'} P(s'|s,a)[R(s,a,s') + \gamma V^\pi(s')]$$

$$Q^\pi(s,a) = \sum_{s'} P(s'|s,a)[R(s,a,s') + \gamma \sum_{a'} \pi(a'|s') Q^\pi(s',a')]$$

### Bellman Optimality Equations
$$
V^*(s) = \max_{a} \sum_{s'} P(s' \mid s,a) \bigl[ R(s,a,s') + \gamma V^*(s') \bigr]
$$

$$
Q^*(s,a) = \sum_{s'} P(s' \mid s,a) \bigl[ R(s,a,s') + \gamma \max_{a'} Q^*(s',a') \bigr]
$$

### Monte Carlo Update

$$V(S_t) \leftarrow V(S_t) + \alpha [G_t - V(S_t)]$$

### TD(0) Update

$$V(S_t) \leftarrow V(S_t) + \alpha [R_{t+1} + \gamma V(S_{t+1}) - V(S_t)]$$

### N-Step Return

$$G_t^{(n)} = R_{t+1} + \gamma R_{t+2} + ... + \gamma^{n-1} R_{t+n} + \gamma^n V(S_{t+n})$$

### TD(λ) Return

$$G_t^\lambda = (1-\lambda) \sum_{n=1}^{\infty} \lambda^{n-1} G_t^{(n)}$$

---

## 10. Grid World Examples

### 10.1 Simple Two-State Example (from lecture)

**Setup**:

- Two states: S1 and S2
- From S1, taking action leads to S2 with reward +2
- From S2, taking action leads to S1 with reward 0
- Deterministic transitions (probability 1)

**Transition Matrix**:

```
P = [0 1]    (Identity transpose - go from one state to the other)
    [1 0]
```

**Reward Vector**: R = [2, 0]

**Solving with γ (discount factor)**:

Using V = R + γPV, we get:

- V(S1) = 2 + γ × 0 = 2 + γV(S2)
- V(S2) = 0 + γ × V(S1)

Solving: V(S1) = 2/(1-γ²), V(S2) = 2γ/(1-γ²)

For γ = 0.9: V(S1) ≈ 10, V(S2) ≈ 9

### 10.2 Grid World Policy Iteration Example

**Setup**: 4×4 grid

- Two terminal states (top-left and bottom-right corners)
- Reward = -1 for each step (encourages finding shortest path)
- Actions: Up, Down, Left, Right
- Deterministic transitions (but hitting wall = stay in place)
- Initial policy: Uniform random (0.25 probability each direction)

**Iteration Process**:

1. **Initialize**: V = 0 for all states, π = uniform random
    
2. **Policy Evaluation**: Calculate V^π using iterative Bellman equation
    
3. **Policy Improvement**: Act greedily - select actions pointing toward higher-value neighbors
    
4. **Repeat** until policy stabilizes
    

**Key Insight**: After just a few iterations, the policy shows optimal arrows pointing toward terminal states via shortest paths.

### 10.3 Special States Example (from Final Exam)

From the Spring 2025 final exam:

**5×5 Grid with Special States**:

- State A: Any action teleports to A' with reward +10
- State B: Any action teleports to B' with reward +5
- Walls give -1 reward, other moves give 0
- γ = 0.9, uniform random policy

**Question**: Why is V(A) < 10 while V(B) > 5?

**Answer**:

- Agent teleported to A' receives immediate reward of 10
    
- But A' might be in a bad location (can hit walls)
    
- The value includes FUTURE expected rewards, not just immediate
    
- If A' is near walls, future expected rewards are negative
    
- So V(A) = 10 + γ × (expected future) < 10
    
- For B, the +5 immediate reward plus favorable location of B' means
    
- V(B) = 5 + γ × (positive expected future) > 5
    

---

## Key Takeaways for the Final

1. **Understand the MDP framework**: States, actions, rewards, transitions, policy, value functions
    
2. **Know both Bellman equations**: Expectation (linear) and Optimality (nonlinear with max)
    
3. **Understand backup trees**: How values propagate from successor states
    
4. **Know Policy Iteration**: Evaluate policy → Improve greedily → Repeat
    
5. **Understand the RL paradigm**: Model-free, learning from experience
    
6. **Monte Carlo**: Sample complete episodes, average returns, unbiased but must wait
    
7. **TD(0)**: Bootstrap from estimated values, can learn online, biased but lower variance
    
8. **TD(λ)**: Interpolates between TD(0) (λ=0) and MC (λ=1) using exponential weighting
    
9. **Be able to work through small examples**: Two-state problems, small grid worlds
    
10. **Understand the intuition**: Value = expected future rewards, policy iteration converges, bootstrapping vs sampling trade-offs
    

---

## Professor's Study Tips

- The professor mentioned he will provide topic hints 48 hours before the exam via Discord
- The final will likely be "easier than the midterm"
- Focus on understanding the grid world examples and how to apply the equations
- Make sure you understand how to evaluate V* after 2-3 iterations of policy iteration
- The recycling robot example from Sutton's book was mentioned - understand V*(high) and V*(low)