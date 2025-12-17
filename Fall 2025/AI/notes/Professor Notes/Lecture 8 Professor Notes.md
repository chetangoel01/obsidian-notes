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