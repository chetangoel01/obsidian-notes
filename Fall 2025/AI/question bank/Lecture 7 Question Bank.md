### **QUESTION SET 1: Tokenization (15 points)**

**QS1A (8 points) - Multiple Choice**

Which of the following statements about Byte Pair Encoding (BPE) tokenization is/are correct?

A. BPE starts with individual characters and iteratively merges the most frequent adjacent pairs to build a vocabulary.

B. The tokenizer used must be tightly coupled with the language model—using a different tokenizer than what the model was trained with will produce correct but suboptimal results.

C. A better tokenizer (with higher compression capability) will result in a larger context size for the same input text.

D. The vocabulary size V and embedding dimension D are typically related—larger vocabularies generally require larger embedding dimensions.

---

**QS1B (7 points) - Short Answer**

Explain why the choice of tokenizer affects a language model's performance on coding tasks. Use the example of GPT-2 vs. GPT-4's tokenizer to illustrate your answer.

---

### **QUESTION SET 2: Word Embeddings & Word2Vec (25 points)**

**QS2A (10 points)**

Consider a vocabulary V = {hotel, motel, car, driver, river, bank} with |V| = 6.

(i) If we use one-hot encoding, what is the dot product between the vectors for "hotel" and "motel"? What does this imply about capturing semantic similarity?

(ii) What property do we want word embeddings to have that one-hot encoding fails to provide?

---

**QS2B (15 points)**

Draw and explain the Skip-gram Word2Vec architecture. Your answer should include:

- The input representation
- The embedding matrix W and its dimensions
- The output projection matrix W' and its dimensions
- The softmax layer
- The loss function used for training

Clearly label the dimensionality at each stage assuming vocabulary size V and embedding dimension D.

---

### **QUESTION SET 3: Context-Free Embeddings (20 points)**

**QS3A (10 points)**

The word "bank" can mean either a financial institution or the edge of a river. In Word2Vec:

(i) How many embedding vectors exist for the word "bank"?

(ii) If your training corpus contains 70% financial news and 30% nature articles, where would you expect the "bank" embedding to be positioned in the D-dimensional space relative to: (a) a corpus of only financial news, and (b) a corpus of only nature articles?

---

**QS3B (10 points)**

Explain why Word2Vec embeddings are called "context-free" despite the fact that the training process uses context windows (nearby words). How do Transformers address this limitation?

---

### **QUESTION SET 4: Recurrent Neural Networks (20 points)**

**QS4A (8 points)**

Write the equation for the hidden state update in a simple RNN. Identify:

- All trainable parameters
- Why the tanh activation function is used instead of sigmoid
- What the "recurrent" aspect of the architecture refers to

---

**QS4B (12 points)**

Compare the following two models:

**Model 1:** s(t) = F(s(t-1), a(t)) where F changes over time

**Model 2:** h(t) = g(h(t-1), x(t); θ) where g is time-invariant

(i) Which model is possible to learn with standard neural network training? Explain why.

(ii) In the RNN formulation, what plays the role of the "action" a(t) from the Kalman filter analogy discussed in class?

---

## **ANSWERS**

---

### **QUESTION SET 1 ANSWERS:**

**QS1A:**

- **A is CORRECT.** BPE starts with individual characters and iteratively merges the most frequent pairs to build subword units.
- **B is INCORRECT.** Using a different tokenizer than what the model was trained with will produce incorrect/poor results, not just suboptimal ones. The tokenizer and model must match.
- **C is INCORRECT.** A better tokenizer with higher compression produces _fewer_ tokens for the same text, resulting in a _smaller_ context size (more information per token, fewer tokens needed).
- **D is CORRECT.** There's an information-theoretic basis for this relationship—larger vocabularies require more dimensions to adequately represent the semantic space.

**Answer: A and D**

---

**QS1B:** GPT-2's tokenizer was not optimized for code, causing Python keywords and common programming constructs to be split into many tokens. For example, common code patterns might produce 186 tokens with GPT-2's tokenizer.

GPT-4's tokenizer was trained with code in mind, allowing single tokens to represent entire Python keywords or common programming patterns. The same code might only produce ~100 tokens.

This matters because:

1. **Context limits:** More tokens means less code can fit in the context window
2. **Computational efficiency:** Fewer tokens means faster processing
3. **Model performance:** When the model sees programming constructs as coherent units (single tokens) rather than fragmented pieces, it can learn patterns more effectively

---

### **QUESTION SET 2 ANSWERS:**

**QS2A:**

(i) The dot product is **0**.

- hotel = [1, 0, 0, 0, 0, 0]
- motel = [0, 1, 0, 0, 0, 0]
- hotel^T · motel = 0

This implies that one-hot encoding treats all words as equally dissimilar, failing to capture that "hotel" and "motel" have very similar meanings.

(ii) We want embeddings where **semantically similar words are close together in vector space**. The dot product (or cosine similarity) between "hotel" and "motel" should be high/positive, indicating their semantic relatedness. Words like "car" and "driver" should also be nearby.

---

**QS2B:**

```
Architecture:

Input: w_t (one-hot, V-dimensional)
         |
         v
    [W matrix: V × D]  ← Embedding matrix
         |
         v
    z (D-dimensional embedding)
         |
         v
    [W' matrix: D × V]  ← Projection/lifting matrix
         |
         v
    z_j' (V-dimensional logits)
         |
         v
    [Softmax]
         |
         v
    ŷ (V-dimensional posterior probability)
```

**Dimensions:**

- Input w_t: V × 1 (one-hot)
- W matrix: V × D (embeds from V to D dimensions)
- z = W^T · w_t: D × 1 (this is just picking the row of W corresponding to the input word)
- W' matrix: D × V (lifts from D back to V dimensions)
- z_j' = W' · z: V × 1 (logits)
- ŷ = softmax(z_j'): V × 1 (probabilities)

**Loss function:** Cross-entropy loss

```
L = -Σ y_j log(ŷ_j)
```

Where y is the one-hot encoding of the actual context word.

For Skip-gram, we predict multiple context words (e.g., 4 words: 2 before, 2 after), so the total loss is the sum of cross-entropy losses for each context position.

**Trainable parameters:** θ = {W, W'_j for each context position}

---

### **QUESTION SET 3 ANSWERS:**

**QS3A:**

(i) **One embedding vector.** Despite having multiple meanings, the word "bank" has exactly one entry in the vocabulary and therefore one row in the W matrix.

(ii) The embedding position would be:

- **Financial corpus only:** The "bank" vector would be positioned near words like "money," "account," "loan," "investment"
- **Nature corpus only:** The "bank" vector would be positioned near words like "river," "stream," "shore," "water"
- **Mixed corpus (70/30):** The "bank" vector would be positioned somewhere between these two locations, likely closer to the financial cluster due to the 70% weighting. It represents an "average" meaning that doesn't perfectly capture either sense.

---

**QS3B:**

Word2Vec embeddings are "context-free" because **the final embedding for each word is fixed after training**—it doesn't change based on the surrounding words at inference time.

During training, context windows ARE used: the model learns to predict context words given a center word (Skip-gram). However, the result of training is a static lookup table (the W matrix) where each word maps to exactly one vector, regardless of usage context.

**How Transformers address this:** Transformers create **contextual embeddings** through the self-attention mechanism. Starting from context-free embeddings (similar to Word2Vec), the attention mechanism allows each token's representation to be modified based on all other tokens in the sequence. The word "bank" in "I went to the bank to deposit money" would have a different representation than "bank" in "We sat on the river bank"—the attention weights would pull each toward the relevant semantic cluster.

---

### **QUESTION SET 4 ANSWERS:**

**QS4A:**

**Hidden state equation:**

```
h(t) = tanh(U^T · x(t) + W · h(t-1) + b)
```

**Trainable parameters:**

- **U:** Weight vector connecting input x(t) to hidden state
- **W:** Weight connecting previous hidden state h(t-1) to current hidden state (the recurrent weight)
- **b:** Bias term

**Why tanh instead of sigmoid:**

- Sigmoid outputs values in [0, 1]
- Tanh outputs values in [-1, 1]
- Tanh is zero-centered, which helps with gradient flow during backpropagation through time
- Having negative values allows the hidden state to decrease as well as increase

**Recurrent aspect:** The "recurrent" nature refers to the feedback connection where the output h(t-1) from the previous timestep is fed back as input to compute h(t). This creates a memory mechanism that allows the network to maintain state information across time steps. The same parameters (U, W, b) are shared across all time steps.

---

**QS4B:**

(i) **Model 2 is possible to learn** with standard neural network training.

Model 1 with F changing over time represents a **non-stationary** or chaotic system where the underlying function itself evolves. Standard neural networks assume the target function is fixed—we're trying to approximate one consistent mapping. If F changes, the "ground truth" keeps shifting, making learning impossible.

Model 2 assumes the function g is **time-invariant** (doesn't change). Even though the inputs change over time, the relationship between inputs and outputs remains consistent, allowing the network to learn a stable set of parameters θ.

(ii) In the RNN formulation, **the arrival of the next token x(t)** plays the role of the "action."

In the Kalman filter, explicit actions (like "move forward") caused state transitions. In RNNs for language modeling, there are no explicit agent actions. Instead, the implicit "action" is the event of receiving the next token in the sequence. Each new token x(t) causes the hidden state to transition from h(t-1) to h(t), analogous to how an action causes state transitions in the Kalman filter framework.