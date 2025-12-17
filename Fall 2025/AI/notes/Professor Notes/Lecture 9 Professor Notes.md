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