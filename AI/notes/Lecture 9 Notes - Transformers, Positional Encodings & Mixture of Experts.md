### Executive Summary

This lecture finishes the core **Transformer** architecture (self-attention, multi-head attention, normalization, positional encodings) and then introduces **Mixture of Experts (MoE)** as a modern scaling technique related to ensemble methods and mixture-of-Gaussians models. The professor also briefly reviews RNN-based sequence-to-sequence translation, beam search, and BLEU to motivate why Transformers are so dominant in language modeling.

---

## 1. Concept: RNN / Seq2Seq Review, Beam Search & BLEU

### 1. High-Level Intuition

Transformers were invented largely because **RNN-based sequence models** struggle to capture long-range dependencies and compress all information into a single vector; this section is the “before picture” that motivates attention-based models.

**Analogy:** Think of an RNN encoder–decoder as a translator who reads an entire paragraph, memorizes it as a single “gist sentence” in their head, and then tries to produce a perfect translation just from that one sentence of memory.

---

### 2. Conceptual Deep Dive

- **RNN language model:** Processes a sequence token by token, maintaining a **hidden state** (a summary of the past).
    
- **Seq2Seq (encoder–decoder):** An **encoder RNN** reads a source sentence and produces a **thought vector** $\phi$ (its final hidden state), which a **decoder RNN** then uses to generate the target sentence.
    
- Limitation: a **single vector** must carry all semantic detail of the input sentence.
    
- **Teacher forcing:** During training, the decoder at time $t$ receives the **ground-truth previous token** $y_{t-1}$; at inference, it has to use its own previous prediction instead.
    
- **Beam search:** Instead of greedily picking the most likely next token at each step, keep the top $B$ partial sequences (the “beam”) ranked by sequence probability to better approximate **maximum likelihood sequence estimation (MLSE)**.
    
- **BLEU metric:** A precision-oriented metric comparing predicted and reference translations via n-gram overlaps.
    

---

### 3. Mathematical Formulation

1. **RNN Language Model**
    
    $$  
    h_t = f(W_{xh} x_t + W_{hh} h_{t-1} + b_h)  
    $$
    
    - $h_t$: hidden state at time $t$
        
    - $x_t$: input token embedding at time $t$
        
    - $W_{xh}, W_{hh}$: input-to-hidden and hidden-to-hidden weight matrices
        
    - $b_h$: bias
        
    - $f$ : nonlinearity (e.g., $\tanh$ or ReLU)
        
    
    $$  
    p(y_t \mid y_{<t}) = \text{softmax}(W_{hy} h_t + b_y)  
    $$
    
    - $W_{hy}$: hidden-to-output weight matrix
        
    - $b_y$: output bias
        
    - $p(y_t \mid y_{<t})$: distribution over next token at time $t$
        
2. **Seq2Seq with Thought Vector**
    
    Encoder:  
    $$  
    h_t^{\text{enc}} = f(W_{xh}^{\text{enc}} x_t + W_{hh}^{\text{enc}} h_{t-1}^{\text{enc}}),\quad  
    \phi = h_T^{\text{enc}}  
    $$
    
    Decoder:  
    $$  
    h_t^{\text{dec}} = f(W_{y h}^{\text{dec}} y_{t-1} + W_{hh}^{\text{dec}} h_{t-1}^{\text{dec}} + W_{\phi h}^{\text{dec}} \phi)  
    $$
    
    - $\phi$: final encoder hidden state (“thought vector”)
        
    - $y_{t-1}$: previous target token (ground truth during training)
        
3. **Sequence Probability & Beam Search**
    
    $$  
    P(\mathbf{y}\mid \mathbf{x}) = \prod_{t=1}^{T'} p(y_t \mid y_{<t}, \mathbf{x})  
    $$
    
    - $\mathbf{x}$: source sentence
        
    - $\mathbf{y}$: candidate translation
        
    - Beam search approximates $\arg\max_{\mathbf{y}} P(\mathbf{y}\mid\mathbf{x})$ by maintaining top-$B$ partial hypotheses.
        
4. **BLEU (simplified)**
    
    For brevity, think:
    
    $$  
    \text{BLEU} \approx \text{BP} \cdot \exp\left( \sum_{n=1}^N w_n \log p_n \right)  
    $$
    
    - $p_n$: modified n-gram precisions
        
    - $w_n$: weights (often $1/N$)
        
    - BP: brevity penalty to discourage too-short translations
        

---

### 4. Worked Toy Example

Take a **tiny vocabulary**: `{"I", "like", "dogs", "<EOS>"}`.

Suppose at some time $t$, the decoder hidden state is  
$h_t = [1, 0]^\top$, and

$$  
W_{hy} =  
\begin{bmatrix}  
2 & 0 \  
1 & 1 \  
0 & 2 \  
-1 & 0  
\end{bmatrix}  
$$

Rows correspond to logits for `[I, like, dogs, <EOS>]`.

Compute logits:

$$  
z = W_{hy} h_t =  
\begin{bmatrix}  
2 & 0 \  
1 & 1 \  
0 & 2 \  
-1 & 0  
\end{bmatrix}  
\begin{bmatrix}  
1 \ 0  
\end{bmatrix}

\begin{bmatrix}  
2 \ 1 \ 0 \ -1  
\end{bmatrix}  
$$

Softmax over $z$ (approx):

- $\exp(2) \approx 7.39$
    
- $\exp(1) \approx 2.72$
    
- $\exp(0) = 1$
    
- $\exp(-1) \approx 0.37$
    

Sum $\approx 7.39 + 2.72 + 1 + 0.37 = 11.48$.

So:

- $p(\text{"I"}) \approx 7.39 / 11.48 \approx 0.64$
    
- $p(\text{"like"}) \approx 0.24$
    
- $p(\text{"dogs"}) \approx 0.09$
    
- $p(\text{""}) \approx 0.03$
    

**Greedy decoding** would choose `"I"`. In **beam search** with beam size $B=2$, we would keep the best two partial sequences for continuation, not just the single best.

---

### 5. Connections & Prerequisites

- These ideas justify why we want an architecture like the Transformer that:
    
    - Can attend to **all positions in parallel**;
        
    - Avoids bottlenecking meaning into a single thought vector;
        
    - Uses beam search & sequence-level scores more naturally.
        

**Prerequisite Refresher:**  
You should already be comfortable with **basic feedforward nets**, vector–matrix multiplication, and **softmax**. Knowing how **backpropagation** works and how probabilities factor over sequences ($\prod_t p(y_t\mid y_{<t})$) will make the rest of the lecture much easier to digest.

---

## 2. Concept: Simple Self-Attention (Non-parameterized)

### 1. High-Level Intuition

Simple self-attention builds **contextual embeddings** by letting each token look at **all other tokens** and take a weighted average of their representations.

**Analogy:** You sit in a meeting and adjust your opinion by listening more to some people (high weight) and less to others (low weight), then averaging their opinions.

---

### 2. Conceptual Deep Dive

- Start with **context-free embeddings**: matrix $X \in \mathbb{R}^{T \times D}$ (T tokens, D-dimensional embedding).
    
- Compute **similarity scores** between every pair of tokens:
    
    - $S = X X^\top \in \mathbb{R}^{T \times T}$.
        
- Apply row-wise **softmax** to each row of $S$ to get **attention weights** $A$.
    
- The new contextual representation is:
    
    $$  
    \hat{X} = A X  
    $$
    
- This is “simple” because it directly uses dot products of embeddings, without learnable query/key/value projections.
    

---

### 3. Mathematical Formulation

1. **Similarity Scores**
    
    $$  
    S = X X^\top  
    $$
    
    - $X$: token embeddings $(T \times D)$
        
    - $S_{ij}$: dot product similarity between token $i$ and token $j$
        
2. **Attention Weights**
    
    $$  
    A_{ij} = \frac{\exp(S_{ij})}{\sum_{k=1}^T \exp(S_{ik})}  
    $$
    
    - Row $i$ of $A$ sums to 1; it’s a distribution over which tokens token $i$ listens to.
        
3. **Contextualized Representations**
    
    $$  
    \hat{X} = A X  
    $$
    
    - Row $\hat{x}_i$ is a weighted average of all token embeddings, with weights from $A_{i\cdot}$.
        

---

### 4. Worked Toy Example

Let’s take 3 tokens ($T=3$), 2-dim embeddings ($D=2$):

$$  
X =  
\begin{bmatrix}  
1 & 0 \ % token 1  
0 & 1 \ % token 2  
1 & 1 % token 3  
\end{bmatrix}  
$$

1. Compute $S = X X^\top$:
    
    - $S_{11} = [1,0]\cdot[1,0] = 1$
        
    - $S_{12} = [1,0]\cdot[0,1] = 0$
        
    - $S_{13} = [1,0]\cdot[1,1] = 1$
        
    - Similarly:
        
        $$  
        S =  
        \begin{bmatrix}  
        1 & 0 & 1 \  
        0 & 1 & 1 \  
        1 & 1 & 2  
        \end{bmatrix}  
        $$
        
2. Compute softmax row-wise (approximate):
    
    - Row 1: $[1,0,1]$
        
        - $\exp(1)=2.72$, $\exp(0)=1$, $\exp(1)=2.72$
            
        - Sum $\approx 6.44$
            
        - row 1 weights $\approx [0.42, 0.16, 0.42]$
            
    
    So $A \approx  
    \begin{bmatrix}  
    0.42 & 0.16 & 0.42 \  
    \dots & \dots & \dots \  
    \dots & \dots & \dots  
    \end{bmatrix}$ (other rows analogous).
    
3. Compute $\hat{x}_1$:
    
    $$  
    \hat{x}_1 = 0.42[1,0] + 0.16[0,1] + 0.42[1,1]  
    = [0.42 + 0 + 0.42,\ 0 + 0.16 + 0.42] = [0.84, 0.58]  
    $$
    

Token 1’s new embedding “blends” information from itself and the others.

---

### 5. Connections & Prerequisites

- This mechanism is the **template** for full Transformer attention; later we just add **learnable projections (Q, K, V)**and scaling/masking.
    
- Simple self-attention already removes recurrency: all tokens attend to all others **in parallel**.
    

**Prerequisite Refresher:**  
You should know how matrix multiplication works and what a **softmax** is. Think of softmax as turning a row of scores into a **probability distribution** over other tokens.

---

## 3. Concept: Learned Self-Attention with Q, K, V (Scaled Dot-Product Attention)

### 1. High-Level Intuition

Instead of using raw embeddings, the Transformer learns three different views of each token:  
**query** = what help I’m looking for,  
**key** = what type of help I provide,  
**value** = the information that will be passed and combined.

**Analogy:** At a party, each person:

- asks: “Who here knows about X?” (query),
    
- advertises: “I’m an expert in Y” (key),
    
- and actually **shares** some knowledge when asked (value).
    

---

### 2. Conceptual Deep Dive

- Each token embedding $x_i$ is linearly projected into:
    
    - **Query** $q_i$: what this token is looking for.
        
    - **Key** $k_i$: what this token claims to be.
        
    - **Value** $v_i$: the representation to be moved.
        
- Matrices:
    
    - $Q = X W_Q$, $K = X W_K$, $V = X W_V$.
        
- Attention scores are computed via **generalized dot product** $Q K^\top$.
    
- Scores are scaled by $1/\sqrt{d_k}$ (where $d_k$ is key dimension) to avoid very large magnitudes that would make the softmax overly “peaky.”
    
- The resulting weights are used to form a weighted sum of values, yielding contextualized representations.
    

---

### 3. Mathematical Formulation

Given $X \in \mathbb{R}^{T \times D}$:

1. **Linear Projections**
    
    $$  
    Q = X W_Q,\quad K = X W_K,\quad V = X W_V  
    $$
    
    - $W_Q, W_K, W_V \in \mathbb{R}^{D \times d_k}$ (often $d_k = D/H$ where $H$ is \#heads)
        
2. **Scaled Dot-Product Attention**
    
    $$  
    S = \frac{1}{\sqrt{d_k}} Q K^\top  
    $$
    
    - $S_{ij}$: scaled similarity between query of token $i$ and key of token $j$
        
    
    $$  
    A = \text{softmax}(S) \quad\text{(row-wise)}  
    $$
    
3. **Output**
    
    $$  
    \text{Attention}(Q,K,V) = A V  
    $$
    
    - For each token $i$, output is $\sum_j A_{ij} v_j$ (weighted sum of value vectors).
        

---

### 4. Worked Toy Example

Let $T=2$, $D=d_k=2$:

$$  
X =  
\begin{bmatrix}  
1 & 0 \ % token 1  
0 & 1 % token 2  
\end{bmatrix},\quad  
W_Q = W_K = W_V = I_2  
$$

Then:

- $Q = K = V = X = \begin{bmatrix}1 & 0\0 & 1\end{bmatrix}$.
    

1. Scores:
    
    $$  
    S = QK^\top = X X^\top =  
    \begin{bmatrix}  
    1 & 0\  
    0 & 1  
    \end{bmatrix}  
    $$
    
2. Scaled scores with $d_k = 2$:
    
    $$  
    S' = \frac{1}{\sqrt{2}} S = \begin{bmatrix}  
    1/\sqrt{2} & 0\  
    0 & 1/\sqrt{2}  
    \end{bmatrix}  
    $$
    
3. Softmax row-wise:
    
    - Row 1: $\text{softmax}([1/\sqrt{2}, 0]) \approx [0.68, 0.32]$
        
    - Row 2: symmetric: $[0.32, 0.68]$
        
4. Output:
    
    $$  
    \text{Att}(Q,K,V) = A V =  
    \begin{bmatrix}  
    0.68 & 0.32\  
    0.32 & 0.68  
    \end{bmatrix}  
    $$
    

So each token now partially attends to the other: token 1’s new rep is mostly itself but with some influence from token 2.

---

### 5. Connections & Prerequisites

- This extends **simple self-attention** by learning **task-specific similarity measures** via $W_Q$ and $W_K$, and task-specific values via $W_V$.
    
- The learned projections let the model represent roles like **subject**, **verb**, and **object** in a more flexible way.
    

**Prerequisite Refresher:**  
You should be comfortable with **linear projections** (matrix multiplication / change of basis) and dot products as similarity. Remember that a dot product is **large** when two vectors point in similar directions.

---

## 4. Concept: Masked Self-Attention for Autoregressive Language Modeling

### 1. High-Level Intuition

During **training** we see full sequences, but during **generation** we only know past tokens. Masked self-attention enforces that **token $t$ can’t peek at future tokens $>t$**.

**Analogy:** While writing a sentence word by word, you’re not allowed to look at words from the future draft—you can only use what you’ve already written.

---

### 2. Conceptual Deep Dive

- In the attention score matrix $S \in \mathbb{R}^{T\times T}$, $S_{ij}$ measures how much token $i$ attends to token $j$.
    
- For **autoregressive (decoder) Transformers**, we must **zero out** attention from token $i$ to all future tokens $j>i$.
    
- Implementation trick:
    
    - Add a **mask** $M$ with $M_{ij} = 0$ if $j \le i$, and $M_{ij} = -\infty$ if $j > i$.
        
    - Compute attention scores as $S' = S + M$.
        
    - Softmax over rows makes attention to future positions effectively zero.
        
- Result: at time position $t$, the representation only depends on tokens $1,\dots,t$.
    

---

### 3. Mathematical Formulation

Let $S$ be the unmasked score matrix:

1. **Mask Construction**
    
    $$  
    M_{ij} =  
    \begin{cases}  
    0 & j \le i \  
    -\infty & j > i  
    \end{cases}  
    $$
    
2. **Masked Scores and Attention Weights**
    
    $$  
    S' = S + M  
    $$  
    $$  
    A_{ij} = \frac{\exp(S'_{ij})}{\sum_{k=1}^T \exp(S'_{ik})}  
    $$
    

- For $j>i$, $S'_{ij} = -\infty$, so $A_{ij} = 0$.
    

---

### 4. Worked Toy Example

Suppose 3 tokens ($T=3$) and (unscaled) scores:

$$  
S =  
\begin{bmatrix}  
1 & 2 & 3\  
1 & 2 & 3\  
1 & 2 & 3  
\end{bmatrix}  
$$

Mask $M$ for causal attention:

$$  
M =  
\begin{bmatrix}  
0 & -\infty & -\infty\  
0 & 0 & -\infty\  
0 & 0 & 0  
\end{bmatrix}  
$$

Then:

$$  
S' = S + M =  
\begin{bmatrix}  
1 & -\infty & -\infty\  
1 & 2 & -\infty\  
1 & 2 & 3  
\end{bmatrix}  
$$

- Row 1 softmax: only position 1 is finite → token 1 attends only to itself.
    
- Row 2 softmax: positions 1 and 2; can’t see token 3.
    
- Row 3: can see all 1–3.
    

---

### 5. Connections & Prerequisites

- Masked self-attention is used in **decoder-only** language models (GPT-style) and in the **decoder** part of encoder–decoder Transformers.
    
- It’s crucial for sequence generation to avoid **information leakage**.
    

**Prerequisite Refresher:**  
You should understand how softmax behaves with **very negative numbers**: $\exp(-10^9) \approx 0$, so they effectively drop out of the probability distribution.

---

## 5. Concept: Multi-Head Self-Attention & the Transformer Layer

### 1. High-Level Intuition

Multi-head self-attention lets the model look at the sequence in **multiple ways at once**—different heads can focus on different relationships (e.g., subject–verb, coreference, positional patterns). The **Transformer layer** then wraps this attention with normalization, residual connections, and an MLP to build deep representations.

**Analogy:** In a team meeting, one specialist tracks timing, another tracks dependencies, and another tracks priorities; afterwards, a manager combines their notes into a single plan.

---

### 2. Conceptual Deep Dive

- **Multi-head attention:**
    
    - Use $H$ different sets of $W_Q^{(h)}, W_K^{(h)}, W_V^{(h)}$.
        
    - Each head attends separately, producing outputs $O^{(h)}$.
        
    - Concatenate $(O^{(1)},\dots,O^{(H)})$ and project with $W^O$ back to dimension $D$.
        
- Intuition: different heads specialize in different “patterns” (loosely, **grammatical or semantic roles**).
    
- **Transformer block (decoder layer as discussed):**
    
    1. Input $X$ → **LayerNorm** → Multi-Head Self-Attention → add residual from $X$ → $Z$.
        
    2. $Z$ → LayerNorm → **Position-wise MLP** → add residual from $Z$ → $V^{\hat{}}$ (new token representations).
        
- Stacking many such layers builds the **body** of the Transformer; a final **linear + softmax head** maps to vocabulary probabilities.
    

---

### 3. Mathematical Formulation

1. **Multi-Head Attention**
    
    For head $h$:
    
    $$  
    Q^{(h)} = X W_Q^{(h)},\quad K^{(h)} = X W_K^{(h)},\quad V^{(h)} = X W_V^{(h)}  
    $$
    
    $$  
    O^{(h)} = \text{Attention}(Q^{(h)}, K^{(h)}, V^{(h)})  
    $$
    
    Concatenate and project:
    
    $$  
    O = \text{Concat}(O^{(1)},\dots,O^{(H)}) W^O  
    $$
    
    - $W^O \in \mathbb{R}^{(H d_k) \times D}$.
        
2. **Transformer Decoder Layer (Pre-Norm)**
    
    - **Attention sublayer:**
        
        $$  
        X' = X + O,\quad\text{where }O = \text{MHA}(\text{LayerNorm}(X))  
        $$
        
    - **MLP sublayer:**
        
        $$  
        V^{\hat{}} = X' + \text{MLP}(\text{LayerNorm}(X'))  
        $$
        
    - $\text{MLP}$ is typically 2 linear layers with a nonlinearity (e.g. GELU) and often an intermediate dimension larger than $D$.
        
3. **Language Modeling Head**
    
    $$  
    \text{logits}_t = W_{\text{LM}} v_t^{\hat{}} + b_{\text{LM}}, \quad  
    p(y_t \mid y_{<t}) = \text{softmax}(\text{logits}_t)  
    $$
    

---

### 4. Worked Toy Example (Tiny 2-Head Setup)

Let $T=2$, $D=2$, and we choose $H=2$ with $d_k=1$:

- $X = \begin{bmatrix}1 & 0\\0 & 1\end{bmatrix}$.
    

Head 1:

- $W_Q^{(1)} = W_K^{(1)} = W_V^{(1)} = \begin{bmatrix}1\\0\end{bmatrix}$ (maps 2D → scalar).
    
- So $Q^{(1)} = K^{(1)} = V^{(1)} = \begin{bmatrix}1\\0\end{bmatrix}$.
    

Scores:

$$  
S^{(1)} = Q^{(1)} (K^{(1)})^\top =  
\begin{bmatrix}  
1\\0  
\end{bmatrix}  
[1, 0] =  
\begin{bmatrix}  
1 & 0\\  
0 & 0  
\end{bmatrix}  
$$

Softmax rows:

- Row 1: $\text{softmax}([1,0]) \approx [0.73,0.27]$
    
- Row 2: $\text{softmax}([0,0]) = [0.5,0.5]$
    

Outputs:

- For simplicity, use $d_k=1$, $V^{(1)}$ is column vector; $O^{(1)} = A^{(1)} V^{(1)}$ is also scalar per token.
    

Head 2 could use a different projection, e.g. focus on 2nd dimension.

Key intuition: **two different heads** can produce different attention patterns; concatenating them and linearly mixing allows richer representations than a single head.

---

### 5. Connections & Prerequisites

- Multi-head attention is the **core building block** used throughout the Transformer stack.
    
- The stacked masked Transformer layers form the **body**, and the linear + softmax head performs **next-token prediction**.
    

**Prerequisite Refresher:**  
You should be familiar with **residual connections** from ResNets and basic **MLPs** (linear → nonlinearity → linear). Knowing why depth can cause training difficulties (vanishing gradients) helps appreciate residuals and normalization.

---

## 6. Concept: Batch Normalization vs Layer Normalization

### 1. High-Level Intuition

Both BatchNorm and LayerNorm try to keep activations in a “nice range” (controlled mean and variance) to stabilize training, but **BatchNorm normalizes across the batch**, while **LayerNorm normalizes across features within a single example**—which is more suitable when batch sizes are small or variable, as in large Transformers.

**Analogy:**

- BatchNorm: “Adjust everyone in a group relative to the group’s average.”
    
- LayerNorm: “Adjust each person based on their own internal balance of traits.”
    

---

### 2. Conceptual Deep Dive

- **Batch Normalization:**
    
    - For each feature dimension, compute mean and variance **over the batch**.
        
    - Normalize activations, then apply learned scale ($\gamma$) and shift ($\beta$).
        
    - Works well when you can use large batches; less ideal with tiny or changing batch sizes.
        
- **Layer Normalization:**
    
    - Normalizes across the **feature dimension** for each example/token independently.
        
    - Perfect when batch may be small (even 1 token), common in Transformers with large context windows and VRAM constraints.
        
- In the Transformer, LayerNorm is applied to each token’s feature vector (row of $X$).
    

---

### 3. Mathematical Formulation

1. **Batch Normalization (for a single feature dimension $j$)**
    
    For batch ${x^{(1)}_j,\dots,x^{(B)}_j}$:
    
    $$  
    \mu_j = \frac{1}{B}\sum_{b=1}^B x^{(b)}_j,\quad  
    \sigma_j^2 = \frac{1}{B} \sum_{b=1}^B (x^{(b)}_j - \mu_j)^2  
    $$
    
    $$  
    \hat{x}^{(b)}_j = \frac{x^{(b)}_j - \mu_j}{\sqrt{\sigma_j^2 + \epsilon}}  
    $$
    
    $$  
    y^{(b)}_j = \gamma_j \hat{x}^{(b)}_j + \beta_j  
    $$
    
    - $B$: batch size
        
    - $\gamma_j, \beta_j$: trainable scale and shift for feature $j$.
        
2. **Layer Normalization (for one token vector $x \in \mathbb{R}^D$)**
    
    $$  
    \mu = \frac{1}{D}\sum_{j=1}^D x_j,\quad  
    \sigma^2 = \frac{1}{D}\sum_{j=1}^D (x_j - \mu)^2  
    $$
    
    $$  
    \hat{x}_j = \frac{x_j - \mu}{\sqrt{\sigma^2 + \epsilon}},\quad  
    y_j = \gamma_j \hat{x}_j + \beta_j  
    $$
    
    - Now statistics are over **features** instead of over the batch.
        

---

### 4. Worked Toy Example

Consider a token vector $x = [2, 4]$ and we do **LayerNorm** with $D=2$.

1. Mean:
    
    $$  
    \mu = (2 + 4)/2 = 3  
    $$
    
2. Variance:
    
    $$  
    \sigma^2 = \frac{1}{2}[(2-3)^2 + (4-3)^2] = \frac{1}{2}(1 + 1) = 1  
    $$
    
3. Normalized:
    
    $$  
    \hat{x} = \left[\frac{2-3}{\sqrt{1+\epsilon}},  
    \frac{4-3}{\sqrt{1+\epsilon}}\right]  
    \approx [-1, 1]  
    $$
    
4. With $\gamma = [1,1]$, $\beta = [0,0]$, output $y = [-1,1]$.
    

The feature vector is now zero-mean and unit variance, which helps stable gradient flow.

---

### 5. Connections & Prerequisites

- LayerNorm is used **inside each Transformer layer** before attention and MLP blocks, while BatchNorm is more common in CNNs and classic feedforward architectures.
    
- Understanding this helps relate Transformers back to earlier network designs (e.g., ResNets).
    

**Prerequisite Refresher:**  
You should know how **mean and variance** work, and why controlling activation scale (e.g., avoiding saturation in sigmoids or extreme ReLU clipping) is useful for optimization.

---

## 7. Concept: Positional Encodings (Fourier / Sinusoidal Method)

### 1. High-Level Intuition

Transformers are **permutation invariant** in their pure form: reshuffling the rows of $X$ gives the same outputs. **Positional encodings** inject information about token order so word meaning can depend on position.

**Analogy:** You and your friends all wear identical clothes (identical embeddings), but your seat number on a bus (position encoding) distinguishes who is “first,” “second,” etc.

---

### 2. Conceptual Deep Dive

- We form a **positional encoding vector** $r_i \in \mathbb{R}^D$ for each position $i$.
    
- Then we **add** it to the original embedding:
    
    $$  
    \tilde{x}_i = x_i + r_i  
    $$
    
    rather than concatenating, to keep dimension fixed and leverage linearity of subsequent layers.
    
- The **Fourier / sinusoidal** method uses sinusoids of varying frequencies across dimensions; positions become unique patterns of sine and cosine values.
    
- Interpretation: similar to encoding position in **binary**, where different bits flip at different frequencies (fastest bit alternates every step, next every 2 steps, etc.).
    
- The sinusoidal version provides **smooth, analog** encodings in $[-1,1]$, which are easier to integrate into the embedding space than hard 0/1 indicators.
    

---

### 3. Mathematical Formulation

Original Vaswani-style formula (for model dimension $D$):

For position $i$ and dimension $2k$ / $2k+1$:

$$  
\text{PE}_{(i,2k)} = \sin\left(\frac{i}{10000^{2k/D}}\right),\quad  
\text{PE}_{(i,2k+1)} = \cos\left(\frac{i}{10000^{2k/D}}\right)  
$$

- $i$: position index (0-based or 1-based depending on convention)
    
- $k$: frequency index
    
- $10000^{2k/D}$ controls the wavelength; low $k$ → high frequency, high $k$ → low frequency.
    

Then we define $\tilde{X}$ by $\tilde{x}_i = x_i + \text{PE}_i$.

---

### 4. Worked Toy Example

Let’s take $D=4$ and positions $i=0,1$, using a **toy base** of 10 instead of 10000 to keep numbers simple.

For $k=0$:

- $\text{PE}_{(i,0)} = \sin(i / 10^{0}) = \sin(i)$
    
- $\text{PE}_{(i,1)} = \cos(i / 10^{0}) = \cos(i)$
    

For $k=1$:

- $\text{PE}_{(i,2)} = \sin(i / 10^{2/4}) = \sin(i / \sqrt{10})$
    
- $\text{PE}_{(i,3)} = \cos(i / 10^{2/4}) = \cos(i / \sqrt{10})$
    

Compute approximate values for $i=0$:

- $\sin(0)=0,\ \cos(0)=1,\ \sin(0)=0,\ \cos(0)=1$  
    → $\text{PE}_0 \approx [0,1,0,1]$
    

For $i=1$:

- $\sin(1)\approx0.84,\ \cos(1)\approx0.54$
    
- $\sin(1/\sqrt{10})\approx\sin(0.316)\approx0.31$
    
- $\cos(1/\sqrt{10})\approx\cos(0.316)\approx0.95$
    

So:

- $\text{PE}_1 \approx [0.84, 0.54, 0.31, 0.95]$
    

Even in this tiny example, positions 0 and 1 have distinct patterns across dimensions, encoding ordering information.

---

### 5. Connections & Prerequisites

- Positional encodings are added **once at the bottom** of the Transformer (or updated per layer in some variants).
    
- Without them, the Transformer would treat the input as a **bag of tokens**, losing sequencing.
    

**Prerequisite Refresher:**  
You should recall basic trigonometric functions and that different **frequencies** (slow vs fast oscillations) can be combined to uniquely encode positions—this is closely related to **Fourier analysis**.

---

## 8. Concept: Mixture of Experts (MoE) & Relation to Ensembles / Mixture of Gaussians

### 1. High-Level Intuition

Mixture of Experts (MoE) models use a **gating network** to route each input through a subset of specialized “experts,” then combine their outputs. This improves **parameter efficiency** and can scale models without activating every parameter for every input.

**Analogy:** Instead of asking a random panel of doctors every time, a receptionist (gating network) decides which specialists are most relevant to your symptoms, then averages their diagnoses.

---

### 2. Conceptual Deep Dive

- Starts from the idea of **ensemble methods**: many weak predictors whose aggregated output is better than any individual one.
    
- **Mixture of experts** is a **conditional ensemble**:
    
    - Different experts specialize on different regions of the data distribution $p_{\text{data}}(x)$.
        
    - A **gating network** $g(x)$ decides how much each expert should contribute for a particular input.
        
- Analogy to **mixture of Gaussians**:
    
    $$  
    p(x) = \sum_{k=1}^K \pi_k \mathcal{N}(x \mid \mu_k, \Sigma_k)  
    $$
    
    where each Gaussian covers a portion of the density and $\pi_k$ are mixing coefficients.
    
- For MoE:
    
    $$  
    \hat{y}(x) = \sum_{k=1}^K g_k(x), f_k(x)  
    $$
    
    - $f_k(x)$: prediction from expert $k$
        
    - $g_k(x)$: gating function output for expert $k$ (often from softmax; $g_k(x) \ge 0$, $\sum_k g_k(x)=1$).
        
- Compared to standard ensembles:
    
    - In ensembles, each model tries to approximate the **entire** $p_{\text{data}}$; in MoE, each expert focuses on a **partition** of the data.
        
    - The gating network is **trainable** and input-dependent.
        
- Error correlation:
    
    - **Best case:** experts’ errors are uncorrelated → averaging drastically reduces error.
        
    - **Worst case:** experts make the same mistakes → no benefit.
        
    - These ideas connect back to ensemble theory (e.g., Goodfellow et al.’s analysis with mean squared error).
        

---

### 3. Mathematical Formulation

1. **Mixture of Gaussians (Analogy)**
    
    $$  
    p(x) = \sum_{k=1}^K \pi_k, \mathcal{N}(x\mid \mu_k,\Sigma_k),  
    \quad \sum_{k=1}^K \pi_k = 1,\ \pi_k \ge 0  
    $$
    
    - Each component $k$ has mean $\mu_k$ and covariance $\Sigma_k$.
        
    - $\pi_k$: mixing coefficients (“responsibilities” of each component).
        
2. **Mixture of Experts Output**
    
    $$  
    \hat{y}(x) = \sum_{k=1}^K g_k(x), f_k(x)  
    $$
    
    - $f_k(x)$: prediction from expert $k$ (could be a neural network).
        
    - $g_k(x)$: gating network output for expert $k$.
        
    - Typically $g(x) = \text{softmax}(W_g x + b_g)$ to ensure $g_k \ge 0$ and $\sum_k g_k = 1$.
        
3. **Constraints on Gating Weights**
    
    $$  
    \sum_{k=1}^K g_k(x) = 1,\quad g_k(x) \ge 0  
    $$
    
    These make $g(x)$ a **probability distribution** over experts for input $x$.
    

---

### 4. Worked Toy Example

Suppose we have 2 experts ($K=2$) doing **scalar regression**:

- Expert 1: $f_1(x) = 2x$
    
- Expert 2: $f_2(x) = -x$
    

Gating network outputs:

- For $x=1$, $g(1) = [0.8, 0.2]$
    
- For $x=-1$, $g(-1) = [0.3, 0.7]$
    

Compute outputs:

1. At $x=1$:
    
    $$  
    \hat{y}(1) = 0.8 \cdot f_1(1) + 0.2 \cdot f_2(1)  
    = 0.8 \cdot 2 + 0.2 \cdot (-1)  
    = 1.6 - 0.2  
    = 1.4  
    $$
    
2. At $x=-1$:
    
    $$  
    \hat{y}(-1) = 0.3 \cdot f_1(-1) + 0.7 \cdot f_2(-1)  
    = 0.3 \cdot (-2) + 0.7 \cdot (1)  
    = -0.6 + 0.7  
    = 0.1  
    $$
    

Different inputs activate experts differently, leading to input-dependent combinations of predictions.

---

### 5. Connections & Prerequisites

- In large language models, MoE layers can be inserted into the Transformer to **activate only a subset of experts per token**, improving efficiency on hardware with limited memory.
    
- The lecture highlights the connection between MoE, **ensembles**, and **mixture of Gaussians**, and mentions that a similar best/worst-case analysis can be done under **cross-entropy** loss.
    

**Prerequisite Refresher:**  
You should remember:

- Basic ensemble learning ideas (averaging predictors),
    
- What a **Gaussian distribution** is,
    
- And how **softmax** can be used to define a probability distribution over discrete options.
    

---

### Key Takeaways & Formulas

- **Seq2Seq with RNNs** compresses an entire sentence into a single vector $\phi$, motivating more flexible architectures like Transformers.
    
- **Self-attention** computes contextualized embeddings via  
    $$  
    \text{Attention}(Q,K,V) = \text{softmax}\left(\frac{QK^\top}{\sqrt{d_k}}\right)V  
    $$  
    with learnable **Q, K, V** projections.
    
- **Masked self-attention** ensures token $t$ only attends to positions $\le t$, which is essential for autoregressive language modeling.
    
- **Multi-head attention + LayerNorm + MLP + residuals** form the **Transformer layer**, stacked many times to build deep sequence models.
    
- **Positional encodings** (sinusoidal/Fourier) inject order information into otherwise permutation-invariant Transformers.
    
- **Mixture of Experts** models use a trainable gating network to combine specialized experts:  
    $$  
    \hat{y}(x) = \sum_{k=1}^K g_k(x), f_k(x),  
    $$  
    linking ideas from ensembles and mixture-of-Gaussians to modern large-scale Transformer architectures.
    

