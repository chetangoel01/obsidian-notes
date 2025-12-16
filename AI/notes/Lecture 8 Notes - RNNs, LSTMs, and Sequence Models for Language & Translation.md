### Executive Summary

This lecture develops the theory and practice of **sequence models** for language: starting from basic **recurrent neural networks (RNNs)**, analyzing **gradient flow** (vanishing/exploding gradients), and then introducing **LSTMs** as a fix. It then applies these architectures to **language modeling**, **neural machine translation (NMT)** with encoder–decoder structures (including **bidirectional RNNs**), and finally to **decoding strategies** like **beam search / MLSE** and evaluation with **BLEU**. The lecture concludes by motivating **Transformers** and **attention** as a non-recurrent, fully parallel alternative that can build richer **contextual embeddings**. 

---

## 1. Concept: Recurrent Neural Networks (RNNs) for Sequences

### 1. High-Level Intuition

**What problem does this solve?**  
RNNs model data where **order matters**, like text or time series, by keeping a **hidden state** that carries information from the past.

**Analogy:**  
Think of reading a sentence word by word while taking notes in the margin; each new word updates your note. The note is the **hidden state**.

---

### 2. Conceptual Deep Dive

An **RNN** processes an input sequence $x_1, x_2, \dots, x_T$ one step at a time and maintains a **hidden state** $h_t$ that summarizes all information seen so far.

Key terms:

- **Input vector** **$x_t$**: representation of the token at time $t$ (often an embedding).
    
- **Hidden state** **$h_t$**: internal memory at time $t$.
    
- **Recurrent weights**: parameters that tell the network how to combine the current input and previous hidden state.
    
- **Unrolling in time**: viewing the recurrent computation as a deep network with one layer per time step. 
    

In the lecture, the instructor first shows **a single recurrent neuron** (one hidden unit) and then **stacks many such units into a layer**, so that $h_t$ becomes a vector rather than a scalar. The **dimension of $h_t$** (“number of neurons”) is a hyperparameter; an intuitive way to choose it is to look at the **spectrum of singular values** (via SVD) of your time-series data matrix and pick the number of significant components. 

To get predictions from the hidden state, we add a **head**:

- For **regression** (e.g., predicting a price), a simple linear unit maps $h_t$ to a scalar. 
    
- For **classification** (e.g., next token), a fully connected layer + **softmax** maps $h_t$ to a probability distribution over classes. 
    

---

### 3. Mathematical Formulation

We’ll use the concatenation form, which matches the lecture: 

1. **Hidden-state update**
    

$$  
h_t = \tanh\big(W [x_t; h_{t-1}] + b\big)  
$$

- $x_t \in \mathbb{R}^{d_{\text{in}}}$: input vector at time $t$.
    
- $h_{t-1} \in \mathbb{R}^{d_h}$: previous hidden state.
    
- $[x_t; h_{t-1}] \in \mathbb{R}^{d_{\text{in}} + d_h}$: concatenation of input and previous state.
    
- $W \in \mathbb{R}^{d_h \times (d_{\text{in}} + d_h)}$: weight matrix combining input and past state.
    
- $b \in \mathbb{R}^{d_h}$: bias vector.
    
- $\tanh(\cdot)$ applied **elementwise** ensures outputs in $(-1, 1)$, which the lecturer prefers to avoid “lots of zeros” in the feedback loop. 
    

2. **Output head – regression**
    

$$  
\hat{y}_t = v^\top h_t + c  
$$

- $v \in \mathbb{R}^{d_h}$, $c \in \mathbb{R}$: parameters of the linear regressor.
    
- $\hat{y}_t \in \mathbb{R}$: scalar prediction (e.g., price at time $t$). 
    

3. **Output head – classification (softmax)**
    

First compute logits:  
$$  
o_t = V h_t + c  
$$

Then apply softmax:  
$$  
P(y_t = k \mid x_{\le t}) = \frac{\exp(o_{t,k})}{\sum_{j=1}^K \exp(o_{t,j})}  
$$

- $V \in \mathbb{R}^{K \times d_h}$: maps hidden state to class logits.
    
- $K$: number of classes (e.g., vocabulary size for tokens).
    
- $o_{t,k}$: logit for class $k$.
    
- $P(y_t = k \mid x_{\le t})$: model’s probability of class $k$ at time $t$. 
    

---

### 4. Worked Toy Example

Consider a **scalar** RNN (1D hidden state, 1D input) to keep arithmetic easy:

- Hidden update:  
    $h_t = \tanh(w_x x_t + w_h h_{t-1} + b)$
    
- Output (regression):  
    $\hat{y}_t = v h_t + c$
    

Let:

- $w_x = 1.0,; w_h = 0.5,; b = 0$
    
- $v = 2.0,; c = 0$
    
- Initial state $h_0 = 0$
    
- Inputs: $x_1 = 1.0,; x_2 = -0.5$
    

**Step 1 (t = 1)**

1. Pre-activation:  
    $a_1 = w_x x_1 + w_h h_0 + b = 1.0 \cdot 1.0 + 0.5 \cdot 0 + 0 = 1.0$
    
2. Hidden state:  
    $h_1 = \tanh(1.0) \approx 0.7616$
    
3. Output:  
    $\hat{y}_1 = v h_1 = 2.0 \cdot 0.7616 \approx 1.5232$
    

**Step 2 (t = 2)**

1. Pre-activation:  
    $a_2 = w_x x_2 + w_h h_1 + b = 1.0 \cdot (-0.5) + 0.5 \cdot 0.7616 = -0.5 + 0.3808 = -0.1192$
    
2. Hidden state:  
    $h_2 = \tanh(-0.1192) \approx -0.1187$
    
3. Output:  
    $\hat{y}_2 = v h_2 = 2.0 \cdot (-0.1187) \approx -0.2374$
    

You can see how the **state carries information** from $x_1$ into the computation at $t=2$.

---

### 5. Connections & Prerequisites

**Prerequisite Refresher:**

- You should understand **basic feedforward networks**, **matrix–vector multiplication**, and **activation functions** like $\tanh$ and ReLU, as used in earlier DNN and CNN lectures.
    
- You should also recall **tokenization and embeddings**: we usually feed an **embedding vector** into the RNN at each time step rather than raw token IDs. 
    

---

## 2. Concept: Gradient Flow in RNNs – Vanishing & Exploding Gradients

### 1. High-Level Intuition

**Problem:**  
When we backpropagate errors through **many time steps**, gradients can either **shrink to almost zero** or **blow up**, making training unstable or ineffective.

**Analogy:**  
Imagine whispering a message down a long line of people: if each person whispers slightly softer, the message vanishes; if each adds drama and shouts louder, it explodes into nonsense.

---

### 2. Conceptual Deep Dive

Training RNNs uses **backpropagation through time (BPTT)**: we unroll the RNN and backpropagate the loss from the last time step all the way to earlier steps. 

The gradient flowing back through the hidden states repeatedly encounters the **recurrent weight matrix** (called $W'$ in the lecture) and the derivative of $\tanh$. The instructor emphasizes: **some tensor along the recurrent path controls the fate of the gradient**, and in this simple RNN that tensor is essentially $W'$ (plus the nonlinearity). 

He then explains via a simple **scalar analogy**:

- If you have $h_t = w h_{t-1}$ with $|w| < 1$, then $|h_t|$ shrinks exponentially — the **vanishing gradient** case.
    
- If $|w| > 1$, $|h_t|$ grows exponentially — the **exploding gradient** case. 
    

In the full vector case, this generalizes to the **eigenvalues** of the recurrent matrix $W'$:

- All eigenvalues with magnitude $< 1$ → vanishing gradients.
    
- Some eigenvalues with magnitude $> 1$ → exploding gradients. 
    

---

### 3. Mathematical Formulation

Consider a simplified RNN:

$$  
h_t = \phi(W' h_{t-1}), \quad \text{(ignore inputs/bias for intuition)}  
$$

Suppose we have a loss $L$ that depends on $h_T$. The gradient w.r.t. $h_t$ is approximately:

$$  
\frac{\partial L}{\partial h_t}  
= \frac{\partial L}{\partial h_T}  
\prod_{k=t+1}^{T}  
\frac{\partial h_k}{\partial h_{k-1}}  
$$

Each term in the product is a Jacobian:

$$  
\frac{\partial h_k}{\partial h_{k-1}}  
\approx D_\phi(W' h_{k-1}) , W'  
$$

- $D_\phi$ is a diagonal matrix with $\phi'$ on the diagonal (for $\tanh$, entries lie in $(0, 1]$).
    

Ignoring $D_\phi$ for intuition, the product behaves like $(W')^{T-t}$. If $\lambda$ is an eigenvalue of $W'$, then along its eigenvector direction, repeated multiplication scales by roughly $\lambda^{T-t}$. Thus:

- If $|\lambda| < 1$, $\lambda^{T-t} \to 0$ as $T-t$ grows → **vanishing gradient**.
    
- If $|\lambda| > 1$, $\lambda^{T-t} \to \infty$ → **exploding gradient**. 
    

---

### 4. Worked Toy Example

Scalar case (1D hidden state):

Let

- $h_t = w h_{t-1}$, no input, loss $L = \tfrac12 h_T^2$, and $h_0 = 1$.
    

Then:

- $h_1 = w h_0 = w$
    
- $h_2 = w h_1 = w^2$
    
- …
    
- $h_T = w^T$
    

Gradient:

$$  
\frac{\partial L}{\partial h_T} = h_T = w^T  
$$

For $t < T$:

$$  
\frac{\partial h_T}{\partial h_t} = w^{T-t}  
\quad\Rightarrow\quad  
\frac{\partial L}{\partial h_t}  
= \frac{\partial L}{\partial h_T} \cdot \frac{\partial h_T}{\partial h_t}  
= w^T \cdot w^{T-t} = w^{2T-t}  
$$

Pick $w = 0.5$, $T=10$:

- $h_{10} = 0.5^{10} \approx 0.00098$
    
- $\frac{\partial L}{\partial h_0} = 0.5^{20} \approx 9.5 \times 10^{-7}$ → **essentially zero** → vanishing gradient.
    

Pick $w = 1.5$, $T=10$:

- $h_{10} = 1.5^{10} \approx 57.7$
    
- $\frac{\partial L}{\partial h_0} = 1.5^{20} \approx 3.3 \times 10^{3}$ → **huge** → exploding gradient.
    

---

### 5. Connections & Prerequisites

**Prerequisite Refresher:**

- You should be comfortable with **matrix powers**, **eigenvalues/eigenvectors**, and **backpropagation** in feedforward networks.
    
- The motivation for LSTMs (next section) is precisely to **control gradient flow** by adding special paths where information and gradients can travel more safely over many time steps. 
    

---

## 3. Concept: Long Short-Term Memory (LSTM) Networks

### 1. High-Level Intuition

**What problem does an LSTM solve?**  
LSTMs modify the RNN cell to **remember important information for a long time** while forgetting irrelevant details, greatly reducing vanishing/exploding gradients.

**Analogy:**  
Imagine you keep a **running summary diary (cell state)**, and every day you decide:

- what to **add** (input gate),
    
- what to **forget** from older pages (forget gate),
    
- and how much to **reveal** to others (output gate).
    

---

### 2. Conceptual Deep Dive

An LSTM introduces **two states** per time step:

- **Cell state** $s_t$ (lecture called this the "long-term hidden state" or highway).
    
- **Hidden state** $h_t$ (short-term state feeding the next layer / output). 
    

Between $s_{t-1}$ and $s_t$, there is a **“highway” connection** that bypasses most nonlinearity, only modulated by the **forget gate** and **input gate**. This supports better gradient flow, similar in spirit to **ResNet skip connections** but over time rather than depth. 

LSTM components (each is a vector of same size as $h_t$):

- **Forget gate** $f_t$: decides how much of $s_{t-1}$ to keep.
    
- **Input gate** $i_t$: decides how much new candidate information to write into $s_t$.
    
- **Output gate** $o_t$: decides how much of $s_t$ to expose as $h_t$.
    
- **Candidate** $g_t$: new content that might be added to the cell state.
    

All gates use a **sigmoid** activation so outputs lie in $(0,1)$, enabling **soft selection** of how much to keep or discard. 

The instructor illustrates this with a **summarization example**: “The bank, faced with political over interest rates, introduced annual savings account.” The network should **ignore** the phrase between commas (less relevant politically charged clause) and focus on the main event: **“bank introduced annual savings account”**. The **input gate** learns to block those less relevant tokens from influencing the long-term memory $s_t$. 

---

### 3. Mathematical Formulation

Standard LSTM equations (vector form):

1. **Gates and candidate**
    

$$  
\begin{aligned}  
i_t &= \sigma(W_i x_t + U_i h_{t-1} + b_i) \  
f_t &= \sigma(W_f x_t + U_f h_{t-1} + b_f) \  
o_t &= \sigma(W_o x_t + U_o h_{t-1} + b_o) \  
g_t &= \tanh(W_g x_t + U_g h_{t-1} + b_g)  
\end{aligned}  
$$

- $x_t$: input vector at time $t$.
    
- $h_{t-1}$: previous hidden state.
    
- $W_* \in \mathbb{R}^{d_h \times d_{\text{in}}}$, $U_* \in \mathbb{R}^{d_h \times d_h}$: learnable weight matrices.
    
- $b_* \in \mathbb{R}^{d_h}$: biases.
    
- $\sigma(\cdot)$: sigmoid, output in $(0,1)$.
    
- $\tanh(\cdot)$: candidate nonlinearity.
    

2. **Cell state update (highway)**
    

$$  
s_t = f_t \odot s_{t-1} + i_t \odot g_t  
$$

- $s_{t-1}$: previous cell (long-term) state.
    
- $\odot$: elementwise multiplication.
    
- $f_t \odot s_{t-1}$: “remember” part.
    
- $i_t \odot g_t$: “write new” part.
    

3. **Hidden state**
    

$$  
h_t = o_t \odot \tanh(s_t)  
$$

- $h_t$: short-term hidden state used for output and next step.
    
- $o_t$: controls how much of the (squashed) cell state is exposed.
    

This architecture allows **gradients to flow more stably** along $s_t$ due to the near-identity path through $f_t$ when $f_t \approx 1$ (“remember this”). 

---

### 4. Worked Toy Example

Consider a **1D LSTM** (all states and weights scalars) at a single time step; suppose:

- Previous cell state: $s_{t-1} = 1.0$
    
- Previous hidden state: $h_{t-1} = 0.0$
    
- Input: $x_t = 2.0$
    

Set weights for simplicity:

- $W_i = 0.5,; U_i = 0,; b_i = 0$
    
- $W_f = 0,; U_f = 1.0,; b_f = 0$
    
- $W_o = 0,; U_o = 1.0,; b_o = 0$
    
- $W_g = 0.5,; U_g = 0,; b_g = 0$
    

**Step 1 – gates & candidate**

[  
\begin{aligned}  
i_t &= \sigma(W_i x_t + U_i h_{t-1}) = \sigma(0.5 \cdot 2.0 + 0) = \sigma(1.0) \approx 0.731 \  
f_t &= \sigma(W_f x_t + U_f h_{t-1}) = \sigma(0 + 1.0 \cdot 0) = \sigma(0) = 0.5 \  
o_t &= \sigma(W_o x_t + U_o h_{t-1}) = \sigma(0 + 1.0 \cdot 0) = 0.5 \  
g_t &= \tanh(W_g x_t + U_g h_{t-1}) = \tanh(0.5 \cdot 2.0 + 0) = \tanh(1.0) \approx 0.7616  
\end{aligned}  
]

**Step 2 – cell state update**

[  
s_t = f_t s_{t-1} + i_t g_t  
= 0.5 \cdot 1.0 + 0.731 \cdot 0.7616  
\approx 0.5 + 0.556 = 1.056  
]

**Step 3 – hidden state**

[  
h_t = o_t \tanh(s_t)  
= 0.5 \cdot \tanh(1.056)  
\approx 0.5 \cdot 0.783 = 0.392  
]

Here, the forget gate **kept half of the previous memory** (0.5), and the input gate added a reasonable chunk of new information; the output gate only exposed half of the updated cell state.

---

### 5. Connections & Prerequisites

**Prerequisite Refresher:**

- LSTMs **extend RNNs** by adding gates and a dedicated cell state $s_t$.
    
- They are explicitly designed to mitigate the **vanishing/exploding gradient** problems described earlier by giving gradients a better pathway along the cell state highway. 
    

---

## 4. Concept: RNN-based Language Modeling

### 1. High-Level Intuition

**Goal:**  
A **language model** assigns probabilities to sequences of words and predicts the **next token** given previous context.

**Analogy:**  
It’s like finishing someone’s sentence; after hearing “The cat sat on the”, your brain assigns high probability to “mat” and low probability to “spaceship”.

---

### 2. Conceptual Deep Dive

A language model models the joint probability of a sequence $w_1, \dots, w_T$ as:

$$  
P(w_1, \dots, w_T) = \prod_{t=1}^T P(w_t \mid w_1, \dots, w_{t-1})  
$$

In an **RNN language model**, we: 

1. **Tokenize** words into IDs (vocabulary of size $V$).
    
2. Map each ID to an **embedding vector** $x_t$ (via a learned embedding matrix).
    
3. Feed embeddings into an **RNN / LSTM** to produce hidden states $h_t$.
    
4. Use a **softmax head** to produce $P(w_t \mid w_{<t})$ over the vocabulary.
    

Lecture structure (simplified): 

- Start at time $t = T - N$ with context window of length $N$.
    
- For each time step:
    
    - Convert token $w_t$ to embedding $x_t$.
        
    - Update hidden state via RNN/LSTM.
        
    - Predict the **next token** $w_{t+1}$ via softmax.
        

A **character-level RNN** is just a special case where tokens are **characters instead of words**; vocabulary is small and the model learns to predict the next character. The instructor references such an example (Karpathy’s character RNN) for hands-on understanding. 

---

### 3. Mathematical Formulation

1. **Embedding lookup**
    

Given token index $w_t \in {1, \dots, V}$, embedding matrix $E \in \mathbb{R}^{d \times V}$:

$$  
x_t = E , e_{w_t}  
$$

- $e_{w_t}$: one-hot vector with 1 at position $w_t$ and 0 elsewhere.
    
- $x_t \in \mathbb{R}^d$: embedding vector.
    

2. **RNN state update** (using LSTM or simple RNN)
    

[  
h_t = \text{RNNCell}(x_t, h_{t-1})  
]

3. **Softmax head (next-token distribution)**
    

[  
\begin{aligned}  
o_t &= W_o h_t + b_o \  
P(w_{t+1} = k \mid w_{\le t}) &= \frac{\exp(o_{t,k})}{\sum_{j=1}^V \exp(o_{t,j})}  
\end{aligned}  
]

4. **Training objective**: **cross-entropy loss** over the true next tokens:
    

$$  
\mathcal{L} = - \sum_{t=1}^{T-1} \log P(w_{t+1}^{\text{(true)}} \mid w_{\le t})  
$$

---

### 4. Worked Toy Example

Very small vocabulary:

- $V = 3$: ${ \text{“I”}=1, \text{“am”}=2, \text{“happy”}=3 }$
    
- Sequence in training: “I am happy” → tokens $(1, 2, 3)$
    

Suppose at time $t=1$ (“I”) the model outputs probabilities for the next word:

- $P(\text{“I”} \mid \text{“I”}) = 0.1$
    
- $P(\text{“am”} \mid \text{“I”}) = 0.7$
    
- $P(\text{“happy”} \mid \text{“I”}) = 0.2$
    

The true next word at $t=1$ is “am”, so the contribution to loss is:

$$  
\mathcal{L}_1 = -\log 0.7  
$$

At $t=2$ (“am”), suppose the model outputs:

- $P(\text{“I”} \mid \text{“I am”}) = 0.05$
    
- $P(\text{“am”} \mid \text{“I am”}) = 0.15$
    
- $P(\text{“happy”} \mid \text{“I am”}) = 0.8$
    

True next word is “happy”, so

$$  
\mathcal{L}_2 = -\log 0.8  
$$

Total sequence loss: $\mathcal{L} = \mathcal{L}_1 + \mathcal{L}_2$.

---

### 5. Connections & Prerequisites

**Prerequisite Refresher:**

- This builds on **embeddings** (mapping tokens → vectors) and **softmax / cross-entropy** from earlier lectures. 
    
- The same RNN language model forms the **decoder** in NMT and is also at the core of **sequence-to-sequence**models.
    

---

## 5. Concept: Encoder–Decoder & Neural Machine Translation (Seq2Seq)

### 1. High-Level Intuition

**Goal:**  
Neural machine translation converts a **source sentence** in one language into a **target sentence** in another using **sequence-to-sequence models**.

**Analogy:**  
Think of one person listening to a sentence, forming a **mental summary (thought vector)**, and then another person using that summary to generate the same meaning in another language.

---

### 2. Conceptual Deep Dive

In the lecture’s NMT setup: 

- Source language tokens: $x_1, x_2, x_3$ (for simplicity).
    
- Target language tokens: $x'_1, x'_2, x'_3, \dots$
    

Architecture: **Encoder–Decoder**

1. **Encoder** (RNN/LSTM):
    
    - Processes the **source sequence** left-to-right:  
        $x_1 \rightarrow h_1,; x_2 \rightarrow h_2,; x_3 \rightarrow h_3$
        
    - Final hidden state $h_3$ (in the toy example) is the **thought vector** summarizing the entire source sentence. 
        
2. **Decoder** (RNN/LSTM language model):
    
    - Initialized with the **thought vector** as its starting hidden state.
        
    - At each step, it takes the **previous target token** (during training, the true one; during inference, the previously predicted one) and outputs a **distribution over the next target token**. 
        
    - Both encoder and decoder parameters are trained **jointly** so that better translations shape a better encoder representation. 
        

The lecturer also mentions **bidirectional encoders**, where the thought vector is built from both a forward RNN and a backward RNN; we’ll detail that later. 

---

### 3. Mathematical Formulation

Let source sentence be $x_1, \dots, x_T$ and target sentence be $y_1, \dots, y_{T'}$.

1. **Encoder**
    

$$  
\begin{aligned}  
h_t^{\text{enc}} &= \text{RNNEncCell}(x_t, h_{t-1}^{\text{enc}}), \quad t=1,\dots,T \  
c &= h_T^{\text{enc}} \quad \text{(context / thought vector)}  
\end{aligned}  
$$

2. **Decoder**
    

Initialize hidden state with $h_0^{\text{dec}} = c$. At each target step $t$:

$$  
\begin{aligned}  
h_t^{\text{dec}} &= \text{RNNDecCell}(e(y_{t-1}), h_{t-1}^{\text{dec}}) \  
o_t &= W_o h_t^{\text{dec}} + b_o \  
P(y_t = k \mid y_{<t}, x_{1:T}) &= \text{softmax}(o_t)_k  
\end{aligned}  
$$

- $e(y_{t-1})$: embedding of previous target token $y_{t-1}$ (special token `<SOS>` for $y_0$).
    
- $c$: context vector from encoder.
    

3. **Training objective**
    

Cross-entropy over all target tokens:

$$  
\mathcal{L}_{\text{NMT}} = - \sum_{t=1}^{T'} \log P(y_t^{\text{(true)}} \mid y_{<t}^{\text{(true)}}, x_{1:T})  
$$

---

### 4. Worked Toy Example

Source: “je t’aime” (French)  
Target: “I love you”

Let’s simplify to token sequences:

- Source: $x_1 = \text{je}$, $x_2 = \text{t’}$, $x_3 = \text{aime}$
    
- Target: $y_1 = \text{I}$, $y_2 = \text{love}$, $y_3 = \text{you}$
    

**Encoder**

Suppose after processing all source tokens, the encoder produces:

- $h_3^{\text{enc}} = c = [0.2, -0.5]^\top$
    

**Decoder (training time – teacher forcing)**

At $t=1$:

- Input token: `<SOS>` (start of sentence), embedding $e(y_0)$
    
- Hidden state: $h_1^{\text{dec}} = \text{RNNDecCell}(e(y_0), c)$
    
- Softmax head outputs probabilities; assume highest is for “I” → $P(y_1=\text{I}|\cdot)=0.7$
    

At $t=2$:

- Input token (training): **true** $y_1 = \text{I}$; embedding $e(\text{I})$
    
- Hidden state: $h_2^{\text{dec}} = \text{RNNDecCell}(e(\text{I}), h_1^{\text{dec}})$
    
- Softmax: highest for “love”, etc.
    

Loss is summed over $t=1,2,3$ comparing predicted distributions to true tokens (“I”, “love”, “you”).

---

### 5. Connections & Prerequisites

**Prerequisite Refresher:**

- Seq2Seq NMT **extends RNN language modeling**: the decoder is essentially a language model conditioned on a **context vector** summarizing the source phrase.
    
- This model will later be enhanced using **attention** (Transformers replace the single context vector with many contextual ones). 
    

---

## 6. Concept: Bidirectional RNNs & Contextual Representations

### 1. High-Level Intuition

**Goal:**  
Bidirectional RNNs use both **past and future context** to encode each token, helpful when word meaning depends on both what comes before and after.

**Analogy:**  
Deciding whether “bank” is a river bank or a financial bank often requires seeing both the **preceding** and **following**words.

---

### 2. Conceptual Deep Dive

A **bidirectional RNN** runs two RNNs over the same input:

- **Forward**: left → right, producing $\overrightarrow{h}_t$.
    
- **Backward**: right → left, producing $\overleftarrow{h}_t$.
    

The combined representation for token $t$ is usually their **concatenation**:

$$  
h_t^{\text{bi}} = [\overrightarrow{h}_t; \overleftarrow{h}_t]  
$$

The lecture gives examples where context from both directions is needed to disambiguate meanings (e.g., names vs. nouns). 

In NMT, a **bidirectional encoder** produces a more informative context (thought vector) for the decoder:

- You can run the encoder forward ($x_1 \to x_T$) and backward ($x_T \to x_1$), then combine their final states to form a richer $c$. 
    

---

### 3. Mathematical Formulation

Forward RNN:

$$  
\overrightarrow{h}_t = \text{RNN}^{\rightarrow}(x_t, \overrightarrow{h}_{t-1})  
$$

Backward RNN:

$$  
\overleftarrow{h}_t = \text{RNN}^{\leftarrow}(x_t, \overleftarrow{h}_{t+1})  
$$

Combined representation:

$$  
h_t^{\text{bi}} = [\overrightarrow{h}_t; \overleftarrow{h}_t]  
$$

For an encoder thought vector, a simple option:

$$  
c = [\overrightarrow{h}_T; \overleftarrow{h}_1]  
$$

---

### 4. Worked Toy Example

Sequence: “GW bridge” (ambiguous “GW” = person vs something else)

Tokens: $x_1 = \text{“GW”}$, $x_2 = \text{“bridge”}$

Forward pass:

- $\overrightarrow{h}_1$: based only on “GW”
    
- $\overrightarrow{h}_2$: based on “GW bridge”
    

Backward pass:

- $\overleftarrow{h}_2$: based only on “bridge”
    
- $\overleftarrow{h}_1$: based on “bridge GW”
    

Combined for token “GW”:

$$  
h_1^{\text{bi}} = [\overrightarrow{h}_1; \overleftarrow{h}_1]  
$$

$\overleftarrow{h}_1$ has already seen **“bridge”**, helping the model realize that “GW” here likely refers to a **person**(“George Washington Bridge”) rather than some other entity. 

---

### 5. Connections & Prerequisites

**Prerequisite Refresher:**

- This builds on the standard **RNN encoder**; we simply run another copy in reverse.
    
- Bidirectionality is especially useful for **encoder-only** tasks (e.g., BERT-style models) but cannot be used directly in **left-to-right decoders** because future tokens are unknown at inference time.
    

---

## 7. Concept: Decoding with Beam Search / MLSE (Viterbi-style)

### 1. High-Level Intuition

**Problem:**  
Greedy decoding (picking the most probable next token at each step) may produce a sequence whose **overall probability**is suboptimal.

**Solution (Beam search / MLSE):**  
Keep several **best partial sequences** at each step and choose the final sequence with the **highest total likelihood**.

**Analogy:**  
Instead of always taking the currently fastest-looking road on a trip (greedy), you keep a small set of promising routes in mind and only decide at the end which was truly fastest.

---

### 2. Conceptual Deep Dive

In NMT decoding, at each step the model outputs a distribution over next tokens. The **greedy approach** always picks the token with the highest local probability, but this can produce worse **sequence-level** probability. 

The lecturer gives a toy example where greedy decoding yields:

> “the sheep passed docked”

but a different sequence with slightly lower local choices has **higher overall likelihood**. 

Beam search (MLSE / Viterbi-style): 

- Maintains top **$K$ candidate sequences** at each time step (beam width $K$).
    
- Works with **log-probabilities** to avoid underflow.
    
- Expands each candidate with all possible next tokens, scoring new sequences by **sum of log-probs**.
    
- At the end (when EOS is reached), picks the sequence with **maximum total log-likelihood**.
    

---

### 3. Mathematical Formulation

Let a candidate sequence up to step $t$ be $y_{1:t}$. Its score is:

$$  
\text{score}(y_{1:t})  
= \log P(y_{1:t} \mid x)  
= \sum_{k=1}^t \log P(y_k \mid y_{<k}, x)  
$$

Beam search algorithm (high level):

1. Initialize beam with `<SOS>`:  
    $\mathcal{B}_0 = { ([\text{}], 0) }$
    
2. At each step $t$:
    
    - For each $(y_{1:t-1}, s)$ in $\mathcal{B}_{t-1}$, expand with all tokens $v \in \mathcal{V}$:  
        $$  
        \tilde{y}_{1:t} = [y_{1:t-1}, v],\quad  
        \tilde{s} = s + \log P(v \mid y_{<t}, x)  
        $$
        
    - Collect all expanded candidates, then keep only the **top $K$** by score to form $\mathcal{B}_t$.
        
3. Stop when all candidates generate EOS or max length; choose the candidate with highest score.
    

---

### 4. Worked Toy Example

Toy example with 2-step translation, beam width $K=2$.

At step 1, decoder outputs probabilities for 3 tokens:

- “sheep”: $0.6$
    
- “ship”: $0.4$
    
- “car”: $0.0$ (ignore)
    

Beam after step 1 (in log space, but we’ll keep raw probs here for simplicity):

- Path A: “sheep” (score $0.6$)
    
- Path B: “ship” (score $0.4$)
    

At step 2, given each first word, the model outputs:

- If first word is “sheep”:
    
    - “passed”: $0.55$
        
    - “docked”: $0.45$
        
- If first word is “ship”:
    
    - “passed”: $0.9$
        
    - “docked”: $0.1$
        

Greedy decoding:

- Step 1: pick “sheep” (0.6)
    
- Step 2: pick “passed” (0.55)
    
- Sequence: “sheep passed” with score $0.6 \times 0.55 = 0.33$
    

Beam search ($K=2$):

**All length-2 candidates:**

- “sheep passed”: $0.6 \times 0.55 = 0.33$
    
- “sheep docked”: $0.6 \times 0.45 = 0.27$
    
- “ship passed”: $0.4 \times 0.9 = 0.36$
    
- “ship docked”: $0.4 \times 0.1 = 0.04$
    

The **best sequence** is **“ship passed”** with **0.36**, even though greedy would have picked the first token “sheep”. This illustrates why **maximizing per-step probability is not equivalent to maximizing sequence-level likelihood**. 

---

### 5. Connections & Prerequisites

**Prerequisite Refresher:**

- Beam search assumes you already have a **trained sequence model** (e.g., NMT decoder) that gives $P(y_t \mid y_{<t}, x)$.
    
- It’s essentially a **search algorithm** over sequences, like Viterbi, optimized for **maximum likelihood sequence estimation (MLSE)** rather than greedy local decisions. 
    

---

## 8. Concept: BLEU – Evaluation Metric for Machine Translation

### 1. High-Level Intuition

**Goal:**  
BLEU score automatically evaluates how close a **machine translation** is to one or more **reference translations** via **n-gram overlap**.

**Analogy:**  
BLEU is like comparing short word chunks between two essays and rewarding overlapping phrasing while penalizing overly short or repetitive summaries.

---

### 2. Conceptual Deep Dive

BLEU is essentially a **precision-oriented** metric over **$n$-grams** (typically up to 4-grams): 

- Let $C$ be the **candidate translation** and $R$ the **reference**.
    
- For each $n$ from 1 to $N$ (e.g., 4), compute **modified precision** $p_n$ = fraction of candidate $n$-grams that appear in the reference, with **clipping** to avoid crediting extra repeated $n$-grams.
    
- BLEU also includes a **brevity penalty** so extremely short translations don’t unfairly score high. 
    

The lecture notes that metrics like BLEU can be **gamed** by weird repetitions (e.g., “the plane the plane the plane”), so BLEU is not perfect but widely used for benchmarking. 

---

### 3. Mathematical Formulation

Let $p_n$ be the modified precision for $n$-grams and $w_n$ be weights (usually $w_n = 1/N$). BLEU is:

$$  
\text{BLEU} = \text{BP} \cdot \exp\left( \sum_{n=1}^N w_n \log p_n \right)  
$$

- **Brevity penalty** ($\text{BP}$):
    
    $$  
    \text{BP} =  
    \begin{cases}  
    1, & \text{if } c > r \  
    \exp(1 - r/c), & \text{if } c \le r  
    \end{cases}  
    $$
    
    - $c$: length of candidate translation.
        
    - $r$: effective reference length.
        
- $p_n$: clipped $n$-gram precision (counts limited by max reference count).
    

---

### 4. Worked Toy Example (Bigram BLEU, no brevity penalty)

Reference: “the cat is on the mat”  
Candidate: “the cat is on mat”

**Unigrams:**

- Candidate: the, cat, is, on, mat (5 tokens)
    
- Overlap with reference (counted with clipping): the (1), cat (1), is (1), on (1), mat (1) → 5 matches
    
- $p_1 = 5/5 = 1.0$
    

**Bigrams:**

- Candidate: “the cat”, “cat is”, “is on”, “on mat” (4)
    
- Reference bigrams: “the cat”, “cat is”, “is on”, “on the”, “the mat”
    
- Overlap: “the cat”, “cat is”, “is on” → 3 matches
    
- $p_2 = 3/4 = 0.75$
    

With $N=2$, $w_1 = w_2 = 0.5$, and assume no brevity penalty (`BP=1` for this toy):

$$  
\text{BLEU} = 1 \cdot \exp\left(0.5 \log 1.0 + 0.5 \log 0.75\right)  
= \exp\left(0 + 0.5 \log 0.75\right)  
\approx 0.866  
$$

---

### 5. Connections & Prerequisites

**Prerequisite Refresher:**

- BLEU is computed **after** you have a translation model and decoding scheme (e.g., beam search).
    
- It’s one of several metrics; understanding it lets you interpret benchmarking tables for NMT systems and compare architectures like RNN-based seq2seq vs Transformers. 
    

---

## 9. Concept: From RNNs to Transformers and Self-Attention (Intro)

### 1. High-Level Intuition

**Goal:**  
Transformers replace **recurrent** processing with **parallel self-attention**, enabling better use of global context and more efficient training.

**Analogy:**  
Instead of reading a sentence strictly left-to-right and only passing notes forward (RNN), you let **every word look at every other word simultaneously** to decide who it cares about (attention).

---

### 2. Conceptual Deep Dive

The lecture motivates Transformers by three key shifts: 

1. **Remove recurrence**: no more $h_t$ derived from $h_{t-1}$; instead, all tokens are processed **in parallel**, making training more efficient.
    
2. **Inject positional information**: without recurrence, the model is **permutation-invariant**, so we must add **positional encodings** so the model knows whether a token appears first, last, etc. 
    
3. **Build contextual embeddings via attention**: representations of tokens change depending on **what other tokens are present** (contextual embeddings), in contrast to static Word2Vec-style embeddings. 
    

The instructor also revisits an **attention mechanism** computed over fixed embeddings (from Word2Vec); this “deterministic” attention is a stepping stone to fully trainable **self-attention** in Transformers. 

---

### 3. Mathematical Formulation (Self-Attention Block)

Given a matrix of input token representations $X \in \mathbb{R}^{T \times d}$:

1. Compute queries, keys, values:
    

$$  
Q = X W^Q, \quad K = X W^K, \quad V = X W^V  
$$

- $W^Q, W^K, W^V \in \mathbb{R}^{d \times d_k}$: learned projection matrices.
    

2. Scaled dot-product attention:
    

$$  
\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^\top}{\sqrt{d_k}}\right) V  
$$

- $QK^\top$ gives pairwise similarity between tokens.
    
- Softmax over rows normalizes attention weights to form a distribution over **which tokens to attend to**.
    
- Multiplying by $V$ gives a **contextualized representation** for each token.
    

Transformers then stack such layers with feedforward networks and add positional encodings to $X$.

---

### 4. Worked Toy Example (Very Small Attention)

Suppose we have 2 tokens (“cat”, “mat”) with 2D embeddings:

- $x_1 = [1, 0]$, $x_2 = [0, 1]$
    
- Let $W^Q = W^K = W^V = I$ (identity), $d_k = 2$, so $Q = K = V = X$.
    

Compute scores:

$$  
QK^\top =  
\begin{bmatrix}  
x_1 \cdot x_1 & x_1 \cdot x_2 \  
x_2 \cdot x_1 & x_2 \cdot x_2  
\end{bmatrix}

\begin{bmatrix}  
1 & 0 \  
0 & 1  
\end{bmatrix}  
$$

Scaled by $\sqrt{2}$:

$$  
\frac{QK^\top}{\sqrt{2}} =  
\begin{bmatrix}  
1/\sqrt{2} & 0 \  
0 & 1/\sqrt{2}  
\end{bmatrix}  
$$

Row-wise softmax: each row puts higher weight on itself:

- For token 1: weights $\approx (0.62, 0.38)$
    
- For token 2: weights $\approx (0.38, 0.62)$
    

Multiply by $V$ to get new representations slightly mixing information from the other token. With more realistic $W^Q, W^K, W^V$, these attention patterns become much richer, enabling complex dependencies without recurrence.

---

### 5. Connections & Prerequisites

**Prerequisite Refresher:**

- Transformers are the natural evolution from **RNNs + attention**: they remove recurrence entirely and rely solely on **stacked self-attention + feedforward** for sequence modeling.
    
- They still need all the earlier components: **tokenization**, **embeddings**, **language modeling objectives**, and **decoding** strategies (often still beam search). The lecture frames them as the **“final destination” architecture** for the course. 
    

---

### Key Takeaways & Formulas

- **RNN core update:**  
    $h_t = \tanh(W [x_t; h_{t-1}] + b)$ — hidden state summarizes past context.
    
- **Gradient issues:**  
    Repeated multiplication by the recurrent weight matrix causes **vanishing gradients** if $|\lambda_{\max}(W')| < 1$ and **exploding gradients** if $|\lambda_{\max}(W')| > 1$.
    
- **LSTM cell:**  
    $s_t = f_t \odot s_{t-1} + i_t \odot g_t$,  
    $h_t = o_t \odot \tanh(s_t)$ — gates control what to remember, forget, and expose.
    
- **Language modeling factorization:**  
    $P(w_1, \dots, w_T) = \prod_{t=1}^T P(w_t \mid w_{<t})$ — predicted via RNN/LSTM + softmax.
    
- **Seq2Seq NMT:**  
    Encoder produces a **context vector** $c$; the decoder is a **conditional language model** $P(y_t \mid y_{<t}, c)$ trained with cross-entropy.
    
- **Beam search (MLSE):**  
    Work with **log-probs**, keep top-$K$ candidate sequences by cumulative log-likelihood, and pick the sequence with highest final score.
    
- **BLEU metric:**  
    $\text{BLEU} = \text{BP} \cdot \exp\left(\sum_{n=1}^N w_n \log p_n\right)$ — evaluates translation quality via clipped $n$-gram precision and brevity penalty.
    
- **Self-attention in Transformers:**  
    $\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^\top}{\sqrt{d_k}}\right)V$ — each token builds a contextual embedding by attending to all others in parallel.
    