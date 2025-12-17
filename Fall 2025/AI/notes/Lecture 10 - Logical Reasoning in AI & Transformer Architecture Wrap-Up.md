### Executive Summary

This lecture serves two purposes: (1) completing the discussion of neural architectures by covering **Mixture of Experts (MOE)** and **Vision Transformers (ViT)**, and (2) transitioning to **symbolic AI** through an introduction to **propositional logic** and **logical reasoning**. The logical reasoning portion introduces the syntax and semantics of propositional logic, knowledge base operations (ASK and TELL), the concepts of entailment, contradiction, and contingency, and concludes with a practical demonstration using the **Wumpus World** game and the **model checking** inference algorithm. These topics form the foundation for understanding how AI agents can reason about their environment using symbolic representations.

---

## Part I: Neural Architecture Wrap-Up

---

## 1. Concept: Transformer Query-Key-Value Projections (Review)

### 1.1 High-Level Intuition

**Problem Solved:** How can we transform context-free word embeddings into representations that capture the meaning of words _in context_?

**Analogy:** Think of entering a professors' cafeteria seeking help with a math problem. You raise your hand (issue a **query**), and professors whose expertise matches your need (**keys**) come to assist you. After receiving help, you leave with enhanced knowledge (**value** transformation). The Q, K, V projections work similarly—they transform tokens so that relevant information can be identified and aggregated.

### 1.2 Conceptual Deep Dive

In a Transformer, input embeddings $X$ are **context-free**—each token's representation is independent of surrounding tokens. The goal of the attention mechanism is to create **contextual representations** where each token's meaning is informed by relevant context.

The three learned weight matrices serve distinct purposes:

- **$W_Q$ (Query matrix):** Projects tokens into a "question" space—"What information do I need?"
- **$W_K$ (Key matrix):** Projects tokens into an "identity" space—"What information can I provide?"
- **$W_V$ (Value matrix):** Projects tokens into a "content" space—"What information will I contribute?"

The projection through these matrices effectively changes the **coordinate system** of the representation space. Just as PCA finds axes that best represent variance in data, these learned projections find axes that best capture semantic relationships (e.g., "object-ness," "verb-ness," grammatical patterns).

### 1.3 Mathematical Formulation

The projections are computed as:

$$ Q = XW_Q, \quad K = XW_K, \quad V = XW_V $$

Where:

- $X \in \mathbb{R}^{n \times d}$ is the input matrix ($n$ tokens, $d$ dimensions)
- $W_Q, W_K, W_V \in \mathbb{R}^{d \times d_k}$ are learned weight matrices
- $Q, K, V$ are the query, key, and value matrices respectively

The attention output is:

$$ \text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V $$

Where:

- $QK^T$ computes similarity scores between all query-key pairs
- $\sqrt{d_k}$ is a scaling factor to prevent vanishing gradients
- The softmax produces attention weights summing to 1
- Multiplication by $V$ aggregates values weighted by attention

### 1.4 Worked Toy Example

Consider 3 tokens with $d = 4$ dimensional embeddings, projecting to $d_k = 2$:

**Input:** $$ X = \begin{bmatrix} 1 & 0 & 1 & 0 \\ 0 & 1 & 0 & 1 \\ 1 & 1 & 0 & 0 \end{bmatrix} $$

**Weight matrices (simplified):** $$ W_Q = \begin{bmatrix} 1 & 0 \\ 0 & 1 \\ 1 & 0 \\ 0 & 1 \end{bmatrix}, \quad W_K = \begin{bmatrix} 0 & 1 \\ 1 & 0 \\ 0 & 1 \\ 1 & 0 \end{bmatrix} $$

**Step 1:** Compute $Q = XW_Q$: $$ Q = \begin{bmatrix} 1 & 0 & 1 & 0 \\ 0 & 1 & 0 & 1 \\ 1 & 1 & 0 & 0 \end{bmatrix} \begin{bmatrix} 1 & 0 \\ 0 & 1 \\ 1 & 0 \\ 0 & 1 \end{bmatrix} = \begin{bmatrix} 2 & 0 \\ 0 & 2 \\ 1 & 1 \end{bmatrix} $$

**Step 2:** Compute $K = XW_K$: $$ K = \begin{bmatrix} 0 & 2 \\ 2 & 0 \\ 1 & 1 \end{bmatrix} $$

**Step 3:** Compute attention scores $QK^T$: $$ QK^T = \begin{bmatrix} 2 & 0 \\ 0 & 2 \\ 1 & 1 \end{bmatrix} \begin{bmatrix} 0 & 2 & 1 \\ 2 & 0 & 1 \end{bmatrix} = \begin{bmatrix} 0 & 4 & 2 \\ 4 & 0 & 2 \\ 2 & 2 & 2 \end{bmatrix} $$

Token 1's query strongly attends to Token 2's key (score = 4), indicating these tokens should share information.

### 1.5 Connections & Prerequisites

**Prerequisite Refresher (Word Embeddings):** Recall that techniques like Word2Vec create embeddings where semantically similar words cluster together, and analogies manifest as vector arithmetic (e.g., $\vec{king} - \vec{man} + \vec{woman} \approx \vec{queen}$). Transformer projections extend this by learning task-specific coordinate transformations.

---

## 2. Concept: Multi-Head Attention

### 2.1 High-Level Intuition

**Problem Solved:** A single attention mechanism can only focus on one type of relationship at a time. How can we capture multiple types of patterns simultaneously?

**Analogy:** When reading code, you simultaneously track: (1) syntax patterns (keywords, brackets), (2) semantic patterns (variable meanings), and (3) structural patterns (function calls, loops). Each attention "head" specializes in detecting one type of pattern.

### 2.2 Conceptual Deep Dive

**Multi-head attention** runs several attention mechanisms in parallel, each with its own learned $W_Q$, $W_K$, $W_V$ matrices. This allows the model to:

- Attend to information from different **representation subspaces**
- Capture different types of relationships (grammatical, semantic, positional)
- Specialize heads for different linguistic or visual patterns

In NLP, one head might track subject-verb agreement while another tracks coreference. In vision (as we'll see), heads might specialize in color, texture, or shape patterns.

To maintain computational efficiency, each head operates on a **reduced dimension** ($d_k = d/h$ where $h$ is the number of heads), keeping the total computation similar to single-head attention.

### 2.3 Mathematical Formulation

$$ \text{MultiHead}(Q, K, V) = \text{Concat}(\text{head}_1, \ldots, \text{head}_h)W^O $$

Where each head is:

$$ \text{head}_i = \text{Attention}(QW_i^Q, KW_i^K, VW_i^V) $$

Where:

- $h$ = number of attention heads
- $W_i^Q, W_i^K, W_i^V \in \mathbb{R}^{d \times d_k}$ are head-specific projection matrices
- $W^O \in \mathbb{R}^{hd_v \times d}$ is the output projection matrix
- $d_k = d_v = d/h$ (dimension per head)

### 2.4 Worked Toy Example

With $d = 4$ and $h = 2$ heads, each head operates on $d_k = 2$ dimensions.

**Head 1** might learn to detect: "Is this token a noun?" **Head 2** might learn to detect: "Is this token related to the subject?"

Each head produces a 2-dimensional output. Concatenating gives a 4-dimensional vector, which $W^O$ projects back to the model dimension.

If Head 1 outputs $[0.8, 0.2]$ (high "noun-ness") and Head 2 outputs $[0.1, 0.9]$ (high "subject-relation"), the concatenated $[0.8, 0.2, 0.1, 0.9]$ captures both patterns simultaneously.

### 2.5 Connections & Prerequisites

**Prerequisite Refresher (CNN Filters):** Recall that in CNNs, multiple convolutional filters detect different spatial patterns (edges, textures, shapes). Multi-head attention is analogous—each head is like a filter that detects different sequential or relational patterns.

---

## 3. Concept: Mixture of Experts (MOE)

### 3.1 High-Level Intuition

**Problem Solved:** How can we scale model capacity without proportionally increasing computation? And what are the theoretical limits of combining multiple predictors?

**Analogy:** Instead of one generalist doctor examining every patient, a hospital routes patients to specialists based on symptoms. The **router** decides which **expert** handles each case, allowing specialized knowledge without every doctor seeing every patient.

### 3.2 Conceptual Deep Dive

**Mixture of Experts** replaces the dense feed-forward layers in a Transformer with multiple "expert" sub-networks. A **routing mechanism** directs each token to a subset of experts (sparse routing) or combines all experts (dense routing).

The theoretical analysis of ensemble methods provides insight into MOE behavior. Consider $K$ predictors combined as:

$$ F(x) = \frac{1}{K}\sum_{i=1}^{K}F_i(x) $$

The ensemble's error depends critically on whether the individual predictors make **correlated** or **uncorrelated** mistakes:

- **Worst case (perfect correlation):** All experts make identical errors → no improvement over a single expert
- **Best case (zero correlation):** Experts make independent errors → error decreases linearly with $K$

### 3.3 Mathematical Formulation

Let $\epsilon_i(x) = F_i(x) - Y(x)$ be the error of the $i$-th predictor.

The **mean squared error of the ensemble** is:

$$ \text{MSE}_{\text{ensemble}} = \mathbb{E}\left[\left(\frac{1}{K}\sum_i \epsilon_i(x)\right)^2\right] $$

After expansion, this equals:

$$ \text{MSE}_{\text{ensemble}} = \frac{1}{K}\bar{V} + \left(1 - \frac{1}{K}\right)\bar{C} $$

Where:

- $\bar{V} = \mathbb{E}[\epsilon_i^2]$ is the **average variance** of individual predictor errors
- $\bar{C} = \mathbb{E}[\epsilon_i \epsilon_j]$ for $i \neq j$ is the **average covariance** between errors

**Best Case** ($\bar{C} = 0$, uncorrelated errors): $$ \text{MSE}_{\text{ensemble}} = \frac{\bar{V}}{K} $$ Error decreases linearly with the number of experts.

**Worst Case** ($\bar{C} = \bar{V}$, perfectly correlated errors): $$ \text{MSE}_{\text{ensemble}} = \bar{V} $$ No improvement—equivalent to a single predictor.

### 3.4 Worked Toy Example

Suppose we have $K = 4$ experts, each with average error variance $\bar{V} = 0.16$.

**Scenario A (Uncorrelated errors, $\bar{C} = 0$):** $$ \text{MSE} = \frac{0.16}{4} = 0.04 $$ A 4× reduction in error!

**Scenario B (Partially correlated, $\bar{C} = 0.08$):** $$ \text{MSE} = \frac{1}{4}(0.16) + \frac{3}{4}(0.08) = 0.04 + 0.06 = 0.10 $$ Only a 1.6× improvement.

**Scenario C (Perfectly correlated, $\bar{C} = 0.16$):** $$ \text{MSE} = \frac{1}{4}(0.16) + \frac{3}{4}(0.16) = 0.16 $$ No improvement at all.

### 3.5 Connections & Prerequisites

**Prerequisite Refresher (Ensemble Methods):** Recall that ensemble methods like bagging and boosting combine weak learners. The key insight is that diversity among predictors is crucial—if all models make the same mistakes, combining them provides no benefit.

**Reference:** For detailed block diagrams and routing equations, see the **DeepSeek MOE** paper.

---

## 4. Concept: Vision Transformers (ViT)

### 4.1 High-Level Intuition

**Problem Solved:** How can we apply the powerful Transformer architecture, designed for sequential data like text, to images?

**Analogy:** Just as we break a sentence into word tokens, we break an image into **patch tokens**. Each patch becomes a "word" that the Transformer can process, allowing the same attention mechanism to find relationships between image regions.

### 4.2 Conceptual Deep Dive

**Vision Transformers (ViT)** adapt the Transformer architecture for images through a simple but effective tokenization scheme:

1. **Patch Extraction:** Divide the image into fixed-size patches (typically 16×16 pixels)
2. **Linear Embedding:** Flatten each patch and project it to the model dimension
3. **Position Embeddings:** Add positional information (interestingly, whether patches are arranged in sequence or with 2D position encoding yields similar performance)
4. **Standard Transformer:** Apply the same attention mechanism as in NLP

The attention mechanism in ViT is highly interpretable. When visualizing attention maps:

- **Queries** represent "what feature am I looking for?" (e.g., a specific color)
- **Keys** represent "what features does this patch contain?"
- **Attention weights** show which patches are relevant to each query

For example, a query for "blue color" will show high attention weights on all blue patches in the image, regardless of their spatial location.

### 4.3 Mathematical Formulation

Given an image $I \in \mathbb{R}^{H \times W \times C}$:

**Step 1: Patch Creation** Split into $N = \frac{HW}{P^2}$ patches, where $P$ is patch size: $$ x_p \in \mathbb{R}^{N \times (P^2 \cdot C)} $$

**Step 2: Linear Projection** $$ z_0 = [x_{\text{class}}; x_p^1 E; x_p^2 E; \ldots; x_p^N E] + E_{\text{pos}} $$

Where:

- $E \in \mathbb{R}^{(P^2 \cdot C) \times D}$ is the patch embedding matrix
- $x_{\text{class}}$ is a learnable classification token
- $E_{\text{pos}} \in \mathbb{R}^{(N+1) \times D}$ are positional embeddings

**Step 3: Transformer Encoding** $$ z_\ell = \text{TransformerBlock}(z_{\ell-1}), \quad \ell = 1, \ldots, L $$

### 4.4 Worked Toy Example

Consider a 32×32 RGB image with patch size $P = 16$:

**Step 1:** Number of patches: $$ N = \frac{32 \times 32}{16 \times 16} = \frac{1024}{256} = 4 \text{ patches} $$

**Step 2:** Each patch has dimension: $$ 16 \times 16 \times 3 = 768 \text{ values (flattened)} $$

**Step 3:** With embedding dimension $D = 512$, the projection matrix $E$ is: $$ E \in \mathbb{R}^{768 \times 512} $$

**Step 4:** Input to Transformer (including class token): $$ z_0 \in \mathbb{R}^{5 \times 512} $$

The 5 tokens (1 class + 4 patches) attend to each other, learning spatial relationships.

### 4.5 Connections & Prerequisites

**Prerequisite Refresher (Positional Embeddings):** Recall that Transformers have no inherent notion of order. Positional embeddings encode position using sinusoidal functions: $$ PE_{(pos, 2i)} = \sin(pos / 10000^{2i/d}) $$ $$ PE_{(pos, 2i+1)} = \cos(pos / 10000^{2i/d}) $$ These are added to token embeddings, allowing the model to distinguish positions.

**Reference:** Chapter 26 of "Foundations of Computer Vision" (MIT, free online) provides detailed ViT coverage.

---

## Part II: Logical Reasoning

---

## 5. Concept: Propositional Logic Syntax

### 5.1 High-Level Intuition

**Problem Solved:** How can we formally represent knowledge about the world in a way that supports automated reasoning?

**Analogy:** Just as programming languages have syntax rules for valid code, propositional logic has syntax rules for valid logical sentences. Learning the syntax is like learning the grammar of a new language for expressing facts and rules.

### 5.2 Conceptual Deep Dive

**Propositional logic** is a formal language consisting of:

**Propositional Symbols:** Variables that can be either TRUE or FALSE. These represent atomic facts about the world:

- $P$ might represent "It is raining"
- $Q$ might represent "The ground is wet"
- $P_{1,2}$ might represent "There is a pit at location (1,2)"

**Logical Operators (Connectives):** Ways to combine propositions:

|Operator|Symbol|Name|Meaning|
|---|---|---|---|
|$\neg$|NOT|Negation|"It is not the case that..."|
|$\land$|AND|Conjunction|"Both ... and ..."|
|$\lor$|OR|Disjunction|"Either ... or ... (or both)"|
|$\Rightarrow$|IF-THEN|Implication|"If ... then ..."|
|$\Leftrightarrow$|IFF|Biconditional|"... if and only if ..."|

**Well-Formed Formulas:** Valid sentences are built recursively:

- Any propositional symbol is a sentence
- If $\alpha$ is a sentence, then $\neg\alpha$ is a sentence
- If $\alpha$ and $\beta$ are sentences, then $(\alpha \land \beta)$, $(\alpha \lor \beta)$, $(\alpha \Rightarrow \beta)$, and $(\alpha \Leftrightarrow \beta)$ are sentences

### 5.3 Mathematical Formulation

**Syntax Grammar (BNF notation):**

$$ \text{Sentence} \rightarrow \text{AtomicSentence} \mid \text{ComplexSentence} $$ $$ \text{AtomicSentence} \rightarrow \text{True} \mid \text{False} \mid P \mid Q \mid \ldots $$ $$ \text{ComplexSentence} \rightarrow \neg\text{Sentence} \mid (\text{Sentence} \land \text{Sentence}) \mid (\text{Sentence} \lor \text{Sentence}) \mid \ldots $$

**Operator Precedence (highest to lowest):**

1. $\neg$ (negation)
2. $\land$ (conjunction)
3. $\lor$ (disjunction)
4. $\Rightarrow$ (implication)
5. $\Leftrightarrow$ (biconditional)

### 5.4 Worked Toy Example

**English:** "If it is raining and I don't have an umbrella, then I will get wet."

**Symbols:**

- $R$ = "It is raining"
- $U$ = "I have an umbrella"
- $W$ = "I will get wet"

**Logical Sentence:** $$ (R \land \neg U) \Rightarrow W $$

**Parsing the structure:**

1. $\neg U$ (negation of U)
2. $R \land \neg U$ (conjunction)
3. $(R \land \neg U) \Rightarrow W$ (implication)

### 5.5 Connections & Prerequisites

This is a foundational concept with no prerequisites. It forms the basis for all subsequent logical reasoning topics.

---

## 6. Concept: Truth Tables and the Implication Operator

### 6.1 High-Level Intuition

**Problem Solved:** How do we determine the truth value of complex sentences given the truth values of their components?

**Analogy:** A truth table is like a complete specification sheet—it lists every possible input combination and the corresponding output. For logic, inputs are truth values of propositions, and output is the truth value of the compound sentence.

### 6.2 Conceptual Deep Dive

**Truth tables** exhaustively enumerate all possible truth value assignments and the resulting sentence values. The most counterintuitive operator is **implication** ($\Rightarrow$).

**Implication Truth Table:**

|$P$|$Q$|$P \Rightarrow Q$|
|---|---|---|
|T|T|T|
|T|F|F|
|F|T|T|
|F|F|T|

The key insight: **implication is only false when the antecedent is true but the consequent is false.** This is called "material implication."

**Intuitive Examples:**

- "If it is raining, then it is cloudy" ($R \Rightarrow C$)
    - Raining and cloudy (T, T) → True ✓
    - Raining but not cloudy (T, F) → False (violated!)
    - Not raining but cloudy (F, T) → True (no claim made about non-rainy days)
    - Not raining and not cloudy (F, F) → True (no claim violated)

**Important Equivalence:** $$ P \Rightarrow Q \equiv \neg P \lor Q $$

This equivalence helps build intuition: "If P then Q" means "Either P is false, or Q is true."

### 6.3 Mathematical Formulation

**Complete Truth Tables for All Operators:**

|$P$|$Q$|$\neg P$|$P \land Q$|$P \lor Q$|$P \Rightarrow Q$|$P \Leftrightarrow Q$|
|---|---|---|---|---|---|---|
|T|T|F|T|T|T|T|
|T|F|F|F|T|F|F|
|F|T|T|F|T|T|F|
|F|F|T|F|F|T|T|

**Biconditional Equivalence:** $$ P \Leftrightarrow Q \equiv (P \Rightarrow Q) \land (Q \Rightarrow P) $$

### 6.4 Worked Toy Example

**Evaluate:** $(P \land \neg Q) \Rightarrow R$ when $P = T$, $Q = T$, $R = F$

**Step 1:** Evaluate $\neg Q$: $$ \neg Q = \neg T = F $$

**Step 2:** Evaluate $P \land \neg Q$: $$ P \land \neg Q = T \land F = F $$

**Step 3:** Evaluate $(P \land \neg Q) \Rightarrow R$: $$ F \Rightarrow F = T $$

The sentence is TRUE because the antecedent is false (and false implies anything).

### 6.5 Connections & Prerequisites

**Prerequisite Refresher (Propositional Syntax):** This concept requires understanding propositional symbols and operators from Concept 5. Each operator has a fixed truth table that defines its semantics.

---

## 7. Concept: Models and Satisfaction

### 7.1 High-Level Intuition

**Problem Solved:** What does it mean for a logical sentence to be "true"? How do we connect abstract symbols to concrete meanings?

**Analogy:** A model is like a possible state of the world. If you're describing a chess game, a model specifies where every piece is on the board. A sentence is "satisfied" by a model if the sentence accurately describes that board state.

### 7.2 Conceptual Deep Dive

A **model** is a complete assignment of truth values to all propositional symbols. Given $n$ symbols, there are $2^n$ possible models.

A model **satisfies** a sentence if the sentence evaluates to TRUE under that assignment. We write:

$$ M \vDash \alpha $$

Read as: "Model $M$ satisfies sentence $\alpha$" or "$\alpha$ is true in $M$."

**Key Definitions:**

- **$M(\alpha)$:** The set of all models that satisfy sentence $\alpha$
- **$M(KB)$:** The set of all models that satisfy the knowledge base (i.e., satisfy ALL sentences in the KB)

For a knowledge base with multiple sentences: $$ M(KB) = M(\alpha_1) \cap M(\alpha_2) \cap \ldots \cap M(\alpha_n) $$

**Critical Insight:** Adding more sentences to a KB can only **decrease** (or maintain) the set of satisfying models. More constraints = fewer valid worlds.

### 7.3 Mathematical Formulation

**Definition of Satisfaction:** A model $M$ satisfies sentence $\alpha$ (written $M \vDash \alpha$) iff $\alpha$ evaluates to TRUE when all symbols are assigned their values from $M$.

**For compound sentences:**

- $M \vDash \neg\alpha$ iff $M \nvDash \alpha$
- $M \vDash \alpha \land \beta$ iff $M \vDash \alpha$ and $M \vDash \beta$
- $M \vDash \alpha \lor \beta$ iff $M \vDash \alpha$ or $M \vDash \beta$
- $M \vDash \alpha \Rightarrow \beta$ iff $M \nvDash \alpha$ or $M \vDash \beta$

**Counting Models:** With $n$ propositional symbols, there are $2^n$ possible models (each symbol can be T or F independently).

### 7.4 Worked Toy Example

**Symbols:** ${R, W}$ (Rain, Wet)

**All possible models:**

|Model|$R$|$W$|
|---|---|---|
|$M_1$|T|T|
|$M_2$|T|F|
|$M_3$|F|T|
|$M_4$|F|F|

**Sentence:** $\alpha = R \Rightarrow W$ ("If rain, then wet")

**Which models satisfy $\alpha$?**

- $M_1$: $T \Rightarrow T = T$ ✓
- $M_2$: $T \Rightarrow F = F$ ✗
- $M_3$: $F \Rightarrow T = T$ ✓
- $M_4$: $F \Rightarrow F = T$ ✓

**Result:** $M(\alpha) = {M_1, M_3, M_4}$

### 7.5 Connections & Prerequisites

**Prerequisite Refresher (Truth Tables):** Models assign truth values to symbols; truth tables tell us how to evaluate compound sentences given those assignments. Together, they allow us to determine which models satisfy which sentences.

---

## 8. Concept: Knowledge Base (KB)

### 8.1 High-Level Intuition

**Problem Solved:** How can an AI agent store and organize everything it knows about the world in a way that supports reasoning?

**Analogy:** A knowledge base is like a detective's case file—it contains all known facts (evidence), rules (physical laws, game rules), and inferences (deductions). When new information arrives, the detective checks if it's consistent with the file and updates accordingly.

### 8.2 Conceptual Deep Dive

A **Knowledge Base (KB)** is a set of logical sentences representing:

1. **Rules of the World:** Background knowledge about how the domain works
    
    - "If there's a pit, adjacent cells have breeze"
    - "The agent starts in a safe cell"
2. **Percepts:** Observations made by the agent
    
    - "There is no breeze at (1,1)"
    - "There is a stench at (1,2)"
3. **Inferences:** Conclusions derived from rules and percepts
    
    - "There is no pit at (2,1)"
    - "The monster is at (1,3)"

**Key Operations:**

|Operation|Name|Description|
|---|---|---|
|TELL(KB, $\alpha$)|Tell|Add sentence $\alpha$ to the knowledge base|
|ASK(KB, $\alpha$)|Ask|Query whether $\alpha$ is true, false, or unknown|

**The Constraint Perspective:** Think of each sentence as a **constraint** on possible worlds. With zero sentences, all $2^n$ models are possible. Each added sentence eliminates some models. The satisfying models for the KB are those that survive all constraints.

### 8.3 Mathematical Formulation

**Knowledge Base Definition:** $$ KB = {\alpha_1, \alpha_2, \ldots, \alpha_m} $$

**Models of KB:** $$ M(KB) = \bigcap_{i=1}^{m} M(\alpha_i) $$

A model satisfies the KB iff it satisfies every sentence: $$ M \vDash KB \iff \forall \alpha_i \in KB: M \vDash \alpha_i $$

### 8.4 Worked Toy Example

**Building a KB for a simple scenario:**

**Symbols:**

- $B_{11}$: Breeze at (1,1)
- $P_{12}$: Pit at (1,2)
- $P_{21}$: Pit at (2,1)

**Initial Rules:**

- $R_1$: $\neg P_{11}$ (No pit at start)
- $R_2$: $B_{11} \Leftrightarrow (P_{12} \lor P_{21})$ (Breeze iff adjacent pit)

**Percept Added:**

- $R_3$: $\neg B_{11}$ (No breeze sensed at (1,1))

**Inference Process:** From $R_2$ and $R_3$:

- $\neg B_{11}$ means $\neg(P_{12} \lor P_{21})$
- By De Morgan's law: $\neg P_{12} \land \neg P_{21}$
- Therefore: No pit at (1,2) AND no pit at (2,1)

### 8.5 Connections & Prerequisites

**Prerequisite Refresher (Models and Satisfaction):** The KB is satisfied by models where ALL its sentences are true. Understanding that adding sentences shrinks the set of satisfying models is crucial for understanding entailment.

---

## 9. Concept: Entailment

### 9.1 High-Level Intuition

**Problem Solved:** How do we formally define what it means for one piece of knowledge to logically follow from another?

**Analogy:** Entailment is like mathematical proof—if the premises are true, the conclusion MUST be true. It's not about probability or likelihood; it's about logical necessity. If your knowledge base entails $\alpha$, then $\alpha$ is guaranteed true in every possible world consistent with your knowledge.

### 9.2 Conceptual Deep Dive

**Entailment** ($\vDash$) is the fundamental concept of logical consequence:

$$ KB \vDash \alpha $$

Read as: "KB entails $\alpha$" or "$\alpha$ is a logical consequence of KB"

**Definition:** KB entails $\alpha$ if and only if **every model that satisfies KB also satisfies $\alpha$**.

**Venn Diagram Interpretation:**

```
[Imagine: Large rectangle = All possible models]
[        Oval inside = M(KB) = Models satisfying KB]
[        Larger oval containing M(KB) = M(α)]
```

If $M(KB) \subseteq M(\alpha)$, then $KB \vDash \alpha$.

**Intuition:** The models where KB is true form a subset of the models where $\alpha$ is true. You can't have a situation where KB is true but $\alpha$ is false.

**Set-Theoretic Definition:** $$ KB \vDash \alpha \iff M(KB) \subseteq M(\alpha) $$

Equivalently: $$ KB \vDash \alpha \iff M(KB) \cap M(\alpha) = M(KB) $$

Adding $\alpha$ to the KB doesn't eliminate any models—they were already eliminated by KB.

### 9.3 Mathematical Formulation

**Formal Definition:** $$ KB \vDash \alpha \iff \forall M: (M \vDash KB) \Rightarrow (M \vDash \alpha) $$

**TELL Operation Response for Entailment:**

When TELL(KB, $\alpha$) is called:

- If $KB \vDash \alpha$: Response is "I already knew that" (entailment)
- The sentence is **not** added as new information
- $M(KB)$ remains unchanged

**Equivalent Characterizations:**

1. Every model of KB is a model of $\alpha$
2. $M(KB) \subseteq M(\alpha)$
3. $M(KB) \cap M(\neg\alpha) = \emptyset$
4. $KB \land \neg\alpha$ is unsatisfiable

### 9.4 Worked Toy Example

**KB Contains:**

- $R_1$: $A$ ("It is morning")
- $R_2$: $A \Rightarrow B$ ("If morning, then sun rises")

**Query:** Does $KB \vDash B$?

**All models with 2 symbols:**

|Model|$A$|$B$|$R_1$|$R_2$|KB|$B$|
|---|---|---|---|---|---|---|
|$M_1$|T|T|T|T|T|T|
|$M_2$|T|F|T|F|F|F|
|$M_3$|F|T|F|T|F|T|
|$M_4$|F|F|F|T|F|F|

**Models satisfying KB:** Only $M_1$ (both $R_1$ and $R_2$ true)

**Check:** Is $B$ true in all models where KB is true?

- In $M_1$: $B = T$ ✓

**Result:** $KB \vDash B$ (Yes, KB entails B)

### 9.5 Connections & Prerequisites

**Prerequisite Refresher (Knowledge Base):** Recall that $M(KB)$ is the intersection of models satisfying all KB sentences. Entailment asks whether this intersection is contained within $M(\alpha)$—i.e., does knowing KB guarantee knowing $\alpha$?

---

## 10. Concept: Contingency and Contradiction

### 10.1 High-Level Intuition

**Problem Solved:** What happens when we try to add information that the knowledge base doesn't already know (contingency) or that conflicts with existing knowledge (contradiction)?

**Analogy:**

- **Contingency:** Learning something genuinely new—like a detective finding new evidence that narrows down the suspects.
- **Contradiction:** Receiving "evidence" that conflicts with established facts—like a witness claiming the suspect was in two places at once.

### 10.2 Conceptual Deep Dive

When TELL(KB, $\alpha$) is called, there are three possible responses:

**1. Entailment:** "I already knew that"

- $M(KB) \subseteq M(\alpha)$
- Adding $\alpha$ doesn't change $M(KB)$

**2. Contingency:** "I didn't know that—updating"

- $M(KB) \cap M(\alpha) \subset M(KB)$ (strict subset)
- AND $M(KB) \cap M(\alpha) \neq \emptyset$ (some models survive)
- Adding $\alpha$ **shrinks** the set of possible models
- This is the "learning" case—new information eliminates possibilities

**3. Contradiction:** "That conflicts with what I know"

- $M(KB) \cap M(\alpha) = \emptyset$
- No model satisfies both KB and $\alpha$
- Adding $\alpha$ would make the KB **unsatisfiable**

**Venn Diagram Visualization:**

```
ENTAILMENT:          CONTINGENCY:         CONTRADICTION:
    ┌─────────┐         ┌─────────┐          ┌─────────┐
    │   M(α)  │         │         │          │         │
    │  ┌───┐  │         │ ┌───┐   │          │ ┌───┐   │ ┌───┐
    │  │KB │  │         │ │KB ├───┼──M(α)    │ │KB │   │ │M(α)│
    │  └───┘  │         │ └───┘   │          │ └───┘   │ └───┘
    └─────────┘         └─────────┘          └─────────┘
   KB ⊆ M(α)          Partial overlap        No overlap
```

### 10.3 Mathematical Formulation

**Contingency Condition:** $$ \emptyset \subset M(KB) \cap M(\alpha) \subset M(KB) $$

Both conditions must hold:

1. $M(KB) \cap M(\alpha) \neq \emptyset$ (not a contradiction)
2. $M(KB) \cap M(\alpha) \neq M(KB)$ (not entailment—$\alpha$ is genuinely new)

**Contradiction Condition:** $$ M(KB) \cap M(\alpha) = \emptyset $$

Equivalently: $KB \vDash \neg\alpha$ (KB entails the negation of $\alpha$)

**After Contingent Update:** $$ M(KB_{new}) = M(KB) \cap M(\alpha) $$

The new KB has fewer satisfying models.

### 10.4 Worked Toy Example

**KB Contains:** $R_1$: $P \lor Q$ ("Either P or Q is true")

**Models satisfying KB:**

|Model|$P$|$Q$|$P \lor Q$|
|---|---|---|---|
|$M_1$|T|T|T|
|$M_2$|T|F|T|
|$M_3$|F|T|T|
|$M_4$|F|F|F|

$M(KB) = {M_1, M_2, M_3}$

**Case 1: TELL(KB, $P$)** — Contingency

- $M(P) = {M_1, M_2}$
- $M(KB) \cap M(P) = {M_1, M_2}$
- This is a strict subset of $M(KB)$
- **Response:** "I didn't know that—updating"
- New models: ${M_1, M_2}$

**Case 2: TELL(KB, $P \lor Q \lor R$)** — Entailment

- Any model satisfying $P \lor Q$ also satisfies $P \lor Q \lor R$
- $M(KB) \subseteq M(P \lor Q \lor R)$
- **Response:** "I already knew that"

**Case 3: TELL(KB, $\neg P \land \neg Q$)** — Contradiction

- $M(\neg P \land \neg Q) = {M_4}$
- $M(KB) \cap {M_4} = \emptyset$
- **Response:** "That conflicts with what I know"

### 10.5 Connections & Prerequisites

**Prerequisite Refresher (Entailment):** Contingency and contradiction are the "not entailment" cases. Understanding entailment as $M(KB) \subseteq M(\alpha)$ helps see that contingency means partial overlap, and contradiction means no overlap.

---

## 11. Concept: ASK Operation

### 11.1 High-Level Intuition

**Problem Solved:** How does an agent query its knowledge base to determine what it knows about a specific proposition?

**Analogy:** Asking the KB is like asking a wise oracle who only speaks truth. The oracle says "yes" if the fact is guaranteed by the knowledge, "no" if the knowledge rules it out, and "I don't know" if the knowledge is insufficient to decide.

### 11.2 Conceptual Deep Dive

The **ASK** operation queries whether a sentence $\alpha$ is:

- **TRUE:** Entailed by the KB ($KB \vDash \alpha$)
- **FALSE:** The negation is entailed ($KB \vDash \neg\alpha$)
- **UNKNOWN:** Neither is entailed (the KB has insufficient information)

**Response Conditions:**

|Response|Condition|Meaning|
|---|---|---|
|TRUE|$KB \vDash \alpha$|$\alpha$ is definitely true given KB|
|FALSE|$KB \vDash \neg\alpha$|$\alpha$ is definitely false given KB|
|UNKNOWN|$KB \nvDash \alpha$ and $KB \nvDash \neg\alpha$|KB doesn't determine $\alpha$'s truth|

**The "Unknown" Case:** This occurs when some models satisfying KB have $\alpha$ true, and others have $\alpha$ false. The KB is compatible with both possibilities.

### 11.3 Mathematical Formulation

**ASK(KB, $\alpha$) Returns:**

**TRUE** if: $$ M(KB) \subseteq M(\alpha) $$

**FALSE** if: $$ M(KB) \subseteq M(\neg\alpha) $$

Equivalently: $M(KB) \cap M(\alpha) = \emptyset$

**UNKNOWN** if: $$ M(KB) \cap M(\alpha) \neq \emptyset \text{ AND } M(KB) \cap M(\neg\alpha) \neq \emptyset $$

### 11.4 Worked Toy Example

**KB Contains:**

- $R_1$: $B_{11} \Leftrightarrow (P_{12} \lor P_{21})$
- $R_2$: $\neg B_{11}$

**Query 1:** ASK(KB, $\neg P_{12}$)

- From $R_2$: $\neg B_{11}$
- From $R_1$: $B_{11} \Leftrightarrow (P_{12} \lor P_{21})$
- Combined: $\neg(P_{12} \lor P_{21})$, so $\neg P_{12} \land \neg P_{21}$
- **Response:** TRUE

**Query 2:** ASK(KB, $P_{22}$)

- KB says nothing about $P_{22}$
- Some satisfying models have $P_{22}$ true, others false
- **Response:** UNKNOWN

**Query 3:** ASK(KB, $P_{12}$)

- We derived $\neg P_{12}$
- **Response:** FALSE

### 11.5 Connections & Prerequisites

**Prerequisite Refresher (Entailment):** ASK returns TRUE when there's entailment, FALSE when the negation is entailed, and UNKNOWN when neither holds. This directly uses the definition $KB \vDash \alpha$ meaning all models of KB are models of $\alpha$.

---

## 12. Concept: Wumpus World — Application Domain

### 12.1 High-Level Intuition

**Problem Solved:** How can we demonstrate logical reasoning in a concrete, interactive scenario where an agent must navigate safely using only logical inference?

**Analogy:** Wumpus World is like a minesweeper game where you can't see the mines directly—you only get hints from adjacent cells. Your survival depends on logically deducing safe cells from incomplete information.

### 12.2 Conceptual Deep Dive

**Wumpus World** is a classic AI testbed—a 4×4 grid where an agent must:

- Find the gold
- Avoid pits (instant death)
- Avoid the Wumpus monster (instant death)
- Return to start and exit

**Environment Features:**

|Entity|Symbol|Agent Perception|
|---|---|---|
|Pit|(deadly)|Breeze in adjacent cells|
|Wumpus|(deadly)|Stench in adjacent cells|
|Gold|(goal)|Glitter in same cell|

**Key Constraints:**

- Agent starts at (1,1), which is guaranteed safe
- Agent can only move to cells it has **proven** safe
- Perceptions come from the current cell only

**Propositional Symbols Used:**

- $P_{x,y}$: Pit at location $(x,y)$
- $W_{x,y}$: Wumpus at location $(x,y)$
- $B_{x,y}$: Breeze perceived at $(x,y)$
- $S_{x,y}$: Stench perceived at $(x,y)$

### 12.3 Mathematical Formulation

**Sample Rules (Knowledge Base):**

**Safety Rules:** $$ R_0: \neg P_{1,1} \quad \text{(No pit at start)} $$ $$ R_0': \neg W_{1,1} \quad \text{(No Wumpus at start)} $$

**Breeze-Pit Relationship:** $$ R_1: B_{1,1} \Leftrightarrow (P_{1,2} \lor P_{2,1}) $$ $$ R_2: B_{2,1} \Leftrightarrow (P_{1,1} \lor P_{2,2} \lor P_{3,1}) $$

General form for interior cell $(x,y)$: $$ B_{x,y} \Leftrightarrow (P_{x-1,y} \lor P_{x+1,y} \lor P_{x,y-1} \lor P_{x,y+1}) $$

**Stench-Wumpus Relationship (similar structure):** $$ S_{x,y} \Leftrightarrow (W_{x-1,y} \lor W_{x+1,y} \lor W_{x,y-1} \lor W_{x,y+1}) $$

### 12.4 Worked Toy Example

**Agent Journey:**

**Step 1:** At (1,1), perceive no breeze, no stench.

- Add to KB: $\neg B_{1,1}$, $\neg S_{1,1}$
- Inference from $R_1$ and $\neg B_{1,1}$:
    - $\neg B_{1,1} \Rightarrow \neg(P_{1,2} \lor P_{2,1})$
    - $\therefore \neg P_{1,2} \land \neg P_{2,1}$
- Both adjacent cells are safe from pits!

**Step 2:** Move to (2,1), perceive breeze.

- Add to KB: $B_{2,1}$
- From $R_2$: $B_{2,1} \Leftrightarrow (P_{1,1} \lor P_{2,2} \lor P_{3,1})$
- We know $\neg P_{1,1}$ (start is safe)
- $\therefore P_{2,2} \lor P_{3,1}$ (pit at (2,2) OR (3,1))
- Can't determine which—both cells are **unsafe** to enter!

**Step 3:** Return to (1,1), move to (1,2), perceive stench.

- Add to KB: $S_{1,2}$
- Adjacent to (1,2): (1,1), (2,2), (1,3)
- We know $\neg W_{1,1}$
- In Step 1, no stench at (1,1), so $\neg W_{1,2}$ and $\neg W_{2,1}$
- $\therefore$ Wumpus must be at (1,3)!

### 12.5 Connections & Prerequisites

**Prerequisite Refresher (KB Operations):** The agent uses TELL to add perceptions and rules, and ASK to determine if cells are safe. Each movement decision requires proving safety through logical inference from the accumulated KB.

---

## 13. Concept: Model Checking (Inference by Enumeration)

### 13.1 High-Level Intuition

**Problem Solved:** How can a computer automatically determine if KB entails a query by exhaustively checking all possible worlds?

**Analogy:** Model checking is like a brute-force password cracker—it tries every possible combination. For logic, it tries every possible truth assignment and checks if the query is true whenever the KB is true. Correct but potentially slow.

### 13.2 Conceptual Deep Dive

**Model checking** determines entailment by enumerating all possible models:

**Algorithm:**

1. List all propositional symbols in KB and query
2. Generate all $2^n$ possible truth assignments (models)
3. For each model:
    - Evaluate whether the model satisfies KB
    - If yes, check if the model satisfies the query
4. If EVERY model satisfying KB also satisfies the query → KB entails query

**Correctness:** Directly implements the definition of entailment.

**Complexity:** $O(2^n)$ where $n$ = number of symbols. Exponential, but guaranteed correct.

**Table Structure:**

|Model|Symbols...|$R_1$|$R_2$|...|KB (all rules)|Query|
|---|---|---|---|---|---|---|
|$M_1$|T/F values|T/F|T/F|...|T/F|T/F|
|...|...|...|...|...|...|...|

Only rows where KB = T matter. If Query = T for ALL such rows, entailment holds.

### 13.3 Mathematical Formulation

**Model Checking Algorithm:**

```
function MODEL-CHECK(KB, α):
    symbols ← all symbols in KB and α
    return CHECK-ALL(KB, α, symbols, {})

function CHECK-ALL(KB, α, symbols, model):
    if symbols is empty:
        if EVALUATE(KB, model) = true:
            return EVALUATE(α, model)
        else:
            return true  // vacuously true
    else:
        P ← first symbol in symbols
        rest ← remaining symbols
        return CHECK-ALL(KB, α, rest, model ∪ {P=true})
           AND CHECK-ALL(KB, α, rest, model ∪ {P=false})
```

**Formal Statement:** $$ KB \vDash \alpha \iff \forall M \in {0,1}^n: (\text{eval}(KB, M) = T) \Rightarrow (\text{eval}(\alpha, M) = T) $$

### 13.4 Worked Toy Example

**Given KB (from Wumpus World at T=4):**

- $R_1$: $\neg P_{1,1}$
- $R_2$: $B_{1,1} \Leftrightarrow (P_{1,2} \lor P_{2,1})$
- $R_3$: $B_{2,1} \Leftrightarrow (P_{1,1} \lor P_{2,2} \lor P_{3,1})$
- $R_4$: $\neg B_{1,1}$
- $R_5$: $B_{2,1}$

**Query:** $\neg P_{1,2}$?

**Symbols:** $B_{1,1}, B_{2,1}, P_{1,1}, P_{1,2}, P_{2,1}, P_{2,2}, P_{3,1}$ (7 symbols)

**Total models:** $2^7 = 128$

**Evaluation Process (abbreviated):**

|#|$B_{1,1}$|$B_{2,1}$|$P_{1,1}$|$P_{1,2}$|$P_{2,1}$|$P_{2,2}$|$P_{3,1}$|$R_1$|$R_2$|$R_3$|$R_4$|$R_5$|KB|$\neg P_{1,2}$|
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
|...|...|...|...|...|...|...|...|...|...|...|...|...|...|...|
|42|F|T|F|F|F|T|F|T|T|T|T|T|**T**|T|
|43|F|T|F|F|F|F|T|T|T|T|T|T|**T**|T|
|...|...|...|...|...|...|...|...|...|...|...|...|...|...|...|

**Key observations:**

- $R_4$ requires $B_{1,1} = F$
- $R_5$ requires $B_{2,1} = T$
- $R_1$ requires $P_{1,1} = F$
- $R_2$ with $B_{1,1} = F$ requires $\neg(P_{1,2} \lor P_{2,1})$, so $P_{1,2} = F$ and $P_{2,1} = F$

**Result:** In ALL rows where KB = T, we have $P_{1,2} = F$, so $\neg P_{1,2} = T$.

**Conclusion:** $KB \vDash \neg P_{1,2}$ ✓

### 13.5 Connections & Prerequisites

**Prerequisite Refresher (Entailment):** Model checking directly implements the definition: $KB \vDash \alpha$ iff every model of KB is a model of $\alpha$. By checking all models, we verify this universally quantified statement.

---

## Key Takeaways & Formulas

### Must-Remember Points:

1. **Transformer Projections Purpose:** The $W_Q$, $W_K$, $W_V$ matrices transform embeddings into a coordinate system where similar meanings cluster and attention can identify relevant context.
    
2. **Mixture of Experts Insight:** Ensemble performance depends on error correlation—uncorrelated errors give linear improvement ($\text{MSE} = \bar{V}/K$), while perfectly correlated errors give no improvement.
    
3. **Entailment Definition:** $KB \vDash \alpha$ means every model satisfying KB also satisfies $\alpha$. Equivalently, $M(KB) \subseteq M(\alpha)$.
    
4. **TELL Operation Responses:**
    
    - **Entailment:** $M(KB) \subseteq M(\alpha)$ → "I already knew that"
    - **Contingency:** $\emptyset \subset M(KB) \cap M(\alpha) \subset M(KB)$ → "Updating with new info"
    - **Contradiction:** $M(KB) \cap M(\alpha) = \emptyset$ → "Conflicts with existing knowledge"
5. **Model Checking:** Enumerate all $2^n$ models, check if query is true in all KB-satisfying models. Exponential but always correct.
    

### Essential Formulas:

$$ \text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V $$

$$ \text{MSE}_{\text{ensemble}} = \frac{1}{K}\bar{V} + \left(1 - \frac{1}{K}\right)\bar{C} $$

$$ KB \vDash \alpha \iff M(KB) \subseteq M(\alpha) \iff \forall M: (M \vDash KB) \Rightarrow (M \vDash \alpha) $$

$$ P \Rightarrow Q \equiv \neg P \lor Q $$

### Critical Table Reference:

For the final exam, bring **Table 7.11** from Russell & Norvig's "AI: A Modern Approach" — the logical equivalences table. Key equivalences include:

- $\alpha \Rightarrow \beta \equiv \neg\alpha \lor \beta$
- $\alpha \Leftrightarrow \beta \equiv (\alpha \Rightarrow \beta) \land (\beta \Rightarrow \alpha)$
- De Morgan's Laws: $\neg(\alpha \land \beta) \equiv \neg\alpha \lor \neg\beta$

![[Pasted image 20251217150936.png]]