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

|Topic|Key Points|
|---|---|
|Transformers Review|WQ, WK, WV project to meaningful coordinate systems; multi-head captures different patterns|
|MoE|Ensemble benefits come from uncorrelated mistakes; C=0 gives linear error reduction|
|Vision Transformers|Image patches = tokens; attention heads can focus on color/texture/shape|
|Propositional Logic|Symbols + operators; models are truth assignments|
|Wumpus World|Demonstrates rule-based reasoning; conservative navigation|
|KB Operations|Tell (entailment/contingency/contradiction), Ask (true/false/unknown)|
|Model Checking|Enumerate all models, check which satisfy KB, evaluate query|
