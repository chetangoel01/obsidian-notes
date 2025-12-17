# Part I: Logical Reasoning — Proof-Based (Syntactic) Method

## Overview and Context

The professor covered two main methods for logical reasoning:

1. **Model Checking** (covered in previous lecture) — Enumerates all possible models and checks truth values semantically
2. **Theorem Proving / Proof-Based Method** (covered this lecture) — Works entirely on syntax using inference rules

**Key distinction**: Model checking works by examining truth tables and possible worlds; proof-based methods work by applying formal rules of inference without considering the actual truth values of propositions.

**Real-world relevance**: AWS uses theorem proving (specifically first-order logic) for their cybersecurity systems. When a customer asks "Is this Docker container connected to the internet?", the system uses logical reasoning to respond. This is called **Automated Theorem Proving**.

---

## Inference Rules

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

## Table 7-11: Logical Equivalences

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

## Worked Example: Wumpus World Proof

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

## Transition to First-Order Logic

The professor noted:

- Propositional logic (what we covered) uses symbols without variables
- **First-order logic** extends this with predicates, quantifiers, and variables
- AWS's cybersecurity reasoning uses first-order logic
- **Neurosymbolic reasoning** combines symbolic representations with neural networks to reduce hallucinations in LLMs

---

# Part II: Classical Planning

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

# Part III: Markov Decision Processes (MDPs) and Introduction to Reinforcement Learning

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

# Summary of Key Concepts

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

# Recommended Resources

1. **Richard Sutton's book**: "Reinforcement Learning: An Introduction" (free online)
2. **David Silver's lectures**: 12-14 lectures on RL (linked on course site)
3. **Deep Reinforcement Learning book**: By Google engineers (O'Reilly Library)
4. **AIMA textbook**: Chapters 3, 4 (problem solving), Chapter 11, Chapters 16 & 17

---

_Notes compiled from Lecture 11, Introduction to AI, NYU, November 21, 2025_