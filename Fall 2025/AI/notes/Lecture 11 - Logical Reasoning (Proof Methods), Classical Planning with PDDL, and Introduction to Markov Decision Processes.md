### Executive Summary

This lecture bridges three interconnected topics in artificial intelligence. First, it completes the treatment of **logical reasoning** by introducing proof-based (syntactical) methods using inference rules—an alternative to the model-checking approach covered previously. Second, it provides a compressed overview of **classical planning** using the Planning Domain Definition Language (PDDL), demonstrating how to formally specify planning problems for automated solvers. Finally, and most substantially, the lecture introduces **Markov Decision Processes (MDPs)**, the mathematical foundation for reinforcement learning. MDPs formalize sequential decision-making under uncertainty, defining the concepts of states, actions, rewards, policies, and value functions that are essential for understanding modern AI systems, including those used to fine-tune large language models.

---

## 1. Concept: Inference Rules for Logical Proof

### 1.1 High-Level Intuition

**Goal:** Derive new true statements from existing knowledge using purely syntactical transformations, without enumerating all possible world models.

**Analogy:** Think of inference rules like the rules of algebra. Just as you can manipulate equations (e.g., adding the same value to both sides) to derive new true equations without plugging in specific numbers, inference rules let you manipulate logical sentences to derive new truths without checking every possible scenario.

### 1.2 Conceptual Deep Dive

While **model checking** (covered in the previous lecture) determines truth by exhaustively examining all possible world states, **proof-based reasoning** operates entirely on the _syntax_ of logical sentences. This approach is the foundation for theorem provers used in formal verification systems, such as those employed by AWS for cybersecurity.

Two fundamental inference rules enable this syntactical manipulation:

**Modus Ponens:** This rule states that if we know "A implies B" is true, and we also know "A" is true, then we can conclude "B" is true. It captures the intuitive notion that implications have consequences when their antecedents are satisfied.

**AND Elimination:** This rule states that if we know "A AND B" is true, then we can conclude that "A" is true (and separately, "B" is true). A conjunction being true guarantees each of its components is true.

These inference rules work in conjunction with **logical equivalences** (documented in Table 7-11 of the textbook), which allow sentences to be rewritten into equivalent forms. Common equivalences include:

- **Biconditional elimination:** $(A \Leftrightarrow B) \equiv (A \Rightarrow B) \land (B \Rightarrow A)$
- **Contraposition:** $(A \Rightarrow B) \equiv (\neg B \Rightarrow \neg A)$
- **De Morgan's Laws:** $\neg(A \lor B) \equiv \neg A \land \neg B$

### 1.3 Mathematical Formulation

**Modus Ponens:**

$$ \frac{\alpha \Rightarrow \beta, \quad \alpha}{\beta} $$

- The numerator contains the **premises** (what we know to be true)
- The denominator contains the **conclusion** (what we can derive)
- $\alpha$ and $\beta$ are logical sentences

**AND Elimination:**

$$ \frac{\alpha \land \beta}{\alpha} $$

- From a conjunction, we can extract either conjunct
- Equivalently: $\frac{\alpha \land \beta}{\beta}$

### 1.4 Worked Toy Example

**Problem:** In the Wumpus World, prove that there is no pit in cells (1,2) or (2,1) given the knowledge base.

**Given from Knowledge Base:**

- R2: $B_{1,1} \Leftrightarrow (P_{1,2} \lor P_{2,1})$ — "There's a breeze in (1,1) if and only if there's a pit in (1,2) or (2,1)"
- R4: $\neg B_{1,1}$ — "There is no breeze in (1,1)"

**Proof:**

| Step       | Statement                                                                                         | Justification                   |
| ---------- | ------------------------------------------------------------------------------------------------- | ------------------------------- |
| R2         | $B_{1,1} \Leftrightarrow (P_{1,2} \lor P_{2,1})$                                                  | Given (Knowledge Base)          |
| R8         | $(B_{1,1} \Rightarrow (P_{1,2} \lor P_{2,1})) \land ((P_{1,2} \lor P_{2,1}) \Rightarrow B_{1,1})$ | Biconditional Elimination on R2 |
| R9         | $(P_{1,2} \lor P_{2,1}) \Rightarrow B_{1,1}$                                                      | AND Elimination on R8           |
| R10        | $\neg B_{1,1} \Rightarrow \neg(P_{1,2} \lor P_{2,1})$                                             | Contraposition on R9            |
| R4         | $\neg B_{1,1}$                                                                                    | Given (Knowledge Base)          |
| R11        | $\neg(P_{1,2} \lor P_{2,1})$                                                                      | Modus Ponens on R10 and R4      |
| **Result** | $\neg P_{1,2} \land \neg P_{2,1}$                                                                 | De Morgan's Law on R11          |

**Conclusion:** There is no pit in (1,2) AND no pit in (2,1). ∎

### 1.5 Connections & Prerequisites

**Prerequisite Refresher on Propositional Logic:** Recall that in propositional logic, we work with atomic propositions (like $P_{1,2}$ meaning "there is a pit at location (1,2)") combined using logical connectives ($\land$, $\lor$, $\neg$, $\Rightarrow$, $\Leftrightarrow$). A knowledge base is a conjunction of sentences known to be true. Model checking evaluates truth by examining all possible truth assignments; proof-based methods derive truth through syntactical transformation.

---

## 2. Concept: Classical Planning with PDDL

### 2.1 High-Level Intuition

**Goal:** Automatically generate a sequence of actions that transforms an initial state into a desired goal state, without interacting with a stochastic environment.

**Analogy:** Think of PDDL planning like writing a detailed recipe. You specify what ingredients you start with (initial state), what dish you want to end up with (goal state), and the possible cooking operations (action schemas with their prerequisites and effects). A solver then figures out the exact sequence of steps to follow.

### 2.2 Conceptual Deep Dive

**Classical planning** addresses deterministic, fully observable environments where an agent must find a sequence of actions to achieve a goal. Unlike reinforcement learning (covered later), there is no stochasticity—actions have predictable outcomes, and the environment doesn't "push back" with random effects.

**PDDL (Planning Domain Definition Language)** is a domain-specific language for expressing planning problems. It separates the specification into two files:

1. **Domain File:** Defines the general "physics" of the world
    
    - **Types:** Classes of objects (e.g., `block`, `location`)
    - **Predicates:** Properties and relations that can be true or false (e.g., `on(block, block)`, `clear(block)`)
    - **Action Schemas:** Templates for actions with parameters, preconditions, and effects
2. **Problem File:** Defines a specific instance
    
    - **Objects:** Specific instances of types (e.g., `blockA`, `blockB`)
    - **Initial State:** Which predicates are true at the start
    - **Goal State:** Which predicates must be true at the end

A **PDDL state** is a conjunction of **grounded atomic fluents**—predicates instantiated with specific objects. For example: $on(A, table) \land on(B, table) \land on(C, A)$ describes a specific arrangement of blocks.

**Action schemas** define:

- **Parameters:** Variables representing objects involved in the action
- **Preconditions:** Conditions that must be true for the action to be executable
- **Effects:** How the world changes when the action is executed (predicates that become true or false)

### 2.3 Mathematical Formulation

An action schema is formally defined as:

$$ \text{Action}(name(parameters)) : \text{Precond}(\phi) \land \text{Effect}(\psi) $$

Where:

- $name$ is the action identifier
- $parameters$ are typed variables
- $\phi$ is a conjunction of literals that must hold before execution
- $\psi$ specifies literals that become true (positive effects) or false (negative effects, prefixed with $\neg$)

**Example - Move Action Schema:**

$$ \text{Action}(\text{Move}(b : Block, x : Location, y : Block)) $$

**Preconditions:** $$ on(b, x) \land clear(b) \land clear(y) \land block(b) \land block(y) \land (b \neq x) \land (x \neq y) \land (b \neq y) $$

**Effects:** $$ on(b, y) \land clear(x) \land \neg on(b, x) \land \neg clear(y) $$

### 2.4 Worked Toy Example

**Blocks World Problem:**

_Initial State:_ Block A is on the table, Block B is on the table, Block C is on top of A.

_Goal State:_ Block B is on top of Block C.

**Initial State in PDDL:**

```
on(A, table) ∧ on(B, table) ∧ on(C, A) ∧ clear(B) ∧ clear(C)
```

**Goal:**

```
on(B, C)
```

**Solution Sequence:**

1. **Move(C, A, table):** Move C from on top of A to the table
    
    - Preconditions satisfied: $on(C, A) \land clear(C) \land clear(table)$ ✓
    - Effects: $on(C, table) \land clear(A) \land \neg on(C, A)$
2. **Move(B, table, C):** Move B from the table onto C
    
    - Preconditions satisfied: $on(B, table) \land clear(B) \land clear(C)$ ✓
    - Effects: $on(B, C) \land \neg clear(C)$

**Final State:** $on(A, table) \land on(C, table) \land on(B, C) \land clear(A) \land clear(B)$

Goal $on(B, C)$ is achieved. ✓

### 2.5 Connections & Prerequisites

**Connection to Logical Reasoning:** PDDL leverages first-order logic concepts (predicates with variables, quantification over objects). The preconditions and effects are logical formulas. Understanding propositional logic and the structure of logical sentences is essential for writing correct PDDL specifications.

**Connection to Reinforcement Learning:** Classical planning assumes a deterministic environment. When environments become stochastic (outcomes are probabilistic), we transition to MDPs and reinforcement learning, which handle uncertainty through expected values and learned policies.

---

## 3. Concept: Markov Decision Process (MDP) Framework

### 3.1 High-Level Intuition

**Goal:** Provide a mathematical framework for sequential decision-making where outcomes are uncertain, enabling an agent to learn optimal behavior through interaction with an environment.

**Analogy:** Imagine playing a board game where rolling dice determines your movement. You choose which direction to try to move (action), but randomness affects where you actually land (stochastic transitions). Some squares give you points, others cost you points (rewards). An MDP formalizes how to make the best choices over many turns to maximize your total score.

### 3.2 Conceptual Deep Dive

A **Markov Decision Process** models an agent interacting with a stochastic environment over discrete time steps. At each step:

1. The agent observes the current **state** $S_t$
2. The agent selects an **action** $A_t$
3. The environment transitions to a new **state** $S_{t+1}$
4. The environment emits a **reward** $R_{t+1}$

The **Markov property** states that the future depends only on the current state, not the history of how we got there. Mathematically: $P(S_{t+1} | S_t, A_t, S_{t-1}, A_{t-1}, ...) = P(S_{t+1} | S_t, A_t)$

Key terminology:

- **Episode:** A complete interaction from start to a terminal state
- **Trajectory:** The sequence of (state, action, reward) tuples in an episode
- **Experience:** Denoted as $(S, A, R, S', A', R', ...)$

The MDP is formally defined by the 5-tuple $\mathcal{M} = (\mathcal{S}, \mathcal{A}, \mathcal{P}, \mathcal{R}, \gamma)$ where:

- $\mathcal{S}$: Set of all possible states
- $\mathcal{A}$: Set of all possible actions
- $\mathcal{P}$: Transition probability model
- $\mathcal{R}$: Reward model
- $\gamma$: Discount factor

### 3.3 Mathematical Formulation

**MDP Dynamics (Joint Distribution):**

$$ p(s', r | s, a) = P(S_{t+1} = s', R_{t+1} = r | S_t = s, A_t = a) $$

This is the probability of transitioning to state $s'$ and receiving reward $r$, given that we're in state $s$ and take action $a$.

**Transition Model (marginalized over rewards):**

$$ p(s' | s, a) = \sum_{r} p(s', r | s, a) $$

- $p(s' | s, a)$: Probability of reaching state $s'$ from state $s$ after taking action $a$

**Reward Model (marginalized over next states):**

$$ p(r | s, a) = \sum_{s'} p(s', r | s, a) $$

**Expected Reward Function (two-argument form):**

$$ r(s, a) = \mathbb{E}[R_t | S_{t-1} = s, A_{t-1} = a] = \sum_{r} r \sum_{s'} p(s', r | s, a) $$

- $r(s, a)$: Expected immediate reward for taking action $a$ in state $s$

### 3.4 Worked Toy Example

**Grid World Setup:**

Consider a 3×4 grid world:

```
+----+----+----+----+
|    |    |    | +1 |  (Goal: +1 reward)
+----+----+----+----+
|    | ## |    | -1 |  (Trap: -1 reward)
+----+----+----+----+
| S  |    |    |    |  (S = Start)
+----+----+----+----+
```

**Transition Model (Stochastic):**

- Intended direction: 80% probability
- Each perpendicular direction: 10% probability each
- Hitting a wall means staying in place

**Example Calculation:**

From state $S_{11}$ (bottom-left), taking action "UP":

|Next State|Probability|Explanation|
|---|---|---|
|$S_{21}$|0.8|Successfully moved up|
|$S_{11}$|0.1|Tried to go left, hit wall, stayed|
|$S_{12}$|0.1|Drifted right|

So the transition model entry: $p(S_{21} | S_{11}, \text{UP}) = 0.8$

**Reward Structure:**

- Reaching $+1$ terminal state: $r = +1$
- Reaching $-1$ terminal state: $r = -1$
- All other transitions: $r = -0.04$ (small penalty to encourage efficiency)

### 3.5 Connections & Prerequisites

**Prerequisite Refresher on Probabilistic Graphical Models:** Recall from Hidden Markov Models that we model sequential processes using directed graphs where nodes represent random variables and edges represent dependencies. In an MDP, the graphical structure shows that $S_{t+1}$ and $R_{t+1}$ depend on $S_t$ and $A_t$. The convention is that rewards and next states are determined _simultaneously_ when the step function executes.

---

## 4. Concept: Return and Discounting

### 4.1 High-Level Intuition

**Goal:** Define a single scalar quantity that captures the total value of all rewards an agent receives over time, accounting for the fact that immediate rewards are more certain than future ones.

**Analogy:** The discount factor is like the "net present value" in finance. A dollar today is worth more than a dollar next year because of uncertainty and opportunity cost. Similarly, a reward received now is more valuable than the same reward received far in the future.

### 4.2 Conceptual Deep Dive

The **return** $G_t$ is the total accumulated reward from time step $t$ onward. However, simply summing all future rewards creates problems:

1. In infinite-horizon problems, the sum might be infinite
2. Future rewards are more uncertain and should count less

The **discount factor** $\gamma \in [0, 1)$ solves both issues by geometrically decreasing the weight of future rewards.

**Interpretation of $\gamma$:**

- $\gamma = 0$: Completely **myopic**—only the immediate reward matters
- $\gamma \rightarrow 1$: **Far-sighted**—future rewards are nearly as important as immediate ones
- Typical values: 0.9 to 0.99

**Infinite horizon** means no deadline is imposed on finding a solution. The interaction terminates when reaching a terminal state, not at a predetermined time. This is actually mathematically simpler than finite-horizon problems (which require time-dependent policies).

### 4.3 Mathematical Formulation

**Return (Discounted Cumulative Reward):**

$$ G_t = R_{t+1} + \gamma R_{t+2} + \gamma^2 R_{t+3} + ... = \sum_{k=0}^{\infty} \gamma^k R_{t+1+k} $$

Where:

- $G_t$: Return starting from time $t$
- $R_{t+k}$: Reward received at time $t+k$
- $\gamma$: Discount factor $(0 \leq \gamma < 1)$
- $k$: Number of steps into the future

**Recursive Property:**

$$ G_t = R_{t+1} + \gamma G_{t+1} $$

This shows that the return at time $t$ equals the immediate reward plus the discounted return from the next time step.

### 4.4 Worked Toy Example

**Scenario:** An agent receives the following reward sequence: $R_1 = 1, R_2 = 2, R_3 = 3, R_4 = 0$ (then terminates)

**Calculate $G_0$ with $\gamma = 0.9$:**

$$ G_0 = R_1 + \gamma R_2 + \gamma^2 R_3 + \gamma^3 R_4 $$

$$ G_0 = 1 + (0.9)(2) + (0.9)^2(3) + (0.9)^3(0) $$

$$ G_0 = 1 + 1.8 + (0.81)(3) + 0 $$

$$ G_0 = 1 + 1.8 + 2.43 = 5.23 $$

**Compare with $\gamma = 0.5$:**

$$ G_0 = 1 + (0.5)(2) + (0.25)(3) + 0 = 1 + 1 + 0.75 = 2.75 $$

**Observation:** With higher $\gamma$, distant rewards contribute more, increasing the total return.

### 4.5 Connections & Prerequisites

**Connection to MDP Dynamics:** The return depends on the trajectory of states and actions, which are governed by the transition model and policy. Since transitions are stochastic, the return $G_t$ is itself a random variable—we typically work with its expected value.

---

## 5. Concept: Policy

### 5.1 High-Level Intuition

**Goal:** Define a decision-making strategy that tells the agent what action to take in each state.

**Analogy:** A policy is like a playbook in sports. It specifies what play to run (action) based on the current game situation (state). A deterministic playbook always calls the same play in the same situation; a stochastic playbook might randomize to keep opponents guessing.

### 5.2 Conceptual Deep Dive

A **policy** $\pi$ is a mapping from states to actions—it completely specifies the agent's behavior. Policies can be:

**Stochastic Policy:** Specifies a probability distribution over actions for each state. The agent samples from this distribution to select actions. Written as $\pi(a|s) = P(A_t = a | S_t = s)$.

**Deterministic Policy:** A special case where one action has probability 1 in each state. The agent always takes the same action in the same state.

Why use stochastic policies?

- **Exploration:** Randomness helps discover new strategies
- **Game theory:** In adversarial settings, randomization prevents exploitation
- **Mathematical convenience:** Many algorithms naturally produce stochastic policies

The goal of reinforcement learning is to find the **optimal policy** $\pi^*$ that maximizes expected return from any starting state.

### 5.3 Mathematical Formulation

**Stochastic Policy:**

$$ \pi(a|s) = P(A_t = a | S_t = s) $$

Where:

- $\pi(a|s)$: Probability of taking action $a$ in state $s$
- Must satisfy: $\sum_{a \in \mathcal{A}} \pi(a|s) = 1$ for all $s$

**Deterministic Policy:**

$$ \pi(a_i|s) = 1 \text{ for some specific } a_i $$

$$ \pi(a_j|s) = 0 \text{ for all } a_j \neq a_i $$

**Example - Uniform Random Policy:**

$$ \pi(a|s) = \frac{1}{|\mathcal{A}|} \text{ for all } a \in \mathcal{A} $$

If there are 4 possible actions: $\pi(a|s) = 0.25$ for each action.

### 5.4 Worked Toy Example

**Grid World with 4 actions:** UP, DOWN, LEFT, RIGHT

**Uniform Random Policy:**

| State | $\pi(\text{UP}|s)$ | $\pi(\text{DOWN}|s)$ | $\pi(\text{LEFT}|s)$ | $\pi(\text{RIGHT}|s)$ | |-------|-----|------|------|-------| | Any $s$ | 0.25 | 0.25 | 0.25 | 0.25 |

**Deterministic Policy (example):**

| State | $\pi(\text{UP}|s)$ | $\pi(\text{DOWN}|s)$ | $\pi(\text{LEFT}|s)$ | $\pi(\text{RIGHT}|s)$ | |-------|-----|------|------|-------| | $S_{11}$ | 1.0 | 0 | 0 | 0 | | $S_{12}$ | 0 | 0 | 0 | 1.0 | | $S_{21}$ | 1.0 | 0 | 0 | 0 |

This deterministic policy always goes UP from $S_{11}$, RIGHT from $S_{12}$, and UP from $S_{21}$.

### 5.5 Connections & Prerequisites

**Important Clarification:** A stochastic policy does NOT make the MDP stochastic. The MDP's stochasticity comes from the transition model $p(s'|s,a)$—the environment's response to actions. The policy determines which action to take; the transition model determines what happens after that action. Even with a deterministic policy, the MDP remains stochastic if $p(s'|s,a)$ is not a delta function.

---

## 6. Concept: State Value Function

### 6.1 High-Level Intuition

**Goal:** Quantify how "good" it is to be in a particular state when following a specific policy—measured by the expected total future reward.

**Analogy:** Think of the value function as a "heat map" over a maze. States near the exit have high value (you're close to the reward), while states far away or near traps have low value. The value tells you the expected "score" you'll achieve starting from each location if you follow your current strategy.

### 6.2 Conceptual Deep Dive

The **state value function** $V^\pi(s)$ answers: "Starting from state $s$ and following policy $\pi$ thereafter, what is my expected return?"

Key insights:

- The value function is defined **with respect to a specific policy** $\pi$
- It's an **expectation** because outcomes are random (stochastic transitions and possibly stochastic policy)
- States leading to high rewards have high value; states leading to penalties have low value

The value function has a recursive structure captured by the **Bellman expectation equation**. This recursion says: the value of a state equals the expected immediate reward plus the discounted value of the next state.

The **backup diagram** visualizes this recursion as a tree: from a state node, actions branch out (weighted by policy probabilities), and from each action, next states branch out (weighted by transition probabilities). The value "backs up" from future states to the current state.

### 6.3 Mathematical Formulation

**State Value Function:**

$$ V^\pi(s) = \mathbb{E}_\pi[G_t | S_t = s] $$

Where:

- $V^\pi(s)$: Value of state $s$ under policy $\pi$
- $\mathbb{E}_\pi$: Expectation over trajectories generated by policy $\pi$
- $G_t$: Return from time $t$

**Expanded Form:**

$$ V^\pi(s) = \mathbb{E}_\pi[R_{t+1} + \gamma R_{t+2} + \gamma^2 R_{t+3} + ... | S_t = s] $$

**Bellman Expectation Equation for $V^\pi$:**

$$ V^\pi(s) = \mathbb{E}_\pi[R_{t+1} + \gamma V^\pi(S_{t+1}) | S_t = s] $$

This can be written explicitly as:

$$ V^\pi(s) = \sum_{a} \pi(a|s) \sum_{s'} p(s'|s,a) \left[ r(s,a,s') + \gamma V^\pi(s') \right] $$

Where:

- $\sum_a \pi(a|s)$: Sum over actions weighted by policy
- $\sum_{s'} p(s'|s,a)$: Sum over next states weighted by transition probabilities
- $r(s,a,s')$: Expected reward for transition
- $\gamma V^\pi(s')$: Discounted value of next state

### 6.4 Worked Toy Example

**Simple 2-State MDP:**

States: ${S_1, S_2}$ where $S_2$ is terminal (absorbing)

From $S_1$: Action "go" transitions to $S_2$ with probability 1, reward = +10

Policy: $\pi(\text{go}|S_1) = 1$

Discount: $\gamma = 0.9$

**Calculate $V^\pi(S_1)$:**

Since $S_2$ is terminal: $V^\pi(S_2) = 0$

Using Bellman equation: $$ V^\pi(S_1) = \sum_a \pi(a|S_1) \sum_{s'} p(s'|S_1, a)[r + \gamma V^\pi(s')] $$

$$ V^\pi(S_1) = 1 \cdot 1 \cdot [10 + 0.9 \cdot 0] = 10 $$

**More Complex Example:**

Now suppose from $S_1$, action "go" leads to:

- $S_2$ (terminal, reward +10) with probability 0.8
- $S_1$ (back to start, reward -1) with probability 0.2

$$ V^\pi(S_1) = 1 \cdot [0.8(10 + 0) + 0.2(-1 + 0.9 \cdot V^\pi(S_1))] $$

$$ V^\pi(S_1) = 8 + 0.2(-1 + 0.9 V^\pi(S_1)) $$

$$ V^\pi(S_1) = 8 - 0.2 + 0.18 V^\pi(S_1) $$

$$ 0.82 V^\pi(S_1) = 7.8 $$

$$ V^\pi(S_1) = \frac{7.8}{0.82} \approx 9.51 $$

### 6.5 Connections & Prerequisites

**Prerequisite Refresher on Expected Value:** The expectation $\mathbb{E}[X]$ of a random variable is its "average" value weighted by probabilities. For discrete distributions: $\mathbb{E}[X] = \sum_x x \cdot P(X=x)$. The subscript $\pi$ in $\mathbb{E}_\pi$ indicates that probabilities come from following policy $\pi$, which determines action selection and thus (via the transition model) the distribution over future states.

**Connection to Backup Trees:** The Bellman equation can be visualized as a tree. The root is state $s$. Branches go to action nodes (weighted by $\pi(a|s)$). From each action, branches go to next-state nodes (weighted by $p(s'|s,a)$). The value "backs up" from leaves to root through this tree structure.

---

## 7. Concept: Prediction vs. Control Problems

### 7.1 High-Level Intuition

**Goal:** Distinguish between evaluating a given policy (prediction) and finding the best policy (control)—the two fundamental problems in reinforcement learning.

**Analogy:** Prediction is like grading a student's existing study strategy—you evaluate how well it works. Control is like being a tutor who helps the student find the optimal study strategy. You must first know how to grade strategies (prediction) before you can improve them (control).

### 7.2 Conceptual Deep Dive

**Prediction Problem:** Given a policy $\pi$, compute the value function $V^\pi(s)$ for all states. This is also called **policy evaluation**. We're answering: "How good is this specific strategy?"

**Control Problem:** Find the optimal policy $\pi^*$ that maximizes value for all states. This involves:

1. Evaluating the current policy (prediction)
2. Improving the policy based on the evaluation
3. Repeating until convergence

The control problem uses **Bellman optimality equations** (to be covered) and algorithms like:

- **Value Iteration:** Directly compute optimal values, then extract policy
- **Policy Iteration:** Alternate between evaluation and improvement

These iterative algorithms are guaranteed to converge to the optimal policy in finite MDPs.

### 7.3 Mathematical Formulation

**Prediction Problem:**

Given: Policy $\pi(a|s)$, MDP $\mathcal{M} = (\mathcal{S}, \mathcal{A}, \mathcal{P}, \mathcal{R}, \gamma)$

Find: $V^\pi(s)$ for all $s \in \mathcal{S}$

**Control Problem:**

Given: MDP $\mathcal{M} = (\mathcal{S}, \mathcal{A}, \mathcal{P}, \mathcal{R}, \gamma)$

Find: $\pi^* = \arg\max_\pi V^\pi(s)$ for all $s \in \mathcal{S}$

**Optimal Value Function:**

$$ V^*(s) = \max_\pi V^\pi(s) $$

**Optimal Policy:** A policy $\pi^_$ is optimal if $V^{\pi^_}(s) \geq V^\pi(s)$ for all states $s$ and all policies $\pi$.

### 7.4 Worked Toy Example

**Prediction Example:**

Given the 4-state grid world and uniform random policy $\pi(a|s) = 0.25$ for all actions:

The prediction problem asks: "What is $V^\pi(s)$ for each grid cell?"

This requires solving a system of linear equations (one Bellman equation per state) or using iterative methods.

**Control Example (Preview):**

After solving prediction, we might find:

- $V^\pi(S_{11}) = 0.5$
- $V^\pi(S_{12}) = 0.8$

Control asks: "Can we do better than the uniform random policy?"

If going RIGHT from $S_{11}$ leads to $S_{12}$ (higher value), we should increase $\pi(\text{RIGHT}|S_{11})$. This is **policy improvement**.

### 7.5 Connections & Prerequisites

**Connection to Bellman Equations:** Prediction uses the Bellman _expectation_ equation (value under a given policy). Control uses the Bellman _optimality_ equation (value under the best possible policy). Understanding prediction is essential before tackling control, as most control algorithms involve repeated policy evaluation steps.

---

### Key Takeaways & Formulas

- **Modus Ponens:** From $\alpha \Rightarrow \beta$ and $\alpha$, conclude $\beta$. This and AND elimination enable purely syntactical logical proofs.
    
- **PDDL Structure:** Domain (types, predicates, action schemas) + Problem (objects, initial state, goal state) → Solver finds action sequence.
    
- **MDP Definition:** $\mathcal{M} = (\mathcal{S}, \mathcal{A}, \mathcal{P}, \mathcal{R}, \gamma)$ — the mathematical framework for sequential decision-making under uncertainty.
    
- **Return with Discounting:** $G_t = \sum_{k=0}^{\infty} \gamma^k R_{t+1+k}$ — future rewards are exponentially discounted; $\gamma$ near 0 is myopic, near 1 is far-sighted.
    
- **Bellman Expectation Equation:** $V^\pi(s) = \mathbb{E}_\pi[R_{t+1} + \gamma V^\pi(S_{t+1}) | S_t = s]$ — connects value of current state to expected immediate reward plus discounted future value.
    
- **Prediction vs. Control:** Prediction evaluates a given policy; control finds the optimal policy. Both rely on the Bellman equations and form the foundation of reinforcement learning algorithms.