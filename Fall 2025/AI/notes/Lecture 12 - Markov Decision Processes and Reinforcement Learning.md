### Executive Summary

This lecture covers the mathematical foundations of sequential decision-making under uncertainty, bridging Markov Decision Processes (MDPs) to Reinforcement Learning (RL). We begin with the MDP framework—states, actions, rewards, and policies—then develop value functions (V and Q) that quantify the "goodness" of states and actions. The Bellman equations provide recursive relationships for computing these values, enabling both prediction (policy evaluation) and control (finding optimal policies). Finally, we transition to reinforcement learning, where the transition and reward models are unknown, introducing Monte Carlo and Temporal Difference methods that learn from experience. This material forms the foundation for understanding modern techniques like PPO used to fine-tune large language models.

---

## 1. Concept: MDP Framework and Dynamics

### High-Level Intuition

**Goal:** Provide a mathematical framework for an agent making sequential decisions in a stochastic environment where current decisions affect future opportunities.

**Analogy:** Think of navigating a city with unreliable GPS. You're at an intersection (state), you choose to turn left (action), but due to one-way streets or traffic, you might not end up exactly where you intended (stochastic transitions). Along the way, you experience satisfaction or frustration (rewards) based on travel time, scenery, etc. Your goal is to find the best driving strategy (policy) to maximize your overall trip satisfaction.

### Conceptual Deep Dive

A **Markov Decision Process** is defined by the tuple $(S, A, P, R, \gamma)$:

- **State space $S$**: The set of all possible situations the agent can be in
- **Action space $A$**: The set of all possible actions the agent can take
- **Transition model $P(s'|s,a)$**: The probability of reaching state $s'$ when taking action $a$ from state $s$
- **Reward function $R(s,a,s')$**: The immediate reward received after transitioning
- **Discount factor $\gamma \in [0,1]$**: How much future rewards are valued relative to immediate rewards

The **policy** $\pi(a|s)$ is a probability distribution over actions given a state. A **deterministic policy** assigns probability 1 to a single action; a **stochastic policy** distributes probability across multiple actions.

The **Markov property** states that the future is independent of the past given the present—all relevant information is captured in the current state.

_Visual description: Imagine a directed graph where nodes are states, edges are actions, edge labels show transition probabilities, and nodes have reward values attached._

### Mathematical Formulation

The **MDP dynamics** combine the transition model and policy:

$$P(s', r | s, a) = P(s' | s, a) \cdot R(s, a, s')$$

The **transition model** (marginalized):

$$P(s'|s,a) = \sum_{r} P(s', r | s, a)$$

The **reward model**:

$$R(s,a) = \sum_{s'} P(s'|s,a) \cdot R(s,a,s')$$

**Annotations:**

- $P(s'|s,a)$: Probability of landing in state $s'$ after taking action $a$ in state $s$
- $R(s,a,s')$: Reward for the specific transition from $s$ to $s'$ via action $a$
- $R(s,a)$: Expected reward when taking action $a$ from state $s$ (averaging over possible next states)

### Worked Toy Example

**Grid World Setup:** Consider a 2×2 grid where an agent starts at position (0,0) and wants to reach goal (1,1).

States: $S = {(0,0), (0,1), (1,0), (1,1)}$

Actions: $A = {\text{up}, \text{down}, \text{left}, \text{right}}$

Transition model (stochastic): When attempting to move right from (0,0):

- $P((0,1)|(0,0), \text{right}) = 0.8$ (succeed)
- $P((0,0)|(0,0), \text{right}) = 0.1$ (hit wall/stay)
- $P((1,0)|(0,0), \text{right}) = 0.1$ (slip down)

Rewards: $R = -1$ for each step (encourages reaching goal quickly), $R = +10$ at goal.

Policy example (uniform random): $\pi(a|s) = 0.25$ for all actions in each state.

### Connections & Prerequisites

This is a foundational concept. Prerequisites include basic probability theory (conditional probability, expectations) and familiarity with directed graphs.

---

## 2. Concept: Return and the Discount Factor

### High-Level Intuition

**Goal:** Define a single quantity that captures the total value of a sequence of rewards, accounting for the fact that immediate rewards are generally preferable to delayed ones.

**Analogy:** The return is like calculating the present value of a stream of future payments. A dollar today is worth more than a dollar next year due to opportunity cost and uncertainty—similarly, a reward now is weighted more heavily than the same reward later.

### Conceptual Deep Dive

The **return** $G_t$ is the total discounted reward from time step $t$ onward. It captures the long-term consequences of being in a state and following a policy.

The **discount factor** $\gamma$ serves multiple purposes:

- **Mathematical**: Ensures the infinite sum converges
- **Economic**: Models preference for immediate rewards
- **Practical**: Allows tuning between short-sighted ($\gamma \approx 0$) and far-sighted ($\gamma \approx 1$) behavior

The return is a **random variable** because both the actions taken (from stochastic policy) and the resulting states (from stochastic transitions) are random.

### Mathematical Formulation

$$G_t = R_{t+1} + \gamma R_{t+2} + \gamma^2 R_{t+3} + \cdots = \sum_{k=0}^{\infty} \gamma^k R_{t+k+1}$$

**Recursive form:**

$$G_t = R_{t+1} + \gamma G_{t+1}$$

**Annotations:**

- $G_t$: Return starting from time $t$
- $R_{t+k+1}$: Reward received at time step $t+k+1$
- $\gamma$: Discount factor; $\gamma^k$ is the weight applied to reward $k$ steps in the future
- When $\gamma = 0$: Agent is completely myopic (only cares about immediate reward)
- When $\gamma = 1$: Agent weights all future rewards equally (only works for finite episodes)

### Worked Toy Example

Consider an episode with rewards: $R_1 = 2, R_2 = 3, R_3 = 1, R_4 = 5$ (terminal)

With $\gamma = 0.9$:

$$G_0 = 2 + 0.9(3) + 0.9^2(1) + 0.9^3(5)$$ $$G_0 = 2 + 2.7 + 0.81 + 3.645 = 9.155$$

With $\gamma = 0.5$:

$$G_0 = 2 + 0.5(3) + 0.25(1) + 0.125(5)$$ $$G_0 = 2 + 1.5 + 0.25 + 0.625 = 4.375$$

Notice how lower $\gamma$ dramatically reduces the contribution of later rewards.

### Connections & Prerequisites

**Prerequisite Refresher on Geometric Series:** Recall that for $|r| < 1$, the infinite sum $\sum_{k=0}^{\infty} r^k = \frac{1}{1-r}$. This is why bounded rewards with $\gamma < 1$ yield finite returns: if $|R| \leq R_{max}$, then $|G_t| \leq \frac{R_{max}}{1-\gamma}$.

---

## 3. Concept: State Value Function V(s)

### High-Level Intuition

**Goal:** Quantify how "good" it is to be in a particular state when following a specific policy—enabling comparison of states and guiding decision-making.

**Analogy:** Think of a chess position's evaluation. A grandmaster can look at a board position and estimate their winning chances (the "value" of that state). This evaluation considers not just the current material but future possibilities under good play (the policy).

### Conceptual Deep Dive

The **state value function** $V^\pi(s)$ represents the expected return when starting from state $s$ and following policy $\pi$ thereafter. It answers: "On average, how much total reward will I accumulate from here?"

Key insights:

- The value depends on the policy—different policies yield different value functions
- States closer to high rewards (spatially or temporally) tend to have higher values
- Terminal/goal states typically have the highest values
- The value function encodes information about the entire future, not just immediate rewards

_Visual description: A heatmap overlay on a grid world, where brighter colors indicate higher state values, with the brightest cell being the goal state._

### Mathematical Formulation

$$V^\pi(s) = \mathbb{E}_\pi[G_t | S_t = s]$$

Expanded form:

$$V^\pi(s) = \mathbb{E}_\pi\left[\sum_{k=0}^{\infty} \gamma^k R_{t+k+1} \bigg| S_t = s\right]$$

**Annotations:**

- $V^\pi(s)$: Value of state $s$ under policy $\pi$
- $\mathbb{E}_\pi[\cdot]$: Expectation over trajectories generated by following policy $\pi$
- $G_t$: Return from time $t$ (random variable)
- $S_t = s$: Conditioning on starting in state $s$ at time $t$

### Worked Toy Example

**Two-State MDP:**

States: ${S_1, S_2}$

Transitions (deterministic):

- From $S_1$: action leads to $S_2$ with reward $R=2$
- From $S_2$: action leads to $S_1$ with reward $R=0$

With $\gamma = 0.9$, let's compute values:

From $S_1$: Get reward 2, go to $S_2$, get reward 0, go to $S_1$, get reward 2, ...

$$V(S_1) = 2 + 0.9(0) + 0.9^2(2) + 0.9^3(0) + \cdots = 2 + 0.81(2) + 0.81^2(2) + \cdots$$ $$V(S_1) = 2 \cdot \frac{1}{1-0.81} = 2 \cdot \frac{1}{0.19} \approx 10.53$$

From $S_2$: Get reward 0, go to $S_1$, then same pattern...

$$V(S_2) = 0 + 0.9(2) + 0.9^2(0) + 0.9^3(2) + \cdots = 0.9 \cdot V(S_1) \approx 9.47$$

### Connections & Prerequisites

**Prerequisite Refresher on Expected Value:** The value function is an expectation—the probability-weighted average of all possible returns. For discrete distributions: $\mathbb{E}[X] = \sum_x x \cdot P(X=x)$.

---

## 4. Concept: Action Value Function Q(s,a)

### High-Level Intuition

**Goal:** Quantify how good it is to take a specific action from a specific state—directly informing which action to choose.

**Analogy:** If $V(s)$ is like knowing a chess position's value, $Q(s,a)$ is like knowing the value after committing to a specific move. A chess engine evaluates each candidate move to find the best one—that's exactly what Q-values enable.

### Conceptual Deep Dive

The **action value function** (or **Q-function**) $Q^\pi(s,a)$ represents the expected return when starting from state $s$, taking action $a$, and then following policy $\pi$ thereafter.

Why Q-functions are powerful:

- They directly enable action selection: choose $a^* = \arg\max_a Q(s,a)$
- No need to know the transition model for decision-making
- Foundation for Q-learning and many modern RL algorithms

The relationship between V and Q:

$$V^\pi(s) = \sum_a \pi(a|s) Q^\pi(s,a)$$

The state value is the policy-weighted average of action values.

### Mathematical Formulation

$$Q^\pi(s,a) = \mathbb{E}_\pi[G_t | S_t = s, A_t = a]$$

Expanded:

$$Q^\pi(s,a) = \sum_{s',r} P(s',r|s,a)\left[r + \gamma V^\pi(s')\right]$$

Or purely in terms of Q:

$$Q^\pi(s,a) = \sum_{s',r} P(s',r|s,a)\left[r + \gamma \sum_{a'} \pi(a'|s') Q^\pi(s',a')\right]$$

**Annotations:**

- $Q^\pi(s,a)$: Value of taking action $a$ in state $s$, then following $\pi$
- $A_t = a$: We commit to action $a$ (not sampled from policy)
- $P(s',r|s,a)$: Joint probability of next state and reward given current state-action
- The inner sum $\sum_{a'} \pi(a'|s') Q^\pi(s',a')$ equals $V^\pi(s')$

### Worked Toy Example

**Grid World:** State $S_{33}$, actions = {up, down, left, right}

Assume $\gamma = 0.9$ and we've computed (somehow):

- $V(S_{43}) = 8.0$ (cell to the right)
- $V(S_{33}) = 5.5$ (current cell—if we stay)
- $V(S_{23}) = 4.0$ (cell above)

Transition probabilities for action "right":

- $P(S_{43}|S_{33}, \text{right}) = 0.8$
- $P(S_{33}|S_{33}, \text{right}) = 0.1$ (hit wall)
- $P(S_{23}|S_{33}, \text{right}) = 0.1$ (slip up)

Reward: $R = -0.04$ for any non-terminal transition

$$Q(S_{33}, \text{right}) = 0.8[-0.04 + 0.9(8.0)] + 0.1[-0.04 + 0.9(5.5)] + 0.1[-0.04 + 0.9(4.0)]$$ $$= 0.8[7.16] + 0.1[4.91] + 0.1[3.56]$$ $$= 5.728 + 0.491 + 0.356 = 6.575$$

### Connections & Prerequisites

**Prerequisite Refresher on V-function:** Recall that $V^\pi(s)$ is the expected return from state $s$ under policy $\pi$. The Q-function extends this by first committing to an action before following the policy.

---

## 5. Concept: Bellman Expectation Equations

### High-Level Intuition

**Goal:** Express the value of a state recursively in terms of the values of successor states—enabling iterative computation without explicit summation over infinite horizons.

**Analogy:** Think of calculating your expected net worth at retirement. Instead of projecting every year's income/expenses to age 65, you can say: "My net worth next year = this year's savings + discounted value of my net worth starting next year." The Bellman equation applies this recursive logic to value functions.

### Conceptual Deep Dive

The **Bellman Expectation Equation** decomposes the value into two parts:

1. **Immediate reward**: What you get right now
2. **Discounted future value**: The value of where you end up

This decomposition enables **bootstrapping**—estimating a value from other estimated values—which is computationally powerful because we can iteratively refine all values simultaneously.

The **backup diagram** visualizes this: from state $s$, branches represent possible actions (weighted by policy), each action branches to possible next states (weighted by transition probabilities), and leaf nodes contribute their values back up.

_Visual description: A tree with state $s$ at root, branching to action nodes, each action branching to next-state nodes, with arrows showing the "backup" of values from leaves to root._

### Mathematical Formulation

**For V-function:**

$$V^\pi(s) = \sum_a \pi(a|s) \sum_{s',r} P(s',r|s,a)\left[r + \gamma V^\pi(s')\right]$$

**For Q-function:**

$$Q^\pi(s,a) = \sum_{s',r} P(s',r|s,a)\left[r + \gamma \sum_{a'} \pi(a'|s') Q^\pi(s',a')\right]$$

**Matrix form for V:**

$$\mathbf{V}^\pi = \mathbf{R}^\pi + \gamma \mathbf{P}^\pi \mathbf{V}^\pi$$

Solving:

$$\mathbf{V}^\pi = (\mathbf{I} - \gamma \mathbf{P}^\pi)^{-1} \mathbf{R}^\pi$$

**Annotations:**

- $\sum_a \pi(a|s)$: Average over actions according to policy
- $\sum_{s',r} P(s',r|s,a)$: Average over possible outcomes
- $r + \gamma V^\pi(s')$: Immediate reward plus discounted future value
- $\mathbf{P}^\pi$: Transition matrix under policy $\pi$
- $\mathbf{R}^\pi$: Expected immediate reward vector under policy $\pi$

### Worked Toy Example

**Two-State System:** $S_1, S_2$ with $\gamma = 0.9$

Transition matrix (deterministic policy):

$$\mathbf{P} = \begin{pmatrix} 0 & 1 \ 1 & 0 \end{pmatrix}$$

Reward vector: $\mathbf{R} = \begin{pmatrix} 2 \ 0 \end{pmatrix}$

Solve $\mathbf{V} = \mathbf{R} + \gamma \mathbf{P} \mathbf{V}$:

$$(\mathbf{I} - \gamma \mathbf{P})\mathbf{V} = \mathbf{R}$$

$$\begin{pmatrix} 1 & -0.9 \ -0.9 & 1 \end{pmatrix} \begin{pmatrix} V_1 \ V_2 \end{pmatrix} = \begin{pmatrix} 2 \ 0 \end{pmatrix}$$

Solving: $V_1 - 0.9V_2 = 2$ and $-0.9V_1 + V_2 = 0 \Rightarrow V_2 = 0.9V_1$

Substituting: $V_1 - 0.81V_1 = 2 \Rightarrow V_1 = \frac{2}{0.19} \approx 10.53$

And $V_2 = 0.9 \times 10.53 \approx 9.47$

### Connections & Prerequisites

**Prerequisite Refresher on Return:** The Bellman equation is derived from the recursive property of returns: $G_t = R_{t+1} + \gamma G_{t+1}$. Taking expectations of both sides yields the Bellman equation.

---

## 6. Concept: Iterative Policy Evaluation

### High-Level Intuition

**Goal:** Compute the value function for a given policy without matrix inversion—using iteration that converges to the true values.

**Analogy:** Imagine estimating house prices in a neighborhood where each house's value depends on nearby houses. You start with guesses, then repeatedly update each price based on neighbors' current estimates. Eventually, prices stabilize at consistent values. This is exactly how iterative policy evaluation works.

### Conceptual Deep Dive

**Matrix inversion is dangerous** for large state spaces—$O(n^3)$ complexity and numerical instability. Instead, we exploit the fact that the Bellman operator is a **contraction mapping**: repeatedly applying it from any starting point converges to the unique fixed point.

The algorithm:

1. Initialize $V(s) = 0$ for all states
2. For each state, update using the Bellman equation
3. Repeat until values stop changing (below threshold $\epsilon$)

Convergence is guaranteed by the Banach fixed-point theorem—the contraction property ensures each iteration brings us closer to the true values.

### Mathematical Formulation

**Iterative update:**

$$V_{k+1}(s) = \sum_a \pi(a|s) \sum_{s',r} P(s',r|s,a)\left[r + \gamma V_k(s')\right]$$

**Contraction property intuition:**

For scalar case: $x_{k+1} = \gamma x_k + c$ converges to $x^* = \frac{c}{1-\gamma}$ for $\gamma < 1$.

**Stopping criterion:**

$$\max_s |V_{k+1}(s) - V_k(s)| < \epsilon$$

**Annotations:**

- $V_k(s)$: Estimate of value at iteration $k$
- $V_{k+1}(s)$: Updated estimate
- The update uses current estimates $V_k(s')$ of successor states (**bootstrapping**)
- $\epsilon$: Convergence threshold

### Worked Toy Example

**4×4 Grid World:** Terminal states at corners (0,0) and (3,3). Uniform random policy. $\gamma = 1$, $R = -1$ per step.

**Iteration 0:** $V_0(s) = 0$ for all states

**Iteration 1:** For interior state (1,1):

- Each action (up/down/left/right) has probability 0.25
- Each leads to a neighbor with current value 0
- $V_1(1,1) = 0.25 \times 4 \times [-1 + 1.0 \times 0] = -1$

All non-terminal states get $V_1 = -1$

**Iteration 2:** For state (1,1):

- Up leads to (0,1) with $V_1 = -1$
- Down leads to (2,1) with $V_1 = -1$
- etc.
- $V_2(1,1) = 0.25 \times 4 \times [-1 + 1.0 \times (-1)] = -2$

States further from terminals have more negative values (more steps needed).

### Connections & Prerequisites

**Prerequisite Refresher on Bellman Equation:** The update rule is simply the Bellman equation with $V^\pi(s')$ replaced by current estimate $V_k(s')$. Convergence means reaching the true $V^\pi$.

---

## 7. Concept: Bellman Optimality Equations

### High-Level Intuition

**Goal:** Characterize the optimal value function—the best possible values achievable by any policy—enabling identification of optimal behavior.

**Analogy:** If the Bellman Expectation Equation asks "what value do I get following this specific strategy?", the Bellman Optimality Equation asks "what's the best value I could possibly achieve with perfect play?" It's like knowing the theoretical maximum score in a game.

### Conceptual Deep Dive

The **optimal value function** $V^*(s)$ is the maximum value achievable from state $s$ over all possible policies:

$$V^*(s) = \max_\pi V^\pi(s)$$

The key insight: instead of averaging over actions (as in expectation equations), we take the **maximum** over actions. The optimal policy simply selects the action that achieves this maximum.

**Crucially, these equations are nonlinear** due to the max operator—we cannot solve them via matrix inversion. This motivates iterative algorithms like value iteration and policy iteration.

### Mathematical Formulation

**V-function optimality:**

$$
V^*(s) = \max_{a} \sum_{s', r} P(s', r \mid s, a)\left[ r + \gamma V^*(s') \right]
$$


**Q-function optimality:**
$$
Q^*(s,a) = \sum_{s', r} P(s', r \mid s, a)
\left[ r + \gamma \max_{a'} Q^*(s', a') \right]
$$


**Relationship:**

$$
V^*(s) = \max_{a} Q^*(s,a)
$$


**Optimal policy extraction:**

$$
\pi^*(a \mid s) =
\begin{cases}
1 & \text{if } a = \arg\max_{a'} Q^*(s,a') \\
0 & \text{otherwise}
\end{cases}
$$


**Annotations:**

- $\max_a$: The nonlinear operation that makes direct solution impossible
- $V^*(s)$: Best achievable expected return from state $s$
- $Q^*(s,a)$: Best achievable expected return after taking $a$ from $s$
- The optimal policy is **greedy** with respect to $Q^*$

### Worked Toy Example

**Recycling Robot:** States = {High, Low} (battery level)

Actions from High: {Search, Wait} Actions from Low: {Search, Wait, Recharge}

Given transitions and rewards, write optimality equation for $V^*(\text{High})$:

$$
V^*(\text{High}) = \max \Bigg\{
$$

**Search:** $P(H|H,S)[R(H,S,H) + \gamma V^*(H)] + P(L|H,S)[R(H,S,L) + \gamma V^*(L)]$

**Wait:** $P(H|H,W)[R(H,W,H) + \gamma V^*(H)]$

$$
\Bigg\}
$$

With $\gamma = 0.9$, $P(H|H,S) = 0.7$, $P(L|H,S) = 0.3$, $R(\text{search}) = 4$, $R(\text{wait}) = 1$:

$$
V^*(H) = \max \left\{
0.7[4 + 0.9V^*(H)] + 0.3[4 + 0.9V^*(L)],\;
1[1 + 0.9V^*(H)]
\right\}
$$

This is a system of nonlinear equations requiring iterative solution.

### Connections & Prerequisites

**Prerequisite Refresher on Bellman Expectation:** The optimality equation replaces $\sum_a \pi(a|s)$ with $\max_a$—instead of averaging over policy, we optimize over actions.


---

## 8. Concept: Policy Iteration

### High-Level Intuition

**Goal:** Find the optimal policy by alternating between evaluating the current policy and improving it greedily—guaranteed to converge to optimality.

**Analogy:** Think of iteratively improving a recipe. You follow your current recipe (policy), taste the result and rate it (evaluation). Then you try variations on each step, keeping changes that taste better (improvement). Repeat until no changes improve the dish—you've found the optimal recipe.

### Conceptual Deep Dive

**Policy Iteration** has two alternating phases:

1. **Policy Evaluation:** Compute $V^\pi$ for current policy $\pi$ (using iterative method from Concept 6)
    
2. **Policy Improvement:** Create better policy $\pi'$ by acting greedily: $$\pi'(s) = \arg\max_a \sum_{s',r} P(s',r|s,a)[r + \gamma V^\pi(s')]$$
    

The **policy improvement theorem** guarantees: if $\pi'$ is greedy w.r.t. $V^\pi$ and $\pi' \neq \pi$, then $V^{\pi'}(s) \geq V^\pi(s)$ for all $s$ (strict inequality for at least one state).

Since there are finitely many deterministic policies and each iteration improves (or equals), convergence to $\pi^*$ is guaranteed.

### Mathematical Formulation

**Algorithm:**

1. Initialize $\pi$ arbitrarily
2. **Policy Evaluation:** Solve for $V^\pi$ using iterative updates
3. **Policy Improvement:** For all $s$: $$\pi'(s) = \arg\max_a \sum_{s'} P(s'|s,a)[R(s,a,s') + \gamma V^\pi(s')]$$
4. If $\pi' = \pi$, stop (optimal found). Else, $\pi \leftarrow \pi'$, go to step 2.

**Greedy action selection:**

$$Q^\pi(s,a) = \sum_{s'} P(s'|s,a)[R(s,a,s') + \gamma V^\pi(s')]$$ $$\pi'(s) = \arg\max_a Q^\pi(s,a)$$

### Worked Toy Example

**4×4 Grid:** Terminal at (0,0), reward -1 per step, $\gamma = 1$.

**Iteration 0:**

- $\pi_0$: uniform random (0.25 each direction)
- Evaluate: $V^{\pi_0}$ computed (states far from terminal have more negative values)

**Iteration 1 (Improvement):**

- At state (1,0): neighbors have values $V(0,0)=0$, $V(1,1)=-14$, $V(2,0)=-18$
- Greedy: choose action "left" toward terminal (highest successor value)
- $\pi_1(1,0) = \text{left}$

**Iteration 1 (Evaluation):**

- Re-evaluate with new deterministic policy
- Values improve (less negative) since we move optimally

**Convergence:**

- After few iterations, arrows point toward terminal from every state
- $V^*(s) = -(\text{Manhattan distance to terminal})$

### Connections & Prerequisites

**Prerequisite Refresher on Iterative Evaluation:** Policy evaluation (step 2) uses the algorithm from Concept 6. Policy iteration wraps this with a greedy improvement step.

---

## 9. Concept: Monte Carlo Methods for RL

### High-Level Intuition

**Goal:** Estimate value functions without knowing the environment dynamics—by averaging observed returns from sampled episodes.

**Analogy:** To estimate average commute time (value), you don't need to know traffic models (dynamics). Just drive the route many times, record each trip's duration, and average them. Monte Carlo does exactly this with episodes.

### Conceptual Deep Dive

In **reinforcement learning**, we don't know $P(s'|s,a)$ or $R(s,a,s')$. We can only interact with the environment and observe outcomes.

**Monte Carlo (MC) method:**

1. Generate complete episodes following policy $\pi$
2. For each state visited, record the return $G_t$ from that point
3. Average returns over many episodes

**Key limitation:** Must wait until episode termination to compute returns. Cannot learn online or from continuing tasks.

**First-visit vs Every-visit:** First-visit MC averages returns only from the first occurrence of state $s$ per episode; every-visit averages all occurrences.

_Visual description: Instead of the full backup tree (breadth-first), MC follows a single path (depth-first) to terminal state, then backs up the observed return._

### Mathematical Formulation

**Sample return:**

$$G_t = R_{t+1} + \gamma R_{t+2} + \cdots + \gamma^{T-t-1}R_T$$

**Incremental mean update:**

$$V(s) \leftarrow V(s) + \alpha [G_t - V(s)]$$

Equivalently:

$$V(s) \leftarrow V(s) + \frac{1}{N(s)}[G_t - V(s)]$$

where $N(s)$ is the visit count for state $s$.

**Annotations:**

- $G_t$: Actual observed return from time $t$ (not estimated!)
- $\alpha = \frac{1}{N(s)}$: Learning rate (inverse of visit count for unbiased estimate)
- $G_t - V(s)$: **MC error**—difference between observed and estimated value
- Update moves estimate toward observed return

### Worked Toy Example

**Episode 1:** $S_A \rightarrow S_B \rightarrow S_C \rightarrow \text{Terminal}$

Rewards: $R_1 = 0, R_2 = 0, R_3 = 1$, $\gamma = 0.9$

Returns:

- $G(S_C) = 1$
- $G(S_B) = 0 + 0.9(1) = 0.9$
- $G(S_A) = 0 + 0.9(0.9) = 0.81$

Initialize $V(s) = 0$ for all $s$.

**After Episode 1:**

- $V(S_A) = 0 + 1.0 \times (0.81 - 0) = 0.81$
- $V(S_B) = 0 + 1.0 \times (0.9 - 0) = 0.9$
- $V(S_C) = 0 + 1.0 \times (1.0 - 0) = 1.0$

**Episode 2:** $S_A \rightarrow S_D \rightarrow \text{Terminal}$, rewards $R_1=0, R_2=0$

- $G(S_A) = 0$ this episode
- $V(S_A) = 0.81 + 0.5 \times (0 - 0.81) = 0.405$ (averaging two returns)

### Connections & Prerequisites

**Prerequisite Refresher on Returns:** MC directly uses observed returns $G_t$, which are sums of discounted rewards along a trajectory. Unlike dynamic programming, returns are computed from actual experience, not from the model.

---

## 10. Concept: Temporal Difference Learning (TD(0))

### High-Level Intuition

**Goal:** Learn value functions online, after every step, without waiting for episode termination—combining MC sampling with DP bootstrapping.

**Analogy:** Instead of waiting until your road trip ends to evaluate your route choice, TD updates your estimate at each city. "Based on reaching Chicago, my estimate of starting from Detroit just improved by this much." You bootstrap from your current belief about Chicago's value.

### Conceptual Deep Dive

**TD(0)** updates values after each transition, using the observed reward plus the estimated value of the next state (bootstrapping):

$$V(S_t) \leftarrow V(S_t) + \alpha[R_{t+1} + \gamma V(S_{t+1}) - V(S_t)]$$

The term $R_{t+1} + \gamma V(S_{t+1})$ is called the **TD target**—our one-step lookahead estimate of return.

The term $\delta_t = R_{t+1} + \gamma V(S_{t+1}) - V(S_t)$ is the **TD error**—how much our estimate was "wrong."

**Key insight:** TD combines:

- **Sampling** (like MC): We follow actual trajectories, not all possible paths
- **Bootstrapping** (like DP): We use estimated values, not actual returns

**Advantage over MC:** Can learn online, from incomplete episodes, and in continuing tasks.

### Mathematical Formulation

**TD(0) update:**

$$V(S_t) \leftarrow V(S_t) + \alpha \delta_t$$

where the **TD error** is:

$$\delta_t = R_{t+1} + \gamma V(S_{t+1}) - V(S_t)$$

**TD target:**

$$\text{Target} = R_{t+1} + \gamma V(S_{t+1})$$

**Annotations:**

- $\alpha$: Learning rate (typically constant, not $1/N$)
- $R_{t+1}$: Actual observed reward (sampled)
- $V(S_{t+1})$: Estimated value of next state (bootstrapped)
- $\delta_t$: TD error—positive means we underestimated, negative means overestimated

### Worked Toy Example

Current estimates: $V(A) = 5.0$, $V(B) = 8.0$, $\alpha = 0.1$, $\gamma = 0.9$

**Transition:** $A \xrightarrow{R=2} B$

TD target: $2 + 0.9 \times 8.0 = 2 + 7.2 = 9.2$

TD error: $\delta = 9.2 - 5.0 = 4.2$

Update: $V(A) \leftarrow 5.0 + 0.1 \times 4.2 = 5.42$

**Interpretation:** We received reward 2 and ended up in a state worth 8.0. This suggests $A$ is worth about 9.2, more than our estimate of 5.0. We adjust upward by 10% of the error.

**Compare to MC:** MC would wait until episode ends, compute actual $G_t$, then update. TD updates immediately using estimated future value.

### Connections & Prerequisites

**Prerequisite Refresher on MC:** Monte Carlo updates: $V(s) \leftarrow V(s) + \alpha[G_t - V(s)]$ where $G_t$ is the actual observed return. TD replaces $G_t$ with the bootstrapped estimate $R_{t+1} + \gamma V(S_{t+1})$.

---

## 11. Concept: TD(λ) and n-Step Returns

### High-Level Intuition

**Goal:** Bridge the gap between TD(0) (one-step lookahead) and MC (full episode)—choosing how far ahead to look before bootstrapping.

**Analogy:** When updating your GPS estimated arrival time, TD(0) updates after each turn, MC updates only at destination. TD(λ) is like updating after every few miles, using a blend of recent observations and projected estimates—more stable than updating every second, more responsive than waiting until arrival.

### Conceptual Deep Dive

**n-step TD** uses returns computed over $n$ steps before bootstrapping:

$$G_t^{(n)} = R_{t+1} + \gamma R_{t+2} + \cdots + \gamma^{n-1}R_{t+n} + \gamma^n V(S_{t+n})$$

- $n=1$: TD(0) (bootstrap after 1 step)
- $n=\infty$: MC (no bootstrapping, wait for terminal)

**TD(λ)** computes a weighted average of all n-step returns:

$$G_t^\lambda = (1-\lambda) \sum_{n=1}^{\infty} \lambda^{n-1} G_t^{(n)}$$

The parameter $\lambda \in [0,1]$ controls the weighting:

- $\lambda = 0$: Only 1-step return (TD(0))
- $\lambda = 1$: Only full return (MC)
- $0 < \lambda < 1$: Blend of all n-step returns

### Mathematical Formulation

**n-step return:**

$$G_t^{(n)} = \sum_{k=0}^{n-1} \gamma^k R_{t+k+1} + \gamma^n V(S_{t+n})$$

**n-step TD update:**

$$V(S_t) \leftarrow V(S_t) + \alpha[G_t^{(n)} - V(S_t)]$$

**λ-return:**

$$G_t^\lambda = (1-\lambda)\sum_{n=1}^{T-t-1} \lambda^{n-1}G_t^{(n)} + \lambda^{T-t-1}G_t$$

**Annotations:**

- $G_t^{(n)}$: n-step return starting at time $t$
- $\gamma^n V(S_{t+n})$: Bootstrapped value after $n$ actual rewards
- $(1-\lambda)\lambda^{n-1}$: Weight for n-step return (geometric decay)
- TD(λ) decouples update timing from bootstrap horizon

### Worked Toy Example

**Trajectory:** $S_0 \xrightarrow{R=1} S_1 \xrightarrow{R=2} S_2 \xrightarrow{R=3} S_T$ (terminal)

$\gamma = 0.9$, current estimates: $V(S_1) = 4.0$, $V(S_2) = 2.5$

**1-step return from $S_0$:** $$G_0^{(1)} = 1 + 0.9 \times 4.0 = 4.6$$

**2-step return from $S_0$:** $$G_0^{(2)} = 1 + 0.9(2) + 0.81 \times 2.5 = 1 + 1.8 + 2.025 = 4.825$$

**3-step (full) return from $S_0$:** $$G_0^{(3)} = G_0 = 1 + 0.9(2) + 0.81(3) = 1 + 1.8 + 2.43 = 5.23$$

**λ-return with $\lambda = 0.5$:** $$G_0^\lambda = 0.5 \times 4.6 + 0.25 \times 4.825 + 0.25 \times 5.23$$ $$= 2.3 + 1.206 + 1.308 = 4.814$$

### Connections & Prerequisites

**Prerequisite Refresher on TD(0):** TD(0) uses the 1-step return $G_t^{(1)} = R_{t+1} + \gamma V(S_{t+1})$. TD(λ) generalizes by mixing all possible n-step returns with exponentially decaying weights.

---

### Key Takeaways & Formulas

- **Value functions quantify long-term reward potential:** $V^\pi(s) = \mathbb{E}_\pi[G_t | S_t=s]$ tells you "how good" a state is under policy $\pi$.
    
- **The Bellman equations enable recursive computation:** $$V^\pi(s) = \sum_a \pi(a|s) \sum_{s'} P(s'|s,a)[R + \gamma V^\pi(s')]$$
    
- **Optimality introduces nonlinearity:** $V^*(s) = \max_{a} Q^*(s,a)$
—the max operator prevents direct matrix solution.
    
- **Policy iteration alternates evaluation and improvement:** Evaluate current policy → Act greedily on values → Repeat until convergence to $\pi^*$.
    
- **TD learning bridges MC and DP:** $$V(S_t) \leftarrow V(S_t) + \alpha[\underbrace{R_{t+1} + \gamma V(S_{t+1})}_{\text{TD target}} - V(S_t)]$$ This enables online learning without knowing environment dynamics—foundational for modern RL methods like PPO used to fine-tune LLMs.