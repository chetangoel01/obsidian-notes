### Executive Summary

This lecture bridges foundational reinforcement learning concepts with modern deep learning fine-tuning techniques used in Large Language Models (LLMs). Starting from TD(λ) methods, the lecture progresses through the taxonomy of RL algorithms (value-based, policy-based, and model-based), then develops the theory behind policy gradient methods (REINFORCE), introduces the actor-critic framework with advantage functions, and culminates in the Proximal Policy Optimization (PPO) algorithm—the backbone of RLHF in systems like ChatGPT. The lecture also contextualizes these methods within the four-stage LLM training pipeline: pre-training → supervised fine-tuning → preference fine-tuning → reasoning fine-tuning.

> **⚠️ Exam Scope Note:** The professor explicitly stated that content from this lecture (TD(λ) onwards through PPO) is **OUT OF SCOPE** for the final exam. Material from previous lectures (MDP, value functions, Q-functions, Monte Carlo, TD(0), policy iteration) remains **IN SCOPE**.

---

## 1. Concept: n-Step Temporal Difference Returns

### 1.1 High-Level Intuition

**Goal:** TD(0) bootstraps from the very next state, while Monte Carlo waits until episode termination—n-step TD provides a middle ground by bootstrapping after n steps.

**Analogy:** Imagine estimating your total trip time. TD(0) is like updating your estimate after each traffic light (immediate feedback). Monte Carlo waits until you arrive (full information). n-step TD is like updating your estimate after every few blocks—you get more information than one step, but don't wait for the entire journey.

### 1.2 Conceptual Deep Dive

The **n-step return** extends the one-step TD target by incorporating multiple actual rewards before bootstrapping from a future state's value estimate. Instead of immediately using the estimated value of the next state, we collect n actual rewards and then bootstrap from the state n steps ahead.

The key insight is that as n increases:

- We rely more on actual experienced rewards (reducing bias)
- We increase variance (more randomness in the trajectory)
- At n = ∞, we recover Monte Carlo methods

The **n-step TD target** replaces the single-step bootstrap with a multi-step lookahead, providing a spectrum between TD(0) and Monte Carlo.

### 1.3 Mathematical Formulation

The **n-step return** is defined as:

$$ G_t^{(n)} = R_{t+1} + \gamma R_{t+2} + \gamma^2 R_{t+3} + \cdots + \gamma^{n-1} R_{t+n} + \gamma^n V(S_{t+n}) $$

Where:

- $G_t^{(n)}$ is the n-step return starting from time $t$
- $R_{t+k}$ is the reward received $k$ steps after time $t$
- $\gamma$ is the discount factor ($0 \leq \gamma \leq 1$)
- $V(S_{t+n})$ is the estimated value of the state reached after $n$ steps

The **value function update** becomes:

$$ V_{t+n}(S_t) = V_{t+n-1}(S_t) + \alpha \left[ G_t^{(n)} - V_{t+n-1}(S_t) \right] $$

Where:

- $\alpha$ is the learning rate
- The update occurs at time $t+n$ (delayed update)

### 1.4 Worked Toy Example

Consider a simple 3-state episode with $\gamma = 0.9$ and current estimates $V(S_1) = 5$, $V(S_2) = 8$, $V(S_3) = 10$.

**Trajectory:** $S_1 \xrightarrow{R=2} S_2 \xrightarrow{R=3} S_3$ (terminal)

**1-step return from $S_1$:** $$G_1^{(1)} = R_2 + \gamma V(S_2) = 2 + 0.9 \times 8 = 2 + 7.2 = 9.2$$

**2-step return from $S_1$:** $$G_1^{(2)} = R_2 + \gamma R_3 + \gamma^2 V(S_3) = 2 + 0.9 \times 3 + 0.81 \times 10 = 2 + 2.7 + 8.1 = 12.8$$

With $\alpha = 0.1$:

- 1-step update: $V(S_1) \leftarrow 5 + 0.1(9.2 - 5) = 5 + 0.42 = 5.42$
- 2-step update: $V(S_1) \leftarrow 5 + 0.1(12.8 - 5) = 5 + 0.78 = 5.78$

### 1.5 Connections & Prerequisites

**Prerequisite Refresher (TD(0)):** Recall that TD(0) updates the value function using a single-step bootstrap: $V(S_t) \leftarrow V(S_t) + \alpha[R_{t+1} + \gamma V(S_{t+1}) - V(S_t)]$. The term $R_{t+1} + \gamma V(S_{t+1})$ is the TD target, and $R_{t+1} + \gamma V(S_{t+1}) - V(S_t)$ is the TD error.

---

## 2. Concept: TD(λ) — Eligibility Traces

### 2.1 High-Level Intuition

**Goal:** Instead of choosing a single n for n-step returns, TD(λ) computes a weighted average of ALL n-step returns, with exponentially decaying weights controlled by parameter λ.

**Analogy:** Think of assigning credit for a team's success. TD(0) only credits the most recent action. Monte Carlo credits everyone equally. TD(λ) is like a manager who gives the most credit to recent contributions but doesn't ignore earlier work—the λ parameter controls how quickly the credit "fades" for older actions.

### 2.2 Conceptual Deep Dive

**TD(λ)** elegantly interpolates between TD(0) and Monte Carlo by computing a geometrically weighted average of all n-step returns. The parameter **λ** (lambda) controls this weighting:

- When **λ = 0**: Only the 1-step return is used → TD(0)
- When **λ = 1**: All returns are weighted equally → Monte Carlo
- When **0 < λ < 1**: Exponentially decaying weights favor shorter returns

The weights follow a geometric distribution: the 1-step return gets weight $(1-\lambda)$, the 2-step return gets weight $(1-\lambda)\lambda$, the 3-step return gets $(1-\lambda)\lambda^2$, and so on. This ensures the weights sum to 1.

_Visual Description: Imagine a bar chart where the x-axis represents n (number of steps) and the y-axis represents weight. Starting at $(1-\lambda)$ for n=1, each subsequent bar is λ times the height of the previous one, creating an exponentially decaying envelope._

### 2.3 Mathematical Formulation

The **λ-return** is defined as:

$$ G_t^{\lambda} = (1 - \lambda) \sum_{n=1}^{\infty} \lambda^{n-1} G_t^{(n)} $$

Where:

- $G_t^{\lambda}$ is the λ-weighted return
- $\lambda \in [0, 1]$ is the trace decay parameter
- $(1-\lambda)\lambda^{n-1}$ is the weight for the n-step return
- $G_t^{(n)}$ is the n-step return defined previously

The **TD(λ) update** rule:

$$ V_{t+n}(S_t) = V_{t+n-1}(S_t) + \alpha \left[ G_t^{\lambda} - V_{t+n-1}(S_t) \right] $$

**Special Cases:**

- $\lambda = 0$: $G_t^{\lambda} = G_t^{(1)} = R_{t+1} + \gamma V(S_{t+1})$ (TD(0))
- $\lambda = 1$: $G_t^{\lambda} = G_t$ (Monte Carlo return)

### 2.4 Worked Toy Example

Consider a 3-step episode with rewards $R_1 = 1$, $R_2 = 2$, $R_3 = 3$ (terminal). Let $\gamma = 1.0$ (no discounting) and $V(S_k) = 0$ for all states initially.

**n-step returns from $S_0$:**

- $G_0^{(1)} = 1 + V(S_1) = 1 + 0 = 1$
- $G_0^{(2)} = 1 + 2 + V(S_2) = 3 + 0 = 3$
- $G_0^{(3)} = 1 + 2 + 3 = 6$ (Monte Carlo, terminal)

**With λ = 0.5:**

Weights: $(1-0.5) = 0.5$ for $G^{(1)}$, $(0.5)(0.5) = 0.25$ for $G^{(2)}$, $(0.5)(0.5)^2 = 0.125$ for $G^{(3)}$

Note: At termination, remaining weight $(1 - 0.5 - 0.25 - 0.125 = 0.125)$ goes to the final return.

$$G_0^{\lambda} = 0.5(1) + 0.25(3) + 0.25(6) = 0.5 + 0.75 + 1.5 = 2.75$$

### 2.5 Connections & Prerequisites

**Prerequisite Refresher (Monte Carlo Returns):** Monte Carlo methods estimate value by averaging complete episode returns: $V(s) \leftarrow V(s) + \alpha[G_t - V(s)]$ where $G_t = R_{t+1} + \gamma R_{t+2} + \cdots + \gamma^{T-t-1}R_T$. The key limitation is that updates can only occur after episode termination.

---

## 3. Concept: Taxonomy of Reinforcement Learning Algorithms

### 3.1 High-Level Intuition

**Goal:** Organize the landscape of RL algorithms into coherent categories based on what they learn and how they make decisions.

**Analogy:** Think of learning to play chess. **Value-based** methods learn which board positions are good (then pick moves leading to good positions). **Policy-based** methods directly learn which moves to make in each position. **Model-based** methods learn how the game works (rules and consequences) and plan ahead.

### 3.2 Conceptual Deep Dive

Reinforcement learning algorithms can be categorized along two major axes:

**Model-Free vs. Model-Based:**

- **Model-Free:** Learn directly from experience without modeling environment dynamics
- **Model-Based:** Learn a model of the environment (transition probabilities, rewards) and use it for planning

**Within Model-Free Methods:**

|Category|What is Learned|Examples|
|---|---|---|
|**Value-Based**|State/action value functions ($V$ or $Q$)|SARSA, DQN, Q-Learning|
|**Policy-Based**|Policy directly ($\pi_\theta$)|REINFORCE, Policy Gradient|
|**Actor-Critic (Hybrid)**|Both value function and policy|A2C, A3C, PPO, TRPO, GRPO|

**Key Insight for LLMs:** Modern LLM fine-tuning (ChatGPT, Claude) primarily uses **Actor-Critic** methods, combining:

- **Actor:** The language model (policy) that generates tokens
- **Critic:** A value network that evaluates response quality

Notable algorithms in the LLM space:

- **PPO (Proximal Policy Optimization):** Adopted by OpenAI
- **GRPO (Group Relative Policy Optimization):** Introduced by DeepSeek as a simplified, more efficient alternative

### 3.3 Mathematical Formulation

**Value-Based Objective:** Learn $Q^_(s,a)$ or $V^_(s)$ satisfying Bellman optimality:

$$ V^_(s) = \max_a \left[ R(s,a) + \gamma \sum_{s'} P(s'|s,a) V^_(s') \right] $$

**Policy-Based Objective:** Directly optimize expected return:

$$ J(\theta) = \mathbb{E}_{\tau \sim \pi_\theta} \left[ G(\tau) \right] $$

Where $\tau$ is a trajectory and $G(\tau)$ is the return.

**Actor-Critic Objective:** Combines both—actor optimizes policy gradient, critic estimates value for variance reduction.

### 3.4 Worked Toy Example

**Scenario:** A robot must navigate a 2×2 grid to reach a goal.

|Value-Based (Q-Learning)|Policy-Based|Actor-Critic|
|---|---|---|
|Learns: $Q(\text{cell}, \text{action})$|Learns: $\pi(\text{action}|\text{cell})$|
|Decides: $\arg\max_a Q(s,a)$|Decides: Sample from $\pi$|Actor samples, Critic evaluates|

### 3.5 Connections & Prerequisites

**Prerequisite Refresher (Value Functions):** The state-value function $V^\pi(s) = \mathbb{E}_\pi[G_t | S_t = s]$ represents expected return starting from state $s$ following policy $\pi$. The action-value function $Q^\pi(s,a)$ additionally conditions on taking action $a$ first.

---

## 4. Concept: Policy Gradient and the REINFORCE Algorithm

### 4.1 High-Level Intuition

**Goal:** Instead of learning value functions and deriving policies, directly learn the policy parameters by gradient ascent on expected returns.

**Analogy:** Imagine learning to throw darts. Value-based learning would be like memorizing how good each board position is, then figuring out how to hit good spots. Policy gradient is like directly adjusting your throwing technique based on whether your throws score well—if a technique scores high, do more of it; if it scores low, do less.

### 4.2 Conceptual Deep Dive

**Policy Gradient Methods** parameterize the policy as $\pi_\theta(a|s)$ and optimize parameters $\theta$ to maximize expected returns. The key challenge is computing the gradient of an expectation.

**The Core Problem:**

- We want to maximize $J(\theta) = \mathbb{E}_{\tau \sim \pi_\theta}[G(\tau)]$
- But trajectories $\tau$ depend on $\theta$ through the policy
- Taking gradients through expectations is non-trivial

**The Log-Derivative Trick** resolves this by transforming the gradient of an expectation into an expectation of gradients:

$$ \nabla_\theta \mathbb{E}_{x \sim p_\theta}[f(x)] = \mathbb{E}_{x \sim p_\theta}[f(x) \nabla_\theta \log p_\theta(x)] $$

This is computable via sampling!

**The REINFORCE Algorithm:**

1. Sample trajectories using current policy $\pi_\theta$
2. Compute returns for each trajectory
3. Update policy: increase probability of actions that led to high returns

**Critical Limitation:** After each policy update, old trajectories become stale (generated by old policy) and must be discarded, making REINFORCE **sample inefficient**.

### 4.3 Mathematical Formulation

**Objective Function:**

$$ J(\pi_\theta) = \mathbb{E}_{\tau \sim \pi_\theta}[G(\tau)] $$

**Policy Gradient Theorem:**

$$ \nabla_\theta J(\pi_\theta) = \mathbb{E}_{\tau \sim \pi_\theta} \left[ G(\tau) \cdot \nabla_\theta \log \pi_\theta(\tau) \right] $$

Where:

- $G(\tau)$ is the return of trajectory $\tau$
- $\pi_\theta(\tau) = \prod_t \pi_\theta(a_t|s_t)$ is the probability of trajectory under policy
- $\nabla_\theta \log \pi_\theta(\tau) = \sum_t \nabla_\theta \log \pi_\theta(a_t|s_t)$

**Practical Form (per-timestep):**

$$ \nabla_\theta J \approx \sum_{t=0}^{T} G_t \cdot \nabla_\theta \log \pi_\theta(a_t|s_t) $$

**Log-Derivative Identity (derivation key step):**

$$ \nabla_\theta p_\theta(x) = p_\theta(x) \nabla_\theta \log p_\theta(x) $$

Since $\nabla \log f = \frac{\nabla f}{f}$, therefore $\nabla f = f \cdot \nabla \log f$.

### 4.4 Worked Toy Example

**Setup:** Two-action MDP. State $s$, actions ${L, R}$. Softmax policy:

$$\pi_\theta(L|s) = \frac{e^{\theta_L}}{e^{\theta_L} + e^{\theta_R}}, \quad \pi_\theta(R|s) = \frac{e^{\theta_R}}{e^{\theta_L} + e^{\theta_R}}$$

Initial: $\theta_L = 0, \theta_R = 0$ → $\pi(L) = \pi(R) = 0.5$

**Episode 1:** Action $L$ taken, return $G = 10$

$$\nabla_\theta \log \pi_\theta(L) = \begin{bmatrix} 1 - \pi(L) \ -\pi(R) \end{bmatrix} = \begin{bmatrix} 0.5 \ -0.5 \end{bmatrix}$$

Update ($\alpha = 0.1$): $$\theta \leftarrow \theta + \alpha \cdot G \cdot \nabla \log \pi = \begin{bmatrix} 0 \ 0 \end{bmatrix} + 0.1 \cdot 10 \cdot \begin{bmatrix} 0.5 \ -0.5 \end{bmatrix} = \begin{bmatrix} 0.5 \ -0.5 \end{bmatrix}$$

New policy: $\pi(L) = \frac{e^{0.5}}{e^{0.5} + e^{-0.5}} \approx 0.73$

The policy now favors action $L$ because it led to a high return!

### 4.5 Connections & Prerequisites

**Prerequisite Refresher (Stochastic Gradient Ascent):** Gradient ascent updates parameters in the direction of increasing objective: $\theta \leftarrow \theta + \alpha \nabla_\theta J(\theta)$. When the true gradient is unavailable, we use sample estimates (stochastic gradient), which are unbiased estimators of the true gradient.

---

## 5. Concept: LLM Fine-Tuning Pipeline

### 5.1 High-Level Intuition

**Goal:** Understand the four-stage pipeline that transforms a raw language model into a helpful, aligned assistant.

**Analogy:** Training an LLM is like educating a person. **Pre-training** is like general education (reading everything). **SFT** is like professional training (learning to answer questions). **Preference fine-tuning** is like feedback from customers (learning what people prefer). **Reasoning fine-tuning** is like specialized problem-solving training (learning to solve math).

### 5.2 Conceptual Deep Dive

Modern LLMs undergo a **four-stage training pipeline**:

**Stage 1: Pre-training**

- Train on trillions of tokens (web text, books, code)
- Objective: Next-token prediction
- Result: A model that can generate coherent text
- Cost: Extremely expensive (millions of dollars)

**Stage 2: Supervised Fine-Tuning (SFT)**

- Train on (instruction, response) pairs
- Objective: Learn to follow instructions and answer questions
- Result: A model that responds helpfully to prompts
- This is the boundary between supervised learning and RL

**Stage 3: Preference Fine-Tuning (RLHF)**

- Use human preferences to improve response quality
- Train a reward model on human comparisons
- Use RL (PPO) to optimize for preferred responses
- Result: Responses that humans prefer (helpful, harmless, honest)

**Stage 4: Reasoning Fine-Tuning**

- Train on reasoning tasks (math, logic, coding)
- Datasets: GSM8K, MATH, coding benchmarks
- Reward: Correctness of final answer
- Result: Improved problem-solving capabilities

### 5.3 Mathematical Formulation

**LLM as MDP:**

- **State $S_t$:** Prompt + all tokens generated so far
- **Action $A_t$:** Selecting the next token from vocabulary $\mathcal{V}$ (e.g., $|\mathcal{V}| \approx 100,000$)
- **Transition:** Deterministic! $S_{t+1} = S_t \oplus A_t$ (concatenation)
- **Policy $\pi_\theta$:** The language model's output distribution over tokens
- **Reward:** Typically sparse—$R = 1$ if final answer correct, $R = 0$ otherwise

$$ S_0 = \text{prompt} \xrightarrow{A_0 \sim \pi_\theta} S_1 = S_0 \oplus A_0 \xrightarrow{A_1 \sim \pi_\theta} \cdots \xrightarrow{} S_T \text{ (terminal)} $$

### 5.4 Worked Toy Example

**GSM8K-style Problem:**

_"Janet has 3 apples. She buys 2 more. How many apples does she have?"_

**Trajectory:**

- $S_0$: "Janet has 3 apples. She buys 2 more. How many apples does she have?"
- $A_0$: "Janet"
- $S_1$: $S_0$ + "Janet"
- $A_1$: "starts"
- ... (many tokens)
- $A_T$: "5"
- $S_T$: Full response ending with "5"

**Reward:** Since 3 + 2 = 5 is correct: $R = 1$

This single reward at termination must be used to update all the policy parameters that generated the entire response.

### 5.5 Connections & Prerequisites

**Prerequisite Refresher (MDP Framework):** A Markov Decision Process is defined by $(S, A, P, R, \gamma)$ where $P(s'|s,a)$ is the transition probability and $R(s,a)$ is the reward. The Markov property states that $P(S_{t+1}|S_t, A_t) = P(S_{t+1}|S_0, A_0, ..., S_t, A_t)$—the future depends only on the present state.

---

## 6. Concept: Baselines and the Advantage Function

### 6.1 High-Level Intuition

**Goal:** Reduce variance in policy gradient estimates by measuring returns relative to a baseline, rather than using absolute returns.

**Analogy:** Consider grading students. Student A consistently scores 80-90%. Student B typically scores 50-60%. If Student B suddenly scores 75%, they've shown remarkable improvement relative to their baseline. Rewarding based on improvement (relative to baseline) rather than absolute score is fairer and provides better learning signals.

### 6.2 Conceptual Deep Dive

**The Variance Problem:** Raw REINFORCE has high variance because:

- Good actions in bad episodes get penalized (low total return)
- Bad actions in good episodes get reinforced (high total return)

**Solution: Subtract a Baseline**

Instead of reinforcing actions proportional to return $G_t$, reinforce proportional to $(G_t - b)$ where $b$ is a baseline. Mathematically, this doesn't change the expected gradient (baseline is independent of action), but dramatically reduces variance.

**The Advantage Function:**

The natural choice of baseline is the **value function** $V(s)$, representing the expected return from state $s$. The difference between actual return and expected return is called the **advantage**:

$$A_t = G_t - V(S_t)$$

Intuition:

- $A_t > 0$: Action performed **better than expected** → increase probability
- $A_t < 0$: Action performed **worse than expected** → decrease probability
- $A_t = 0$: Action performed **as expected** → no change

**Actor-Critic Architecture:**

- **Actor:** Policy network $\pi_\theta$ that selects actions
- **Critic:** Value network $V_\phi$ that estimates expected returns

Both networks are trained simultaneously:

- Critic learns to predict returns (regression)
- Actor uses advantage for policy gradient (with reduced variance)

### 6.3 Mathematical Formulation

**Advantage Function:**

$$ A_t = R_t - V_\phi(S_t) $$

Or more generally (using TD-style estimates):

$$ A_t = R_{t+1} + \gamma V_\phi(S_{t+1}) - V_\phi(S_t) $$

**Actor Loss (Policy Gradient with Advantage):**

$$ L^{\text{actor}}(\theta) = \sum_{t=0}^{T} \log \pi_\theta(A_t|S_t) \cdot A_t $$

**Critic Loss (Value Function Regression):**

$$ L^{\text{critic}}(\phi) = \sum_{t=0}^{T} \left( V_\phi(S_t) - G_t \right)^2 $$

Where:

- $\theta$ are actor (policy) parameters
- $\phi$ are critic (value) parameters
- $G_t$ is the actual return (training target for critic)

### 6.4 Worked Toy Example

**Setup:** Agent in state $S$ with $V_\phi(S) = 10$ (critic's prediction).

**Episode 1:** Takes action $A$, receives return $G = 15$

- Advantage: $A = 15 - 10 = +5$
- Interpretation: Action was **better than expected**
- Actor update: **Increase** $\pi_\theta(A|S)$

**Episode 2:** Takes action $A'$, receives return $G = 7$

- Advantage: $A = 7 - 10 = -3$
- Interpretation: Action was **worse than expected**
- Actor update: **Decrease** $\pi_\theta(A'|S)$

**Critic Update (after both episodes):**

- Observed returns: 15, 7
- Current prediction: 10
- New prediction should move toward average (≈11)

### 6.5 Connections & Prerequisites

**Prerequisite Refresher (Variance Reduction):** In statistics, variance measures the spread of estimates. High variance means estimates fluctuate wildly between samples. Control variates reduce variance by subtracting a correlated baseline: $\text{Var}(X - b) < \text{Var}(X)$ when $b$ correlates with $X$ but has known expectation.

---

## 7. Concept: Proximal Policy Optimization (PPO)

### 7.1 High-Level Intuition

**Goal:** Enable sample-efficient policy gradient learning by reusing trajectories across multiple update epochs, while preventing destructive large policy updates.

**Analogy:** Imagine learning a new skill from video tutorials. REINFORCE is like watching each tutorial once, practicing once, then deleting it forever (wasteful). PPO is like rewatching and practicing with the same tutorials multiple times—but being careful not to "overfit" to any single tutorial by limiting how much your technique changes per session.

### 7.2 Conceptual Deep Dive

**The Sample Efficiency Problem:**

Basic policy gradient (REINFORCE) suffers from poor sample efficiency:

- Generate trajectories with policy $\pi_{\theta_\text{old}}$
- Update policy once: $\theta_\text{old} \to \theta_\text{new}$
- Trajectories are now "stale" (generated by wrong policy)
- Must discard all data and regenerate!

**Solution Part 1: Importance Sampling**

Importance sampling allows us to compute expectations under one distribution using samples from another:

$$ \mathbb{E}_{x \sim p}[f(x)] = \mathbb{E}_{x \sim q}\left[\frac{p(x)}{q(x)} f(x)\right] $$

For PPO, this means we can reuse trajectories from $\pi_{\theta_\text{old}}$ to estimate gradients for $\pi_\theta$:

$$ \nabla_\theta J \approx \mathbb{E}_{\tau \sim \pi_{\theta_\text{old}}} \left[ \frac{\pi_\theta(a|s)}{\pi_{\theta_\text{old}}(a|s)} A(s,a) \right] $$

The ratio $r_t(\theta) = \frac{\pi_\theta(a_t|s_t)}{\pi_{\theta_\text{old}}(a_t|s_t)}$ measures how much the policy has changed.

**Solution Part 2: Clipping**

Importance sampling can be unstable when policies diverge too much (large ratios). PPO introduces a **clipped objective** that constrains policy changes:

$$ L^{\text{CLIP}}(\theta) = \mathbb{E}_t \left[ \min\left( r_t(\theta) A_t, \text{clip}(r_t(\theta), 1-\epsilon, 1+\epsilon) A_t \right) \right] $$

Where $\epsilon \approx 0.2$ is a hyperparameter.

**Effect of Clipping:**

- If $A_t > 0$ (good action): Don't let $r_t$ exceed $1+\epsilon$ (don't increase probability too much)
- If $A_t < 0$ (bad action): Don't let $r_t$ fall below $1-\epsilon$ (don't decrease probability too much)

This creates a "trust region" where policy updates are safe.

### 7.3 Mathematical Formulation

**Probability Ratio:**

$$ r_t(\theta) = \frac{\pi_\theta(a_t|s_t)}{\pi_{\theta_\text{old}}(a_t|s_t)} $$

**PPO Clipped Objective:**

$$ L^{\text{CLIP}}(\theta) = \mathbb{E}_t \left[ \min\left( r_t(\theta) A_t, \text{clip}(r_t(\theta), 1-\epsilon, 1+\epsilon) A_t \right) \right] $$

Where:

- $r_t(\theta)$ is the probability ratio (how much policy changed)
- $A_t$ is the advantage estimate
- $\epsilon$ is the clipping parameter (typically 0.1-0.2)
- $\text{clip}(x, a, b) = \max(a, \min(x, b))$

**Clipping Constraint:**

$$ 1 - \epsilon \leq r_t(\theta) \leq 1 + \epsilon $$

**Combined PPO Loss:**

$$ L(\theta, \phi) = L^{\text{CLIP}}(\theta) - c_1 L^{\text{critic}}(\phi) + c_2 H[\pi_\theta] $$

Where $H[\pi_\theta]$ is an entropy bonus encouraging exploration.

### 7.4 Worked Toy Example

**Setup:** $\epsilon = 0.2$, so valid ratio range is $[0.8, 1.2]$

**Scenario 1:** Good action ($A_t = +5$), policy increased ($r_t = 1.5$)

- Unclipped: $1.5 \times 5 = 7.5$
- Clipped: $\text{clip}(1.5, 0.8, 1.2) \times 5 = 1.2 \times 5 = 6.0$
- PPO uses: $\min(7.5, 6.0) = 6.0$ ← Clipping limits over-reinforcement

**Scenario 2:** Bad action ($A_t = -3$), policy decreased ($r_t = 0.6$)

- Unclipped: $0.6 \times (-3) = -1.8$
- Clipped: $\text{clip}(0.6, 0.8, 1.2) \times (-3) = 0.8 \times (-3) = -2.4$
- PPO uses: $\min(-1.8, -2.4) = -2.4$ ← Takes the worse (more negative) value

**Scenario 3:** Within trust region ($r_t = 1.1$, $A_t = +2$)

- Unclipped: $1.1 \times 2 = 2.2$
- Clipped: $\text{clip}(1.1, 0.8, 1.2) \times 2 = 1.1 \times 2 = 2.2$
- PPO uses: $\min(2.2, 2.2) = 2.2$ ← No clipping needed

### 7.5 Connections & Prerequisites

**Prerequisite Refresher (Importance Sampling):** Importance sampling is a technique to estimate expectations under distribution $p$ using samples from distribution $q$: $\mathbb{E}_p[f(X)] = \mathbb{E}_q[\frac{p(X)}{q(X)}f(X)]$. The ratio $\frac{p(x)}{q(x)}$ reweights samples to correct for the distributional mismatch.

---

## 8. Concept: PPO Algorithm — Complete Pseudocode

### 8.1 High-Level Intuition

**Goal:** Put all the pieces together into a complete, practical algorithm for training policies with actor-critic methods.

**Analogy:** PPO is like a well-organized study routine: gather material (collect trajectories), review multiple times (multiple epochs), but don't cram too much at once (clipping prevents overfitting to any single batch).

### 8.2 Conceptual Deep Dive

The **PPO algorithm** combines:

1. **Trajectory Collection:** Roll out the current policy to gather experience
2. **Advantage Estimation:** Use the critic to compute advantages
3. **Multi-Epoch Updates:** Reuse trajectories across several gradient steps
4. **Clipped Objective:** Prevent destructive policy updates
5. **Critic Training:** Improve value estimates for better advantages

**Key Implementation Details:**

- **Generalized Advantage Estimation (GAE):** Often used instead of raw advantages for smoother estimates
- **Multiple Epochs:** Typically 3-10 epochs per batch of trajectories
- **Mini-batching:** Trajectories are divided into mini-batches for each gradient step
- **Entropy Bonus:** Added to encourage exploration and prevent premature convergence

### 8.3 Mathematical Formulation

**PPO Algorithm Pseudocode:**

```
Initialize policy parameters θ
Initialize value function parameters φ

for iteration = 1, 2, ... do
    // Step 1: Collect trajectories
    Collect set of trajectories {τ_i} by running π_θ
    
    // Step 2: Compute advantages
    Compute discounted returns G_t for all timesteps
    Compute advantages A_t = G_t - V_φ(S_t)
    
    // Step 3: Multi-epoch updates (KEY DIFFERENCE FROM REINFORCE)
    for epoch = 1, ..., K do
        for each mini-batch do
            // Actor update
            Compute ratio r_t(θ) = π_θ(a_t|s_t) / π_θ_old(a_t|s_t)
            Compute clipped objective L^CLIP
            Update θ using ∇_θ L^CLIP
            
            // Critic update
            Compute critic loss L^critic = (V_φ(s_t) - G_t)²
            Update φ using ∇_φ L^critic
        end for
    end for
    
    θ_old ← θ  // Update old policy for next iteration
end for
```

**Practical Hyperparameters:**

- Clipping parameter $\epsilon$: 0.1-0.2
- Number of epochs $K$: 3-10
- Learning rate: $3 \times 10^{-4}$ (often with decay)
- GAE parameter $\lambda$: 0.95
- Discount factor $\gamma$: 0.99

### 8.4 Worked Toy Example

**Mini PPO Training Loop (Conceptual):**

**Iteration 1:**

1. Collect 100 trajectories using $\pi_{\theta_0}$
2. Compute advantages using $V_{\phi_0}$
3. Run 4 epochs of updates:
    - Epoch 1: $\theta_0 \to \theta_1$, $\phi_0 \to \phi_1$
    - Epoch 2: $\theta_1 \to \theta_2$, $\phi_1 \to \phi_2$ (same data!)
    - Epoch 3: $\theta_2 \to \theta_3$, $\phi_2 \to \phi_3$ (same data!)
    - Epoch 4: $\theta_3 \to \theta_4$, $\phi_3 \to \phi_4$ (same data!)
4. Set $\theta_{\text{old}} = \theta_4$ for next iteration

**Iteration 2:**

1. Collect NEW 100 trajectories using $\pi_{\theta_4}$
2. Repeat...

**Sample Efficiency Gain:** REINFORCE would need 400 trajectory collections for the same number of gradient steps!

### 8.5 Connections & Prerequisites

**Prerequisite Refresher (TRPO):** Trust Region Policy Optimization preceded PPO and used a hard KL-divergence constraint: $D_{KL}(\pi_{\theta}||\pi_{\theta_{old}}) \leq \delta$. This required complex second-order optimization. PPO achieves similar stability through the simpler clipping mechanism, making it more practical for large-scale applications.

---

### Key Takeaways & Formulas

- **TD(λ) interpolates** between TD(0) ($\lambda=0$) and Monte Carlo ($\lambda=1$) using exponentially weighted n-step returns: $G_t^{\lambda} = (1-\lambda)\sum_{n=1}^{\infty}\lambda^{n-1}G_t^{(n)}$
    
- **Policy Gradient Theorem** enables direct policy optimization: $\nabla_\theta J = \mathbb{E}[\sum_t G_t \nabla_\theta \log \pi_\theta(a_t|s_t)]$
    
- **The Advantage Function** $A_t = G_t - V(S_t)$ measures performance relative to baseline, reducing variance while preserving unbiased gradients
    
- **PPO's Clipped Objective** enables sample-efficient learning by reusing trajectories: $L^{CLIP} = \mathbb{E}[\min(r_t A_t, \text{clip}(r_t, 1-\epsilon, 1+\epsilon)A_t)]$
    
- **LLM Fine-Tuning Pipeline:** Pre-training → SFT → Preference Fine-Tuning (RLHF) → Reasoning Fine-Tuning, with RL methods (especially PPO/GRPO) driving the preference and reasoning stages
    

---

### Quick Reference: Algorithm Comparison

|Algorithm|Category|Sample Efficiency|Complexity|LLM Usage|
|---|---|---|---|---|
|REINFORCE|Policy Gradient|Low|Low|Baseline|
|Actor-Critic|Hybrid|Medium|Medium|Foundation|
|PPO|Hybrid + Clipping|High|Medium|OpenAI (ChatGPT)|
|GRPO|Simplified PPO|High|Low|DeepSeek|

---

### Recommended Further Study

1. **Section 2.3.1** of "Foundations of Deep Reinforcement Learning" (Python RL Book) for complete log-derivative trick derivation
2. **Hugging Face TRL Library** for practical PPO implementation with small language models
3. **Original PPO Paper:** "Proximal Policy Optimization Algorithms" (Schulman et al., 2017)