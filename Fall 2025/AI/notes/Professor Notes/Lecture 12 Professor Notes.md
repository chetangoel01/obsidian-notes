## 1. Markov Decision Processes (MDPs) - Foundations

### 1.1 What is an MDP?

A Markov Decision Process is a mathematical framework for modeling sequential decision-making problems where outcomes are partly random and partly under the control of a decision-maker (agent).

**The MDP Tuple**: An MDP is defined by five components:

$$\mathcal{M} = (\mathcal{S}, \mathcal{P}, \mathcal{R}, \mathcal{A}, \gamma)$$

Where:

- **𝒮 (State Space)**: The set of all possible states the environment can be in
- **𝒫 (Transition Model)**: Probability of transitioning between states given actions
- **ℛ (Reward Function)**: The set of rewards or reward function
- **𝒜 (Action Space)**: The set of all possible actions the agent can take
- **γ (Discount Factor)**: A value between 0 and 1 that discounts future rewards

### 1.2 The Agent-Environment Interaction Loop

The professor emphasized this core interaction cycle:

1. Agent is in state **S**
2. Agent decides to act with action **A** (based on policy π)
3. Environment transitions to new state **S'**
4. Agent receives reward **R**
5. Process repeats

This interaction generates a sequence of **experiences** that are encoded into probability distributions.

### 1.3 The Transition Model

The transition model P(s'|s, a) describes the probability of ending up in state s' given that you were in state s and took action a.

**Key Properties**:

- The environment is **stochastic** - same action from same state may lead to different outcomes
- The Markov property: The future depends only on the current state, not the history

**Example (Grid World)**: If you're in a grid cell and choose to go "up":

- With probability 0.8, you actually go up
- With probability 0.1, you slip left
- With probability 0.1, you slip right

This is captured in transition probability tables - one for each action.

### 1.4 The Reward Function

The reward function can take two forms:

**Two-parameter version**: R(s, a)

- Expected reward received after executing action a from state s

**Three-parameter version**: R(s, a, s')

- Expected reward received after executing action a from state s and landing in state s'

**Mathematical Definition**: $$R(s, a) = \mathbb{E}[R_t | S_t = s, A_t = a]$$

### 1.5 The Policy

A **policy π** is the agent's strategy - it defines which action to take in each state.

**Stochastic Policy**: π(a|s) = P(A_t = a | S_t = s)

- Gives a probability distribution over actions for each state

**Deterministic Policy**: Directly maps states to actions

- π(s) = a

**Important Clarification from Lecture**: A stochastic policy does NOT make the MDP stochastic. The stochasticity of the MDP comes from the transition model itself.

### 1.6 Episodes and Trajectories

**Episode**: A complete interaction sequence from start to termination

- From t = 0 to t = T-1

**Trajectory (τ)**: The sequence of experiences over an episode $$\tau = (S_0, A_0, R_1, S_1, A_1, R_2, ..., S_T)$$

**Termination** occurs when:

- Agent reaches a terminal state
- Agent decides to terminate
- Running out of time (finite horizon problems)

### 1.7 The Return

The **return G_t** is the cumulative discounted reward from time t onwards:

$$G_t = R_{t+1} + \gamma R_{t+2} + \gamma^2 R_{t+3} + ... = \sum_{k=0}^{\infty} \gamma^k R_{t+k+1}$$

**Why discount?**

- Mathematical convenience (ensures finite returns for infinite horizons)
- Represents uncertainty about the future
- Models preference for immediate rewards
- Common values: γ = 0.9 or γ = 0.99

---

## 2. Value Functions

Value functions are the **objective functions** we optimize in MDPs. They estimate "how good" it is to be in a given state or to take a given action.

### 2.1 State Value Function V(s)

The **state value function** V^π(s) answers: "What is the expected return starting from state s and following policy π?"

$$V^\pi(s) = \mathbb{E}_\pi[G_t | S_t = s]$$

**Expanded form**: $$V^\pi(s) = \mathbb{E}_\pi[R_{t+1} + \gamma R_{t+2} + \gamma^2 R_{t+3} + ... | S_t = s]$$

**Intuition**: Think of a grid world where you need to exit a room. States closer to the door have higher value because you expect to receive rewards sooner.

**Visual Representation**: In a grid world, the value function is a matrix where every state/cell has a value number.

### 2.2 State-Action Value Function Q(s, a)

The **state-action value function** (Q-function) Q^π(s, a) answers: "What is the expected return starting from state s, taking action a, and then following policy π?"

$$Q^\pi(s, a) = \mathbb{E}_\pi[G_t | S_t = s, A_t = a]$$

**Why Q is "better" than V for acting optimally**: The Q function directly tells us the value of each action, making it easier to select optimal actions. With V alone, you need to know the transition model to compare action values.

---

## 3. Bellman Equations

The Bellman equations are fundamental recursive relationships that connect the value of a state to the values of successor states. They are named after Richard Bellman, a pioneer in dynamic programming and optimal control.

### 3.1 Bellman Expectation Equation for V

This equation connects the value of the current state to immediate reward plus discounted value of next states:

$$V^\pi(s) = \mathbb{E}_\pi[R_{t+1} + \gamma V^\pi(S_{t+1}) | S_t = s]$$

**Expanded form with explicit sums**: $$V^\pi(s) = \sum_a \pi(a|s) \sum_{s'} P(s'|s,a)[R(s,a,s') + \gamma V^\pi(s')]$$

**Interpretation**: The value of a state equals the expected immediate reward plus the discounted expected value of the next state.

### 3.2 The Backup Tree Concept

The professor emphasized understanding backup trees:

```
        (s)           ← Starting state
       / | \
      a1 a2 a3        ← Possible actions (determined by π)
     /|\ |  |\ 
   s' s'' s'''        ← Possible next states (determined by P)
```

- **Solid lines** (from state to action): Determined by policy π(a|s)
- **Dashed lines** (from action to next state): Determined by transition model P(s'|s,a)
- Each next state has its own value V(s')
- We "back up" values from successor states to compute current state value

### 3.3 Bellman Expectation Equation for Q

$$Q^\pi(s,a) = \mathbb{E}_\pi[R_{t+1} + \gamma Q^\pi(S_{t+1}, A_{t+1}) | S_t = s, A_t = a]$$

**Expanded**: $$Q^\pi(s,a) = \sum_{s'} P(s'|s,a)[R(s,a,s') + \gamma \sum_{a'} \pi(a'|s') Q^\pi(s',a')]$$

### 3.4 Bellman Optimality Equations (Nonlinear!)

When we want to find the **optimal** value function (V* or Q*), we introduce the **max** operator, which makes the equations **nonlinear**.

**For V***:
$$
V^*(s) = \max_{a} \sum_{s'} P(s' \mid s,a) \bigl[ R(s,a,s') + \gamma V^*(s') \bigr]
$$


**For Q***: 
$$
Q^*(s,a) = \sum_{s'} P(s' \mid s,a) \bigl[ R(s,a,s') + \gamma \max_{a'} Q^*(s',a') \bigr]
$$


**Why nonlinear?** The max operator is nonlinear - we can't solve these with simple matrix inversion like the expectation equations. We need iterative methods like **policy iteration** or **value iteration**.

### 3.5 Solving Bellman Equations

**Linear Case (Bellman Expectation)**:

- Can be written as: **V = R + γPV**
- Rearranges to: **(I - γP)V = R**
- Solution: **V = (I - γP)^(-1) R**

**However**, the professor strongly warned against matrix inversion in practice:

> "Every time you see matrix inversion, don't implement it - it will more likely blow up."

Instead, use **iterative methods** based on the fact that the Bellman operator is a **contraction**.

### 3.6 The Bellman Operator as a Contraction

A **contraction** mapping has the property that applying it repeatedly converges to a fixed point, regardless of starting point.

**Simple scalar example**: $$x_{k+1} = \gamma x_k + c$$

For γ < 1, this converges to x* = c/(1-γ)

**The Bellman operator is a contraction** with factor γ: $$V_{k+1}(s) = \sum_a \pi(a|s) \sum_{s'} P(s'|s,a)[R(s,a,s') + \gamma V_k(s')]$$

This iterative method is called **Iterative Policy Evaluation**.

---

## 4. Prediction vs Control Problems

### 4.1 Prediction Problem (Policy Evaluation)

**Goal**: Given a policy π, calculate V^π(s) for all states

**Question**: "How good is this policy?"

**Method**: Use Bellman Expectation equations iteratively

**Example**: Given a uniform random policy (equal probability for all actions), what is the value of each state?

### 4.2 Control Problem

**Goal**: Find the optimal policy π* that maximizes value

**Question**: "What is the best way to act?"

**Involves two sub-problems**:

1. Policy Evaluation: Evaluate current policy
2. Policy Improvement: Use greedy action selection to improve policy

**Methods**: Policy Iteration, Value Iteration

---

## 5. Policy Iteration

Policy Iteration is the key algorithm for solving MDPs (finding π*). It alternates between two steps until convergence.

### 5.1 The Algorithm

```
1. Initialize π arbitrarily (e.g., uniform random)
2. Repeat until convergence:
   a. Policy Evaluation: Compute V^π for current policy
   b. Policy Improvement: For each state, update policy to be greedy w.r.t. V^π
      π'(s) = argmax_a Σ_{s'} P(s'|s,a)[R(s,a,s') + γV^π(s')]
3. Return π* (optimal policy)
```

### 5.2 Policy Evaluation (Step 2a)

Use iterative policy evaluation:

$$V_{k+1}(s) = \sum_a \pi(a|s) \sum_{s'} P(s'|s,a)[R(s,a,s') + \gamma V_k(s')]$$

Repeat until V converges (change is below threshold).

### 5.3 Policy Improvement (Step 2b)

**Greedy Policy Improvement**: For each state, select the action that maximizes expected value:

$$\pi'(s) = \arg\max_a \sum_{s'} P(s'|s,a)[R(s,a,s') + \gamma V^\pi(s')]$$

This **eliminates suboptimal actions** from the policy.

### 5.4 Convergence Guarantee

The professor emphasized a key insight:

> "The optimal policy may converge much sooner than the value function. Despite the fact that the value function may change from one iteration to another, the relative benefits or relative values between adjacent states that drive the decision-making do not change, and therefore the optimal policy may have converged much sooner."

### 5.5 Grid World Example

**Setup**: 4x4 grid, two terminal states (corners), reward of -1 for each step

**Iteration 0**:

- Initialize V = 0 for all states
- Policy = uniform random (0.25 probability for each direction)

**Iteration 1**:

- Evaluate policy → Get new V values
- Act greedily → Arrows now point toward terminal states

**Iteration 2+**:

- Continue until policy no longer changes
- Final result: Optimal arrows in each cell pointing toward shortest path to terminal

---

## 6. Reinforcement Learning Fundamentals

### 6.1 The Key Difference: Model-Free Learning

In **MDP** (what we covered earlier):

- We **know** the transition model P(s'|s,a)
- We **know** the reward function R(s,a,s')
- We can compute value functions exactly

In **Reinforcement Learning**:

- We **do NOT know** the transition model
- We **do NOT know** the reward function
- We must **learn** from experience (interactions with environment)

### 6.2 The Backup Tree Comparison

The professor drew a clear distinction using backup trees:

**Dynamic Programming (Full Backup)**:

```
        (s)
      /  |  \
    a1  a2  a3      ← Know all actions
   /|\  |   /|\
 s' s'' s'''        ← Know all possible next states and their probabilities
```

- **Breadth-first expansion**
- Know exactly all states we can transition to
- Know exactly all rewards
- Full backup from all successor states

**Monte Carlo / Reinforcement Learning (Sample Backup)**:

```
   (s)
    |
    a1                ← Take ONE action
    |
   s'                 ← Observe ONE next state
    |
    a2
    |
   s''
    ...
    |
 Terminal            ← Complete trajectory
```

- **Depth-first sampling**
- Environment is a **black box**
- We just take actions and observe what happens
- Generate complete trajectories (episodes)

### 6.3 The Learning Paradigm

In RL, we:

1. Interact with the environment
2. Collect experiences (s, a, r, s')
3. Learn value functions from these experiences
4. Use estimated values to improve policy

---

## 7. Monte Carlo Methods

### 7.1 Core Idea

Monte Carlo (MC) methods learn from **complete episodes**. We:

1. Generate many trajectories by interacting with the environment
2. Observe the actual returns (rewards) received
3. Estimate value as the **sample mean** of observed returns

### 7.2 The Monte Carlo Value Update

**Basic idea**: Visit a state multiple times, track returns, compute average

For each state s visited in an episode:

1. Increment visit counter: N(s) = N(s) + 1
2. Update total return: G_total(s) = G_total(s) + G_t
3. Estimate value: V(s) = G_total(s) / N(s)

**Incremental Sample Mean Form** (connects to Kalman filter lecture!):

$$V(s) \leftarrow V(s) + \alpha [G_t - V(s)]$$

Where:

- α = 1/N(s) (or a constant learning rate)
- G_t = actual return observed from state s in this episode
- [G_t - V(s)] = "error" or "surprise" - difference between observed and expected

### 7.3 The Monte Carlo Equation

$$V^\pi(S_t) = V^\pi(S_t) + \alpha [G_t - V^\pi(S_t)]$$

**Interpretation**:

- New estimate = Old estimate + learning_rate × (target - old estimate)
- Target is the **actual return G_t** observed
- This is exactly the incremental sample mean we studied in Kalman filters!

### 7.4 Advantages of Monte Carlo

- Simple and intuitive
- No bias (uses actual returns, not estimates)
- Works well with episodic tasks
- Does not require knowledge of environment dynamics

### 7.5 Major Limitation of Monte Carlo

**You must wait until the episode terminates to learn!**

The professor used a driving analogy:

> "It's okay to wait until the car is crashing, and then go back and say 'okay, I now have an estimate of the value of not going 100 miles per hour in a corner.' In reality, I prefer schemes that allow me to act as I go."

This limitation leads us to **Temporal Difference methods**.

---

## 8. Temporal Difference Learning

### 8.1 TD(0) - One-Step Temporal Difference

**Core Innovation**: Don't wait for episode end - bootstrap from estimated values!

**TD(0) Update Equation**: $$V(S_t) \leftarrow V(S_t) + \alpha [R_{t+1} + \gamma V(S_{t+1}) - V(S_t)]$$

**The TD Target**: $$\text{TD Target} = R_{t+1} + \gamma V(S_{t+1})$$

**The TD Error (δ)**: $$\delta_t = R_{t+1} + \gamma V(S_{t+1}) - V(S_t)$$

### 8.2 Comparing MC and TD(0)

|Aspect|Monte Carlo|TD(0)|
|---|---|---|
|**Target**|Actual return G_t|R_{t+1} + γV(S_{t+1})|
|**Must wait for**|Episode end|Next time step|
|**Bias**|Unbiased|Biased (uses estimates)|
|**Variance**|High|Lower|
|**Bootstrapping**|No|Yes|

### 8.3 The Secret of TD(0)

The professor emphasized this key insight:

> "TD(0) combines the **bootstrapping** of dynamic programming with the **sampling** of Monte Carlo."

- **From DP**: We use estimated values of successor states (bootstrapping)
- **From MC**: We sample trajectories through the environment

### 8.4 The Brooklyn-to-Jersey Analogy

The professor gave this memorable analogy:

> "If I am in Brooklyn and want to go back to Jersey, I have two options - Brooklyn Bridge or Manhattan Bridge."
> 
> "**TD(0) says**: The moment you cross the Manhattan Bridge, go ahead and update the value of Brooklyn."
> 
> "**TD(λ) says**: Wait until you exit the Holland Tunnel and reach New Jersey before you update Brooklyn, because you never know what is waiting for you in Canal Street."

### 8.5 N-Step TD Returns

Instead of bootstrapping after 1 step, we can wait n steps:

**n-step Return**: $$G_t^{(n)} = R_{t+1} + \gamma R_{t+2} + ... + \gamma^{n-1} R_{t+n} + \gamma^n V(S_{t+n})$$

**n-step TD Update**: $$V(S_t) \leftarrow V(S_t) + \alpha [G_t^{(n)} - V(S_t)]$$

**Special Cases**:

- n = 1: TD(0)
- n = ∞: Monte Carlo (full return)

### 8.6 TD(λ) - Combining All N-Step Returns

TD(λ) takes a weighted average of ALL n-step returns:

**The λ-return**: $$G_t^\lambda = (1-\lambda) \sum_{n=1}^{\infty} \lambda^{n-1} G_t^{(n)}$$

**Properties**:

- **λ = 0**: Reduces to TD(0) (one-step bootstrap)
- **λ = 1**: Reduces to Monte Carlo (full return)
- **0 < λ < 1**: Weighted combination

**Exponential Weighting**: The weights decay exponentially:

- G_t^(1) gets weight (1-λ)
- G_t^(2) gets weight (1-λ)λ
- G_t^(3) gets weight (1-λ)λ²
- And so on...

**TD(λ) Update**: $$V_{t+n}(S_t) = V_{t+n-1}(S_t) + \alpha [G_t^\lambda - V_{t+n-1}(S_t)]$$

**Key Benefit**: TD(λ) allows **decoupling** the time that actions are taken from when value function updates are done.

---

## 9. Key Equations Summary

### MDP Definitions

$$G_t = \sum_{k=0}^{\infty} \gamma^k R_{t+k+1}$$

$$V^\pi(s) = \mathbb{E}_\pi[G_t | S_t = s]$$

$$Q^\pi(s,a) = \mathbb{E}_\pi[G_t | S_t = s, A_t = a]$$

### Bellman Expectation Equations

$$V^\pi(s) = \sum_a \pi(a|s) \sum_{s'} P(s'|s,a)[R(s,a,s') + \gamma V^\pi(s')]$$

$$Q^\pi(s,a) = \sum_{s'} P(s'|s,a)[R(s,a,s') + \gamma \sum_{a'} \pi(a'|s') Q^\pi(s',a')]$$

### Bellman Optimality Equations
$$
V^*(s) = \max_{a} \sum_{s'} P(s' \mid s,a) \bigl[ R(s,a,s') + \gamma V^*(s') \bigr]
$$

$$
Q^*(s,a) = \sum_{s'} P(s' \mid s,a) \bigl[ R(s,a,s') + \gamma \max_{a'} Q^*(s',a') \bigr]
$$

### Monte Carlo Update

$$V(S_t) \leftarrow V(S_t) + \alpha [G_t - V(S_t)]$$

### TD(0) Update

$$V(S_t) \leftarrow V(S_t) + \alpha [R_{t+1} + \gamma V(S_{t+1}) - V(S_t)]$$

### N-Step Return

$$G_t^{(n)} = R_{t+1} + \gamma R_{t+2} + ... + \gamma^{n-1} R_{t+n} + \gamma^n V(S_{t+n})$$

### TD(λ) Return

$$G_t^\lambda = (1-\lambda) \sum_{n=1}^{\infty} \lambda^{n-1} G_t^{(n)}$$

---

## 10. Grid World Examples

### 10.1 Simple Two-State Example (from lecture)

**Setup**:

- Two states: S1 and S2
- From S1, taking action leads to S2 with reward +2
- From S2, taking action leads to S1 with reward 0
- Deterministic transitions (probability 1)

**Transition Matrix**:

```
P = [0 1]    (Identity transpose - go from one state to the other)
    [1 0]
```

**Reward Vector**: R = [2, 0]

**Solving with γ (discount factor)**:

Using V = R + γPV, we get:

- V(S1) = 2 + γ × 0 = 2 + γV(S2)
- V(S2) = 0 + γ × V(S1)

Solving: V(S1) = 2/(1-γ²), V(S2) = 2γ/(1-γ²)

For γ = 0.9: V(S1) ≈ 10, V(S2) ≈ 9

### 10.2 Grid World Policy Iteration Example

**Setup**: 4×4 grid

- Two terminal states (top-left and bottom-right corners)
- Reward = -1 for each step (encourages finding shortest path)
- Actions: Up, Down, Left, Right
- Deterministic transitions (but hitting wall = stay in place)
- Initial policy: Uniform random (0.25 probability each direction)

**Iteration Process**:

1. **Initialize**: V = 0 for all states, π = uniform random
    
2. **Policy Evaluation**: Calculate V^π using iterative Bellman equation
    
3. **Policy Improvement**: Act greedily - select actions pointing toward higher-value neighbors
    
4. **Repeat** until policy stabilizes
    

**Key Insight**: After just a few iterations, the policy shows optimal arrows pointing toward terminal states via shortest paths.

### 10.3 Special States Example (from Final Exam)

From the Spring 2025 final exam:

**5×5 Grid with Special States**:

- State A: Any action teleports to A' with reward +10
- State B: Any action teleports to B' with reward +5
- Walls give -1 reward, other moves give 0
- γ = 0.9, uniform random policy

**Question**: Why is V(A) < 10 while V(B) > 5?

**Answer**:

- Agent teleported to A' receives immediate reward of 10
    
- But A' might be in a bad location (can hit walls)
    
- The value includes FUTURE expected rewards, not just immediate
    
- If A' is near walls, future expected rewards are negative
    
- So V(A) = 10 + γ × (expected future) < 10
    
- For B, the +5 immediate reward plus favorable location of B' means
    
- V(B) = 5 + γ × (positive expected future) > 5
    

---

## Key Takeaways for the Final

1. **Understand the MDP framework**: States, actions, rewards, transitions, policy, value functions
    
2. **Know both Bellman equations**: Expectation (linear) and Optimality (nonlinear with max)
    
3. **Understand backup trees**: How values propagate from successor states
    
4. **Know Policy Iteration**: Evaluate policy → Improve greedily → Repeat
    
5. **Understand the RL paradigm**: Model-free, learning from experience
    
6. **Monte Carlo**: Sample complete episodes, average returns, unbiased but must wait
    
7. **TD(0)**: Bootstrap from estimated values, can learn online, biased but lower variance
    
8. **TD(λ)**: Interpolates between TD(0) (λ=0) and MC (λ=1) using exponential weighting
    
9. **Be able to work through small examples**: Two-state problems, small grid worlds
    
10. **Understand the intuition**: Value = expected future rewards, policy iteration converges, bootstrapping vs sampling trade-offs
    

---

## Professor's Study Tips

- The professor mentioned he will provide topic hints 48 hours before the exam via Discord
- The final will likely be "easier than the midterm"
- Focus on understanding the grid world examples and how to apply the equations
- Make sure you understand how to evaluate V* after 2-3 iterations of policy iteration
- The recycling robot example from Sutton's book was mentioned - understand V*(high) and V*(low)