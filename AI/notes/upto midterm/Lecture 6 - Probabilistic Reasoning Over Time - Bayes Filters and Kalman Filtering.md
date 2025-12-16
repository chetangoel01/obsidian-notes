### Executive Summary

This lecture introduces the foundational framework for reasoning about dynamic systems under uncertainty—a core capability of autonomous robots and intelligent agents. We move from static perception (classifying a single image) to **temporal reasoning** where measurements arrive sequentially and states evolve over time. The lecture covers two major filtering algorithms: the **Bayes Filter** for discrete state spaces and the **Kalman Filter** for continuous state spaces. Both use a two-step recursive approach: **prediction** (where we estimate the next state based on motion models) and **update**(where we correct our estimate using noisy sensor measurements). This framework is fundamental to robotics applications like localization, tracking, and sensor fusion.

---

## 1. Concept: States and State Representation

### High-Level Intuition

A **state** captures all the information needed to describe a system at a particular moment in time—for a robot vacuum, this might be its position and orientation in a room. The challenge is that we can't directly observe the true state; we must estimate it from noisy measurements.

**Analogy:** Think of state estimation like navigating in fog. You know approximately where you are (your estimated state), but you can only see blurry landmarks (noisy measurements). As you move and take new observations, you continuously refine your belief about your location.

### Conceptual Deep Dive

There are three categories of state representation:

1. **Atomic States**: The state is indivisible and represented as a single discrete value. Example: In a grid-based planning problem, the state might be "square 42" with no internal structure.
    
2. **Factored States**: The state is a vector composed of multiple random variables, each with its own probability distribution. This is what we use in most robotics applications.
    
3. **Structured States**: The state includes relational information between entities (e.g., "bottle is on table"). This is used in more advanced reasoning tasks.
    

For a vehicle, a typical **factored state vector** is:

$$\mathbf{s}_t = [x, y, z, \phi, \theta, \psi, \dot{x}, \dot{y}, \dot{z}, \dot{\phi}, \dot{\theta}, \dot{\psi}]^T$$

Where:

- $(x, y, z)$ = position in 3D space
- $(\phi, \theta, \psi)$ = roll, pitch, yaw (orientation angles)
- $(\dot{x}, \dot{y}, \dot{z})$ = linear velocities
- $(\dot{\phi}, \dot{\theta}, \dot{\psi})$ = angular velocities

This 12-dimensional state vector captures the complete **pose** (position + orientation) and velocities of the vehicle.

### Mathematical Formulation

The state at time $t$ is denoted $\mathbf{s}_t$. For a continuous state space:

$$\mathbf{s}_t \in \mathbb{R}^n$$

where $n$ is the dimensionality of the state space.

For discrete state spaces (like grid cells):

$$\mathbf{s}_t \in {s_1, s_2, \ldots, s_N}$$

where $N$ is the number of possible discrete states.

### Worked Toy Example

Consider a 1D robot on a line segment. Its state is just its position:

$$\mathbf{s}_t = [x]$$

- At $t=0$: Robot is at position $x_0 = 5.0$ meters
- The state vector is simply: $\mathbf{s}_0 = [5.0]$

For a 2D robot (like a vacuum cleaner), the state includes position and orientation:

$$\mathbf{s}_t = [x, y, \theta]$$

- At $t=0$: Robot is at $(x_0, y_0, \theta_0) = (2.0, 3.0, 0.785)$
    - Position: (2m, 3m)
    - Heading: 0.785 radians (45 degrees)
- State vector: $\mathbf{s}_0 = [2.0, 3.0, 0.785]^T$

### Connections & Prerequisites

**Prerequisite Refresher on Random Variables:** Recall that a random variable is a quantity whose value is uncertain. In probabilistic robotics, we represent states as random variables because we never know the exact state—we only have probabilistic beliefs about it. A probability distribution $P(\mathbf{s}_t)$ describes our uncertainty about the state at time $t$.

---

## 2. Concept: Notation and Random Variables in Temporal Models

### High-Level Intuition

In temporal reasoning, we need a systematic way to represent how states change over time, what actions cause those changes, and what measurements we receive. We use subscript notation to track time and introduce three key random variables: **state** ($\mathbf{s}_t$), **action** ($\mathbf{a}_t$), and **measurement** ($\mathbf{z}_t$).

**Analogy:** Think of state estimation like keeping a diary. The **state** is where you are, the **action** is what you did to get there, and the **measurement** is what you observed when you arrived. Each entry in your diary corresponds to a different time step.

### Conceptual Deep Dive

**Convention:** The agent first takes an action, then receives a measurement.

1. **State**: $\mathbf{s}_t$ - The configuration of the system at time $t$ (hidden/unobserved in most cases)
    
2. **Action**: $\mathbf{a}_t$ - The control input applied at time $t$ (e.g., "move forward 1 meter", "rotate 30 degrees"). Actions are **observed** because we command them.
    
3. **Measurement**: $\mathbf{z}_t$ - The sensor observation received at time $t$ (e.g., "door appears open", "obstacle detected at 2m"). Measurements are **observed** but noisy.
    

**Sequence Notation**: We use colon notation to denote sequences:

$$\mathbf{s}_{1:T} = {\mathbf{s}_1, \mathbf{s}_2, \ldots, \mathbf{s}_T}$$

This means "all states from time 1 to time T, inclusive."

### Mathematical Formulation

At each time step $t$, we have:

$$\mathbf{s}_t \in \mathbb{R}^n \quad \text{(state vector)}$$ $$\mathbf{a}_t \in \mathbb{R}^m \quad \text{(action vector)}$$ $$\mathbf{z}_t \in \mathbb{R}^j \quad \text{(measurement vector)}$$

**Example for a robotic vacuum:**

- State: $\mathbf{s}_t = [x, y, \theta]$ (position and heading)
- Action: $\mathbf{a}_t = [v, \omega]$ (linear velocity, angular velocity)
- Measurement: $\mathbf{z}_t = [d_1, d_2, \ldots, d_k]$ (distances from LIDAR rays)

### Worked Toy Example

Consider a robot on a 1D track:

**Time t=0:**

- State: $\mathbf{s}_0 = [3.0]$ meters
- (No action yet)
- (No measurement yet)

**Time t=1:**

- Action: $\mathbf{a}_1 = [+1.0]$ (move forward 1 meter)
- Agent performs action
- True state becomes: $\mathbf{s}_1 = [4.0]$ meters (but we don't know this exactly)
- Measurement: $\mathbf{z}_1 = [4.2]$ meters (noisy sensor reading)

**Sequence up to t=2:**

- States: $\mathbf{s}_{0:2} = {[3.0], [4.0], [5.1]}$
- Actions: $\mathbf{a}_{1:2} = {[+1.0], [+1.0]}$
- Measurements: $\mathbf{z}_{1:2} = {[4.2], [5.3]}$

### Connections & Prerequisites

**Prerequisite Refresher on Conditional Probability:** Recall that $P(A|B)$ represents the probability of event $A$ occurring given that event $B$ has occurred. In temporal models, we'll condition future states on past states, actions, and measurements to capture how the system evolves over time.

---

## 3. Concept: Probabilistic Models - State Transition and Measurement Models

### High-Level Intuition

To reason about dynamic systems, we need two key probabilistic models: one that describes how states **evolve** when we take actions (state transition model), and one that describes how **measurements relate to states** (measurement model). These models capture the inherent uncertainty in both motion and sensing.

**Analogy:** Think of the state transition model as a weather forecast (predicting tomorrow's weather based on today's and what actions nature takes), and the measurement model as a thermometer reading (telling you about the temperature, but with some error).

### Conceptual Deep Dive

**1. State Transition Model (Motion Model)**

This is a **generative model** for state evolution:

$$P(\mathbf{s}_t | \mathbf{s}_{0:t-1}, \mathbf{a}_{1:t}, \mathbf{z}_{1:t-1})$$

This reads as: "The probability of being in state $\mathbf{s}_t$ given:

- All previous states from $\mathbf{s}_0$ to $\mathbf{s}_{t-1}$
- All actions from $\mathbf{a}_1$ to $\mathbf{a}_t$
- All measurements from $\mathbf{z}_1$ to $\mathbf{z}_{t-1}$"

**Problem:** This full conditional is intractable—it requires storing and conditioning on the entire history!

**2. Measurement Model (Sensor Model)**

This describes how measurements depend on states:

$$P(\mathbf{z}_t | \mathbf{s}_t)$$

This reads as: "The probability of observing measurement $\mathbf{z}_t$ given that the true state is $\mathbf{s}_t$."

**Key insight:** Measurements are noisy. Even if you know the exact state, the sensor reading will vary due to noise.

### Mathematical Formulation

**Full State Transition Model (before simplification):**

$$P(\mathbf{s}_t | \mathbf{s}_{0:t-1}, \mathbf{a}_{1:t}, \mathbf{z}_{1:t-1})$$

**Measurement Model:**

$$P(\mathbf{z}_t | \mathbf{s}_t)$$

**Intuition:**

- The transition model is **forward-looking**: "If I'm here and do this action, where will I end up?"
- The measurement model is **observation-based**: "If I'm at this location, what will my sensor read?"

### Worked Toy Example

**Scenario:** Robot vacuum with proximity sensor

**State Transition Model Example:**

- Current state: $\mathbf{s}_0 = \text{"at position } (2,3)\text{"}$
- Action: $\mathbf{a}_1 = \text{"move forward 1m"}$
- Transition model: $P(\mathbf{s}_1 = (3,3) | \mathbf{s}_0 = (2,3), \mathbf{a}_1 = \text{forward}) = 0.8$
    - 80% chance of successfully moving forward
    - 20% chance of slipping on carpet and staying at $(2,3)$

**Measurement Model Example:**

- True state: $\mathbf{s}_t = \text{"next to wall at 0.5m"}$
- Sensor measures distance to wall
- Measurement model: $P(\mathbf{z}_t = 0.55\text{m} | \mathbf{s}_t = 0.5\text{m true}) = 0.4$
    - Sensor is noisy, so it reads 0.55m when true distance is 0.5m with 40% probability
    - Could also read 0.45m, 0.50m, 0.60m, etc. with other probabilities

### Connections & Prerequisites

**Prerequisite Refresher on Generative Models:** A generative model describes the joint distribution of variables—it can "generate" samples. Here, the state transition model can simulate how a robot would move if we repeatedly sampled from $P(\mathbf{s}_t | \mathbf{s}_{t-1}, \mathbf{a}_t)$. This is in contrast to discriminative models (like classifiers) that only model $P(Y|X)$.

---

## 4. Concept: Markovian Assumption (Complete State Assumption)

### High-Level Intuition

The **Markovian assumption** is a simplifying assumption that says "the future depends only on the present, not on the past." If we know the current state $\mathbf{s}_{t-1}$ and the action $\mathbf{a}_t$, we have all the information needed to predict $\mathbf{s}_t$—the entire history before $\mathbf{s}_{t-1}$ is irrelevant.

**Analogy:** Think of a chess game. If I tell you the current board position, you can predict the next position after a move. You don't need to know all the previous moves in the game—the board state "summarizes" the past.

### Conceptual Deep Dive

The **Markovian Assumption** (also called the **Complete State Assumption**) states:

> The state $\mathbf{s}_t$ is a **complete summary** of the past. In statistical terms, $\mathbf{s}_t$ is a **sufficient statistic** for past measurements and actions.

This allows us to simplify the unwieldy full transition model:

$$P(\mathbf{s}_t | \mathbf{s}_{0:t-1}, \mathbf{a}_{1:t}, \mathbf{z}_{1:t-1}) \quad \rightarrow \quad P(\mathbf{s}_t | \mathbf{s}_{t-1}, \mathbf{a}_t)$$

**Drastic Reduction:**

- Before: Conditioning on potentially thousands of variables (entire history)
- After: Conditioning on just 2 variables (previous state and current action)

**Why is this reasonable?** If the state truly captures everything relevant (position, velocity, map knowledge, etc.), then the history is encoded in $\mathbf{s}_{t-1}$.

**Measurement Model Assumption:** Similarly, measurements depend only on the current state:

$$P(\mathbf{z}_t | \mathbf{s}_{0:t}, \mathbf{a}_{1:t}, \mathbf{z}_{1:t-1}) \quad \rightarrow \quad P(\mathbf{z}_t | \mathbf{s}_t)$$

### Mathematical Formulation

**Markov Property:**

$$P(\mathbf{s}_t | \mathbf{s}_{0:t-1}, \mathbf{a}_{1:t}, \mathbf{z}_{1:t-1}) = P(\mathbf{s}_t | \mathbf{s}_{t-1}, \mathbf{a}_t)$$

**Simplified Models:**

1. **State Transition Model** (Markovian): $$P(\mathbf{s}_t | \mathbf{s}_{t-1}, \mathbf{a}_t)$$
    
2. **Measurement Model**: $$P(\mathbf{z}_t | \mathbf{s}_t)$$
    

These two models are **the only probabilistic models** we need to define for Bayesian filtering!

### Worked Toy Example

Consider a robot at discrete locations on a 1D track: ${0, 1, 2, 3, 4, 5}$

**Without Markov Assumption:**

- At $t=3$, to predict $\mathbf{s}_3$, we need:
    - $\mathbf{s}_0 = 0, \mathbf{s}_1 = 1, \mathbf{s}_2 = 2$
    - $\mathbf{a}_1 = \text{right}, \mathbf{a}_2 = \text{right}, \mathbf{a}_3 = \text{right}$
    - $\mathbf{z}_1 = \text{wall on left}, \mathbf{z}_2 = \text{wall on left}$
    - Total: 7 pieces of information

**With Markov Assumption:**

- At $t=3$, to predict $\mathbf{s}_3$, we only need:
    - $\mathbf{s}_2 = 2$ (current state)
    - $\mathbf{a}_3 = \text{right}$ (current action)
    - Total: 2 pieces of information

**Example Probability:** $$P(\mathbf{s}_3 = 3 | \mathbf{s}_2 = 2, \mathbf{a}_3 = \text{right}) = 0.9$$

This says: "If I'm at position 2 and move right, I'll be at position 3 with 90% probability (might slip and stay at 2 with 10% probability)."

### Connections & Prerequisites

**Prerequisite Refresher on Sufficient Statistics:** In statistics, a **sufficient statistic** is a function of the data that captures all the information needed for inference. For example, the sample mean is a sufficient statistic for estimating the mean of a Gaussian distribution. Here, the state $\mathbf{s}_t$ is a sufficient statistic for predicting future states—we don't need the raw history.

---

## 5. Concept: Dynamic Bayesian Networks and Hidden Markov Models (HMMs)

### High-Level Intuition

A **Dynamic Bayesian Network (DBN)** is a graphical representation of how states evolve over time and how measurements relate to states. When the states are hidden (unobserved) and the Markov property holds, this is called a **Hidden Markov Model (HMM)**. The graphical structure makes the probabilistic dependencies clear at a glance.

**Analogy:** Think of a DBN as a flowchart for probability. Arrows show "influences"—if there's an arrow from $\mathbf{s}_{t-1}$ to $\mathbf{s}_t$, it means the previous state influences the current state.

### Conceptual Deep Dive

**Dynamic Bayesian Network Structure:**

```
   a₁        a₂        a₃        aₜ
   ↓         ↓         ↓         ↓
s₀ → s₁  →  s₂  →  s₃  →  ... → sₜ
     ↓       ↓       ↓           ↓
     z₁      z₂      z₃          zₜ
```

**Key Properties:**

1. **Nodes** = Random variables
    
    - **Shaded nodes** = Observed (actions $\mathbf{a}_t$, measurements $\mathbf{z}_t$)
    - **Unshaded nodes** = Hidden/Unobserved (states $\mathbf{s}_t$)
2. **Directed edges** = Probabilistic dependencies
    
    - $\mathbf{s}_{t-1} \rightarrow \mathbf{s}_t$: Previous state influences current state
    - $\mathbf{a}_t \rightarrow \mathbf{s}_t$: Action influences current state
    - $\mathbf{s}_t \rightarrow \mathbf{z}_t$: State determines measurement
3. **Markov property** encoded graphically:
    
    - $\mathbf{s}_t$ only receives arrows from $\mathbf{s}_{t-1}$ and $\mathbf{a}_t$ (not from earlier states)
    - $\mathbf{z}_t$ only receives arrow from $\mathbf{s}_t$ (measurements depend only on current state)

**Why "Hidden" Markov Model?** The states $\mathbf{s}_t$ are **hidden**—we never directly observe them. We only observe actions (which we control) and measurements (which are noisy). Our goal is to infer the hidden states from the observations.

### Mathematical Formulation

The HMM factorizes the joint distribution over all variables:

$$P(\mathbf{s}_{0:T}, \mathbf{a}_{1:T}, \mathbf{z}_{1:T}) = P(\mathbf{s}_0) \prod_{t=1}^{T} P(\mathbf{s}_t | \mathbf{s}_{t-1}, \mathbf{a}_t) P(\mathbf{z}_t | \mathbf{s}_t)$$

Where:

- $P(\mathbf{s}_0)$ = Initial state distribution (prior belief)
- $P(\mathbf{s}_t | \mathbf{s}_{t-1}, \mathbf{a}_t)$ = State transition model
- $P(\mathbf{z}_t | \mathbf{s}_t)$ = Measurement model

**Conditional Independence:** The graphical structure encodes:

- $\mathbf{z}_t$ is conditionally independent of all other variables given $\mathbf{s}_t$
- $\mathbf{s}_t$ is conditionally independent of ${\mathbf{s}_{0:t-2}, \mathbf{a}_{1:t-1}, \mathbf{z}_{1:t-1}}$ given $\mathbf{s}_{t-1}$ and $\mathbf{a}_t$

### Worked Toy Example

**Scenario:** Robot in hallway with 3 positions: ${A, B, C}$

**Graphical Model:**

```
   a₁ = stay    a₂ = right
       ↓            ↓
    s₀ → s₁  →  s₂
    A     ↓       ↓
         z₁      z₂
      "at A"   "at B"
```

**Factorization:** $$P(s_0, s_1, s_2, a_1, a_2, z_1, z_2) = P(s_0=A) \cdot P(s_1|s_0=A, a_1=\text{stay}) \cdot P(z_1|s_1) \cdot P(s_2|s_1, a_2=\text{right}) \cdot P(z_2|s_2)$$

**Reading the graph:**

- We start with belief $P(s_0 = A) = 1.0$ (certain we're at A)
- We take action "stay", so likely $s_1 = A$ also
- We measure $z_1 = \text{"at A"}$ (confirms our belief)
- We take action "right", so likely $s_2 = B$
- We measure $z_2 = \text{"at B"}$ (confirms transition)

**Conditional Independence Example:**

- $P(z_2 | s_2=B, z_1=\text{"at A"}) = P(z_2 | s_2=B)$
    - The past measurement $z_1$ doesn't matter for predicting $z_2$ once we know $s_2$

### Connections & Prerequisites

**Prerequisite Refresher on Graphical Models:** In probabilistic graphical models, nodes represent random variables and edges represent direct dependencies. If there's no edge between two nodes, they are conditionally independent given their parents. This structure allows us to factor complex joint distributions into simpler conditional distributions.

---

## 6. Concept: Bayes Filter - General Recursive Framework

### High-Level Intuition

The **Bayes Filter** is a two-step recursive algorithm for estimating the state of a dynamic system. Step 1 (**Prediction**): Use the motion model to predict where you'll be after taking an action (you're "blind" to measurements here). Step 2 (**Update**): When a measurement arrives, correct your prediction using Bayes' rule. Repeat forever.

**Analogy:** Think of navigating while blindfolded. First, you take a step (prediction—you estimate where you've moved based on your action). Then you briefly open your eyes (measurement update—you correct your estimate based on what you see).

### Conceptual Deep Dive

The Bayes Filter maintains a **belief** about the state:

**Belief** = Probability distribution over states given all available information

- **bel⁻($\mathbf{s}_t$)** = "belief bar" = Prior belief = Prediction before receiving measurement $\mathbf{z}_t$
- **bel($\mathbf{s}_t$)** = Posterior belief = Updated belief after receiving measurement $\mathbf{z}_t$

**The Two-Step Cycle:**

1. **Prediction Step:**
    
    - **Input:** Previous belief bel($\mathbf{s}_{t-1}$), action $\mathbf{a}_t$
    - **Output:** Prior belief bel⁻($\mathbf{s}_t$)
    - **Operation:** Apply motion model
    - **Interpretation:** "Given my belief about where I was and the action I took, where do I expect to be now?"
2. **Measurement Update Step:**
    
    - **Input:** Prior belief bel⁻($\mathbf{s}_t$), measurement $\mathbf{z}_t$
    - **Output:** Posterior belief bel($\mathbf{s}_t$)
    - **Operation:** Apply Bayes' rule
    - **Interpretation:** "My sensor says something—how should I adjust my prediction to account for this evidence?"

**Why Bayesian?** The measurement update step is literally Bayes' rule:

$$P(\mathbf{s}_t | \mathbf{z}_t) = \frac{P(\mathbf{z}_t | \mathbf{s}_t) P(\mathbf{s}_t)}{P(\mathbf{z}_t)} = \eta \cdot P(\mathbf{z}_t | \mathbf{s}_t) P(\mathbf{s}_t)$$

Where $\eta = 1/P(\mathbf{z}_t)$ is a normalization constant.

### Mathematical Formulation

**Bayes Filter Algorithm:**

**Function:** `BayesFilter(bel($\mathbf{s}_{t-1}$), $\mathbf{a}_t$, $\mathbf{z}_t$)`

**Step 1: Prediction** $$\text{bel}^-(\mathbf{s}_t) = \sum_{\mathbf{s}_{t-1}} P(\mathbf{s}_t | \mathbf{s}_{t-1}, \mathbf{a}_t) \cdot \text{bel}(\mathbf{s}_{t-1})$$

This is the **sum rule** (marginalization): $$P(X) = \sum_Y P(X, Y) = \sum_Y P(X|Y)P(Y)$$

**Step 2: Measurement Update** $$\text{bel}(\mathbf{s}_t) = \eta \cdot P(\mathbf{z}_t | \mathbf{s}_t) \cdot \text{bel}^-(\mathbf{s}_t)$$

Where:

- $P(\mathbf{z}_t | \mathbf{s}_t)$ = Likelihood (from measurement model)
- bel⁻($\mathbf{s}_t$) = Prior (from prediction step)
- $\eta$ = Normalization constant ensuring $\sum_{\mathbf{s}_t} \text{bel}(\mathbf{s}_t) = 1$

This is **Bayes' rule**: $$P(Y|X) = \eta \cdot P(X|Y) P(Y)$$

**Return:** bel($\mathbf{s}_t$)

### Worked Toy Example

**Scenario:** 1D robot at positions ${0, 1, 2}$

**Initialization:**

- bel($\mathbf{s}_0$) = [0.33, 0.33, 0.34] (uniform—no idea where we are)

**Time t=1:**

- Action: $\mathbf{a}_1 = \text{"move right"}$
- Measurement: $\mathbf{z}_1 = \text{"sensor says position 1"}$

**Step 1: Prediction**

Transition model: Moving right succeeds with 0.8 prob, stays in place with 0.2 prob

$$\text{bel}^-(s_1=0) = P(s_1=0|s_0=0, \text{right}) \cdot 0.33 + P(s_1=0|s_0=1, \text{right}) \cdot 0.33 + P(s_1=0|s_0=2, \text{right}) \cdot 0.34$$ $$= 0.2 \cdot 0.33 + 0 \cdot 0.33 + 0 \cdot 0.34 = 0.066$$

$$\text{bel}^-(s_1=1) = 0.8 \cdot 0.33 + 0.2 \cdot 0.33 + 0 \cdot 0.34 = 0.33$$

$$\text{bel}^-(s_1=2) = 0 \cdot 0.33 + 0.8 \cdot 0.33 + 0.2 \cdot 0.34 = 0.332$$

Result: bel⁻($s_1$) = [0.066, 0.33, 0.332] (still uncertain, but shifted right)

**Step 2: Update**

Measurement model: Sensor correct with 0.9 prob, off by 1 with 0.1 prob

$$\text{bel}(s_1=0) = \eta \cdot P(z_1=1|s_1=0) \cdot \text{bel}^-(s_1=0) = \eta \cdot 0.1 \cdot 0.066 = \eta \cdot 0.0066$$ $$\text{bel}(s_1=1) = \eta \cdot 0.9 \cdot 0.33 = \eta \cdot 0.297$$ $$\text{bel}(s_1=2) = \eta \cdot 0.1 \cdot 0.332 = \eta \cdot 0.0332$$

Sum = $0.0066 + 0.297 + 0.0332 = 0.3368$, so $\eta = 1/0.3368 = 2.97$

$$\text{bel}(s_1) = [0.02, 0.88, 0.10]$$

**Result:** After measurement, we're 88% confident we're at position 1!

### Connections & Prerequisites

**Prerequisite Refresher on Marginalization:** The sum rule says $P(X) = \sum_Y P(X,Y)$. In the prediction step, we marginalize out (sum over) all possible previous states $\mathbf{s}_{t-1}$ to get the marginal distribution of $\mathbf{s}_t$. This accounts for uncertainty in where we were before.

---

## 7. Concept: Discrete Bayes Filter Applied - The Door Example

### High-Level Intuition

This example demonstrates the Bayes Filter on a simple binary state system: a robot with a noisy sensor trying to determine if a door is open or closed. We'll see how the belief evolves as the robot takes actions (push door) and receives measurements (sensor reading).

**Analogy:** You're in a dark room trying to find out if a door is open. You push on it (action) and listen for a creak (measurement). Based on whether you hear a creak or not, you update your belief about whether the door is open.

### Conceptual Deep Dive

**Setup:**

- **State space:** $\mathbf{s}_t \in {\text{Open}, \text{Closed}}$ (binary)
- **Action space:** $\mathbf{a}_t \in {\text{Push}, \text{DoNothing}}$ (binary)
- **Measurement space:** $\mathbf{z}_t \in {\text{SenseOpen}, \text{SenseClosed}}$ (binary)

**Initial Belief:** $$\text{bel}(\mathbf{s}_0) = [P(\text{Open}), P(\text{Closed})] = [0.5, 0.5]$$

Complete uncertainty about initial door state.

**Measurement Model:** Calibrated from sensor tests:

| True State | P(Sense Open | State) | P(Sense Closed | State) | |------------|------------------------|------------------------| | Open | 0.6 | 0.4 | | Closed | 0.2 | 0.8 |

Note the **asymmetry**: Easier to detect "closed" (beam reflects back) than "open" (beam goes through door).

**State Transition Model:**

| Current State | Action    | Next State | Probability |
| ------------- | --------- | ---------- | ----------- |
| Open          | Push      | Open       | 1.0         |
| Open          | DoNothing | Open       | 1.0         |
| Closed        | Push      | Open       | 0.8         |
| Closed        | Push      | Closed     | 0.2         |
| Closed        | DoNothing | Closed     | 1.0         |

Interpretation: Pushing an open door keeps it open; pushing a closed door opens it 80% of the time.

### Mathematical Formulation

**At t=1: Action = DoNothing, Measurement = SenseOpen**

**Prediction Step:** $$\text{bel}^-(s_1 = \text{Open}) = \sum_{s_0} P(s_1=\text{Open}|s_0, \text{DoNothing}) \cdot \text{bel}(s_0)$$

$$= P(\text{Open}|\text{Open}, \text{DoNothing}) \cdot 0.5 + P(\text{Open}|\text{Closed}, \text{DoNothing}) \cdot 0.5$$ $$= 1.0 \cdot 0.5 + 0.0 \cdot 0.5 = 0.5$$

Similarly: $$\text{bel}^-(s_1 = \text{Closed}) = 0.5$$

**Update Step:** $$\text{bel}(s_1 = \text{Open}) = \eta \cdot P(\text{SenseOpen}|\text{Open}) \cdot \text{bel}^-(s_1=\text{Open})$$ $$= \eta \cdot 0.6 \cdot 0.5 = \eta \cdot 0.3$$

$$\text{bel}(s_1 = \text{Closed}) = \eta \cdot 0.2 \cdot 0.5 = \eta \cdot 0.1$$

Normalization: $\eta = 1/(0.3 + 0.1) = 2.5$

$$\text{bel}(s_1) = [0.75, 0.25]$$

**Interpretation:** The measurement "SenseOpen" increased our belief that the door is open from 50% to 75%.

### Worked Toy Example

Let's trace through **t=2** with **Action = Push, Measurement = SenseOpen**

**Prediction Step:**

$$\text{bel}^-(s_2 = \text{Open}) = P(\text{Open}|\text{Open}, \text{Push}) \cdot 0.75 + P(\text{Open}|\text{Closed}, \text{Push}) \cdot 0.25$$ $$= 1.0 \cdot 0.75 + 0.8 \cdot 0.25 = 0.75 + 0.2 = 0.95$$

$$\text{bel}^-(s_2 = \text{Closed}) = 0 \cdot 0.75 + 0.2 \cdot 0.25 = 0.05$$

**Interpretation:** After pushing, we're 95% confident door is open (pushed it open from closed, or kept it open).

**Update Step:**

$$\text{bel}(s_2 = \text{Open}) = \eta \cdot 0.6 \cdot 0.95 = \eta \cdot 0.57$$ $$\text{bel}(s_2 = \text{Closed}) = \eta \cdot 0.2 \cdot 0.05 = \eta \cdot 0.01$$

Normalization: $\eta = 1/(0.57 + 0.01) = 1.724$

$$\text{bel}(s_2) = [0.98, 0.02]$$

**Final Result:** After pushing and sensing open again, we're now 98% confident the door is open!

### Connections & Prerequisites

**Prerequisite Refresher on Lookup Tables:** In discrete Bayes Filters, the transition and measurement models are stored as lookup tables. When you need $P(s_t|s_{t-1}, a_t)$, you index into the transition table with the specific values of $s_{t-1}$ and $a_t$. For $N$ states, $M$ actions, and $J$ measurements, you need an $N \times N \times M$ transition table and an $N \times J$ measurement table.

---

## 8. Concept: Transition to Continuous State Spaces - The Kalman Filter

### High-Level Intuition

Real robots operate in continuous spaces (position is a real number, not a discrete grid cell). The **Kalman Filter** is the continuous version of the Bayes Filter, specialized for **linear-Gaussian systems**. If states and measurements are continuous, motion models are linear, and noise is Gaussian, the Kalman Filter gives the optimal estimate.

**Analogy:** The discrete Bayes Filter is like having a multiple-choice question (robot is in room A, B, or C). The Kalman Filter is like having a fill-in-the-blank question (robot is at position x = 3.7412 meters), which requires different math.

### Conceptual Deep Dive

**Why do we need a different filter for continuous spaces?**

In the discrete case, we could represent the belief as a probability mass function (list of probabilities): $$\text{bel}(s_t) = [0.1, 0.3, 0.2, 0.15, 0.25]$$

In continuous spaces, the belief is a probability density function. We can't store a density for every real number—we need a **parametric representation**.

**Gaussian Assumption:** The Kalman Filter assumes all distributions are Gaussian:

- Prior: $\text{bel}^-(\mathbf{s}_t) \sim \mathcal{N}(\boldsymbol{\mu}_t^-, \boldsymbol{\Sigma}_t^-)$
- Posterior: $\text{bel}(\mathbf{s}_t) \sim \mathcal{N}(\boldsymbol{\mu}_t, \boldsymbol{\Sigma}_t)$

**Sufficient Statistics for Gaussians:** A Gaussian is completely specified by two parameters:

1. **Mean vector** $\boldsymbol{\mu} \in \mathbb{R}^n$
2. **Covariance matrix** $\boldsymbol{\Sigma} \in \mathbb{R}^{n \times n}$

So instead of tracking a full density function, we just track $(\boldsymbol{\mu}, \boldsymbol{\Sigma})$!

**Key Properties:**

1. **Closure under linear transformations:** If $X \sim \mathcal{N}(\mu, \sigma^2)$, then $Y = aX + b \sim \mathcal{N}(a\mu + b, a^2\sigma^2)$
2. **Product of Gaussians is Gaussian:** If we multiply two Gaussian densities, the result is (proportional to) a Gaussian

These properties make the math tractable.

### Mathematical Formulation

**Kalman Filter vs. Discrete Bayes Filter:**

|Discrete Bayes Filter|Kalman Filter|
|---|---|
|$\text{bel}(\mathbf{s}_t) = [p_1, p_2, \ldots, p_N]$|$\text{bel}(\mathbf{s}_t) \sim \mathcal{N}(\boldsymbol{\mu}_t, \boldsymbol{\Sigma}_t)$|
|Summation over discrete states|Integration over continuous states|
|Lookup tables for models|Linear equations for models|

**Linear-Gaussian System:**

1. **State Transition (Motion Model):** $$\mathbf{s}_t = \mathbf{A}_t \mathbf{s}_{t-1} + \mathbf{B}_t \mathbf{a}_t + \boldsymbol{\epsilon}_t$$

Where:

- $\mathbf{A}_t \in \mathbb{R}^{n \times n}$ = State transition matrix
- $\mathbf{B}_t \in \mathbb{R}^{n \times m}$ = Control input matrix
- $\boldsymbol{\epsilon}_t \sim \mathcal{N}(\mathbf{0}, \mathbf{R}_t)$ = Process noise with covariance $\mathbf{R}_t$

2. **Measurement Model:** $$\mathbf{z}_t = \mathbf{C}_t \mathbf{s}_t + \boldsymbol{\delta}_t$$

Where:

- $\mathbf{C}_t \in \mathbb{R}^{j \times n}$ = Measurement matrix
- $\boldsymbol{\delta}_t \sim \mathcal{N}(\mathbf{0}, \mathbf{Q}_t)$ = Measurement noise with covariance $\mathbf{Q}_t$

**Why "Linear-Gaussian"?**

- **Linear:** The equations are linear in the state and action
- **Gaussian:** All noise terms are Gaussian

### Worked Toy Example

**1D Drone Localization:**

State: $\mathbf{s}_t = [x]$ (position on x-axis)

**Motion Model:** Drone hovers (no movement): $$x_t = x_{t-1} + \epsilon_t, \quad \epsilon_t \sim \mathcal{N}(0, 0.01)$$

This gives:

- $\mathbf{A}_t = [1]$ (1×1 matrix)
- $\mathbf{B}_t = [0]$ (no control input)
- $\mathbf{R}_t = [0.01]$ (small process noise)

**Measurement Model:** Radar measures position with noise: $$z_t = x_t + \delta_t, \quad \delta_t \sim \mathcal{N}(0, 0.25)$$

This gives:

- $\mathbf{C}_t = [1]$ (1×1 matrix—direct observation)
- $\mathbf{Q}_t = [0.25]$ (larger measurement noise)

**Initial Belief:** $$\text{bel}(\mathbf{s}_0) = \mathcal{N}(\mu_0 = 0, \Sigma_0 = 10)$$

Very uncertain about initial position (large variance).

**Time t=1:** We'll compute the prediction and update with $z_1 = 0.5$ in the next section.

### Connections & Prerequisites

**Prerequisite Refresher on Gaussian Distributions:** A univariate Gaussian $X \sim \mathcal{N}(\mu, \sigma^2)$ has pdf: $p(x) = \frac{1}{\sqrt{2\pi\sigma^2}} e^{-\frac{(x-\mu)^2}{2\sigma^2}}$. For multivariate Gaussian $\mathbf{X} \sim \mathcal{N}(\boldsymbol{\mu}, \boldsymbol{\Sigma})$, the pdf is: $p(\mathbf{x}) = \frac{1}{\sqrt{(2\pi)^n|\boldsymbol{\Sigma}|}} e^{-\frac{1}{2}(\mathbf{x}-\boldsymbol{\mu})^T\boldsymbol{\Sigma}^{-1}(\mathbf{x}-\boldsymbol{\mu})}$. The covariance matrix $\boldsymbol{\Sigma}$ captures correlations between dimensions.

---

## 9. Concept: Recursive Maximum Likelihood Estimation and Kalman Gain

### High-Level Intuition

Before deriving the full Kalman Filter, we build intuition by looking at a simpler problem: estimating the mean of a Gaussian from sequential measurements. We'll discover that the optimal estimate can be written **recursively**: new estimate = old estimate + gain × (measurement - old estimate). This same structure appears in the Kalman Filter.

**Analogy:** You're estimating the temperature outside. Each time you check your thermometer, you don't throw away your previous belief—you combine it with the new reading. If you trust the thermometer, you weight it heavily; if it's unreliable, you weight your previous estimate more.

### Conceptual Deep Dive

**Problem Setup:**

- Measurements arrive sequentially: $z_1, z_2, \ldots, z_t$
- Each $z_i \sim \mathcal{N}(x, \sigma_z^2)$ (Gaussian noise around true value $x$)
- Goal: Estimate $x$ after seeing $t$ measurements

**Maximum Likelihood Estimate (MLE):**

From basic statistics, the MLE of the mean is the sample mean:

$$\hat{x}_t^{\text{MLE}} = \frac{1}{t} \sum_{i=1}^t z_i$$

**Recursive Formulation:**

Can we compute $\hat{x}_t^{\text{MLE}}$ from $\hat{x}_{t-1}^{\text{MLE}}$ without storing all measurements?

Yes! Here's the derivation:

$$\hat{x}_t^{\text{MLE}} = \frac{1}{t} \sum_{i=1}^t z_i = \frac{1}{t}\left(\sum_{i=1}^{t-1} z_i + z_t\right)$$

$$= \frac{1}{t} \cdot (t-1) \cdot \frac{1}{t-1}\sum_{i=1}^{t-1} z_i + \frac{1}{t} z_t$$

$$= \frac{t-1}{t} \hat{x}_{t-1}^{\text{MLE}} + \frac{1}{t} z_t$$

$$= \left(1 - \frac{1}{t}\right) \hat{x}_{t-1}^{\text{MLE}} + \frac{1}{t} z_t$$

$$= \hat{x}_{t-1}^{\text{MLE}} + \frac{1}{t}(z_t - \hat{x}_{t-1}^{\text{MLE}})$$

**Final Form:**

$$\boxed{\hat{x}_t^{\text{MLE}} = \hat{x}_{t-1}^{\text{MLE}} + K_t (z_t - \hat{x}_{t-1}^{\text{MLE}})}$$

Where **Kalman Gain**: $K_t = \frac{1}{t}$

### Mathematical Formulation

**Three Interpretations of the Recursive Update:**

**Form 1:** Weighted average $$\hat{x}_t = \frac{t-1}{t} \hat{x}_{t-1} + \frac{1}{t} z_t$$

**Form 2:** Previous estimate + correction $$\hat{x}_t = \hat{x}_{t-1} + K_t \cdot \underbrace{(z_t - \hat{x}_{t-1})}_{\text{innovation/residual}}$$

**Form 3:** Expanded $$\hat{x}_t = (1 - K_t) \hat{x}_{t-1} + K_t z_t$$

**Key Terms:**

1. **Kalman Gain** $K_t = \frac{1}{t} \in [0, 1]$
    
    - Controls how much we trust the new measurement vs. previous estimate
    - $K_t \to 0$ as $t \to \infty$ (trust accumulates in prior)
2. **Innovation (or Residual)**: $(z_t - \hat{x}_{t-1})$
    
    - How much the measurement differs from prediction
    - Also called "measurement surprise"

**Behavior Over Time:**

- **Early (t small):** $K_t$ is large → trust measurements heavily
- **Late (t large):** $K_t$ is small → trust accumulated estimate

### Worked Toy Example

**Measuring Room Temperature:**

True temperature: $x = 20°C$ (unknown to us) Measurements: $z_t \sim \mathcal{N}(20, 1)$ (thermometer has $\sigma = 1°C$ noise)

**Initial estimate:** $\hat{x}_0 = 18°C$ (our guess)

**t=1:** Measurement $z_1 = 21.5°C$

$$K_1 = \frac{1}{1} = 1.0$$ $$\hat{x}_1 = 18 + 1.0 \cdot (21.5 - 18) = 18 + 3.5 = 21.5°C$$

We fully trust the first measurement (Gain = 1).

**t=2:** Measurement $z_2 = 19.0°C$

$$K_2 = \frac{1}{2} = 0.5$$ $$\hat{x}_2 = 21.5 + 0.5 \cdot (19.0 - 21.5) = 21.5 - 1.25 = 20.25°C$$

We balance the measurement and previous estimate.

**t=3:** Measurement $z_3 = 20.5°C$

$$K_3 = \frac{1}{3} \approx 0.33$$ $$\hat{x}_3 = 20.25 + 0.33 \cdot (20.5 - 20.25) = 20.25 + 0.083 = 20.33°C$$

Less weight on new measurement (Gain decreasing).

**t=10:** Measurement $z_{10} = 22.0°C$

$$K_{10} = \frac{1}{10} = 0.1$$ $$\hat{x}_{10} = \hat{x}_9 + 0.1 \cdot (22.0 - \hat{x}_9)$$

Now we heavily trust our accumulated estimate—a single noisy reading doesn't move us much.

**Trajectory:**

```
t=0:  ̂x=18.0°C  (initial guess)
t=1:  ̂x=21.5°C  (big jump—fully trust first measurement)
t=2:  ̂x=20.25°C (converging toward true value)
t=3:  ̂x=20.33°C
...
t=10: ̂x≈20.1°C  (small adjustments now)
```

**Observation:** The estimate converges to the true value as more measurements arrive, and later measurements have diminishing influence.

### Connections & Prerequisites

**Prerequisite Refresher on Sample Mean:** The sample mean $\bar{x} = \frac{1}{n}\sum_{i=1}^n x_i$ is the MLE for the mean of a Gaussian distribution. It's also an unbiased estimator (expected value equals true mean) and has variance $\sigma^2/n$ (uncertainty decreases as $1/\sqrt{n}$). The recursive formulation avoids storing all past measurements, making it memory-efficient for real-time systems.

---

## 10. Concept: Product of Gaussians and Measurement Update Intuition

### High-Level Intuition

When we combine a prediction (prior Gaussian) with a measurement (likelihood Gaussian) using Bayes' rule, the result (posterior) is also Gaussian. The **product of two Gaussians is a Gaussian**, and this closed-form property is what makes the Kalman Filter tractable. The posterior mean is a weighted average of prior and measurement, where weights depend on their relative uncertainties.

**Analogy:** Two witnesses give testimony about a crime time: one says "around 3pm" (uncertain), another says "3:15pm" (confident). You combine them by trusting the confident witness more—the posterior belief is closer to 3:15pm but adjusted slightly by the uncertain witness.

### Conceptual Deep Dive

**Bayes' Rule for Gaussians:**

Prior: $P(x) = \mathcal{N}(x | \mu_{\text{prior}}, \sigma_{\text{prior}}^2)$ Likelihood: $P(z|x) = \mathcal{N}(z | x, \sigma_z^2)$ (measurement $z$ centered at true state $x$)

Posterior: $$P(x|z) = \eta \cdot P(z|x) P(x)$$

**Result (Product of Gaussians):**

The posterior is also Gaussian: $$P(x|z) = \mathcal{N}(x | \mu_{\text{post}}, \sigma_{\text{post}}^2)$$

Where:

$$\mu_{\text{post}} = \frac{\sigma_z^2 \mu_{\text{prior}} + \sigma_{\text{prior}}^2 z}{\sigma_{\text{prior}}^2 + \sigma_z^2}$$

$$\sigma_{\text{post}}^2 = \frac{\sigma_{\text{prior}}^2 \sigma_z^2}{\sigma_{\text{prior}}^2 + \sigma_z^2}$$

**Key Insights:**

1. **Variance decreases:** $\sigma_{\text{post}}^2 < \min(\sigma_{\text{prior}}^2, \sigma_z^2)$
    
    - We're always more certain after incorporating a measurement!
2. **Mean is weighted average:** More certain source gets more weight
    
    - If $\sigma_{\text{prior}} \ll \sigma_z$ (prior very certain), then $\mu_{\text{post}} \approx \mu_{\text{prior}}$
    - If $\sigma_z \ll \sigma_{\text{prior}}$ (measurement very certain), then $\mu_{\text{post}} \approx z$
3. **Symmetric influence:** Both prior and likelihood contribute
    

### Mathematical Formulation

**Alternative Form (Precision-Weighted):**

Using **precision** (inverse variance) $\lambda = 1/\sigma^2$:

$$\lambda_{\text{post}} = \lambda_{\text{prior}} + \lambda_z$$

$$\mu_{\text{post}} = \frac{\lambda_{\text{prior}} \mu_{\text{prior}} + \lambda_z z}{\lambda_{\text{post}}}$$

This shows: **precisions add** (easier to combine information), and mean is precision-weighted average.

**Kalman Gain Form:**

We can rewrite the mean update as:

$$\mu_{\text{post}} = \mu_{\text{prior}} + K(z - \mu_{\text{prior}})$$

Where: $$K = \frac{\sigma_{\text{prior}}^2}{\sigma_{\text{prior}}^2 + \sigma_z^2}$$

This is the **Kalman Gain** for the 1D case!

**Interpretation:**

- If $\sigma_{\text{prior}}^2 \gg \sigma_z^2$ (prior uncertain), then $K \approx 1$ → trust measurement
- If $\sigma_{\text{prior}}^2 \ll \sigma_z^2$ (measurement uncertain), then $K \approx 0$ → trust prior

### Worked Toy Example

**Three Scenarios:**

**Scenario 1: Agreement**

- Prior: $\mathcal{N}(10, 1)$ (believe we're at position 10, std dev = 1)
- Measurement: $z = 10$ with $\sigma_z = 1$ (measurement also says 10)

$$\mu_{\text{post}} = \frac{1 \cdot 10 + 1 \cdot 10}{1 + 1} = 10$$

$$\sigma_{\text{post}}^2 = \frac{1 \cdot 1}{1 + 1} = 0.5, \quad \sigma_{\text{post}} = 0.707$$

Result: $\mathcal{N}(10, 0.5)$ → Mean stays at 10, but variance decreased from 1 to 0.5 (more certain!)

**Scenario 2: Disagreement with Equal Uncertainty**

- Prior: $\mathcal{N}(10, 1)$
- Measurement: $z = 12$ with $\sigma_z = 1$

$$\mu_{\text{post}} = \frac{1 \cdot 10 + 1 \cdot 12}{1 + 1} = 11$$

$$\sigma_{\text{post}}^2 = 0.5$$

Result: $\mathcal{N}(11, 0.5)$ → Mean is exactly in the middle (equal weighting)

**Scenario 3: Disagreement with Unequal Uncertainty**

- Prior: $\mathcal{N}(10, 4)$ (uncertain prior, $\sigma = 2$)
- Measurement: $z = 12$ with $\sigma_z = 1$ (confident measurement)

$$\mu_{\text{post}} = \frac{1 \cdot 10 + 4 \cdot 12}{4 + 1} = \frac{10 + 48}{5} = 11.6$$

$$\sigma_{\text{post}}^2 = \frac{4 \cdot 1}{4 + 1} = 0.8$$

Result: $\mathcal{N}(11.6, 0.8)$ → Mean closer to measurement (because it's more confident)

**Visual Intuition:**

```
Scenario 1: Agreement
Prior:       ___/\___         (centered at 10)
Measurement:    /\            (also at 10)
Posterior:     /||\           (at 10, narrower)

Scenario 2: Equal Disagreement
Prior:       __/\__           (centered at 10)
Measurement:      __/\__      (centered at 12)
Posterior:      __|__         (centered at 11)

Scenario 3: Unequal Disagreement  
Prior:      ___/\___          (at 10, wide/uncertain)
Measurement:       /|\        (at 12, narrow/certain)
Posterior:        /||\        (at 11.6, closer to measurement)
```

### Connections & Prerequisites

**Prerequisite Refresher on Bayes' Rule:** $P(A|B) = \frac{P(B|A)P(A)}{P(B)}$. In state estimation, $A$ is the state and $B$ is the measurement. The prior $P(A)$ captures our belief before measurement; the likelihood $P(B|A)$ models the sensor; the posterior $P(A|B)$ is our updated belief after measurement. The denominator $P(B)$ is just a normalizing constant.

---

## 11. Concept: The Kalman Filter - Complete Algorithm

### High-Level Intuition

The **Kalman Filter** is the optimal recursive state estimator for linear-Gaussian systems. It extends the ideas we've developed (recursive estimation, product of Gaussians) to multi-dimensional systems with motion models. Like the Bayes Filter, it has two steps: **predict** (propagate uncertainty through motion model) and **update** (reduce uncertainty using measurement).

**Analogy:** You're tracking a ball thrown in the air. Prediction: "Given its current position and velocity, where will it be in 0.1 seconds?" (physics model + uncertainty). Update: "My camera sees it at position x—let me correct my prediction" (sensor fusion).

### Conceptual Deep Dive

**System Assumptions:**

1. **Linear State Transition:** $$\mathbf{s}_t = \mathbf{A}_t \mathbf{s}_{t-1} + \mathbf{B}_t \mathbf{a}_t + \boldsymbol{\epsilon}_t, \quad \boldsymbol{\epsilon}_t \sim \mathcal{N}(\mathbf{0}, \mathbf{R}_t)$$
    
2. **Linear Measurement:** $$\mathbf{z}_t = \mathbf{C}_t \mathbf{s}_t + \boldsymbol{\delta}_t, \quad \boldsymbol{\delta}_t \sim \mathcal{N}(\mathbf{0}, \mathbf{Q}_t)$$
    
3. **Gaussian Belief:** $$\text{bel}(\mathbf{s}_t) = \mathcal{N}(\boldsymbol{\mu}_t, \boldsymbol{\Sigma}_t)$$
    

**Algorithm Structure:**

The Kalman Filter maintains the **sufficient statistics** $(\boldsymbol{\mu}_t, \boldsymbol{\Sigma}_t)$ and updates them recursively.

**Input:** Previous belief $\mathcal{N}(\boldsymbol{\mu}_{t-1}, \boldsymbol{\Sigma}_{t-1})$, action $\mathbf{a}_t$, measurement $\mathbf{z}_t$

**Output:** Updated belief $\mathcal{N}(\boldsymbol{\mu}_t, \boldsymbol{\Sigma}_t)$

**Two Steps:**

1. **Prediction:** Propagate belief through motion model
    
    - Mean prediction: $\boldsymbol{\mu}_t^- = \mathbf{A}_t \boldsymbol{\mu}_{t-1} + \mathbf{B}_t \mathbf{a}_t$
    - Covariance prediction: $\boldsymbol{\Sigma}_t^- = \mathbf{A}_t \boldsymbol{\Sigma}_{t-1} \mathbf{A}_t^T + \mathbf{R}_t$
    - **Uncertainty increases** due to process noise $\mathbf{R}_t$
2. **Update:** Correct prediction using measurement
    
    - Kalman Gain: $\mathbf{K}_t = \boldsymbol{\Sigma}_t^- \mathbf{C}_t^T (\mathbf{C}_t \boldsymbol{\Sigma}_t^- \mathbf{C}_t^T + \mathbf{Q}_t)^{-1}$
    - Mean update: $\boldsymbol{\mu}_t = \boldsymbol{\mu}_t^- + \mathbf{K}_t (\mathbf{z}_t - \mathbf{C}_t \boldsymbol{\mu}_t^-)$
    - Covariance update: $\boldsymbol{\Sigma}_t = (\mathbf{I} - \mathbf{K}_t \mathbf{C}_t) \boldsymbol{\Sigma}_t^-$
    - **Uncertainty decreases** (measurement reduces uncertainty)

### Mathematical Formulation

**Kalman Filter Algorithm:**

```
Function: KalmanFilter(μ_{t-1}, Σ_{t-1}, a_t, z_t)

PREDICTION STEP:
1. Predicted mean:
   μ_t^- = A_t μ_{t-1} + B_t a_t

2. Predicted covariance:
   Σ_t^- = A_t Σ_{t-1} A_t^T + R_t

UPDATE STEP:
3. Kalman Gain:
   K_t = Σ_t^- C_t^T (C_t Σ_t^- C_t^T + Q_t)^{-1}

4. Updated mean:
   μ_t = μ_t^- + K_t (z_t - C_t μ_t^-)

5. Updated covariance:
   Σ_t = (I - K_t C_t) Σ_t^-

Return: (μ_t, Σ_t)
```

**Matrix Dimensions:**

- State: $\mathbf{s}_t \in \mathbb{R}^n$
- Action: $\mathbf{a}_t \in \mathbb{R}^m$
- Measurement: $\mathbf{z}_t \in \mathbb{R}^j$
- $\mathbf{A}_t$: $n \times n$ (state transition)
- $\mathbf{B}_t$: $n \times m$ (control input)
- $\mathbf{C}_t$: $j \times n$ (measurement projection)
- $\mathbf{R}_t$: $n \times n$ (process noise covariance)
- $\mathbf{Q}_t$: $j \times j$ (measurement noise covariance)
- $\mathbf{K}_t$: $n \times j$ (Kalman gain)

**Key Properties:**

1. **Innovation:** $(\mathbf{z}_t - \mathbf{C}_t \boldsymbol{\mu}_t^-)$ is the "surprise" in the measurement
2. **Kalman Gain** balances trust between prediction and measurement:
    - High $\mathbf{K}_t$ → trust measurement more
    - Low $\mathbf{K}_t$ → trust prediction more
3. **Covariance never increases** in update step: $\boldsymbol{\Sigma}_t \leq \boldsymbol{\Sigma}_t^-$

### Worked Toy Example

**1D Position Tracking:**

A stationary drone (hovers at fixed position) with radar measurement.

**State:** $\mathbf{s}_t = [x]$ (1D position)

**Models:**

- Motion: $x_t = x_{t-1} + \epsilon_t$ with $\epsilon_t \sim \mathcal{N}(0, 0.01)$
    - $\mathbf{A}_t = [1]$, $\mathbf{B}_t = [0]$, $\mathbf{R}_t = [0.01]$
- Measurement: $z_t = x_t + \delta_t$ with $\delta_t \sim \mathcal{N}(0, 0.25)$
    - $\mathbf{C}_t = [1]$, $\mathbf{Q}_t = [0.25]$

**Initialization:**

- $\mu_0 = 0$, $\Sigma_0 = 10$ (very uncertain initial belief)

**Time t=1:** Measurement $z_1 = 0.5$

**Prediction:** $$\mu_1^- = 1 \cdot 0 + 0 = 0$$ $$\Sigma_1^- = 1 \cdot 10 \cdot 1 + 0.01 = 10.01$$

(Slightly more uncertain due to process noise)

**Update:**

Kalman Gain: $$K_1 = \frac{10.01 \cdot 1}{1 \cdot 10.01 \cdot 1 + 0.25} = \frac{10.01}{10.26} = 0.976$$

Mean: $$\mu_1 = 0 + 0.976 \cdot (0.5 - 0) = 0.488$$

Covariance: $$\Sigma_1 = (1 - 0.976) \cdot 10.01 = 0.24$$

**Result:** $\text{bel}(\mathbf{s}_1) = \mathcal{N}(0.488, 0.24)$

**Interpretation:**

- Prior very uncertain ($\Sigma_0 = 10$) → high gain ($K_1 = 0.976$) → mostly trust measurement
- Posterior much more certain ($\Sigma_1 = 0.24$)

**Time t=2:** Measurement $z_2 = 0.6$

**Prediction:** $$\mu_2^- = 1 \cdot 0.488 = 0.488$$ $$\Sigma_2^- = 1 \cdot 0.24 \cdot 1 + 0.01 = 0.25$$

**Update:**

Kalman Gain: $$K_2 = \frac{0.25}{0.25 + 0.25} = 0.5$$

Mean: $$\mu_2 = 0.488 + 0.5 \cdot (0.6 - 0.488) = 0.488 + 0.056 = 0.544$$

Covariance: $$\Sigma_2 = (1 - 0.5) \cdot 0.25 = 0.125$$

**Result:** $\text{bel}(\mathbf{s}_2) = \mathcal{N}(0.544, 0.125)$

**Interpretation:**

- Prediction and measurement equally uncertain → gain = 0.5 → equal weighting
- Uncertainty continues to decrease

### Connections & Prerequisites

**Prerequisite Refresher on Matrix Operations:** The transpose $\mathbf{A}^T$ swaps rows and columns. For covariance propagation, $\boldsymbol{\Sigma}' = \mathbf{A}\boldsymbol{\Sigma}\mathbf{A}^T$ transforms the covariance under linear transformation by $\mathbf{A}$. The inverse $\mathbf{M}^{-1}$ satisfies $\mathbf{M}\mathbf{M}^{-1} = \mathbf{I}$. In the Kalman Gain, we invert the **innovation covariance**$(\mathbf{C}_t \boldsymbol{\Sigma}_t^- \mathbf{C}_t^T + \mathbf{Q}_t)$, which represents total uncertainty in the measurement space.

---

## 12. Concept: Kalman Filter Behavior - Prediction vs. Update Dynamics

### High-Level Intuition

The Kalman Filter exhibits interesting temporal behavior: **prediction increases uncertainty** (we're "blind" without measurements), while **update decreases uncertainty** (measurements provide information). Over time, the filter converges—early on it trusts measurements heavily, but as confidence builds in the state estimate, predictions become more trusted.

**Analogy:** Imagine walking with your eyes closed for 5 seconds (uncertainty grows), then opening them briefly (uncertainty drops), then closing again. Over many cycles, you learn the environment well enough that even with eyes closed, you're fairly confident where you are.

### Conceptual Deep Dive

**Prediction Step Dynamics:**

Starting from $\boldsymbol{\Sigma}_{t-1}$, the predicted covariance is:

$$\boldsymbol{\Sigma}_t^- = \mathbf{A}_t \boldsymbol{\Sigma}_{t-1} \mathbf{A}_t^T + \mathbf{R}_t$$

**Observation:** Even if $\mathbf{A}_t = \mathbf{I}$ (identity—no change), we have: $$\boldsymbol{\Sigma}_t^- = \boldsymbol{\Sigma}_{t-1} + \mathbf{R}_t$$

The process noise $\mathbf{R}_t$ is always **added**, so uncertainty increases (or at best stays constant if $\mathbf{R}_t = \mathbf{0}$).

**Update Step Dynamics:**

The updated covariance is: $$\boldsymbol{\Sigma}_t = (\mathbf{I} - \mathbf{K}_t \mathbf{C}_t) \boldsymbol{\Sigma}_t^-$$

Since $\mathbf{K}_t$ is designed such that $\mathbf{0} \preceq (\mathbf{I} - \mathbf{K}_t \mathbf{C}_t) \preceq \mathbf{I}$ (in the matrix sense), we have:

$$\boldsymbol{\Sigma}_t \preceq \boldsymbol{\Sigma}_t^-$$

The measurement **always reduces** (or at worst maintains) uncertainty.

**Kalman Gain Evolution:**

Recall: $K_t = \frac{\Sigma_t^-}{\Sigma_t^- + Q_t}$ (1D case)

- **Early iterations:** $\Sigma_t^-$ large → $K_t$ close to 1 → trust measurements
- **Later iterations:** $\Sigma_t^-$ small → $K_t$ close to 0 → trust predictions

This is the **convergence behavior**: the filter learns over time.

### Mathematical Formulation

**Uncertainty Trajectory:**

Define the **trace** of covariance as a scalar measure of total uncertainty: $$\text{Uncertainty}(t) = \text{tr}(\boldsymbol{\Sigma}_t)$$

The trace sums diagonal elements (variances in each dimension).

**Prediction Step:** $$\text{tr}(\boldsymbol{\Sigma}_t^-) \geq \text{tr}(\boldsymbol{\Sigma}_{t-1})$$

Uncertainty increases (or stays same).

**Update Step:** $$\text{tr}(\boldsymbol{\Sigma}_t) \leq \text{tr}(\boldsymbol{\Sigma}_t^-)$$

Uncertainty decreases (or stays same).

**Steady-State:**

For time-invariant systems (constant $\mathbf{A}, \mathbf{C}, \mathbf{R}, \mathbf{Q}$), the covariance converges to a **steady-state** value $\boldsymbol{\Sigma}_{\infty}$ satisfying:

$$\boldsymbol{\Sigma}_{\infty} = (\mathbf{I} - \mathbf{K}_{\infty} \mathbf{C}) (\mathbf{A} \boldsymbol{\Sigma}_{\infty} \mathbf{A}^T + \mathbf{R})$$

Where $\mathbf{K}_{\infty}$ is the steady-state Kalman gain.

### Worked Toy Example

**Scenario:** Track 1D position with measurements every $\Delta t = 1$ second

- Process noise: $R = 0.01$ (small—system is stable)
- Measurement noise: $Q = 1.0$ (moderate sensor noise)
- Initial uncertainty: $\Sigma_0 = 100$ (no idea where we are)

**Time Series:**

|Time|Prediction $\Sigma_t^-$|Update $\Sigma_t$|Gain $K_t$|
|---|---|---|---|
|t=0|—|100.0|—|
|t=1|100.01|0.99|0.99|
|t=2|1.0|0.50|0.50|
|t=3|0.51|0.34|0.34|
|t=5|0.35|0.26|0.26|
|t=10|0.26|0.21|0.21|
|t=50|0.21|0.17|0.17|
|t=∞|~0.20|~0.17|~0.17|

**Observations:**

1. **First update (t=1):** Huge drop from 100 → 0.99 (measurement vastly more informative than prior)
2. **Gain decreases:** From 0.99 → 0.50 → 0.17 (trusting measurements less over time)
3. **Convergence:** Around t=50, values stabilize (steady-state reached)

**Physical Interpretation:**

```
t=0:   [==================] Prior: Very uncertain
        ↓ (measurement arrives)
t=1:   [==] Posterior: Much more certain

t=1:   [==] Prior (slight growth due to process noise)
        ↓ (measurement)
t=2:   [=] Posterior (slightly more certain)

t=∞:   [=] Steady-state (prediction and update balance)
```

### Connections & Prerequisites

**Prerequisite Refresher on Matrix Trace:** For a square matrix $\mathbf{M} \in \mathbb{R}^{n \times n}$, the trace is $\text{tr}(\mathbf{M}) = \sum_{i=1}^n M_{ii}$ (sum of diagonal elements). For a covariance matrix, diagonal elements are variances, so the trace measures "total variance" across all dimensions. The trace has useful properties: $\text{tr}(\mathbf{A} + \mathbf{B}) = \text{tr}(\mathbf{A}) + \text{tr}(\mathbf{B})$ and $\text{tr}(\mathbf{A}\mathbf{B}) = \text{tr}(\mathbf{B}\mathbf{A})$.

---

## 13. Concept: Limitations and Extensions of the Kalman Filter

### High-Level Intuition

The Kalman Filter is optimal **only for linear-Gaussian systems**. Real robots operate in nonlinear worlds (e.g., rotation dynamics are nonlinear). When linearity or Gaussianity breaks down, we need extensions: the **Extended Kalman Filter (EKF)** linearizes nonlinear systems, and the **Particle Filter** handles non-Gaussian distributions.

**Analogy:** The Kalman Filter is like navigating with a map and compass on flat terrain (linear, well-behaved). The EKF is like navigating hilly terrain—you approximate each hill as locally flat. The Particle Filter is like using a swarm of scouts to explore a complex maze where no simplifications work.

### Conceptual Deep Dive

**When Kalman Filter Fails:**

1. **Nonlinear Motion Model:**
    
    - Example: $s_t = s_{t-1}^2 + a_t$ (quadratic dynamics)
    - Kalman Filter assumes $\mathbf{s}_t = \mathbf{A}\mathbf{s}_{t-1} + \mathbf{B}\mathbf{a}_t$
2. **Nonlinear Measurement Model:**
    
    - Example: Radar measures angle and distance $(r, \theta)$, not Cartesian $(x, y)$
    - Conversion: $x = r\cos\theta, y = r\sin\theta$ (nonlinear)
3. **Non-Gaussian Noise:**
    
    - Example: Outlier measurements (heavy-tailed distribution)
    - Gaussian assumption breaks down

**Extensions:**

**1. Extended Kalman Filter (EKF):**

For nonlinear systems: $$\mathbf{s}_t = f(\mathbf{s}_{t-1}, \mathbf{a}_t) + \boldsymbol{\epsilon}_t$$ $$\mathbf{z}_t = h(\mathbf{s}_t) + \boldsymbol{\delta}_t$$

**Approach:** Linearize $f$ and $h$ using first-order Taylor expansion:

$$\mathbf{A}_t = \left.\frac{\partial f}{\partial \mathbf{s}}\right|_{\mathbf{s}_{t-1}}, \quad \mathbf{C}_t = \left.\frac{\partial h}{\partial \mathbf{s}}\right|_{\boldsymbol{\mu}_t^-}$$

Then apply standard Kalman Filter with these local linearizations.

**Trade-off:** Works well for "mildly nonlinear" systems, but can diverge if linearization is poor.

**2. Unscented Kalman Filter (UKF):**

Uses "sigma points" to better capture nonlinear transformations of distributions.

**3. Particle Filter:**

Represents belief as a set of samples (particles): $$\text{bel}(\mathbf{s}_t) \approx {(\mathbf{s}_t^{(i)}, w_t^{(i)})}_{i=1}^N$$

Can handle arbitrary nonlinearities and non-Gaussian distributions.

**Trade-off:** Computationally expensive (needs many particles for high dimensions).

### Mathematical Formulation

**Extended Kalman Filter (EKF) - Brief Overview:**

**Nonlinear System:**

- Motion: $\mathbf{s}_t = f(\mathbf{s}_{t-1}, \mathbf{a}_t) + \boldsymbol{\epsilon}_t$
- Measurement: $\mathbf{z}_t = h(\mathbf{s}_t) + \boldsymbol{\delta}_t$

**EKF Prediction:**

$$\boldsymbol{\mu}_t^- = f(\boldsymbol{\mu}_{t-1}, \mathbf{a}_t)$$

$$\mathbf{F}_t = \left.\frac{\partial f}{\partial \mathbf{s}}\right|_{\boldsymbol{\mu}_{t-1}, \mathbf{a}_t}$$

$$\boldsymbol{\Sigma}_t^- = \mathbf{F}_t \boldsymbol{\Sigma}_{t-1} \mathbf{F}_t^T + \mathbf{R}_t$$

**EKF Update:**

$$\mathbf{H}_t = \left.\frac{\partial h}{\partial \mathbf{s}}\right|_{\boldsymbol{\mu}_t^-}$$

$$\mathbf{K}_t = \boldsymbol{\Sigma}_t^- \mathbf{H}_t^T (\mathbf{H}_t \boldsymbol{\Sigma}_t^- \mathbf{H}_t^T + \mathbf{Q}_t)^{-1}$$

$$\boldsymbol{\mu}_t = \boldsymbol{\mu}_t^- + \mathbf{K}_t (\mathbf{z}_t - h(\boldsymbol{\mu}_t^-))$$

$$\boldsymbol{\Sigma}_t = (\mathbf{I} - \mathbf{K}_t \mathbf{H}_t) \boldsymbol{\Sigma}_t^-$$

**Key Difference:** We compute Jacobians $\mathbf{F}_t, \mathbf{H}_t$ at each step.

### Worked Toy Example

**Problem:** Track a robot with nonlinear rotation

**State:** $\mathbf{s}_t = [x, y, \theta]$ (position and heading)

**Nonlinear Motion Model:** $$x_t = x_{t-1} + v\Delta t \cos\theta_{t-1}$$ $$y_t = y_{t-1} + v\Delta t \sin\theta_{t-1}$$ $$\theta_t = \theta_{t-1} + \omega\Delta t$$

Where $v$ = velocity, $\omega$ = angular velocity.

**Why Nonlinear?** The $\cos\theta$ and $\sin\theta$ terms are nonlinear functions of the state.

**EKF Approach:**

Compute Jacobian: $$\mathbf{F}_t = \begin{bmatrix} 1 & 0 & -v\Delta t \sin\theta_{t-1} \ 0 & 1 & v\Delta t \cos\theta_{t-1} \ 0 & 0 & 1 \end{bmatrix}$$

This matrix locally approximates how small changes in $(x, y, \theta)$ propagate.

**Example Calculation:**

At $\theta = 0$ (facing right): $$\mathbf{F}_t = \begin{bmatrix} 1 & 0 & 0 \ 0 & 1 & v\Delta t \ 0 & 0 & 1 \end{bmatrix}$$

At $\theta = \pi/2$ (facing up): $$\mathbf{F}_t = \begin{bmatrix} 1 & 0 & -v\Delta t \ 0 & 1 & 0 \ 0 & 0 & 1 \end{bmatrix}$$

The linearization changes based on current heading!

### Connections & Prerequisites

**Prerequisite Refresher on Jacobians:** For a function $\mathbf{f}: \mathbb{R}^n \to \mathbb{R}^m$, the Jacobian is the $m \times n$ matrix of partial derivatives: $\mathbf{J}_{ij} = \frac{\partial f_i}{\partial x_j}$. It represents the best linear approximation to $\mathbf{f}$ near a point. In the EKF, Jacobians replace the constant matrices $\mathbf{A}$ and $\mathbf{C}$ from the standard Kalman Filter, making them state-dependent.

---

## 14. Concept: Practical Considerations and Implementation

### High-Level Intuition

Implementing a Kalman Filter in practice requires careful attention to numerical stability, tuning of noise parameters, and handling edge cases. The filter is robust but not magic—poor parameter choices or model mismatch can lead to divergence or overconfidence.

**Analogy:** The Kalman Filter is like autopilot on a plane—it works great when properly configured, but if you tell it the wind speed is 10× what it actually is, it'll overcorrect and crash. Tuning is critical.

### Conceptual Deep Dive

**Key Implementation Challenges:**

**1. Numerical Stability:**

Matrix inversions (in Kalman Gain computation) can be numerically unstable. Solutions:

- Use **Joseph form** for covariance update (more stable): $$\boldsymbol{\Sigma}_t = (\mathbf{I} - \mathbf{K}_t\mathbf{C}_t)\boldsymbol{\Sigma}_t^-(\mathbf{I} - \mathbf{K}_t\mathbf{C}_t)^T + \mathbf{K}_t\mathbf{Q}_t\mathbf{K}_t^T$$
- Use **square root filters** that propagate $\boldsymbol{\Sigma}^{1/2}$ instead of $\boldsymbol{\Sigma}$

**2. Parameter Tuning:**

The covariances $\mathbf{R}_t$ (process noise) and $\mathbf{Q}_t$ (measurement noise) are often **not known exactly**. They must be:

- Estimated from data (offline calibration)
- Tuned empirically (adjust until performance is good)
- Adapted online (in adaptive Kalman Filters)

**Guidelines:**

- **Underestimating $\mathbf{Q}_t$** (trusting sensors too much) → filter becomes overconfident, ignores good measurements
- **Overestimating $\mathbf{Q}_t$** → filter becomes too cautious, oscillates
- **Underestimating $\mathbf{R}_t$** → filter trusts predictions too much
- **Overestimating $\mathbf{R}_t$** → filter has high uncertainty, sluggish convergence

**3. Observability:**

Not all states may be observable from measurements. Example: Measuring only position but trying to estimate velocity and acceleration.

**Observability Condition:** A system is observable if you can uniquely determine the state from a sequence of measurements.

**4. Data Association:**

In multi-object tracking, you must match measurements to tracks. Wrong associations corrupt the filter.

### Mathematical Formulation

**Process vs. Measurement Noise Balance:**

The **Kalman Gain** encodes the noise trade-off:

$$\mathbf{K}_t = \boldsymbol{\Sigma}_t^- \mathbf{C}_t^T (\mathbf{C}_t \boldsymbol{\Sigma}_t^- \mathbf{C}_t^T + \mathbf{Q}_t)^{-1}$$

Rewrite as: $$\mathbf{K}_t = \boldsymbol{\Sigma}_t^- \mathbf{C}_t^T [\mathbf{C}_t \boldsymbol{\Sigma}_t^- \mathbf{C}_t^T + \mathbf{Q}_t]^{-1}$$

**Two extremes:**

1. **$\mathbf{Q}_t \to 0$ (perfect sensors):** $$\mathbf{K}_t \to [\mathbf{C}_t^T\mathbf{C}_t]^{-1}\mathbf{C}_t^T$$ Filter trusts measurements completely.
    
2. **$\mathbf{Q}_t \to \infty$ (useless sensors):** $$\mathbf{K}_t \to \mathbf{0}$$ Filter ignores measurements, relies on prediction.
    

**Divergence Detection:**

Monitor the **innovation sequence** $\mathbf{r}_t = \mathbf{z}_t - \mathbf{C}_t\boldsymbol{\mu}_t^-$:

- Should be zero-mean Gaussian with covariance $\mathbf{S}_t = \mathbf{C}_t\boldsymbol{\Sigma}_t^-\mathbf{C}_t^T + \mathbf{Q}_t$
- If innovations are consistently large, filter may be diverging (model mismatch)

### Worked Toy Example

**Scenario:** Tracking a car with GPS (measurement) and odometry (motion model)

**State:** $\mathbf{s} = [x, v]$ (position and velocity)

**Motion Model:** $$\begin{bmatrix} x_t \ v_t \end{bmatrix} = \begin{bmatrix} 1 & \Delta t \ 0 & 1 \end{bmatrix} \begin{bmatrix} x_{t-1} \ v_{t-1} \end{bmatrix} + \text{noise}$$

**Measurement:** GPS measures position only: $$z_t = \begin{bmatrix} 1 & 0 \end{bmatrix} \begin{bmatrix} x_t \ v_t \end{bmatrix} + \text{noise}$$

**Tuning Experiment:**

**Case 1: Accurate Noise Estimates**

- $\mathbf{R} = \begin{bmatrix} 0.1 & 0 \ 0 & 0.1 \end{bmatrix}$, $\mathbf{Q} = [1.0]$
- Result: Filter converges smoothly, tracks ground truth well

**Case 2: Overconfident in Sensors ($\mathbf{Q}$ too small)**

- $\mathbf{R} = \begin{bmatrix} 0.1 & 0 \ 0 & 0.1 \end{bmatrix}$, $\mathbf{Q} = [0.01]$
- Result: Filter "jumps" to every noisy measurement, high variance in estimates

**Case 3: Underconfident in Model ($\mathbf{R}$ too large)**

- $\mathbf{R} = \begin{bmatrix} 10 & 0 \ 0 & 10 \end{bmatrix}$, $\mathbf{Q} = [1.0]$
- Result: Filter has large uncertainty, slow to converge, doesn't trust predictions

**Best Practice:**

Start with **conservative estimates** (larger noise values), then gradually reduce based on observed performance. Use validation data to tune.

### Connections & Prerequisites

**Prerequisite Refresher on Observability:** A system is observable if the observability matrix $\mathcal{O} = [\mathbf{C}^T, (\mathbf{CA})^T, (\mathbf{CA}^2)^T, \ldots]^T$ has full column rank. Intuitively, this means you can "see" all states through measurements over time. For the position-velocity example, measuring only position makes the system observable because velocity affects how position changes over time.

---

## Key Takeaways & Formulas

### Core Principles

1. **Recursive Estimation Philosophy:**
    
    - New estimate = Previous estimate + Gain × Innovation
    - This pattern appears throughout: from sample means to Kalman Filters
2. **Two-Step Structure:**
    
    - **Prediction (Blind):** Use motion model to forecast next state → uncertainty increases
    - **Update (Correction):** Use measurement to refine forecast → uncertainty decreases
3. **Markovian Assumption:**
    
    - Current state is a sufficient statistic for the past
    - Simplifies $P(\mathbf{s}_t | \text{history})$ to $P(\mathbf{s}_t | \mathbf{s}_{t-1}, \mathbf{a}_t)$

### Essential Formulas

**Bayes Filter (Discrete):**

$$\text{bel}^-(\mathbf{s}_t) = \sum_{\mathbf{s}_{t-1}} P(\mathbf{s}_t|\mathbf{s}_{t-1}, \mathbf{a}_t) \cdot \text{bel}(\mathbf{s}_{t-1})$$

$$\text{bel}(\mathbf{s}_t) = \eta \cdot P(\mathbf{z}_t|\mathbf{s}_t) \cdot \text{bel}^-(\mathbf{s}_t)$$

**Kalman Filter (Continuous):**

_Prediction:_ $$\boldsymbol{\mu}_t^- = \mathbf{A}_t\boldsymbol{\mu}_{t-1} + \mathbf{B}_t\mathbf{a}_t$$ $$\boldsymbol{\Sigma}_t^- = \mathbf{A}_t\boldsymbol{\Sigma}_{t-1}\mathbf{A}_t^T + \mathbf{R}_t$$

_Update:_ $$\mathbf{K}_t = \boldsymbol{\Sigma}_t^-\mathbf{C}_t^T(\mathbf{C}_t\boldsymbol{\Sigma}_t^-\mathbf{C}_t^T + \mathbf{Q}_t)^{-1}$$ $$\boldsymbol{\mu}_t = \boldsymbol{\mu}_t^- + \mathbf{K}_t(\mathbf{z}_t - \mathbf{C}_t\boldsymbol{\mu}_t^-)$$ $$\boldsymbol{\Sigma}_t = (\mathbf{I} - \mathbf{K}_t\mathbf{C}_t)\boldsymbol{\Sigma}_t^-$$

**Product of Gaussians (1D):**

$$\mu_{\text{post}} = \frac{\sigma_z^2\mu_{\text{prior}} + \sigma_{\text{prior}}^2 z}{\sigma_{\text{prior}}^2 + \sigma_z^2}$$

$$\sigma_{\text{post}}^2 = \frac{\sigma_{\text{prior}}^2 \sigma_z^2}{\sigma_{\text{prior}}^2 + \sigma_z^2}$$

### Exam-Relevant Intuitions

- **Why the filter is "Bayesian":** The update step implements Bayes' rule (likelihood × prior → posterior)
- **Why it's a "filter":** Continuous input stream (measurements) → continuous output stream (state estimates)
- **Kalman Gain behavior:** High when uncertain about prediction, low when confident
- **Uncertainty evolution:** Increases during prediction, decreases during update
- **Convergence:** Early iterations trust measurements; later iterations trust predictions more

### Common Pitfalls

1. **Forgetting normalization** in discrete Bayes Filter update (probabilities must sum to 1)
2. **Confusing prior and posterior** in the two-step cycle
3. **Assuming Kalman Filter works for nonlinear systems** (need EKF or UKF)
4. **Poor noise parameter tuning** leads to divergence or oscillation
5. **Matrix dimension errors** in multi-dimensional implementations

---

**Preparation Tip:** Focus on understanding the **conceptual flow** (why each step is needed) rather than memorizing formulas. Practice deriving the recursive update from first principles. For the exam, be prepared for intuitive questions about when to trust measurements vs. predictions, how uncertainty evolves, and what the Markovian assumption means in practice.