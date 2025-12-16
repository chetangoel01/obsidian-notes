---
tags:
  - study-guide
  - course/artificial-intelligence
---
### Executive Summary

This inaugural lecture establishes the foundational framework for understanding AI systems. It introduces the canonical **AI agent architecture** with its core subsystems (perception, reasoning over time, planning, and control), then transitions into the mathematical foundations of **statistical learning theory**. The lecture covers supervised learning fundamentals including hypothesis selection, loss function design, the critical challenge of overfitting, regularization techniques, and gradient descent optimization—all illustrated through a polynomial regression example.

---

## 1. Concept: AI Agent Architecture

### High-Level Intuition

An AI agent is a system that perceives its environment through sensors, makes decisions about actions to take, and executes those actions to achieve goals—think of it as the "brain" that controls a robot or autonomous vehicle.

### Conceptual Deep Dive

The canonical AI agent consists of five interconnected subsystems operating in a continuous loop:

**Environment**: The external world containing all entities and state outside the agent's physical boundaries. This includes other agents, obstacles, dynamic objects, and environmental conditions. The environment is typically **stochastic** (probabilistic/unpredictable) and only **partially observable** (the agent cannot see everything).

**Perception Subsystem**: Receives high-dimensional sensory input **X ∈ ℝⁿ** from the environment (e.g., camera images with n = width × height pixels) and produces predictions **Ŷ** about the state (e.g., "Is there a pedestrian?"). These predictions are produced by **statistical machines** that are reflexive and probabilistic—they can make temporary mistakes.

**Probabilistic Reasoning Over Time**: Corrects for the statistical nature of perception by observing predictions over multiple time steps. This subsystem performs **tracking** and handles **missing data** (occlusions) to produce reliable **state estimates**.

**Planning**: Uses the state estimate to determine a sequence of actions that achieve a goal state. This could be:

- **Global planning**: High-level path from point A to point B
- **Local planning**: Trajectory generation (e.g., the green curve in autonomous driving)

**Controller**: Translates planned actions into low-level commands for actuators (e.g., voltages for motors, steering angles). The executed action modifies the environment state, which is then observed through new sensory input, closing the loop.

This architecture also includes an optional **reward signal** from the environment, used in reinforcement learning to evaluate action quality.

### Mathematical Formulation

**Sensory Input:** $$\mathbf{X} \in \mathbb{R}^n$$

Where $\mathbf{X}$ is the sensory vector (e.g., flattened image), and $n$ is the dimensionality (e.g., $n = 320 \times 320 = 102,400$ for an image).

**Perception Function:** $$\hat{Y} = g(\mathbf{X}, \mathbf{W})$$

Where $\hat{Y}$ is the predicted target variable (e.g., binary: 0 = no pedestrian, 1 = pedestrian), and $\mathbf{W}$ represents the model parameters.

### Worked Toy Example

**Scenario:** A simple robot vacuum cleaner

- **Environment**: Floor with obstacles (chairs, walls)
- **Perception**: Camera detects obstacles → predicts "obstacle at position (x, y)"
- **Reasoning over time**: Tracks obstacles as robot moves; if obstacle disappears briefly (behind leg), infers it's still there
- **Planning**: Computes path to cover all floor space, avoiding obstacles
- **Controller**: Sends motor commands (left wheel: 5V, right wheel: 3V) to execute path
- **Loop**: New camera input shows robot has moved → update state estimate → replan if needed

### Connections & Prerequisites

**Prerequisite Refresher**: This concept requires understanding that **sensors produce data**, which can be represented as vectors (ordered lists of numbers), and that **computation** can transform these vectors into decisions. No advanced mathematics is needed yet—just the intuition that systems can process inputs to produce outputs.

---

## 2. Concept: The Statistical Learning Machine (Vapnik's Framework)

### High-Level Intuition

Machine learning is fundamentally about guessing a function that maps inputs to outputs based on limited examples—like trying to figure out the recipe for a dish by tasting a few samples and observing the ingredients.

### Conceptual Deep Dive

Vladimir Vapnik formalized the learning problem with a block diagram consisting of:

**Data Generator**: Nature (or a data collection process) produces inputs **X** according to some unknown distribution. We have no control over what data we receive.

**Target Function f**: An unknown, potentially very complex function that maps **X → Y**. This is the "ground truth" relationship we're trying to learn. We never know **f** directly.

**Sampler**: Provides us with a finite set of **m examples** (pairs of inputs and outputs): **{(X₁, Y₁), (X₂, Y₂), ..., (Xₘ, Yₘ)}**. The sampler can introduce challenges like **class imbalance** (e.g., 10 examples of class 1, 1 million of class 2) or **missing data**.

---

**The horizontal line** divides what we cannot control (above) from what we design (below):

**Hypothesis Set**: Our "guesses" about what **f** might look like. These are functions **g(X, W)** parameterized by a vector **W**. By changing **W**, we generate different hypotheses.

**Objective/Loss Function** ℒ: Evaluates how good a hypothesis is by comparing predictions **Ŷ** to true labels **Y**. Returns a scalar **L** (a single number indicating quality).

**Optimization Algorithm**: Iteratively adjusts **W** to minimize **L**, eventually producing **W*** (the optimal parameters).

**Final Hypothesis**: __g(X, W_)_* is the model we deploy for making predictions on new, unseen data.

### Mathematical Formulation

**Training Dataset:** $$\mathcal{D} = {(\mathbf{X}_i, Y_i)}_{i=1}^{m}$$

Where $m$ is the number of training examples.

**Hypothesis:** $$g(\mathbf{X}, \mathbf{W})$$

Where $\mathbf{W} = [w_0, w_1, ..., w_M]^T$ is the parameter vector.

**Objective Function:** $$L = \mathcal{L}(g(\mathbf{X}, \mathbf{W}), Y)$$

Where $L \in \mathbb{R}^+$ is a positive scalar representing the loss.

**Optimization Goal:** $$\mathbf{W}^* = \arg\min_{\mathbf{W}} L$$

Find the parameters $\mathbf{W}^*$ that minimize the loss $L$.

### Worked Toy Example

**Scenario:** Predicting house prices

- **Data Generator**: Real estate market produces listings
- **Target Function f**: Unknown "true" relationship between square footage and price
- **Sampler**: Gives us 10 houses: {(1000 sq ft, $200k), (1500 sq ft, $280k), ...}
- **Hypothesis Set**: We guess f is a straight line: **g(X, W) = w₀ + w₁ × X**
- **Loss Function**: Mean Squared Error (explained later)
- **Optimization**: Find __w₀_ = $50k_*, __w₁_ = $200/sq ft_* that minimize error
- **Final Model**: For a new house with 1800 sq ft → predict $50k + $200 × 1800 = $410k

### Connections & Prerequisites

**Prerequisite Refresher**: Understanding that a **function** is a rule that transforms inputs into outputs (e.g., **f(x) = 2x** doubles any input). A **parameter** is a tunable knob in a function (like the "2" in **2x**). **Optimization** means finding the best value for these knobs.

---

## 3. Concept: Supervised Learning Problem Types

### High-Level Intuition

Supervised learning comes in two flavors based on what we're predicting: **regression** (predicting numbers, like "How much?") and **classification** (predicting categories, like "Which type?").

### Conceptual Deep Dive

The **type of supervised learning problem** is determined by the nature of the target variable **Y**:

**Regression**: When **Y** is a **continuous real-valued number** (Y ∈ ℝ). Examples:

- Predicting house prices ($200k, $315k, $1.2M)
- Estimating temperature (72.5°F, 89.2°F)
- Forecasting stock prices

**Classification**: When **Y** is a **discrete category** from a finite set. Examples:

- **Binary classification**: Y ∈ {0, 1} (e.g., "pedestrian" vs. "no pedestrian")
- **Multi-class classification**: Y ∈ {0, 1, 2, ..., K} (e.g., "cat", "dog", "bird")

The lecture focuses on **regression** using a polynomial fitting example.

### Mathematical Formulation

**Regression:** $$Y \in \mathbb{R}$$

The target variable is a real number on the continuous number line.

**Binary Classification:** $$Y \in {0, 1}$$

The target variable is one of two discrete categories.

**Multi-class Classification:** $$Y \in {0, 1, 2, \ldots, K-1}$$

The target variable is one of $K$ discrete categories.

### Worked Toy Example

**Regression Example:**

- **Task**: Predict student exam score based on study hours
- **Input X**: Hours studied (2.5, 4.0, 6.5)
- **Output Y**: Exam score (65.2, 78.8, 91.5) ← continuous numbers

**Classification Example:**

- **Task**: Predict if email is spam based on word counts
- **Input X**: Number of times "free" appears (0, 5, 12)
- **Output Y**: Spam label (0, 0, 1) ← discrete categories (0=not spam, 1=spam)

### Connections & Prerequisites

**Prerequisite Refresher**: Recall that **continuous variables** can take any value in a range (like measuring liquid in a container—2.5 liters, 2.51 liters, 2.512 liters, etc.), while **discrete variables** can only take specific separate values (like counting people—1, 2, 3, but never 2.5 people).

---

## 4. Concept: Hypothesis Sets and Model Complexity

### High-Level Intuition

A hypothesis set is like a family of candidate functions—think of it as having different-order polynomial "lenses" through which to view the data, from simple straight lines to complex curves.

### Conceptual Deep Dive

For the polynomial regression example with **one feature** (e.g., square footage), we define hypothesis sets of increasing **model complexity** denoted by order **M**:

**Constant Hypothesis (M=0):** $$g_0(X, \mathbf{W}) = w_0$$ A flat horizontal line. Predicts the same value regardless of input. This is the simplest possible model.

**Linear Hypothesis (M=1):** $$g_1(X, \mathbf{W}) = w_0 + w_1 X$$ A straight line with slope $w_1$ and intercept $w_0$. Can model linear relationships.

**Quadratic Hypothesis (M=2):** $$g_2(X, \mathbf{W}) = w_0 + w_1 X + w_2 X^2$$ A parabola. Can model curves with one bend.

**Cubic Hypothesis (M=3):** $$g_3(X, \mathbf{W}) = w_0 + w_1 X + w_2 X^2 + w_3 X^3$$ Can model more complex curves with up to two bends.

**General M-th Order Polynomial:** $$g_M(X, \mathbf{W}) = \sum_{j=0}^{M} w_j X^j = w_0 + w_1 X + w_2 X^2 + \cdots + w_M X^M$$

Key insight: **Each hypothesis set contains infinitely many hypotheses** by varying the parameter vector **W**. Changing **W** shifts, rotates, or reshapes the curve.

**Model Complexity**: As **M** increases, the model can fit more complex patterns, but this is a double-edged sword (see Overfitting section).

### Mathematical Formulation

**Parameter Vector for M-th Order Polynomial:** $$\mathbf{W} = [w_0, w_1, w_2, \ldots, w_M]^T \in \mathbb{R}^{M+1}$$

The parameter vector has $M+1$ dimensions (M+1 tunable knobs).

**Hypothesis:** $$g_M(X, \mathbf{W}) = \sum_{j=0}^{M} w_j X^j$$

Where $X^j$ represents $X$ raised to the power $j$ (with $X^0 = 1$ by convention).

### Worked Toy Example

Given data points: (1, 2), (2, 5), (3, 8)

**M=0 Hypothesis**: $g_0 = w_0 = 5$ (flat line at Y=5)  
**M=1 Hypothesis**: $g_1 = 1 + 3X$ (straight line)

- At X=1: Ŷ = 1 + 3(1) = 4
- At X=2: Ŷ = 1 + 3(2) = 7
- At X=3: Ŷ = 1 + 3(3) = 10

**M=2 Hypothesis**: $g_2 = -1 + 2X + 1X^2$ (parabola)

- At X=1: Ŷ = -1 + 2(1) + 1(1)² = 2 ✓ (exact match!)
- At X=2: Ŷ = -1 + 2(2) + 1(4) = 7 (error = |5-7| = 2)
- At X=3: Ŷ = -1 + 2(3) + 1(9) = 14 (error = |8-14| = 6)

Different values of **W** produce different curves through the data.

### Connections & Prerequisites

**Prerequisite Refresher**: A **polynomial** is a mathematical expression formed by adding together terms, where each term is a coefficient multiplied by a variable raised to a non-negative integer power. For example, $5 + 3x + 2x^2$ is a polynomial. The **degree** of a polynomial is the highest power (2 in this example). Polynomials can approximate smooth curves—the **Taylor series** in calculus formalizes this idea.

---

## 5. Concept: Mean Squared Error (MSE) Loss Function

### High-Level Intuition

We need a "report card" that gives our hypothesis a single grade based on how close its predictions are to the true answers—Mean Squared Error does this by averaging the squared distances between predictions and reality.

### Conceptual Deep Dive

The **loss function** is our evaluation metric. For regression problems, the standard choice is **Mean Squared Error (MSE)**.

**Design Rationale:**

1. **Difference**: We want small differences between predictions **Ŷᵢ** and true labels **Yᵢ**. The error for example **i** is **(Ŷᵢ - Yᵢ)**.
    
2. **Squared**: We square the difference for two reasons:
    
    - Makes all errors positive (we don't care if we overestimate or underestimate—both are equally bad)
    - Penalizes large errors more heavily (error of 10 contributes 100, while error of 1 contributes 1)
3. **Mean**: We average across all **m** training examples to get a single number. This prevents the loss from growing just because we have more data.
    

**Weak Assumption of Local Smoothness**: MSE implicitly assumes the target function **f** is **locally smooth** (no sudden discontinuities). This means that for similar inputs **X**, the outputs **Y** should be similar. Therefore, if our model predicts well on the training **X**'s, it should predict well on similar (nearby) test **X**'s.

### Mathematical Formulation

**Mean Squared Error:** $$L_{MSE} = \frac{1}{m} \sum_{i=1}^{m} (\hat{Y}_i - Y_i)^2$$

Where:

- $m$ = number of training examples
- $\hat{Y}_i = g(\mathbf{X}_i, \mathbf{W})$ = prediction for example $i$
- $Y_i$ = true label for example $i$
- $(\hat{Y}_i - Y_i)^2$ = squared error for example $i$

**Alternative notation (emphasizing dependence on W):** $$L_{MSE}(\mathbf{W}) = \frac{1}{m} \sum_{i=1}^{m} [g(\mathbf{X}_i, \mathbf{W}) - Y_i]^2$$

### Worked Toy Example

**Dataset:**

|i|Xᵢ|Yᵢ (true)|
|---|---|---|
|1|1.0|3.0|
|2|2.0|5.0|
|3|3.0|7.0|

**Hypothesis**: $g(X, \mathbf{W}) = 2 + 1.5X$ (we guessed **w₀=2**, **w₁=1.5**)

**Step 1 - Compute predictions:**

- Ŷ₁ = 2 + 1.5(1) = 3.5
- Ŷ₂ = 2 + 1.5(2) = 5.0
- Ŷ₃ = 2 + 1.5(3) = 6.5

**Step 2 - Compute squared errors:**

- (Ŷ₁ - Y₁)² = (3.5 - 3.0)² = (0.5)² = 0.25
- (Ŷ₂ - Y₂)² = (5.0 - 5.0)² = (0.0)² = 0.00
- (Ŷ₃ - Y₃)² = (6.5 - 7.0)² = (-0.5)² = 0.25

**Step 3 - Compute mean:** $$L_{MSE} = \frac{0.25 + 0.00 + 0.25}{3} = \frac{0.50}{3} = 0.167$$

The loss is **0.167**. Lower values indicate better fit.

### Connections & Prerequisites

**Prerequisite Refresher**: **Mean** (average) of a set of numbers is their sum divided by how many there are: $\text{mean of } {2, 5, 8} = \frac{2+5+8}{3} = 5$. **Squaring** a number means multiplying it by itself: $(-3)^2 = (-3) \times (-3) = 9$. Note that squaring always gives a non-negative result, which is why we use it to make errors positive.

---

## 6. Concept: Overfitting - Detection

### High-Level Intuition

Overfitting is like memorizing answers to practice problems without understanding the underlying concepts—you ace the practice test but fail when given new questions. We detect this by holding out some data as a "surprise quiz."

### Conceptual Deep Dive

**Overfitting** occurs when a model is too complex for the underlying data, capturing noise and random fluctuations rather than the true underlying pattern. The model performs excellently on training data but poorly on new, unseen data.

**The Problem**: If we only evaluate models on training data, the most complex model (e.g., M=9 polynomial passing through all 10 points) would always win because it achieves **zero training error**. But this model is "too wiggly" and makes terrible predictions on new data (e.g., predicting $1.3M for a house that should cost $300k).

**The Solution - Train/Test Split**:

1. **Split the dataset** into two disjoint subsets:
    
    - **Training Set**: ~80% of data (**m_train** examples) → used to optimize **W**
    - **Test Set**: ~20% of data (**m_test** examples) → **held out**, never used during training
2. **Compute two losses**:
    
    - **L_train**: MSE computed on training set (measures how well we fit the training data)
    - **L_test**: MSE computed on test set (measures how well we **generalize** to new data)
3. **Detection Rule**:
    
    - If **L_test ≈ L_train** → model generalizes well ✓
    - If **L_test >> L_train** → **overfitting detected** ✗

**Observable Pattern**: As model complexity **M** increases from 0 to 9:

- **L_train** continuously decreases (more complex models fit training data better)
- **L_test** initially decreases, reaches a minimum around M=3-4, then **shoots up** at M=8,9 (the "overfitting zone")

The dramatic gap between **L_train** and **L_test** is the smoking gun of overfitting.

### Mathematical Formulation

**Train/Test Split:** $$\mathcal{D} = \mathcal{D}_{train} \cup \mathcal{D}_{test}$$ $$\mathcal{D}_{train} \cap \mathcal{D}_{test} = \emptyset$$

Where $\mathcal{D}_{train}$ contains $m_{train}$ examples and $\mathcal{D}_{test}$ contains $m_{test}$ examples, with $m_{train} + m_{test} = m$.

**Training Loss:** $$L_{train} = \frac{1}{m_{train}} \sum_{i \in \mathcal{D}_{train}} (\hat{Y}_i - Y_i)^2$$

**Test Loss:** $$L_{test} = \frac{1}{m_{test}} \sum_{i \in \mathcal{D}_{test}} (\hat{Y}_i - Y_i)^2$$

**Overfitting Indicator:** $$\text{If } L_{test} - L_{train} > \epsilon \text{ (some threshold)}, \text{ overfitting is detected}$$

### Worked Toy Example

**Original Dataset**: 10 house prices  
**Split**: 8 for training, 2 for testing (held out)

**Training Set** (8 points): (1000, $200k), (1200, $220k), (1400, $260k), (1500, $280k), (1800, $320k), (2000, $360k), (2200, $400k), (2500, $450k)

**Test Set** (2 points - HIDDEN during training): (1100, $210k), (1900, $340k)

**Results for different models:**

|Model|L_train|L_test|Gap|Status|
|---|---|---|---|---|
|M=0 (flat line)|12,500|12,800|300|Underfitting|
|M=1 (straight line)|400|450|50|Good ✓|
|M=3 (cubic)|150|180|30|Good ✓|
|M=8|5|8,200|8,195|**Overfitting!** ✗|

The M=8 model has nearly zero training error but massive test error → **detected overfitting**.

### Connections & Prerequisites

**Prerequisite Refresher**: **Generalization** means applying learned knowledge to new situations. A student who understands calculus can solve new derivative problems, not just memorized ones. In ML, a model **generalizes** if it performs well on data it has never seen before. The **test set** simulates this "unseen data" scenario.

---

## 7. Concept: Regularization (L2/Ridge Regression)

### High-Level Intuition

Regularization is like adding a "complexity penalty" to discourage the model from being overly complicated—it's a soft constraint that nudges the optimizer toward simpler solutions without explicitly forbidding complex ones.

### Conceptual Deep Dive

**The Root Cause**: Overfitting is driven by minimizing **MSE alone**, which rewards arbitrarily complex models that perfectly fit training data (even noise).

**The Solution**: Modify the loss function to include a **penalty term** that punishes model complexity.

**Regularized Loss Function:** $$L_{total} = L_{MSE} + L_{penalty}$$

**Where does the penalty come from?** We need an **organic signal**—something measurable within the problem itself that indicates overfitting.

**Key Observation**: When a high-order polynomial (e.g., M=9) overfits, its parameter vector **W** exhibits **huge swings**—some coefficients become very large positive or negative numbers to force the curve through all data points. In contrast, simpler models (M=3) have modest coefficient values.

**L2 Regularization (Ridge Regression)**: We use the **squared L2 norm** of **W** as the penalty:

$$L_{penalty} = \lambda |\mathbf{W}|_2^2 = \lambda \sum_{j=0}^{M} w_j^2$$

Where:

- **λ** (lambda) is a **hyperparameter** controlling the penalty strength (set by us, not by the optimizer)
- When coefficients **w_j** are large → penalty is large → total loss is large → optimizer avoids this region
- When coefficients are small → penalty is small → optimizer is free to fit the data

**Effect of λ**:

- **λ ≈ 0**: Penalty is negligible → behaves like unregularized MSE → overfitting
- **λ optimal**: Balances fit to data with model simplicity → good generalization
- **λ very large**: Penalty dominates → coefficients forced toward zero → **underfitting** (model too simple to capture patterns)

**Key Insight**: We "bet on" a complex model (M=9) but use regularization to **softly switch off** higher-order terms. The result resembles a simpler model (M=3) but without explicitly limiting model complexity.

### Mathematical Formulation

**Regularized Loss (Ridge Regression):** $$L_{reg}(\mathbf{W}) = \frac{1}{m} \sum_{i=1}^{m} [g(\mathbf{X}_i, \mathbf{W}) - Y_i]^2 + \lambda \sum_{j=0}^{M} w_j^2$$

Where:

- First term: **Mean Squared Error** (data fit)
- Second term: **L2 penalty** (complexity penalty)
- $\lambda \in \mathbb{R}^+$: **Hyperparameter** controlling the regularization strength

**Optimization Goal:** $$\mathbf{W}_{reg}^* = \arg\min_{\mathbf{W}} L_{reg}(\mathbf{W})$$

Find parameters that minimize the regularized loss (trading off fit and complexity).

**L2 Norm Squared:** $$|\mathbf{W}|_2^2 = \sum_{j=0}^{M} w_j^2 = w_0^2 + w_1^2 + w_2^2 + \cdots + w_M^2$$

This is always non-negative and grows with the magnitude of the coefficients.

### Worked Toy Example

**Scenario**: M=9 polynomial on 10 data points (prone to overfitting)

**Unregularized (λ=0)**:

- **W_unreg** = [5, 300, -1200, 8000, -15000, 12000, -5000, 1000, -80, 2]
- Huge coefficient swings! Curve oscillates wildly.
- **L_train** = 0.01, **L_test** = 5600 → overfitting

**Regularized (λ=0.01)**:

- **W_reg** = [2, 15, -5, 8, 0.1, 0.05, 0.01, 0.001, 0.0001, 0.00001]
- Higher-order terms (w₅ through w₉) are nearly zero → effectively M≈3
- **L_train** = 12, **L_test** = 18 → generalization improved!

**Penalty Calculation**:

- **P_unreg** = 5² + 300² + 1200² + ... ≈ 3.2 × 10⁸ (massive!)
- **P_reg** = 2² + 15² + 5² + 8² + (tiny terms) ≈ 318 (small)

The optimizer **avoids the high-penalty region** (large coefficients), naturally finding simpler solutions.

### Connections & Prerequisites

**Prerequisite Refresher**: A **norm** is a mathematical measure of the "size" or "magnitude" of a vector. The **L2 norm** (Euclidean norm) is the square root of the sum of squared components: $|\mathbf{v}| = \sqrt{v_1^2 + v_2^2 + \cdots}$. For the vector [3, 4], the L2 norm is $\sqrt{3^2 + 4^2} = \sqrt{25} = 5$. The **L2 norm squared** just omits the square root: $|\mathbf{v}|^2 = 3^2 + 4^2 = 25$.

---

## 8. Concept: Gradient Descent Optimization

### High-Level Intuition

Gradient descent is like hiking down a mountain in the fog—you can't see the bottom, but you can feel the slope beneath your feet, so you repeatedly take small steps in the direction of steepest descent until you reach a valley.

### Conceptual Deep Dive

**The Challenge**: We have a loss function **L(W)** that we want to minimize. For complex models, this function has thousands to billions of dimensions—we can't visualize it or solve it analytically.

**The Solution**: An **iterative algorithm** that starts from a random guess and repeatedly improves it.

**Gradient Descent Algorithm**:

1. **Initialize**: Start with a random parameter vector **W₀** (random guess)
    
2. **Compute Gradient**: Calculate **∇L(W)**, which is a vector pointing in the direction of **steepest increase** in loss
    
3. **Update Rule**: Move in the **opposite direction** (steepest decrease): $$\mathbf{W}_{t+1} = \mathbf{W}_t - \eta \nabla L(\mathbf{W}_t)$$
    
    Where:
    
    - **η** (eta) is the **learning rate** (step size)
    - The negative sign means we go **downhill**
4. **Iterate**: Repeat steps 2-3 until convergence (loss stops decreasing significantly)
    

**Key Components**:

**Gradient ∇L(W)**: A vector of partial derivatives—one for each parameter: $$\nabla L(\mathbf{W}) = \left[\frac{\partial L}{\partial w_0}, \frac{\partial L}{\partial w_1}, \ldots, \frac{\partial L}{\partial w_M}\right]^T$$

Each component tells us how the loss changes if we nudge that particular parameter.

**Learning Rate η**: Controls step size.

- Too small → convergence is very slow (takes forever)
- Too large → we overshoot the minimum, potentially diverging
- Just right → steady progress toward minimum

**Convergence**: At a **local minimum**, the gradient becomes approximately zero (slope is flat in all directions). The algorithm oscillates near this point.

**Limitations**:

- May converge to **local minima** (valleys) rather than the **global minimum** (lowest valley)
- **Good local minima** (close to global minimum) are acceptable
- **Bad local minima** (far from global minimum) lead to poor performance

### Mathematical Formulation

**Gradient Descent Update Rule:** $$\mathbf{W}_{t+1} = \mathbf{W}_t - \eta \nabla L(\mathbf{W}_t)$$

Where:

- $t$ = iteration number (0, 1, 2, ...)
- $\mathbf{W}_t$ = parameter vector at iteration $t$
- $\eta \in \mathbb{R}^+$ = learning rate (small positive number, e.g., 0.01)
- $\nabla L(\mathbf{W}_t)$ = gradient (vector of partial derivatives)

**Gradient for MSE Loss (Linear Regression Example):**

For the simple case $L(\mathbf{W}) = w^2$: $$\frac{\partial L}{\partial w} = 2w$$

At the minimum ($w^* = 0$), the derivative is zero: $\frac{\partial L}{\partial w} = 0$.

### Worked Toy Example

**Toy Loss Function**: $L(w) = w^2$ (one parameter, parabola)  
**Goal**: Find $w^* = 0$ (the minimum)

**Initialization**: $w_0 = 4$ (random starting point)  
**Learning Rate**: $\eta = 0.3$

**Iteration 1**:

- Gradient: $\nabla L(w_0) = 2w_0 = 2(4) = 8$
- Update: $w_1 = w_0 - \eta \nabla L(w_0) = 4 - 0.3(8) = 4 - 2.4 = 1.6$
- Loss: $L(w_1) = (1.6)^2 = 2.56$

**Iteration 2**:

- Gradient: $\nabla L(w_1) = 2(1.6) = 3.2$
- Update: $w_2 = 1.6 - 0.3(3.2) = 1.6 - 0.96 = 0.64$
- Loss: $L(w_2) = (0.64)^2 = 0.41$

**Iteration 3**:

- Gradient: $\nabla L(w_2) = 2(0.64) = 1.28$
- Update: $w_3 = 0.64 - 0.3(1.28) = 0.64 - 0.384 = 0.256$
- Loss: $L(w_3) = (0.256)^2 = 0.066$

**Convergence**: After ~10 iterations, $w_t \approx 0.001 \approx 0$ → minimum reached!

### Connections & Prerequisites

**Prerequisite Refresher**: The **derivative** of a function $f(x)$ at point $x$ is the **slope of the tangent line** to $f$ at that point. It tells us the rate of change: if $f'(x) > 0$, the function is increasing; if $f'(x) < 0$, it's decreasing; if $f'(x) = 0$, we're at a peak or valley. For $f(x) = x^2$, the derivative is $f'(x) = 2x$. The **partial derivative** $\frac{\partial f}{\partial x_i}$ measures the rate of change with respect to one variable while holding others constant.

---

## 9. Concept: Visualizing Optimization - Parameter Space vs. Hypothesis Space

### High-Level Intuition

When we optimize a model, there are two parallel "worlds" to think about: the abstract **parameter space** where the algorithm searches, and the concrete **hypothesis space** where we see the actual curves fitting our data.

### Conceptual Deep Dive

This concept bridges the abstract optimization process with its practical effect on predictions.

**Parameter Space**: An M-dimensional space where each axis represents one parameter (w₀, w₁, ..., wₘ). Each **point** in this space represents a specific combination of parameter values, i.e., a specific **W** vector.

**Hypothesis Space**: The original X-Y space where we plot data points and fitted curves. Each **curve** in this space corresponds to a specific hypothesis (a specific **W** from parameter space).

**The Mapping**: There is a one-to-one correspondence:

- **Point in parameter space** ↔ **Curve in hypothesis space**
- Moving from one point to another in parameter space = changing from one curve to another in hypothesis space

**Gradient Descent Visualization**:

1. **Start**: Random point in parameter space → random curve in hypothesis space (usually a poor fit)
    
2. **Gradient Step**: The optimizer computes gradient and moves to a new point in parameter space → this manifests as a new (slightly better) curve in hypothesis space
    
3. **Iteration**: Each gradient descent step corresponds to the curve "morphing" closer to the data
    
4. **Convergence**: Final point in parameter space (W*) → optimal curve that fits data well
    

**Example (M=1, straight line)**:

- Parameter space is 2D: axes are w₀ (intercept) and w₁ (slope)
- Point (w₀=50, w₁=200) → line Y = 50 + 200X in hypothesis space
- Gradient descent traces a path through parameter space → we see a sequence of lines progressively fitting the data better

### Mathematical Formulation

**Parameter Space:** $$\mathbf{W} \in \mathbb{R}^{M+1}$$

Each parameter vector is a point in $(M+1)$-dimensional space.

**Hypothesis Space:** $$\hat{Y} = g(X, \mathbf{W})$$

Each value of $\mathbf{W}$ defines a function curve in the X-Y plane.

**Gradient Descent Path in Parameter Space:** $${\mathbf{W}_0, \mathbf{W}_1, \mathbf{W}_2, \ldots, \mathbf{W}_T, \ldots, \mathbf{W}^*}$$

This is a trajectory of points.

**Corresponding Curve Sequence in Hypothesis Space:** $${g(X, \mathbf{W}_0), g(X, \mathbf{W}_1), \ldots, g(X, \mathbf{W}^*)}$$

A sequence of curves converging to the optimal fit.

### Worked Toy Example

**Task**: Fit a straight line to 3 points: (1, 3), (2, 5), (3, 7)

**Parameter Space** (2D):

- Axes: w₀ (vertical) and w₁ (horizontal)

**Gradient Descent**:

|Iteration|w₀|w₁|Point in Param Space|Hypothesis|Visual|
|---|---|---|---|---|---|
|0|10|-1|(10, -1)|Ŷ = 10 - X|Negative slope, poor fit|
|1|8|0.5|(8, 0.5)|Ŷ = 8 + 0.5X|Slight positive slope, still off|
|5|3|1.8|(3, 1.8)|Ŷ = 3 + 1.8X|Getting closer|
|10|1.2|1.95|(1.2, 1.95)|Ŷ = 1.2 + 1.95X|Very close to optimal|
|∞|1|2|(1, 2)|Ŷ = 1 + 2X|**Optimal** (perfect fit!)|

**Parameter Space View**: We see the point moving from (10, -1) → (8, 0.5) → ... → (1, 2)  
**Hypothesis Space View**: We see the line changing from steep negative to flat to steep positive, finally settling on Y = 1 + 2X

### Connections & Prerequisites

**Prerequisite Refresher**: A **coordinate system** assigns numbers (coordinates) to geometric positions. In 2D, the point (3, 5) means "move 3 units right, 5 units up." In parameter space with M=1, the point (w₀=1, w₁=2) represents the line equation Y = 1 + 2X. The **trajectory** is the path traced by a moving point—like connecting dots in the order they were visited.

---

## 10. Concept: Model Selection via Occam's Razor

### High-Level Intuition

When multiple models perform equally well on test data, choose the simplest one—it's cheaper to run and less likely to be "overfitted by accident." This is the principle of **Occam's Razor**: simpler explanations are preferable.

### Conceptual Deep Dive

After detecting overfitting and applying regularization, we often find a **plateau** in the test error curve—a range of model complexities (e.g., M=3 to M=8) that all achieve similar test performance.

**Selection Criteria**:

1. **Performance**: All models in the plateau have L_test ≈ minimum (within ~5% is typically acceptable)
    
2. **Simplicity**: Among equally-performing models, prefer the one with **fewer parameters**
    
3. **Computational Cost**: Simpler models have lower **inference costs** (cost of making predictions in production):
    
    - M=3 requires evaluating $w_0 + w_1 X + w_2 X^2 + w_3 X^3$ → 3 multiplications, 3 additions
    - M=8 requires evaluating 8 terms → 8 multiplications, 8 additions
    - For billions of predictions per day, this difference matters!

**Occam's Razor**: "Entities should not be multiplied without necessity." In ML: **Choose the model that is as simple as possible, but no simpler.**

**Practical Impact**:

- **Training costs**: Usually manageable (one-time expense)
- **Inference costs**: Ongoing expense, charged per prediction. For large-scale applications (e.g., serving Google search results), inference costs dominate.

**Caveat**: In the toy example from lecture, M=3 through M=8 all performed similarly—this is **artificial**. In real problems, test error usually has a clearer minimum, and we select that model. The principle still applies: if multiple models tie, prefer simplicity.

### Mathematical Formulation

**Model Complexity (for polynomials):** $$\text{Complexity}(M) = M + 1$$

Number of parameters in an $M$-th order polynomial.

**Inference Cost (operations per prediction):** $$\text{Cost} \propto M$$

Roughly linear in model complexity (each term requires multiplication and addition).

**Selection Rule:** $$\mathbf{W}^* = \arg\min_{\mathbf{W} \in \mathcal{M}} L_{test}(\mathbf{W}) \quad \text{subject to} \quad \text{Complexity}(\mathbf{W}) \text{ minimal}$$

Among models $\mathcal{M}$ with near-optimal test loss, choose the one with minimal complexity.

### Worked Toy Example

**Results from the plateau region:**

|Model|Params|L_test|Status|
|---|---|---|---|
|M=3|4|0.15|✓ Simplest, good performance|
|M=4|5|0.14|Slightly better, but more complex|
|M=5|6|0.145|Marginal improvement|
|M=6|7|0.143|Marginal improvement|
|M=7|8|0.141|Marginal improvement|
|M=8|9|0.14|Best test error, but most complex|

**Decision**: Choose M=3. Why?

- L_test(M=3) = 0.15 vs. L_test(M=8) = 0.14 → only 7% worse
- But M=3 has 4 parameters vs. M=8 has 9 parameters → 55% fewer parameters
- **Inference cost savings**: If serving 1 billion predictions/day, this translates to significant compute savings

### Connections & Prerequisites

**Prerequisite Refresher**: **Computational complexity** measures how much work an algorithm does. For a polynomial of degree M, evaluating it requires M multiplications (for $X, X^2, X^3, \ldots, X^M$) and M additions (summing the terms). If we process 1 billion data points, a model with M=8 does 8 billion operations, while M=3 does 3 billion—that's a 62% reduction! In production ML systems serving millions of users, these savings matter.

---

## 11. Concept: Model Compression via Pruning

### High-Level Intuition

After training a large model with regularization, we can "prune" the near-zero parameters to create a smaller, faster model without sacrificing performance—like trimming dead branches from a tree.

### Conceptual Deep Dive

**The Dilemma**: Regularization helps us train with high model complexity (M=9) while achieving the generalization of a simpler model (M=3). But we still have a 9-billion-parameter model in memory—wasteful!

**The Solution**: **Model compression** via **pruning**:

1. **Train** with regularization: Use the full complex model (M=9) with L2 penalty
    
2. **Observe** the trained parameters: Many coefficients (e.g., w₅ through w₉) are nearly zero
    
3. **Prune** near-zero parameters: Remove terms whose coefficients are below a threshold (e.g., |wⱼ| < 0.001)
    
4. **Deploy** the compressed model: The remaining model (effectively M≈3) is what goes to production
    

**Tools**: NVIDIA's TensorRT and similar **model compression frameworks** automate this process, identifying and removing parameters that contribute negligibly to predictions.

**Result**: We get the **best of both worlds**:

- The **regularization training process** automatically finds which parameters should be zero
- The **compressed deployed model** has low inference costs

**Key Insight**: Regularization is a **training technique** that softly discovers model structure. Compression is a **deployment technique** that removes discovered redundancy.

### Mathematical Formulation

**Trained Parameter Vector (with regularization):** $$\mathbf{W}_{reg}^* = [w_0, w_1, w_2, w_3, w_4, w_5, w_6, w_7, w_8, w_9]^T$$

Example values: $[2.1, 15.3, -4.8, 7.9, 0.05, 0.002, 0.0008, 0.0001, 0.00003, 0.000001]^T$

**Pruning Rule:** $$\text{If } |w_j| < \epsilon, \text{ set } w_j = 0$$

Where $\epsilon$ is a pruning threshold (e.g., 0.01).

**Compressed Parameter Vector:** $$\mathbf{W}_{compressed}^* = [2.1, 15.3, -4.8, 7.9, 0, 0, 0, 0, 0, 0]^T$$

Only the first 4 parameters are non-zero → effectively M=3.

**Compressed Hypothesis:** $$g_{compressed}(X, \mathbf{W}_{compressed}^*) = 2.1 + 15.3X - 4.8X^2 + 7.9X^3$$

A 4-parameter model, despite starting with 10 parameters.

### Worked Toy Example

**After Training M=9 Model with λ=0.01**:

**Full model coefficients:**

```
w0 = 5.2
w1 = 12.8
w2 = -3.1
w3 = 6.7
w4 = 0.08    ← small
w5 = 0.003   ← very small
w6 = 0.0009  ← very small
w7 = 0.0001  ← negligible
w8 = 0.00002 ← negligible
w9 = 0.000001 ← negligible
```

**Pruning (ε = 0.01)**:

- Keep: w0, w1, w2, w3 (all > 0.01)
- Remove: w4 through w9 (all < 0.01)

**Compressed model**: Ŷ = 5.2 + 12.8X - 3.1X² + 6.7X³

**Before compression**:

- Model size: 10 parameters × 4 bytes = 40 bytes
- Operations per prediction: 10 multiplies + 10 adds = 20 ops

**After compression**:

- Model size: 4 parameters × 4 bytes = 16 bytes (60% smaller!)
- Operations per prediction: 4 multiplies + 4 adds = 8 ops (60% faster!)

### Connections & Prerequisites

**Prerequisite Refresher**: **Sparsity** means most values in a vector are zero. A **sparse vector** like [5, 0, 0, 12, 0, 0, 0, 3] stores mostly zeros, which don't need to be computed (skip the multiplication by zero). **Compression** is the process of representing information more efficiently. Here, we recognize that "w₈ = 0.00001" can be approximated as "w₈ = 0" without losing accuracy.

---

### Key Takeaways & Formulas

**Core Concepts:**

1. **AI Agent Architecture**: Perception → Reasoning over Time → Planning → Control (with environment feedback loop)
2. **Supervised Learning**: Learn function $f: X \rightarrow Y$ from examples ${(X_i, Y_i)}_{i=1}^m$
3. **Overfitting**: Model too complex → memorizes training data → poor generalization (detected via test set)
4. **Regularization**: Add penalty for complexity → prevent overfitting → L2: $\lambda |\mathbf{W}|^2$
5. **Gradient Descent**: Iteratively minimize loss → $\mathbf{W}_{t+1} = \mathbf{W}_t - \eta \nabla L(\mathbf{W}_t)$

**Must-Remember Formulas:**

**Mean Squared Error:** $$L_{MSE} = \frac{1}{m} \sum_{i=1}^{m} (\hat{Y}_i - Y_i)^2$$

**Regularized Loss (Ridge):** $$L_{reg} = \frac{1}{m} \sum_{i=1}^{m} (\hat{Y}_i - Y_i)^2 + \lambda \sum_{j=0}^{M} w_j^2$$

**Gradient Descent Update:** $$\mathbf{W}_{t+1} = \mathbf{W}_t - \eta \nabla L(\mathbf{W}_t)$$

**Polynomial Hypothesis:** $$g_M(X, \mathbf{W}) = \sum_{j=0}^{M} w_j X^j$$

**Critical Exam Insight**: Always justify model selection using **both test error** (performance) and **model complexity** (simplicity). A complete answer explains the bias-variance tradeoff: simpler models may underfit (high bias), complex models may overfit (high variance), and regularization helps find the sweet spot.