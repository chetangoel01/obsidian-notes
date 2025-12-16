---
tags:
  - study-guide
---
### Executive Summary

This lecture bridges deterministic machine learning approaches (like basic gradient descent) with probabilistic frameworks that enable confidence quantification in predictions. We explore stochastic gradient descent as a solution to computational complexity and local minima problems, introduce foundational information theory concepts (entropy, cross-entropy), develop maximum likelihood estimation as our core parameter optimization framework, and transition regression and classification from absolute predictions to probabilistic models that quantify uncertainty.

---

## 1. Concept: The Learning Framework - Vapnik Diagram Components

### High-Level Intuition

**The Problem:** We need a systematic way to describe how machines learn from finite data samples to make predictions on unseen data.

**Analogy:** Think of machine learning like teaching someone to identify birds. The **sampler** is like showing them specific bird photos from a book (finite examples from infinite possibilities), the **hypothesis set** is like their mental library of "what makes a bird" (e.g., "has feathers," "has wings"), and the **learning algorithm** is the process of refining their mental model by comparing their guesses to correct answers.

### Conceptual Deep Dive

The **Vapnik diagram** provides the conceptual framework for supervised learning. It consists of three key components:

1. **The Sampler**: Draws finite samples from infinite possible (x, y) mappings. Given an input space X and output space Y, the sampler provides us with m examples: {(x₁, y₁), (x₂, y₂), ..., (xₘ, yₘ)}.
    
2. **The Hypothesis Set**: A family of functions g(x, w) parameterized by w that could potentially approximate the true underlying function f(x). The key insight: we start with a _complex_ hypothesis set (e.g., a large neural network) and let regularization push us toward simpler solutions.
    
3. **The Optimization Algorithm**: Searches through the hypothesis set to find the parameters w* that minimize some loss function, effectively selecting our final predictor ĝ(x, w*).
    

### Mathematical Formulation

The hypothesis is defined as: $$g(x, w) : \mathcal{X} \rightarrow \mathcal{Y}$$

where:

- $x \in \mathcal{X}$ is the input
- $w$ represents the parameters (weights) we need to learn
- $g(x, w)$ produces predictions in the output space $\mathcal{Y}$

For linear regression specifically: $$g(x, w) = w_0 + w_1 x$$

where $w = [w_0, w_1]^T$ are the parameters to be learned.

### Worked Toy Example

**Problem:** Fit a line to three data points: (1, 3), (2, 5), (3, 7)

**Step 1:** Choose hypothesis class: $g(x, w) = w_0 + w_1 x$

**Step 2:** Initial guess: $w_0 = 0, w_1 = 1$

- Prediction at x=1: ĝ(1) = 0 + 1(1) = 1 (true y=3, error = 2)
- Prediction at x=2: ĝ(2) = 0 + 1(2) = 2 (true y=5, error = 3)
- Prediction at x=3: ĝ(3) = 0 + 1(3) = 3 (true y=7, error = 4)

**Step 3:** After optimization (which we'll cover next), we'd find: $w_0 = 1, w_1 = 2$

- Now ĝ(1) = 1 + 2(1) = 3 ✓
- ĝ(2) = 1 + 2(2) = 5 ✓
- ĝ(3) = 1 + 2(3) = 7 ✓

### Connections & Prerequisites

This is the foundational framework upon which all subsequent concepts build. Understanding that learning = search through hypothesis space is critical for everything that follows.

---

## 2. Concept: Multi-Part Loss Functions and Regularization

### High-Level Intuition

**The Problem:** We want our model to fit training data well (low training error) but also generalize to new data (avoid overfitting).

**Analogy:** Imagine memorizing every question on a practice test versus understanding the underlying concepts. A student who memorizes might ace the practice test (low training error) but fail the real exam (poor generalization). Regularization is like a teacher saying "you must understand the fundamentals" - it penalizes overly complex memorization strategies.

### Conceptual Deep Dive

A **multi-part loss function** combines multiple objectives simultaneously. The general form:

$$\mathcal{L}_{total}(w) = \mathcal{L}_{data}(w) + \mathcal{L}_{penalty}(w)$$

The **data loss** $\mathcal{L}_{data}(w)$ measures how well the model fits the training data (e.g., mean squared error for regression).

The **penalty term** $\mathcal{L}_{penalty}(w)$ discourages overfitting by penalizing complexity. This penalty creates a "landscape" in parameter space where:

- **High-penalty regions** correspond to overfit models (large weights, high-order terms active)
- **Low-penalty regions** correspond to simpler, more generalizable models
- The optimal w* lives in a region balancing both terms

The key insight from lecture: In practice, regularization can completely **switch off** high-order terms, not just reduce them, leading to true sparsity in solutions.

### Mathematical Formulation

For linear regression with L2 regularization: $$\mathcal{L}(w) = \frac{1}{m}\sum_{i=1}^{m}(y_i - g(x_i, w))^2 + \lambda ||w||^2$$

where:

- First term: Mean Squared Error (data fit)
- $\lambda$: Regularization strength (hyperparameter)
- $||w||^2 = w_0^2 + w_1^2 + ... + w_n^2$: L2 penalty on weights

### Worked Toy Example

**Setup:** Three data points with a quadratic model

- Data: (1, 2), (2, 3), (3, 5)
- Model: $g(x, w) = w_0 + w_1 x + w_2 x^2$

**Without regularization (λ = 0):**

- Might find: $w = [1, 0.5, 0.3]$
- Data loss: very low (fits training perfectly)
- Problem: $w_2$ term might be fitting noise

**With regularization (λ = 0.1):**

- Total loss = Data loss + 0.1(w₀² + w₁² + w₂²)
- Might find: $w = [1.2, 0.6, 0.05]$
- $w_2$ is heavily suppressed
- Better generalization: simpler model preferred

### Connections & Prerequisites

**Prerequisite Refresher on Overfitting:** Recall that overfitting occurs when a model learns the training data _too well_, capturing noise rather than the underlying pattern. A model with many parameters can perfectly fit training data by using high-order terms to "wiggle" through every data point, but this doesn't generalize to new data.

---

## 3. Concept: Gradient Descent - Basic Optimization

### High-Level Intuition

**The Problem:** We have a loss function L(w) and need to find the parameters w* that minimize it.

**Analogy:** Imagine you're blindfolded on a hilly terrain and need to reach the lowest valley. Gradient descent is like feeling the slope beneath your feet and taking small steps downhill. The **gradient** tells you which direction is steepest, and the **learning rate** determines how big each step should be.

### Conceptual Deep Dive

**Gradient Descent** is an iterative optimization algorithm that updates parameters in the direction opposite to the gradient (the direction of steepest increase in the loss).

The algorithm:

1. Initialize parameters randomly: $w^{(0)}$
2. Compute the gradient: $\nabla \mathcal{L}(w^{(k)})$
3. Update: $w^{(k+1)} = w^{(k)} - \eta \nabla \mathcal{L}(w^{(k)})$
4. Repeat until convergence

**Key hyperparameter:** The **learning rate** η controls step size:

- Too small → slow convergence, may not reach optimal in time
- Too large → oscillations, may diverge
- Just right → smooth convergence to w*

The gradient must be computed over **all m training examples**, which becomes computationally expensive for large datasets.

### Mathematical Formulation

For mean squared error with hypothesis $g(x, w) = w_0 + w_1 x$:

$$\nabla \mathcal{L}(w) = \begin{bmatrix} \frac{\partial \mathcal{L}}{\partial w_0} \ \frac{\partial \mathcal{L}}{\partial w_1} \end{bmatrix}$$

The partial derivatives are: $$\frac{\partial \mathcal{L}}{\partial w_0} = \frac{1}{m}\sum_{i=1}^{m} 2(w_0 + w_1 x_i - y_i)$$

$$\frac{\partial \mathcal{L}}{\partial w_1} = \frac{1}{m}\sum_{i=1}^{m} 2(w_0 + w_1 x_i - y_i) \cdot x_i$$

Update rule: $$w_0^{(k+1)} = w_0^{(k)} - \eta \cdot \frac{\partial \mathcal{L}}{\partial w_0}$$ $$w_1^{(k+1)} = w_1^{(k)} - \eta \cdot \frac{\partial \mathcal{L}}{\partial w_1}$$

### Worked Toy Example

**Data:** (1, 3), (2, 5)  
**Model:** $g(x, w) = w_0 + w_1 x$  
**Learning rate:** η = 0.1  
**Initial:** $w_0 = 0, w_1 = 0$

**Iteration 1:**

- Predictions: ĝ(1) = 0, ĝ(2) = 0
- Errors: e₁ = 0 - 3 = -3, e₂ = 0 - 5 = -5
- Gradients:
    - $\frac{\partial \mathcal{L}}{\partial w_0} = \frac{1}{2}[2(-3) + 2(-5)] = -8$
    - $\frac{\partial \mathcal{L}}{\partial w_1} = \frac{1}{2}[2(-3)(1) + 2(-5)(2)] = -13$
- Updates:
    - $w_0^{(1)} = 0 - 0.1(-8) = 0.8$
    - $w_1^{(1)} = 0 - 0.1(-13) = 1.3$

**Iteration 2:**

- New predictions: ĝ(1) = 0.8 + 1.3(1) = 2.1, ĝ(2) = 0.8 + 1.3(2) = 3.4
- Continue iterations until convergence...

### Connections & Prerequisites

This builds directly on the loss function concept - we're now finding the w that minimizes the loss we defined earlier.

---

## 4. Concept: Stochastic Gradient Descent (SGD)

### High-Level Intuition

**The Problem:** Standard gradient descent requires computing gradients over ALL m examples each iteration, which is slow for large datasets. Additionally, it can get stuck in local minima of complex loss surfaces.

**Analogy:** Instead of surveying every inch of the valley before taking a step (expensive!), stochastic gradient descent is like taking quick steps based on random samples of the terrain nearby. You might not always step in the _perfect_ direction, but you explore more of the landscape and move faster on average. The randomness also helps you escape from shallow local valleys.

### Conceptual Deep Dive

**Stochastic Gradient Descent** addresses two critical problems:

1. **Computational Complexity:** Instead of summing over all m examples, SGD uses a **mini-batch** of size B << m to estimate the gradient.
    
2. **Local Minima Escape:** The noise introduced by sampling makes the gradient point in slightly different directions each iteration. This randomness allows the algorithm to:
    
    - Escape shallow local minima
    - Explore the parameter space more thoroughly
    - Potentially find better global solutions

**Key concepts:**

- **Mini-batch:** A randomly sampled subset of training data
- **Epoch:** One complete pass through the entire dataset
- **Sampling with replacement:** The same example can appear in multiple mini-batches

The gradient becomes a **noisy estimator** - it points approximately toward the minimum but with variance. This noise is a feature, not a bug!

### Mathematical Formulation

**Standard GD gradient (all m examples):** $$\nabla \mathcal{L}(w) = \frac{1}{m}\sum_{i=1}^{m} \nabla \mathcal{L}_i(w)$$

**SGD gradient (mini-batch of size B):** $$\nabla \mathcal{L}_{batch}(w) = \frac{1}{B}\sum_{i \in \text{mini-batch}} \nabla \mathcal{L}_i(w)$$

where B might be 8, 16, 32, etc. (often much smaller than m)

**Learning rate scheduling** (optional but recommended): $$\eta(t) = \frac{\eta_0}{1 + decay \cdot t}$$

Start with larger steps, gradually reduce step size as you approach minimum.

### Worked Toy Example

**Dataset:** m = 200 points  
**Mini-batch size:** B = 8  
**Learning rate:** η = 0.1

**Standard GD - Iteration 1:**

- Compute gradient over all 200 points
- Single update to w
- Cost: 200 gradient calculations

**SGD - Iteration 1:**

- Randomly sample 8 points: {x₃, x₁₂, x₇, x₉₅, x₁₈, x₁₄₄, x₆₇, x₁₀₁}
- Compute gradient over just these 8 points
- Update w
- Cost: 8 gradient calculations (25× faster!)

**Per epoch:**

- Standard GD: 1 update (evaluates all 200 points once)
- SGD: ~25 updates (evaluates all 200 points in batches of 8)

Result: SGD explores parameter space much more rapidly and with beneficial noise.

### Connections & Prerequisites

**Prerequisite Refresher on Gradient Descent:** Recall that GD iteratively updates parameters by moving in the direction opposite to the gradient: $w^{(k+1)} = w^{(k)} - \eta \nabla \mathcal{L}(w^{(k)})$. SGD modifies this by using an approximate (noisy) gradient computed over a small subset of data, trading perfect gradient information for computational efficiency and exploration benefits.

---

## 5. Concept: Probability Distributions in Machine Learning

### High-Level Intuition

**The Problem:** Until now, we've treated data as deterministic. But real-world data comes from complex, uncertain processes. We need a mathematical framework to model this uncertainty.

**Analogy:** When a sensor measures temperature, it doesn't give you a single "true" value - there's measurement noise, environmental fluctuations, etc. A probability distribution describes the range of likely values and their relative frequencies. In ML, we model both our inputs (x) and outputs (y) as random variables drawn from distributions.

### Conceptual Deep Dive

The lecture transitions from deterministic to **probabilistic modeling**:

1. **Data Generation Model:**
    
    - Input x is sampled from some distribution: $x \sim p_{data}(x)$
    - Given x, output y is sampled from: $y \sim p_{data}(y|x)$
    - The **sampler** provides the joint distribution: $p_{data}(x, y)$
2. **Product Rule (Factorization):** $$p_{data}(x, y) = p_{data}(y|x) \cdot p_{data}(x)$$
    
    This breaks the joint into a **conditional** (y given x) and a **marginal** (x alone).
    
3. **Model Distribution:**
    
    - We propose a model: $p_{model}(y|x, w)$
    - Our hypothesis is now probabilistic
    - Goal: Make $p_{model}$ as close as possible to $p_{data}$

**Key terminology:**

- **Random variable:** A variable whose value is determined by a random process
- **Conditional probability:** $p(y|x)$ means "probability of y given we know x"
- **Marginal distribution:** $p(x)$ obtained by "summing out" other variables

### Mathematical Formulation

**Product Rule:** $$p(x, y) = p(y|x) \cdot p(x) = p(x|y) \cdot p(y)$$

**Sum Rule (Marginalization):** $$p(x) = \sum_y p(x, y) \quad \text{(discrete)}$$ $$p(x) = \int p(x, y) , dy \quad \text{(continuous)}$$

**Independence:** If x₁ and x₂ are independent: $$p(x_1, x_2) = p(x_1) \cdot p(x_2)$$

**Example distributions:**

- **Normal (Gaussian):** $\mathcal{N}(\mu, \sigma^2) = \frac{1}{\sqrt{2\pi\sigma^2}}\exp\left(-\frac{(x-\mu)^2}{2\sigma^2}\right)$
- **Bernoulli:** $p(x=1) = p$, $p(x=0) = 1-p$ (for binary outcomes)

### Worked Toy Example

**Bivariate Gaussian - Temperature Sensors**

**Setup:**

- Sensor 1 (indoor): $x_1 \sim \mathcal{N}(20°C, 1°C^2)$
- Sensor 2 (outdoor): $x_2 \sim \mathcal{N}(15°C, 4°C^2)$
- These are **correlated**: when outdoor temp drops, indoor tends to drop too

**Joint distribution:** $p(x_1, x_2)$ is a bivariate Gaussian

- Mean vector: $\mu = [20, 15]^T$
- Covariance matrix captures correlation: $$\Sigma = \begin{bmatrix} 1 & 0.8 \ 0.8 & 4 \end{bmatrix}$$

**Interpretation:** The off-diagonal 0.8 indicates positive correlation.

**Marginal distributions:**

- $p(x_1) = \mathcal{N}(20, 1)$ (just integrate out x₂)
- $p(x_2) = \mathcal{N}(15, 4)$ (just integrate out x₁)

### Connections & Prerequisites

This is foundational probability theory needed for all subsequent probabilistic ML concepts. The key insight: we're now modeling our hypothesis not as a function but as a probability distribution.

---

## 6. Concept: Information Theory - Entropy and Value of Information

### High-Level Intuition

**The Problem:** How do we quantify the "information content" or "surprise" of an event? How much should we pay to acquire information?

**Analogy:** Imagine you're a hedge fund manager receiving two sealed envelopes. One says "the Holland Tunnel will be empty Monday at 8am" (surprising!), the other says "the Holland Tunnel will be packed Monday at 8am" (not surprising). Which has more value? The surprising information is more valuable because it deviates from what you expect - that's the essence of information theory.

### Conceptual Deep Dive

**Value of Information** for an event x: $$I(x) = -\log p(x) = \log \frac{1}{p(x)}$$

Key insights:

- **Rare events** (low p(x)) have **high information value**
- **Common events** (high p(x)) have **low information value**
- Base of logarithm determines units:
    - log₂ → bits
    - ln (natural log) → nats

**Entropy** aggregates information value across all possible events: $$H(p) = -\sum_i p(x_i) \log p(x_i) = \mathbb{E}_{x \sim p}[-\log p(x)]$$

Entropy measures:

- **Average surprise** in the distribution
- **Uncertainty** in the random variable
- **Minimum bits** needed to encode outcomes on average

**Special case - Binary (Bernoulli) Entropy:** For a coin with P(heads) = p: $$H(p) = -p\log_2(p) - (1-p)\log_2(1-p)$$

Maximum entropy: p = 0.5 (maximum uncertainty)  
Minimum entropy: p = 0 or p = 1 (no uncertainty)

### Mathematical Formulation

**Information value:** $$I(x) = -\log_2 p(x) \quad \text{[bits]}$$

**Entropy (discrete):** $$H(X) = -\sum_{i=1}^{n} p(x_i) \log_2 p(x_i)$$

where:

- $X$ is the random variable
- $p(x_i)$ is the probability of outcome $x_i$
- Sum is over all possible outcomes

**Properties:**

1. $H(X) \geq 0$ (always non-negative)
2. $H(X) = 0$ iff one outcome has p = 1
3. Maximum when uniform distribution

### Worked Toy Example

**Example: English letters + space (27 outcomes)**

From the Linux FAQ text analysis:

- Space symbol: p = 0.18 → I = -log₂(0.18) = 2.47 bits
- Letter 'e': p = 0.10 → I = -log₂(0.10) = 3.32 bits
- Letter 'j': p = 0.001 → I = -log₂(0.001) = 9.97 bits

**Entropy calculation:** $$H = \sum_{i=1}^{27} p_i \log_2 p_i$$

If we compute this for all 27 symbols, we get approximately **4.1 bits** per character.

**Interpretation:** On average, you need 4.1 bits to encode each character in English text (with these frequencies). This is why compression algorithms can reduce text size - they exploit the non-uniform distribution.

**Binary entropy example:** For a fair coin (p = 0.5): $$H = -0.5\log_2(0.5) - 0.5\log_2(0.5) = 1 \text{ bit}$$

For a biased coin (p = 0.9): $$H = -0.9\log_2(0.9) - 0.1\log_2(0.1) = 0.47 \text{ bits}$$

### Connections & Prerequisites

This information theory foundation is critical for understanding loss functions in ML. The next concept (cross-entropy) extends this to measure distance between distributions.

---

## 7. Concept: Cross-Entropy - Measuring Distribution Distance

### High-Level Intuition

**The Problem:** We need a way to measure how "different" two probability distributions are. In ML, we want to know: how far is our model distribution $p_{model}$ from the true data distribution $p_{data}$?

**Analogy:** Imagine you have two translators. The first speaks your language (p_data), the second speaks a slightly different dialect (p_model). Cross-entropy measures the "extra communication cost" - the additional bits needed when you encode messages optimized for the first translator but have to send them through the second.

### Conceptual Deep Dive

**Cross-Entropy** between distributions p and q: $$H(p, q) = -\mathbb{E}_{x \sim p}[\log q(x)] = -\sum_x p(x) \log q(x)$$

**Interpretation:**

- Measures the average number of bits needed to encode samples from p using an encoding optimized for q
- Always ≥ entropy H(p)
- Equals H(p) only when p = q (perfect match)

**KL Divergence** (Kullback-Leibler): $$D_{KL}(p || q) = \mathbb{E}_{x \sim p}\left[\log \frac{p(x)}{q(x)}\right] = H(p, q) - H(p)$$

Properties:

- $D_{KL}(p || q) \geq 0$
- $D_{KL}(p || q) = 0$ iff p = q
- **Not symmetric:** $D_{KL}(p || q) \neq D_{KL}(q || p)$

**Connection to ML Loss Functions:** When we minimize cross-entropy between $p_{data}$ and $p_{model}$, we're minimizing KL divergence because H(p_data) is constant: 
$$\min_w H(p_{data}, p_{model}) \equiv \min_w D_{KL}(p_{data} || p_{model})$$

### Mathematical Formulation

**Cross-entropy loss (general):** $$\mathcal{L}(w) = -\mathbb{E}_{(x,y) \sim p_{data}}[\log p_{model}(y|x, w)]$$

**Discrete form (practical):** $$\mathcal{L}(w) = -\frac{1}{m}\sum_{i=1}^{m} \log p_{model}(y_i | x_i, w)$$

where:

- m is the number of training examples
- $(x_i, y_i)$ are training pairs
- $p_{model}(y|x, w)$ is our probabilistic hypothesis

**Binary classification special case:** $$\mathcal{L}(w) = -\frac{1}{m}\sum_{i=1}^{m} [y_i \log p_i + (1-y_i)\log(1-p_i)]$$

where $p_i = p_{model}(y=1|x_i, w)$

### Worked Toy Example

**Setup:** Binary classification with 4 examples

|Example|True y|Model probability p̂ = p(y=1)|
|---|---|---|
|1|1|0.9|
|2|1|0.7|
|3|0|0.2|
|4|0|0.1|

**Cross-entropy calculation:**

For each example:

- Ex 1: $-[1 \cdot \log(0.9) + 0 \cdot \log(0.1)] = -\log(0.9) = 0.046$
- Ex 2: $-[1 \cdot \log(0.7) + 0 \cdot \log(0.3)] = -\log(0.7) = 0.155$
- Ex 3: $-[0 \cdot \log(0.2) + 1 \cdot \log(0.8)] = -\log(0.8) = 0.097$
- Ex 4: $-[0 \cdot \log(0.1) + 1 \cdot \log(0.9)] = -\log(0.9) = 0.046$

**Average loss:** $$\mathcal{L} = \frac{1}{4}(0.046 + 0.155 + 0.097 + 0.046) = 0.086$$

**Interpretation:** Lower cross-entropy = model probabilities align well with true labels.

### Connections & Prerequisites

**Prerequisite Refresher on Entropy:** Recall that entropy H(p) measures the average information content of distribution p. Cross-entropy extends this to measure the cost of using the "wrong" distribution (q) to encode data from the "right" distribution (p). The difference between them (KL divergence) quantifies how suboptimal our encoding is.

---

## 8. Concept: Maximum Likelihood Estimation (MLE)

### High-Level Intuition

**The Problem:** Given data samples and a parametric model (e.g., Gaussian with unknown μ and σ²), how do we find the best parameters?

**Analogy:** Imagine you're trying to guess the settings on a radio by listening to fragments of music. MLE says: "Find the dial settings (parameters) that would make the music you're hearing (the data) as likely as possible." You're asking, "Which parameter values make this observed data least surprising?"

### Conceptual Deep Dive

**Maximum Likelihood Estimation** is the workhorse of modern ML. The core idea:

1. **Likelihood:** Given parameters w, how probable is it that we'd observe our data? $$L(w) = p_{model}(\text{data} | w)$$
    
2. **Goal:** Find parameters that maximize this likelihood: $$w^* = \arg\max_w L(w)$$
    

**Key assumptions:**

- **IID (Independent and Identically Distributed):** Each data point is drawn independently from the same distribution
- This allows factorization: $L(w) = \prod_{i=1}^{m} p_{model}(x_i | w)$

**Log-likelihood trick:** Products of small probabilities cause numerical underflow. Taking logarithms:

- Converts products to sums (computationally stable)
- Preserves the location of the maximum (log is monotonic)
- Easier to differentiate

$$\log L(w) = \sum_{i=1}^{m} \log p_{model}(x_i | w)$$

**Optimization:** Maximizing log-likelihood ≡ Minimizing negative log-likelihood: $$w^* = \arg\min_w \left[-\sum_{i=1}^{m} \log p_{model}(x_i | w)\right]$$

This is exactly **cross-entropy loss**!

### Mathematical Formulation

**Marginal MLE (density estimation):** $$w^* = \arg\max_w \prod_{i=1}^{m} p_{model}(x_i | w)$$

**Log-likelihood:** $$\ell(w) = \sum_{i=1}^{m} \log p_{model}(x_i | w)$$

**Example: Gaussian MLE**

If $p_{model}(x|w) = \mathcal{N}(\mu, \sigma^2)$: $$\ell(\mu, \sigma^2) = \sum_{i=1}^{m} \log\left[\frac{1}{\sqrt{2\pi\sigma^2}}\exp\left(-\frac{(x_i-\mu)^2}{2\sigma^2}\right)\right]$$

Simplifying: $$\ell(\mu, \sigma^2) = -\frac{m}{2}\log(2\pi) - \frac{m}{2}\log(\sigma^2) - \frac{1}{2\sigma^2}\sum_{i=1}^{m}(x_i - \mu)^2$$

Taking derivatives and setting to zero yields: $$\mu^* = \frac{1}{m}\sum_{i=1}^{m} x_i \quad \text{(sample mean)}$$ $$\sigma^{2*} = \frac{1}{m}\sum_{i=1}^{m}(x_i - \mu^*)^2 \quad \text{(sample variance)}$$

### Worked Toy Example

**Data:** Five samples on the real line: {3, 4, 6, 7, 5}

**Model:** $p_{model}(x | \mu, \sigma^2) = \mathcal{N}(\mu, \sigma^2)$

**Step 1: Calculate sufficient statistics**

- Sample mean: $\mu = \frac{3+4+6+7+5}{5} = 5$
- Sample variance: $$\sigma^2 = \frac{(3-5)^2 + (4-5)^2 + (6-5)^2 + (7-5)^2 + (5-5)^2}{5} = \frac{4+1+1+4+0}{5} = 2$$

**Step 2: Compare two hypotheses**

**Hypothesis 1:** $\mu_1 = 5, \sigma_1^2 = 2$ (our MLE estimate)

Log-likelihood: $$\ell_1 = \sum_{i=1}^{5} \log \mathcal{N}(x_i | 5, 2)$$

For each point:

- $\log \mathcal{N}(3|5,2) = \log\frac{1}{\sqrt{4\pi}}e^{-\frac{4}{4}} \approx -1.84$
- $\log \mathcal{N}(4|5,2) \approx -1.61$
- Similar for others...
- Total: $\ell_1 \approx -8.3$

**Hypothesis 2:** $\mu_2 = 8, \sigma_2^2 = 2$ (wrong mean)

This would give $\ell_2 \approx -15.2$ (much worse!)

**Conclusion:** Hypothesis 1 has higher likelihood, so MLE correctly identifies μ = 5.

### Connections & Prerequisites

**Prerequisite Refresher on Cross-Entropy:** MLE directly connects to cross-entropy. When we maximize $\sum \log p_{model}(x_i|w)$, we're minimizing the cross-entropy between the empirical data distribution $\hat{p}_{data}$ and our model $p_{model}$. This is why cross-entropy is our fundamental loss function.

---

## 9. Concept: Probabilistic Linear Regression

### High-Level Intuition

**The Problem:** Previously, regression gave us a single number (point estimate) for each prediction. But how confident should we be? A prediction of "house price = $350k" without uncertainty information is incomplete.

**Analogy:** Instead of saying "the temperature tomorrow will be exactly 75°F," a probabilistic model says "the temperature will be around 75°F, with a typical variation of ±3°F." The ±3°F represents our confidence (variance), while 75°F is our best guess (mean).

### Conceptual Deep Dive

**Transition to Probabilistic Regression:**

Old deterministic view: $\hat{y} = g(x, w) = w_0 + w_1 x$

New probabilistic view: $$p_{model}(y | x, w) = \mathcal{N}(g(x,w), \sigma^2)$$

**Key assumptions:**

1. The true relationship: $y = f(x) + \epsilon$
2. Noise term: $\epsilon \sim \mathcal{N}(0, \sigma^2)$
3. Therefore: $y | x \sim \mathcal{N}(f(x), \sigma^2)$

**Our model:**

- **Mean of distribution:** $\mu(x) = g(x, w) = w_0 + w_1 x$ (the regression line)
- **Variance:** $\sigma^2$ (constant uncertainty across all x)

**Geometric interpretation:**

- For each fixed x value, imagine a vertical Gaussian distribution
- The mean of each Gaussian lies on the regression line
- The variance σ² represents the spread (confidence interval width)
- As we vary x, the means trace out the red regression line

**Training via MLE:** $$w^* = \arg\max_w \prod_{i=1}^{m} p_{model}(y_i | x_i, w)$$ $$= \arg\min_w \left[-\sum_{i=1}^{m} \log \mathcal{N}(y_i | g(x_i, w), \sigma^2)\right]$$

**Remarkable result:** Maximizing likelihood for Gaussian model is equivalent to minimizing mean squared error!

### Mathematical Formulation

**Probabilistic model:** $$p_{model}(y | x, w, \sigma^2) = \frac{1}{\sqrt{2\pi\sigma^2}} \exp\left(-\frac{(y - g(x,w))^2}{2\sigma^2}\right)$$

where $g(x, w) = w_0 + w_1 x$ for linear regression.

**Negative log-likelihood loss:** $$\mathcal{L}(w) = -\sum_{i=1}^{m} \log p_{model}(y_i | x_i, w)$$ $$= \frac{m}{2}\log(2\pi\sigma^2) + \frac{1}{2\sigma^2}\sum_{i=1}^{m}(y_i - g(x_i, w))^2$$

**Key insight:** Minimizing this is equivalent to minimizing MSE: $$\min_w \sum_{i=1}^{m}(y_i - g(x_i, w))^2$$

The σ² terms are constant w.r.t. w, so they don't affect the argmin.

**Prediction with uncertainty:** Given a new $x_{new}$: $$\hat{y}_{new} = g(x_{new}, w^*) \pm 2\sigma \quad \text{(95% confidence)}$$

### Worked Toy Example

**Data:** Three points (1, 3), (2, 5), (3, 6)

**Step 1: Assume Gaussian noise model** $$y_i = w_0 + w_1 x_i + \epsilon_i, \quad \epsilon_i \sim \mathcal{N}(0, \sigma^2)$$

**Step 2: MLE optimization** Using gradient descent on negative log-likelihood finds:

- $w_0^* = 1.67$
- $w_1^* = 1.5$
- $\sigma^{2*} = 0.22$ (estimated from residuals)

**Step 3: Make predictions**

For $x = 2.5$:

- Mean prediction: $\hat{y} = 1.67 + 1.5(2.5) = 5.42$
- Standard deviation: $\sigma = \sqrt{0.22} = 0.47$
- 95% confidence interval: $5.42 \pm 2(0.47) = [4.48, 6.36]$

**Visualization:**

- At x = 1: Gaussian centered at y = 3.17, spread ±0.47
- At x = 2: Gaussian centered at y = 4.67, spread ±0.47
- At x = 3: Gaussian centered at y = 6.17, spread ±0.47
- The means (3.17, 4.67, 6.17) form the regression line

### Connections & Prerequisites

**Prerequisite Refresher on MLE:** Recall that MLE finds parameters maximizing $p_{model}(\text{data}|w)$. For regression with Gaussian noise, maximizing likelihood is mathematically equivalent to minimizing sum of squared errors - this connects our earlier deterministic regression to the new probabilistic framework.

---

## 10. Concept: Binary Classification and Detection Theory

### High-Level Intuition

**The Problem:** We need to make binary decisions (attack/no attack, spam/not spam, cancer/healthy) based on noisy measurements. Where should we set the decision threshold to balance different types of errors?

**Analogy:** WWII radar operators had to decide: "Is this signal an enemy bomber or just birds?" Set the threshold too sensitive → false alarms, people lose trust. Too insensitive → miss real attacks, people die. Binary classification finds the optimal balance.

### Conceptual Deep Dive

**Setup:**

- **Feature:** x (e.g., radar signal strength)
- **Label:** y ∈ {0, 1} (negative class, positive class)
- **Decision rule:** If x > w (threshold), predict ŷ = 1, else ŷ = 0

**The Confusion Matrix:**

||ŷ = 1 (Predict Positive)|ŷ = 0 (Predict Negative)|
|---|---|---|
|y = 1 (True)|True Positive (TP)|False Negative (FN)|
|y = 0 (True)|False Positive (FP)|True Negative (TN)|

**Key rates:**

- **False Positive Rate (FPR):** P(ŷ=1 | y=0) - "false alarm rate"
- **False Negative Rate (FNR):** P(ŷ=0 | y=1) - "miss rate"
- **True Positive Rate (TPR):** P(ŷ=1 | y=1) - "detection rate" or "sensitivity"

**The fundamental problem:** We cannot make both FPR and FNR zero simultaneously when class-conditional distributions overlap: $$p(x | y=0) \text{ and } p(x | y=1) \text{ overlap}$$

**Optimal threshold (Bayes decision rule):** Set threshold where total error is minimized: $$P(\text{error}) = P(\text{FP}) + P(\text{FN})$$ $$= \int_{x \in R_1} p(x|y=0)p(y=0)dx + \int_{x \in R_0} p(x|y=1)p(y=1)dx$$

where $R_1 = {x : x > w}$ and $R_0 = {x : x \leq w}$

### Mathematical Formulation

**Class-conditional densities:** $$p(x | y=0) \quad \text{(distribution of x when negative class)}$$ $$p(x | y=1) \quad \text{(distribution of x when positive class)}$$

**Decision regions:**

- $R_1 = {x : x > w}$ → predict positive
- $R_0 = {x : x \leq w}$ → predict negative

**Error probability:** $$P_{error}(w) = \int_{-\infty}^{w} p(x|y=1)p(y=1)dx + \int_{w}^{\infty} p(x|y=0)p(y=0)dx$$

Components:

- First term (magenta area): FN rate
- Second term (green area): FP rate

**Optimal threshold:** $w^*$ minimizes $P_{error}(w)$

**Graphically:** The optimal threshold is approximately where the two class-conditional distributions intersect (assuming equal priors).

### Worked Toy Example

**Scenario:** Radar detection with signal strength x

**Distributions:**

- No attack: $p(x|y=0) = \mathcal{N}(3, 1)$ (mean=3, variance=1)
- Attack: $p(x|y=1) = \mathcal{N}(7, 1.5)$ (mean=7, variance=1.5)

**Prior probabilities:**

- P(y=0) = 0.9 (no attack is common)
- P(y=1) = 0.1 (attacks are rare)

**Trial threshold: w = 5**

**Calculate error rates:**

FNR (miss an attack): $$P(x < 5 | y=1) = P(X < 5 \text{ when } X \sim \mathcal{N}(7, 1.5))$$ Using standard normal tables: ≈ 0.05

FPR (false alarm): $$P(x > 5 | y=0) = P(X > 5 \text{ when } X \sim \mathcal{N}(3, 1))$$ Using standard normal tables: ≈ 0.02

**Total error:** $$P_{error} = 0.1 \times 0.05 + 0.9 \times 0.02 = 0.005 + 0.018 = 0.023$$

**Try w = 4.5:** (moving threshold left)

- FNR decreases (catches more attacks)
- FPR increases (more false alarms)
- Need to compute total error to compare

**Optimal threshold:** Found by calculus or grid search, approximately w* ≈ 4.8 for this example.

### Connections & Prerequisites

**Prerequisite Refresher on Probability Distributions:** Recall that class-conditional distributions $p(x|y=0)$ and $p(x|y=1)$ describe how the feature x is distributed within each class. The overlap between these distributions determines the fundamental limit on classification accuracy - perfect separation is only possible when distributions don't overlap.

---

## Key Takeaways & Formulas

### Critical Concepts to Master

1. **Stochastic Gradient Descent (SGD):**
    
    - Uses mini-batches (B << m) instead of full dataset
    - Trades perfect gradients for computational efficiency + exploration
    - Formula: $w^{(k+1)} = w^{(k)} - \eta \cdot \frac{1}{B}\sum_{i \in batch} \nabla \mathcal{L}_i(w^{(k)})$
2. **Information Theory Foundations:**
    
    - **Entropy:** $H(p) = -\sum_i p(x_i) \log p(x_i)$ measures uncertainty
    - **Cross-entropy:** $H(p,q) = -\sum_i p(x_i) \log q(x_i)$ measures distribution distance
    - Cross-entropy = KL divergence + entropy (constant)
3. **Maximum Likelihood Estimation (MLE):**
    
    - Core principle: Find parameters that make observed data most probable
    - $w^* = \arg\max_w \prod_{i=1}^m p_{model}(x_i|w) = \arg\min_w [-\sum_{i=1}^m \log p_{model}(x_i|w)]$
    - Minimizing negative log-likelihood = minimizing cross-entropy
4. **Probabilistic Regression:**
    
    - Model: $p(y|x,w) = \mathcal{N}(g(x,w), \sigma^2)$ where $g(x,w) = w_0 + w_1x$
    - MLE for Gaussian regression ≡ MSE minimization
    - Predictions include confidence: $\hat{y} \pm 2\sigma$
5. **Binary Classification:**
    
    - Decision boundary creates confusion matrix: TP, TN, FP, FN
    - Overlapping class distributions → irreducible error
    - Optimal threshold minimizes: $P(error) = P(FP) + P(FN)$

### Must-Memorize Formulas

**Gradient descent update:** $$w^{(k+1)} = w^{(k)} - \eta \nabla \mathcal{L}(w^{(k)})$$

**Cross-entropy loss:** $$\mathcal{L}(w) = -\frac{1}{m}\sum_{i=1}^m \log p_{model}(y_i|x_i, w)$$

**Gaussian density:** $$\mathcal{N}(x|\mu,\sigma^2) = \frac{1}{\sqrt{2\pi\sigma^2}}\exp\left(-\frac{(x-\mu)^2}{2\sigma^2}\right)$$

**Entropy (binary):** $$H(p) = -p\log p - (1-p)\log(1-p)$$