---
tags:
---
### Executive Summary

This lecture introduces neural network architectures for classification, starting from binary classification metrics (ROC curves, confusion matrices) and progressing through logistic regression to multi-layer neural networks. The core focus is understanding how neural networks automatically learn features through gradient-based optimization, culminating in the backpropagation algorithm. Key topics include binary cross-entropy loss, sigmoid and ReLU activation functions, dense layers, softmax for multi-class classification, and the computational graph framework for efficient gradient computation.

---

## 1. Concept: Binary Classification Metrics and Trade-offs

### High-Level Intuition

When building a binary classifier, we need to decide where to set the decision threshold. This choice involves a fundamental trade-off: reducing one type of error (false negatives) often increases another type (false positives). Think of it like airport security—setting stricter screening reduces the chance of missing a threat (false negative) but increases the chance of falsely flagging innocent passengers (false positive).

### Conceptual Deep Dive

In binary classification, we have four possible outcomes forming a **confusion matrix**:

- **True Positive (TP)**: Correctly predicted positive class
- **True Negative (TN)**: Correctly predicted negative class
- **False Positive (FP)**: Incorrectly predicted positive (Type I error)
- **False Negative (FN)**: Incorrectly predicted negative (Type II error)

The decision boundary is controlled by a **threshold** θ (not to be confused with model parameters). The axis is divided into two regions:

- **R₀**: Values below θ (predicted as negative)
- **R₁**: Values above θ (predicted as positive)

The **probability of error** consists of two components:

- **False Negative Rate**: P(predict negative | actually positive) = ∫_{R₀} P(x|y=1)dx
- **False Positive Rate**: P(predict positive | actually negative) = ∫_{R₁} P(x|y=0)dx

### Mathematical Formulation

**Key Classification Metrics:**

**Recall (True Positive Rate, Sensitivity, Probability of Detection):** $$\text{Recall} = \frac{TP}{TP + FN}$$

**Precision:** $$\text{Precision} = \frac{TP}{TP + FP}$$

**False Positive Rate:** $$\text{FPR} = \frac{FP}{FP + TN}$$

Where:

- TP = number of true positives
- FN = number of false negatives
- FP = number of false positives
- TN = number of true negatives

### Worked Toy Example

Consider a cyber attack detection system with threshold θ:

|Scenario|True Label|Prediction (x < θ)|Classification|
|---|---|---|---|
|x = 0.2|y = 0|Negative|True Negative|
|x = 0.4|y = 1|Negative|False Negative|
|x = 0.7|y = 1|Positive|True Positive|
|x = 0.9|y = 0|Positive|False Positive|

If θ = 0.5:

- Recall = 1/(1+1) = 0.5 (caught 1 of 2 attacks)
- Precision = 1/(1+1) = 0.5 (1 of 2 positive predictions correct)

Moving θ left increases recall but decreases precision.

### Connections & Prerequisites

**Foundation for**: ROC curves, loss function design, threshold selection strategies

---

## 2. Concept: ROC Curves and Classifier Comparison

### High-Level Intuition

An ROC (Receiver Operating Characteristic) curve visualizes classifier performance across all possible threshold settings. It's like testing a metal detector at every possible sensitivity level and plotting how many real metals you find versus how many false alarms you trigger. The curve closer to the top-left corner represents the best classifier.

### Conceptual Deep Dive

The **ROC curve** plots:

- X-axis: False Positive Rate (FPR)
- Y-axis: True Positive Rate (Recall)

Each point on the curve represents a different threshold θ setting. The curve is constrained to the unit square [0,1] × [0,1].

**Classifier Quality:**

- **Best**: Curve close to (0, 1) - perfect classifier achieving 100% recall with 0% FPR
- **Worst**: Diagonal line where TPR = FPR - random guessing, no better than coin flip
- **Comparison**: For the same FPR, the classifier with higher TPR is superior

The **optimal threshold** θ* minimizes total probability of error, though in practice you may want to trade off based on application needs (e.g., medical diagnosis prioritizes minimizing false negatives).

### Mathematical Formulation

For a given classifier with threshold θ:

**True Positive Rate:** $$TPR(θ) = \frac{TP(θ)}{TP(θ) + FN(θ)}$$

**False Positive Rate:**  
$$FPR(θ) = \frac{FP(θ)}{FP(θ) + TN(θ)}$$

The ROC curve is the set of points: $$\text{ROC} = {(FPR(θ), TPR(θ)) : θ \in \mathbb{R}}$$

### Worked Toy Example

Consider three classifiers A, B, C with performance at θ = 0.5:

|Classifier|FPR|TPR|Performance|
|---|---|---|---|
|A|0.1|0.9|Excellent|
|B|0.2|0.7|Good|
|C|0.5|0.5|Random|

Classifier A is best because for low FPR (0.1), it achieves high TPR (0.9). Classifier C is worst—it's no better than random guessing.

### Connections & Prerequisites

**Prerequisite**: Understanding confusion matrix and classification metrics **Used in**: Model selection, performance evaluation, threshold tuning

---

## 3. Concept: Binary Cross-Entropy Loss

### High-Level Intuition

Binary cross-entropy measures how well your classifier's probability predictions match the true labels. Think of it as a penalty function that strongly punishes confident wrong predictions. If you're 95% sure an email is spam but it's actually legitimate, the penalty is much higher than if you were only 55% sure.

### Conceptual Deep Dive

The **cross-entropy loss** generalizes to both regression and classification. For binary classification, we model the label y as a **Bernoulli random variable** (like a coin flip) where ŷ represents the probability of the positive class.

Key insight: Since we're modeling a discrete binary outcome, a **Gaussian distribution is inappropriate**. Instead, we use the **Bernoulli distribution**:

$$P(y|x, w) = \hat{y}^y (1-\hat{y})^{(1-y)}$$

This formula elegantly handles both cases:

- When y = 1: P(y=1|x,w) = ŷ
- When y = 0: P(y=0|x,w) = 1 - ŷ

The model output ŷ must be interpreted as the **posterior probability of the positive event**.

### Mathematical Formulation

**General Cross-Entropy:** $$\mathcal{L} = -\mathbb{E}_{(x,y) \sim P_{\text{data}}}[\log P_{\text{model}}(y|x,w)]$$

**Binary Cross-Entropy (per example):** $$\ell(y, \hat{y}) = -[y \log(\hat{y}) + (1-y)\log(1-\hat{y})]$$

**Binary Cross-Entropy (dataset average):** $$\mathcal{L}(\mathbf{y}, \hat{\mathbf{y}}) = -\frac{1}{m}\sum_{i=1}^{m}[y_i \log(\hat{y}_i) + (1-y_i)\log(1-\hat{y}_i)]$$

Where:

- $m$ = number of training examples
- $y_i \in {0, 1}$ = true label for example $i$
- $\hat{y}_i \in [0, 1]$ = predicted probability for example $i$
- $w$ = model parameters

### Worked Toy Example

Consider 3 examples with predictions and true labels:

|Example|True y|Predicted ŷ|Loss Calculation|Loss Value|
|---|---|---|---|---|
|1|1|0.95|-[1·log(0.95) + 0·log(0.05)]|0.051|
|2|1|0.05|-[1·log(0.05) + 0·log(0.95)]|2.996|
|3|0|0.10|-[0·log(0.10) + 1·log(0.90)]|0.105|

**Analysis:**

- Example 1: Confident and correct → small loss
- Example 2: Confident and wrong → very large loss
- Example 3: Somewhat correct → moderate loss

Average loss: (0.051 + 2.996 + 0.105) / 3 ≈ 1.05

### Connections & Prerequisites

**Prerequisite**: Understanding of probability distributions, logarithms, Bernoulli distribution **Leads to**: Gradient computation for training, logistic regression loss function

---

## 4. Concept: Logistic Regression (Binary Classification)

### High-Level Intuition

Logistic regression is the simplest neural classifier. Think of it as linear regression followed by a "squashing function" that converts any real number into a probability between 0 and 1. It's like having a linear decision boundary, but instead of giving hard yes/no answers, it gives you confidence scores.

### Conceptual Deep Dive

Logistic regression addresses a key problem: linear models (y = w^T x) can output any real number, but we need outputs in [0, 1] to interpret as probabilities.

**Solution**: Apply the **sigmoid function** σ(·) after the linear combination:

$$\hat{y} = \sigma(w^T x + b) = \sigma(a)$$

Where $a = w^T x + b$ is called the **logit** or **activation**.

The **sigmoid function** has important properties:

- Maps (-∞, ∞) → (0, 1)
- Smooth and differentiable everywhere
- S-shaped curve with inflection point at 0.5
- Asymptotically approaches 0 and 1

**Historical note**: The steep limit of sigmoid becomes the **Perceptron** (Rosenblatt, 1960s), which acts like a binary switch.

### Mathematical Formulation

**Logistic Regression Hypothesis:** $$\hat{y} = \sigma(w^T x + b)$$

**Sigmoid Function:** $$\sigma(a) = \frac{1}{1 + e^{-a}}$$

**Decision Rule:** $$\text{Class} = \begin{cases} 1 & \text{if } \hat{y} \geq 0.5 \ 0 & \text{if } \hat{y} < 0.5 \end{cases}$$

**Full Model (with features):** $$\hat{y} = \sigma(w^T \phi(x) + b)$$

Where:

- $w \in \mathbb{R}^d$ = weight vector
- $x \in \mathbb{R}^n$ = input features
- $b \in \mathbb{R}$ = bias term
- $\phi(x)$ = feature transformation (optional)
- $a$ = logit (pre-activation)

### Worked Toy Example

**Setup**: Classify whether signal strength indicates an attack.

Given: $x = 0.7$ (signal strength), $w = 2.0$, $b = -1.0$

**Step 1**: Compute logit $$a = w^T x + b = 2.0(0.7) + (-1.0) = 0.4$$

**Step 2**: Apply sigmoid $$\hat{y} = \sigma(0.4) = \frac{1}{1 + e^{-0.4}} = \frac{1}{1 + 0.670} = 0.599$$

**Step 3**: Make decision Since ŷ = 0.599 > 0.5 → Predict class 1 (attack detected)

**Interpretation**: The model is about 60% confident there's an attack.

### Connections & Prerequisites

**Prerequisite**: Linear regression, understanding of probability **Foundation for**: Neural networks, multi-layer perceptrons **Note**: Despite the name "regression," this is a classification method

---

## 5. Concept: Dense Layers and Neural Network Architecture

### High-Level Intuition

A dense layer (fully-connected layer) is like having multiple logistic regression units working in parallel, each learning to detect different patterns. Think of it as a team of specialists—one might detect edges, another curves, another textures—and their combined insights form richer features that subsequent layers can use for better decisions.

### Conceptual Deep Dive

A **dense layer** generalizes the single neuron to multiple parallel neurons. Key architectural insight: we split the network into two parts:

1. **Body**: Learns features automatically from raw data
2. **Head**: Combines learned features to make final predictions

**Critical property**: The features produced by the body are **not independent** of the head's performance. During training via stochastic gradient descent, all parameters (both body and head) are **jointly optimized**. This means features adapt to improve the final prediction.

**Dense Layer Structure:**

- Input: vector $x \in \mathbb{R}^{n_x}$
- Linear transformation: $z = Wx + b$
- Nonlinear activation: $h = \text{activation}(z)$
- Output: vector $h \in \mathbb{R}^{n_z}$

The **weight matrix** $W$ has dimensions $n_z \times n_x$, where typically $n_z < n_x$ (dimensionality reduction), implementing the "pyramid" structure.

### Mathematical Formulation

**Dense Layer Equations:**

**Linear transformation:** $$z = Wx + b$$

**Activation:** $$h = \text{ReLU}(z) = \max(0, z)$$

**Dimension matching:**

- $x \in \mathbb{R}^{n_x \times 1}$
- $W \in \mathbb{R}^{n_z \times n_x}$
- $b \in \mathbb{R}^{n_z \times 1}$
- $z \in \mathbb{R}^{n_z \times 1}$
- $h \in \mathbb{R}^{n_z \times 1}$

**ReLU Activation (element-wise):** $$\text{ReLU}(z_i) = \max(0, z_i) = \begin{cases} z_i & \text{if } z_i > 0 \ 0 & \text{if } z_i \leq 0 \end{cases}$$

**Simple Neural Network (2-layer):** $$h_1 = \text{ReLU}(W_1 x + b_1)$$ $$\hat{y} = \sigma(W_2 h_1 + b_2)$$

Note: The final layer for binary classification uses **sigmoid**, not ReLU.

### Worked Toy Example

**Dense layer with $n_x = 3$ (input), $n_z = 2$ (output):**

$$W = \begin{bmatrix} w_{11} & w_{12} & w_{13} \ w_{21} & w_{22} & w_{23} \end{bmatrix}, \quad x = \begin{bmatrix} x_1 \ x_2 \ x_3 \end{bmatrix}, \quad b = \begin{bmatrix} b_1 \ b_2 \end{bmatrix}$$

**Computation:**

$$z = Wx + b = \begin{bmatrix} w_{11}x_1 + w_{12}x_2 + w_{13}x_3 + b_1 \ w_{21}x_1 + w_{22}x_2 + w_{23}x_3 + b_2 \end{bmatrix}$$

$$h = \text{ReLU}(z) = \begin{bmatrix} \max(0, z_1) \ \max(0, z_2) \end{bmatrix}$$

**Concrete Example:** Let $W = \begin{bmatrix} 1 & -2 & 0.5 \ 0.5 & 1 & -1 \end{bmatrix}$, $x = \begin{bmatrix} 2 \ 1 \ -1 \end{bmatrix}$, $b = \begin{bmatrix} 0 \ 1 \end{bmatrix}$

$$z_1 = 1(2) + (-2)(1) + 0.5(-1) + 0 = -0.5$$ $$z_2 = 0.5(2) + 1(1) + (-1)(-1) + 1 = 4$$

$$h = \begin{bmatrix} \max(0, -0.5) \ \max(0, 4) \end{bmatrix} = \begin{bmatrix} 0 \ 4 \end{bmatrix}$$

### Connections & Prerequisites

**Prerequisite**: Understanding of logistic regression, matrix multiplication **Key insight**: Body delivers features to head; features adapt during training **Parameter count**: For $n_z \times n_x$ matrix plus bias = $n_z \cdot n_x + n_z$ parameters

---

## 6. Concept: Multi-Class Classification and Softmax

### High-Level Intuition

Softmax generalizes binary classification to multiple classes. Instead of outputting a single probability (positive class), it outputs a probability distribution over all classes. Think of it as a competition where each class gets a confidence score, and softmax ensures these scores are positive and sum to 1.

### Conceptual Deep Dive

**Multi-class classification** requires predicting one of $K$ classes (K > 2). The output is a **posterior probability distribution**:

$$\hat{\mathbf{y}} = [\hat{y}_1, \hat{y}_2, ..., \hat{y}_K]^T$$

Where $\hat{y}_k$ = P(class = k | x).

**Softmax function** converts arbitrary real-valued logits into valid probabilities:

1. **Exponentiation**: Makes all values positive (handles negative logits)
2. **Normalization**: Divides by sum to ensure total = 1

**Key property**: Softmax creates competition among classes. A strong prediction for one class suppresses confidence in others. If two classes are close in score, both will have moderate confidence.

**Decision rule**: $\text{Predicted class} = \arg\max_k \hat{y}_k$

### Mathematical Formulation

**Softmax Function:** $$\hat{y}_k = \frac{e^{z_k}}{\sum_{j=1}^{K} e^{z_j}}$$

Where:

- $z_k$ = logit for class $k$ (output before softmax)
- $\hat{y}_k$ = probability for class $k$
- $K$ = number of classes

**Properties:** $$\sum_{k=1}^{K} \hat{y}_k = 1 \quad \text{and} \quad \hat{y}_k \in (0, 1) \quad \forall k$$

**Multi-class Network Architecture:** $$h = \text{ReLU}(W_1 x + b_1) \quad \text{(body)}$$ $$z = W_2 h + b_2 \quad \text{(head, pre-softmax)}$$ $$\hat{\mathbf{y}} = \text{softmax}(z) \quad \text{(final probabilities)}$$

**Categorical Cross-Entropy Loss:** $$\mathcal{L} = -\frac{1}{m}\sum_{i=1}^{m}\sum_{k=1}^{K} y_{ik} \log(\hat{y}_{ik})$$

### Worked Toy Example

**Fashion MNIST Example** (10 classes):

Given logits from network: $z = [2.1, -0.5, 1.3, 0.8, -1.2, 0.3, 1.7, -0.8, 0.5, 1.0]$

**Step 1**: Compute exponentials $$e^{z_1} = e^{2.1} = 8.17, \quad e^{z_2} = e^{-0.5} = 0.61, \quad e^{z_3} = e^{1.3} = 3.67, \text{ etc.}$$

**Step 2**: Sum all exponentials $$\text{sum} = 8.17 + 0.61 + 3.67 + ... \approx 22.5$$

**Step 3**: Normalize $$\hat{y}_1 = \frac{8.17}{22.5} = 0.363 \quad \text{(T-shirt class)}$$ $$\hat{y}_2 = \frac{0.61}{22.5} = 0.027 \quad \text{(Trouser class)}$$ $$\hat{y}_3 = \frac{3.67}{22.5} = 0.163 \quad \text{(Pullover class)}$$

**Decision**: Predict class 1 (highest probability = 0.363)

**Confidence interpretation**: 36.3% confident it's a T-shirt, with pullover as second choice (16.3%).

### Connections & Prerequisites

**Prerequisite**: Binary classification, probability distributions, logistic regression **Extends**: Binary classification to K classes **Used with**: Categorical cross-entropy loss **Note**: Final layer dimensions must equal number of classes

---

## 7. Concept: Backpropagation Algorithm

### High-Level Intuition

Backpropagation is the algorithm that makes neural network training efficient. Think of it as a smart accounting system: instead of recalculating how every parameter affects the loss from scratch, it reuses intermediate calculations by flowing gradients backward through the network. Like calculating tax deductions—once you know your total income, you can efficiently figure out each deduction's impact.

### Conceptual Deep Dive

**The Problem**: Computing gradients for thousands of parameters appears to require expensive symbolic differentiation for each parameter individually—computationally infeasible for large networks.

**The Solution**: Backpropagation exploits the **chain rule** and **locality** of computations.

**Computational Graph**: Represent the function as a directed acyclic graph where:

- **Nodes** = elementary operations (addition, multiplication, sigmoid, etc.)
- **Edges** = tensors flowing between operations

**Two Passes**:

1. **Forward Pass**: Compute function value bottom-up, storing intermediate results
    
    - Start from inputs
    - Calculate each node's output using stored values
    - Store all intermediate tensors for backward pass
2. **Backward Pass**: Compute gradients top-down using chain rule
    
    - Start from final loss (gradient = 1)
    - Each gate receives **upstream gradient** from above
    - Gate computes **downstream gradients** using local derivatives
    - Pass downstream gradients to gates below

**Key Advantages**:

1. **Massive parallelization**: Gates with no dependencies can compute in parallel
2. **Lookup table approach**: Local derivatives computed symbolically once, then reused
3. **Memory reuse**: Each gradient calculation uses precomputed values from forward pass

### Mathematical Formulation

**Chain Rule Template** (for gate g):

**Forward pass**: $$z = g(x, y)$$

**Backward pass**: Given upstream gradient $\frac{\partial L}{\partial z}$, compute:

$$\frac{\partial L}{\partial x} = \frac{\partial L}{\partial z} \cdot \frac{\partial g}{\partial x}$$

$$\frac{\partial L}{\partial y} = \frac{\partial L}{\partial z} \cdot \frac{\partial g}{\partial y}$$

**Junction Rule** (when tensor used by multiple gates): $$\frac{\partial L}{\partial x} = \sum_{\text{paths}} \frac{\partial L}{\partial x_{\text{path}}}$$

**Common Gate Derivatives**:

- Addition: $\frac{\partial}{\partial x}(x + y) = 1$
- Multiplication: $\frac{\partial}{\partial x}(xy) = y$
- Sigmoid: $\frac{\partial}{\partial x}\sigma(x) = \sigma(x)(1-\sigma(x))$
- ReLU: $\frac{\partial}{\partial x}\max(0,x) = \begin{cases} 1 & x > 0 \ 0 & x \leq 0 \end{cases}$

### Worked Toy Example

**Function**: $f(x, y) = \frac{x + \sigma(y)}{\sigma(x) + (x+y)^2}$

**Forward Pass** (given x=2, y=1):

1. $\sigma_y = \sigma(1) = 0.731$
2. $\text{nu} = x + \sigma_y = 2 + 0.731 = 2.731$
3. $\sigma_x = \sigma(2) = 0.881$
4. $x_py = x + y = 3$
5. $x_py_sqr = 9$
6. $\text{denom} = \sigma_x + x_py_sqr = 9.881$
7. $\text{invdenom} = 1/9.881 = 0.101$
8. $f = \text{nu} \times \text{invdenom} = 0.276$

**Backward Pass** (starting from top):

**Gate 8** (multiplication):

- Upstream: $\frac{\partial f}{\partial f} = 1$
- Downstream:
    - $\frac{\partial f}{\partial \text{nu}} = 1 \cdot \text{invdenom} = 0.101$
    - $\frac{\partial f}{\partial \text{invdenom}} = 1 \cdot \text{nu} = 2.731$

**Gate 7** (inversion $z = 1/x$):

- Upstream: $\frac{\partial f}{\partial \text{invdenom}} = 2.731$
- Downstream: $\frac{\partial f}{\partial \text{denom}} = 2.731 \cdot (-1/\text{denom}^2) = 2.731 \cdot (-0.0102) = -0.028$

**Gate 6** (addition):

- Upstream: $\frac{\partial f}{\partial \text{denom}} = -0.028$
- Downstream:
    - $\frac{\partial f}{\partial \sigma_x} = -0.028 \cdot 1 = -0.028$
    - $\frac{\partial f}{\partial x_py_sqr} = -0.028 \cdot 1 = -0.028$

_...continue for remaining gates..._

**Final gradients**: $\frac{\partial f}{\partial x}$ and $\frac{\partial f}{\partial y}$ are computed by accumulating contributions through all paths.

### Connections & Prerequisites

**Prerequisite**: Calculus (chain rule, partial derivatives), computational graphs **Critical for**: Training all neural networks efficiently **Key insight**: Locality + chain rule + caching = efficiency **Analogy**: Think of gradient flow like water flow—junctions sum flows, gates transform them

---

### Key Takeaways & Formulas

**Essential Formulas**:

1. **Binary Cross-Entropy**: $\mathcal{L} = -\frac{1}{m}\sum_{i=1}^{m}[y_i \log(\hat{y}_i) + (1-y_i)\log(1-\hat{y}_i)]$
    
2. **Sigmoid**: $\sigma(a) = \frac{1}{1 + e^{-a}}$
    
3. **ReLU**: $\text{ReLU}(z) = \max(0, z)$
    
4. **Softmax**: $\hat{y}_k = \frac{e^{z_k}}{\sum_{j=1}^{K} e^{z_j}}$
    
5. **Backpropagation Chain Rule**: $\frac{\partial L}{\partial x} = \frac{\partial L}{\partial z} \cdot \frac{\partial g}{\partial x}$
    

**Core Concepts**:

- Binary classification requires balancing false positives vs false negatives (ROC curves visualize this trade-off)
- Cross-entropy is the principled loss for classification, derived from maximum likelihood with appropriate probability distributions
- Neural networks = feature learning (body) + decision making (head), jointly optimized
- ReLU activation in body, sigmoid/softmax in head (depending on binary/multi-class)
- Backpropagation enables efficient gradient computation through locality and reuse