### Executive Summary

This lecture transitions from dense neural networks to Convolutional Neural Networks (CNNs), introducing the foundational concepts of computer vision. The session covers the correlation operation, convolutional layers, pooling mechanisms, and architectural patterns in CNNs. Key topics include how filters detect spatial features through correlation, the mathematics of multi-dimensional convolutions, gradient flow challenges in deep networks, and how ResNets solve these problems through skip connections. The lecture emphasizes the shift from hand-designed features to learned feature extraction in neural networks optimized for image processing tasks.

---

## 1. Concept: Image Representation and Spatial Correlation

### High-Level Intuition

**Goal:** Detect specific patterns or features within an image without knowing their exact location.

**Analogy:** Think of searching for a specific shape in a large buffer like radar detection. You have a template (the pattern you're looking for) and you slide it across the entire signal space, measuring how well it matches at each location. When you find the strongest match (peak correlation), you've located your pattern.

### Conceptual Deep Dive

Images are represented as matrices where each pixel contains intensity values. For **grayscale images**, each pixel has a value typically ranging from 0 (black) to 255 (white), often normalized to [0, 1] by dividing by 255. For **color images**, we have three channels (Red, Green, Blue), creating a 3D volume of data.

The **correlation operation** is the fundamental mechanism for detecting features in images. Instead of having a full template of the object we're searching for, we use small primitive shapes called **patches** or **kernels** (typically 3×3 or 5×5). These kernels are slid across the image systematically, and at each position, we compute a **dot product** between the kernel values and the overlapping image region.

When the kernel pattern strongly matches the underlying image content, the correlation produces a large value. Multiple different kernels can detect different features: vertical edges, horizontal edges, curves, corners, etc. This multi-kernel approach allows us to extract diverse features from a single input image.

### Mathematical Formulation

For a 2D correlation operation at spatial location $(i, j)$:

$$Z(i,j) = \sum_{u} \sum_{v} X(i+u, j+v) \cdot W(u,v)$$

Where:

- $Z(i,j)$ is the output value at spatial coordinates $(i,j)$
- $X(i+u, j+v)$ represents the image intensity at the shifted position
- $W(u,v)$ is the kernel/filter weight at position $(u,v)$
- The summations iterate over the kernel dimensions

### Worked Toy Example

Consider a 5×5 grayscale image section and a 3×3 edge detection kernel:

**Image X:**

```
[0   0   0   255 255]
[0   0   0   255 255]
[0   0   0   255 255]
[0   0   0   255 255]
[0   0   0   255 255]
```

**Kernel W (vertical edge detector):**

```
[-1  0  1]
[-1  0  1]
[-1  0  1]
```

**Calculation at position (1,1):**

- Overlay kernel on image region starting at (1,1)
- Dot product: $(-1)(0) + (0)(0) + (1)(0) + (-1)(0) + (0)(0) + (1)(0) + (-1)(0) + (0)(0) + (1)(0) = 0$

**Calculation at position (1,2):**

- $(-1)(0) + (0)(0) + (1)(255) + (-1)(0) + (0)(0) + (1)(255) + (-1)(0) + (0)(0) + (1)(255) = 765$

This large positive value indicates a strong vertical edge at this location.

### Connections & Prerequisites

**Prerequisite Refresher:** This builds on the concept of matrix operations from linear algebra. The dot product is extended from vectors to 2D regions, where we multiply corresponding elements and sum all results. Understanding matrix element-wise operations and summations is essential for grasping spatial correlation.

---

## 2. Concept: Convolutional Layers and Feature Maps

### High-Level Intuition

**Goal:** Transform an input volume (image with depth) into an output volume that captures detected features while preserving spatial relationships.

**Analogy:** Think of a convolutional layer as a team of specialized detectives, each looking for different clues across an entire scene. Each detective (kernel) has their own specialty (edge detection, color patterns, textures) and produces a map showing where they found their clues (feature map). The collection of all detective reports forms the complete output volume.

### Conceptual Deep Dive

A **convolutional layer** processes 3D input volumes and produces 3D output volumes through learned filters. Unlike dense layers that flatten spatial structure, convolutional layers preserve spatial relationships between pixels.

The **input feature map** has dimensions: Height ($H_{L-1}$) × Width ($W_{L-1}$) × Depth ($M_{L-1}$). For RGB images at the first layer, depth = 3.

A **filter** is a 3D volume with dimensions: kernel_height × kernel_width × $M_{L-1}$. Crucially, the filter depth must equal the input depth to perform the 3D correlation. Each filter produces one "slice" of the output volume.

The **output feature map** has dimensions: $H_L$ × $W_L$ × $M_L$, where $M_L$ equals the number of filters used. This is a design choice - more filters allow the network to detect more diverse features.

Key architectural parameters include:

- **Stride**: How many pixels the filter moves at each step (stride=1 means no skipping)
- **Padding**: Adding zeros around the image border to control output dimensions and improve edge feature detection

### Mathematical Formulation

For a 3D convolutional operation producing output at position $(i, j, k_L)$:

$$Z(i,j,k_L) = \sum_{k_{L-1}=1}^{M_{L-1}} \sum_{u} \sum_{v} X(i+u, j+v, k_{L-1}) \cdot W(u, v, k_{L-1}, k_L) + b_{k_L}$$

Where:

- $Z(i,j,k_L)$ is the output value at spatial position $(i,j)$ in the $k_L$-th output channel
- $X(i+u, j+v, k_{L-1})$ is the input at the overlapping position in channel $k_{L-1}$
- $W(u, v, k_{L-1}, k_L)$ is the weight in filter $k_L$ at position $(u,v)$ in input channel $k_{L-1}$
- $b_{k_L}$ is the bias term for output channel $k_L$
- The outer sum aggregates across all input channels
- The inner sums perform spatial correlation

**Output spatial dimensions:**

$$H_L = \left\lfloor \frac{H_{L-1} + 2P - K}{S} \right\rfloor + 1$$

Where $P$ = padding, $K$ = kernel size, $S$ = stride

### Worked Toy Example

**Input volume:** 4×4×2 (height × width × depth)

**Filter specifications:** 3×3×2, using 1 filter

**Parameters:** stride=1, no padding

**Channel 1 of input:**

```
[1 2 3 4]
[5 6 7 8]
[9 8 7 6]
[5 4 3 2]
```

**Channel 2 of input:**

```
[2 1 0 1]
[3 2 1 0]
[4 3 2 1]
[5 4 3 2]
```

**Filter kernel (3×3 for each of 2 channels):**

- Channel 1 kernel: all ones
- Channel 2 kernel: all zeros

**Calculation at position (0,0):**

- From channel 1: $(1×1 + 2×1 + 3×1 + 5×1 + 6×1 + 7×1 + 9×1 + 8×1 + 7×1) = 48$
- From channel 2: $(2×0 + 1×0 + ... ) = 0$
- Total: $Z(0,0) = 48 + 0 + b = 48 + b$

**Output dimensions:** $\lfloor(4-3)/1\rfloor + 1 = 2$ × $2$ × $1$ (since we used 1 filter)

### Connections & Prerequisites

**Prerequisite Refresher:** Understanding requires comfort with 3D array indexing and the concept that depth represents different "views" or channels of the same spatial region. Each channel might represent color (RGB) or, in deeper layers, abstract learned features.

---

## 3. Concept: Convolutional Neuron and Nonlinearity

### High-Level Intuition

**Goal:** Create a computational unit that extracts spatial features and introduces nonlinearity to enable learning complex patterns.

**Analogy:** A convolutional neuron is like a feature extraction module on an assembly line. The raw data (image) enters, the correlation operation detects patterns, but before passing forward, the ReLU activation acts as a quality filter that removes negative/weak signals and amplifies positive detections, making the features more robust.

### Conceptual Deep Dive

A **convolutional neuron** combines three operations:

1. **Linear correlation** (the 3D dot product described above)
2. **Bias addition** (shifting the activation threshold)
3. **Nonlinear activation** (typically ReLU)

The equation for a convolutional neuron is:

$$H = \text{ReLU}(Z + b)$$

where $Z$ is the result of the 3D correlation operation.

The **nonlinearity is essential** for the same reason as in dense networks: without it, stacking multiple convolutional layers would be equivalent to a single linear transformation. The ReLU (Rectified Linear Unit) function is most common: $\text{ReLU}(x) = \max(0, x)$.

**Parameter count:** For a layer with:

- Input depth: $M_{L-1}$
- Filter size: $K \times K$
- Number of filters: $M_L$

Total parameters = $(K \times K \times M_{L-1}) \times M_L + M_L$

The first term counts weights, the second counts biases (one per filter).

### Mathematical Formulation

Complete convolutional neuron operation:

$$H_{L}(i,j,k_L) = \sigma\left(\sum_{k_{L-1}=1}^{M_{L-1}} \sum_{u=0}^{K-1} \sum_{v=0}^{K-1} X_{L-1}(i+u, j+v, k_{L-1}) \cdot W(u, v, k_{L-1}, k_L) + b_{k_L}\right)$$

Where:

- $H_L(i,j,k_L)$ is the activated output (feature map) at position $(i,j)$ in channel $k_L$
- $\sigma$ is the activation function (typically ReLU)
- All other terms defined as before

**ReLU activation function:**

$$\text{ReLU}(z) = \max(0, z) = \begin{cases} z & \text{if } z > 0 \ 0 & \text{if } z \leq 0 \end{cases}$$

### Worked Toy Example

Continuing the previous example with calculated $Z(0,0) = 48 + b$:

**Assume bias $b = -45$:**

$Z(0,0) = 48 - 45 = 3$

**Apply ReLU:**

$H(0,0) = \text{ReLU}(3) = \max(0, 3) = 3$

**If we had different position with $Z(1,1) = -10 + b = -55$:**

$H(1,1) = \text{ReLU}(-55) = \max(0, -55) = 0$

**Parameter count example:**

- Input: 32×32×3 (RGB image)
- Layer: 64 filters of size 3×3
- Parameters = $(3 \times 3 \times 3) \times 64 + 64 = 27 \times 64 + 64 = 1,728 + 64 = 1,792$ parameters

### Connections & Prerequisites

**Prerequisite Refresher:** The need for nonlinearity comes from the Universal Approximation Theorem - without nonlinear activations, even deep networks can only learn linear mappings. ReLU has become dominant because it avoids vanishing gradients (for positive values) and is computationally efficient.

---

## 4. Concept: Pooling Layers and Dimensionality Reduction

### High-Level Intuition

**Goal:** Reduce spatial dimensions while retaining the most important features, improving computational efficiency and providing translation invariance.

**Analogy:** Think of pooling as creating a highlight reel. Instead of keeping every frame of a sports game (all pixel details), you keep only the most exciting moments (maximum activations). This gives you the essential information in a compressed form that's also more robust to small shifts in the original footage.

### Conceptual Deep Dive

**Pooling layers** perform dimensionality reduction by applying a function over local regions without any learned parameters. They operate independently on each depth channel.

**Max Pooling** (most common): Selects the maximum value from each region. For a 2×2 pooling window:

- Slide window across feature map
- Output the maximum value in each window
- Typically use stride equal to window size (non-overlapping)

**Benefits:**

1. **Translation invariance**: Small shifts in input don't dramatically change output
2. **Computational efficiency**: Reduces spatial dimensions, lowering computation in subsequent layers
3. **Feature distillation**: Propagates strongest activations (most confident feature detections)

**Trade-offs:**

- **Information loss**: Discards spatial details and non-maximum activations
- Increasingly questioned in modern architectures (some newer networks avoid pooling)

**Average Pooling** (less common): Computes mean of each region instead of maximum.

### Mathematical Formulation

**Max Pooling operation:**

$$H_L(i, j, k) = \max_{(u,v) \in \mathcal{R}_{i,j}} X_{L-1}(u, v, k)$$

Where:

- $H_L(i,j,k)$ is the pooled output at position $(i,j)$ in channel $k$
- $\mathcal{R}_{i,j}$ is the rectangular pooling region corresponding to output position $(i,j)$
- The max operation selects the largest value in that region
- Operates independently on each channel $k$

**Output dimensions with pooling:**

$$H_L = \left\lfloor \frac{H_{L-1} - K_{pool}}{S_{pool}} \right\rfloor + 1$$

Where $K_{pool}$ is pooling window size and $S_{pool}$ is pooling stride

### Worked Toy Example

**Input feature map (4×4, single channel):**

```
[1  3  2  4]
[5  6  7  8]
[9  2  1  3]
[4  5  6  7]
```

**Max pooling parameters:** 2×2 window, stride=2

**Pooling regions and outputs:**

- **Region (0,0):** $[1, 3, 5, 6] \rightarrow \max = 6$
- **Region (0,2):** $[2, 4, 7, 8] \rightarrow \max = 8$
- **Region (2,0):** $[9, 2, 4, 5] \rightarrow \max = 9$
- **Region (2,2):** $[1, 3, 6, 7] \rightarrow \max = 7$

**Output feature map (2×2):**

```
[6  8]
[9  7]
```

**Dimension reduction:** 4×4 = 16 values reduced to 2×2 = 4 values (75% reduction)

### Connections & Prerequisites

**Prerequisite Refresher:** Pooling can be viewed as a special case of convolution where: (1) there are no learned parameters, (2) the operation is not a weighted sum but a selection/aggregation function, and (3) the "filter" typically moves with stride equal to its size, creating non-overlapping regions.

---

## 5. Concept: CNN Architecture Patterns and Filter Depth Progression

### High-Level Intuition

**Goal:** Design deep networks that progressively build from simple to complex feature representations while managing computational cost.

**Analogy:** Think of a CNN as a pyramid of abstraction. At the base (early layers), simple pattern detectors spot edges and colors - like identifying individual LEGO blocks. Middle layers combine these into shapes and textures - like recognizing LEGO structures. Top layers understand complete objects and scenes - like identifying a LEGO castle. As we move up, we need more "experts" (filters) to capture increasing complexity, but we're examining smaller spatial regions.

### Conceptual Deep Dive

**Typical CNN architecture pattern:**

```
Input → [Conv → ReLU → Pool] × N → Flatten → Dense → Output
```

**Key architectural trends:**

1. **Increasing depth:** As we go deeper, the number of filters typically increases (e.g., 32 → 64 → 128 → 256 → 512)
    
2. **Decreasing spatial dimensions:** Pooling and strided convolutions reduce height and width
    
3. **Why more filters at deeper layers:** Early layers detect primitive features (edges, colors) that are spatially local. Deeper layers detect abstract, semantic features (object parts, patterns) requiring more diverse combinations of lower-level features. More filters = more ways to combine information.
    
4. **Transition to dense layers:** Eventually spatial dimensions become small (e.g., 7×7), and we **flatten** the volume into a 1D vector and use dense layers for final classification.
    

**Cost analysis:** Dense layers often dominate parameter count despite CNNs being "expensive." For a 6,272-dimensional flattened vector connected to a dense layer with 128 neurons: $6,272 \times 128 = 802,816$ parameters just for that single connection.

### Mathematical Formulation

**Typical architecture progression:**

$$\text{Input: } 224 \times 224 \times 3$$ $$\text{Conv1: } 224 \times 224 \times 64 \quad \text{(64 filters, 3×3)}$$ $$\text{Pool1: } 112 \times 112 \times 64 \quad \text{(2×2 max pool)}$$ $$\text{Conv2: } 112 \times 112 \times 128 \quad \text{(128 filters, 3×3)}$$ $$\text{Pool2: } 56 \times 56 \times 128$$ $$\text{...}$$ $$\text{Flatten: } [n] \quad \text{where } n = H_L \times W_L \times M_L$$ $$\text{Dense: } [n] \rightarrow [k] \quad \text{where } k = \text{num classes}$$

**Parameter counts:**

For convolutional layer: $(K \times K \times M_{in}) \times M_{out} + M_{out}$

For dense layer: $(n_{in} \times n_{out}) + n_{out}$

### Worked Toy Example

**Small CNN for binary classification (cats vs dogs):**

**Architecture:**

```
Input: 64×64×3
Conv1: 64×64×32 (3×3 filters, padding=1, stride=1)
Pool1: 32×32×32 (2×2 max pool, stride=2)
Conv2: 32×32×64 (3×3 filters, padding=1, stride=1)
Pool2: 16×16×64 (2×2 max pool, stride=2)
Flatten: 16,384 dimensions (16×16×64)
Dense1: 128 neurons
Output: 1 neuron (sigmoid activation)
```

**Parameter counts:**

- **Conv1:** $(3 \times 3 \times 3) \times 32 + 32 = 864 + 32 = 896$
- **Conv2:** $(3 \times 3 \times 32) \times 64 + 64 = 18,432 + 64 = 18,496$
- **Dense1:** $(16,384 \times 128) + 128 = 2,097,152 + 128 = 2,097,280$
- **Output:** $(128 \times 1) + 1 = 129$

**Total:** ~2.1 million parameters, with 98% in the dense layer!

### Connections & Prerequisites

**Prerequisite Refresher:** Understanding requires grasping that features become increasingly abstract through hierarchical composition. Each layer builds representations from the previous layer's outputs. Early layers might detect edges, mid layers detect textures or simple shapes, and deep layers detect high-level concepts like "cat face" or "wheel."

---

## 6. Concept: 1×1 Convolutions

### High-Level Intuition

**Goal:** Transform the depth dimension of a feature map while preserving spatial structure, enabling dimensionality reduction or expansion across channels.

**Analogy:** Think of a 1×1 convolution as a "channel mixer" that operates at each spatial location independently. It's like having a cocktail mixer at every pixel that combines the different flavor channels (depth dimensions) into new flavor combinations, without affecting the spatial arrangement of pixels.

### Conceptual Deep Dive

**1×1 convolutions** use filters of size 1×1×$M_{in}$ and are unique because:

1. **Preserve spatial dimensions:** Output height and width equal input (with appropriate padding/stride)
    
2. **Mix channel information:** At each spatial location $(i,j)$, combine all input channels using learned weights
    
3. **Dimensionality control:** Can increase or decrease depth by choosing number of filters
    

**Use cases:**

- **Dimensionality reduction:** Reduce depth before expensive 3×3 or 5×5 convolutions (used in Inception networks)
- **Dimensionality expansion:** Increase channels for richer representations
- **Adding nonlinearity:** Introduce additional ReLU activations without spatial filtering
- **Channel-wise feature combination:** Learn which channel combinations are useful

**Comparison to dense layers:** A 1×1 convolution is equivalent to applying a dense layer independently at each spatial location.

### Mathematical Formulation

**1×1 convolution operation:**

$$H_L(i,j,k_L) = \sigma\left(\sum_{k_{L-1}=1}^{M_{L-1}} X_{L-1}(i, j, k_{L-1}) \cdot W(k_{L-1}, k_L) + b_{k_L}\right)$$

Where:

- $H_L(i,j,k_L)$ is output at spatial position $(i,j)$ in output channel $k_L$
- $X_{L-1}(i,j,k_{L-1})$ is input at same spatial position in input channel $k_{L-1}$
- $W(k_{L-1}, k_L)$ is the weight connecting input channel $k_{L-1}$ to output channel $k_L$
- No spatial summation occurs (u=0, v=0 only)
- $\sigma$ is activation function (typically ReLU)

**Key observation:** At each spatial location, this is equivalent to a fully connected layer applied across channels.

### Worked Toy Example

**Input:** 4×4×3 feature map

**Operation:** 1×1 convolution with 2 filters

**Input at position (1,2) across all channels:**

- Channel 0: $X(1,2,0) = 5$
- Channel 1: $X(1,2,1) = 3$
- Channel 2: $X(1,2,2) = 7$

**Filter 1 weights:** $W(:,0) = [0.5, -0.3, 0.2]$, bias $b_0 = 1$

**Filter 2 weights:** $W(:,1) = [0.1, 0.4, -0.2]$, bias $b_1 = 0$

**Calculation at position (1,2):**

**Output channel 0:** $$Z(1,2,0) = (5)(0.5) + (3)(-0.3) + (7)(0.2) + 1 = 2.5 - 0.9 + 1.4 + 1 = 4.0$$ $$H(1,2,0) = \text{ReLU}(4.0) = 4.0$$

**Output channel 1:** $$Z(1,2,1) = (5)(0.1) + (3)(0.4) + (7)(-0.2) + 0 = 0.5 + 1.2 - 1.4 = 0.3$$ $$H(1,2,1) = \text{ReLU}(0.3) = 0.3$$

**Output:** 4×4×2 feature map (spatial dimensions unchanged, depth reduced from 3 to 2)

**Parameter count:** $(1 \times 1 \times 3) \times 2 + 2 = 6 + 2 = 8$ parameters total

### Connections & Prerequisites

**Prerequisite Refresher:** Understanding 1×1 convolutions requires recognizing that "convolution" is really about how we connect layers, not necessarily about spatial filtering. When the kernel is 1×1, we're performing weighted combinations of channels rather than detecting spatial patterns, making it a powerful tool for controlling information flow between layers.

---

## 7. Concept: Residual Networks (ResNets) and Skip Connections

### High-Level Intuition

**Goal:** Enable training of very deep networks (50+ layers) by addressing the gradient flow problem through alternate paths.

**Analogy:** Imagine information flowing through a company hierarchy. In a traditional organization (standard CNN), information must pass through every manager from bottom to top, and each layer can distort the message. Skip connections create "express elevators" where information can bypass middle managers, ensuring the original message reaches the top intact while still allowing middle layers to add refinements.

### Conceptual Deep Dive

**The gradient flow problem:** Before ResNets, networks deeper than ~16-20 layers suffered from vanishing/exploding gradients during backpropagation. Each layer's gradient depends on multiplying many small derivatives, causing gradients to either vanish (→0) or explode (→∞).

**Skip connections (residual connections):** Add the input directly to the output of one or more layers:

$$Y_i = F_i(Y_{i-1}) + Y_{i-1}$$

Where $F_i$ represents the transformation of layer(s) $i$.

**Why this helps:**

1. **Gradient highways:** During backpropagation, gradients can flow directly through skip connections without attenuation
    
2. **Identity mapping:** If a layer learns the zero transformation, the skip connection maintains identity mapping, preventing performance degradation
    
3. **Multiple gradient paths:** Creates many paths of varying depths through the network
    
4. **Ensemble effect:** The network implicitly behaves like an ensemble of shallow networks
    

**Architectural pattern:** Residual blocks typically contain:

```
Input → Conv → BatchNorm → ReLU → Conv → BatchNorm → (+Input) → ReLU → Output
```

### Mathematical Formulation

**Single residual block:**

$$Y_i = F_i(Y_{i-1}) + Y_{i-1}$$

**Expanded through multiple blocks:**

$$Y_3 = F_3(F_2(F_1(Y_0) + Y_0) + F_2(F_1(Y_0) + Y_0)) + F_2(F_1(Y_0) + Y_0) + F_1(Y_0) + Y_0$$

**Simplified notation:**

$$Y_3 = F_3(Y_2) + Y_2$$ $$Y_2 = F_2(Y_1) + Y_1$$ $$Y_1 = F_1(Y_0) + Y_0$$

**Gradient flow advantage:**

$$\frac{\partial \mathcal{L}}{\partial Y_0} = \frac{\partial \mathcal{L}}{\partial Y_3} \left( \frac{\partial F_3}{\partial Y_2} \frac{\partial F_2}{\partial Y_1} \frac{\partial F_1}{\partial Y_0} + \frac{\partial F_3}{\partial Y_2} \frac{\partial F_2}{\partial Y_1} + \frac{\partial F_3}{\partial Y_2} + 1 \right)$$

The "+1" term provides a gradient highway - even if all $\frac{\partial F_i}{\partial Y_i}$ terms are small, gradient still flows through.

### Worked Toy Example

**Scenario:** 3-block ResNet vs standard network

**Standard network backpropagation:**

- Gradient must pass through all layers: $\frac{\partial \mathcal{L}}{\partial Y_0} = \frac{\partial \mathcal{L}}{\partial Y_3} \cdot \frac{\partial F_3}{\partial Y_2} \cdot \frac{\partial F_2}{\partial Y_1} \cdot \frac{\partial F_1}{\partial Y_0}$
- If each derivative ≈ 0.5: $(0.5)^3 = 0.125$ (gradient shrinks exponentially)

**ResNet backpropagation paths:**

Path 1 (shortest): $\frac{\partial \mathcal{L}}{\partial Y_3} \cdot 1$ (direct connection, no attenuation)

Path 2: $\frac{\partial \mathcal{L}}{\partial Y_3} \cdot \frac{\partial F_3}{\partial Y_2}$ (through 1 layer)

Path 3: $\frac{\partial \mathcal{L}}{\partial Y_3} \cdot \frac{\partial F_3}{\partial Y_2} \cdot \frac{\partial F_2}{\partial Y_1}$ (through 2 layers)

Path 4: Full path through all 3 layers

**Result:** Even if paths 2-4 have vanishing gradients, path 1 ensures $Y_0$ receives strong gradient signal.

**Numerical example:**

- Assume $\frac{\partial \mathcal{L}}{\partial Y_3} = 1.0$
- Standard net: gradient at $Y_0$ ≈ 0.125
- ResNet: gradient at $Y_0$ ≈ 1.0 + 0.5 + 0.25 + 0.125 = 1.875 (much stronger)

### Connections & Prerequisites

**Prerequisite Refresher:** Understanding requires familiarity with the chain rule in backpropagation. When computing $\frac{\partial \mathcal{L}}{\partial \theta}$, we multiply derivatives along each path from loss to parameter. In deep networks, this multiplication of many small numbers causes vanishing gradients. Skip connections add parallel paths where fewer multiplications occur, maintaining gradient strength.

---

## 8. Concept: Ensemble Learning in ResNets

### High-Level Intuition

**Goal:** Leverage the principle that combining multiple weak predictors can achieve better performance than a single strong predictor.

**Analogy:** A ResNet behaves like a committee making decisions. Instead of one expert who went through extensive training (deep network), you have many experts with varying levels of experience (different path depths through skip connections). Some made quick judgments (shallow paths), others deliberated deeply (full-depth paths). The final decision aggregates all perspectives, reducing individual errors.

### Conceptual Deep Dive

**Ensemble learning principle:** A committee of $K$ weak predictors can outperform a single strong predictor if:

1. **Diversity:** Predictors make different types of errors (uncorrelated mistakes)
2. **Independence:** Predictors use different data or learning approaches
3. **Weak competence:** Each predictor performs better than random

**Performance bounds:**

- **Lower bound:** Performance of single weak predictor
- **Upper bound:** Achieved when predictor errors are completely uncorrelated

**ResNets as implicit ensembles:** The network can be viewed as an exponential ensemble of paths of varying depths:

- In a 3-block ResNet: 8 possible paths (2³) from input to output
- Each path represents a different "predictor"
- Paths have different computational depths (weak vs strong)
- Training optimizes all paths simultaneously

### Concept: Ensemble Learning in ResNets (continued)

**Diversity mechanisms in CNNs:**

1. **Data augmentation:** Random transformations create different training examples
2. **Dropout:** Randomly deactivates neurons, training different sub-networks
3. **Different architectures:** Combining CNN, logistic regression, decision trees
4. **Stochastic gradient descent:** Mini-batch sampling introduces randomness

**ResNet's advantage:** Skip connections naturally create architectural diversity without explicitly training separate models. The network learns to use different path combinations for different inputs.

### Mathematical Formulation

**General ensemble prediction:**

$$Y_{committee} = \frac{1}{K} \sum_{k=1}^{K} f_k(X)$$

Where $f_k$ represents the $k$-th weak predictor.

**ResNet as implicit ensemble:** For a 3-block ResNet, the paths from $Y_0$ to $Y_3$ are:

Path A: $Y_0 \rightarrow Y_3$ (identity, 0 layers) Path B: $F_1(Y_0) + Y_0 \rightarrow Y_3$ (1 layer) Path C: $F_2(F_1(Y_0) + Y_0) + F_1(Y_0) + Y_0 \rightarrow Y_3$ (2 layers) Path D: Full expression with all 3 $F$ functions (3 layers)

**Expanded form shows implicit ensemble:**

$$Y_3 = F_3(Y_2) + F_2(Y_1) + F_1(Y_0) + Y_0 + \text{interaction terms}$$

This resembles weighted sum of predictors at different depths.

**Committee prediction strength:** If $K$ uncorrelated predictors each have error probability $p < 0.5$, the majority vote error probability is:

$$P_{ensemble} = \sum_{k=\lceil K/2 \rceil}^{K} \binom{K}{k} p^k (1-p)^{K-k}$$

This decreases exponentially with $K$ when errors are uncorrelated.

### Worked Toy Example

**Scenario:** Binary classification with 3 weak predictors

**Individual predictor accuracies:** 60%, 62%, 58% (all better than random 50%)

**Predictions on test sample:**

- Predictor 1: Class A (correct)
- Predictor 2: Class B (incorrect)
- Predictor 3: Class A (correct)

**Majority vote:** 2 votes for Class A → Ensemble predicts Class A (correct)

**Expected ensemble performance:** If errors are uncorrelated and each predictor has accuracy 60%:

For 3 predictors, ensemble is correct if ≥2 predictors are correct:

- All 3 correct: $(0.6)^3 = 0.216$
- Exactly 2 correct: $\binom{3}{2}(0.6)^2(0.4)^1 = 3 \times 0.36 \times 0.4 = 0.432$
- Total: $0.216 + 0.432 = 0.648$ (64.8% accuracy)

**Ensemble improvement:** 64.8% vs 60% - nearly 5% gain from combining weak predictors!

**Importance of diversity:** If all predictors make the same mistakes (perfectly correlated errors), ensemble accuracy = 60% (no improvement). Diversity is crucial.

### Connections & Prerequisites

**Prerequisite Refresher:** Ensemble learning builds on probability theory and the law of large numbers. When combining independent random variables (predictor outputs), variance decreases while bias remains constant. This is why ensembles are more stable and reliable than individual models, even if each component is relatively weak. ResNets exploit this principle through their multi-path architecture.

---

## 9. Concept: Data Augmentation for CNNs

### High-Level Intuition

**Goal:** Artificially expand the training dataset by applying transformations that preserve semantic content while creating variation, helping prevent overfitting.

**Analogy:** If you're teaching someone to recognize dogs using only 10 photos, they might memorize those specific images rather than learning "dogness." But if you show those same 10 photos from different angles, lighting conditions, with crops, and flips, you create hundreds of varied examples that force learning of robust features that generalize beyond the specific training images.

### Conceptual Deep Dive

**The overfitting problem in computer vision:** CNNs have millions of parameters (e.g., 3.4M in the cats vs dogs example). With limited data (e.g., 2,000 cats + 2,000 dogs), the model memorizes training examples rather than learning generalizable features.

**Data augmentation** creates additional training examples through label-preserving transformations:

**Common transformations:**

1. **Geometric:** Rotation, translation, scaling, flipping, shearing
2. **Color:** Brightness/contrast adjustment, hue shift, saturation changes
3. **Cropping:** Random crops, center crops
4. **Noise:** Gaussian noise, blur, cutout (masking patches)
5. **Advanced:** Mixup (blending images), CutMix, AutoAugment

**Implementation considerations:**

- Applied randomly during training (different each epoch)
- Not applied during validation/testing (except possibly center crop)
- Some libraries use GPU acceleration (e.g., Kornia) for efficiency
- Others are CPU-based (e.g., Albumentations)

**Benefits:**

- Reduces overfitting by increasing effective dataset size
- Improves model robustness to variations in real-world data
- Acts as implicit regularization
- Helps model learn invariant features

### Mathematical Formulation

**Augmentation as transformation function:**

$$X_{aug} = T(X, \theta)$$

Where:

- $X$ is original image
- $T$ is transformation function (rotation, flip, etc.)
- $\theta$ are random parameters (e.g., rotation angle sampled from distribution)
- Label remains unchanged: $y_{aug} = y$

**Effective dataset size:**

$$N_{effective} = N_{original} \times E \times A$$

Where:

- $N_{original}$ = original dataset size
- $E$ = number of epochs
- $A$ = average number of unique augmentations per image per epoch

**Example transformations:**

**Horizontal flip:** $$X_{flip}(i, j) = X(i, W - j - 1)$$

**Rotation by angle $\theta$:** $$\begin{bmatrix} i' \ j' \end{bmatrix} = \begin{bmatrix} \cos\theta & -\sin\theta \ \sin\theta & \cos\theta \end{bmatrix} \begin{bmatrix} i \ j \end{bmatrix}$$

**Brightness adjustment:** $$X_{bright}(i,j,k) = \text{clip}(X(i,j,k) + \delta, 0, 1)$$

Where $\delta \sim \mathcal{U}(-0.2, 0.2)$

### Worked Toy Example

**Original dataset:** 100 cat images, 100 dog images (200 total)

**Augmentation strategy per image:**

1. Random horizontal flip (50% probability)
2. Random rotation (-15° to +15°)
3. Random brightness (±20%)
4. Random crop (90% of original size)

**Single image transformations:**

**Original cat image:** 224×224×3

**Epoch 1, Image 1:**

- Flip: Yes → horizontally flipped version
- Rotation: +7° → slightly rotated
- Brightness: +10% → slightly brighter
- Crop: Random 200×200 region → resized to 224×224
- Result: Unique augmented image A

**Epoch 2, Image 1 (same original, different random parameters):**

- Flip: No → original orientation
- Rotation: -12° → rotated opposite direction
- Brightness: -15% → darker
- Crop: Different 200×200 region
- Result: Different augmented image B

**Effective training examples:**

- Original: 200 images
- 50 epochs with random augmentations
- Effective: ~200 × 50 = 10,000 unique training examples
- Model sees orders of magnitude more variation

**Impact on overfitting:**

**Without augmentation:**

- Training loss: 0.05 → Validation loss: 0.35 (large gap = overfitting)

**With augmentation:**

- Training loss: 0.15 → Validation loss: 0.20 (smaller gap = better generalization)

### Connections & Prerequisites

**Prerequisite Refresher:** Data augmentation relies on the assumption that certain transformations don't change semantic content. For images, flipping a cat horizontally still shows a cat. This differs from tasks like text classification where character-level changes alter meaning. Understanding which transformations are "label-preserving" for your specific task is critical for effective augmentation.

---

## 10. Concept: Feature Abstraction Hierarchy in CNNs

### High-Level Intuition

**Goal:** Understand how CNNs progressively transform low-level pixel patterns into high-level semantic concepts through hierarchical feature learning.

**Analogy:** Think of reading a book. Early layers are like recognizing individual letters (low-level features). Middle layers are like identifying words and phrases (mid-level patterns). Deep layers are like understanding paragraphs and themes (high-level concepts). Each level builds meaning from the previous level's outputs.

### Conceptual Deep Dive

**Feature hierarchy in trained CNNs:**

**Layer 1-2 (Early layers):**

- Detect primitive features: edges, corners, colors, simple textures
- Filters learn Gabor-like patterns (oriented edges at various angles)
- Feature maps retain clear spatial/geometric structure
- Relatively few filters needed (32-64) since primitives are limited

**Layers 3-4 (Middle layers):**

- Combine primitives into parts: curves, simple shapes, texture patterns
- Begin losing clear geometric correspondence to input
- More abstract than raw pixels but still somewhat interpretable
- Filter count increases (128-256) to capture diverse combinations

**Layers 5+ (Deep layers):**

- Detect high-level concepts: object parts, faces, wheels, windows
- Feature maps appear abstract, artistic, difficult to interpret visually
- Encode semantic information (what's in the image) over spatial information (where things are)
- Maximum filter count (512+) to capture rich semantic combinations

**Why increasing depth matters:**

- Each layer's receptive field grows (sees larger input regions)
- Hierarchical composition enables exponential expressiveness
- Later layers need more filters to combine diverse lower-level features in meaningful ways

**Visualizing learned features:**

- **Filter visualization:** Show what patterns maximize filter activation
- **Feature map visualization:** Show layer outputs for specific inputs
- **Activation maximization:** Generate synthetic images that maximally activate neurons
- **Attention maps:** Highlight input regions most relevant to predictions (e.g., Grad-CAM)

### Mathematical Formulation

**Receptive field growth:**

For layer $L$ with filter size $K$ and stride $S$:

$$RF_L = RF_{L-1} + (K-1) \prod_{i=1}^{L-1} S_i$$

Where $RF_L$ is the receptive field size at layer $L$.

**Example:** 3 layers, each with 3×3 filters, stride 1:

- Layer 1: $RF_1 = 3$
- Layer 2: $RF_2 = 3 + (3-1) \times 1 = 5$
- Layer 3: $RF_3 = 5 + (3-1) \times 1 = 7$

With pooling (stride 2 every other layer), growth is faster.

**Feature composition:** Output of layer $L$ can be expressed recursively:

$$H_L = \sigma(W_L * H_{L-1} + b_L)$$ $$H_L = \sigma(W_L * \sigma(W_{L-1} * H_{L-2} + b_{L-1}) + b_L)$$

Each layer composes functions, creating exponentially complex mappings.

**Dimensionality trend in typical CNN:**

|Layer|Spatial Dims|Depth|Total Elements|
|---|---|---|---|
|Input|224×224|3|150,528|
|Conv1|224×224|64|3,211,264|
|Pool1|112×112|64|802,816|
|Conv2|112×112|128|1,605,632|
|Pool2|56×56|128|401,408|
|Conv3|56×56|256|802,816|

Spatial dimensions ↓, Depth ↑, capturing more abstract features in smaller spatial regions.

### Worked Toy Example

**Scenario:** Analyzing learned features in a CNN trained on ImageNet

**Layer 1 filter visualization:**

- Filter 1: Detects horizontal edges (strong response to $\begin{bmatrix} 1 & 1 & 1 \ 0 & 0 & 0 \ -1 & -1 & -1 \end{bmatrix}$ pattern)
- Filter 2: Detects vertical edges
- Filter 3: Detects diagonal edges
- Filter 4: Detects red-green color contrast

**Layer 2 filter visualization:**

- Filter 1: Combines multiple edge filters → detects corners
- Filter 2: Combines color and edge → detects texture patterns
- Still somewhat interpretable as geometric patterns

**Layer 5 filter visualization:**

- Filter 1: Activates on dog faces (not just edges or shapes)
- Filter 2: Activates on wheels/circular objects
- Filter 3: Activates on text/windows patterns
- Highly abstract, difficult to describe geometrically

**Input image: Photo of taxi**

**Feature map progression:**

- **Layer 1 output:** Shows all edges in scene (car, buildings, roads clearly outlined)
- **Layer 3 output:** Shows texture regions (car body, window glass, pavement) less sharp
- **Layer 5 output:** Abstract patterns, spatial structure largely lost, but "vehicle-ness" encoded

**Why this matters for depth:**

- Early layers learn universal features (edges work for any image)
- Deep layers learn task-specific features (car parts only useful for vehicle classification)
- Need more filters at depth to capture diverse high-level concepts

### Connections & Prerequisites

**Prerequisite Refresher:** Understanding feature hierarchies connects to the concept of compositionality in mathematics. Simple functions compose to create complex functions: $f(g(h(x)))$ can represent arbitrarily complex mappings even if $f$, $g$, $h$ are relatively simple. CNNs exploit this through layer stacking, where each "simple" convolutional layer composes with others to learn rich representations.

---

### Key Takeaways & Formulas

**Critical Concepts:**

1. **Spatial correlation is fundamental:** CNNs detect features by sliding learned kernels across images, computing dot products at each location
2. **Filter depth must match input depth:** A 3D filter with depth = input depth produces one output slice; multiple filters create output volumes
3. **Architecture patterns matter:** Increasing filter count with depth captures hierarchical abstraction while spatial dimensions shrink
4. **Skip connections enable deep networks:** ResNets solve gradient flow problems through multiple paths, enabling 50-100+ layer networks
5. **Data augmentation is essential:** With limited data and millions of parameters, augmentation prevents overfitting and improves generalization

**Must-Remember Formulas:**

**Output spatial dimensions:** $$H_L = \left\lfloor \frac{H_{L-1} + 2P - K}{S} \right\rfloor + 1$$

**3D Convolution:** $$Z(i,j,k_L) = \sum_{k_{L-1}=1}^{M_{L-1}} \sum_{u} \sum_{v} X(i+u, j+v, k_{L-1}) \cdot W(u, v, k_{L-1}, k_L) + b_{k_L}$$

**Convolutional layer parameters:** $$(K \times K \times M_{in}) \times M_{out} + M_{out}$$

**ResNet skip connection:** $$Y_i = F_i(Y_{i-1}) + Y_{i-1}$$

**ReLU activation:** $$\text{ReLU}(z) = \max(0, z)$$