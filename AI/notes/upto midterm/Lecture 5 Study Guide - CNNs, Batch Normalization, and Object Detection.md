# Convolutional Neural Networks (CNNs)

## 1. High-Level Intuition
A CNN processes images much like a detective scanning a scene with magnifying glasses of different shapes. Imagine sliding a small stamp (filter) over a photo: each stamp looks for a particular pattern (edge, texture, etc.) and lights up where it finds a match. This creates a _feature map_ that highlights where that feature exists in the image. Stacking many such filters (stamps) of different patterns lets the network detect complex shapes in a hierarchical way.

## 2. Conceptual Deep Dive
A _convolutional neuron_ performs a local linear operation: it "slides" a 3D filter (kernel) across the 2D input image (or previous feature map) and computes a weighted sum (correlation) plus a bias, followed by a nonlinearity (e.g. ReLU). Each filter has learnable weights, so it can detect a specific pattern (e.g. vertical edge, color blob). Using multiple filters per layer produces multiple output channels (forming a volume). Key terms: **filter/kernel** (the small weight matrix), **feature map** (result of filtering), **stride** (step size of the slide), **padding** (how image edges are handled), and **ReLU activation** (keeps positive signals). Deep networks stack convolutional layers so that higher layers detect higher-level features.

## 3. Mathematical Formulation
For a single-channel input $I$ and a filter $K$, the convolution output $O$ at position $(i,j)$ is

$$O[i,j] = (I * K)[i,j] + b = \sum_{u,v} I[i+u, j+v] K[u,v] + b,$$

where $b$ is a bias term. For a multi-channel input and multiple filters,

$$O_k[i,j] = \sum_{c=1}^C \sum_{u,v} I_c[i+u,j+v] K_{k,c}[u,v] + b_k,$$

where $I_c$ is channel $c$ of the input, $K_{k,c}$ is filter $k$ on channel $c$, and $O_k$ is the $k$-th output channel. Finally, a ReLU activation is applied: $A_k[i,j]=\max(0,O_k[i,j])$. Each equation term corresponds to the convolution operation and bias before activation.

## 4. Worked Toy Example
Suppose we have a simple $3\times3$ input image and a $2\times2$ filter:

**Input image $I$:**

$$\begin{bmatrix}1 & 2 & 3\\ 4 & 5 & 6\\ 7 & 8 & 9\end{bmatrix}$$

**Filter $K$ (no bias, single output channel):**

$$\begin{bmatrix}1 & 0\\ 0 & -1\end{bmatrix}$$

**Stride = 1, no padding.**

Convolve the filter over the image:

- At top-left corner: $$O[0,0] = 1\cdot1 + 4\cdot0 + 2\cdot0 + 5\cdot(-1) = 1 - 5 = -4.$$

- At top-middle: $$O[0,1] = 2\cdot1 + 5\cdot0 + 3\cdot0 + 6\cdot(-1) = 2 - 6 = -4.$$

- And so on for each position. Applying ReLU (zeroing negatives), the feature map highlights where the pattern (here, "1 then -1") is present. This simple numeric example shows how local dot-products produce each output pixel.

## 5. Connections & Prerequisites
CNNs build on linear algebra and signal processing concepts (convolution/correlation). A **Prerequisite Refresher**: know how matrix multiplication or convolution works in 2D, and understand simple neural network layers. CNNs generalize dense layers by sharing weights across space and exploiting locality.

---

# Batch Normalization

## 1. High-Level Intuition
Batch normalization is like calibrating every ingredient in a recipe so each batch tastes consistent. Just as a cook might adjust a batch of soup to have the same saltiness each time, a neural network adjusts activations to have a consistent distribution. This keeps the "flavor" (data distribution) stable as it passes through layers, preventing training from getting skewed by wildly varying scales.

## 2. Conceptual Deep Dive
Internal covariate shift refers to layer inputs changing distribution during training, which can slow learning. Batch normalization solves this by normalizing the activations **per batch**. Specifically, for each mini-batch and for each feature, BN computes the batch mean and variance and subtracts/divides to produce zero-mean, unit-variance activations. Then it applies learnable **scale ($\gamma$)** and **shift ($\beta$)** parameters so the network can choose any optimal mean and variance. Thus BN layers always see inputs of controlled scale, stabilizing gradient flow and often speeding up convergence. The lecture notes that BN "estimates the mean and variance of the activations over the minibatch" and learns $\gamma,\beta$ to let the network find the right scale and location (spread and offset).

## 3. Mathematical Formulation
For a layer input $x_i$ over a minibatch $\{x_i\}_{i=1}^m$, compute batch statistics:

$$\mu_B = \frac{1}{m}\sum_{i=1}^m x_i, \quad \sigma_B^2 = \frac{1}{m}\sum_{i=1}^m (x_i - \mu_B)^2$$

Normalize each activation:

$$\hat{x}_i = \frac{x_i - \mu_B}{\sqrt{\sigma^2_B + \epsilon}},$$

where $\epsilon$ is a small constant to avoid division by zero. Then scale and shift:

$$y_i = \gamma \hat{x}_i + \beta,$$

where $\gamma$ (scale) and $\beta$ (shift) are learned parameters. This ensures the output $y_i$ can adopt any needed distribution while training remains stable.

## 4. Worked Toy Example
Consider a batch of 3 activations $\{x_i\} = \{1, 2, 3\}$. Compute BN with $\gamma=1$, $\beta=0$ for simplicity:

- Batch mean: $\mu_B = (1+2+3)/3 = 2.$

- Batch variance: $\sigma^2_B = [(1-2)^2 + (2-2)^2 + (3-2)^2]/3 = (1+0+1)/3 = 0.67.$

- Normalize: $\hat{x}_1 = (1-2)/\sqrt{0.67+\epsilon} \approx -1.22, \hat{x}_2=0, \hat{x}_3\approx +1.22.$

- Scale/shift: $y_i = 1\cdot\hat{x}_i + 0 = \hat{x}_i$.

The output $\{-1.22, 0, +1.22\}$ has mean 0 and variance 1 (approx). A learnable $\gamma,\beta$ could then adjust this if needed. This toy example shows how batch normalization centers and rescales values each batch.

## 5. Connections & Prerequisites
BN relies on basic statistics (mean/variance) and how gradients propagate through normalization. A **Prerequisite Refresher**: be comfortable computing mean and variance, and know that standardizing inputs often helps learning. BN is conceptually similar to input normalization, but applied at every hidden layer.

---

# Object Detection

## 1. High-Level Intuition
Object detection is like playing "I spy" with a camera: the system must _find_ (localize) and _name_ (classify) objects in an image. For example, a self-driving car's vision system detects where pedestrians or cars are and what they are. This requires not just saying "there's a car" (classification) but drawing a box around each car (localization).

## 2. Conceptual Deep Dive
An object detector performs **two tasks** for each target object: (1) _Classification_ – assigning a class label (e.g. "dog", "car"), and (2) _Localization/Regression_ – predicting a bounding box around the object. In practice, a model might scan many candidate regions or proposals and output a class score plus a bounding box adjustment for each. Common pipelines generate region proposals (candidate boxes) and then use a CNN to classify each and refine its box. Importantly, detectors often include a "background" or "no object" class so regions with no target become negatives. Unlike plain classification, object detection has no true negatives (we don't train on "not a dog" at every location; we care only about finding positives).

## 3. Mathematical Formulation
Each predicted bounding box is usually represented by parameters $(x,y,w,h)$ or $(x_{\text{min}},y_{\text{min}},x_{\text{max}},y_{\text{max}})$. The network outputs a class probability vector $p$ and a box prediction $t$ for each candidate. Training involves a **multi-task loss**: e.g. a cross-entropy loss for the class label plus a regression loss (often smooth L1) for the box coordinates:

$$L = \frac{1}{N_{\text{cls}}} \sum_i \mathrm{CE}(p_i, p_i^*) + \frac{\lambda}{N_{\text{reg}}} \sum_{i: p_i^*>0} \text{smooth}_{L1}(t_i - t_i^*)$$

where $p_i^*$ and $t_i^*$ are the ground-truth class and box, and $[p^*_i>0]$ means box loss only applies to object (non-background) regions. (Faster R-CNN uses a similar loss in its RPN.) This formula shows classification loss plus a box regression loss for positive samples.

## 4. Worked Toy Example
Suppose an image has one dog and one cat. We generate 2 proposals: Box A overlapping the dog, Box B overlapping the cat. The detector predicts:

- Box A: class "dog" (score 0.9), box coords close to the true dog box.

- Box B: class "cat" (score 0.8), box coords close to true cat box.

Here both are correct detections. If Box B instead was misclassified as "dog", it becomes a **false positive** for dog and a **false negative** for cat (since the true cat wasn't found). In training, Box A's class loss would target "dog"=1 and box loss minimize coordinate error to the dog's ground truth; Box B's class loss would target "cat"=1 but the prediction was wrong, incurring classification loss, and its box loss would target the cat's ground truth. This illustrates how both regression and classification errors contribute to the total loss in object detection.

## 5. Connections & Prerequisites
Object detection builds on image classification and regression knowledge. A **Prerequisite Refresher**: recall how a softmax classifier works for multiple classes, and how regression (like linear regression) predicts continuous values. In detection, these combine: the network has a final softmax or sigmoid for classes and outputs numeric offsets for box coordinates.

---

# Segmentation

## 1. High-Level Intuition
Segmentation is like a digital coloring book: each pixel in an image is "colored" or labeled with its object category (semantic segmentation) or instance ID (instance segmentation). Imagine highlighting every road in a street photo one color and every sidewalk another; that's semantic segmentation. Going further, coloring each **object instance** (e.g. each person) a different color is instance segmentation.

## 2. Conceptual Deep Dive
Segmentation provides much finer detail than bounding boxes. For **semantic segmentation**, the model outputs a label for every pixel, grouping all objects of the same class into one mask. For example, in autonomous driving, labeling which pixels belong to "road," "pedestrian," "car," etc., is crucial for path planning. The lecture notes that simply drawing a big box around the road would erroneously include pedestrians and cars on the road, whereas segmentation identifies exactly which pixels are free space vs. obstacles. **Instance segmentation** goes further: it assigns a unique label to each object instance of a class (so two people are distinguished). The professor described that in instance segmentation "each one of you [in the camera frame] will be a different color," capturing object-level uniqueness. Segmentation models typically use a fully convolutional network and output a mask of the same spatial size as the image.

## 3. Mathematical Formulation
A segmentation model outputs a tensor $S$ of size (classes × height × width), where $S_{c,i,j}$ is the score (logit) for class $c$ at pixel $(i,j)$. A common training loss is the sum of pixel-wise cross-entropies:

$$L = - \frac{1}{HW} \sum_{i,j} \sum_{c} y_{c,i,j} \log \mathrm{softmax}(S_{:,i,j})_c,$$

where $y_{c,i,j}=1$ if pixel $(i,j)$ has class $c$, else 0. In instance segmentation, often a combination of box proposals and mask prediction is used (e.g. Mask R-CNN). For evaluation, segmentation quality can also be measured by IoU on masks (the Jaccard index between predicted and true masks).

## 4. Worked Toy Example
Consider a tiny $3\times3$ image with two classes: 0=background, 1=object. Suppose the ground-truth mask is:

```
1 1 0
0 1 0
0 0 0
```

(Three pixels are object). A model outputs logits; after softmax, suppose predicted probabilities for class 1 are:

```
0.8 0.6 0.1
0.2 0.7 0.3
0.1 0.2 0.0
```

We assign pixels ≥0.5 to object (1). The predicted mask becomes:

```
1 1 0
0 1 0
0 0 0
```

which matches ground truth exactly. We compute pixel-wise accuracy = 100% for this example. If a pixel was incorrect, segmentation IoU could be used: e.g. IoU = (intersection of true/pred masks)/(union of true/pred masks) to quantify overlap. This tiny example shows how each pixel's class is predicted.

## 5. Connections & Prerequisites
Segmentation extends classification to a per-pixel level. A **Prerequisite Refresher**: understand that semantic segmentation is essentially applying a classifier to each pixel. Knowing convolutional feature maps and upsampling (to map CNN outputs back to the image size) is helpful. Segmentation models often build on CNN backbones with deconvolutional or upsampling layers.

---

# Evaluation Metrics (IoU, Precision/Recall)

## 1. High-Level Intuition
Evaluating detection is like grading a test: we want to see how many objects the model got right (true positives) versus how many it missed or got wrong. Intersection-over-Union (IoU) is like how much the predicted "answer" (bounding box) overlaps with the actual answer. Precision and recall measure quality: precision is "of all things we said were there, how many really were?"; recall is "of all real objects, how many did we find?". The precision-recall curve and its area (AP) summarize performance across different confidence thresholds.

## 2. Conceptual Deep Dive
**Intersection over Union (IoU)** quantifies overlap between a predicted box and the ground truth: it is the area of overlap divided by the area of union. Mathematically, for a predicted box $B_p$ and ground-truth box $B_{gt}$:

$$\text{IoU}(B_p,B_{gt}) = \frac{\text{area}(B_p \cap B_{gt})}{\text{area}(B_p \cup B_{gt})},$$

which lies in $[0,1]$. A higher IoU means better alignment. In detection, a common convention is that a detection counts as correct (True Positive) if IoU exceeds a threshold $T$ (e.g. 0.5) and the class is correct; otherwise it is a False Positive. If a ground-truth object has no matching prediction (IoU<T for all), it's a False Negative. (We don't typically count True Negatives in detection.)

From these counts we compute **Precision** and **Recall**:

$$\text{Precision} = \frac{\text{TP}}{\text{TP}+\text{FP}}, \quad \text{Recall} = \frac{\text{TP}}{\text{TP}+\text{FN}}.$$

Varying the detector's confidence threshold yields different (precision, recall) points. Plotting Precision vs. Recall gives the **PR curve**, whose area is the **Average Precision (AP)**. The lecture notes explain that AP is the area under the interpolated precision-recall curve. Finally, **mean Average Precision (mAP)** is the mean of AP across all classes. (Caution: the lecture highlights that using only mAP can obscure performance differences for individual classes.)

## 3. Mathematical Formulation
From the definitions above, we have

$$\text{IoU} = \frac{|B_p \cap B_{gt}|}{|B_p \cup B_{gt}|},$$

and thresholding: if $\text{IoU}\ge T$ we count a detection as a TP. Precision and recall formulas are as given above. The **PR curve** is obtained by plotting precision vs. recall as threshold varies. AP is computed as an integral (or sum) over recall of precision, often via interpolation.

## 4. Worked Toy Example
Imagine there are 2 ground-truth boxes: Box G1 and G2. A detector outputs 3 predicted boxes: P1, P2, P3. Suppose IoUs and predicted classes are:

- P1: IoU=0.7 with G1 (class correct) ⇒ counts as a TP (if threshold $T=0.5$).

- P2: IoU=0.4 with G1 (class correct) ⇒ IoU<T, so this is a False Positive (even if class correct).

- P3: IoU=0.8 with G2 (class incorrect, say predicted "cat" but G2 is "dog") ⇒ for "dog" detector, P3 is a False Positive (wrong class) and G2 is a False Negative.

Here, TP=1 (P1), FP=2 (P2, P3), FN=1 (missed G2). Precision = 1/(1+2)=0.33, Recall = 1/(1+1)=0.5.

If we vary a confidence threshold, some low-score predictions might be dropped. For instance, if P2 was low confidence and excluded, we'd have TP=1, FP=1, FN=1 ⇒ Precision=0.5, Recall=0.5. Plotting all such points gives the PR curve. The area under that curve is the AP. This simple count shows how IoU and classification affect TP/FP counts and thus precision/recall.

## 5. Connections & Prerequisites
Evaluating detection builds on basic classification metrics. A **Prerequisite Refresher**: ensure you understand how binary classification metrics (TP, FP, FN) are defined, and how they extend to detection via IoU. Familiarity with plotting curves and computing area (integral) under a curve is also helpful for interpreting AP.

---

# Faster R-CNN Architecture

## 1. High-Level Intuition
Faster R-CNN is like an assembly line that finds and classifies objects in one integrated pipeline. Think of it as an upgraded search-and-classify system: first it scans an image to propose where objects might be (using learned "detectors"), then it classifies and refines those proposals. Real-world analogy: imagine a security team scanning an area with drones for suspicious regions, then having specialists identify exactly what's in each region.

## 2. Conceptual Deep Dive
Faster R-CNN consists of:

- **Backbone CNN:** Extracts a feature map from the whole image (e.g. ResNet). Importantly, _we run the image through this CNN only once._

- **Region Proposal Network (RPN):** A small CNN that slides over the feature map and, at each location, predicts a set of bounding boxes (anchors) likely to contain objects. For each anchor (of several scales/aspect ratios), the RPN outputs an _objectness_ score (object vs. background) and box adjustments. The lecture explains that at every location the RPN defines $K$ anchors of different sizes/shapes (often $K\approx9$). Anchors with high IoU to a ground-truth are treated as positives, others as negatives.

- **ROI Pooling:** The proposed boxes are then "projected" onto the shared feature map (since the backbone CNN is done). An ROI (Region of Interest) pooling layer crops each proposal's region from the feature map and pools it to a fixed size (e.g. 7×7). This lets the next layers handle each proposal uniformly.

- **Detection Head:** Each pooled ROI is passed through fully-connected layers that output a final class prediction (softmax over $K$ classes + background) and a refined bounding box (regression).

**Shared Computation:** Crucially, the RPN and the detection head **share the bottom convolutional layers**. This means the feature extraction is done only once and used for both proposing and classifying boxes, making it efficient. The lecture notes that after computing convolutional features, Faster R-CNN adds the RPN "to replace" the proposal algorithm with a CNN that shares layers.

## 3. Mathematical Formulation
The core ideas: the RPN has its own loss similar to the detector (classification + bbox regression) applied to anchors. For example, if $p_i$ is the predicted objectness for anchor $i$ and $p_i^*$ its label (1 if positive, 0 if negative), and $t_i$/$t_i^*$ the predicted/true box offsets, the RPN loss is

$$L_{\text{RPN}} = \sum_i \mathrm{CE}(p_i, p_i^*) + \sum_i p_i^* \text{smooth}_{L1}(t_i - t_i^*)$$

This mirrors the two-headed loss described in lecture. After ROI pooling, the detection head similarly uses a softmax loss for class and a regression loss for box. ROI pooling itself can be written as dividing each proposal into a grid (say $H\times W$) of sub-windows and applying max/average pooling in each, but the key is that every ROI yields a fixed-size feature tensor for the FC layers.

## 4. Worked Toy Example
**ROI Pooling Example:** Suppose the CNN feature map for an image is

$$F = \begin{bmatrix}1 & 2 & 3 & 4\\5 & 6 & 7 & 8\\9 & 0 & 1 & 2\\3 & 4 & 5 & 6\end{bmatrix}$$

(shape $4\times4$). A proposal ROI covers the whole map (coordinates [0,0]–[3,3]) and we want to pool it to a $2\times2$ output. We split the $4\times4$ ROI into four $2\times2$ bins:

- Bin1 (top-left 2×2): $\{1,2;5,6\}$, max=6

- Bin2 (top-right 2×2): $\{3,4;7,8\}$, max=8

- Bin3 (bottom-left 2×2): $\{9,0;3,4\}$, max=9

- Bin4 (bottom-right 2×2): $\{1,2;5,6\}$, max=6

ROI pooling outputs the pooled map $\begin{bmatrix}6 & 8\\ 9 & 6\end{bmatrix}$, a fixed size. This shows how any sized region is downsampled to a uniform 2×2 feature map for the next layers, as described in the lecture.

## 5. Connections & Prerequisites
Faster R-CNN combines many ideas: convolutional features, sliding-window detection, and ROI pooling. A **Prerequisite Refresher**: know the concept of sliding-window classifiers and how sharing features (fully convolutional networks) speeds things up. Also recall what a softmax classifier and bounding-box regressor are. Having grasped CNNs and region proposals, Faster R-CNN logically unites them into one end-to-end network.

---

# Key Takeaways & Formulas

- **Convolution:** A convolutional layer computes $O[i,j] = \sum_{u,v} I[i+u,j+v] K[u,v] + b$, typically followed by ReLU. Filters detect local patterns and weight sharing makes CNNs efficient.

- **Batch Normalization:** Normalizes each batch by $\hat{x} = (x - \mu_B)/\sqrt{\sigma_B^2+\epsilon}$, then scales/shifts: $y = \gamma\hat{x}+\beta$. ($\gamma,\beta$ are learned).

- **Intersection over Union (IoU):** $$\mathrm{IoU}(B_p,B_{gt}) = \frac{\mathrm{area}(B_p \cap B_{gt})}{\mathrm{area}(B_p \cup B_{gt})},$$ in $[0,1]$. IoU ≥ threshold (e.g. 0.5) denotes a correct detection.

- **Precision & Recall:** $$\text{Precision} = \frac{TP}{TP+FP}, \quad \text{Recall} = \frac{TP}{TP+FN}.$$ These are plotted in a Precision-Recall curve. The area under this curve is **Average Precision (AP)**. Mean AP (mAP) averages AP over classes.

- **Faster R-CNN Loss:** The RPN and detection heads use a combined loss: classification (cross-entropy) plus box regression (smooth L1/Huber) for positive samples. Anchors are labeled positive/negative based on IoU, and losses are summed over all anchors and all proposals.

These are the fundamental formulas and ideas to recall for the topics covered.