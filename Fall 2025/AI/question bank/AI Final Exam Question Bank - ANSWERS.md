**Topics Covered:** Word2vec | Transformer | MDP Estimation | MDP Optimal Solution

---

## Section 1: Word2vec

**A1.**

- **W (embedding) matrix:** Dimensions V × D = 100,000 × 300
- **W' (lifting) matrix:** Dimensions D × V = 300 × 100,000

**Roles:**

- W matrix: Takes the one-hot encoded center word (V-dimensional) and projects/embeds it into the lower D-dimensional space to produce vector Z
- W' matrix: Takes the embedding Z and "lifts" it back to V-dimensional space to produce logits, which are then passed through softmax to get probability distribution over vocabulary

**A2.**

- **Context words:** "The", "quick", "fox", "jumps" (2 words before and 2 words after "brown")
- **Number of predictions:** 4 separate predictions (one for each context word)

The model jointly optimizes the loss across all context word predictions. The total loss is the sum of cross-entropy losses for predicting each context word.

**A3.** The loss function is **cross-entropy loss**: $$L = -\sum_j Y_j \log(\hat{Y}_j)$$

- **Ŷ (Y-hat):** The V-dimensional posterior probability distribution output by softmax, representing P(context word | center word)
- **Y:** The one-hot encoded ground truth vector for the actual context word

Cross-entropy is appropriate because:

1. We're doing multi-class classification over V classes
2. It measures the difference between predicted probability distribution and true distribution
3. It encourages the model to assign high probability to the correct context words

**A4.** You upload the **W matrix** (embedding matrix), not W'.

**Reason:** The W matrix directly produces the embeddings. When you want to get the embedding for a word, you simply look up the corresponding row in W (equivalent to multiplying W^T by the one-hot vector). The W' matrix was only needed for training (to compute predictions and losses) and is discarded after training.

**A5.** Word2vec produces **context-free** embeddings.

**Context-free:** Each word has a single fixed vector representation regardless of surrounding context. The word "bears" always maps to the same embedding vector.

**Contextual:** The representation of a word changes based on surrounding words. The word "bears" would have different representations in:

- "I love bears" (animals)
- "The team bears the name" (carries/holds)
- "Chicago Bears won" (sports team)

In Word2vec, "bears" would have ONE embedding that tries to capture an average of all these meanings, positioned somewhere in between animal, verb, and sports team clusters in the embedding space.

**A6.** Mathematical forward pass:

1. Center word one-hot vector: x (dimension V × 1)
2. Embedding: Z = W^T · x (dimension D × 1) — effectively picks row from W
3. Logits: Z' = W' · Z (dimension V × 1)
4. Probability distribution: Ŷ = softmax(Z') (dimension V × 1)

The softmax produces: P(word_j | center word) = exp(Z'_j) / Σ_k exp(Z'_k)

**A7.** Tokenizers and models are **jointly developed** and must be compatible:

- The vocabulary used during Word2vec training defines which words have embeddings
- The tokenizer must use the same vocabulary

**Unknown words:** If a word appears during inference that wasn't in training vocabulary:

- It gets assigned a special [UNK] token
- Having many unknown tokens degrades performance
- Modern approaches use subword tokenization (BPE) to handle this — unknown words are broken into known subword pieces

---

## Section 2: Transformer Architecture

**A8.** **Query (Q):** What the token is "looking for" to help determine its contextual meaning

- "bears" in "I love bears" might issue query: "I'm looking for a verb that describes the relationship to me"

**Key (K):** What the token "advertises" about itself that could help others

- "love" would emit key: "I am a verb"

**Value (V):** The actual content/information the token contributes to the weighted sum

- Decoupled from Q and K; represents where the token should move in the embedding space

When Q and K have high dot product (similar directions), the attention weight is high, meaning that key's corresponding value will strongly influence the query token's new representation.

**A9.** Given X: T × D

- **Q = X · W_Q:** T × D (same as input)
- **K = X · W_K:** T × D
- **V = X · W_V:** T × D
- **Score S = Q · K^T:** T × T
- **Attention A = softmax(S/√D):** T × T
- **Output V̂ = A · V:** T × D

**A10.** **Purpose of 1/√D scaling:**

- Without scaling, dot products grow with dimension D (variance increases by factor of D)
- Large dot product values cause softmax to saturate (output becomes nearly one-hot)
- Saturated softmax means only ONE token provides attention (sparse attention)

**With scaling:** Softmax outputs are more distributed, allowing MANY tokens to contribute to the contextual representation. This is desirable for rich contextual embeddings.

**A11.** **Masking** prevents future tokens from influencing current token predictions in decoder architectures.

**Why necessary:**

- During training, all tokens are available (teacher forcing)
- During inference, only past tokens exist
- Masking ensures training matches inference conditions (no "cheating")

**Implementation:**

- Set attention weights for future positions to 0
- OR set input to softmax to very large negative number (−∞) for future positions, causing softmax output ≈ 0

For position t, only positions 1 to t-1 can provide attention.

**A12.** **Need for multi-head attention:**

- Single head captures one type of relationship/pattern
- Multiple heads capture different types of relationships in parallel
- Analogous to CNN filters: different filters detect different spatial patterns; different heads detect different temporal/semantic patterns

**Example:**

- Head 1: Captures subject-verb relationships
- Head 2: Captures adjective-noun relationships
- Head 3: Captures long-range dependencies

**Complexity:** H heads with reduced dimension D/H each maintains same overall computation as single head with full D.

**A13.** **Why needed:** Transformer processes all tokens in parallel with no inherent notion of order. Without positional information, "dog bites man" and "man bites dog" would have identical representations.

**Sinusoidal positional embeddings:** $$PE_{(pos, 2i)} = \sin(pos / 10000^{2i/D})$$ $$PE_{(pos, 2i+1)} = \cos(pos / 10000^{2i/D})$$

- Each position gets a unique D-dimensional vector
- Values bounded between -1 and 1
- Different frequencies allow model to learn relative positions
- Added (not concatenated) to input embeddings

**A14.** **Transformer Block Structure:**

```
Input X
    ↓
Layer Norm → Multi-Head Self-Attention → + (skip connection from X)
    ↓
    Z
    ↓  
Layer Norm → Feed-Forward Network (MLP) → + (skip connection from Z)
    ↓
Output
```

- **Layer Normalization:** Normalizes activations for stable training
- **Skip connections:** Enable gradient flow, allow learning identity mappings
- **MLP:** Introduces nonlinearity (typically two linear layers with ReLU/GELU)

**A15.** Attention matrix size: **T × T = 1000 × 1000 = 1,000,000 elements**

**Bottleneck:**

- Memory grows as O(T²)
- Computation grows as O(T²)
- For T = 100,000, matrix would have 10 billion elements
- This limits practical context lengths

**A16.** **Simple dot-product (X·X^T):**

- Attention determined entirely by context-free embeddings
- Fixed, deterministic mapping
- No learning of what relationships matter

**Learnable attention (Q·K^T = XW_Q · (XW_K)^T):**

- W_Q, W_K, W_V are trainable parameters
- Model learns WHAT to look for (queries) and WHAT to advertise (keys)
- Can learn task-specific attention patterns
- Different heads can specialize in different relationship types

**A17.** **Two properties of attention weights:**

1. **Non-negative:** A_ij ≥ 0
2. **Sum to one:** Σ_j A_ij = 1 (for each row)

**Why softmax:**

- Naturally produces valid probability distribution
- Differentiable for gradient-based learning
- Creates competition among tokens (higher scores suppress others)
- Allows interpretation as "attention probabilities"

---

## Section 3: MDP Estimation (Prediction Problem)

**A18.** **State value function:** $$V^\pi(s) = \mathbb{E}_\pi[G_t | S_t = s]$$

Where the return G_t is:
$$G_t = R_{t+1} + \gamma R_{t+2} + \gamma^2 R_{t+3} + ... = \sum_{k=0}^{\infty} \gamma^k R_{t+k+1}$$

It represents the expected cumulative discounted reward starting from state s and following policy π thereafter.

**A19.** $$v(s) = \sum_a \pi(a|s) \sum_{s'} p(s'|s,a) [r(s,a,s') + \gamma v(s')]$$

**Components:**

- **π(a|s):** Policy — probability of taking action a in state s
- **p(s'|s,a):** Transition model — probability of reaching s' given state s and action a
- **r(s,a,s'):** Immediate reward for transition
- **γ:** Discount factor (0 < γ < 1)
- **v(s'):** Value of successor state (bootstrapping)

**Interpretation:** Value of s = weighted average over actions of [immediate reward + discounted future value]

**A20.** Using Bellman equation with γ = 0.9:

**V(S1):** From S1, go to S2 with reward 2 $$V(S1) = 2 + 0.9 \cdot V(S2)$$

**V(S2):** From S2, go to S1 with reward 0 $$V(S2) = 0 + 0.9 \cdot V(S1)$$

**System of equations:** $$V(S1) = 2 + 0.9 \cdot V(S2)$$ $$V(S2) = 0.9 \cdot V(S1)$$

**Solution:** Substituting: V(S1) = 2 + 0.9(0.9 · V(S1)) = 2 + 0.81 · V(S1) V(S1)(1 - 0.81) = 2 V(S1) = 2/0.19 ≈ **10.53** V(S2) = 0.9 × 10.53 ≈ **9.47**

**A21.** **State A value < 10:**

- After teleporting to A' with reward 10, the agent follows random policy
- From A', agent may move into walls (reward -1) or wander
- The γv(s') term for successor states is NEGATIVE on average
- Value = 10 + γ(negative expected future) < 10

**State B value > 5:**

- After teleporting to B' with reward 5, agent is in a favorable position
- B' has positive expected future value (close to high-value regions)
- Value = 5 + γ(positive expected future) > 5

**A22.** **Matrix form:** V = R + γPV

**Rearranging:** $$V - \gamma PV = R$$ $$(I - \gamma P)V = R$$ $$V = (I - \gamma P)^{-1}R$$

**Problems with direct solution:**

1. **Computational complexity:** Matrix inversion is O(|S|³)
2. **Memory:** Storing full matrix requires O(|S|²) space
3. **Numerical instability:** Can blow up for ill-conditioned matrices
4. **Not scalable:** Real problems have millions of states

**A23.** **Iterative Policy Evaluation:** $$V_{k+1}(s) = \sum_a \pi(a|s) \sum_{s'} p(s'|s,a)[r(s,a,s') + \gamma V_k(s')]$$

**Contraction property:** The Bellman operator T is a γ-contraction: $$||TV_1 - TV_2||_\infty \leq \gamma ||V_1 - V_2||_\infty$$

Since γ < 1, repeated application converges to unique fixed point V^π:

- Like the scalar iteration x_{k+1} = γx_k + c converging to c/(1-γ)
- Convergence guaranteed regardless of initialization

**A24.** **Number of deterministic policies:** |Π| = |A|^|S|

**Counting states:**

- Total cells: 3 × 4 = 12
- Terminal states: 2 (cells (1,4) and (2,4))
- Wall: 1 (cell (2,2))
- Non-terminal states: |S| = 12 - 2 - 1 = **9**

**Actions:** |A| = 4 (north, south, east, west)

**Number of policies:** 4^9 = **262,144**

**A25.** **Transition model P(s'|s,a):** Probability of next state given current state and action

- Marginalized over rewards
- Used for state transitions only

**MDP dynamics P(s',r|s,a):** Joint probability of next state AND reward

- Full specification of environment
- More detailed than transition model alone

**Reward model computation:** $$R(s,a) = \mathbb{E}[R_t | S_t=s, A_t=a] = \sum_{s'}\sum_r r \cdot P(s',r|s,a)$$

---

## Section 4: MDP Optimal Solution (Control Problem)

**A26.** **Bellman Optimality Equation:** 
$$
V^*(s) = \max_a \sum_{s'} p(s' \mid s,a)\left[r(s,a,s') + \gamma V^*(s')\right]
$$

**Why nonlinear:** The **max** operator makes this nonlinear (max is not a linear operation).

**Difference from expectation equation:**

- Expectation equation: averages over actions according to policy π(a|s)
- Optimality equation: takes maximum over actions
- Cannot solve optimality equation with simple matrix inversion

**A27.** **State-action value function:** 
$$Q^\pi(s,a) = \mathbb{E}_\pi[G_t | S_t=s, A_t=a]$$

Expected return starting from state s, taking action a, then following policy π.

**Relationship to V:** 
$$
V^*(s) = \max_a Q^*(s,a)
$$

Q function is more useful for control because it directly tells you which action is best.

**A28.** **Policy Iteration Algorithm:**

**Step 1 - Policy Evaluation:** Given current policy π_k, compute V^{π_k} by solving Bellman expectation equation (iteratively until convergence)

**Step 2 - Policy Improvement:** Create new policy π_{k+1} by acting greedily with respect to V^{π_k}: $$\pi_{k+1}(s) = \arg\max_a \sum_{s'} p(s'|s,a)[r(s,a,s') + \gamma V^{π_k}(s')]$$

**Repeat** until policy no longer changes (convergence to π*)

**Guarantees:** Each iteration produces a policy at least as good as previous; terminates at optimal policy.

**A29.** **Greedy Policy Improvement:**

Given value function V(s), for each state s:

1. Compute Q(s,a) for all actions: Q(s,a) = Σ_{s'} p(s'|s,a)[r + γV(s')]
2. Select action with highest Q-value: π'(s) = argmax_a Q(s,a)

**Why it improves:** Policy improvement theorem guarantees that greedy policy w.r.t. V^π is at least as good as π. If strictly better for any state, overall policy is strictly better.

**A30.** __Bellman Optimality for V_(high):_* 
$$
V^*(\text{high})
= \max\{Q^*(\text{high}, \text{search}),
        Q^*(\text{high}, \text{wait}),
        Q^*(\text{high}, \text{recharge})\}
$$


Where each Q* is: 
$$
Q^*(\text{high}, \text{search})
= \sum_{s'} p(s' \mid \text{high}, \text{search})
\left[r(\text{high}, \text{search}, s') + \gamma V^*(s')\right]
$$

The optimal value equals the maximum expected return over all possible actions from the high battery state.

**A31.** __Why π_ converges before V_:**

The optimal policy depends on **relative** values between states, not absolute values.

Example: If V(s1) = 10 and V(s2) = 8, we prefer s1. After more iterations: V(s1) = 15 and V(s2) = 12, we STILL prefer s1.

The ranking/ordering of state values stabilizes before the actual numbers converge. Since greedy policy only cares about which state is better (not by how much), policy converges first.

**A32.** **Why γ reduction works:**

- Smaller γ means less weight on distant future rewards
- Fewer effective time steps to consider
- Value function "horizon" is approximately 1/(1-γ)

**Selection strategy:**

- γ = 0.9 → horizon ≈ 10 steps
- γ = 0.5 → horizon ≈ 2 steps
- Smaller γ = faster convergence

**Trade-off:**

- Smaller γ: faster computation but more myopic policy (ignores long-term consequences)
- Larger γ: better long-term planning but slower convergence
- Must balance computational savings against policy quality

**A33.** **Multiple actions with equal Q-values:**

This means the optimal policy is **not unique** — there are multiple optimal policies.

In gridworld, this often happens:

- Near goal states where multiple paths are equally good
- In symmetric situations
- Any action achieving the maximum Q-value is optimal

The agent can break ties arbitrarily or randomly; all such policies are equally optimal.

---

## Section 5: Reinforcement Learning (Monte Carlo & TD Methods)

**A34.** **Key difference:**

**MDP (Dynamic Programming):**

- Full knowledge of transition model P(s'|s,a)
- Full knowledge of reward function R(s,a)
- Can compute exact expectations
- "Planning" with known model

**Reinforcement Learning:**

- NO knowledge of P or R
- Must learn from experience/interaction
- Environment is a "black box"
- Sample-based estimation

**A35.** **Monte Carlo Update:** $$V(S_t) \leftarrow V(S_t) + \alpha[G_t - V(S_t)]$$

**G_t (sample return):** Actual cumulative discounted reward observed from time t until episode termination: $$G_t = R_{t+1} + \gamma R_{t+2} + \gamma^2 R_{t+3} + ... + \gamma^{T-t-1}R_T$$

**Why wait for termination:** Cannot compute G_t until episode ends because we need all future rewards. This is Monte Carlo's main limitation — cannot learn online during episode.

**A36.** **Connection to Kalman filter:**

Incremental mean update: μ_k = μ_{k-1} + (1/k)(x_k - μ_{k-1})

The (1/k) factor is analogous to **Kalman gain** — it determines how much to trust new information vs. prior estimate.

**Role of α:**

- α = 1/N(s): Exact sample mean (stationary environment)
- Fixed α: Weighted average favoring recent samples (non-stationary)
- Larger α: Faster adaptation, more noise
- Smaller α: Slower adaptation, more stable

**A37.** **TD(0) Update:** $$V(S_t) \leftarrow V(S_t) + \alpha[R_{t+1} + \gamma V(S_{t+1}) - V(S_t)]$$

**TD Target:** R_{t+1} + γV(S_{t+1})

- One-step estimate of return
- Uses actual reward R_{t+1} (sampling)
- Bootstraps from estimated V(S_{t+1}) (DP-like)

**Combination:**

- From **MC:** Sampling trajectories from unknown environment
- From **DP:** Bootstrapping from value estimates instead of waiting for actual returns
- Result: Can update immediately after each transition (online learning)

**A38.** **MC Limitation:** Must wait for episode termination

- Cannot learn during episode
- Cannot handle continuing (non-episodic) tasks
- Long episodes = long wait for updates

**When to prefer TD:**

- Continuing tasks with no natural termination
- When immediate learning is important
- Long episodes where waiting is impractical
- When bootstrapping improves sample efficiency

**When MC might be better:**

- When environment is highly stochastic (less bootstrap bias)
- When episodes are short
- When unbiased estimates are crucial

**A39.** **TD(λ) as weighted combination:** $$G_t^\lambda = (1-\lambda)\sum_{n=1}^{\infty} \lambda^{n-1} G_t^{(n)}$$

Where G_t^{(n)} is the n-step return.

**Special cases:**

- **λ = 0:** Only G_t^{(1)} matters → **TD(0)** (one-step bootstrap)
- **λ = 1:** All weight on G_t^{(∞)} → **Monte Carlo** (full return)

Intermediate λ values blend short and long-term estimates.

**A40.** **Backup Diagrams:**

**Dynamic Programming (full backup):**

```
        s
      / | \
     a1 a2 a3   (all actions)
    /|\ ...
   s' s' s'     (all successor states)
```

Width: ALL actions and ALL successors Depth: One step

**Monte Carlo (sample to termination):**

```
    s
    |
    a  (sampled action)
    |
    s' (sampled state)
    |
    a
    |
    ...
    |
  terminal
```

Width: ONE sampled path Depth: Until termination

**TD(0) (one-step sample):**

```
    s
    |
    a  (sampled action)
    |
    s' (sampled state, bootstrap here)
```

Width: ONE sampled transition Depth: One step, then bootstrap

---

## Section 6: Mixed/Application Questions (Exam-Style)

**A41.** **(a) Architecture:**

```
Text descriptions ──→ [Text Encoder (Transformer)] ──→ Text embeddings (T₁, T₂, T₃)
                                                              ↓
                                                        Dot products
                                                              ↑
Images ──→ [Image Encoder (ResNet/ViT)] ──→ [Projection] ──→ Image embeddings (I₁, I₂, I₃)
```

Key components:

- Text encoder: Transformer to encode descriptions
- Image encoder: ResNet/CNN OR Vision Transformer (ViT)
- Projection layer: Projects image features to same dimension as text features
- Similarity: Computed via dot product between text and image embeddings

**(b) Loss function:**

**L+ (Positive component):** Encourages matching image-text pairs to be similar

- Maximizes dot product T_i · I_i for correct pairs
- Pulls matching pairs together in embedding space

**L- (Negative component):** Encourages non-matching pairs to be dissimilar

- Minimizes dot product T_i · I_j for i ≠ j
- Pushes non-matching pairs apart

**Total loss:** L = L+ + L-

Similarity computed as: sim(T, I) = T · I (dot product)

**A42.** **Vocabulary size (32K → 128K):**

- Larger vocabulary = more tokens represented without [UNK]
- Better handling of rare words, multilingual text, code
- Trade-off: Larger embedding matrix, but better coverage

**Context length (4K → 8K tokens):**

- Longer context = can process longer documents
- Better for tasks requiring long-range dependencies
- Enables better in-context learning with more examples
- Trade-off: O(T²) attention cost increases

**Training tokens (2T → 15T):**

- More training data = better generalization
- Sees more diverse examples and patterns
- Reduces overfitting, improves factual knowledge
- Interdependency: More parameters (8B vs 7B) can utilize more data without memorizing

**A43.** **Role of head:**

- Final linear layer followed by softmax
- Projects transformer body output to vocabulary dimension
- Produces probability distribution over next token

**Softmax dimensions:** Operates over **V dimensions** (vocabulary size, e.g., 100,000+)

**Relationship:**

- Output dimension = vocabulary size
- Each position in softmax output = probability of that token being next
- Highest probability token (or sampled token) is the prediction

**A44.** **Four stages:**

1. **Pre-training:** Next-token prediction on trillions of tokens. Produces base model that can generate text but doesn't follow instructions well.
    
2. **Supervised Fine-tuning (SFT):** Train on instruction-response pairs. Model learns to follow instructions and answer questions.
    
3. **Preference Fine-tuning (RLHF):** Use human preferences to train reward model. Fine-tune with RL (PPO) so model produces preferred responses. This is where ChatGPT-like behavior emerges.
    
4. **Reasoning Fine-tuning:** Train on reasoning benchmarks (e.g., GSM8K math problems). RL rewards correct answers. Improves chain-of-thought and problem-solving.
    

**RL Role:** In stages 3 & 4, RL optimizes the policy (model) to maximize rewards (human preferences or correctness), going beyond what supervised learning can achieve.

**A45.** **Effect of α on convergence:**

**Large α (e.g., 0.4):**

- Fast initial convergence
- More responsive to recent samples
- Higher variance, may oscillate around true value
- Risk of not converging if too large

**Small α (e.g., 0.01):**

- Slow convergence
- More stable, less noise
- May take many episodes to reach true value
- Safer but less efficient

**Trade-off:**

- Need to balance speed vs. stability
- In practice, often start with larger α and decay over time
- Optimal α depends on environment stochasticity

---

## Section 7: True/False & Multiple Choice

**A46.** Correct answers: **A and C**

- A. TRUE — W' has dimensions D × V (lifts from D back to V)
- B. FALSE — Skip-gram predicts CONTEXT words given CENTER word (CBOW does opposite)
- C. TRUE — Softmax outputs V-dimensional probability distribution
- D. FALSE — Word2vec produces context-FREE embeddings

**A47.** Correct answers: **A, B, and D**

- A. TRUE — Attention matrix is T × T (each token attends to all tokens)
- B. TRUE — Without positional info, transformer treats input as a set
- C. FALSE — Masking is needed during BOTH training and inference for decoder
- D. TRUE — Scaling prevents softmax from saturating to one-hot

**A48.** Correct answers: **B and D**

- A. FALSE — Bellman EXPECTATION equation is linear; OPTIMALITY equation has max
- B. TRUE — This is the essence of the recursive definition
- C. FALSE — γ = 1 can cause divergence (infinite returns)
- D. TRUE — V = (I - γP)^(-1)R works for small state spaces

**A49.** Correct answers: **A, B, and D**

- A. TRUE — MC needs G_t which requires full episode
- B. TRUE — TD target uses estimated V(S_{t+1})
- C. FALSE — MC has HIGHER variance (uses full noisy return); TD has bias but lower variance
- D. TRUE — TD updates after each step

**A50.** Correct answers: **B and C**

- A. FALSE — Optimal policy can have ties (multiple equally good actions)
- B. TRUE — Value iteration is an alternative algorithm
- C. TRUE — Finite state/action space guarantees finite convergence
- D. FALSE — Policy improvement theorem guarantees policy never gets worse