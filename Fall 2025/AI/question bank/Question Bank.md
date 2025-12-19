---

## **QUESTION BANK**

---
### **TOPIC 1: WORD2VEC**

**Q1.1** What are context-free embeddings, and why are they called "context-free"?

**Q1.2** Explain the core idea behind Word2Vec embeddings inspired by the linguist Firth's distributional semantics hypothesis.

**Q1.3** In the Word2Vec architecture, you have two matrices W and W'. After training, which matrix is used to generate embeddings, and why don't we need to perform matrix multiplication to retrieve an embedding for a specific word?

**Q1.4** The word "bank" can mean a financial institution or a riverbank. Explain why Word2Vec embeddings cannot distinguish between these two meanings. What is the fundamental limitation this illustrates?

**Q1.5** What is the relationship between the vocabulary size V and the embedding dimension D in the W matrix used for Word2Vec? What are the dimensions of this matrix?

**Q1.6** If a word is not present in the vocabulary during inference (unknown word), what happens in the language model, and why is having many unknown tokens problematic?

**Q1.7** Describe the training objective of Word2Vec. What prediction task is the neural network solving, and what loss function is typically used?

---

### **TOPIC 2: TRANSFORMER**

**Q2.1** The transformer architecture eliminates recurrent connections found in RNNs. What new challenge does this create, and how is it addressed?

**Q2.2** In the simple self-attention mechanism, explain the role of the score matrix S = XX^T and how it relates to creating contextual embeddings.

**Q2.3** Explain why softmax is applied to the score matrix row-by-row, and what properties does this give to the attention weights?

**Q2.4** What is the purpose of the scaling factor 1/√D in the attention mechanism? Explain what happens to the softmax output if this scaling is not applied.

**Q2.5** In the learnable attention mechanism, tokens emit three vectors: Query (Q), Key (K), and Value (V). Explain the role of each using an intuitive example.

**Q2.6** Given three sentences: "I love bears," "Bears the pain," and "Bears won the game," explain how the Query and Key vectors help the word "bears" receive help from different context words in each sentence.

**Q2.7** Write the mathematical expression for single-head self-attention output V̂ in terms of Q, K, V matrices. Include all operations from input X to output V̂.

**Q2.8** What is multi-head self-attention, and why is it beneficial? How does it relate to the concept of multiple filters in CNNs?

**Q2.9** Explain why masking is required in decoder-based transformer architectures during training.

**Q2.10** What are the two methods for positional embeddings discussed in class? Why is addition preferred over concatenation when combining positional embeddings with context-free embeddings?

**Q2.11** What is the role of layer normalization and skip connections in the transformer architecture?

**Q2.12** Explain the difference between context-free embeddings (from Word2Vec) and contextual embeddings (from the transformer attention mechanism).

---

### **TOPIC 3: MDP ESTIMATION**

**Q3.1** Define an MDP formally. What are the five components (S, A, P, R, γ) that specify an MDP?

**Q3.2** Write the Bellman expectation equation for the state value function v_π(s). Explain each term in the equation.

**Q3.3** In a gridworld where an agent operates under a uniform random policy (π(a|s) = 0.25 for all actions), state A teleports the agent to A' with reward 10, and state B teleports to B' with reward 5. Using the Bellman equation, explain conceptually why v(A) < 10 but v(B) > 5.

**Q3.4** What is the difference between the prediction problem and the control problem in MDPs?

**Q3.5** Explain why the Bellman expectation equation results in a system of linear equations. How can this system be solved?

**Q3.6** In a 3×4 gridworld with 2 terminal states and 1 wall (impassable), what is the number of possible deterministic policies? Show your calculation.

**Q3.7** What is the state-action value function Q_π(s,a)? How does it differ from the state value function V_π(s)?

**Q3.8** Explain the concept of "bootstrapping" in the context of value function estimation.

**Q3.9** What is the iterative policy evaluation algorithm? Write the update equation and explain why convergence is guaranteed.

**Q3.10** Explain the effect of the discount factor γ on the value function. What happens when γ is close to 0 versus close to 1?

**Q3.11** You are asked to reduce computational complexity of solving the MDP by selecting the discount factor. Explain why this is a good idea and how you would select the value.

---

### **TOPIC 4: MDP OPTIMAL SOLUTION**

**Q4.1** Write the Bellman optimality equation for V*(s). Why is this equation nonlinear, unlike the Bellman expectation equation?

**Q4.2** What is the difference between the Bellman expectation equation and the Bellman optimality equation?

**Q4.3** Describe the policy iteration algorithm. What are its two main steps, and how do they interact?

**Q4.4** In policy iteration, why might the optimal policy π* converge before the value function V* fully converges?

**Q4.5** Consider a simple MDP with two states S1 and S2. From S1, the action leads to S2 with reward 2. From S2, the action leads to S1 with reward 0. With γ = 0.8, set up and solve the system of equations for V(S1) and V(S2).

**Q4.6** Explain the "greedy" approach to policy improvement. Given a value function, how do you determine the improved policy?

**Q4.7** What is the contraction property, and why is it important for the convergence of policy iteration?

**Q4.8** In the recycling robot example with states {High, Low} and actions {Search, Wait, Recharge}, write the Bellman optimality equation for V*(High). (You don't need exact probabilities, but show the structure.)

**Q4.9** Compare and contrast policy iteration and value iteration algorithms.

**Q4.10** Given a grid world with initial uniform random policy, describe the process of one complete iteration of policy iteration (policy evaluation followed by policy improvement).

---

## **ANSWERS**

---

### **TOPIC 1: WORD2VEC ANSWERS**

**A1.1** Context-free embeddings are vector representations of words/tokens that do not change based on surrounding context. They are called "context-free" because the same word always maps to the same vector regardless of what other tokens are adjacent to it. For example, "bank" always has the same embedding whether it appears in "river bank" or "bank account."

**A1.2** Firth's distributional semantics hypothesis states: "A word's meaning is given by words that frequently appear close by." Word2Vec operationalizes this by training a neural network to predict context words given a center word (or vice versa). Words appearing in similar contexts end up with similar embeddings, placing semantically related words close together in the embedding space.

**A1.3** The W* (W-star) matrix is used to generate embeddings. It has dimensions V × D (vocabulary size × embedding dimension). We don't need matrix multiplication because the input is one-hot encoded. Multiplying a one-hot vector by W* simply selects the row corresponding to the word's index. So for word "bank" with index 359, we just pick up the 359th row of W*.

**A1.4** Word2Vec produces one embedding per word type in the vocabulary, not per word instance. During training, the representation averages across all contexts where the word appears. If "bank" appears in both financial and geographical contexts, the resulting embedding is an average that doesn't distinguish meanings. This is why they are called context-free—the embedding cannot adapt to the specific sentence context.

**A1.5** The W matrix has dimensions V × D, where V is the vocabulary size and D is the embedding dimension (typically hundreds). Each row corresponds to one word's embedding vector.

**A1.6** Unknown words are assigned a special [UNK] token. Having many unknown tokens is problematic because: (1) the model receives no semantic information about the actual word, (2) it degrades prediction performance, and (3) tokenizers and models are jointly developed to minimize unknowns—if there's a mismatch, performance suffers significantly.

**A1.7** Word2Vec solves a prediction task: given a center word, predict the surrounding context words (Skip-gram), or given context words, predict the center word (CBOW). The loss function is the composition of cross-entropy losses for predicting each context word. The network minimizes this loss through backpropagation, and the W matrix that results is used for embeddings.

---

### **TOPIC 2: TRANSFORMER ANSWERS**

**A2.1** Eliminating recurrent connections means the transformer processes all tokens in parallel, which creates the challenge that positional information is lost (the architecture becomes permutation invariant). This is addressed through positional embeddings—vectors added to the context-free embeddings that encode each token's position in the sequence.

**A2.2** The score matrix S = XX^T computes dot products between all pairs of tokens. Each element S_ij measures the similarity between tokens i and j. High scores indicate tokens that should influence each other more. This allows us to determine which tokens should help "move" each other in the embedding space to create contextual representations.

**A2.3** Softmax is applied row-by-row to create attention weights A where: (1) A_ij ≥ 0 (non-negative weights), and (2) Σ_j A_ij = 1 (weights sum to 1 for each row). This creates a normalized weighted combination where attention weights represent how much each context token contributes to updating the current token's representation—like a competition among tokens to contribute information.

**A2.4** The scaling 1/√D prevents the dot products from becoming too large when D is large. Without scaling, large dot products cause softmax to produce very sparse (nearly one-hot) outputs, meaning only one or two tokens dominate the attention. With proper scaling, the softmax produces smoother distributions allowing many tokens to contribute information. Mathematically, dot products have variance proportional to D, so dividing by √D normalizes the variance back to 1.

**A2.5**

- **Query (Q)**: What am I looking for? What kind of help do I need? (e.g., "I'm looking for a verb")
- **Key (K)**: What am I? What can I offer? (e.g., "I am a verb")
- **Value (V)**: What meaning do I carry to transfer? This is the starting representation that gets moved to a new location.

Analogy: You enter a professors' cafeteria (Query: "I need help with math"). Professors raise their hands based on expertise (Keys: "I teach calculus"). Those who match your query help you, and your knowledge (Value) is updated to a new state (V̂).

**A2.6**

- "I love bears" → bears is an **object**. Query asks: "I need help from a verb." The verb "love" has a Key saying "I am a verb," so it attends strongly.
- "Bears the pain" → bears is a **verb**. Query asks: "I need help from an object/noun." "Pain" responds with its Key.
- "Bears won the game" → bears is a **subject** (sports team). Query asks: "I need help from a verb or object." Both "won" and "game" may attend.

**A2.7**

1. Q = XW_Q (queries), K = XW_K (keys), V = XW_V (values)
2. S = QK^T (score matrix, T×T)
3. A = softmax(S/√D) (attention weights, T×T)
4. V̂ = AV (output contextual embeddings, T×D)

Or in one equation: V̂ = softmax((XW_Q)(XW_K)^T / √D) × (XW_V)

**A2.8** Multi-head self-attention uses H parallel attention mechanisms (heads), each with its own W_Q, W_K, W_V matrices. The outputs are concatenated/linearly combined. Benefits: (1) Different heads can capture different patterns (grammatical, semantic, syntactic), (2) Similar to how CNN filters extract different spatial features, heads extract different contextual relationships, (3) Architectures typically use 8-32+ heads.

**A2.9** Masking is required because during inference, we only have access to previously generated tokens, not future ones. To avoid "cheating" during training (where all tokens are available), we mask future positions so that when computing attention for position i, only positions 1 to i-1 can contribute. This ensures training matches inference conditions.

**A2.10** Two methods: (1) **Fourier/sinusoidal method**: Uses sin/cos functions at different frequencies to create position-dependent vectors, (2) **Learnable embeddings**: Position vectors are learned during training. Addition is preferred over concatenation because: (1) it doesn't increase the D dimension (which affects computational complexity), (2) linear operations preserve the addition (can separate components mathematically), (3) the positional vectors are designed to not disrupt the original context-free vectors significantly.

**A2.11** **Layer normalization**: Normalizes activations across features for each token independently (unlike batch norm). Helps stabilize training, sometimes called RMS normalization. **Skip connections**: Add the input directly to the output (residual connections). Ensures gradient flow during backpropagation (prevents vanishing gradients), similar to ResNets.

**A2.12** **Context-free embeddings** (Word2Vec): Same vector for same word regardless of context. Determined externally to transformer. "Bank" → same vector always. **Contextual embeddings** (Transformer output): Vector depends on surrounding tokens. Determined by attention mechanism during forward pass. "Bank" → different vectors in "river bank" vs "bank account."

---

### **TOPIC 3: MDP ESTIMATION ANSWERS**

**A3.1** An MDP is defined by the 5-tuple (S, A, P, R, γ):

- **S**: Set of states (calligraphic S)
- **A**: Set of actions
- **P**: Transition model p(s'|s,a) - probability of reaching s' given state s and action a
- **R**: Reward function r(s,a,s') - reward received for transition
- **γ**: Discount factor (0 ≤ γ < 1) - weights future rewards

**A3.2** v_π(s) = Σ_a π(a|s) Σ_{s'} p(s'|s,a) [r(s,a,s') + γv_π(s')]

- π(a|s): Policy probability of action a in state s
- p(s'|s,a): Transition probability to s' given s,a
- r(s,a,s'): Immediate reward
- γ: Discount factor
- v_π(s'): Value of next state (bootstrapping)

This equation says: the value of state s equals the expected immediate reward plus discounted expected future value, averaged over all actions (per policy) and all possible next states.

**A3.3** For state A (reward 10 → teleport to A'):

- v(A) < 10 because after teleporting to A', the agent follows a random policy and can hit walls (reward -1) with some probability. The negative future values from A' are added with weight γ, reducing total value below 10.

For state B (reward 5 → teleport to B'):

- v(B) > 5 because B' is in a better location (higher value region of the grid). The positive future values from B' are added with weight γ, increasing total value above 5.

**A3.4**

- **Prediction problem**: Given a fixed policy π, evaluate/compute the value function V_π(s) for all states. "How good is this policy?"
- **Control problem**: Find the optimal policy π* that maximizes value. Involves both evaluation and policy improvement.

**A3.5** The Bellman equation is linear because it expresses v(s) as a linear combination of v(s') values (no max operators). For |S| states, we get |S| linear equations with |S| unknowns: v = R + γPv, which can be rewritten as (I - γP)v = R. This can be solved by matrix inversion: v = (I - γP)^{-1}R, or iteratively.

**A3.6**

- Total cells: 3 × 4 = 12
- Terminal states: 2 (no decisions made there)
- Wall: 1 (impassable)
- Non-terminal states: |S| = 12 - 2 - 1 = 9
- Actions per state: |A| = 4

Number of deterministic policies: |Π| = |A|^{|S|} = 4^9 = 262,144

**A3.7** Q_π(s,a) = E_π[G_t | S_t=s, A_t=a]

It's the expected return starting from state s, taking action a, then following policy π. Unlike V_π(s) which averages over actions, Q_π(s,a) commits to a specific first action. Q is better for acting optimally because we can directly compare actions: π(s) = argmax_a Q(s,a).

**A3.8** Bootstrapping means estimating a value using estimates of other values rather than waiting for actual returns. In the Bellman equation, we estimate v(s) using v(s') of successor states. This allows updates before episode completion (unlike Monte Carlo which needs complete returns).

**A3.9** Iterative policy evaluation update: V_{k+1}(s) = Σ_a π(a|s) Σ_{s'} p(s'|s,a)[r(s,a,s') + γV_k(s')]

Start with V_0 = 0 for all states. Convergence is guaranteed because the Bellman operator is a contraction (similar to x_{k+1} = γx_k + c converging when γ < 1). Each iteration reduces the distance to the true value function.

**A3.10**

- **γ close to 0**: Agent is "myopic," valuing only immediate rewards. Return ≈ R_{t+1}. Short-term planning.
- **γ close to 1**: Agent values long-term rewards. Full future trajectory matters. Long-term planning.

Interpretation: Like net present value of money—$1 today is worth more than $1 tomorrow due to uncertainty. γ discounts future rewards because they're uncertain.

**A3.11** A smaller γ reduces computational complexity because:

- It shortens the effective planning horizon
- Bellman updates require fewer steps of value propagation across states
- The backup tree depth is limited
- Expected return converges faster

Trade-off: Smaller γ may miss important long-term rewards. For applications like games with long horizons, larger γ is needed despite higher computation.

---

### **TOPIC 4: MDP OPTIMAL SOLUTION ANSWERS**

**A4.1** V*(s) = max_a Σ_{s'} p(s'|s,a)[r(s,a,s') + γV*(s')]

This is nonlinear because of the **max** operator over actions. The expectation equation uses Σ_a π(a|s)... (weighted average), which is linear. The max operator makes it impossible to solve with simple matrix inversion—requires iterative methods.

**A4.2**

- **Bellman expectation equation**: Evaluates a given policy π. Uses weighted average over actions per policy π(a|s). Linear equation.
- **Bellman optimality equation**: Finds the optimal policy. Uses max over actions. Nonlinear equation.

The expectation equation asks "what's the value of following π?" The optimality equation asks "what's the value of acting optimally?"

**A4.3** Policy iteration has two steps:

1. **Policy Evaluation**: Given current policy π_k, compute V_{π_k}(s) for all states using Bellman expectation equation (iterate until convergence)
2. **Policy Improvement**: Create new policy π_{k+1}(s) = argmax_a Q_{π_k}(s,a) (act greedily with respect to current value function)

Repeat until policy stops changing. Guaranteed to converge to π*.

**A4.4** The optimal policy can converge before V* because what matters for policy decisions are the **relative** values between adjacent states, not absolute values. Once relative ordering stabilizes (which state is better than which), the greedy policy is optimal even if exact values are still being refined.

**A4.5** Bellman equations with γ = 0.8:

- V(S1) = r_{S1→S2} + γV(S2) = 2 + 0.8V(S2)
- V(S2) = r_{S2→S1} + γV(S1) = 0 + 0.8V(S1)

Substitute: V(S1) = 2 + 0.8(0.8V(S1)) = 2 + 0.64V(S1) 0.36V(S1) = 2 → V(S1) = 2/0.36 ≈ 5.56

V(S2) = 0.8 × 5.56 ≈ 4.44

**A4.6** Greedy policy improvement: For each state s, select the action that maximizes expected value: π'(s) = argmax_a Σ_{s'} p(s'|s,a)[r(s,a,s') + γV(s')]

This means: look at all possible actions, compute expected value for each, pick the best one. This is guaranteed to improve or maintain policy quality.

**A4.7** The contraction property states that applying the Bellman operator reduces the distance between any value function and the true value: ||TV - V*|| ≤ γ||V - V*||

Since γ < 1, repeated application contracts the distance, guaranteeing convergence. Like the scalar example x_{k+1} = γx_k + c converging to c/(1-γ).

**A4.8** V*(High) = max{ [p(H|H,search)·(r + γV*(H)) + p(L|H,search)·(r' + γV*(L))], ← search [p(H|H,wait)·(r'' + γV*(H))] ← wait }

The structure shows: for each action, compute expected reward + discounted future value, then take max over actions.

**A4.9**

|Aspect|Policy Iteration|Value Iteration|
|---|---|---|
|Evaluation|Full policy evaluation each iteration|Single sweep of value updates|
|Improvement|Explicit policy improvement step|Implicit (max in update)|
|Convergence|Fewer iterations, each more expensive|More iterations, each cheaper|
|Policy|Explicit π at each step|Policy extracted at end|

Both guaranteed to converge to V* and π*.

**A4.10** Example iteration:

1. **Start**: Uniform random policy (π(a|s) = 0.25), V_0 = 0 everywhere
2. **Policy Evaluation**: Apply Bellman equation iteratively until V_π converges to stable values for each grid cell
3. **Policy Improvement**: At each cell, choose action leading to highest-value neighbor:
    - If V(up) > V(down), V(left), V(right) → policy picks "up"
    - Eliminate suboptimal actions
4. **Result**: New policy with arrows pointing toward higher-value states
5. **Repeat**: Evaluate new policy, improve again, until π stops changing