**Topics Covered:** Word2vec | Transformer | MDP Estimation | MDP Optimal Solution

---

## Section 1: Word2vec

**Q1.** In the Word2vec Skip-gram architecture, you have a vocabulary of size V = 100,000 and an embedding dimension D = 300. Describe the dimensions of the W (embedding) matrix and the W' (lifting) matrix. Explain what role each matrix plays in the forward pass.

**Q2.** Given a center word, the Skip-gram model predicts context words. If the context window size is 2 (two words before and two words after), and you have a training sentence "The quick brown fox jumps", with "brown" as the center word:

- What are the context words?
- How many separate predictions does the model need to make?

**Q3.** Explain the loss function used in Word2vec. Why is cross-entropy loss appropriate for this task? In your answer, describe what Ŷ and Y represent.

**Q4.** After training Word2vec, you have both W and W' matrices. Which matrix do you upload to a hub for others to use as embeddings? Explain why.

**Q5.** Word2vec produces what type of embeddings: context-free or contextual? Explain the difference between these two types of embeddings and give an example of how the word "bears" would be represented differently under each approach.

**Q6.** In the Word2vec architecture, after the center word is embedded into vector Z (dimension D), describe mathematically how the model produces a probability distribution over the entire vocabulary for predicting context words.

**Q7.** How does the tokenizer vocabulary relate to the embedding model in Word2vec? What happens if a word appears during inference that was not seen during training?

---

## Section 2: Transformer Architecture

**Q8.** In the self-attention mechanism, explain the roles of Query (Q), Key (K), and Value (V) matrices. Using the example of the word "bears" in the sentence "I love bears", describe intuitively what query the token "bears" might issue and what keys from other tokens might respond.

**Q9.** Given an input matrix X of dimensions T × D (where T is context length and D is embedding dimension), write down the dimensions of:

- The Q, K, V matrices after projection
- The score matrix S = QK^T
- The attention weights matrix A
- The output V̂ matrix

**Q10.** Explain the purpose of the scaling factor 1/√D in the scaled dot-product attention mechanism. What happens to the softmax output if this scaling is not applied?

**Q11.** What is masking in decoder-based transformer architectures? Why is it necessary, and how is it implemented in the attention mechanism?

**Q12.** Explain the need for multi-head self-attention. How is it analogous to multiple filters in CNNs? If you have H heads, how does this affect the computational complexity?

**Q13.** Why are positional embeddings necessary in transformer architectures? Describe one method for generating positional embeddings using sinusoidal functions.

**Q14.** Draw or describe the complete transformer block, including:

- Layer normalization
- Multi-head self-attention
- Skip connections
- Feed-forward network (MLP)

**Q15.** In a transformer with context size T = 1000 and embedding dimension D = 512, what is the size of the attention matrix? Why does this become a computational bottleneck for long sequences?

**Q16.** Compare the simple dot-product attention (using X·X^T) with the learnable attention (using Q·K^T). What advantage does the learnable version provide?

**Q17.** The softmax in attention produces attention weights. What two properties do these weights satisfy? Why is softmax used instead of other normalization methods?

---

## Section 3: MDP Estimation (Prediction Problem)

**Q18.** Define the state value function $V^π(s)$. Write the mathematical definition using the expected return.

**Q19.** Write the Bellman expectation equation for V^π(s). Explain each component of the equation: $$v(s) = \sum_a \pi(a|s) \sum_{s'} p(s'|s,a) [r(s,a,s') + \gamma v(s')]$$

**Q20.** Consider a simple MDP with two states S1 and S2:

- From S1, taking action leads to S2 with probability 1 and reward 2
- From S2, taking action leads to S1 with probability 1 and reward 0
- Discount factor γ = 0.9

Set up the system of linear equations to solve for V(S1) and V(S2).

**Q21.** In a 5×5 gridworld environment with a uniform random policy (probability 0.25 for each of 4 actions):

- State A teleports to state A' with reward 10
- State B teleports to state B' with reward 5

Using the Bellman equation, explain (not numerically) why the value of state A might be less than 10 while the value of state B might be greater than 5.

**Q22.** The Bellman expectation equation can be written in matrix form as V = R + γPV. Show how this can be rearranged to solve for V directly using matrix inversion. Why is this direct solution problematic in practice?

**Q23.** Explain the iterative policy evaluation algorithm. Write the update equation and explain why the Bellman operator is a contraction that guarantees convergence.

**Q24.** In a gridworld with the following structure:

```
    1    2    3    4
1  (1,1) (1,2) (1,3) +1 (terminal)
2  (2,1) wall  (2,3) -1 (terminal)
3  (3,1) (3,2) (3,3) (3,4)
```

What is the number of possible deterministic policies? Show your calculation.

**Q25.** Explain the difference between the transition model P(s'|s,a) and the MDP dynamics P(s',r|s,a). How is the reward model R(s,a) computed from the MDP dynamics?

---

## Section 4: MDP Optimal Solution (Control Problem)

**Q26.** Write the Bellman optimality equation for V*(s). Explain why this equation is nonlinear and how it differs from the Bellman expectation equation.

**Q27.** Define the state-action value function Q^π(s,a). How does it relate to the state value function V^π(s)?

**Q28.** Describe the policy iteration algorithm. What are the two main steps, and how do they interact to find the optimal policy?

**Q29.** In policy iteration, explain how "greedy policy improvement" works. Given a value function V(s), how do you derive a better policy?

**Q30.** Consider the recycling robot example with states {high, low} representing battery level. The robot can choose to search, wait, or recharge. Write the Bellman optimality equation for V*(high).

**Q31.** Explain why the optimal policy π* might converge before the value function V* converges in policy iteration. What property of the value function allows this?

**Q32.** You are asked to reduce computational complexity of solving the MDP by selecting the discount factor γ. Explain:

- Why this is a valid approach
- How you would select γ to reduce computation
- What is the trade-off involved

**Q33.** In a grid world, after running policy iteration for several iterations, you observe that some states have multiple actions with equal Q-values. What does this mean for the optimal policy?

---

## Section 5: Reinforcement Learning (Monte Carlo & TD Methods)

**Q34.** What is the key difference between MDP (dynamic programming) and Reinforcement Learning in terms of knowledge about the environment?

**Q35.** Write the Monte Carlo update equation for estimating V^π(s). Explain what G_t represents and why we need to wait until episode termination.

**Q36.** The incremental sample mean update is: $$V(S_t) \leftarrow V(S_t) + \alpha[G_t - V(S_t)]$$

Relate this to the Kalman filter update equation. What role does α play?

**Q37.** Explain the TD(0) update equation: $$V(S_t) \leftarrow V(S_t) + \alpha[R_{t+1} + \gamma V(S_{t+1}) - V(S_t)]$$

What is the "TD target"? How does TD(0) combine bootstrapping from dynamic programming with sampling from Monte Carlo?

**Q38.** What is the main limitation of Monte Carlo methods that TD methods address? In what scenarios would you prefer TD over Monte Carlo?

**Q39.** Explain TD(λ) as a weighted combination of n-step returns. What values of λ give you:

- TD(0)?
- Monte Carlo?

**Q40.** Draw the backup diagram for:

- Dynamic programming (full backup)
- Monte Carlo (sample backup to termination)
- TD(0) (one-step sample backup)

---

## Section 6: Mixed/Application Questions (Exam-Style)

**Q41.** You are designing a CLIP-like architecture to learn mappings between images of geometric shapes ("red circle", "blue triangle") and their text descriptions.

**(a)** Draw an architecture showing the text encoder, image encoder, and how they interact.

**(b)** The loss function has two components L+ and L-. Explain what each component does and how the similarity between image and text features is computed.

**Q42.** Compare LLaMA 3-8B to LLaMA 2-7B on the following features. For each, explain the advantage (do NOT just say "bigger is better"):

- Vocabulary size (32,000 vs 128,000)
- Context length (4,096 vs 8,192 tokens)
- Training tokens (~2 trillion vs ~15 trillion)

**Q43.** In the transformer architecture for language models:

- What is the role of the "head" (final layer before output)?
- How many dimensions does the softmax operate over?
- What is the relationship between vocabulary size and the output of the transformer?

**Q44.** Explain the four stages of training an LLM:

1. Pre-training
2. Supervised fine-tuning (SFT)
3. Preference fine-tuning
4. Reasoning fine-tuning

What is the role of reinforcement learning in stages 3 and 4?

**Q45.** You have a simple environment for Monte Carlo estimation. After generating 1000 trajectories with different values of learning rate α, you observe different convergence curves. Explain:

- How does α affect convergence speed?
- What is the trade-off between large and small α values?

---

## Section 7: True/False & Multiple Choice

**Q46.** Which of the following statements about Word2vec are correct?

- A. The W' (lifting) matrix has dimensions D × V
- B. Skip-gram predicts the center word given context words
- C. The output of softmax is a V-dimensional probability distribution
- D. Word2vec embeddings are contextual

**Q47.** Which statements about the transformer attention mechanism are TRUE?

- A. The attention weights matrix A has dimensions T × T
- B. Without positional embeddings, the transformer is permutation invariant
- C. Masking is only needed during training, not inference
- D. The scaling factor √D prevents softmax saturation

**Q48.** In the Bellman expectation equation, which of the following is TRUE?

- A. The equation is nonlinear due to the max operator
- B. The equation relates the value of a state to values of successor states
- C. γ = 1 always leads to convergence
- D. The equation can be solved directly using matrix inversion for small state spaces

**Q49.** Which statements about Monte Carlo vs TD methods are correct?

- A. Monte Carlo requires episode termination to update values
- B. TD(0) uses bootstrapping from estimated values
- C. Monte Carlo has lower variance than TD methods
- D. TD methods can learn online (during the episode)

**Q50.** In policy iteration:

- A. The policy is always deterministic after convergence
- B. Value iteration is an alternative that combines policy evaluation and improvement
- C. The algorithm always terminates in a finite number of iterations
- D. Greedy policy improvement can make the policy worse