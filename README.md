# Ternary Baum-Welch

Baum-Welch training algorithm for **ternary Hidden Markov Models (HMMs)** where both states and emissions are drawn from {-1, 0, +1}. Implements the full EM loop — forward, backward, E-step (γ, ξ), M-step — plus Viterbi decoding and sequence probability computation.

## Why It Matters

Hidden Markov Models are the workhorse of sequence modeling — speech recognition, bioinformatics, time-series analysis. Standard HMMs operate on arbitrary discrete alphabets, but constraining both hidden states and emissions to ternary {-1, 0, +1} creates a uniquely compact model:

- **3 states, 3 emissions**: The full HMM is described by 3 initial probabilities + 9 transition + 9 emission = 21 parameters.
- **Closed-form updates**: With only 3 states, every E-step and M-step computation is a fixed-size matrix operation.
- **Ternary semantics**: States map to meaningful categories — negative/neutral/positive, reject/hold/accept, below/on/above.

The Baum-Welch algorithm (a special case of Expectation-Maximization) iteratively refines HMM parameters to maximize the likelihood of observed sequences. This crate provides the complete implementation with convergence detection and monotonic likelihood guarantees.

## How It Works

### Forward Algorithm

Computes $\alpha_t(i) = P(o_1, \ldots, o_t, q_t = i \mid \lambda)$ for all $t$ and states $i$:

$$\alpha_1(i) = \pi_i \cdot b_i(o_1)$$
$$\alpha_{t+1}(j) = \left[\sum_{i=0}^{2} \alpha_t(i) \cdot a_{ij}\right] \cdot b_j(o_{t+1})$$

The log-likelihood is $\ln P(O \mid \lambda) = \ln \sum_i \alpha_T(i)$.

### Backward Algorithm

Computes $\beta_t(i) = P(o_{t+1}, \ldots, o_T \mid q_t = i, \lambda)$:

$$\beta_T(i) = 1$$
$$\beta_t(i) = \sum_{j=0}^{2} a_{ij} \cdot b_j(o_{t+1}) \cdot \beta_{t+1}(j)$$

### E-Step: Posterior Probabilities

$$\gamma_t(i) = P(q_t = i \mid O, \lambda) = \frac{\alpha_t(i) \beta_t(i)}{P(O \mid \lambda)}$$

$$\xi_t(i, j) = P(q_t = i, q_{t+1} = j \mid O, \lambda) = \frac{\alpha_t(i) \cdot a_{ij} \cdot b_j(o_{t+1}) \cdot \beta_{t+1}(j)}{P(O \mid \lambda)}$$

### M-Step: Parameter Updates

$$\pi_i^{(\text{new})} = \gamma_0(i)$$
$$a_{ij}^{(\text{new})} = \frac{\sum_{t=0}^{T-2} \xi_t(i,j)}{\sum_{t=0}^{T-2} \gamma_t(i)}$$
$$b_i(k)^{(\text{new})} = \frac{\sum_{t: o_t = k} \gamma_t(i)}{\sum_{t=0}^{T-1} \gamma_t(i)}$$

### Convergence

The log-likelihood is **guaranteed monotonically non-decreasing** across EM iterations:

$$\ell(\lambda^{(t+1)}) \geq \ell(\lambda^{(t)})$$

Training stops when $|\ell^{(t+1)} - \ell^{(t)}| < \epsilon$.

### Viterbi Decoding

Finds the most likely state sequence $\hat{q}^* = \arg\max_q P(q, O \mid \lambda)$ using dynamic programming:

$$\delta_t(j) = \max_i [\delta_{t-1}(i) + \ln a_{ij}] + \ln b_j(o_t)$$

with backtrace via $\psi_t(j)$ pointers. Complexity: O(T · 3²) = O(T).

### Complexity

| Operation | Time | Space |
|-----------|------|-------|
| `forward(obs)` | O(T · 3²) = O(T) | O(T · 3) |
| `backward(obs)` | O(T) | O(T · 3) |
| `e_step(obs)` | O(T) | O(T · 3 + T · 9) |
| `m_step(obs, γ, ξ)` | O(T) | O(21) |
| `train(obs, max_iter, tol)` | O(I · T) | O(T · 12) |
| `viterbi(hmm, obs)` | O(T) | O(T · 3) |

Where T = sequence length, I = iterations until convergence.

## Quick Start

```rust
use ternary_baum_welch::{TernaryHmm, viterbi, sequence_probability};

// Create a ternary HMM
let hmm = TernaryHmm::new(
    vec![0.6, 0.2, 0.2],                     // pi: initial probabilities
    vec![
        vec![0.7, 0.2, 0.1],                  // transitions from state -1
        vec![0.1, 0.7, 0.2],                  // transitions from state 0
        vec![0.2, 0.1, 0.7],                  // transitions from state +1
    ],
    vec![
        vec![0.8, 0.1, 0.1],                  // emissions from state -1
        vec![0.1, 0.8, 0.1],                  // emissions from state 0
        vec![0.1, 0.1, 0.8],                  // emissions from state +1
    ],
).unwrap();

let obs = vec![-1, -1, 0, 0, 1, 1, -1, -1];

// Train (refine parameters)
let (trained, ll_history) = hmm.train(&obs, 200, 1e-8);
// ll_history is monotonically non-decreasing

// Most likely state sequence
let path = viterbi(&trained, &obs);

// Sequence probability
let p = sequence_probability(&trained, &obs);
```

## API

### `TernaryHmm`

| Method | Description |
|--------|-------------|
| `uniform()` | Equal-probability initialization |
| `new(pi, a, b)` | Explicit parameter construction |
| `forward(&obs) → (alpha, log_lik)` | Forward probabilities + log-likelihood |
| `backward(&obs) → beta` | Backward probabilities |
| `e_step(&obs) → (gamma, xi)` | Posterior state and transition probs |
| `m_step(&obs, gamma, xi) → TernaryHmm` | Updated parameters |
| `train(&obs, max_iter, tol) → (TernaryHmm, Vec<f64>)` | Full EM training |

### Standalone Functions

| Function | Description |
|----------|-------------|
| `viterbi(&hmm, &obs) → Vec<i32>` | Most likely state sequence |
| `sequence_probability(&hmm, &obs) → f64` | P(O \| λ) |

## Architecture Notes

The Baum-Welch algorithm maintains the **γ + η = C** conservation link through its probabilistic invariant:

- **γ (structure)**: the transition matrix $A$ — the Markov chain topology
- **η (perturbation)**: the observation sequence — evidence that reshapes beliefs
- **C (conservation)**: probability axioms — all distributions sum to 1, all parameters remain valid

The EM guarantee (monotonic likelihood increase) is itself a conservation law: the free energy $\mathcal{F}(q, \lambda) = \mathbb{E}_q[\ln P(O, Q \mid \lambda)] + H(q)$ is non-decreasing, and its fixed points are stationary points of the likelihood.

## References

- Baum, L.E. & Petrie, T. (1966). *Statistical Inference for Probabilistic Functions of Finite State Markov Chains*. Annals of Mathematical Statistics.
- Rabiner, L. (1989). *A Tutorial on Hidden Markov Models and Selected Applications in Speech Recognition*. Proceedings of the IEEE.
- Bilmes, J. (1998). *A Gentle Tutorial of the EM Algorithm and its Application to Parameter Estimation for Gaussian Mixture and Hidden Markov Models*. ICSI Technical Report.
- Durbin, R. et al. (1998). *Biological Sequence Analysis*. Cambridge University Press.

## License: MIT
