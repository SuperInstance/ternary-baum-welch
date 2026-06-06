# ternary-baum-welch

Baum-Welch training algorithm for ternary Hidden Markov Models (HMMs) where both hidden states and observable emissions belong to the set **{-1, 0, +1}**.

## Overview

This crate provides a complete implementation of the Baum-Welch (Expectation-Maximization) algorithm specifically designed for 3-state ternary HMMs. The ternary constraint — working with exactly three states and three emission symbols — enables optimized implementations while maintaining full mathematical rigor.

The Baum-Welch algorithm iteratively refines HMM parameters (initial probabilities, transition matrix, emission matrix) to maximize the likelihood of observed data. This is the standard unsupervised training method for HMMs.

## Features

- **Forward Algorithm** — Computes forward probabilities α(t,i) = P(o₁…oₜ, state_t = i) for all time steps and states
- **Backward Algorithm** — Computes backward probabilities β(t,i) = P(o_{t+1}…o_T | state_t = i)
- **E-Step** — Calculates posterior state probabilities (gamma) and state transition probabilities (xi) given current parameters
- **M-Step** — Updates HMM parameters (π, A, B) to maximize expected log-likelihood based on E-step posteriors
- **Training Loop** — Full EM iteration with configurable convergence detection (tolerance-based stopping) and log-likelihood tracking
- **Viterbi Decoding** — Find the most likely state sequence via the Viterbi algorithm
- **Sequence Probability** — Compute P(O|λ) for an observation sequence

## Usage

```rust
use ternary_baum_welch::{TernaryHmm, viterbi, sequence_probability};

// Create an HMM with custom parameters
let hmm = TernaryHmm::new(
    vec![0.5, 0.3, 0.2],                          // Initial state probabilities
    vec![
        vec![0.6, 0.3, 0.1],                       // Transition matrix
        vec![0.2, 0.5, 0.3],
        vec![0.1, 0.3, 0.6],
    ],
    vec![
        vec![0.7, 0.2, 0.1],                       // Emission matrix
        vec![0.1, 0.8, 0.1],
        vec![0.1, 0.2, 0.7],
    ],
).unwrap();

// Observation sequence of ternary values
let observations = vec![1, 0, -1, 1, 0, -1, 1];

// Train the model (up to 100 iterations, convergence tolerance 1e-8)
let (trained_hmm, ll_history) = hmm.train(&observations, 100, 1e-8);

// The log-likelihood history shows monotonic improvement
println!("Initial LL: {}", ll_history.first().unwrap());
println!("Final LL:   {}", ll_history.last().unwrap());

// Decode the most likely state sequence
let states = viterbi(&trained_hmm, &observations);
println!("Most likely states: {:?}", states);

// Compute sequence probability
let prob = sequence_probability(&trained_hmm, &observations);
```

## Algorithm Details

### Forward-Backward Procedure

The forward variable α(t,i) is computed recursively:
- **Initialization**: α(0,i) = πᵢ · bᵢ(o₀)
- **Induction**: α(t,j) = Σᵢ α(t-1,i) · aᵢⱼ · bⱼ(oₜ)

The backward variable β(t,i):
- **Initialization**: β(T-1,i) = 1
- **Induction**: β(t,i) = Σⱼ aᵢⱼ · bⱼ(o_{t+1}) · β(t+1,j)

### E-Step (Expectation)

Computes sufficient statistics:
- **Gamma**: γ(t,i) = P(state_t = i | O, λ) = α(t,i)·β(t,i) / P(O)
- **Xi**: ξ(t,i,j) = P(state_t = i, state_{t+1} = j | O, λ) = α(t,i)·aᵢⱼ·bⱼ(o_{t+1})·β(t+1,j) / P(O)

### M-Step (Maximization)

Updates parameters:
- **πᵢ** = γ(0,i)
- **aᵢⱼ** = Σₜ ξ(t,i,j) / Σₜ γ(t,i)
- **bᵢ(k)** = Σ_{t: oₜ=k} γ(t,i) / Σₜ γ(t,i)

### Convergence

Training stops when |LL(t) - LL(t-1)| < tolerance, guaranteeing monotonic likelihood improvement (EM property).

## Ternary Representation

Internal indexing maps ternary values to array indices:
| Ternary Value | Internal Index |
|:-:|:-:|
| -1 | 0 |
| 0 | 1 |
| +1 | 2 |

## Performance

- O(T · N²) per iteration where T = sequence length, N = 3 (fixed for ternary)
- Memory: O(T · N) for alpha/beta/gamma, O(T · N²) for xi
- Convergence typically achieved in 10-100 iterations for well-structured data

## License

MIT
