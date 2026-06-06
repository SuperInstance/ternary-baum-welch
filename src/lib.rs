//! # ternary-baum-welch
//!
//! Baum-Welch training algorithm for ternary Hidden Markov Models (HMMs)
//! where both states and emissions belong to the set {-1, 0, +1}.



/// A ternary HMM with 3 states labeled -1, 0, +1 (indexed as 0, 1, 2 internally).
#[derive(Debug, Clone)]
pub struct TernaryHmm {
    /// Initial state probabilities: pi[i] = P(state_0 = i)
    pub pi: Vec<f64>,
    /// Transition matrix: a[i][j] = P(state_t+1 = j | state_t = i)
    pub a: Vec<Vec<f64>>,
    /// Emission matrix: b[i][k] = P(emission_k | state_i)
    /// emissions indexed: -1→0, 0→1, +1→2
    pub b: Vec<Vec<f64>>,
}

impl TernaryHmm {
    /// Create a new ternary HMM with uniform initialization.
    pub fn uniform() -> Self {
        let v = 1.0 / 3.0;
        TernaryHmm {
            pi: vec![v; 3],
            a: vec![vec![v; 3]; 3],
            b: vec![vec![v; 3]; 3],
        }
    }

    /// Create from explicit parameters.
    pub fn new(pi: Vec<f64>, a: Vec<Vec<f64>>, b: Vec<Vec<f64>>) -> Result<Self, String> {
        if pi.len() != 3 || a.len() != 3 || b.len() != 3 {
            return Err("All parameter arrays must have 3 entries (for states -1, 0, +1)".into());
        }
        for row in &a {
            if row.len() != 3 {
                return Err("Transition matrix must be 3x3".into());
            }
        }
        for row in &b {
            if row.len() != 3 {
                return Err("Emission matrix must be 3x3".into());
            }
        }
        Ok(TernaryHmm { pi, a, b })
    }

    /// Convert ternary value (-1, 0, +1) to internal index (0, 1, 2).
    pub fn emit_to_idx(emit: i32) -> usize {
        match emit {
            -1 => 0,
            0 => 1,
            1 => 2,
            _ => panic!("Invalid ternary value: {emit}"),
        }
    }

    /// Convert internal index to ternary value.
    pub fn idx_to_emit(idx: usize) -> i32 {
        match idx {
            0 => -1,
            1 => 0,
            2 => 1,
            _ => panic!("Invalid index: {idx}"),
        }
    }

    /// Forward algorithm: compute alpha[t][i] = P(o_1..o_t, state_t = i)
    /// Returns alpha matrix and the log-likelihood.
    pub fn forward(&self, obs: &[i32]) -> (Vec<Vec<f64>>, f64) {
        let t_max = obs.len();
        let mut alpha = vec![vec![0.0; 3]; t_max];

        // Initialization
        let o0 = Self::emit_to_idx(obs[0]);
        for i in 0..3 {
            alpha[0][i] = self.pi[i] * self.b[i][o0];
        }

        // Induction
        for t in 1..t_max {
            let ot = Self::emit_to_idx(obs[t]);
            for j in 0..3 {
                let sum: f64 = (0..3).map(|i| alpha[t - 1][i] * self.a[i][j]).sum();
                alpha[t][j] = sum * self.b[j][ot];
            }
        }

        // Log-likelihood
        let ll: f64 = alpha[t_max - 1].iter().sum();
        (alpha, ll.ln())
    }

    /// Backward algorithm: compute beta[t][i] = P(o_{t+1}..o_T | state_t = i)
    pub fn backward(&self, obs: &[i32]) -> Vec<Vec<f64>> {
        let t_max = obs.len();
        let mut beta = vec![vec![0.0; 3]; t_max];

        // Initialization
        for i in 0..3 {
            beta[t_max - 1][i] = 1.0;
        }

        // Induction (backwards)
        for t in (0..t_max - 1).rev() {
            let ot1 = Self::emit_to_idx(obs[t + 1]);
            for i in 0..3 {
                beta[t][i] = (0..3)
                    .map(|j| self.a[i][j] * self.b[j][ot1] * beta[t + 1][j])
                    .sum();
            }
        }

        beta
    }

    /// E-step: compute gamma and xi matrices.
    ///
    /// gamma[t][i] = P(state_t = i | O)
    /// xi[t][i][j] = P(state_t = i, state_{t+1} = j | O)
    pub fn e_step(&self, obs: &[i32]) -> (Vec<Vec<f64>>, Vec<Vec<Vec<f64>>>) {
        let (alpha, _) = self.forward(obs);
        let beta = self.backward(obs);
        let t_max = obs.len();

        let prob_o: f64 = alpha[t_max - 1].iter().sum();

        // Gamma
        let mut gamma = vec![vec![0.0; 3]; t_max];
        for t in 0..t_max {
            for i in 0..3 {
                gamma[t][i] = alpha[t][i] * beta[t][i] / prob_o;
            }
        }

        // Xi
        let mut xi = vec![vec![vec![0.0; 3]; 3]; t_max - 1];
        for t in 0..t_max - 1 {
            let ot1 = Self::emit_to_idx(obs[t + 1]);
            for i in 0..3 {
                for j in 0..3 {
                    xi[t][i][j] =
                        alpha[t][i] * self.a[i][j] * self.b[j][ot1] * beta[t + 1][j] / prob_o;
                }
            }
        }

        (gamma, xi)
    }

    /// M-step: update parameters from gamma and xi.
    pub fn m_step(
        &self,
        obs: &[i32],
        gamma: &[Vec<f64>],
        xi: &[Vec<Vec<f64>>],
    ) -> TernaryHmm {
        let t_max = obs.len();

        // Update pi
        let pi: Vec<f64> = (0..3).map(|i| gamma[0][i]).collect();

        // Update transition matrix
        let mut a = vec![vec![0.0; 3]; 3];
        for i in 0..3 {
            let gamma_sum: f64 = (0..t_max - 1).map(|t| gamma[t][i]).sum();
            for j in 0..3 {
                let xi_sum: f64 = (0..t_max - 1).map(|t| xi[t][i][j]).sum();
                a[i][j] = if gamma_sum > 0.0 { xi_sum / gamma_sum } else { 1.0 / 3.0 };
            }
        }

        // Update emission matrix
        let mut b = vec![vec![0.0; 3]; 3];
        for i in 0..3 {
            let gamma_sum: f64 = (0..t_max).map(|t| gamma[t][i]).sum();
            for k in 0..3 {
                let emit_sum: f64 = (0..t_max)
                    .filter(|&t| Self::emit_to_idx(obs[t]) == k)
                    .map(|t| gamma[t][i])
                    .sum();
                b[i][k] = if gamma_sum > 0.0 { emit_sum / gamma_sum } else { 1.0 / 3.0 };
            }
        }

        TernaryHmm { pi, a, b }
    }

    /// Train the HMM using Baum-Welch (EM) algorithm.
    ///
    /// Returns a vector of log-likelihoods at each iteration.
    pub fn train(
        &self,
        obs: &[i32],
        max_iter: usize,
        tol: f64,
    ) -> (TernaryHmm, Vec<f64>) {
        let mut hmm = self.clone();
        let mut ll_history = Vec::new();

        let (_, ll) = hmm.forward(obs);
        ll_history.push(ll);

        let mut prev_ll = ll;

        for _ in 0..max_iter {
            let (gamma, xi) = hmm.e_step(obs);
            hmm = hmm.m_step(obs, &gamma, &xi);

            let (_, new_ll) = hmm.forward(obs);
            ll_history.push(new_ll);

            if (new_ll - prev_ll).abs() < tol {
                break;
            }
            prev_ll = new_ll;
        }

        (hmm, ll_history)
    }
}

/// Compute the probability of an observation sequence given an HMM.
pub fn sequence_probability(hmm: &TernaryHmm, obs: &[i32]) -> f64 {
    let (_, ll) = hmm.forward(obs);
    ll.exp()
}

/// Compute the most likely state sequence using the Viterbi algorithm.
pub fn viterbi(hmm: &TernaryHmm, obs: &[i32]) -> Vec<i32> {
    let t_max = obs.len();
    let mut delta = vec![vec![0.0_f64; 3]; t_max];
    let mut psi = vec![vec![0_usize; 3]; t_max];

    let o0 = TernaryHmm::emit_to_idx(obs[0]);
    for i in 0..3 {
        delta[0][i] = hmm.pi[i].ln() + hmm.b[i][o0].ln();
    }

    for t in 1..t_max {
        let ot = TernaryHmm::emit_to_idx(obs[t]);
        for j in 0..3 {
            let (best_val, best_idx) = (0..3)
                .map(|i| (delta[t - 1][i] + hmm.a[i][j].ln(), i))
                .fold((f64::NEG_INFINITY, 0_usize), |acc, (v, idx)| {
                    if v > acc.0 { (v, idx) } else { acc }
                });
            delta[t][j] = best_val + hmm.b[j][ot].ln();
            psi[t][j] = best_idx;
        }
    }

    // Backtrace
    let mut path = vec![0_usize; t_max];
    let (_best_val, best_idx) = delta[t_max - 1]
        .iter()
        .enumerate()
        .fold((f64::NEG_INFINITY, 0_usize), |acc, (idx, &v)| {
            if v > acc.0 { (v, idx) } else { acc }
        });
    path[t_max - 1] = best_idx;

    for t in (0..t_max - 1).rev() {
        path[t] = psi[t + 1][path[t + 1]];
    }

    path.iter().map(|&i| TernaryHmm::idx_to_emit(i)).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn near(a: f64, b: f64, eps: f64) -> bool {
        (a - b).abs() < eps
    }

    #[test]
    fn test_forward_probabilities_sum_correctly() {
        let hmm = TernaryHmm::new(
            vec![0.5, 0.3, 0.2],
            vec![
                vec![0.6, 0.3, 0.1],
                vec![0.2, 0.5, 0.3],
                vec![0.1, 0.3, 0.6],
            ],
            vec![
                vec![0.7, 0.2, 0.1],
                vec![0.1, 0.8, 0.1],
                vec![0.1, 0.2, 0.7],
            ],
        )
        .unwrap();

        let obs = vec![1, 0, -1, 1, 0];
        let (alpha, ll) = hmm.forward(&obs);

        // Alpha at each time step should sum to P(o_1..o_t)
        for t in 0..obs.len() {
            let sum: f64 = alpha[t].iter().sum();
            assert!(sum > 0.0, "Alpha sum at t={t} should be positive");
            assert!(sum <= 1.0 + 1e-9, "Alpha sum at t={t} should be <= 1");
        }

        // Final alpha sum should match exp(ll)
        let final_sum: f64 = alpha[obs.len() - 1].iter().sum();
        assert!(near(final_sum, ll.exp(), 1e-9));
    }

    #[test]
    fn test_backward_matches_forward() {
        let hmm = TernaryHmm::new(
            vec![1.0 / 3.0; 3],
            vec![vec![0.5, 0.3, 0.2], vec![0.2, 0.6, 0.2], vec![0.3, 0.3, 0.4]],
            vec![vec![0.6, 0.3, 0.1], vec![0.1, 0.8, 0.1], vec![0.2, 0.2, 0.6]],
        )
        .unwrap();

        let obs = vec![-1, 0, 1, -1, 0, 1];
        let (alpha, _) = hmm.forward(&obs);
        let beta = hmm.backward(&obs);

        let prob_forward: f64 = alpha[obs.len() - 1].iter().sum();

        // P(O) via backward: sum_i pi[i] * b[i][o0] * beta[0][i]
        let o0 = TernaryHmm::emit_to_idx(obs[0]);
        let prob_backward: f64 = (0..3)
            .map(|i| hmm.pi[i] * hmm.b[i][o0] * beta[0][i])
            .sum();

        assert!(
            near(prob_forward, prob_backward, 1e-9),
            "Forward and backward should give same P(O): {prob_forward} vs {prob_backward}"
        );
    }

    #[test]
    fn test_estep_valid_gamma() {
        let hmm = TernaryHmm::new(
            vec![0.4, 0.4, 0.2],
            vec![
                vec![0.5, 0.3, 0.2],
                vec![0.2, 0.5, 0.3],
                vec![0.1, 0.4, 0.5],
            ],
            vec![
                vec![0.6, 0.3, 0.1],
                vec![0.1, 0.8, 0.1],
                vec![0.2, 0.3, 0.5],
            ],
        )
        .unwrap();

        let obs = vec![0, 1, -1, 0, 1];
        let (gamma, xi) = hmm.e_step(&obs);

        // Gamma rows should sum to 1
        for t in 0..obs.len() {
            let row_sum: f64 = gamma[t].iter().sum();
            assert!(near(row_sum, 1.0, 1e-9), "Gamma at t={t} sums to {row_sum}");
            for i in 0..3 {
                assert!(gamma[t][i] >= 0.0, "Gamma should be non-negative");
            }
        }

        // Xi at each t should sum to 1
        for t in 0..obs.len() - 1 {
            let xi_sum: f64 = xi[t].iter().flat_map(|r| r.iter()).sum();
            assert!(near(xi_sum, 1.0, 1e-9), "Xi at t={t} sums to {xi_sum}");
        }
    }

    #[test]
    fn test_mstep_improves_likelihood() {
        let hmm = TernaryHmm::new(
            vec![0.4, 0.4, 0.2],
            vec![
                vec![0.5, 0.3, 0.2],
                vec![0.2, 0.5, 0.3],
                vec![0.1, 0.4, 0.5],
            ],
            vec![
                vec![0.6, 0.3, 0.1],
                vec![0.1, 0.8, 0.1],
                vec![0.2, 0.3, 0.5],
            ],
        )
        .unwrap();

        let obs = vec![1, 1, 0, 1, -1, 0, 1, 1, 0, 1];
        let (_, ll_before) = hmm.forward(&obs);

        let (gamma, xi) = hmm.e_step(&obs);
        let new_hmm = hmm.m_step(&obs, &gamma, &xi);
        let (_, ll_after) = new_hmm.forward(&obs);

        assert!(
            ll_after >= ll_before - 1e-9,
            "Likelihood should not decrease after M-step: {ll_before} -> {ll_after}"
        );
    }

    #[test]
    fn test_convergence_detected() {
        // HMM with some structure, train on data that should converge quickly
        let hmm = TernaryHmm::new(
            vec![0.6, 0.2, 0.2],
            vec![
                vec![0.7, 0.2, 0.1],
                vec![0.1, 0.7, 0.2],
                vec![0.2, 0.1, 0.7],
            ],
            vec![
                vec![0.8, 0.1, 0.1],
                vec![0.1, 0.8, 0.1],
                vec![0.1, 0.1, 0.8],
            ],
        )
        .unwrap();

        let obs = vec![-1, -1, 0, 0, 1, 1, -1, -1];
        let (_, ll_history) = hmm.train(&obs, 1000, 1e-6);

        // Should converge well before 1000 iterations
        assert!(
            ll_history.len() < 1001,
            "Should converge before max iterations, got {} entries",
            ll_history.len()
        );

        // Log-likelihood should be monotonically non-decreasing
        for w in ll_history.windows(2) {
            assert!(
                w[1] >= w[0] - 1e-9,
                "LL should be non-decreasing: {} -> {}",
                w[0],
                w[1]
            );
        }
    }

    #[test]
    fn test_manual_verification_small_example() {
        // Simple HMM with known structure: state -1 mostly emits -1, etc.
        let hmm = TernaryHmm::new(
            vec![1.0, 0.0, 0.0],
            vec![
                vec![0.1, 0.8, 0.1],
                vec![0.1, 0.1, 0.8],
                vec![0.8, 0.1, 0.1],
            ],
            vec![
                vec![0.9, 0.05, 0.05],
                vec![0.05, 0.9, 0.05],
                vec![0.05, 0.05, 0.9],
            ],
        )
        .unwrap();

        // obs = [-1, 0, 1]: should cycle through states
        let obs = vec![-1, 0, 1];
        let (alpha, _) = hmm.forward(&obs);

        // Manually compute alpha[0]:
        // alpha[0][0] = pi[0] * b[0][0] = 1.0 * 0.9 = 0.9
        // alpha[0][1] = pi[1] * b[1][0] = 0.0 * 0.05 = 0.0
        // alpha[0][2] = pi[2] * b[2][0] = 0.0 * 0.05 = 0.0
        assert!(near(alpha[0][0], 0.9, 1e-10));
        assert!(near(alpha[0][1], 0.0, 1e-10));
        assert!(near(alpha[0][2], 0.0, 1e-10));

        // alpha[1][1] = sum_i alpha[0][i] * a[i][1] * b[1][1]
        //   = 0.9 * 0.8 * 0.9 = 0.648
        assert!(near(alpha[1][1], 0.648, 1e-10));

        // alpha[2][2] should be dominant since the chain cycles to state +1
        let max_state = alpha[2]
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
            .unwrap()
            .0;
        assert_eq!(max_state, 2, "State +1 should be most likely at t=2");
    }

    #[test]
    fn test_train_on_deterministic_sequence() {
        let hmm = TernaryHmm::uniform();
        let obs = vec![-1, -1, -1, -1, -1, -1, -1, -1, -1, -1];

        let (trained, _ll_history) = hmm.train(&obs, 200, 1e-8);

        // After training on all -1s, one state should strongly prefer emitting -1
        let best_state = (0..3)
            .max_by(|&i, &j| trained.b[i][0].partial_cmp(&trained.b[j][0]).unwrap())
            .unwrap();
        assert!(
            trained.b[best_state][0] > 0.8,
            "Expected a state to strongly emit -1 after training, got {}",
            trained.b[best_state][0]
        );
    }

    #[test]
    fn test_viterbi_basic() {
        let hmm = TernaryHmm::new(
            vec![1.0, 0.0, 0.0],
            vec![
                vec![0.1, 0.9, 0.0],
                vec![0.0, 0.1, 0.9],
                vec![0.9, 0.0, 0.1],
            ],
            vec![
                vec![0.95, 0.025, 0.025],
                vec![0.025, 0.95, 0.025],
                vec![0.025, 0.025, 0.95],
            ],
        )
        .unwrap();

        let obs = vec![-1, 0, 1];
        let path = viterbi(&hmm, &obs);

        // Best path should be -1 → 0 → +1 matching the cyclic structure
        assert_eq!(path, vec![-1, 0, 1]);
    }

    #[test]
    fn test_sequence_probability() {
        let hmm = TernaryHmm::new(
            vec![1.0 / 3.0; 3],
            vec![
                vec![0.5, 0.25, 0.25],
                vec![0.25, 0.5, 0.25],
                vec![0.25, 0.25, 0.5],
            ],
            vec![
                vec![0.7, 0.2, 0.1],
                vec![0.1, 0.8, 0.1],
                vec![0.1, 0.2, 0.7],
            ],
        )
        .unwrap();

        let prob = sequence_probability(&hmm, &vec![0, 0, 0]);
        assert!(prob > 0.0 && prob < 1.0, "P(O) should be in (0,1), got {prob}");
    }

    #[test]
    fn test_empty_and_single_observation() {
        let hmm = TernaryHmm::uniform();
        let obs = vec![1];
        let (alpha, ll) = hmm.forward(&obs);
        assert_eq!(alpha.len(), 1);
        assert!(ll.is_finite());

        let beta = hmm.backward(&obs);
        assert_eq!(beta.len(), 1);
        assert!(near(beta[0][0], 1.0, 1e-10));
    }
}
