//! # Algoritmo de Viterbi — Decodificação de Sequências CRF
//!
//! O algoritmo de Viterbi é um método de **programação dinâmica** que encontra
//! a sequência de tags mais provável de forma eficiente.
//!
//! ## Intuição
//!
//! Imagine que para cada token temos 9 tags possíveis. Uma busca exaustiva
//! teria complexidade `O(9^N)` para N tokens — impraticável. O Viterbi explora
//! que a **melhor sequência até o token i com tag t** depende apenas da
//! **melhor sequência até o token i-1 com alguma tag anterior** → `O(N × T²)`.
//!
//! ## Algoritmo
//!
//! ```text
//! Inicialização: viterbi[0][t] = emission_score(t, x_0)
//!
//! Recursão: viterbi[i][t] = max_{t'} [viterbi[i-1][t'] + transition(t', t)] + emission(t, x_i)
//!
//! Backtracking: reconstruo o caminho ótimo de trás pra frente
//! ```

use serde::{Deserialize, Serialize};

use crate::crf::{compute_emission_scores, CrfModel};
use crate::features::FeatureVector;
use crate::tagger::Tag;

/// Estado do Viterbi em um instante (para visualização passo a passo)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ViterbiStep {
    /// Índice do token sendo processado
    pub token_index: usize,
    /// Scores acumulados para cada tag neste passo: (label, score, best_prev_label)
    pub scores: Vec<TagScore>,
    /// A tag com maior score neste passo
    pub best_tag: String,
    /// Score do melhor caminho até aqui
    pub best_score: f64,
}

/// Score de uma tag individual no Viterbi
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TagScore {
    /// Label da tag (ex: "B-PER")
    pub tag: String,
    /// Score acumulado até este passo com esta tag
    pub score: f64,
    /// Label da tag anterior que gerou este score ótimo
    pub best_prev: String,
    /// Score de emissão neste step
    pub emission: f64,
    /// Score de transição da tag anterior para esta
    pub transition: f64,
}

/// Resultado completo do Viterbi
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ViterbiResult {
    /// Sequência de tags mais provável (uma por token)
    pub best_sequence: Vec<Tag>,
    /// Probabilidade (não-normalizada) da melhor sequência
    pub best_score: f64,
    /// Tabela completa de scores (para visualização)
    pub steps: Vec<ViterbiStep>,
}

/// Executa o algoritmo de Viterbi sobre os features de uma sequência
///
/// # Parâmetros
/// - `model`: modelo CRF com pesos
/// - `feature_vectors`: features de cada token
///
/// # Retorno
/// [`ViterbiResult`] com a sequência ótima e a tabela de scores para visualização
pub fn viterbi_decode(model: &CrfModel, feature_vectors: &[FeatureVector]) -> ViterbiResult {
    if feature_vectors.is_empty() {
        return ViterbiResult {
            best_sequence: vec![],
            best_score: 0.0,
            steps: vec![],
        };
    }

    let n_tokens = feature_vectors.len();
    let tags = Tag::all();
    let n_tags = tags.len();

    // Pré-calcula scores de emissão: emission[i][t]
    let emission = compute_emission_scores(model, feature_vectors);

    // Tabela Viterbi: viterbi[t] = melhor score acumulado para tag t no token atual
    let mut viterbi: Vec<f64> = vec![f64::NEG_INFINITY; n_tags];
    // Backpointer: backptr[i][t] = índice da tag anterior que maximiza o score
    let mut backptr: Vec<Vec<usize>> = vec![vec![0usize; n_tags]; n_tokens];
    // Steps para visualização
    let mut steps: Vec<ViterbiStep> = Vec::with_capacity(n_tokens);

    // === Inicialização (token 0) ===
    // Sem transição para o primeiro token, só usamos o score de emissão
    for t in 0..n_tags {
        viterbi[t] = emission[0][t];
        backptr[0][t] = t; // aponta para si mesmo
    }

    let (best_tag_0, best_score_0) = best_in_slice(&viterbi);
    steps.push(ViterbiStep {
        token_index: 0,
        scores: (0..n_tags)
            .map(|t| TagScore {
                tag: tags[t].label(),
                score: viterbi[t],
                best_prev: tags[t].label(), // sem anterior no primeiro token
                emission: emission[0][t],
                transition: 0.0,
            })
            .collect(),
        best_tag: tags[best_tag_0].label(),
        best_score: best_score_0,
    });

    // === Recursão (tokens 1..N-1) ===
    for i in 1..n_tokens {
        let mut new_viterbi = vec![f64::NEG_INFINITY; n_tags];

        let mut step_scores = Vec::with_capacity(n_tags);

        for t in 0..n_tags {
            // Encontra a melhor tag anterior para esta tag t
            let mut best_prev_score = f64::NEG_INFINITY;
            let mut best_prev_tag = 0;
            let mut best_transition = 0.0;

            for prev_t in 0..n_tags {
                let trans = model.transition_score(&tags[prev_t], &tags[t]);
                let score = viterbi[prev_t] + trans;
                if score > best_prev_score {
                    best_prev_score = score;
                    best_prev_tag = prev_t;
                    best_transition = trans;
                }
            }

            // Penaliza transições inválidas no esquema BIO
            if !Tag::is_valid_transition(&tags[best_prev_tag], &tags[t]) {
                // Pequena penalidade para manter o esquema BIO
                new_viterbi[t] = best_prev_score + emission[i][t] - 10.0;
            } else {
                new_viterbi[t] = best_prev_score + emission[i][t];
            }

            backptr[i][t] = best_prev_tag;

            step_scores.push(TagScore {
                tag: tags[t].label(),
                score: new_viterbi[t],
                best_prev: tags[best_prev_tag].label(),
                emission: emission[i][t],
                transition: best_transition,
            });
        }

        viterbi = new_viterbi;

        let (best_t, best_s) = best_in_slice(&viterbi);
        steps.push(ViterbiStep {
            token_index: i,
            scores: step_scores,
            best_tag: tags[best_t].label(),
            best_score: best_s,
        });
    }

    // === Backtracking ===
    let (mut best_last, best_total_score) = best_in_slice(&viterbi);
    let mut best_sequence: Vec<Tag> = vec![tags[0].clone(); n_tokens];
    best_sequence[n_tokens - 1] = tags[best_last].clone();

    for i in (0..n_tokens - 1).rev() {
        best_last = backptr[i + 1][best_last];
        best_sequence[i] = tags[best_last].clone();
    }

    ViterbiResult {
        best_sequence,
        best_score: best_total_score,
        steps,
    }
}

/// Retorna (índice, valor) do máximo em um slice
fn best_in_slice(scores: &[f64]) -> (usize, f64) {
    scores
        .iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
        .map(|(i, &v)| (i, v))
        .unwrap_or((0, f64::NEG_INFINITY))
}

/// Converte scores Viterbi em probabilidades softmax (para confiança)
pub fn scores_to_probs(scores: &[f64]) -> Vec<f64> {
    if scores.is_empty() {
        return vec![];
    }
    let max_score = scores.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let exps: Vec<f64> = scores.iter().map(|&s| (s - max_score).exp()).collect();
    let sum: f64 = exps.iter().sum();
    if sum == 0.0 {
        return vec![1.0 / scores.len() as f64; scores.len()];
    }
    exps.iter().map(|e| e / sum).collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::crf::CrfModel;
    use crate::features::FeatureVector;
    use crate::tagger::EntityCategory;

    fn make_fv_with_capitalized(index: usize, capitalized: bool) -> FeatureVector {
        let mut fv = FeatureVector::new(index);
        fv.features.insert("bias".to_string(), 1.0);
        if capitalized {
            fv.features.insert("is_capitalized".to_string(), 1.0);
        }
        fv
    }

    #[test]
    fn test_viterbi_prefers_capitalized_as_per() {
        let mut model = CrfModel::new();
        // Palavra capitalizada → forte sinal para B-PER
        model.set_emission(
            "is_capitalized",
            &Tag::Begin(EntityCategory::Per),
            5.0,
        );
        // Penaliza O para palavras capitalizadas
        model.set_emission("is_capitalized", &Tag::Outside, -3.0);
        // Facilita transição B-PER → I-PER
        model.set_transition(
            &Tag::Begin(EntityCategory::Per),
            &Tag::Inside(EntityCategory::Per),
            3.0,
        );

        let fvs = vec![
            make_fv_with_capitalized(0, true),  // "Lula"
            make_fv_with_capitalized(1, false),  // "é"
        ];

        let result = viterbi_decode(&model, &fvs);
        assert_eq!(result.best_sequence.len(), 2);
        // Primeiro token capitalizado deve ser B-PER
        assert_eq!(result.best_sequence[0], Tag::Begin(EntityCategory::Per));
    }

    #[test]
    fn test_viterbi_empty() {
        let model = CrfModel::new();
        let result = viterbi_decode(&model, &[]);
        assert!(result.best_sequence.is_empty());
    }

    #[test]
    fn test_softmax_sums_to_one() {
        let scores = vec![1.0, 2.0, 3.0, 0.5, -1.0];
        let probs = scores_to_probs(&scores);
        let sum: f64 = probs.iter().sum();
        assert!((sum - 1.0).abs() < 1e-9);
    }
}
