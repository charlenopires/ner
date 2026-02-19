//! # CRF — Conditional Random Field Linear-Chain
//!
//! Implementação didática do CRF para marcação de sequências (NER).
//! O CRF é um modelo probabilístico que aprende a probabilidade condicional
//! `P(y|x)` onde `y` é a sequência de tags e `x` é a sequência de tokens.
//!
//! ## Intuição
//!
//! Diferente do HMM (que faz suposições de independência), o CRF pode usar
//! features arbitrárias do texto inteiro ao decidir cada tag. Para "São Paulo",
//! o CRF sabe que se "São" foi marcado como B-LOC, então "Paulo" quase certamente
//! deve ser I-LOC — isso é capturado pela **matriz de transição**.
//!
//! ## Estrutura do Modelo
//!
//! Score total de uma sequência de tags:
//!
//! ```text
//! score(y, x) = Σ_i [Σ_k w_k * f_k(y_{i-1}, y_i, x, i)]
//!             = Σ_i [emission_score(y_i, x, i) + transition_score(y_{i-1}, y_i)]
//! ```
//!
//! A probabilidade é a softmax dos scores (via normalização Z).

use std::collections::HashMap;

use serde::{Deserialize, Serialize};

use crate::features::FeatureVector;
use crate::tagger::Tag;

/// Modelo CRF com pesos aprendidos/definidos
///
/// Contém:
/// - `emission_weights`: mapa feature_name×tag_label → peso
/// - `transition_weights`: matriz tag_prev×tag_next → peso
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CrfModel {
    /// Pesos de emissão: (feature_name + "|" + tag_label) → f64
    pub emission_weights: HashMap<String, f64>,
    /// Pesos de transição: indexed by [prev_tag_idx][next_tag_idx]
    pub transition_weights: Vec<Vec<f64>>,
}

impl CrfModel {
    /// Cria um modelo CRF com pesos zerados
    pub fn new() -> Self {
        let n = Tag::COUNT;
        Self {
            emission_weights: HashMap::new(),
            transition_weights: vec![vec![0.0f64; n]; n],
        }
    }

    /// Calcula o score de emissão para uma tag num token com features dadas
    ///
    /// `score = Σ_k w_{k, tag} * f_k(x, i)`
    pub fn emission_score(&self, features: &FeatureVector, tag: &Tag) -> f64 {
        let tag_label = tag.label();
        features
            .features
            .iter()
            .map(|(feat_name, feat_val)| {
                let key = format!("{feat_name}|{tag_label}");
                feat_val * self.emission_weights.get(&key).unwrap_or(&0.0)
            })
            .sum()
    }

    /// Calcula o score de transição de uma tag para outra
    pub fn transition_score(&self, prev: &Tag, next: &Tag) -> f64 {
        self.transition_weights[prev.index()][next.index()]
    }

    /// Pontua todas as tags possíveis para um token → retorna vetor de (tag, score)
    pub fn score_all_tags(&self, features: &FeatureVector) -> Vec<(Tag, f64)> {
        Tag::all()
            .into_iter()
            .map(|tag| {
                let score = self.emission_score(features, &tag);
                (tag, score)
            })
            .collect()
    }

    /// Configura um peso de emissão
    pub fn set_emission(&mut self, feature: &str, tag: &Tag, weight: f64) {
        let key = format!("{feature}|{}", tag.label());
        self.emission_weights.insert(key, weight);
    }

    /// Configura um peso de transição
    pub fn set_transition(&mut self, from: &Tag, to: &Tag, weight: f64) {
        self.transition_weights[from.index()][to.index()] = weight;
    }
}

impl Default for CrfModel {
    fn default() -> Self {
        Self::new()
    }
}

/// Resultado completo do scoring CRF para uma sequência
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CrfScores {
    /// Para cada token: vetor de (tag, emission_score)
    pub emission_scores: Vec<Vec<(String, f64)>>,
    /// Matriz de transições (para visualização)
    pub transition_matrix: Vec<Vec<f64>>,
    /// Labels das tags (para indexação da visualização)
    pub tag_labels: Vec<String>,
}

/// Calcula os scores de emissão para todos os tokens e tags
pub fn compute_emission_scores(
    model: &CrfModel,
    feature_vectors: &[FeatureVector],
) -> Vec<Vec<f64>> {
    let tags = Tag::all();
    feature_vectors
        .iter()
        .map(|fv| tags.iter().map(|tag| model.emission_score(fv, tag)).collect())
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tagger::EntityCategory;

    #[test]
    fn test_emission_score_positive() {
        let mut model = CrfModel::new();
        let tag = Tag::Begin(EntityCategory::Per);
        model.set_emission("is_capitalized", &tag, 2.5);

        let mut fv = FeatureVector::new(0);
        fv.features.insert("is_capitalized".to_string(), 1.0);

        let score = model.emission_score(&fv, &tag);
        assert!((score - 2.5).abs() < 1e-9);
    }

    #[test]
    fn test_transition_score() {
        let mut model = CrfModel::new();
        let b_per = Tag::Begin(EntityCategory::Per);
        let i_per = Tag::Inside(EntityCategory::Per);
        model.set_transition(&b_per, &i_per, 3.0);

        assert!((model.transition_score(&b_per, &i_per) - 3.0).abs() < 1e-9);
        // Transição default é 0
        assert!((model.transition_score(&Tag::Outside, &i_per)).abs() < 1e-9);
    }
}
