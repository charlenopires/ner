//! # Averaged Perceptron para NER
//!
//! Algoritmo online simples e eficiente, similar ao CRF mas mais rápido de treinar.
//! Utiliza "Lazy Averaging" para evitar custo O(N*T) na atualização dos pesos médios.

use std::collections::{HashMap, HashSet};
use serde::{Deserialize, Serialize};
use crate::corpus::AnnotatedSentence;
use crate::features::{self, FeatureVector, Gazetteers};

/// Modelo Perceptron Médio (Averaged Perceptron).
///
/// O Perceptron é um algoritmo de aprendizado **online** e **mistake-driven**:
/// ele processa uma sentença por vez e só atualiza os pesos se errar a predição.
///
/// # Averaged Perceptron
/// A versão padrão do Perceptron oscila muito. O "Averaged" usa a **média** dos pesos
/// de todas as iterações como modelo final, o que reduz overfitting e estabiliza o aprendizado.
///
/// # Lazy Averaging
/// Calcular a média real a cada passo seria $O(N \cdot T)$. Esta implementação usa
/// "Lazy Averaging" para atualizar a média de uma feature **apenas quando ela é ativa**,
/// mantendo complexidade constante por passo.
///
/// Isso resulta no mesmo modelo matemático que o Averaged Perceptron padrão,
/// mas com eficiência computacional muito maior.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerceptronModel {
    /// Tokenizer interno (para testes e uso standalone).
    /// Para simplificar, o modelo recebe tokens pré-processados ou usa o tokenizador padrão se necessário.
    
    /// Pesos atuais $w$: (feature_name, tag) -> weight.
    weights: HashMap<(String, String), f64>,
    /// Soma acumulada dos pesos: (feature_name, tag) -> $\sum w_t$.
    total_weights: HashMap<(String, String), f64>,
    /// Último passo em que o peso foi atualizado (timestamp $t$).
    last_update: HashMap<(String, String), usize>,
    /// Número total de passos de treino (amostras processadas).
    steps: usize,
    /// Tags conhecidas.
    tags: Vec<String>,
}

impl PerceptronModel {
    pub fn new() -> Self {
        Self {
            weights: HashMap::new(),
            total_weights: HashMap::new(),
            last_update: HashMap::new(),
            steps: 0,
            tags: Vec::new(),
        }
    }

    /// Treina o modelo (Online Learning).
    ///
    /// O algoritmo itera pelo corpus várias vezes (`iterations`). Para cada sentença:
    /// 1. Faz uma predição com os pesos atuais.
    /// 2. Se a predição estiver errada, atualiza os pesos (promove a tag correta, penaliza a errada).
    ///
    /// Ao final, calcula a média dos pesos (finalize_weights) para obter o modelo robusto.
    pub fn train(&mut self, corpus: &[AnnotatedSentence], iterations: usize) {
        // Coleta tags
        let mut tag_set = HashSet::new();
        for s in corpus {
            for (_, tag) in s.annotations {
                tag_set.insert(tag.to_string());
            }
        }
        self.tags = tag_set.into_iter().collect();
        self.tags.sort();

        let gaz = Gazetteers::new();

        for _ in 0..iterations {
            for sentence in corpus {
                // Reconstrói tokens (simplificação)
                let tokens: Vec<crate::tokenizer::Token> = sentence.annotations.iter().enumerate().map(|(i, (text, _))| {
                    crate::tokenizer::Token {
                        text: text.to_string(),
                        start: 0,
                        end: 0,
                        index: i,
                    }
                }).collect();

                let feature_vectors = features::extract_features(&tokens, &gaz);

                for (i, fv) in feature_vectors.iter().enumerate() {
                    let true_tag = sentence.annotations[i].1;
                    
                    // Predição usando pesos REAIS (não averaged durante treino)
                    let pred_tag = self.predict_single(fv, false);

                    // Atualiza apenas em caso de erro (mistake-driven)
                    if pred_tag != true_tag {
                        self.update(fv, true_tag, &pred_tag);
                    }
                    
                    self.steps += 1;
                }
            }
        }
        
        // Finaliza: Atualiza total de todos os pesos até o passo final e calcula média
        self.finalize_weights();
    }

    fn predict_single(&self, fv: &FeatureVector, use_averaged: bool) -> String {
        let mut best_tag = if self.tags.is_empty() { String::new() } else { self.tags[0].clone() };
        let mut best_score = f64::NEG_INFINITY;

        for tag in &self.tags {
            let score = self.score_tag(fv, tag, use_averaged);
            if score > best_score {
                best_score = score;
                best_tag = tag.clone();
            }
        }
        best_tag
    }
    
    fn score_tag(&self, fv: &FeatureVector, tag: &str, _use_averaged: bool) -> f64 {
        let mut score = 0.0;
        // Nota: se use_averaged for true, assume-se que finalize_weights já rodou e weights contém as médias.
        let map = &self.weights;
        
        for (fname, fval) in &fv.features {
            if let Some(w) = map.get(&(fname.clone(), tag.to_string())) {
                score += w * fval;
            }
        }
        score
    }

    /// Atualiza os pesos quando o modelo erra.
    ///
    /// $w_{correto} \leftarrow w_{correto} + \phi(x)$
    /// $w_{errado} \leftarrow w_{errado} - \phi(x)$
    fn update(&mut self, fv: &FeatureVector, true_tag: &str, pred_tag: &str) {
        // Para cada feature ativa
        for (fname, _fval) in &fv.features {
            // Nota: Perceptron binário assume fval=1.0 geralmente, mas aqui usamos generalizado.
            // Para simplificar, assumimos features binárias ou multiplicamos pelo valor.
            
            // Tag correta (promote)
            self.update_feature(fname, true_tag, 1.0);
            // Tag predita (demote)
            self.update_feature(fname, pred_tag, -1.0);
        }
    }
    
    /// Atualiza uma feature específica aplicando Lazy Averaging.
    fn update_feature(&mut self, fname: &str, tag: &str, delta: f64) {
        let key = (fname.to_string(), tag.to_string());
        
        // 1. Atualiza o total acumulado até agora com o peso ANTIGO
        //    (simula que o peso ficou constante desde a última atualização até agora)
        let current_w = *self.weights.get(&key).unwrap_or(&0.0);
        let last_step = *self.last_update.get(&key).unwrap_or(&0);
        let steps_since_update = (self.steps - last_step) as f64;
        
        *self.total_weights.entry(key.clone()).or_insert(0.0) += steps_since_update * current_w;
        self.last_update.insert(key.clone(), self.steps);
        
        // 2. Atualiza o peso atual com a mudança (delta)
        *self.weights.entry(key).or_insert(0.0) += delta;
    }

    /// Finaliza o treinamento calculando as médias finais.
    fn finalize_weights(&mut self) {
        // Itera sobre todas as chaves conhecidas para atualizar o acumulado até o final
        let keys: Vec<(String, String)> = self.weights.keys().cloned().collect();
        
        for key in keys {
            let current_w = *self.weights.get(&key).unwrap_or(&0.0);
            let last_step = *self.last_update.get(&key).unwrap_or(&0);
            let steps_since_update = (self.steps - last_step) as f64;
            
            *self.total_weights.entry(key.clone()).or_insert(0.0) += steps_since_update * current_w;
        }
        
        // Substitui os pesos atuais pelas médias ($ \sum w_t / T $)
        let steps_f64 = self.steps as f64;
        if steps_f64 > 0.0 {
            for (key, total) in &self.total_weights {
                self.weights.insert(key.clone(), total / steps_f64);
            }
        }
        
        // Limpa estruturas auxiliares para economizar memória
        self.total_weights.clear();
        self.last_update.clear();
    }

    /// Predição final (usando pesos médios)
    pub fn predict(&self, tokens: &[String]) -> Vec<String> {
        let gaz = Gazetteers::new();
        let input_tokens: Vec<crate::tokenizer::Token> = tokens.iter().enumerate().map(|(i, text)| {
             crate::tokenizer::Token {
                text: text.clone(),
                start: 0,
                end: 0,
                index: i,
            }
        }).collect();

        let feature_vectors = features::extract_features(&input_tokens, &gaz);
        let mut result = Vec::with_capacity(tokens.len());

        for fv in feature_vectors {
            // Usa weights (que agora são averages)
            result.push(self.predict_single(&fv, true));
        }
        result
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_perceptron_learning_lazy() {
        let corpus = vec![
            AnnotatedSentence {
                text: "Lula é presidente",
                domain: "test",
                annotations: &[("Lula", "B-PER"), ("é", "O"), ("presidente", "O")]
            }
        ];

        let mut model = PerceptronModel::new();
        model.train(&corpus, 5);

        let tokens = vec!["Lula".to_string(), "é".to_string()];
        let tags = model.predict(&tokens);

        assert_eq!(tags[0], "B-PER");
    }
}
