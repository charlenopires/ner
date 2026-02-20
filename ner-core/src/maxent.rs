//! # Maximum Entropy (Logistic Regression) para NER
//!
//! Implementação de um classificador discriminativo log-linear (MaxEnt).
//! Diferente do HMM (generativo), este modelo aprende pesos para combinar
//! features arbitrárias.
//!
//! ## Algoritmo
//! - **Treinamento**: Stochastic Gradient Descent (SGD) com regularização L2.
//! - **Predição**: Classificação local (greedy) ou MEMM (se features de transição forem usadas).
//!
//! O modelo calcula: P(tag | features) ~ exp(dot(weights, features))

use std::collections::{HashMap, HashSet};
use serde::{Deserialize, Serialize};
use crate::corpus::AnnotatedSentence;
use crate::features::{self, FeatureVector, Gazetteers};


/// Modelo de Entropia Máxima (MaxEnt), também conhecido como Regressão Logística Multinomial.
///
/// Diferente do HMM, que modela como os dados foram gerados ($P(x,y)$), o MaxEnt é **discriminativo**:
/// modela diretamente a probabilidade da classe dado o dado ($P(y|x)$).
///
/// # Vantagens
/// - Permite usar features arbitrárias e sobrepostas (ex: "palavra anterior" E "sufixo da palavra atual")
///   sem violar suposições de independência (como no Naive Bayes ou HMM).
/// - Estado da arte antes do Deep Learning (junto com CRF).
///
/// # Fórmula
/// $$ P(y|x) = \frac{\exp(\sum_i w_i \cdot f_i(x,y))}{Z(x)} $$
/// Onde $Z(x)$ é o fator de normalização (soma de todos os numeradores possíveis).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MaxEntModel {
    /// Mapa de pesos $w_{feature, tag}$.
    /// Chave: `(feature_name, tag)`. Valor: peso.
    /// Pesos positivos indicam correlação positiva, negativos correlação inversa.
    weights: HashMap<(String, String), f64>,
    /// Lista de todas as tags possíveis (labels de classe).
    tags: Vec<String>,
}

impl MaxEntModel {
    pub fn new() -> Self {
        Self {
            weights: HashMap::new(),
            tags: Vec::new(),
        }
    }

    /// Treina o modelo usando **Stochastic Gradient Descent (SGD)**.
    ///
    /// Diferente do HMM que conta frequências, o MaxEnt é treinado iterativamente para
    /// ajustar os pesos e minimizar o erro de classificação no treino.
    ///
    /// # Parâmetros
    /// * `corpus` - Dados anotados para treino.
    /// * `iterations` - Número de épocas (passadas completas pelo corpus).
    /// * `learning_rate` ($\eta$) - Taxa de aprendizado (tamanho do passo do gradiente).
    /// * `lambda` ($\lambda$) - Fator de regularização L2 (ajuda a evitar overfitting punindo pesos muito grandes).
    pub fn train(&mut self, corpus: &[AnnotatedSentence], iterations: usize, learning_rate: f64, lambda: f64) {
        // 1. Coleta todas as tags e inicializa estrutura
        let mut tag_set = HashSet::new();
        for s in corpus {
            for (_, tag) in s.annotations {
                tag_set.insert(tag.to_string());
            }
        }
        self.tags = tag_set.into_iter().collect();
        self.tags.sort();

        let gaz = Gazetteers::new(); // Gazetteers vazios por enquanto ou passados como arg

        for epoch in 0..iterations {
            let mut correct = 0;
            let mut total = 0;

            for sentence in corpus {
                // Tokeniza e extrai features
                // Em um cenário real, tokenização deve alinhar perfeitamente.
                // Aqui reconstruímos tokens simples baseados na anotação para garantir alinhamento.
                let tokens: Vec<crate::tokenizer::Token> = sentence.annotations.iter().enumerate().map(|(i, (text, _))| {
                    crate::tokenizer::Token {
                        text: text.to_string(),
                        start: 0, // irrelevante para features de treino simples
                        end: 0,
                        index: i,
                    }
                }).collect();

                let feature_vectors = features::extract_features(&tokens, &gaz);

                for (i, fv) in feature_vectors.iter().enumerate() {
                    let true_tag = sentence.annotations[i].1;

                    // 1. Predição (Forward step)
                    let scores = self.compute_scores(fv);
                    let probs = self.softmax(&scores);

                    // Apenas para log de acurácia
                    let (pred_tag, _) = self.predict_best(&scores);
                    if pred_tag == true_tag {
                        correct += 1;
                    }
                    total += 1;

                    // 2. Atualização (Backward step - SGD)
                    // Para cada classe, ajustamos os pesos das features ativas.
                    // Regra de atualização: w = w + rate * (indicador_classe_correta - prob_predita)
                    
                    for (tag_idx, tag) in self.tags.iter().enumerate() {
                        let prob = probs[tag_idx];
                        let indicator = if tag == true_tag { 1.0 } else { 0.0 };
                        let error = indicator - prob; // Gradiente do erro

                        // Otimização: só atualiza se o erro for significativo
                        if error.abs() > 1e-6 {
                            for (fname, fval) in &fv.features {
                                let key = (fname.clone(), tag.clone());
                                let current_w = *self.weights.get(&key).unwrap_or(&0.0);
                                
                                // Update com regularização L2 (Ridge)
                                // w_new = w_old + rate * (error * feature_val - lambda * w_old)
                                let grad = error * fval;
                                let reg = lambda * current_w;
                                let new_w = current_w + learning_rate * (grad - reg);
                                
                                // Pruning de pesos muito próximos de zero (sparsity)
                                if new_w.abs() > 1e-9 {
                                    self.weights.insert(key, new_w);
                                } else {
                                    self.weights.remove(&key);
                                }
                            }
                        }
                    }
                }
            }
            
            if epoch % 5 == 0 {
                println!("Epoch {}: Accuracy {:.2}%", epoch, (correct as f64 / total as f64) * 100.0);
            }
        }
    }

    /// Prediz tags para uma sentença (Greedy Decoding).
    ///
    /// # Nota
    /// Nesta implementação simplificada, a decisão é **Local** (Greedy):
    /// Para cada token, escolhemos a tag com maior probabilidade isoladamente.
    ///
    /// Em implementações mais avançadas (MEMM), usaríamos Viterbi considerando
    /// a tag anterior como uma feature.
    pub fn predict(&self, tokens: &[String]) -> Vec<String> {
        let gaz = Gazetteers::new();
        // Reconstrói tokens
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

        // TODO: Suportar features de transição (prev_tag) passando a tag prevista anterior
        // Por enquanto, features.rs busca "prev_word" etc, mas "prev_tag" seria um feature extra.
        // O extract_features padrão não usa prev_tag dinâmico.

        for fv in feature_vectors {
            let scores = self.compute_scores(&fv);
            let (best_tag, _) = self.predict_best(&scores);
            result.push(best_tag);
        }

        result
    }

    fn compute_scores(&self, fv: &FeatureVector) -> HashMap<String, f64> {
        let mut scores = HashMap::new();
        for tag in &self.tags {
            let mut score = 0.0;
            for (fname, fval) in &fv.features {
                if let Some(w) = self.weights.get(&(fname.clone(), tag.clone())) {
                    score += w * fval;
                }
            }
            scores.insert(tag.clone(), score);
        }
        scores
    }

    fn softmax(&self, scores: &HashMap<String, f64>) -> Vec<f64> {
        let max_score = scores.values().cloned().fold(f64::NEG_INFINITY, f64::max);
        let mut exps = Vec::with_capacity(self.tags.len());
        let mut sum = 0.0;

        for tag in &self.tags {
            let s = scores.get(tag).unwrap_or(&0.0);
            let e = (s - max_score).exp();
            exps.push(e);
            sum += e;
        }

        exps.iter().map(|e| e / sum).collect()
    }

    fn predict_best(&self, scores: &HashMap<String, f64>) -> (String, f64) {
        let mut best_tag = self.tags[0].clone();
        let mut best_val = f64::NEG_INFINITY;

        for (tag, &val) in scores {
            if val > best_val {
                best_val = val;
                best_tag = tag.clone();
            }
        }
        (best_tag, best_val)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_maxent_simple_learning() {
        let corpus = vec![
            AnnotatedSentence {
                text: "Lula é presidente",
                domain: "test",
                annotations: &[("Lula", "B-PER"), ("é", "O"), ("presidente", "O")]
            },
            AnnotatedSentence {
                text: "Dilma foi presidente",
                domain: "test",
                annotations: &[("Dilma", "B-PER"), ("foi", "O"), ("presidente", "O")]
            }
        ];

        let mut model = MaxEntModel::new();
        // Mais iterações ou LR maior para garantir convergência em teste pequeno
        model.train(&corpus, 20, 0.1, 0.001); 

        let tokens = vec!["Lula".to_string(), "foi".to_string()];
        let tags = model.predict(&tokens);

        assert_eq!(tags[0], "B-PER"); // Deve aprender que Lula é PER
    }
}
