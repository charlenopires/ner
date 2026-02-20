//! # Span-based NER
//!
//! Abordagem moderna que classifica diretamente trechos de texto (spans) em vez de tokens individuais.
//! Isso permite naturalmente lidar com **entidades aninhadas** e resolve problemas de consistência do BIO.
//!
//! ## Algoritmo
//! 1. Gera todos os candidatos a span até um tamanho máximo (ex: 6 tokens).
//! 2. Extrai features ricas para cada span (bordas, conteúdo, contexto).
//! 3. Classifica cada span independentemente (ou com estrutura).
//! 4. Retorna todos os spans classificados como entidade (score > limiar ou argmax != O).

use std::collections::{HashMap, HashSet};
use serde::{Deserialize, Serialize};
use crate::corpus::AnnotatedSentence;
use crate::features::{FeatureVector, Gazetteers};
use crate::tokenizer::Token;

/// Representa um span (intervalo) de tokens com uma label associada.
///
/// # Exemplo
/// Em "Universidade de São Paulo", o span "São Paulo":
/// `Span { start: 2, end: 4, label: "LOC" }`
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Span {
    /// Índice do token inicial (inclusivo)
    pub start: usize,
    /// Índice do token final (exclusivo)
    pub end: usize,
    /// Rótulo da entidade (ex: "PER", "ORG")
    pub label: String,
}

/// Modelo NER baseado em Spans.
///
/// Diferente dos modelos de sequência (CRF, HMM, Perceptron) que classificam cada token
/// com tags BIO (Begin, Inside, Outside), este modelo classifica **diretamente trechos de texto** (spans).
///
/// # Como funciona
/// 1. **Geração de Candidatos**: Enumera todos os spans possíveis até um tamanho máximo $L$
///    (ex: [0,1], [0,2], [1,2]...).
/// 2. **Extração de Features**: Cria um vetor de features para o span inteiro (bordas, conteúdo interno, contexto).
/// 3. **Classificação**: Decide se o span é uma entidade (PER, ORG, LOC) ou não ("O").
///
/// # Vantagens
/// - **Fim do BIO**: Não precisa reconstruir entidades a partir de fragmentos.
/// - **Entidades Aninhadas**: Pode detectar "Universidade de [São Paulo]" (ORG que contém LOC).
/// - **Features Globais**: Pode olhar para o span inteiro de uma vez (ex: "tem 3 palavras e a do meio é 'de'").
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SpanModel {
    /// Pesos do modelo linear: (feature_name, label) -> peso.
    weights: HashMap<(String, String), f64>,
    /// Lista de labels conhecidos (ex: "PER", "ORG", "LOC", "O").
    tags: Vec<String>,
    /// Tamanho máximo de span a ser considerado (otimização).
    max_span_len: usize,
}

impl SpanModel {
    pub fn new() -> Self {
        Self {
            weights: HashMap::new(),
            tags: Vec::new(),
            max_span_len: 6,
        }
    }

    /// Treina o modelo Span-based.
    ///
    /// Utiliza um algoritmo do tipo Perceptron/SGD Estruturado ou Local:
    ///
    /// 1. **Geração de Candidatos**: Para cada sentença, gera todos os spans válidos (dentro de `max_span_len`).
    /// 2. **Feedback Loop**:
    ///    - Compara o span candidato com o Gold Standard (convertido de BIO para Spans).
    ///    - Se o modelo prever errado para aquele span específico, atualiza os pesos.
    /// 3. **Observação**: Atualmente treina de forma independente (cada span é classificado isoladamente).
    pub fn train(&mut self, corpus: &[AnnotatedSentence], iterations: usize) {
        // 1. Coleta tags (excluindo O/B-/I- prefixos, queremos apenas categorias reais + "O")
        let mut tag_set = HashSet::new();
        tag_set.insert("O".to_string());
        
        for s in corpus {
            for (_i, (_word, tag)) in s.annotations.iter().enumerate() {
                if tag != &"O" {
                    let clean_tag = tag.trim_start_matches("B-").trim_start_matches("I-");
                    tag_set.insert(clean_tag.to_string());
                }
            }
        }
        self.tags = tag_set.into_iter().collect();
        self.tags.sort();

        let gaz = Gazetteers::new();

        for _ in 0..iterations {
            for sentence in corpus {
                // Tokens
                let tokens: Vec<Token> = sentence.annotations.iter().enumerate().map(|(i, (text, _))| {
                    Token { text: text.to_string(), start: 0, end: 0, index: i }
                }).collect();
                
                // Extrai Gold Spans do BIO (converte anotação sequencial para spans)
                let bio_tags: Vec<&str> = sentence.annotations.iter().map(|(_, t)| *t).collect();
                let gold_spans = bio_to_spans(&bio_tags);
                // Set para busca rápida: (start, end, label)
                let gold_span_set: HashSet<(usize, usize, String)> = gold_spans.iter()
                    .map(|s| (s.start, s.end, s.label.clone()))
                    .collect();

                // Gera candidatos
                let candidates = self.generate_candidates(tokens.len());
                
                for (start, end) in candidates {
                    let fv = self.extract_span_features(&tokens, start, end, &gaz);
                    
                    // Determina label correto para este span candidato
                    // Se o span start..end estiver no gold set, usa aquele label. Caso contrário, é "O".
                    let true_label = gold_span_set.iter()
                        .find(|(s, e, _)| *s == start && *e == end)
                        .map(|(_, _, l): &(usize, usize, String)| l.clone())
                        .unwrap_or_else(|| "O".to_string());

                    // Predição
                    let pred_label = self.predict_single(&fv);

                    if pred_label != true_label {
                        self.update(&fv, &true_label, &pred_label);
                    }
                }
            }
        }
    }

    /// Prediz entidades em uma lista de tokens.
    ///
    /// Retorna uma lista de objetos `Span` encontrados.
    pub fn predict(&self, tokens: &[String]) -> Vec<Span> {
        let gaz = Gazetteers::new();
        let input_tokens: Vec<Token> = tokens.iter().enumerate().map(|(i, text)| {
             Token { text: text.clone(), start: 0, end: 0, index: i }
        }).collect();

        let candidates = self.generate_candidates(tokens.len());
        let mut results = Vec::new();

        for (start, end) in candidates {
            let fv = self.extract_span_features(&input_tokens, start, end, &gaz);
            let label = self.predict_single(&fv);
            
            if label != "O" {
                results.push(Span {
                    start,
                    end,
                    label,
                });
            }
        }
        
        // Nota: Esta implementação ingênua pode retornar spans sobrepostos (ex: [0,2] PER e [0,1] LOC).
        // Um sistema real aplicaria NMS (Non-Maximum Suppression) ou Programação Dinâmica para resolver conflitos.
        results
    }

    fn generate_candidates(&self, n_tokens: usize) -> Vec<(usize, usize)> {
        let mut spans = Vec::new();
        for len in 1..=self.max_span_len {
            for start in 0..n_tokens {
                let end = start + len;
                if end <= n_tokens {
                    spans.push((start, end));
                }
            }
        }
        spans
    }

    fn extract_span_features(&self, tokens: &[Token], start: usize, end: usize, gaz: &Gazetteers) -> FeatureVector {
        let mut fv = FeatureVector::new(start);
        
        // Features de borda (Boundary features)
        let first_token = &tokens[start];
        let last_token = &tokens[end - 1];
        
        fv.insert(format!("span_first={}", first_token.text.to_lowercase()), 1.0);
        fv.insert(format!("span_last={}", last_token.text.to_lowercase()), 1.0);
        
        // Contexto
        if start > 0 {
            fv.insert(format!("ctx_prev={}", tokens[start - 1].text.to_lowercase()), 1.0);
        }
        if end < tokens.len() {
            fv.insert(format!("ctx_next={}", tokens[end].text.to_lowercase()), 1.0);
        }
        
        // Tamanho
        fv.insert(format!("span_len={}", end - start), 1.0);
        
        // Bag of words interno
        for i in start..end {
            fv.insert(format!("in_span={}", tokens[i].text.to_lowercase()), 1.0);
             if tokens[i].text.chars().next().unwrap().is_uppercase() {
                 fv.insert("span_has_cap", 1.0);
             }
        }

        // Gazetteer match (se o span inteiro bater com gazetteer)
        let span_text: String = tokens[start..end].iter().map(|t| t.text.as_str()).collect::<Vec<_>>().join(" ").to_lowercase();
        if gaz.persons.contains(&span_text) { fv.insert("span_is_person_gaz", 1.0); }
        if gaz.locations.contains(&span_text) { fv.insert("span_is_loc_gaz", 1.0); }
        if gaz.organizations.contains(&span_text) { fv.insert("span_is_org_gaz", 1.0); }

        fv
    }

    fn predict_single(&self, fv: &FeatureVector) -> String {
        let mut best_label = "O".to_string();
        let mut best_score = f64::NEG_INFINITY;

        for tag in &self.tags {
            let score = self.score_label(fv, tag);
            if score > best_score {
                best_score = score;
                best_label = tag.clone();
            }
        }
        best_label
    }

    fn score_label(&self, fv: &FeatureVector, label: &str) -> f64 {
        let mut score = 0.0;
        for (fname, fval) in &fv.features {
            if let Some(w) = self.weights.get(&(fname.clone(), label.to_string())) {
                score += w * fval;
            }
        }
        score
    }

    fn update(&mut self, fv: &FeatureVector, true_label: &str, pred_label: &str) {
        // Perceptron update simples
        for (fname, _fval) in &fv.features {
            *self.weights.entry((fname.clone(), true_label.to_string())).or_insert(0.0) += 1.0;
            *self.weights.entry((fname.clone(), pred_label.to_string())).or_insert(0.0) -= 1.0;
        }
    }
}

/// Helper para converter tags BIO em spans
pub fn bio_to_spans(tags: &[&str]) -> Vec<Span> {
    let mut spans = Vec::new();
    let mut current_start: Option<usize> = None;
    let mut current_label: Option<String> = None;

    for (i, tag) in tags.iter().enumerate() {
        if tag.starts_with("B-") {
            if let Some(start) = current_start {
                spans.push(Span { start, end: i, label: current_label.take().unwrap() });
            }
            current_start = Some(i);
            current_label = Some(tag[2..].to_string());
        } else if tag.starts_with("I-") {
            // Verifica consistência: I deve seguir B ou I do mesmo tipo
            if let Some(ref label) = current_label {
                if &tag[2..] != label {
                    // Inconsistência (novo tipo começou sem B): trata como novo B
                     if let Some(start) = current_start {
                        spans.push(Span { start, end: i, label: current_label.take().unwrap() });
                    }
                    current_start = Some(i);
                    current_label = Some(tag[2..].to_string());
                }
            } else {
                 // Começou com I: trata como B
                current_start = Some(i);
                current_label = Some(tag[2..].to_string());
            }
        } else { // O
            if let Some(start) = current_start {
                spans.push(Span { start, end: i, label: current_label.take().unwrap() });
                current_start = None;
                current_label = None;
            }
        }
    }
    
    // Fecha último span se aberto
    if let Some(start) = current_start {
        spans.push(Span { start, end: tags.len(), label: current_label.take().unwrap() });
    }

    spans
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_bio_to_spans() {
        let tags = vec!["O", "B-PER", "I-PER", "O", "B-LOC"];
        let spans = bio_to_spans(&tags);
        assert_eq!(spans.len(), 2);
        assert_eq!(spans[0], Span { start: 1, end: 3, label: "PER".to_string() });
        assert_eq!(spans[1], Span { start: 4, end: 5, label: "LOC".to_string() });
    }

    #[test]
    fn test_span_learning() {
        let corpus = vec![
            AnnotatedSentence {
                text: "Lula é presidente",
                domain: "test",
                annotations: &[("Lula", "B-PER"), ("é", "O"), ("presidente", "O")]
            }
        ];

        let mut model = SpanModel::new();
        model.train(&corpus, 5);

        let tokens = vec!["Lula".to_string(), "é".to_string()];
        let spans = model.predict(&tokens);

        assert_eq!(spans.len(), 1);
        assert_eq!(spans[0].label, "PER");
        assert_eq!(spans[0].start, 0);
        assert_eq!(spans[0].end, 1);
    }
}
