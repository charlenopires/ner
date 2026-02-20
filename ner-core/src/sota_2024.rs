//! # SOTA 2024/2025: Simulador GLiNER (Generalist Model for NER)
//!
//! Este módulo contém uma versão **simulada** e **educacional** do conceito por trás 
//! dos modelos estado-da-arte atuais (SOTA), como o GLiNER.
//!
//! ## Como modelos antigos vs GLiNER funcionam:
//! - **Antigo (ex: CRF/BERT-token-level)**: Classifica *tokens individuais* com tags BIO (B-PER, I-PER).
//!   É ineficiente para entidades longas ou aninhadas, e treina com um número fixo de categorias.
//! - **SOTA (GLiNER)**: É um *Span-based model*. Ele **vetoriza** todas as partes do texto (spans)
//!   e também **vetoriza** representações textuais das categorias desejadas (ex: vetor de "Pessoa").
//!   A similaridade entre o vetor do Span e o vetor da Categoria (Dot Product) deita a predição.
//!   Eso permite Zero-Shot NER (reconhecer qualquer categoria digitada pelo usuário on-the-fly).

use crate::tokenizer::Token;
use rayon::prelude::*;
use serde::{Deserialize, Serialize};

/// Custom span struct for SOTA to allow infinite String categories 
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SotaEntitySpan {
    pub start_token: usize,
    pub end_token: usize,
    pub start: usize,
    pub end: usize,
    pub category: String,
    pub text: String,
    pub confidence: f64,
}

/// Estrutura para descrever a simulação de uma predição SOTA
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SotaPrediction {
    pub entity: SotaEntitySpan,
    pub class_name: String,
    pub similarity_score: f32, // O "Dot Product" simulado
}

/// Um "embedding" simulado para um conceito
type Embedding = Vec<f32>;

/// Dicionário simulado de embeddings para as nossas categorias
fn get_class_embedding(class: &str) -> Embedding {
    // Retorna vetores fixos fictícios que representam o significado das classes no espaço
    match class.to_uppercase().as_str() {
        "PESSOA" | "PER" => vec![0.9, 0.1, 0.2, 0.0, -0.4],
        "LOCAL" | "LOC" => vec![0.1, 0.9, 0.0, 0.3, 0.1],
        "ORGANIZACAO" | "ORG" => vec![0.2, 0.2, 0.8, -0.1, 0.5],
        "DATA" | "DATE" => vec![0.0, 0.0, 0.1, 0.9, 0.0],
        _ => vec![0.0, 0.0, 0.0, 0.0, 0.0],
    }
}

/// O texto de um *span* é convertido em um embedding simples (simulando um Bi-Encoder)
fn get_span_embedding(span_text: &str) -> Embedding {
    let lower = span_text.to_lowercase();
    
    // Hardcoded logic para simular a intuição de uma rede neural que "entendeu" o texto:
    if lower.contains("lula") || lower.contains("silva") || lower.contains("paris hilton") {
        vec![0.85, 0.15, 0.1, 0.0, -0.3] // Próximo de Pessoa
    } else if lower.contains("brasil") || lower.contains("frança") || lower.contains("paris") {
        vec![0.15, 0.88, 0.05, 0.2, 0.1] // Próximo de Local
    } else if lower.contains("apple") || lower.contains("banco") || lower.contains("stf") {
        vec![0.1, 0.1, 0.9, 0.0, 0.6] // Próximo de Org
    } else if lower.contains("ontem") || lower.contains("2024") || lower.contains("março") {
        vec![0.05, 0.05, 0.05, 0.95, -0.1] // Próximo de Data
    } else {
        // Fallback genérico, sem significado forte
        vec![0.0, 0.0, 0.0, 0.0, 0.0]
    }
}

/// Produto Escalar (Dot Product) ou Coseno de Similaridade
fn dot_product(v1: &[f32], v2: &[f32]) -> f32 {
    let mut dot = 0.0;
    for i in 0..v1.len() {
        dot += v1[i] * v2[i];
    }
    // Para simplificar a simulação visual, vamos normalizar grosseiramente para [0, 1]
    (dot.max(0.0) / 1.5).min(1.0)
}

/// Simula o processo de um modelo SOTA Span-based:
/// 1. Avalia todos os pedaços (spans) possíveis do texto até um certo tamanho max.
/// 2. Para cada pedaço, tira o Dot Product contra os embeddings de TODAS as classes pedidas pelo user.
/// 3. Retorna os pedaços com score > Threshold.
pub fn simulate_gliner(
    tokens: &[Token],
    user_classes: &[String],
    threshold: f32,
    max_span_length: usize,
) -> Vec<SotaPrediction> {
    
    // Computa o embedding para as classes solicitadas (uma única vez - "Prompting")
    let class_embeddings: Vec<(String, Embedding)> = user_classes
        .iter()
        .map(|c| (c.clone(), get_class_embedding(c)))
        .collect();

    // Cria as combinações de (Início do Span, Fim do Span)
    let mut span_ranges = Vec::new();
    let n = tokens.len();
    for i in 0..n {
        for j in i..=(i + max_span_length - 1).min(n - 1) {
            span_ranges.push((i, j));
        }
    }

    // Processamento estonteante paralelo de todas as spans contra todas as classes via Rayon
    let mut predictions: Vec<SotaPrediction> = span_ranges
        .par_iter()
        .flat_map(|&(start_tok, end_tok)| {
            let start_byte = tokens[start_tok].start;
            let end_byte = tokens[end_tok].end;
            
            // Reconstrói texto basico juntando os tokens com espaço (simplificação)
            let span_text = tokens[start_tok..=end_tok]
                .iter()
                .map(|t| t.text.as_str())
                .collect::<Vec<_>>()
                .join(" ");

            let span_emb = get_span_embedding(&span_text);
            let mut local_preds = Vec::new();

            // Testa esse pedaço de texto contra as representações das "Ideias Platônicas" (Classes)
            if span_emb.iter().any(|&v| v != 0.0) { // otimizacao simples: pula spans sem "sentido" na simulação
                for (class_name, class_emb) in &class_embeddings {
                    let score = dot_product(&span_emb, class_emb);
                    
                    if score > threshold {
                        local_preds.push(SotaPrediction {
                            entity: SotaEntitySpan {
                                start_token: start_tok,
                                end_token: end_tok,
                                start: start_byte,
                                end: end_byte,
                                category: class_name.clone(),
                                text: span_text.clone(),
                                confidence: score as f64,
                            },
                            class_name: class_name.clone(),
                            similarity_score: score,
                        });
                    }
                }
            }

            local_preds
        })
        .collect();

    // Resolução de NMS (Non-Maximum Suppression) simulada para evitar sobreposição
    // Se há spans cobrindo a mesma área, mantém o de maior score
    predictions.sort_by(|a, b| b.similarity_score.partial_cmp(&a.similarity_score).unwrap());
    
    let mut final_preds = Vec::new();
    let mut used_tokens = vec![false; n];
    
    for pred in predictions {
        let mut overlap = false;
        for i in pred.entity.start_token..=pred.entity.end_token {
            if used_tokens[i] {
                overlap = true;
                break;
            }
        }
        
        if !overlap {
            final_preds.push(pred.clone());
            for i in pred.entity.start_token..=pred.entity.end_token {
                used_tokens[i] = true;
            }
        }
    }

    final_preds
}
