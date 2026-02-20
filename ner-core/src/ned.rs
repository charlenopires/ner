//! # Named Entity Disambiguation (NED)
//!
//! Este módulo resolve a ambiguidade de entidades nomeadas analisando o contexto
//! ao redor da entidade. Por exemplo, distinguindo "Paris" (a cidade) de "Paris"
//! (a pessoa, em "Paris Hilton").
//!
//! A estratégia básica envolve perfis de contexto esperados para certos tipos de categorias.

use crate::tagger::EntitySpan;
use crate::tokenizer::Token;
use serde::{Deserialize, Serialize};

/// Resultado da desambiguação para uma entidade
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DisambiguatedEntity {
    pub entity: EntitySpan,
    pub original_tag: String,
    pub resolved_tag: String,
    pub confidence: f32,
    pub context_clues: Vec<String>,
}

/// Analisa os tokens e as entidades extraídas pelo NER para refinar suas categorias.
pub fn disambiguate(
    tokens: &[Token],
    entities: &[EntitySpan],
) -> Vec<DisambiguatedEntity> {
    let mut results = Vec::new();

    for entity in entities {
        let (resolved_tag, confidence, clues) = analyze_context(tokens, entity);
        results.push(DisambiguatedEntity {
            entity: entity.clone(),
            original_tag: entity.category.name().to_string(),
            resolved_tag,
            confidence,
            context_clues: clues,
        });
    }

    results
}

fn analyze_context(tokens: &[Token], entity: &EntitySpan) -> (String, f32, Vec<String>) {
    let mut clues = Vec::new();
    let mut resolved_tag = entity.category.name().to_string();
    let confidence;

    // Obtém janela de contexto de +/- 3 tokens
    let start_idx = entity.start_token.saturating_sub(3);
    let end_idx = (entity.end_token + 3).min(tokens.len() - 1);

    let text_lower = entity.text.to_lowercase();

    // Regras Hardcoded simples para propósito educacional:
    if text_lower.contains("paris") {
        let mut is_person = false;
        let mut is_loc = false;

        for i in start_idx..=end_idx {
            let token_lower = tokens[i].text.to_lowercase();
            if token_lower == "hilton" || token_lower == "socialite" || token_lower == "atriz" {
                is_person = true;
                clues.push(format!("Encontrado indicador de pessoa: '{}'", tokens[i].text));
            }
            if token_lower == "frança" || token_lower == "cidade" || token_lower == "capital" {
                is_loc = true;
                clues.push(format!("Encontrado indicador de local: '{}'", tokens[i].text));
            }
        }

        if is_person {
            resolved_tag = "PER".to_string();
            confidence = 0.95;
        } else if is_loc || entity.category.name().contains("LOC") {
            resolved_tag = "LOC".to_string();
            confidence = 0.85;
        } else {
            // Se "Paris" não tiver contexto de pessoa, assumimos LOC como padrão estatístico
            resolved_tag = "LOC".to_string();
            confidence = 0.60;
            clues.push("Nenhum contexto forte, assumindo classe majoritária (Local)".to_string());
        }
    } else {
        // Sem regras específicas, mantém a tag do NER
        confidence = 0.80;
        clues.push("Nenhuma regra de desambiguação específica aplicada".to_string());
    }

    (resolved_tag, confidence, clues)
}
