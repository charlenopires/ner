//! # Named Entity Linking (NEL)
//!
//! Este módulo faz o "Linking" ou "Grounding" de entidades desambiguadas para uma
//! Base de Conhecimento (Knowledge Base - KB). O NEL é crucial para resolver
//! sinônimos ou variações ortográficas para a mesma entidade no mundo real.

use crate::ned::DisambiguatedEntity;
use serde::{Deserialize, Serialize};

/// Um registro simulado em nossa Base de Conhecimento "Wikidata Mock"
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct KbRecord {
    pub id: String,
    pub name: String,
    pub description: String,
    pub url: String,
}

/// Entidade após a etapa de Linking
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LinkedEntity {
    pub disambiguated: DisambiguatedEntity,
    pub kb_match: Option<KbRecord>,
    pub match_score: f32,
}

/// Simulated Knowledge Base with predefined entities
pub struct KnowledgeBase {
    records: Vec<KbRecord>,
}

impl KnowledgeBase {
    pub fn new() -> Self {
        Self {
            records: vec![
                KbRecord {
                    id: "Q36098".to_string(),
                    name: "Luiz Inácio Lula da Silva".to_string(),
                    description: "39º presidente do Brasil".to_string(),
                    url: "https://www.wikidata.org/wiki/Q36098".to_string(),
                },
                KbRecord {
                    id: "Q155".to_string(),
                    name: "Brasil".to_string(),
                    description: "República Federativa do Brasil, país na América do Sul".to_string(),
                    url: "https://www.wikidata.org/wiki/Q155".to_string(),
                },
                KbRecord {
                    id: "Q47454".to_string(),
                    name: "Paris Hilton".to_string(),
                    description: "Personalidade de televisão, empresária e socialite americana".to_string(),
                    url: "https://www.wikidata.org/wiki/Q47454".to_string(),
                },
                KbRecord {
                    id: "Q90".to_string(),
                    name: "Paris".to_string(),
                    description: "Capital e a cidade mais populosa da França".to_string(),
                    url: "https://www.wikidata.org/wiki/Q90".to_string(),
                },
                KbRecord {
                    id: "Q312".to_string(),
                    name: "Apple Inc.".to_string(),
                    description: "Empresa multinacional norte-americana de eletrônicos e software".to_string(),
                    url: "https://www.wikidata.org/wiki/Q312".to_string(),
                },
            ],
        }
    }

    /// Realiza a busca ingênua (naive) na base de conhecimento usando match parcial
    pub fn link(&self, entities: &[DisambiguatedEntity]) -> Vec<LinkedEntity> {
        let mut results = Vec::new();

        for ent in entities {
            let mut best_match = None;
            let mut best_score = 0.0;
            let query = ent.entity.text.to_lowercase();

            for record in &self.records {
                let name_lower = record.name.to_lowercase();
                
                // Métrica muito simples:
                // Se a busca é exata ou uma contém a outra, e o tipo sugerido do NED faz sentido:
                // Ex: Se o NED diz PER e o record id="Q47454" (Paris Hilton), pontuação sobe.
                let mut score = 0.0;
                
                if name_lower == query {
                    score += 0.8;
                } else if name_lower.contains(&query) || query.contains(&name_lower) {
                    score += 0.5;
                }
                
                // Refinamento baseado na tag do NED (hardcoded simulation):
                if score > 0.0 {
                    if ent.resolved_tag == "PER" && (record.id == "Q36098" || record.id == "Q47454") {
                        score += 0.15;
                    }
                    if ent.resolved_tag == "LOC" && (record.id == "Q155" || record.id == "Q90") {
                        score += 0.15;
                    }
                    if ent.resolved_tag == "ORG" && record.id == "Q312" {
                        score += 0.15;
                    }
                }

                if score > best_score {
                    best_score = score;
                    best_match = Some(record.clone());
                }
            }

            // Apenas ligamos se o score for aceitável
            if best_score >= 0.5 {
                results.push(LinkedEntity {
                    disambiguated: ent.clone(),
                    kb_match: best_match,
                    match_score: best_score,
                });
            } else {
                results.push(LinkedEntity {
                    disambiguated: ent.clone(),
                    kb_match: None,
                    match_score: 0.0,
                });
            }
        }

        results
    }
}

impl Default for KnowledgeBase {
    fn default() -> Self {
        Self::new()
    }
}
