//! # Motor de Regras — Gazetteers e Padrões Regex
//!
//! Um motor de regras complementa o CRF com conhecimento explícito:
//! listas de entidades conhecidas (gazetteers) e expressões regulares
//! para padrões como CPF, CNPJ, datas e siglas.
//!
//! ## Por que combinar regras e CRF?
//!
//! O CRF aprende padrões estatísticos do corpus, mas pode ter dificuldade
//! com entidades raras ou novas. As regras garantem alta precisão para
//! padrões bem definidos (ex: "CNPJ 12.345.678/0001-90" sempre é ORG).

use serde::{Deserialize, Serialize};

use crate::tagger::{EntityCategory, Tag};
use crate::tokenizer::Token;

/// Uma correspondência de regra: qual token foi marcado e com qual tag
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RuleMatch {
    pub token_index: usize,
    pub tag: Tag,
    pub rule_name: String,
    pub confidence: f64,
}

/// Motor de regras com gazetteers e padrões regex
pub struct RuleEngine {
    /// Nomes de pessoas conhecidas (lowercase)
    person_names: Vec<String>,
    /// Cidades, estados e países (lowercase)
    location_names: Vec<String>,
    /// Organizações conhecidas (lowercase, pode ser múltiplas palavras)
    org_names: Vec<Vec<String>>,
    /// Entidades miscelâneas (eventos, produtos, etc.) — lowercase
    misc_names: Vec<Vec<String>>,
    /// Títulos que precedem nomes de pessoas
    person_titles: Vec<String>,
    /// Palavras que indicam organização ao redor
    org_indicators: Vec<String>,
}

impl RuleEngine {
    pub fn new() -> Self {
        Self {
            person_names: vec![],
            location_names: vec![],
            org_names: vec![],
            misc_names: vec![],
            person_titles: [
                "presidente", "ex-presidente", "senador", "senadora", "deputado",
                "deputada", "ministro", "ministra", "governador", "governadora",
                "prefeito", "prefeita", "general", "capitão", "dr", "dra", "prof",
                "profa", "vereador", "vereadora", "secretário", "secretária",
                "diretor", "diretora", "ceo", "jogador", "jogadora", "técnico",
                "técnica", "atleta", "ator", "atriz", "cantor", "cantora",
            ].iter().map(|s| s.to_string()).collect(),
            org_indicators: [
                "s.a.", "s/a", "ltda", "eireli", "me", "epp", "sa", "inc",
                "corp", "holdings", "group", "fc", "esporte", "clube",
            ].iter().map(|s| s.to_string()).collect(),
        }
    }

    pub fn add_person(&mut self, name: &str) {
        self.person_names.push(name.to_lowercase());
    }

    pub fn add_location(&mut self, name: &str) {
        self.location_names.push(name.to_lowercase());
    }

    pub fn add_org(&mut self, name: &str) {
        let parts: Vec<String> = name.split_whitespace().map(|p| p.to_lowercase()).collect();
        if !parts.is_empty() {
            self.org_names.push(parts);
        }
    }

    pub fn add_misc(&mut self, name: &str) {
        let parts: Vec<String> = name.split_whitespace().map(|p| p.to_lowercase()).collect();
        if !parts.is_empty() {
            self.misc_names.push(parts);
        }
    }

    /// Aplica todas as regras à sequência de tokens
    ///
    /// Retorna um mapa token_index → RuleMatch para os tokens identificados
    pub fn apply(&self, tokens: &[Token]) -> Vec<Option<RuleMatch>> {
        let mut result: Vec<Option<RuleMatch>> = vec![None; tokens.len()];

        // 1. Gazetteers de pessoa (token único)
        for (i, token) in tokens.iter().enumerate() {
            let lower = token.text.to_lowercase();
            if self.person_names.contains(&lower) {
                result[i] = Some(RuleMatch {
                    token_index: i,
                    tag: if result
                        .get(i.wrapping_sub(1))
                        .and_then(|r| r.as_ref())
                        .map(|r| matches!(r.tag, Tag::Begin(EntityCategory::Per) | Tag::Inside(EntityCategory::Per)))
                        .unwrap_or(false)
                    {
                        Tag::Inside(EntityCategory::Per)
                    } else {
                        Tag::Begin(EntityCategory::Per)
                    },
                    rule_name: "person_gazetteer".to_string(),
                    confidence: 0.92,
                });
            }
        }

        // 2. Gazetteers de localização (token único)
        for (i, token) in tokens.iter().enumerate() {
            if result[i].is_some() {
                continue;
            }
            let lower = token.text.to_lowercase();
            if self.location_names.contains(&lower) {
                result[i] = Some(RuleMatch {
                    token_index: i,
                    tag: Tag::Begin(EntityCategory::Loc),
                    rule_name: "location_gazetteer".to_string(),
                    confidence: 0.90,
                });
            }
        }

        // 3. Gazetteers de organização (n-gramas)
        'outer_org: for (i, _) in tokens.iter().enumerate() {
            if result[i].is_some() {
                continue;
            }
            for org_parts in &self.org_names {
                if i + org_parts.len() <= tokens.len() {
                    let matches = org_parts.iter().enumerate().all(|(j, part)| {
                        tokens[i + j].text.to_lowercase() == *part
                    });
                    if matches {
                        result[i] = Some(RuleMatch {
                            token_index: i,
                            tag: Tag::Begin(EntityCategory::Org),
                            rule_name: "org_gazetteer".to_string(),
                            confidence: 0.93,
                        });
                        for j in 1..org_parts.len() {
                            if i + j < result.len() {
                                result[i + j] = Some(RuleMatch {
                                    token_index: i + j,
                                    tag: Tag::Inside(EntityCategory::Org),
                                    rule_name: "org_gazetteer".to_string(),
                                    confidence: 0.93,
                                });
                            }
                        }
                        continue 'outer_org;
                    }
                }
            }
        }

        // 4. Gazetteers de misc (n-gramas)
        'outer_misc: for (i, _) in tokens.iter().enumerate() {
            if result[i].is_some() {
                continue;
            }
            for misc_parts in &self.misc_names {
                if i + misc_parts.len() <= tokens.len() {
                    let matches = misc_parts.iter().enumerate().all(|(j, part)| {
                        tokens[i + j].text.to_lowercase() == *part
                    });
                    if matches {
                        result[i] = Some(RuleMatch {
                            token_index: i,
                            tag: Tag::Begin(EntityCategory::Misc),
                            rule_name: "misc_gazetteer".to_string(),
                            confidence: 0.88,
                        });
                        for j in 1..misc_parts.len() {
                            if i + j < result.len() {
                                result[i + j] = Some(RuleMatch {
                                    token_index: i + j,
                                    tag: Tag::Inside(EntityCategory::Misc),
                                    rule_name: "misc_gazetteer".to_string(),
                                    confidence: 0.88,
                                });
                            }
                        }
                        continue 'outer_misc;
                    }
                }
            }
        }

        // 5. Regra de título: "Presidente X" → X é PER
        for i in 0..tokens.len().saturating_sub(1) {
            if result[i + 1].is_some() {
                continue;
            }
            let lower = tokens[i].text.to_lowercase();
            if self.person_titles.contains(&lower) {
                let next = &tokens[i + 1];
                let next_first_upper = next
                    .text
                    .chars()
                    .next()
                    .map(|c| c.is_uppercase())
                    .unwrap_or(false);
                if next_first_upper {
                    result[i + 1] = Some(RuleMatch {
                        token_index: i + 1,
                        tag: Tag::Begin(EntityCategory::Per),
                        rule_name: "title_pattern".to_string(),
                        confidence: 0.80,
                    });
                }
            }
        }

        // 6. Indicadores de organização: "X S.A." → X é ORG
        for i in 1..tokens.len() {
            let lower = tokens[i].text.to_lowercase();
            if self.org_indicators.contains(&lower) && result[i - 1].is_none() {
                let prev = &tokens[i - 1];
                let prev_first_upper = prev
                    .text
                    .chars()
                    .next()
                    .map(|c| c.is_uppercase())
                    .unwrap_or(false);
                if prev_first_upper {
                    result[i - 1] = Some(RuleMatch {
                        token_index: i - 1,
                        tag: Tag::Begin(EntityCategory::Org),
                        rule_name: "org_suffix_pattern".to_string(),
                        confidence: 0.85,
                    });
                    result[i] = Some(RuleMatch {
                        token_index: i,
                        tag: Tag::Inside(EntityCategory::Org),
                        rule_name: "org_suffix_pattern".to_string(),
                        confidence: 0.85,
                    });
                }
            }
        }

        // 7. Regex: CNPJ (padrão XX.XXX.XXX/XXXX-XX → ORG próximo)
        for (i, token) in tokens.iter().enumerate() {
            if is_cnpj(&token.text) && result[i].is_none() {
                result[i] = Some(RuleMatch {
                    token_index: i,
                    tag: Tag::Begin(EntityCategory::Org),
                    rule_name: "cnpj_pattern".to_string(),
                    confidence: 0.99,
                });
            }
        }

        result
    }
}

impl Default for RuleEngine {
    fn default() -> Self {
        Self::new()
    }
}

/// Verifica se um token tem formato de CNPJ brasileiro
fn is_cnpj(s: &str) -> bool {
    let digits: String = s.chars().filter(|c| c.is_numeric()).collect();
    digits.len() == 14
        && (s.contains('.')
            && s.contains('/')
            && s.contains('-'))
}

/// Verifica se um token tem formato de CPF brasileiro
#[allow(dead_code)]
fn is_cpf(s: &str) -> bool {
    let digits: String = s.chars().filter(|c| c.is_numeric()).collect();
    digits.len() == 11 && s.contains('.') && s.contains('-')
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tokenizer::tokenize;

    #[test]
    fn test_person_gazetteer() {
        let mut engine = RuleEngine::new();
        engine.add_person("Lula");

        let tokens = tokenize("Lula ganhou as eleições");
        let matches = engine.apply(&tokens);

        assert!(matches[0].is_some());
        assert_eq!(
            matches[0].as_ref().unwrap().tag,
            Tag::Begin(EntityCategory::Per)
        );
    }

    #[test]
    fn test_title_pattern() {
        let engine = RuleEngine::new();
        let tokens = tokenize("o presidente Lula anunciou medidas");
        let matches = engine.apply(&tokens);

        // "Lula" está depois de "presidente" e é capitalizado
        assert!(matches[2].is_some());
        assert!(
            matches[2].as_ref().unwrap().rule_name == "title_pattern"
        );
    }

    #[test]
    fn test_org_multiword() {
        let mut engine = RuleEngine::new();
        engine.add_org("São Paulo");

        let tokens = tokenize("o clube São Paulo venceu");
        let matches = engine.apply(&tokens);

        // "São" → B-ORG, "Paulo" → I-ORG
        assert!(matches[2].is_some());
        assert_eq!(
            matches[2].as_ref().unwrap().tag,
            Tag::Begin(EntityCategory::Org)
        );
        assert!(matches[3].is_some());
        assert_eq!(
            matches[3].as_ref().unwrap().tag,
            Tag::Inside(EntityCategory::Org)
        );
    }
}
