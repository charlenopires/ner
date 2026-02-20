//! # Engenharia de Features para NER
//!
//! Para cada token, extrai um vetor de features binárias que o CRF utiliza
//! para calcular probabilidades de tags. Features capuram informações
//! ortográficas, lexicais e contextuais.
//!
//! ## Features Implementadas
//!
//! ### Features do token atual
//! - Forma da palavra (lowercase)
//! - Capitalização: IsCapitalized, IsAllCaps, IsMixed
//! - Prefixos de 2, 3 e 4 caracteres
//! - Sufixos de 2, 3 e 4 caracteres
//! - Contém dígitos, hífens, pontos
//! - É apenas dígito
//!
//! ### Features de contexto (janela de 2 tokens)
//! - Palavra anterior e posterior
//! - Tag da palavra anterior (para features de transição)
//!
//! ### Features de Gazetteer
//! - Pertence à lista de nomes de pessoas
//! - Pertence à lista de cidades/estados
//! - Pertence à lista de organizações

use std::collections::{HashMap, HashSet};

use serde::{Deserialize, Serialize};

use crate::tokenizer::Token;

/// Estrutura para representar as características de um token.
///
/// Utilizamos um mapa esparso (`HashMap<String, f64>`) porque o espaço de features é potencialmente
/// infinito (ex: "word=abacaxi", "suffix3=axi"), mas cada token ativa apenas um pequeno subconjunto.
///
/// # Por que f64?
/// Embora a maioria das features sejam binárias (0.0 ou 1.0), usar `f64` permite:
/// - Features contínuas (ex: TF-IDF, embeddings).
/// - Operações vetoriais eficientes (produto escalar).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FeatureVector {
    /// O mapa de features ativas. Ex: `{"is_capitalized": 1.0, "word=Brasil": 1.0}`.
    pub features: HashMap<String, f64>,
    /// Referência ao índice do token original na sentença.
    pub token_index: usize,
}

impl FeatureVector {
    pub fn new(token_index: usize) -> Self {
        Self {
            features: HashMap::new(),
            token_index,
        }
    }

    /// Adiciona uma feature ao vetor com valor 1.0 (binária) ou customizado.
    pub fn insert(&mut self, key: impl Into<String>, value: f64) {
        self.features.insert(key.into(), value);
    }

    /// Calcula o produto escalar (dot product) com um vetor de pesos.
    ///
    /// $$ \text{score} = \sum (w_i \cdot f_i) $$
    ///
    /// Usado pelo Perceptron e CRF para calcular a "força" de ativação de uma tag.
    pub fn dot(&self, weights: &HashMap<String, f64>) -> f64 {
        self.features
            .iter()
            .map(|(k, v)| v * weights.get(k).unwrap_or(&0.0))
            .sum()
    }
}

/// Listas de gazetteer compiladas a partir do corpus PT-BR
#[derive(Debug, Clone)]
pub struct Gazetteers {
    pub persons: HashSet<String>,
    pub locations: HashSet<String>,
    pub organizations: HashSet<String>,
    pub misc: HashSet<String>,
}

impl Gazetteers {
    pub fn new() -> Self {
        Self {
            persons: HashSet::new(),
            locations: HashSet::new(),
            organizations: HashSet::new(),
            misc: HashSet::new(),
        }
    }
}

impl Default for Gazetteers {
    fn default() -> Self {
        Self::new()
    }
}

/// Gera vetores de features para toda a sequência de tokens.
///
/// Esta função orquestra a chamada de `extract_for_token` para cada posição,
/// garantindo que o contexto (janela deslizante) seja respeitado.
///
/// # Parâmetros
/// - `tokens`: A lista completa de tokens da sentença.
/// - `gazetteers`: Acesso rápido a listas de entidades conhecidas (O(1)).
///
/// # Retorno
/// Retorna um `Vec<FeatureVector>` alinhado com os tokens de entrada.
/// O índice `i` do retorno corresponde ao token `i` da entrada.
///
/// # Exemplo
/// Para "O Brasil venceu", o vetor do índice 1 ("Brasil") conterá:
/// - `word=brasil`
/// - `is_capitalized`
/// - `prev_word=o`
/// - `next_word=venceu`
/// - `in_location_gazetteer` (se estiver no gazetteer)
pub fn extract_features(tokens: &[Token], gazetteers: &Gazetteers) -> Vec<FeatureVector> {
    tokens
        .iter()
        .enumerate()
        .map(|(i, _)| extract_for_token(tokens, i, gazetteers))
        .collect()
}

/// Extrai features para um único token em seu contexto
///
/// Implementa a lógica detalhada de extração, cobrindo:
/// 1. **Morfologia**: Sufixos, prefixos, capitalização.
/// 2. **Contexto**: Palavras vizinhas (unigramas e bigramas).
/// 3. **Conhecimento Externo**: Verificação em gazetteers.
/// 4. **Posição**: Se é início ou fim de frase.
pub fn extract_for_token(tokens: &[Token], i: usize, gazetteers: &Gazetteers) -> FeatureVector {
    let mut fv = FeatureVector::new(i);
    let token = &tokens[i];
    let word = &token.text;
    let lower = word.to_lowercase();

    // === Features da palavra atual ===
    fv.insert(format!("word={lower}"), 1.0);
    fv.insert("bias", 1.0);

    // Capitalização
    let first_char_upper = word.chars().next().map(|c| c.is_uppercase()).unwrap_or(false);
    let all_upper = word.chars().all(|c| c.is_uppercase() || !c.is_alphabetic());
    let has_upper_in_middle = word.chars().skip(1).any(|c| c.is_uppercase());

    if first_char_upper {
        fv.insert("is_capitalized", 1.0);
    }
    if all_upper && word.len() > 1 {
        fv.insert("is_all_caps", 1.0);
    }
    if has_upper_in_middle {
        fv.insert("is_mixed_case", 1.0);
    }

    // Prefixos e sufixos
    let chars: Vec<char> = word.chars().collect();
    for n in 2..=4 {
        if chars.len() >= n {
            let prefix: String = chars[..n].iter().collect();
            let suffix: String = chars[chars.len() - n..].iter().collect();
            fv.insert(format!("prefix{n}={}", prefix.to_lowercase()), 1.0);
            fv.insert(format!("suffix{n}={}", suffix.to_lowercase()), 1.0);
        }
    }

    // Padrões numéricos e de pontuação
    if word.chars().all(char::is_numeric) {
        fv.insert("is_digit", 1.0);
    }
    if word.contains('-') {
        fv.insert("has_hyphen", 1.0);
    }
    if word.contains('.') {
        fv.insert("has_period", 1.0);
    }
    if word.len() == 1 && !word.chars().next().unwrap().is_alphanumeric() {
        fv.insert("is_punctuation", 1.0);
    }

    // Posição na sequência
    if i == 0 {
        fv.insert("is_first", 1.0);
    }
    if i == tokens.len() - 1 {
        fv.insert("is_last", 1.0);
    }

    // === Features de contexto ===

    // Token anterior
    if i > 0 {
        let prev = &tokens[i - 1];
        fv.insert(format!("prev_word={}", prev.text.to_lowercase()), 1.0);
        let prev_first_upper = prev
            .text
            .chars()
            .next()
            .map(|c| c.is_uppercase())
            .unwrap_or(false);
        if prev_first_upper {
            fv.insert("prev_is_capitalized", 1.0);
        }
    } else {
        fv.insert("BOS", 1.0); // Beginning Of Sentence
    }

    // Token dois posições antes
    if i > 1 {
        let prev2 = &tokens[i - 2];
        fv.insert(format!("prev2_word={}", prev2.text.to_lowercase()), 1.0);
    }

    // Token seguinte
    if i + 1 < tokens.len() {
        let next = &tokens[i + 1];
        fv.insert(format!("next_word={}", next.text.to_lowercase()), 1.0);
        let next_first_upper = next
            .text
            .chars()
            .next()
            .map(|c| c.is_uppercase())
            .unwrap_or(false);
        if next_first_upper {
            fv.insert("next_is_capitalized", 1.0);
        }
    } else {
        fv.insert("EOS", 1.0); // End Of Sentence
    }

    // Token dois posições depois
    if i + 2 < tokens.len() {
        let next2 = &tokens[i + 2];
        fv.insert(format!("next2_word={}", next2.text.to_lowercase()), 1.0);
    }

    // Bigramas de contexto
    if i > 0 && i + 1 < tokens.len() {
        let bigram = format!(
            "bigram={}_{}",
            tokens[i - 1].text.to_lowercase(),
            tokens[i + 1].text.to_lowercase()
        );
        fv.insert(bigram, 1.0);
    }

    // === Features de Gazetteer ===
    let word_lower = word.to_lowercase();

    if gazetteers.persons.contains(&word_lower)
        || gazetteers.persons.contains(word.as_str())
    {
        fv.insert("in_person_gazetteer", 1.0);
    }
    if gazetteers.locations.contains(&word_lower)
        || gazetteers.locations.contains(word.as_str())
    {
        fv.insert("in_location_gazetteer", 1.0);
    }
    if gazetteers.organizations.contains(&word_lower)
        || gazetteers.organizations.contains(word.as_str())
    {
        fv.insert("in_org_gazetteer", 1.0);
    }
    if gazetteers.misc.contains(&word_lower) || gazetteers.misc.contains(word.as_str()) {
        fv.insert("in_misc_gazetteer", 1.0);
    }

    fv
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tokenizer::tokenize;

    #[test]
    fn test_capitalization_feature() {
        let tokens = tokenize("Lula é presidente");
        let gaz = Gazetteers::default();
        let features = extract_features(&tokens, &gaz);

        // "Lula" é capitalizado
        assert_eq!(features[0].features.get("is_capitalized"), Some(&1.0));
        // "é" não é capitalizado
        assert!(features[1].features.get("is_capitalized").is_none());
    }

    #[test]
    fn test_prefix_suffix_features() {
        let tokens = tokenize("Petrobras");
        let gaz = Gazetteers::default();
        let features = extract_features(&tokens, &gaz);

        assert!(features[0].features.contains_key("prefix2=pe"));
        assert!(features[0].features.contains_key("suffix3=ras"));
    }

    #[test]
    fn test_context_features() {
        let tokens = tokenize("o presidente Lula anunciou");
        let gaz = Gazetteers::default();
        let features = extract_features(&tokens, &gaz);

        // Features de "Lula" (índice 2) devem ter prev_word=presidente
        let lula_features = &features[2].features;
        assert!(lula_features.contains_key("prev_word=presidente"));
        assert!(lula_features.contains_key("next_word=anunciou"));
    }

    #[test]
    fn test_gazetteer_feature() {
        let tokens = tokenize("Brasília é bonita");
        let mut gaz = Gazetteers::default();
        gaz.locations.insert("brasília".to_string());

        let features = extract_features(&tokens, &gaz);
        assert_eq!(
            features[0].features.get("in_location_gazetteer"),
            Some(&1.0)
        );
    }
}
