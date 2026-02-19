//! # Tokenizador para Português Brasileiro
//!
//! Responsável por dividir o texto bruto em tokens individuais (palavras, pontuações).
//! Cada token preserva sua posição original no texto (offset) para permitir
//! destacar entidades na interface web.
//!
//! ## Esquema de Tokenização
//!
//! - Palavras são separadas por espaços e pontuações
//! - Contrações PT-BR são preservadas como palavras individuais
//! - Pontuações são tokens separados
//! - Abreviações comuns ("Dr.", "Sr.", "Gov.") são detectadas e não quebradas

use serde::{Deserialize, Serialize};

/// Um token extraído do texto original.
///
/// Mantém a posição exata (`start`, `end`) para que a interface possa
/// destacar as entidades no texto original sem modificá-lo.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Token {
    /// O texto do token (ex: "Lula", ",", "presidente")
    pub text: String,
    /// Índice de byte inicial no texto original
    pub start: usize,
    /// Índice de byte final (exclusivo) no texto original
    pub end: usize,
    /// Índice do token na sequência (0-based)
    pub index: usize,
}

/// Abreviações comuns em PT-BR que não devem ter o ponto tratado como fim de sentença
const ABBREVIATIONS: &[&str] = &[
    "Dr", "Dra", "Sr", "Sra", "Prof", "Profa", "Gov", "Dep", "Sen", "Min",
    "Gen", "Cap", "Sgt", "Cel", "Brig", "Adm", "Des", "Pres", "Eng", "Arq",
    "km", "cm", "mm", "kg", "mg", "ml", "dl", "ha", "etc", "vol", "núm",
    "art", "pág", "pag", "cap", "tel", "fax", "av", "pg", "ibid", "op",
];

/// Tokeniza um texto em português brasileiro.
///
/// # Algoritmo
///
/// 1. Percorre o texto char a char acumulando caracteres alfanuméricos como tokens
/// 2. Ao encontrar pontuação, fecha o token atual e cria um token de pontuação
/// 3. Trata casos especiais: abreviações (Dr., Sr.), números decimais (1.234,56)
///
/// # Exemplos
///
/// ```rust
/// # use ner_core::tokenizer::tokenize;
/// let tokens = tokenize("Lula ganhou as eleições em 2022.");
/// assert_eq!(tokens.len(), 7); // Lula, ganhou, as, eleições, em, 2022, .
/// ```
pub fn tokenize(text: &str) -> Vec<Token> {
    let mut tokens = Vec::new();
    let mut current_start = 0;
    let mut current_text = String::new();
    let chars: Vec<(usize, char)> = text.char_indices().collect();
    let mut i = 0;

    while i < chars.len() {
        let (byte_pos, ch) = chars[i];

        if ch.is_alphanumeric() || ch == '-' && !current_text.is_empty() {
            if current_text.is_empty() {
                current_start = byte_pos;
            }
            current_text.push(ch);
        } else if ch == '.' && !current_text.is_empty() {
            // Verifica se é abreviação (ex: "Dr.") ou número decimal (ex: "1.234")
            let is_abbrev = ABBREVIATIONS.contains(&current_text.as_str());
            let next_is_upper = chars
                .get(i + 1)
                .map(|(_, c)| c.is_uppercase())
                .unwrap_or(false);
            let next_is_space_then_lower = chars
                .get(i + 1)
                .map(|(_, c)| c.is_whitespace())
                .unwrap_or(false)
                && chars
                    .get(i + 2)
                    .map(|(_, c)| c.is_lowercase())
                    .unwrap_or(false);
            let current_is_num = current_text.chars().all(char::is_numeric);
            let next_is_num = chars
                .get(i + 1)
                .map(|(_, c)| c.is_numeric())
                .unwrap_or(false);

            if is_abbrev || (current_is_num && next_is_num) || next_is_space_then_lower {
                // Inclui o ponto no token atual
                current_text.push('.');
            } else if next_is_upper || is_abbrev {
                // Abreviação seguida de maiúscula: inclui o ponto
                current_text.push('.');
                flush_token(&mut tokens, &mut current_text, current_start, byte_pos + 1);
                i += 1;
                continue;
            } else {
                // Fim de sentença: fecha o token e adiciona o ponto separado
                let end = byte_pos;
                flush_token(&mut tokens, &mut current_text, current_start, end);
                push_token(&mut tokens, ".".to_string(), byte_pos, byte_pos + 1);
            }
        } else if ch == '\'' || ch == '\u{2019}' {
            // Apóstrofo em contrações (ex: d'água) — mantém junto
            if current_text.is_empty() {
                current_start = byte_pos;
            }
            current_text.push(ch);
        } else if ch.is_whitespace() {
            // Espaço: fecha o token atual
            let end = byte_pos;
            flush_token(&mut tokens, &mut current_text, current_start, end);
        } else {
            // Pontuação: fecha o token atual e adiciona a pontuação como token separado
            let end = byte_pos;
            flush_token(&mut tokens, &mut current_text, current_start, end);
            // Calcula o tamanho em bytes do char de pontuação
            let ch_len = ch.len_utf8();
            push_token(&mut tokens, ch.to_string(), byte_pos, byte_pos + ch_len);
        }

        i += 1;
    }

    // Fecha qualquer token restante
    let end = text.len();
    flush_token(&mut tokens, &mut current_text, current_start, end);

    // Atribui índices sequenciais
    for (idx, token) in tokens.iter_mut().enumerate() {
        token.index = idx;
    }

    tokens
}

/// Fecha o token acumulado e adiciona à lista (se não vazio)
fn flush_token(tokens: &mut Vec<Token>, text: &mut String, start: usize, end: usize) {
    if !text.is_empty() {
        let t = Token {
            text: text.clone(),
            start,
            end,
            index: 0, // será atribuído depois
        };
        tokens.push(t);
        text.clear();
    }
}

/// Adiciona um token de pontuação diretamente
fn push_token(tokens: &mut Vec<Token>, text: String, start: usize, end: usize) {
    tokens.push(Token {
        text,
        start,
        end,
        index: 0,
    });
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_tokenize_basic() {
        let tokens = tokenize("Lula ganhou as eleições em 2022.");
        let texts: Vec<&str> = tokens.iter().map(|t| t.text.as_str()).collect();
        assert!(texts.contains(&"Lula"));
        assert!(texts.contains(&"ganhou"));
        assert!(texts.contains(&"2022"));
        assert!(texts.contains(&"."));
    }

    #[test]
    fn test_offsets_correct() {
        let text = "São Paulo é linda";
        let tokens = tokenize(text);
        for token in &tokens {
            assert_eq!(&text[token.start..token.end], token.text.as_str());
        }
    }

    #[test]
    fn test_indices_sequential() {
        let tokens = tokenize("Brasil é o maior país da América do Sul");
        for (i, t) in tokens.iter().enumerate() {
            assert_eq!(t.index, i);
        }
    }

    #[test]
    fn test_empty_text() {
        let tokens = tokenize("");
        assert!(tokens.is_empty());
    }

    #[test]
    fn test_punctuation_separated() {
        let tokens = tokenize("Olá, mundo!");
        let texts: Vec<&str> = tokens.iter().map(|t| t.text.as_str()).collect();
        assert!(texts.contains(&"Olá"));
        assert!(texts.contains(&","));
        assert!(texts.contains(&"mundo"));
        assert!(texts.contains(&"!"));
    }
}
