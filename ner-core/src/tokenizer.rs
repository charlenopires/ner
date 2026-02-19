//! # Tokenizador para Português Brasileiro
//!
//! Responsável por dividir o texto bruto em tokens individuais (palavras, pontuações).
//! Cada token preserva sua posição original no texto (offset) para permitir
//! destacar entidades na interface web.
//!
//! ## Esquema de Tokenização
//!
//! - **Standard**: Palavras separadas por espaços/pontuações. Preserva abreviações comuns.
//! - **CharLevel**: Cada caractere é um token (bom para redes neurais profundas/OOV).
//! - **Aggressive**: Separa sufixos comuns e clíticos (ex: "curou-se" -> "curou", "-", "se").
//! - **Conservative**: Preserva locuções e nomes compostos (ex: "São Paulo").
//! - **BpeLite**: Simulação de BPE baseada em frequência de sub-palavras.
//!
//! ## Exemplo de Uso
//!
//! ```rust
//! use ner_core::tokenizer::{tokenize_with_mode, TokenizerMode};
//!
//! let text = "Dr. Silva curou-se.";
//!
//! // Modo Standard: "Dr.", "Silva", "curou-se", "."
//! let tokens = tokenize_with_mode(text, TokenizerMode::Standard);
//!
//! // Modo Aggressive: "Dr.", "Silva", "curou", "-", "se", "."
//! let aggressive = tokenize_with_mode(text, TokenizerMode::Aggressive);
//! ```

use serde::{Deserialize, Serialize};

/// Um token extraído do texto original.
///
/// O `Token` é a unidade atômica de processamento do pipeline. Ele mantém a referência
/// exata de sua posição no texto original (`start` e `end`), o que é crucial para:
/// 1. Extração de features baseada no texto cru.
/// 2. Destaque (highlight) das entidades na interface gráfica sem alterar a formatação original.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct Token {
    /// O texto do token (ex: "Lula", ",", "presidente").
    pub text: String,
    /// Índice de byte inicial no texto original (inclusive).
    pub start: usize,
    /// Índice de byte final no texto original (exclusivo).
    pub end: usize,
    /// Índice sequencial do token na lista (0, 1, 2...).
    pub index: usize,
}

/// Estratégias de Tokenização disponíveis.
///
/// A escolha do tokenizador impacta diretamente quais "unidades" o modelo verá.
/// Diferentes estratégias podem ser úteis para diferentes tipos de texto ou modelos.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum TokenizerMode {
    /// **Padrão**: Separa por espaços e pontuações, mas preserva abreviações comuns (ex: "Dr.", "Sra.")
    /// e números flutuantes. Ideal para textos formais e jornalísticos.
    Standard,
    /// **Caractere**: Cada caractere é um token. Útil para modelos profundos (Deep Learning) que
    /// aprendem morfologia do zero ou para lidar com muito ruído (erros de digitação).
    CharLevel,
    /// **Agressivo**: Separa clíticos verbais ("curou-se" -> "curou", "-", "se") e sufixos comuns.
    /// Aumenta o vocabulário conhecido pelo modelo ao reduzir palavras à sua raiz.
    Aggressive,
    /// **Conservador**: Mantém entidades conhecidas juntas (ex: "São Paulo" vira um único token).
    /// Facilita para o modelo aprender que aquele bloco é uma entidade única.
    Conservative,
    /// **Sub-word (BPE Lite)**: Simulação didática de Byte-Pair Encoding. Agrupa caracteres frequentes
    /// (ex: "q"+"u"+"e" -> "que"). Reduz o tamanho do vocabulário mantendo partes significativas.
    BpeLite,
}

impl Default for TokenizerMode {
    fn default() -> Self {
        TokenizerMode::Standard
    }
}

/// Abreviações comuns em PT-BR que não devem ter o ponto tratado como fim de sentença
const ABBREVIATIONS: &[&str] = &[
    "Dr", "Dra", "Sr", "Sra", "Prof", "Profa", "Gov", "Dep", "Sen", "Min",
    "Gen", "Cap", "Sgt", "Cel", "Brig", "Adm", "Des", "Pres", "Eng", "Arq",
    "km", "cm", "mm", "kg", "mg", "ml", "dl", "ha", "etc", "vol", "núm",
    "art", "pág", "pag", "cap", "tel", "fax", "av", "pg", "ibid", "op",
];

/// Sufixos e clíticos para o modo Aggressive
const CLITICS: &[&str] = &["-se", "-nos", "-lhe", "-lhes", "-me", "-te", "-o", "-a", "-los", "-las"];
const SUFFIXES: &[&str] = &["mente", "ção", "ções", "ista", "ismo", "dade"];

/// Locuções comuns para o modo Conservative
const COMPOUNDS: &[&str] = &[
    "são paulo", "rio de janeiro", "minas gerais", "espírito santo",
    "mato grosso", "mato grosso do sul", "rio grande do sul", "rio grande do norte",
    "estados unidos", "reino unido", "nova iorque", "sem teto", "pôr do sol",
];

/// Tokeniza um texto usando o algoritmo padrão (compatibilidade).
pub fn tokenize(text: &str) -> Vec<Token> {
    tokenize_with_mode(text, TokenizerMode::Standard)
}

/// Tokeniza um texto com o modo especificado.
pub fn tokenize_with_mode(text: &str, mode: TokenizerMode) -> Vec<Token> {
    let mut tokens = match mode {
        TokenizerMode::CharLevel => tokenize_char_level(text),
        TokenizerMode::Aggressive => tokenize_aggressive(text),
        TokenizerMode::Conservative => tokenize_conservative(text),
        TokenizerMode::BpeLite => tokenize_bpe_lite(text),
        TokenizerMode::Standard => tokenize_standard(text),
    };

    // Re-indexa os tokens
    for (i, token) in tokens.iter_mut().enumerate() {
        token.index = i;
    }
    tokens
}

fn tokenize_char_level(text: &str) -> Vec<Token> {
    text.char_indices()
        .map(|(i, c)| Token {
            text: c.to_string(),
            start: i,
            end: i + c.len_utf8(),
            index: 0,
        })
        .collect()
}

fn tokenize_aggressive(text: &str) -> Vec<Token> {
    // Primeiro tokeniza standard, depois pós-processa
    let standard_tokens = tokenize_standard(text);
    let mut expanded_tokens = Vec::new();

    for token in standard_tokens {
        // Verifica clíticos (ex: encontrou-se)
        let mut handled = false;
        
        // Separação de clíticos com hífen
        if let Some((base, clitic)) = token.text.rsplit_once('-') {
             // Reconstrói o clítico com hífen para checar na lista (ex: "-se")
            let clitic_with_hyphen = format!("-{}", clitic);
            if CLITICS.contains(&clitic_with_hyphen.as_str()) && !base.is_empty() {
                // Split: base, "-", clitic
                let base_len = base.len();
                let hyphen_len = 1; // assumindo 1 byte '-'
                
                // Base
                expanded_tokens.push(Token {
                    text: base.to_string(),
                    start: token.start,
                    end: token.start + base_len,
                    index: 0,
                });
                // Hífen
                expanded_tokens.push(Token {
                    text: "-".to_string(),
                    start: token.start + base_len,
                    end: token.start + base_len + hyphen_len,
                    index: 0,
                });
                // Clítico
                expanded_tokens.push(Token {
                    text: clitic.to_string(),
                    start: token.start + base_len + hyphen_len,
                    end: token.end,
                    index: 0,
                });
                handled = true;
            }
        }
        
        if !handled {
            // Tenta separar sufixos conhecidos (ex: rapida+mente)
            // Apenas se a palavra for longa o suficiente
            let mut suffix_handled = false;
            // Verifica apenas palavras alfabéticas
            if token.text.len() > 6 && token.text.chars().all(char::is_alphabetic) {
                 for &suffix in SUFFIXES {
                     if token.text.ends_with(suffix) {
                         let split_idx = token.text.len() - suffix.len();
                         let (base, suf) = token.text.split_at(split_idx);
                         
                         // Base
                         expanded_tokens.push(Token {
                             text: base.to_string(),
                             start: token.start,
                             end: token.start + base.len(),
                             index: 0,
                         });
                         // Sufixo (marcado com + para visualização, mas texto original preservado na teoria)
                         // Aqui vamos apenas quebrar
                         expanded_tokens.push(Token {
                             text: suf.to_string(),
                             start: token.start + base.len(),
                             end: token.end,
                             index: 0,
                         });
                         suffix_handled = true;
                         break;
                     }
                 }
            }
            
            if !suffix_handled {
                expanded_tokens.push(token);
            }
        }
    }
    
    expanded_tokens
}

fn tokenize_conservative(text: &str) -> Vec<Token> {
    let standard = tokenize_standard(text);
    if standard.is_empty() { return standard; }

    let mut merged = Vec::new();
    let mut i = 0;
    
    while i < standard.len() {
        // Tenta encontrar o maior match de locução começando em i
        let mut best_match_len = 0;
        
        // Verifica até 4 tokens à frente (ex: "Rio", "Grande", "do", "Sul")
        for window in 2..=5 {
            if i + window > standard.len() { break; }
            
            let candidate_slice = &standard[i..i+window];
            // Verifica se os tokens são adjacentes no texto original
            let is_adjacent = candidate_slice.windows(2).all(|w| w[1].start == w[0].end || 
                (w[1].start > w[0].end && text[w[0].end..w[1].start].trim().is_empty()));
             
             if is_adjacent {
                 let combined_text = candidate_slice.iter().map(|t| t.text.as_str()).collect::<Vec<_>>().join(" ");
                 if COMPOUNDS.contains(&combined_text.to_lowercase().as_str()) {
                     best_match_len = window;
                 }
             }
        }
        
        if best_match_len > 0 {
            // Cria token mergeado
            let first = &standard[i];
            let last = &standard[i + best_match_len - 1];
            merged.push(Token {
                text: text[first.start..last.end].to_string(),
                start: first.start,
                end: last.end,
                index: 0,
            });
            i += best_match_len;
        } else {
            merged.push(standard[i].clone());
            i += 1;
        }
    }
    
    merged
}

fn tokenize_bpe_lite(text: &str) -> Vec<Token> {
    // Simulação simplificada de BPE:
    // 1. Quebra em caracteres
    // 2. Faz merges de pares frequentes conhecidos (hardcoded para demonstração)
    let mut tokens = tokenize_char_level(text);
    
    // Pares para merge (ordem importa: prioridade)
    let merges = &[
        ("e", "s"), ("a", "s"), ("o", "s"), // plurais
        ("d", "e"), ("d", "o"), ("d", "a"), // preposições
        ("q", "u"), ("u", "e"), ("e", "m"), // "que", "em"
        ("ã", "o"), ("ç", "a"), ("ç", "o"), // nasais/cedilha
        ("r", "e"), ("i", "n"), ("t", "e"), // prefixos/sufixos
    ];
    
    // Aplica N passadas de merge
    for _ in 0..3 {
        let mut new_tokens = Vec::new();
        let mut i = 0;
        while i < tokens.len() {
            if i + 1 < tokens.len() {
                let t1 = &tokens[i];
                let t2 = &tokens[i+1];
                
                // Só merge se forem adjacentes
                if t1.end == t2.start {
                    let pair = (t1.text.as_str(), t2.text.as_str());
                    if merges.contains(&pair) {
                        new_tokens.push(Token {
                            text: format!("{}{}", t1.text, t2.text),
                            start: t1.start,
                            end: t2.end,
                            index: 0,
                        });
                        i += 2;
                        continue;
                    }
                }
            }
            new_tokens.push(tokens[i].clone());
            i += 1;
        }
        tokens = new_tokens;
    }
    
    tokens
}

fn tokenize_standard(text: &str) -> Vec<Token> {
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
            // Verifica se é abreviação (ex: "Dr.")
            let is_abbrev = ABBREVIATIONS.contains(&current_text.as_str());
            // Lógica simplificada para número (ex: 1.234)
            let current_is_num = current_text.chars().all(char::is_numeric);
             let next_is_num = chars
                .get(i + 1)
                .map(|(_, c)| c.is_numeric())
                .unwrap_or(false);
            
            // Check for next char logic for abbreviations (like Next is uppercase)
             let next_is_upper = chars
                .get(i + 1)
                .map(|(_, c)| c.is_uppercase())
                .unwrap_or(false);

            if is_abbrev || (current_is_num && next_is_num) {
                current_text.push('.');
            } else if is_abbrev && next_is_upper {
                 // Logic from original: abbr followed by upper -> keep dot
                 current_text.push('.');
            } else {
                // Termina token atual
                let end = byte_pos;
                flush_token(&mut tokens, &mut current_text, current_start, end);
                // Ponto separado
                push_token(&mut tokens, ".".to_string(), byte_pos, byte_pos + 1);
            }
        } else if ch == '\'' || ch == '\u{2019}' {
             if current_text.is_empty() { current_start = byte_pos; }
             current_text.push(ch);
        } else if ch.is_whitespace() {
            let end = byte_pos;
            flush_token(&mut tokens, &mut current_text, current_start, end);
        } else {
            let end = byte_pos;
            flush_token(&mut tokens, &mut current_text, current_start, end);
            let ch_len = ch.len_utf8();
            push_token(&mut tokens, ch.to_string(), byte_pos, byte_pos + ch_len);
        }
        i += 1;
    }
    
    let end = text.len();
    flush_token(&mut tokens, &mut current_text, current_start, end);

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
    fn test_tokenize_standard_basic() {
        let tokens = tokenize("Lula ganhou 2022.");
        assert_eq!(tokens.len(), 4);
    }
    
    #[test]
    fn test_tokenize_char_level() {
        let tokens = tokenize_with_mode("Oi", TokenizerMode::CharLevel);
        assert_eq!(tokens.len(), 2);
        assert_eq!(tokens[0].text, "O");
        assert_eq!(tokens[1].text, "i");
    }

    #[test]
    fn test_tokenize_aggressive() {
        let tokens = tokenize_with_mode("curou-se rapidamente", TokenizerMode::Aggressive);
        // curou-se -> curou, -, se (3)
        // rapidamente -> rapida, mente (2)
        // total 5 tokens (se ignorar espaço)
        let texts: Vec<&str> = tokens.iter().map(|t| t.text.as_str()).collect();
        assert!(texts.contains(&"curou"));
        assert!(texts.contains(&"-"));
        assert!(texts.contains(&"se"));
        assert!(texts.contains(&"rapida"));
        assert!(texts.contains(&"mente"));
    }

    #[test]
    fn test_tokenize_conservative() {
        let tokens = tokenize_with_mode("Visitei São Paulo ontem.", TokenizerMode::Conservative);
        let texts: Vec<&str> = tokens.iter().map(|t| t.text.as_str()).collect();
        // Espera-se "São Paulo" junto
        assert!(texts.contains(&"São Paulo"));
    }
    
    #[test]
    fn test_tokenize_bpe_lite() {
        // "que" -> "q"+"u"+"e" -> "qu"+"e" -> "que" (depende da ordem)
        // merges: q+u, u+e ...
        let tokens = tokenize_with_mode("quem", TokenizerMode::BpeLite);
        // q, u, e, m -> qu, e, m -> que, m -> quem (se tiver e+m)
        let texts: Vec<&str> = tokens.iter().map(|t| t.text.as_str()).collect();
        // Verificar se houve algum merge
        assert!(tokens.len() < 4); 
    }
}
