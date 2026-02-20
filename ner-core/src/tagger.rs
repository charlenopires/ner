//! # Esquema de Tags BIO e Tipos de Entidade
//!
//! Define o esquema de anotação **BIO** (Beginning-Inside-Outside) utilizado
//! para rotular tokens no reconhecimento de entidades nomeadas.
//!
//! ## Categorias de Entidades
//!
//! | Prefixo | Significado         | Exemplos                          |
//! |---------|---------------------|-----------------------------------|
//! | PER     | Pessoa              | Lula, Bolsonaro, Pelé             |
//! | ORG     | Organização         | Petrobras, Embraer, FIFA          |
//! | LOC     | Local/Geográfico    | São Paulo, Amazônia, Brasil       |
//! | MISC    | Miscelânea          | Copa do Mundo, PIB, COVID-19      |
//! | O       | Fora de entidade    | (qualquer palavra não-entidade)   |
//!
//! ## Esquema BIO
//!
//! - `B-TAG`: Begin — primeiro token de uma entidade
//! - `I-TAG`: Inside — tokens subsequentes da mesma entidade
//! - `O`: Outside — não é parte de nenhuma entidade

use serde::{Deserialize, Serialize};

use crate::tokenizer::Token;

/// Categorias de entidade reconhecidas pelo sistema NER.
///
/// Estas categorias definem o "vocabulário" semântico do modelo.
/// Adicionar novas categorias exigiria retreinar o modelo e atualizar o corpus.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum EntityCategory {
    /// **Pessoa**: Nomes de humanos reais, fictícios ou grupos musicais. Ex: "Machado de Assis", "Beatles".
    Per,
    /// **Organização**: Empresas, instituições, órgãos públicos, times. Ex: "Google", "STF", "Flamengo".
    Org,
    /// **Localização**: Países, cidades, estados, rios, montanhas. Ex: "Brasil", "Tietê", "Everest".
    Loc,
    /// **Miscelânea**: O que não se encaixa nas anteriores (eventos, obras de arte, leis). Ex: "Copa 2014", "Lei Áurea".
    Misc,
}

impl EntityCategory {
    /// Nome da categoria como string (para serialização e UI)
    pub fn name(&self) -> &'static str {
        match self {
            EntityCategory::Per => "PER",
            EntityCategory::Org => "ORG",
            EntityCategory::Loc => "LOC",
            EntityCategory::Misc => "MISC",
        }
    }

    /// Cor CSS para highlight na UI
    pub fn color(&self) -> &'static str {
        match self {
            EntityCategory::Per => "#3b82f6",  // azul
            EntityCategory::Org => "#10b981",  // verde esmeralda
            EntityCategory::Loc => "#f59e0b",  // âmbar
            EntityCategory::Misc => "#8b5cf6", // violeta
        }
    }

    /// Ícone emoji para a categoria
    pub fn icon(&self) -> &'static str {
        match self {
            EntityCategory::Per => "👤",
            EntityCategory::Org => "🏢",
            EntityCategory::Loc => "📍",
            EntityCategory::Misc => "🔖",
        }
    }

    /// Tenta parsear a partir de string (ex: "PER" → Some(Per))
    pub fn from_str(s: &str) -> Option<Self> {
        match s {
            "PER" => Some(EntityCategory::Per),
            "ORG" => Some(EntityCategory::Org),
            "LOC" => Some(EntityCategory::Loc),
            "MISC" => Some(EntityCategory::Misc),
            _ => None,
        }
    }
}

/// Tag BIO aplicada a um token.
///
/// O esquema BIO permite representar entidades de múltiplos tokens.
/// O modelo preverá uma dessas tags para cada palavra.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum Tag {
    /// **Begin**: Marca o INÍCIO de uma entidade. Ex: **São** (B-LOC) Paulo.
    Begin(EntityCategory),
    /// **Inside**: Marca a CONTINUAÇÃO de uma entidade. Ex: São **Paulo** (I-LOC).
    Inside(EntityCategory),
    /// **Outside**: O token não faz parte de nenhuma entidade.
    Outside,
}

impl Tag {
    /// Representação textual da tag (ex: "B-PER", "I-ORG", "O")
    pub fn label(&self) -> String {
        match self {
            Tag::Begin(cat) => format!("B-{}", cat.name()),
            Tag::Inside(cat) => format!("I-{}", cat.name()),
            Tag::Outside => "O".to_string(),
        }
    }

    /// Índice numérico da tag para matrizes CRF/Viterbi.
    /// Mapeia cada possibilidade para um inteiro 0..8.
    pub fn index(&self) -> usize {
        match self {
            Tag::Outside => 0,
            Tag::Begin(EntityCategory::Per) => 1,
            Tag::Inside(EntityCategory::Per) => 2,
            Tag::Begin(EntityCategory::Org) => 3,
            Tag::Inside(EntityCategory::Org) => 4,
            Tag::Begin(EntityCategory::Loc) => 5,
            Tag::Inside(EntityCategory::Loc) => 6,
            Tag::Begin(EntityCategory::Misc) => 7,
            Tag::Inside(EntityCategory::Misc) => 8,
        }
    }

    /// Número total de tags possíveis
    pub const COUNT: usize = 9;

    /// Todas as tags em ordem (para iteração)
    pub fn all() -> [Tag; 9] {
        [
            Tag::Outside,
            Tag::Begin(EntityCategory::Per),
            Tag::Inside(EntityCategory::Per),
            Tag::Begin(EntityCategory::Org),
            Tag::Inside(EntityCategory::Org),
            Tag::Begin(EntityCategory::Loc),
            Tag::Inside(EntityCategory::Loc),
            Tag::Begin(EntityCategory::Misc),
            Tag::Inside(EntityCategory::Misc),
        ]
    }

    /// Retorna a categoria desta tag (se for B- ou I-)
    pub fn category(&self) -> Option<EntityCategory> {
        match self {
            Tag::Begin(c) | Tag::Inside(c) => Some(*c),
            Tag::Outside => None,
        }
    }

    /// Verifica se a transição tag_prev → self é válida no esquema BIO
    ///
    /// Regras:
    /// - `I-X` só pode seguir `B-X` ou `I-X` (mesma categoria)
    /// - `B-X` pode seguir qualquer tag
    /// - `O` pode seguir qualquer tag
    pub fn is_valid_transition(prev: &Tag, next: &Tag) -> bool {
        match next {
            Tag::Inside(cat) => match prev {
                Tag::Begin(prev_cat) | Tag::Inside(prev_cat) => prev_cat == cat,
                _ => false,
            },
            _ => true,
        }
    }

    /// Parseia uma tag a partir de string (ex: "B-PER" → Begin(Per))
    pub fn from_label(s: &str) -> Option<Self> {
        if s == "O" {
            return Some(Tag::Outside);
        }
        let parts: Vec<&str> = s.splitn(2, '-').collect();
        if parts.len() != 2 {
            return None;
        }
        let cat = EntityCategory::from_str(parts[1])?;
        match parts[0] {
            "B" => Some(Tag::Begin(cat)),
            "I" => Some(Tag::Inside(cat)),
            _ => None,
        }
    }
}

impl std::fmt::Display for Tag {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.label())
    }
}

/// Um token com sua tag BIO e probabilidade de confiança
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TaggedToken {
    pub token: Token,
    pub tag: Tag,
    /// Probabilidade/confiança desta atribuição (0.0 a 1.0)
    pub confidence: f64,
}

/// Uma entidade identificada no texto (spans de múltiplos tokens)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EntitySpan {
    /// Texto da entidade (ex: "São Paulo")
    pub text: String,
    /// Categoria da entidade
    pub category: EntityCategory,
    /// Índice do primeiro token
    pub start_token: usize,
    /// Índice do último token (inclusivo)
    pub end_token: usize,
    /// Posição de byte inicial no texto original
    pub start: usize,
    /// Posição de byte final no texto original
    pub end: usize,
    /// Confiança média dos tokens
    pub confidence: f64,
    /// Fonte: foi identificada por "rule" ou "crf"
    pub source: String,
}

/// Converte uma sequência de tokens classificados (BIO) em spans de entidades.
///
/// Implementa a máquina de estados finita do esquema BIO para reconstruir as entidades completas:
/// - Inicia uma nova entidade ao encontrar `B-XXX`.
/// - Continua a entidade enquanto encontrar `I-XXX` da **mesma** categoria.
/// - Finaliza a entidade ao encontrar `O`, `B-YYY` ou `I-YYY` (de outra categoria).
///
/// Este passo é fundamental para transformar a saída "token a token" do modelo
/// em objetos estruturados úteis para a aplicação final.
///
/// # Exemplo
/// `[B-PER, I-PER, O, B-LOC]` -> `[EntitySpan(PER), EntitySpan(LOC)]`
pub fn tokens_to_spans(tagged: &[TaggedToken], original_text: &str) -> Vec<EntitySpan> {
    let mut spans = Vec::new();
    let mut i = 0;

    while i < tagged.len() {
        if let Tag::Begin(cat) = &tagged[i].tag {
            let cat = *cat;
            let start_token = tagged[i].token.index;
            let start_byte = tagged[i].token.start;
            let mut end_token = start_token;
            let mut end_byte = tagged[i].token.end;
            let mut conf_sum = tagged[i].confidence;
            let mut count = 1usize;

            // Acumula tokens I-XXX consecutivos da mesma categoria
            let mut j = i + 1;
            while j < tagged.len() {
                if let Tag::Inside(next_cat) = &tagged[j].tag {
                    if *next_cat == cat {
                        end_token = tagged[j].token.index;
                        end_byte = tagged[j].token.end;
                        conf_sum += tagged[j].confidence;
                        count += 1;
                        j += 1;
                        continue;
                    }
                }
                break;
            }

            let entity_text = original_text[start_byte..end_byte].trim().to_string();
            spans.push(EntitySpan {
                text: entity_text,
                category: cat,
                start_token,
                end_token,
                start: start_byte,
                end: end_byte,
                confidence: conf_sum / count as f64,
                source: "crf".to_string(),
            });

            i = j;
        } else {
            i += 1;
        }
    }

    spans
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_tag_labels() {
        assert_eq!(Tag::Outside.label(), "O");
        assert_eq!(Tag::Begin(EntityCategory::Per).label(), "B-PER");
        assert_eq!(Tag::Inside(EntityCategory::Loc).label(), "I-LOC");
    }

    #[test]
    fn test_valid_transitions() {
        assert!(Tag::is_valid_transition(
            &Tag::Begin(EntityCategory::Per),
            &Tag::Inside(EntityCategory::Per)
        ));
        assert!(!Tag::is_valid_transition(
            &Tag::Outside,
            &Tag::Inside(EntityCategory::Per)
        ));
        assert!(!Tag::is_valid_transition(
            &Tag::Begin(EntityCategory::Org),
            &Tag::Inside(EntityCategory::Per)
        ));
    }

    #[test]
    fn test_tag_from_label() {
        assert_eq!(Tag::from_label("O"), Some(Tag::Outside));
        assert_eq!(
            Tag::from_label("B-PER"),
            Some(Tag::Begin(EntityCategory::Per))
        );
        assert_eq!(
            Tag::from_label("I-LOC"),
            Some(Tag::Inside(EntityCategory::Loc))
        );
    }

    #[test]
    fn test_all_tags_have_unique_indices() {
        let all = Tag::all();
        let mut indices: Vec<usize> = all.iter().map(|t| t.index()).collect();
        indices.sort();
        indices.dedup();
        assert_eq!(indices.len(), Tag::COUNT);
    }
}
