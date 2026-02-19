//! # ner-core — Sistema de Reconhecimento de Entidades Nomeadas (NER)
//!
//! Este crate implementa um pipeline completo para Extração de Informação em textos em Português Brasileiro.
//! Ele foi projetado para ser didático, modular e extensível, permitindo a comparação entre diferentes
//! abordagens de NLP (Processamento de Linguagem Natural).
//!
//! ## Arquitetura do Sistema
//!
//! O sistema segue uma arquitetura de pipeline linear, onde o dado flui e é transformado passo a passo:
//!
//! 1.  **Entrada**: Texto bruto (String).
//! 2.  **Tokenização** ([`tokenizer`]): O texto é dividido em unidades (tokens), preservando offsets originais.
//! 3.  **Extração de Features** ([`features`]): Cada token é convertido em um vetor de características (ex: "começa com maiúscula", "sufixo=ão").
//! 4.  **Decodificação/Tagging** ([`tagger`]):
//!     *   **Regras/Gazetteers** ([`rule_based`]): Identificação direta por dicionários e expressões regulares.
//!     *   **Modelos Probabilísticos** ([`hmm`]): Hidden Markov Models para sequências.
//!     *   **Modelos Discriminativos** ([`maxent`, `perceptron`, `crf`]): Classificação baseada em features.
//! 5.  **Saída**: Lista de [`EntitySpan`] (ex: "Lula" -> PER, "Brasil" -> LOC).
//!
//! ## Exemplo de Uso
//!
//! ```rust
//! use ner_core::{NerPipeline, AlgorithmMode, TokenizerMode};
//!
//! // 1. Instancia o pipeline (carrega modelos e gazetteers)
//! let pipeline = NerPipeline::new();
//!
//! // 2. Texto para análise
//! let text = "O Supremo Tribunal Federal decidiu ontem.";
//!
//! // 3. Executa a análise usando modo Híbrido (Regras + CRF)
//! let (tokens, entities) = pipeline.analyze_with_mode(
//!     text,
//!     AlgorithmMode::Hybrid,
//!     TokenizerMode::Standard
//! );
//!
//! // 4. Exibe as entidades encontradas
//! for entity in entities {
//!     println!("Entidade: {} ({:?}) - Score: {:.2}", entity.text, entity.tag, entity.score);
//! }
//! ```
//!
//! ## Módulos Principais
//!
//! - [`pipeline`]: Orquestrador principal que conecta todos os estágios.
//! - [`tokenizer`]: Responsável pela segmentação do texto.
//! - [`features`]: Engenharia de características para modelos de ML.
//! - [`corpus`]: Dados de treinamento e teste anotados (BIO).


pub mod corpus;
pub mod crf;
pub mod features;
pub mod model;
pub mod pipeline;
pub mod rule_based;
pub mod tagger;
pub mod tokenizer;
pub mod hmm;
pub mod maxent;
pub mod perceptron;
pub mod span;
pub mod viterbi;

pub use pipeline::{AlgorithmMode, NerPipeline, PipelineEvent};
pub use tagger::{EntitySpan, Tag, TaggedToken};
pub use tokenizer::{Token, TokenizerMode};
