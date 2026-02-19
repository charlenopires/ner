//! # ner-core — Reconhecimento de Entidades Nomeadas em Português Brasileiro
//!
//! Esta crate implementa um sistema NER didático que combina:
//! - **Motor de Regras** (Gazetteers + Regex)
//! - **CRF (Conditional Random Field)** com decodificação **Viterbi**
//!
//! ## Fluxo do Pipeline
//!
//! ```text
//! Texto → Tokenização → Features → Regras + CRF → Viterbi → Entidades
//! ```
//!
//! Cada passo emite eventos via [`pipeline::PipelineEvent`] para visualização em tempo real.

pub mod corpus;
pub mod crf;
pub mod features;
pub mod model;
pub mod pipeline;
pub mod rule_based;
pub mod tagger;
pub mod tokenizer;
pub mod viterbi;

pub use pipeline::{AlgorithmMode, NerPipeline, PipelineEvent};
pub use tagger::{EntitySpan, Tag, TaggedToken};
pub use tokenizer::Token;
