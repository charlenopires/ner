//! # Pipeline NER — Orquestrador com Eventos Observáveis
//!
//! O pipeline coordena todos os módulos (tokenizador, features, regras, CRF/Viterbi)
//! e emite eventos em cada passo via um canal Rust (`mpsc`), permitindo que
//! o servidor WebSocket transmita o progresso em tempo real para o cliente.

use std::sync::mpsc;

use serde::{Deserialize, Serialize};

use crate::features::{extract_features, FeatureVector};
use crate::model::NerModel;
use crate::tagger::{tokens_to_spans, EntitySpan, Tag, TaggedToken};
use crate::tokenizer::{tokenize, Token};
use crate::viterbi::{viterbi_decode, ViterbiStep};

/// Modo de operação do algoritmo NER
///
/// O usuário pode escolher qual combinação de algoritmos usar para analisar o texto.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum AlgorithmMode {
    /// Usa regras + CRF/Viterbi (padrão) — produz os melhores resultados
    Hybrid,
    /// Usa apenas o motor de regras (gazetteers + regex)
    RulesOnly,
    /// Usa apenas o CRF + Viterbi (sem regras)
    CrfOnly,
    /// Apenas tokenização e features (sem classificação)
    FeaturesOnly,
}

impl Default for AlgorithmMode {
    fn default() -> Self { AlgorithmMode::Hybrid }
}

/// Eventos emitidos pelo pipeline durante o processamento
/// Cada evento corresponde a um passo visualizável na interface
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type", content = "data")]
pub enum PipelineEvent {
    /// Tokenização concluída
    TokenizationDone {
        tokens: Vec<Token>,
        total: usize,
    },
    /// Features extraídas para um token
    FeaturesComputed {
        token_index: usize,
        token_text: String,
        top_features: Vec<(String, f64)>, // top 10 features mais relevantes
    },
    /// Uma regra foi aplicada a um token
    RuleApplied {
        token_index: usize,
        token_text: String,
        tag: String,
        rule_name: String,
        confidence: f64,
    },
    /// Um passo do algoritmo Viterbi foi processado
    ViterbiStep {
        step: ViterbiStep,
        token_text: String,
    },
    /// Tag final atribuída a um token
    TagAssigned {
        token_index: usize,
        token_text: String,
        tag: String,
        confidence: f64,
        source: String, // "rule" ou "crf"
    },
    /// Pipeline concluído com as entidades identificadas
    Done {
        entities: Vec<EntitySpan>,
        tagged_tokens: Vec<TaggedToken>,
        total_tokens: usize,
        processing_ms: u64,
    },
    /// Erro durante processamento
    Error {
        message: String,
    },
}

/// O pipeline NER principal
///
/// Encapsula o modelo e oferece dois modos de execução:
/// - [`analyze`]: execução síncrona, retorna resultado direto
/// - [`analyze_streaming`]: execução com eventos para WebSocket
pub struct NerPipeline {
    pub model: NerModel,
}

impl NerPipeline {
    /// Cria o pipeline com o modelo padrão baseado no corpus PT-BR
    pub fn new() -> Self {
        Self {
            model: NerModel::default(),
        }
    }

    /// Análise direta (sem streaming) — retorna entidades e tokens anotados
    pub fn analyze(&self, text: &str) -> (Vec<TaggedToken>, Vec<EntitySpan>) {
        self.analyze_with_mode(text, AlgorithmMode::Hybrid)
    }

    /// Análise direta com modo de algoritmo escolhido
    pub fn analyze_with_mode(&self, text: &str, mode: AlgorithmMode) -> (Vec<TaggedToken>, Vec<EntitySpan>) {
        let (tx, rx) = mpsc::channel();
        self.analyze_streaming(text, mode, tx);
        let mut tagged = vec![];
        let mut entities = vec![];
        while let Ok(event) = rx.recv() {
            if let PipelineEvent::Done {
                tagged_tokens,
                entities: ents,
                ..
            } = event
            {
                tagged = tagged_tokens;
                entities = ents;
            }
        }
        (tagged, entities)
    }

    /// Análise com streaming de eventos via canal mpsc
    ///
    /// Envia [`PipelineEvent`]s para o canal à medida que cada passo é concluído.
    /// O canal pode ser conectado a um WebSocket para transmissão em tempo real.
    /// O modo pode ser `Hybrid`, `RulesOnly`, `CrfOnly` ou `FeaturesOnly`.
    pub fn analyze_streaming(&self, text: &str, mode: AlgorithmMode, tx: mpsc::Sender<PipelineEvent>) {
        let start = std::time::Instant::now();

        // === Passo 1: Tokenização ===
        let tokens = tokenize(text);
        let total = tokens.len();
        let _ = tx.send(PipelineEvent::TokenizationDone {
            tokens: tokens.clone(),
            total,
        });

        if tokens.is_empty() {
            let _ = tx.send(PipelineEvent::Done {
                entities: vec![],
                tagged_tokens: vec![],
                total_tokens: 0,
                processing_ms: start.elapsed().as_millis() as u64,
            });
            return;
        }

        // === Passo 2: Extração de Features ===
        let gazetteers = self.model.gazetteers();
        let feature_vectors: Vec<FeatureVector> =
            extract_features(&tokens, &gazetteers);

        for (i, fv) in feature_vectors.iter().enumerate() {
            // Envia as top 10 features por importância
            let mut sorted: Vec<(String, f64)> = fv
                .features
                .iter()
                .map(|(k, v)| (k.clone(), *v))
                .collect();
            sorted.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
            sorted.truncate(10);

            let _ = tx.send(PipelineEvent::FeaturesComputed {
                token_index: i,
                token_text: tokens[i].text.clone(),
                top_features: sorted,
            });
        }

        // === Passo 3: Motor de Regras (pula se CrfOnly ou FeaturesOnly) ===
        let mut rule_tags: Vec<Option<(Tag, String, f64)>> = vec![None; tokens.len()];

        if mode != AlgorithmMode::CrfOnly && mode != AlgorithmMode::FeaturesOnly {
            let rule_results = self.model.rule_engine.apply(&tokens);
            for (i, maybe_match) in rule_results.iter().enumerate() {
                if let Some(rm) = maybe_match {
                    let _ = tx.send(PipelineEvent::RuleApplied {
                        token_index: i,
                        token_text: tokens[i].text.clone(),
                        tag: rm.tag.label(),
                        rule_name: rm.rule_name.clone(),
                        confidence: rm.confidence,
                    });
                    rule_tags[i] = Some((rm.tag.clone(), rm.rule_name.clone(), rm.confidence));
                }
            }
        }

        // Se RulesOnly: aplica apenas as regras e conclui
        if mode == AlgorithmMode::RulesOnly || mode == AlgorithmMode::FeaturesOnly {
            let tagged_tokens: Vec<TaggedToken> = tokens
                .iter()
                .enumerate()
                .map(|(i, token)| {
                    if let Some((rule_tag, rule_name, rule_conf)) = &rule_tags[i] {
                        let _ = tx.send(PipelineEvent::TagAssigned {
                            token_index: i,
                            token_text: token.text.clone(),
                            tag: rule_tag.label(),
                            confidence: *rule_conf,
                            source: rule_name.clone(),
                        });
                        TaggedToken { token: token.clone(), tag: rule_tag.clone(), confidence: *rule_conf }
                    } else {
                        let _ = tx.send(PipelineEvent::TagAssigned {
                            token_index: i,
                            token_text: token.text.clone(),
                            tag: Tag::Outside.label(),
                            confidence: 1.0,
                            source: if mode == AlgorithmMode::FeaturesOnly { "features_only".into() } else { "no_rule".into() },
                        });
                        TaggedToken { token: token.clone(), tag: Tag::Outside, confidence: 1.0 }
                    }
                })
                .collect();

            let entities = tokens_to_spans(&tagged_tokens, text);
            let _ = tx.send(PipelineEvent::Done {
                entities,
                tagged_tokens,
                total_tokens: total,
                processing_ms: start.elapsed().as_millis() as u64,
            });
            return;
        }

        // === Passo 4: Viterbi (CRF) — pula se RulesOnly ===
        let viterbi_result = viterbi_decode(&self.model.crf, &feature_vectors);

        for (i, step) in viterbi_result.steps.iter().enumerate() {
            let _ = tx.send(PipelineEvent::ViterbiStep {
                step: step.clone(),
                token_text: tokens[i].text.clone(),
            });
        }

        // === Passo 5: Fusão de Resultados ===
        // No modo Hybrid: Regras prevalecem; no CrfOnly: apenas CRF
        let tag_probs: Vec<Vec<f64>> = viterbi_result.steps.iter().map(|step| {
            let scores: Vec<f64> = step.scores.iter().map(|s| s.score).collect();
            crate::viterbi::scores_to_probs(&scores)
        }).collect();

        let tagged_tokens: Vec<TaggedToken> = tokens
            .iter()
            .enumerate()
            .map(|(i, token)| {
                let crf_tag = viterbi_result
                    .best_sequence
                    .get(i)
                    .cloned()
                    .unwrap_or(Tag::Outside);
                let crf_confidence = tag_probs
                    .get(i)
                    .and_then(|probs| probs.get(crf_tag.index()))
                    .copied()
                    .unwrap_or(0.5);

                // Modo Hybrid: regra vence se disponível; CrfOnly: ignora regras
                if mode == AlgorithmMode::Hybrid {
                    if let Some((rule_tag, rule_name, rule_conf)) = &rule_tags[i] {
                        let _ = tx.send(PipelineEvent::TagAssigned {
                            token_index: i,
                            token_text: token.text.clone(),
                            tag: rule_tag.label(),
                            confidence: *rule_conf,
                            source: rule_name.clone(),
                        });
                        return TaggedToken {
                            token: token.clone(),
                            tag: rule_tag.clone(),
                            confidence: *rule_conf,
                        };
                    }
                }

                let _ = tx.send(PipelineEvent::TagAssigned {
                    token_index: i,
                    token_text: token.text.clone(),
                    tag: crf_tag.label(),
                    confidence: crf_confidence,
                    source: "crf".to_string(),
                });
                TaggedToken {
                    token: token.clone(),
                    tag: crf_tag,
                    confidence: crf_confidence,
                }
            })
            .collect();

        // === Passo 6: Agrupamento de Entidades ===
        let mut entities = tokens_to_spans(&tagged_tokens, text);
        for span in &mut entities {
            if mode == AlgorithmMode::Hybrid {
                if let Some(Some((_, rule_name, _))) = rule_tags.get(span.start_token) {
                    span.source = rule_name.clone();
                }
            }
        }

        let elapsed = start.elapsed().as_millis() as u64;

        let _ = tx.send(PipelineEvent::Done {
            entities: entities.clone(),
            tagged_tokens: tagged_tokens.clone(),
            total_tokens: total,
            processing_ms: elapsed,
        });
    }
}

impl Default for NerPipeline {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_pipeline_basic() {
        let pipeline = NerPipeline::new();
        let (tagged, entities) = pipeline.analyze(
            "Lula foi eleito presidente do Brasil em 2002 com apoio da Petrobras.",
        );
        assert!(!tagged.is_empty());
        // Deve encontrar pelo menos uma entidade
        assert!(!entities.is_empty());
    }

    #[test]
    fn test_pipeline_empty() {
        let pipeline = NerPipeline::new();
        let (tagged, entities) = pipeline.analyze("");
        assert!(tagged.is_empty());
        assert!(entities.is_empty());
    }

    #[test]
    fn test_pipeline_events_streaming() {
        let pipeline = NerPipeline::new();
        let (tx, rx) = mpsc::channel();
        pipeline.analyze_streaming("São Paulo é a maior cidade do Brasil.", AlgorithmMode::Hybrid, tx);

        let events: Vec<PipelineEvent> = rx.try_iter().collect();
        assert!(!events.is_empty());

        // Deve ter TokenizationDone como primeiro evento
        assert!(
            matches!(&events[0], PipelineEvent::TokenizationDone { .. }),
            "Primeiro evento deve ser TokenizationDone"
        );

        // Deve ter Done como último evento
        let last = events.last().unwrap();
        assert!(
            matches!(last, PipelineEvent::Done { .. }),
            "Último evento deve ser Done"
        );
    }
}
