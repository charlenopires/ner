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
use crate::tokenizer::{tokenize_with_mode, Token, TokenizerMode};
use crate::viterbi::{viterbi_decode, ViterbiStep};

/// Modo de operação do algoritmo NER.
///
/// O usuário pode escolher qual combinação de algoritmos usar para analisar o texto.
/// Cada modo oferece um balanço diferente entre precisão e explicabilidade.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum AlgorithmMode {
    /// **Híbrido (Recomendado)**: Combina Regras + CRF + Viterbi.
    /// - Primeiro aplica regras determinísticas (gazetteers, regex).
    /// - Onde as regras não cobrem, usa o modelo estatístico (CRF).
    /// - Produz os melhores resultados gerais.
    Hybrid,
    /// **Apenas Regras**: Usa somente gazetteers e padrões.
    /// Ítil para debugging ou quando se quer controle total sobre a saída.
    /// Não generaliza para entidades não vistas.
    RulesOnly,
    /// **Apenas CRF**: Usa apenas o modelo estatístico.
    /// Ignora listas manuais, dependendo totalmente do aprendizado de máquina.
    /// Bom para avaliar a capacidade de generalização do modelo.
    CrfOnly,
    /// **Apenas Features**: Executa tokenização e extração de features, mas não classifica.
    /// Usado para inspecionar o que o modelo "vê" (ex: quais sufixos foram detectados).
    FeaturesOnly,
    /// **HMM (Hidden Markov Model)**: Modelo probabilístico baseado em transições e emissões simples.
    /// Menos preciso que o CRF, mas treina muito rápido.
    Hmm,
    /// **MaxEnt (Entropia Máxima)**: Classificador logístico que não considera a sequência (histórico).
    /// Classifica cada token independentemente.
    MaxEnt,
    /// **Perceptron Médio**: Algoritmo online simples e eficaz.
    /// Aprende iterativamente a separar as classes.
    Perceptron,
    /// **Span-Based**: Abordagem experimental que classifica spans inteiros em vez de tokens.
    SpanBased,
}

impl Default for AlgorithmMode {
    fn default() -> Self { AlgorithmMode::Hybrid }
}

/// Eventos emitidos pelo pipeline durante o processamento.
///
/// Estes eventos permitem que a UI (frontend) visualize o "raciocínio" do modelo passo-a-passo.
/// Cada variante carrega os dados necessários para renderizar uma etapa da visualização.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type", content = "data")]
pub enum PipelineEvent {
    /// **Passo 1**: Tokenização concluída.
    /// Retorna a lista de tokens e o total.
    TokenizationDone {
        tokens: Vec<Token>,
        total: usize,
    },
    /// **Passo 2**: Features extraídas para um token específico.
    /// Mostra quais atributos (ex: "é maiúscula", "sufixo=ão") foram ativados.
    FeaturesComputed {
        token_index: usize,
        token_text: String,
        /// Lista das 10 features com maiores pesos para visualização.
        top_features: Vec<(String, f64)>,
    },
    /// **Passo 3 (Opcional)**: Uma regra manual foi aplicada com sucesso.
    /// Indica que o sistema "cortou caminho" usando conhecimento prévio.
    RuleApplied {
        token_index: usize,
        token_text: String,
        tag: String,
        rule_name: String,
        confidence: f64,
    },
    /// **Passo 4**: Um passo do algoritmo de decodificação Viterbi.
    /// Mostra as probabilidades acumuladas para cada tag naquele ponto da frase.
    ViterbiStep {
        step: ViterbiStep,
        token_text: String,
    },
    /// **Passo Final**: Tag definitiva atribuída a um token.
    /// Pode vir de uma regra ou do cálculo do Viterbi/CRF.
    TagAssigned {
        token_index: usize,
        token_text: String,
        tag: String,
        confidence: f64,
        source: String, // "rule" ou "crf"
    },
    /// **Conclusão**: O processo terminou com sucesso.
    /// Retorna todas as entidades estruturadas e estatísticas de tempo.
    Done {
        entities: Vec<EntitySpan>,
        tagged_tokens: Vec<TaggedToken>,
        total_tokens: usize,
        processing_ms: u64,
    },
    /// **Falha**: Ocorreu um erro irrecuperável.
    Error {
        message: String,
    },
}

/// O pipeline NER principal.
///
/// Atua como o **controlador** do sistema, orquestrando:
/// 1. Tokenização do texto bruto.
/// 2. Extração de features para cada token.
/// 3. Aplicação opcional de regras manuais.
/// 4. Decodificação Viterbi (CRF) para predição probabilística.
/// 5. Fusão dos resultados e construção das entidades finais.
///
/// # Modos de Uso
/// - **Sync**: Método `analyze` para scripts e chamadas diretas.
/// - **Streaming**: Método `analyze_streaming` para UIs reativas (via WebSocket).
pub struct NerPipeline {
    pub model: NerModel,
}

impl NerPipeline {
    /// Cria o pipeline carregando o modelo padrão com pesos heurísticos.
    pub fn new() -> Self {
        Self {
            model: NerModel::default(),
        }
    }

    /// Processa o texto de forma síncrona e retorna o resultado final.
    ///
    /// Ideal para processamento em lote ou quando não há necessidade de feedback visual.
    /// Usa internamente o modo `Hybrid` e tokenização `Standard`.
    pub fn analyze(&self, text: &str) -> (Vec<TaggedToken>, Vec<EntitySpan>) {
        self.analyze_with_mode(text, AlgorithmMode::Hybrid, TokenizerMode::Standard)
    }

    /// Processa o texto de forma síncrona, configurando o algoritmo e tokenizador.
    ///
    /// Útil para debugging ou comparações de performance entre modos.
    pub fn analyze_with_mode(&self, text: &str, mode: AlgorithmMode, tokenizer_mode: TokenizerMode) -> (Vec<TaggedToken>, Vec<EntitySpan>) {
        let (tx, rx) = mpsc::channel();
        self.analyze_streaming(text, mode, tokenizer_mode, tx);
        let mut tagged = vec![];
        let mut entities = vec![];
        
        // Consome todos os eventos até o fim
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

    /// Executa o pipeline enviando eventos de progresso em tempo real.
    ///
    /// Este método é o coração da interface visual. Ele não retorna valores diretamente,
    /// mas sim "empurra" `PipelineEvent`s pelo canal `tx`.
    ///
    /// # Fluxo de Eventos
    /// 1. `TokenizationDone`: Tokens gerados.
    /// 2. `FeaturesComputed` (Loop): Features de cada token (se aplicável).
    /// 3. `RuleApplied` (Loop): Regras que "bateram" (se modo híbrido).
    /// 4. `ViterbiStep` (Loop): Passos do algoritmo de decodificação.
    /// 5. `TagAssigned` (Loop): Decisão final para cada token.
    /// 6. `Done`: Resultado final consolidado.
    pub fn analyze_streaming(&self, text: &str, mode: AlgorithmMode, tokenizer_mode: TokenizerMode, tx: mpsc::Sender<PipelineEvent>) {
        let start = std::time::Instant::now();

        // === Passo 1: Tokenização ===
        let tokens = tokenize_with_mode(text, tokenizer_mode);
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

        match mode {
            AlgorithmMode::Hybrid | AlgorithmMode::RulesOnly | AlgorithmMode::CrfOnly | AlgorithmMode::FeaturesOnly => {
                 self.analyze_streaming_standard(text, &tokens, mode, &tx, start);
            }
            AlgorithmMode::Hmm | AlgorithmMode::MaxEnt | AlgorithmMode::Perceptron => {
                 self.analyze_streaming_ml(text, &tokens, mode, &tx, start);
            }
             AlgorithmMode::SpanBased => {
                 self.analyze_streaming_span(text, &tokens, &tx, start);
             }
        }
    }

    fn analyze_streaming_standard(&self, text: &str, tokens: &[Token], mode: AlgorithmMode, tx: &mpsc::Sender<PipelineEvent>, start: std::time::Instant) {
         // === Passo 2: Extração de Features ===
        let gazetteers = self.model.gazetteers();
        let feature_vectors: Vec<FeatureVector> =
            extract_features(tokens, &gazetteers);

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
            let rule_results = self.model.rule_engine.apply(tokens);
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
                total_tokens: tokens.len(),
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
            total_tokens: tokens.len(),
            processing_ms: elapsed,
        });
    }

    fn analyze_streaming_ml(&self, text: &str, tokens: &[Token], mode: AlgorithmMode, tx: &mpsc::Sender<PipelineEvent>, start: std::time::Instant) {
        // Envia features se for MaxEnt ou Perceptron
        if mode == AlgorithmMode::MaxEnt || mode == AlgorithmMode::Perceptron {
             let gazetteers = self.model.gazetteers();
             let feature_vectors = extract_features(tokens, &gazetteers);
             for (i, fv) in feature_vectors.iter().enumerate() {
                // Top features logic clone from standard
                let mut sorted: Vec<(String, f64)> = fv.features.iter().map(|(k, v)| (k.clone(), *v)).collect();
                sorted.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
                sorted.truncate(10);
                let _ = tx.send(PipelineEvent::FeaturesComputed {
                    token_index: i,
                    token_text: tokens[i].text.clone(),
                    top_features: sorted,
                });
            }
        }

        let token_strs: Vec<String> = tokens.iter().map(|t| t.text.clone()).collect();
        let pred_tags = match mode {
            AlgorithmMode::Hmm => self.model.hmm.predict(&token_strs),
            AlgorithmMode::MaxEnt => self.model.maxent.predict(&token_strs),
            AlgorithmMode::Perceptron => self.model.perceptron.predict(&token_strs),
            _ => unreachable!(),
        };

        let tagged_tokens: Vec<TaggedToken> = tokens.iter().zip(pred_tags.iter()).enumerate().map(|(i, (token, tag_str))| {
            let tag = Tag::from_label(tag_str).unwrap_or(Tag::Outside);
            let _ = tx.send(PipelineEvent::TagAssigned {
                token_index: i,
                token_text: token.text.clone(),
                tag: tag.label(),
                confidence: 1.0, 
                source: format!("{:?}", mode).to_lowercase(),
            });
            TaggedToken { token: token.clone(), tag, confidence: 1.0 }
        }).collect();

        let entities = tokens_to_spans(&tagged_tokens, text);
        let _ = tx.send(PipelineEvent::Done {
            entities,
            tagged_tokens,
            total_tokens: tokens.len(),
            processing_ms: start.elapsed().as_millis() as u64,
        });
    }

    fn analyze_streaming_span(&self, text: &str, tokens: &[Token], tx: &mpsc::Sender<PipelineEvent>, start: std::time::Instant) {
        let token_strs: Vec<String> = tokens.iter().map(|t| t.text.clone()).collect();
        let spans = self.model.span.predict(&token_strs);

        // Dummy tagged tokens (converte spans de volta para BIO para visualização seria ideal, mas complexo com overlaps)
        // Para simplificar, gera tudo como O, exceto se eu quiser reconstruir BIO sem overlap.
        let mut tagged_tokens: Vec<TaggedToken> = tokens.iter().map(|t| TaggedToken {
            token: t.clone(),
            tag: Tag::Outside,
            confidence: 1.0
        }).collect();

        // Tenta marcar BIO para o primeiro layer de spans
        let mut occupied = vec![false; tokens.len()];
        for span in &spans {
            // Ignora spans que colidem
             let range = span.start..span.end;
             if range.clone().any(|i| i < occupied.len() && occupied[i]) {
                 continue; // Skip overlapping span for BIO visualization
             }
             
             if let Some(cat) = crate::tagger::EntityCategory::from_str(&span.label) {
                 if span.start < tagged_tokens.len() {
                    tagged_tokens[span.start].tag = Tag::Begin(cat);
                    occupied[span.start] = true;
                    for i in (span.start + 1)..span.end {
                        if i < tagged_tokens.len() {
                            tagged_tokens[i].tag = Tag::Inside(cat);
                            occupied[i] = true;
                        }
                    }
                 }
             }
        }

        // For Done event, TagAssigned events
        for (i, tt) in tagged_tokens.iter().enumerate() {
             let _ = tx.send(PipelineEvent::TagAssigned {
                token_index: i,
                token_text: tt.token.text.clone(),
                tag: tt.tag.label(),
                confidence: 1.0, 
                source: "span_based".to_string(),
            });
        }

        let mut entities_vec = Vec::new();
        for span in spans {
             if span.start < tokens.len() && span.end <= tokens.len() {
                let start_char = tokens[span.start].start;
                let end_char = tokens[span.end - 1].end;
                
                let cat = crate::tagger::EntityCategory::from_str(&span.label).unwrap_or(crate::tagger::EntityCategory::Misc);
                
                entities_vec.push(EntitySpan {
                    text: text[start_char..end_char].to_string(),
                    category: cat,
                    start_token: span.start,
                    end_token: span.end - 1,
                    start: start_char,
                    end: end_char,
                    confidence: 1.0,
                    source: "span_model".to_string(),
                });
            }
        }

        let _ = tx.send(PipelineEvent::Done {
            entities: entities_vec,
            tagged_tokens,
            total_tokens: tokens.len(),
            processing_ms: start.elapsed().as_millis() as u64,
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
        pipeline.analyze_streaming("São Paulo é a maior cidade do Brasil.", AlgorithmMode::Hybrid, TokenizerMode::Standard, tx);

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
