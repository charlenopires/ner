//! # Hidden Markov Model (HMM) para NER
//!
//! Implementação clássica de HMM onde:
//! - **Estados Ocultos**: Tags (B-PER, I-ORG, O, etc.)
//! - **Observações**: Tokens (palavras)
//!
//! O modelo aprende:
//! 1. Probabilidade de Transição: P(tag_atual | tag_anterior)
//! 2. Probabilidade de Emissão: P(palavra | tag)
//! 3. Probabilidade Inicial: P(tag_inicial)
//!
//! A decodificação é feita via algoritmo de Viterbi, maximizando P(tags | palavras).

use std::collections::{HashMap, HashSet};
use serde::{Deserialize, Serialize};
use crate::corpus::AnnotatedSentence;


/// Modelo HMM (Hidden Markov Model) treinado para NER.
///
/// O HMM é um modelo **generativo** que modela a probabilidade conjunta $P(x, y)$
/// de observações $x$ (tokens) e estados ocultos $y$ (tags).
///
/// # Componentes
/// - **Transição**: Probabilidade de uma tag seguir outra ($P(y_i | y_{i-1})$).
/// - **Emissão**: Probabilidade de um token ser gerado por uma tag ($P(x_i | y_i)$).
/// - **Inicial**: Probabilidade de uma tag começar a frase ($P(y_0)$).
///
/// # Armazenamento
/// As probabilidades são armazenadas em **log-space** para evitar underflow numérico
/// ao multiplicar muitas probabilidades pequenas.
/// $$ \log(A \cdot B) = \log(A) + \log(B) $$
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HmmModel {
    /// $P(y_i | y_{i-1})$ em log-space. Chave: `(prev_tag, curr_tag)`.
    transition_probs: HashMap<(String, String), f64>,
    /// $P(x_i | y_i)$ em log-space. Chave: `(tag, word)`.
    emission_probs: HashMap<(String, String), f64>,
    /// $P(y_0)$ em log-space. Chave: `tag`.
    start_probs: HashMap<String, f64>,
    /// Lista ordenada de todas as tags conhecidas.
    all_tags: Vec<String>,
    /// Vocabulário conhecido (para identificar e tratar tokens desconhecidos `<UNK>`).
    vocab: HashSet<String>,
}

impl HmmModel {
    pub fn new() -> Self {
        Self {
            transition_probs: HashMap::new(),
            emission_probs: HashMap::new(),
            start_probs: HashMap::new(),
            all_tags: Vec::new(),
            vocab: HashSet::new(),
        }
    }

    /// Treina o HMM com o corpus fornecido (Supervised Learning).
    ///
    /// # Processo de Treinamento
    /// 1. **Contagem**: Itera sobre o corpus e conta quantas vezes cada tag aparece, 
    ///    quantas vezes uma tag segue outra (transição) e quantas vezes uma tag gera uma palavra (emissão).
    /// 2. **Smoothing (Suavização)**: Aplica *Add-1 Smoothing* (Laplace) para garantir que
    ///    nenhuma probabilidade seja zero (o que quebraria o logaritmo).
    /// 3. **Log-Probabilidades**: Converte tudo para logaritmo para estabilidade numérica.
    ///
    /// # Exemplo
    /// ```rust
    /// // Suponha corpus com [("Lula", "B-PER"), ("é", "O")]
    /// // P("Lula" | "B-PER") = count("Lula", "B-PER") / count("B-PER")
    /// ```
    pub fn train(&mut self, corpus: &[AnnotatedSentence]) {
        let mut transition_counts: HashMap<(String, String), u32> = HashMap::new();
        let mut emission_counts: HashMap<(String, String), u32> = HashMap::new();
        let mut start_counts: HashMap<String, u32> = HashMap::new();
        let mut tag_counts: HashMap<String, u32> = HashMap::new();
        let mut vocab: HashSet<String> = HashSet::new();
        let mut all_tags_set: HashSet<String> = HashSet::new();

        // 1. Contagem das frequências brutas
        for sentence in corpus {
            let mut prev_tag: Option<String> = None;

            for (i, (word, tag)) in sentence.annotations.iter().enumerate() {
                let w = word.to_string();
                let t = tag.to_string();

                vocab.insert(w.clone());
                all_tags_set.insert(t.clone());
                *tag_counts.entry(t.clone()).or_insert(0) += 1;

                // Emissão: quantas vezes a tag T gerou a palavra W?
                *emission_counts.entry((t.clone(), w)).or_insert(0) += 1;

                if i == 0 {
                    // Start: quantas vezes a sentença começou com a tag T?
                    *start_counts.entry(t.clone()).or_insert(0) += 1;
                } else if let Some(prev) = prev_tag {
                    // Transição: quantas vezes a tag PREV foi seguida por T?
                    *transition_counts.entry((prev, t.clone())).or_insert(0) += 1;
                }

                prev_tag = Some(t);
            }
        }

        self.vocab = vocab;
        self.all_tags = all_tags_set.into_iter().collect();
        self.all_tags.sort(); // Garante ordem determinística

        // 2. Normalização e Cálculo de Probabilidades (com Smoothing)
        let vocab_size = self.vocab.len() as f64;
        let num_tags = self.all_tags.len() as f64;

        // Probabilidades Iniciais P(tag)
        let total_starts = corpus.len() as f64;
        for tag in &self.all_tags {
            let count = *start_counts.get(tag).unwrap_or(&0) as f64;
            // Add-1 smoothing: evita log(0) se uma tag nunca começar frases (raro, mas possível)
            let prob = (count + 1.0) / (total_starts + num_tags);
            self.start_probs.insert(tag.clone(), prob.ln());
        }

        // Probabilidades de Transição P(curr | prev)
        for prev in &self.all_tags {
            let prev_count = *tag_counts.get(prev).unwrap_or(&0) as f64;
            for curr in &self.all_tags {
                let count = *transition_counts.get(&(prev.clone(), curr.clone())).unwrap_or(&0) as f64;
                // Add-1 smoothing: importante para combinações de tags inéditas
                let prob = (count + 1.0) / (prev_count + num_tags);
                self.transition_probs.insert((prev.clone(), curr.clone()), prob.ln());
            }
        }

        // Probabilidades de Emissão P(word | tag)
        // Inclui probabilidade para token especial <UNK> (desconhecido)
        for tag in &self.all_tags {
            let tag_count = *tag_counts.get(tag).unwrap_or(&0) as f64;
            
            // Para cada palavra conhecida no vocabulário
            for word in &self.vocab {
                let count = *emission_counts.get(&(tag.clone(), word.clone())).unwrap_or(&0) as f64;
                // Add-1 smoothing
                let prob = (count + 1.0) / (tag_count + vocab_size + 1.0);
                self.emission_probs.insert((tag.clone(), word.clone()), prob.ln());
            }

            // Probabilidade reservada para palavras desconhecidas (<UNK>)
            // Simula ter visto <UNK> 0 vezes, mas com add-1 vira 1.
            let prob_unk = 1.0 / (tag_count + vocab_size + 1.0);
            self.emission_probs.insert((tag.clone(), "<UNK>".to_string()), prob_unk.ln());
        }
    }

    /// Decodifica uma sequência de tokens para encontrar a melhor sequência de tags.
    ///
    /// Utiliza o **Algoritmo de Viterbi**, que é um algoritmo de programação dinâmica
    /// para encontrar o caminho mais provável em um HMM.
    ///
    /// # Complexidade
    /// $O(N \cdot T^2)$, onde $N$ é o número de tokens e $T$ o número de tags possíveis.
    ///
    /// # Retorno
    /// Retorna a lista de tags preditas (ex: `["B-PER", "O", "O"]`) alinhada com os tokens de entrada.
    pub fn predict(&self, tokens: &[String]) -> Vec<String> {
        if tokens.is_empty() {
            return Vec::new();
        }

        let n_tokens = tokens.len();
        let n_tags = self.all_tags.len();
        
        // viterbi[t][s] = log-prob do melhor caminho terminando no tempo t com estado s
        let mut viterbi = vec![vec![f64::NEG_INFINITY; n_tags]; n_tokens];
        // backptr[t][s] = índice do estado anterior que maximizou viterbi[t, s]
        let mut backptr = vec![vec![0usize; n_tags]; n_tokens];

        // 1. Inicialização (t=0)
        let first_token = if self.vocab.contains(&tokens[0]) { &tokens[0] } else { "<UNK>" };
        
        for (s, tag) in self.all_tags.iter().enumerate() {
            let start_p = self.start_probs.get(tag).cloned().unwrap_or(f64::NEG_INFINITY);
            let emit_p = self.emission_probs.get(&(tag.clone(), first_token.to_string())).cloned().unwrap_or(f64::NEG_INFINITY);
            viterbi[0][s] = start_p + emit_p;
        }

        // 2. Recursão (t=1..N)
        for t in 1..n_tokens {
            let token = if self.vocab.contains(&tokens[t]) { &tokens[t] } else { "<UNK>" };
            
            for (s, curr_tag) in self.all_tags.iter().enumerate() {
                let emit_p = self.emission_probs.get(&(curr_tag.clone(), token.to_string())).cloned().unwrap_or(f64::NEG_INFINITY);
                
                let mut best_prob = f64::NEG_INFINITY;
                let mut best_prev = 0;

                for (prev_s, prev_tag) in self.all_tags.iter().enumerate() {
                    let trans_p = self.transition_probs.get(&(prev_tag.clone(), curr_tag.clone())).cloned().unwrap_or(f64::NEG_INFINITY);
                    let prob = viterbi[t-1][prev_s] + trans_p + emit_p;
                    
                    if prob > best_prob {
                        best_prob = prob;
                        best_prev = prev_s;
                    }
                }
                
                viterbi[t][s] = best_prob;
                backptr[t][s] = best_prev;
            }
        }

        // 3. Terminação (encontrar melhor estado final)
        let mut best_last_prob = f64::NEG_INFINITY;
        let mut best_last_tag_idx = 0;
        
        for s in 0..n_tags {
            if viterbi[n_tokens-1][s] > best_last_prob {
                best_last_prob = viterbi[n_tokens-1][s];
                best_last_tag_idx = s;
            }
        }

        // 4. Backtracking (reconstrução do caminho)
        let mut best_path = vec![String::new(); n_tokens];
        let mut curr_idx = best_last_tag_idx;
        
        best_path[n_tokens-1] = self.all_tags[curr_idx].clone();
        
        for t in (1..n_tokens).rev() {
            curr_idx = backptr[t][curr_idx];
            best_path[t-1] = self.all_tags[curr_idx].clone();
        }

        best_path
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_hmm_basic_training() {
        let corpus = vec![
            AnnotatedSentence {
                text: "Lula é presidente",
                domain: "test",
                annotations: &[("Lula", "B-PER"), ("é", "O"), ("presidente", "O")]
            }
        ];

        let mut model = HmmModel::new();
        model.train(&corpus);

        // Deve ter aprendido as tags
        assert!(model.all_tags.contains(&"B-PER".to_string()));
        assert!(model.all_tags.contains(&"O".to_string()));

        // Deve prever a mesma sentença corretamente
        let tokens = vec!["Lula".to_string(), "é".to_string(), "presidente".to_string()];
        let tags = model.predict(&tokens);
        
        assert_eq!(tags[0], "B-PER");
        assert_eq!(tags[1], "O");
        assert_eq!(tags[2], "O");
    }

    #[test]
    fn test_hmm_unknown_word() {
        let corpus = vec![
            AnnotatedSentence {
                text: "Brasil é lindo",
                domain: "test",
                annotations: &[("Brasil", "B-LOC"), ("é", "O"), ("lindo", "O")]
            }
        ];

        let mut model = HmmModel::new();
        model.train(&corpus);

        // "Japão" é desconhecido, mas deve ser tratado via UNK. 
        // Como B-LOC -> O tem alta prob, e B-LOC emite UNK com certa prob,
        // o comportamento exato depende dos pesos, mas não deve panic.
        let tokens = vec!["Japão".to_string(), "é".to_string(), "lindo".to_string()];
        let tags = model.predict(&tokens);
        
        // Pelo menos o tamanho deve ser igual
        assert_eq!(tags.len(), 3);
    }
}
