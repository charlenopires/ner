//! # Modelo NER Pré-treinado
//!
//! O modelo encapsula:
//! - **Pesos CRF** estimados a partir do corpus PT-BR usando frequências de tags
//! - **Gazetteers** compilados automaticamente do corpus + listas manuais
//! - **Motor de Regras** configurado com entidades brasileiras conhecidas
//!
//! ## Como os pesos foram derivados
//!
//! Os pesos do CRF foram estimados de forma heurística a partir das frequências
//! observadas no corpus anotado. Em um sistema real, seriam treinados via
//! máxima verossimilhança condicional com L-BFGS. Para fins didáticos,
//! codificamos pesos que refletem os padrões mais fortes do corpus.

use crate::corpus::extract_gazetteers_from_corpus;
use crate::corpus::get_corpus;
use crate::crf::CrfModel;
use crate::features::Gazetteers;
use crate::hmm::HmmModel;
use crate::maxent::MaxEntModel;
use crate::perceptron::PerceptronModel;
use crate::rule_based::RuleEngine;
use crate::span::SpanModel;
use crate::tagger::{EntityCategory, Tag};

/// O modelo NER completo, agregando todos os sub-modelos e recursos.
///
/// Este struct serve como o "cérebro" do sistema, contendo:
/// - **CRF**: O modelo estatístico principal (pesos).
/// - **Regras**: O motor de regras determinísticas.
/// - **Gazelleers**: As listas de entidades conhecidas.
/// - **Outros Modelos**: HMM, MaxEnt, Perceptron, SpanModel (para experimentação).
pub struct NerModel {
    /// ## Exemplos
    ///
    /// Se o modelo for configurado com pesos manuais (como em `build()`), ele
    /// usará o conhecimento embutido sobre língua portuguesa (sufixos, prefixos, listas)
    /// para pontuar as tags candidatas.
    pub crf: CrfModel,
    /// Modelo HMM (Hidden Markov Model)
    pub hmm: HmmModel,
    /// Modelo de Maxima Entropia
    pub maxent: MaxEntModel,
    /// Modelo Perceptron
    pub perceptron: PerceptronModel,
    /// Modelo Span
    pub span: SpanModel,
    /// Motor de regras para aplicação de dicionários e regex
    pub rule_engine: RuleEngine,
    /// Cache interno de gazetteers para acesso rápido
    gazetteers_cache: Gazetteers,
}

impl NerModel {
    /// Constrói o modelo padrão com pesos derivados heuristicamente do corpus PT-BR.
    ///
    /// Em um cenário de produção real, estes pesos seriam aprendidos via treinamento (L-BFGS).
    /// Aqui, eles são definidos manualmente para refletir intuições linguísticas sobre o português.
    pub fn build() -> Self {
        let crf = build_crf_model();
        let mut rule_engine = build_rule_engine();
        // Os gazetteers alimentam tanto o motor de regras quanto a extração de features
        let gazetteers = build_gazetteers(&mut rule_engine);
        let corpus = get_corpus();

        // Treinamento rápido dos modelos secundários para demonstração
        let mut hmm = HmmModel::new();
        hmm.train(&corpus);

        let mut maxent = MaxEntModel::new();
        maxent.train(&corpus, 10, 0.1, 0.01);

        let mut perceptron = PerceptronModel::new();
        perceptron.train(&corpus, 5);

        let mut span = SpanModel::new();
        span.train(&corpus, 5);

        Self {
            crf,
            hmm,
            maxent,
            perceptron,
            span,
            rule_engine,
            gazetteers_cache: gazetteers,
        }
    }

    /// Retorna uma cópia dos gazetteers para uso no extrator de features.
    ///
    /// # Importância
    ///
    /// O extrator de features (`features.rs`) precisa saber quais palavras são
    /// entidades conhecidas para gerar features binárias como `in_person_gazetteer`.
    /// Este método provê acesso seguro a esses dados compartilhados.
    pub fn gazetteers(&self) -> Gazetteers {
        self.gazetteers_cache.clone()
    }
}

impl Default for NerModel {
    fn default() -> Self {
        Self::build()
    }
}

/// Constrói o modelo CRF com pesos heurísticos baseados no corpus.
///
/// Define manualmente a "importância" de cada feature para cada tag.
///
/// # Exemplos de Intuição
/// - Se a palavra está nos **Gazetteers de Pessoa**, a chance de ser `B-PER` aumenta muito (+5.0).
/// - Se a palavra começa com maiúscula (`is_capitalized`), há uma boa chance de ser uma entidade (+2.8).
/// - Se a palavra anterior for "Presidente", a próxima provavelmente é `B-PER` (+2.5).
fn build_crf_model() -> CrfModel {
    let mut model = CrfModel::new();

    // =====================================================================
    // PESOS DE EMISSÃO (Feature -> Tag)
    // =====================================================================

    // --- PESSOA (PER) ---
    // Capitalização é um forte indício, mas não garantia (início de frase).
    model.set_emission("is_capitalized", &Tag::Begin(EntityCategory::Per), 2.8);
    model.set_emission("is_capitalized", &Tag::Begin(EntityCategory::Org), 1.5);
    model.set_emission("is_capitalized", &Tag::Begin(EntityCategory::Loc), 1.5);

    // Presença em listas conhecidas (Gazetteers) é o sinal mais forte.
    model.set_emission("in_person_gazetteer", &Tag::Begin(EntityCategory::Per), 5.0);
    model.set_emission("in_person_gazetteer", &Tag::Inside(EntityCategory::Per), 4.5);
    model.set_emission("in_location_gazetteer", &Tag::Begin(EntityCategory::Loc), 5.0);
    model.set_emission("in_location_gazetteer", &Tag::Inside(EntityCategory::Loc), 4.5);
    model.set_emission("in_org_gazetteer", &Tag::Begin(EntityCategory::Org), 5.0);
    model.set_emission("in_org_gazetteer", &Tag::Inside(EntityCategory::Org), 4.5);
    model.set_emission("in_misc_gazetteer", &Tag::Begin(EntityCategory::Misc), 5.0);
    model.set_emission("in_misc_gazetteer", &Tag::Inside(EntityCategory::Misc), 4.5);

    // Sufixo "-inho", "-inha" → frequentemente apelidos de pessoas
    model.set_emission("suffix3=nho", &Tag::Begin(EntityCategory::Per), 1.0);
    model.set_emission("suffix3=nha", &Tag::Begin(EntityCategory::Per), 1.0);

    // Sufixo "ão" ou "ões" pode ser nome de pessoa ou lugar
    model.set_emission("suffix2=ão", &Tag::Begin(EntityCategory::Per), 0.5);
    model.set_emission("suffix2=ão", &Tag::Begin(EntityCategory::Loc), 0.5);

    // Palavra "presidente", "senador" etc. antes → feature de contexto
    model.set_emission("prev_word=presidente", &Tag::Begin(EntityCategory::Per), 2.5);
    model.set_emission("prev_word=governador", &Tag::Begin(EntityCategory::Per), 2.5);
    model.set_emission("prev_word=deputado", &Tag::Begin(EntityCategory::Per), 2.0);
    model.set_emission("prev_word=senador", &Tag::Begin(EntityCategory::Per), 2.0);
    model.set_emission("prev_word=ministro", &Tag::Begin(EntityCategory::Per), 2.0);
    model.set_emission("prev_word=ministra", &Tag::Begin(EntityCategory::Per), 2.0);
    model.set_emission("prev_word=jogador", &Tag::Begin(EntityCategory::Per), 1.8);
    model.set_emission("prev_word=atleta", &Tag::Begin(EntityCategory::Per), 1.8);
    model.set_emission("prev_word=dr", &Tag::Begin(EntityCategory::Per), 1.8);
    model.set_emission("prev_word=prof", &Tag::Begin(EntityCategory::Per), 1.8);
    model.set_emission("prev_word=general", &Tag::Begin(EntityCategory::Per), 1.8);
    model.set_emission("prev_word=escritor", &Tag::Begin(EntityCategory::Per), 1.5);
    model.set_emission("prev_word=ator", &Tag::Begin(EntityCategory::Per), 1.5);
    model.set_emission("prev_word=cantor", &Tag::Begin(EntityCategory::Per), 1.5);
    model.set_emission("prev_word=dom", &Tag::Begin(EntityCategory::Per), 2.0);

    // Prefixo comum de primeiro nome BR
    for prefix in &["lu", "ma", "jo", "an", "ca", "fe", "ro", "pe", "fa", "ri"] {
        model.set_emission(
            &format!("prefix2={prefix}"),
            &Tag::Begin(EntityCategory::Per),
            0.3,
        );
    }

    // --- ORGANIZAÇÃO (ORG) ---
    // Palavra após "da" ou "do" e capitalizada → frequentemente ORG ou LOC
    model.set_emission("prev_word=ministério", &Tag::Begin(EntityCategory::Org), 2.5);
    model.set_emission("prev_word=instituto", &Tag::Begin(EntityCategory::Org), 2.0);
    model.set_emission("prev_word=tribunal", &Tag::Begin(EntityCategory::Org), 2.0);
    model.set_emission("prev_word=empresa", &Tag::Begin(EntityCategory::Org), 1.5);
    model.set_emission("prev_word=clube", &Tag::Begin(EntityCategory::Org), 2.0);
    model.set_emission("prev_word=equipe", &Tag::Begin(EntityCategory::Org), 1.5);
    model.set_emission("prev_word=banco", &Tag::Begin(EntityCategory::Org), 2.0);
    model.set_emission("prev_word=universidade", &Tag::Begin(EntityCategory::Org), 2.0);
    model.set_emission("prev_word=startup", &Tag::Begin(EntityCategory::Org), 2.0);

    // Sufixo "-ras" como em "Petrobras", "Eletrobras"
    model.set_emission("suffix3=ras", &Tag::Begin(EntityCategory::Org), 1.8);
    // Sufixo "-itec" ou "-tech"
    model.set_emission("suffix3=ech", &Tag::Begin(EntityCategory::Org), 1.2);
    model.set_emission("suffix4=bank", &Tag::Begin(EntityCategory::Org), 2.0);

    // SIGLE / siglas: palavras todas maiúsculas com 2-5 chars → podem ser ORG ou MISC
    model.set_emission("is_all_caps", &Tag::Begin(EntityCategory::Org), 1.5);
    model.set_emission("is_all_caps", &Tag::Begin(EntityCategory::Misc), 1.0);

    // --- LOCALIZAÇÃO (LOC) ---
    model.set_emission("prev_word=cidade", &Tag::Begin(EntityCategory::Loc), 1.8);
    model.set_emission("prev_word=estado", &Tag::Begin(EntityCategory::Loc), 1.8);
    model.set_emission("prev_word=rio", &Tag::Begin(EntityCategory::Loc), 2.0);
    model.set_emission("prev_word=região", &Tag::Begin(EntityCategory::Loc), 1.5);
    model.set_emission("prev_word=fronteira", &Tag::Begin(EntityCategory::Loc), 1.5);
    model.set_emission("prev_word=município", &Tag::Begin(EntityCategory::Loc), 2.0);
    model.set_emission("prev_word=país", &Tag::Begin(EntityCategory::Loc), 1.8);
    model.set_emission("prev_word=floresta", &Tag::Begin(EntityCategory::Loc), 1.5);
    model.set_emission("prev_word=estádio", &Tag::Begin(EntityCategory::Loc), 2.0);
    model.set_emission("prev_word=palácio", &Tag::Begin(EntityCategory::Loc), 2.0);
    model.set_emission("prev_word=aeroporto", &Tag::Begin(EntityCategory::Loc), 2.0);
    model.set_emission("prev_word=em", &Tag::Begin(EntityCategory::Loc), 0.8);
    model.set_emission("prev_word=no", &Tag::Begin(EntityCategory::Loc), 0.8);
    model.set_emission("prev_word=na", &Tag::Begin(EntityCategory::Loc), 0.8);
    model.set_emission("prev_word=do", &Tag::Begin(EntityCategory::Loc), 0.5);
    model.set_emission("prev_word=da", &Tag::Begin(EntityCategory::Loc), 0.5);

    // Sufixos comuns de cidades/estados BR
    model.set_emission("suffix3=lis", &Tag::Begin(EntityCategory::Loc), 1.2); // Brasília, Fortaleza
    model.set_emission("suffix4=ília", &Tag::Begin(EntityCategory::Loc), 1.5);
    model.set_emission("suffix2=as", &Tag::Begin(EntityCategory::Loc), 0.4);

    // --- MISC ---
    model.set_emission("prev_word=copa", &Tag::Begin(EntityCategory::Misc), 2.0);
    model.set_emission("prev_word=campeonato", &Tag::Begin(EntityCategory::Misc), 2.0);
    model.set_emission("prev_word=taxa", &Tag::Begin(EntityCategory::Misc), 1.5);
    model.set_emission("prev_word=lei", &Tag::Begin(EntityCategory::Misc), 1.5);
    model.set_emission("prev_word=vírus", &Tag::Begin(EntityCategory::Misc), 1.8);
    model.set_emission("prev_word=vacina", &Tag::Begin(EntityCategory::Misc), 1.0);
    model.set_emission("prev_word=satélite", &Tag::Begin(EntityCategory::Misc), 1.8);
    model.set_emission("prev_word=operação", &Tag::Begin(EntityCategory::Misc), 1.5);
    model.set_emission("prev_word=fórmula", &Tag::Begin(EntityCategory::Misc), 2.0);

    // Palavra comum → Outside
    model.set_emission("BOS", &Tag::Outside, 0.5);
    model.set_emission("bias", &Tag::Outside, 1.0);

    // Pontuação → sempre Outside
    model.set_emission("is_punctuation", &Tag::Outside, 5.0);

    // Dígito puro → geralmente Outside (anos, números)
    model.set_emission("is_digit", &Tag::Outside, 2.0);

    // =====================================================================
    // PESOS DE TRANSIÇÃO
    // Capturam a regularidade das sequências BIO
    // =====================================================================

    let tags = Tag::all();

    // Penaliza fortemente todas as transições inválidas
    for prev in &tags {
        for next in &tags {
            if !Tag::is_valid_transition(prev, next) {
                model.set_transition(prev, next, -8.0);
            }
        }
    }

    // Transições válidas B→I da mesma categoria têm alto peso
    let categories = [
        EntityCategory::Per,
        EntityCategory::Org,
        EntityCategory::Loc,
        EntityCategory::Misc,
    ];
    for cat in &categories {
        let b = Tag::Begin(*cat);
        let i = Tag::Inside(*cat);
        model.set_transition(&b, &i, 4.0);   // B-PER → I-PER: muito provável
        model.set_transition(&i, &i, 3.5);   // I-PER → I-PER: "João da Silva"
        model.set_transition(&b, &Tag::Outside, 2.0); // entidade de um token
        model.set_transition(&i, &Tag::Outside, 2.5); // fim de entidade
        model.set_transition(&Tag::Outside, &b, 1.5); // início de nova entidade
    }

    // Outside → Outside é muito comum
    model.set_transition(&Tag::Outside, &Tag::Outside, 2.5);

    model
}

/// Constrói os gazetteers a partir do corpus e de listas manuais
fn build_gazetteers(rule_engine: &mut RuleEngine) -> Gazetteers {
    let (corpus_persons, corpus_locs, corpus_orgs, corpus_misc) =
        extract_gazetteers_from_corpus();

    let mut gaz = Gazetteers::new();

    // Inclui entidades do corpus
    for p in &corpus_persons {
        for word in p.split_whitespace() {
            if word.len() > 2 {
                gaz.persons.insert(word.to_lowercase());
                rule_engine.add_person(word);
            }
        }
        rule_engine.add_person(p);
    }
    for l in &corpus_locs {
        for word in l.split_whitespace() {
            if word.len() > 3 {
                gaz.locations.insert(word.to_lowercase());
            }
        }
        rule_engine.add_location(l);
    }
    for o in &corpus_orgs {
        for word in o.split_whitespace() {
            if word.len() > 3 {
                gaz.organizations.insert(word.to_lowercase());
            }
        }
        rule_engine.add_org(o);
    }
    for m in &corpus_misc {
        for word in m.split_whitespace() {
            if word.len() > 3 {
                gaz.misc.insert(word.to_lowercase());
            }
        }
        rule_engine.add_misc(m);
    }

    // Listas manuais estendidas — Políticos e figuras históricas do Brasil
    let extra_persons = vec![
        "Getúlio", "Vargas", "Juscelino", "Kubitschek", "Jânio", "Quadros",
        "Costa", "Silva", "Geisel", "Figueiredo", "Sarney", "Collor", "Itamar",
        "Franco", "Cardoso", "Rousseff", "Temer", "Bolsonaro", "Haddad",
        "Mantega", "Meirelles", "Guedes", "Ciro", "Alckmin", "Moro",
        "Senna", "Pelé", "Ronaldo", "Ronaldinho", "Zico", "Garrincha",
        "Neymar", "Vini", "Rodrygo", "Casemiro", "Marquinhos",
        "Gisele", "Bündchen", "Xuxa", "Ivete", "Sangalo", "Anitta",
        "Caetano", "Veloso", "Gilberto", "Gil", "Chico", "Buarque",
        "Machado", "Assis", "Guimarães", "Rosa", "Clarice", "Lispector",
        "Oswald", "Andrade", "Drummond", "Pessoa",
    ];
    for p in extra_persons {
        gaz.persons.insert(p.to_lowercase());
        rule_engine.add_person(p);
    }

    // Cidades e locais do Brasil
    let extra_locs = vec![
        "Brasília", "São Paulo", "Rio de Janeiro", "Salvador", "Fortaleza",
        "Manaus", "Curitiba", "Recife", "Porto Alegre", "Belém", "Goiânia",
        "Florianópolis", "Maceió", "Natal", "Teresina", "Campo Grande",
        "João Pessoa", "Aracaju", "Cuiabá", "Macapá", "Porto Velho",
        "Boa Vista", "Palmas", "Rio Branco", "Vitória", "São Luís",
        "Amazônia", "Pantanal", "Cerrado", "Caatinga", "Pampa",
        "Nordeste", "Sudeste", "Norte", "Sul", "Centro-Oeste",
        "Maracanã", "Itaquerão", "Arena", "Mineirão", "Beira-Rio",
        "Planalto", "Palácio", "Congresso", "Senado", "Câmara",
        "Supremo", "STF", "STJ", "TSE", "TRF",
        "Argentina", "Chile", "Colômbia", "Peru", "Venezuela", "Uruguai",
        "Paraguai", "Bolívia", "Equador", "Qatar", "Japão", "Coreia",
        "Alemanha", "França", "Espanha", "Portugal", "Itália", "Inglaterra",
        "Estados Unidos", "China", "Rússia", "Índia", "África",
        "Europa", "Ásia", "América", "Latina", "Caribe",
        "Ipiranga", "Tietê", "São Francisco", "Paraná", "Tocantins",
        "Xingu", "Negro", "Solimões", "Tapajós",
    ];
    for l in extra_locs {
        for word in l.split_whitespace() {
            if word.len() > 3 {
                gaz.locations.insert(word.to_lowercase());
            }
        }
        rule_engine.add_location(l);
    }

    // Organizações brasileiras
    let extra_orgs = vec![
        "Petrobras", "Vale", "Embraer", "Nubank", "Itaú", "Bradesco", "Santander",
        "Caixa", "Econômica", "Federal", "BNDES", "IBGE", "INPE", "Fiocruz",
        "Anvisa", "Anatel", "Aneel", "ANS", "ANP", "CADE",
        "Partidos", "PT", "PL", "MDB", "PSDB", "PDT", "PSB", "Republicanos",
        "Podemos", "União", "Brasil", "Solidariedade", "Avante",
        "Flamengo", "Palmeiras", "Corinthians", "São Paulo", "Grêmio",
        "Internacional", "Atlético", "Cruzeiro", "Fluminense", "Vasco",
        "Botafogo", "Santos", "Sport", "Bahia", "Ceará", "Fortaleza",
        "McLaren", "Ferrari", "Mercedes", "Red Bull", "Alpine",
        "ONU", "UNESCO", "UNICEF", "OMS", "FMI", "Banco Mundial",
        "BRICS", "Mercosul", "ALBA", "UNASUL", "CELAC",
        "FIFA", "CBF", "COI", "COB",
        "USP", "Unicamp", "UFRJ", "UnB", "UFMG", "UFRGS",
        "Globo", "Record", "SBT", "Band", "CNN Brasil", "UOL", "Folha",
        "Estadão", "O Globo", "Veja", "Época", "IstoÉ",
    ];
    for o in extra_orgs {
        for word in o.split_whitespace() {
            if word.len() > 2 {
                gaz.organizations.insert(word.to_lowercase());
            }
        }
        rule_engine.add_org(o);
    }

    // Miscelânea (eventos, produtos, leis, etc.)
    let extra_misc = vec![
        "Copa do Mundo", "Olimpíadas", "Jogos Olímpicos", "Paralímpicos",
        "Libertadores", "Copa América", "Europeu", "Champions League",
        "Fórmula 1", "MotoGP", "Rally Dakar",
        "Carnaval", "Réveillon", "Natal", "São João", "Festa Junina",
        "COVID-19", "Dengue", "Febre Amarela", "Zika", "Malária",
        "PIB", "Selic", "IPCA", "IBOV", "FGTS", "INSS", "SUS",
        "Constituição", "Marco Civil", "Lei Maria da Penha", "ECA",
        "Operação Lava Jato", "Mensalão", "Privatizações",
        "Independência", "República", "Proclamação", "Abolição",
        "Inconfidência Mineira", "Revolução de 1930", "AI-5",
        "Amazônia-1", "SGDC", "VLS",
        "Gabriela Cravo e Canela", "Grande Sertão Veredas",
    ];
    for m in extra_misc {
        for word in m.split_whitespace() {
            if word.len() > 3 {
                gaz.misc.insert(word.to_lowercase());
            }
        }
        rule_engine.add_misc(m);
    }

    gaz
}

/// Constrói o motor de regras base (sem gazetteers, que são adicionados depois)
fn build_rule_engine() -> RuleEngine {
    RuleEngine::new()
}
