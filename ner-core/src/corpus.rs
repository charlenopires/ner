//! # Corpus em Português Brasileiro com Anotações BIO
//!
//! Corpus de texto anotado manualmente cobrindo domínios temáticos do Brasil.
//! Cada sentença está anotada no formato BIO para treinamento e demonstração do NER.
//!
//! ## Domínios Cobertos
//! - Saúde e medicina
//! - Bem-estar e qualidade de vida
//! - Religião e espiritualidade
//! - História do Brasil
//! - Economia e negócios
//! - Esportes
//! - Ciência e tecnologia
//! - Cultura e entretenimento
//! - Meio ambiente
//! - Educação

/// Uma sentença anotada no formato BIO
///
/// O formato BIO (Begin, Inside, Outside) é padrão para NER:
/// - **B-TYPE**: Início de uma entidade do tipo TYPE.
/// - **I-TYPE**: Continuação de uma entidade do tipo TYPE.
/// - **O**: Fora de qualquer entidade.
pub struct AnnotatedSentence {
    /// O texto completo da sentença (idealmente sem tokenização prévia,
    /// mas aqui já estruturado para facilitar).
    pub text: &'static str,
    /// Domínio temático (utilizado para análises de performance por área).
    pub domain: &'static str,
    /// Pares (palavra, tag_BIO).
    /// Exemplo: `[("Lula", "B-PER"), ("viajou", "O")]`
    pub annotations: &'static [(&'static str, &'static str)],
}

/// Retorna o corpus completo em PT-BR
pub fn get_corpus() -> Vec<AnnotatedSentence> {
    vec![
        // ===== SAÚDE =====
        AnnotatedSentence {
            text: "A Fiocruz desenvolveu a vacina contra a dengue aprovada pela Anvisa em 2023.",
            domain: "saúde",
            annotations: &[
                ("A", "O"), ("Fiocruz", "B-ORG"), ("desenvolveu", "O"), ("a", "O"),
                ("vacina", "O"), ("contra", "O"), ("a", "O"), ("dengue", "B-MISC"),
                ("aprovada", "O"), ("pela", "O"), ("Anvisa", "B-ORG"), ("em", "O"), ("2023", "O"), (".", "O"),
            ],
        },
        AnnotatedSentence {
            text: "O Hospital Albert Einstein em São Paulo é referência em cardiologia e oncologia no Brasil.",
            domain: "saúde",
            annotations: &[
                ("O", "O"), ("Hospital", "B-ORG"), ("Albert", "I-ORG"), ("Einstein", "I-ORG"),
                ("em", "O"), ("São", "B-LOC"), ("Paulo", "I-LOC"), ("é", "O"),
                ("referência", "O"), ("em", "O"), ("cardiologia", "O"), ("e", "O"),
                ("oncologia", "O"), ("no", "O"), ("Brasil", "B-LOC"), (".", "O"),
            ],
        },
        AnnotatedSentence {
            text: "A pesquisadora Margareth Dalcolmo foi um dos principais rostos da ciência durante a pandemia de Covid-19.",
            domain: "saúde",
            annotations: &[
                ("A", "O"), ("pesquisadora", "O"),
                ("Margareth", "B-PER"), ("Dalcolmo", "I-PER"),
                ("foi", "O"), ("um", "O"), ("dos", "O"), ("principais", "O"),
                ("rostos", "O"), ("da", "O"), ("ciência", "O"), ("durante", "O"),
                ("a", "O"), ("pandemia", "O"), ("de", "O"), ("Covid-19", "B-MISC"), (".", "O"),
            ],
        },
        AnnotatedSentence {
            text: "O Instituto Butantan é responsável por produzir milhões de doses de vacinas para o Sistema Único de Saúde.",
            domain: "saúde",
            annotations: &[
                ("O", "O"), ("Instituto", "B-ORG"), ("Butantan", "I-ORG"),
                ("é", "O"), ("responsável", "O"), ("por", "O"), ("produzir", "O"),
                ("milhões", "O"), ("de", "O"), ("doses", "O"), ("de", "O"),
                ("vacinas", "O"), ("para", "O"), ("o", "O"),
                ("Sistema", "B-ORG"), ("Único", "I-ORG"), ("de", "I-ORG"), ("Saúde", "I-ORG"), (".", "O"),
            ],
        },
        AnnotatedSentence {
            text: "O médico Drauzio Varella é um dos mais conhecidos divulgadores científicos do Brasil.",
            domain: "saúde",
            annotations: &[
                ("O", "O"), ("médico", "O"), ("Drauzio", "B-PER"), ("Varella", "I-PER"),
                ("é", "O"), ("um", "O"), ("dos", "O"), ("mais", "O"), ("conhecidos", "O"),
                ("divulgadores", "O"), ("científicos", "O"), ("do", "O"), ("Brasil", "B-LOC"), (".", "O"),
            ],
        },
        AnnotatedSentence {
            text: "A Organização Mundial da Saúde declarou o fim da emergência global da Covid-19 em maio de 2023.",
            domain: "saúde",
            annotations: &[
                ("A", "O"), ("Organização", "B-ORG"), ("Mundial", "I-ORG"), ("da", "I-ORG"), ("Saúde", "I-ORG"),
                ("declarou", "O"), ("o", "O"), ("fim", "O"), ("da", "O"), ("emergência", "O"),
                ("global", "O"), ("da", "O"), ("Covid-19", "B-MISC"), ("em", "O"),
                ("maio", "O"), ("de", "O"), ("2023", "O"), (".", "O"),
            ],
        },

        // ===== BEM-ESTAR =====
        AnnotatedSentence {
            text: "A prática do yoga e da meditação tem crescido entre os brasileiros nos últimos anos.",
            domain: "bem-estar",
            annotations: &[
                ("A", "O"), ("prática", "O"), ("do", "O"), ("yoga", "B-MISC"), ("e", "O"),
                ("da", "O"), ("meditação", "O"), ("tem", "O"), ("crescido", "O"),
                ("entre", "O"), ("os", "O"), ("brasileiros", "O"), ("nos", "O"),
                ("últimos", "O"), ("anos", "O"), (".", "O"),
            ],
        },
        AnnotatedSentence {
            text: "O Centro de Bem-Estar Animal de Curitiba oferece atendimento veterinário gratuito à população.",
            domain: "bem-estar",
            annotations: &[
                ("O", "O"), ("Centro", "B-ORG"), ("de", "I-ORG"), ("Bem-Estar", "I-ORG"),
                ("Animal", "I-ORG"), ("de", "O"), ("Curitiba", "B-LOC"),
                ("oferece", "O"), ("atendimento", "O"), ("veterinário", "O"),
                ("gratuito", "O"), ("à", "O"), ("população", "O"), (".", "O"),
            ],
        },
        AnnotatedSentence {
            text: "A nutricionista Ana Paula Torres recomenda a dieta mediterrânea para a prevenção de doenças cardiovasculares.",
            domain: "bem-estar",
            annotations: &[
                ("A", "O"), ("nutricionista", "O"),
                ("Ana", "B-PER"), ("Paula", "I-PER"), ("Torres", "I-PER"),
                ("recomenda", "O"), ("a", "O"), ("dieta", "O"), ("mediterrânea", "B-MISC"),
                ("para", "O"), ("a", "O"), ("prevenção", "O"), ("de", "O"),
                ("doenças", "O"), ("cardiovasculares", "O"), (".", "O"),
            ],
        },
        AnnotatedSentence {
            text: "O Parque Estadual da Cantareira em São Paulo é ideal para trilhas e reconexão com a natureza.",
            domain: "bem-estar",
            annotations: &[
                ("O", "O"), ("Parque", "B-LOC"), ("Estadual", "I-LOC"), ("da", "I-LOC"), ("Cantareira", "I-LOC"),
                ("em", "O"), ("São", "B-LOC"), ("Paulo", "I-LOC"),
                ("é", "O"), ("ideal", "O"), ("para", "O"), ("trilhas", "O"), ("e", "O"),
                ("reconexão", "O"), ("com", "O"), ("a", "O"), ("natureza", "O"), (".", "O"),
            ],
        },

        // ===== RELIGIÃO E ESPIRITUALIDADE =====
        AnnotatedSentence {
            text: "Nossa Senhora de Aparecida é a padroeira do Brasil, venerada em Aparecida do Norte no estado de São Paulo.",
            domain: "religião",
            annotations: &[
                ("Nossa", "B-PER"), ("Senhora", "I-PER"), ("de", "I-PER"), ("Aparecida", "I-PER"),
                ("é", "O"), ("a", "O"), ("padroeira", "O"), ("do", "O"), ("Brasil", "B-LOC"),
                (",", "O"), ("venerada", "O"), ("em", "O"),
                ("Aparecida", "B-LOC"), ("do", "I-LOC"), ("Norte", "I-LOC"),
                ("no", "O"), ("estado", "O"), ("de", "O"), ("São", "B-LOC"), ("Paulo", "I-LOC"), (".", "O"),
            ],
        },
        AnnotatedSentence {
            text: "O Candomblé é uma das religiões de matriz africana mais praticadas no Brasil, especialmente na Bahia.",
            domain: "religião",
            annotations: &[
                ("O", "O"), ("Candomblé", "B-MISC"), ("é", "O"), ("uma", "O"), ("das", "O"),
                ("religiões", "O"), ("de", "O"), ("matriz", "O"), ("africana", "O"),
                ("mais", "O"), ("praticadas", "O"), ("no", "O"), ("Brasil", "B-LOC"),
                (",", "O"), ("especialmente", "O"), ("na", "O"), ("Bahia", "B-LOC"), (".", "O"),
            ],
        },
        AnnotatedSentence {
            text: "O padre Fábio de Melo é um dos sacerdotes mais populares do Brasil e autor de diversos livros espirituais.",
            domain: "religião",
            annotations: &[
                ("O", "O"), ("padre", "O"), ("Fábio", "B-PER"), ("de", "I-PER"), ("Melo", "I-PER"),
                ("é", "O"), ("um", "O"), ("dos", "O"), ("sacerdotes", "O"), ("mais", "O"),
                ("populares", "O"), ("do", "O"), ("Brasil", "B-LOC"), ("e", "O"),
                ("autor", "O"), ("de", "O"), ("diversos", "O"), ("livros", "O"), ("espirituais", "O"), (".", "O"),
            ],
        },
        AnnotatedSentence {
            text: "A Basílica de Nossa Senhora de Nazaré em Belém recebe milhões de fiéis durante o Círio de Nazaré.",
            domain: "religião",
            annotations: &[
                ("A", "O"), ("Basílica", "B-LOC"), ("de", "I-LOC"), ("Nossa", "I-LOC"),
                ("Senhora", "I-LOC"), ("de", "I-LOC"), ("Nazaré", "I-LOC"),
                ("em", "O"), ("Belém", "B-LOC"), ("recebe", "O"), ("milhões", "O"),
                ("de", "O"), ("fiéis", "O"), ("durante", "O"), ("o", "O"),
                ("Círio", "B-MISC"), ("de", "I-MISC"), ("Nazaré", "I-MISC"), (".", "O"),
            ],
        },
        AnnotatedSentence {
            text: "A Umbanda surgiu no Brasil no início do século XX, combinando elementos do Candomblé, do Espiritismo e do catolicismo.",
            domain: "religião",
            annotations: &[
                ("A", "O"), ("Umbanda", "B-MISC"), ("surgiu", "O"), ("no", "O"), ("Brasil", "B-LOC"),
                ("no", "O"), ("início", "O"), ("do", "O"), ("século", "O"), ("XX", "O"),
                (",", "O"), ("combinando", "O"), ("elementos", "O"), ("do", "O"),
                ("Candomblé", "B-MISC"), (",", "O"), ("do", "O"), ("Espiritismo", "B-MISC"),
                ("e", "O"), ("do", "O"), ("catolicismo", "O"), (".", "O"),
            ],
        },
        AnnotatedSentence {
            text: "Allan Kardec codificou o Espiritismo na França no século XIX, obra que se tornou base para o espiritismo brasileiro.",
            domain: "religião",
            annotations: &[
                ("Allan", "B-PER"), ("Kardec", "I-PER"), ("codificou", "O"),
                ("o", "O"), ("Espiritismo", "B-MISC"), ("na", "O"), ("França", "B-LOC"),
                ("no", "O"), ("século", "O"), ("XIX", "O"), (",", "O"), ("obra", "O"),
                ("que", "O"), ("se", "O"), ("tornou", "O"), ("base", "O"),
                ("para", "O"), ("o", "O"), ("espiritismo", "O"), ("brasileiro", "O"), (".", "O"),
            ],
        },

        // ===== HISTÓRIA DO BRASIL =====
        AnnotatedSentence {
            text: "Dom Pedro I proclamou a Independência do Brasil às margens do Rio Ipiranga em 1822.",
            domain: "história",
            annotations: &[
                ("Dom", "B-PER"), ("Pedro", "I-PER"), ("I", "I-PER"), ("proclamou", "O"), ("a", "O"),
                ("Independência", "B-MISC"), ("do", "I-MISC"), ("Brasil", "I-MISC"),
                ("às", "O"), ("margens", "O"), ("do", "O"), ("Rio", "B-LOC"), ("Ipiranga", "I-LOC"),
                ("em", "O"), ("1822", "O"), (".", "O"),
            ],
        },
        AnnotatedSentence {
            text: "Tiradentes foi enforcado em 21 de abril de 1792 no Rio de Janeiro por liderar a Inconfidência Mineira.",
            domain: "história",
            annotations: &[
                ("Tiradentes", "B-PER"), ("foi", "O"), ("enforcado", "O"), ("em", "O"),
                ("21", "O"), ("de", "O"), ("abril", "O"), ("de", "O"), ("1792", "O"),
                ("no", "O"), ("Rio", "B-LOC"), ("de", "I-LOC"), ("Janeiro", "I-LOC"),
                ("por", "O"), ("liderar", "O"), ("a", "O"),
                ("Inconfidência", "B-MISC"), ("Mineira", "I-MISC"), (".", "O"),
            ],
        },
        AnnotatedSentence {
            text: "Zumbi dos Palmares foi o líder do Quilombo dos Palmares e símbolo da resistência negra no Brasil colonial.",
            domain: "história",
            annotations: &[
                ("Zumbi", "B-PER"), ("dos", "I-PER"), ("Palmares", "I-PER"), ("foi", "O"),
                ("o", "O"), ("líder", "O"), ("do", "O"), ("Quilombo", "B-LOC"),
                ("dos", "I-LOC"), ("Palmares", "I-LOC"), ("e", "O"), ("símbolo", "O"),
                ("da", "O"), ("resistência", "O"), ("negra", "O"), ("no", "O"),
                ("Brasil", "B-LOC"), ("colonial", "O"), (".", "O"),
            ],
        },
        AnnotatedSentence {
            text: "A Semana de Arte Moderna de 1922 em São Paulo marcou o início do Modernismo na cultura brasileira.",
            domain: "história",
            annotations: &[
                ("A", "O"), ("Semana", "B-MISC"), ("de", "I-MISC"), ("Arte", "I-MISC"),
                ("Moderna", "I-MISC"), ("de", "O"), ("1922", "O"), ("em", "O"),
                ("São", "B-LOC"), ("Paulo", "I-LOC"), ("marcou", "O"), ("o", "O"),
                ("início", "O"), ("do", "O"), ("Modernismo", "B-MISC"),
                ("na", "O"), ("cultura", "O"), ("brasileira", "O"), (".", "O"),
            ],
        },
        AnnotatedSentence {
            text: "Getúlio Vargas governou o Brasil em dois períodos distintos e criou a Consolidação das Leis do Trabalho.",
            domain: "história",
            annotations: &[
                ("Getúlio", "B-PER"), ("Vargas", "I-PER"), ("governou", "O"), ("o", "O"),
                ("Brasil", "B-LOC"), ("em", "O"), ("dois", "O"), ("períodos", "O"),
                ("distintos", "O"), ("e", "O"), ("criou", "O"), ("a", "O"),
                ("Consolidação", "B-MISC"), ("das", "I-MISC"), ("Leis", "I-MISC"),
                ("do", "I-MISC"), ("Trabalho", "I-MISC"), (".", "O"),
            ],
        },
        AnnotatedSentence {
            text: "Princesa Isabel assinou a Lei Áurea em 13 de maio de 1888, abolindo a escravidão no Brasil.",
            domain: "história",
            annotations: &[
                ("Princesa", "O"), ("Isabel", "B-PER"), ("assinou", "O"), ("a", "O"),
                ("Lei", "B-MISC"), ("Áurea", "I-MISC"), ("em", "O"), ("13", "O"),
                ("de", "O"), ("maio", "O"), ("de", "O"), ("1888", "O"), (",", "O"),
                ("abolindo", "O"), ("a", "O"), ("escravidão", "O"), ("no", "O"), ("Brasil", "B-LOC"), (".", "O"),
            ],
        },
        AnnotatedSentence {
            text: "Santos Dumont realizou o primeiro voo reconhecido da história com o 14-Bis em Paris em 1906.",
            domain: "história",
            annotations: &[
                ("Santos", "B-PER"), ("Dumont", "I-PER"), ("realizou", "O"), ("o", "O"),
                ("primeiro", "O"), ("voo", "O"), ("reconhecido", "O"), ("da", "O"),
                ("história", "O"), ("com", "O"), ("o", "O"),
                ("14-Bis", "B-MISC"), ("em", "O"), ("Paris", "B-LOC"), ("em", "O"), ("1906", "O"), (".", "O"),
            ],
        },

        // ===== ECONOMIA =====
        AnnotatedSentence {
            text: "A Petrobras anunciou lucro recorde de 50 bilhões de reais no terceiro trimestre.",
            domain: "economia",
            annotations: &[
                ("A", "O"), ("Petrobras", "B-ORG"), ("anunciou", "O"), ("lucro", "O"), ("recorde", "O"),
                ("de", "O"), ("50", "O"), ("bilhões", "O"), ("de", "O"), ("reais", "O"),
                ("no", "O"), ("terceiro", "O"), ("trimestre", "O"), (".", "O"),
            ],
        },
        AnnotatedSentence {
            text: "O Banco Central do Brasil manteve a taxa Selic em 10,5% ao ano.",
            domain: "economia",
            annotations: &[
                ("O", "O"), ("Banco", "B-ORG"), ("Central", "I-ORG"), ("do", "I-ORG"), ("Brasil", "I-ORG"),
                ("manteve", "O"), ("a", "O"), ("taxa", "O"), ("Selic", "B-MISC"),
                ("em", "O"), ("10,5%", "O"), ("ao", "O"), ("ano", "O"), (".", "O"),
            ],
        },
        AnnotatedSentence {
            text: "A Vale é a maior mineradora do Brasil e uma das maiores do mundo.",
            domain: "economia",
            annotations: &[
                ("A", "O"), ("Vale", "B-ORG"), ("é", "O"), ("a", "O"), ("maior", "O"),
                ("mineradora", "O"), ("do", "O"), ("Brasil", "B-LOC"), ("e", "O"), ("uma", "O"),
                ("das", "O"), ("maiores", "O"), ("do", "O"), ("mundo", "O"), (".", "O"),
            ],
        },
        AnnotatedSentence {
            text: "A Embraer assinou contrato com a Boeing para fornecimento de peças aeronáuticas.",
            domain: "economia",
            annotations: &[
                ("A", "O"), ("Embraer", "B-ORG"), ("assinou", "O"), ("contrato", "O"),
                ("com", "O"), ("a", "O"), ("Boeing", "B-ORG"), ("para", "O"), ("fornecimento", "O"),
                ("de", "O"), ("peças", "O"), ("aeronáuticas", "O"), (".", "O"),
            ],
        },

        // ===== ESPORTES =====
        AnnotatedSentence {
            text: "Pelé é considerado o maior jogador de futebol de todos os tempos.",
            domain: "esportes",
            annotations: &[
                ("Pelé", "B-PER"), ("é", "O"), ("considerado", "O"), ("o", "O"), ("maior", "O"),
                ("jogador", "O"), ("de", "O"), ("futebol", "O"), ("de", "O"), ("todos", "O"),
                ("os", "O"), ("tempos", "O"), (".", "O"),
            ],
        },
        AnnotatedSentence {
            text: "O Flamengo venceu o Fluminense por 3 a 1 no Maracanã pelo Campeonato Brasileiro.",
            domain: "esportes",
            annotations: &[
                ("O", "O"), ("Flamengo", "B-ORG"), ("venceu", "O"), ("o", "O"),
                ("Fluminense", "B-ORG"), ("por", "O"), ("3", "O"), ("a", "O"), ("1", "O"),
                ("no", "O"), ("Maracanã", "B-LOC"), ("pelo", "O"), ("Campeonato", "B-MISC"),
                ("Brasileiro", "I-MISC"), (".", "O"),
            ],
        },
        AnnotatedSentence {
            text: "Ayrton Senna foi tricampeão mundial de Fórmula 1 pela equipe McLaren.",
            domain: "esportes",
            annotations: &[
                ("Ayrton", "B-PER"), ("Senna", "I-PER"), ("foi", "O"), ("tricampeão", "O"),
                ("mundial", "O"), ("de", "O"), ("Fórmula", "B-MISC"), ("1", "I-MISC"),
                ("pela", "O"), ("equipe", "O"), ("McLaren", "B-ORG"), (".", "O"),
            ],
        },
        AnnotatedSentence {
            text: "Beatriz Souza conquistou a medalha de ouro no judô nos Jogos Olímpicos de Paris em 2024.",
            domain: "esportes",
            annotations: &[
                ("Beatriz", "B-PER"), ("Souza", "I-PER"), ("conquistou", "O"), ("a", "O"),
                ("medalha", "O"), ("de", "O"), ("ouro", "O"), ("no", "O"), ("judô", "O"),
                ("nos", "O"), ("Jogos", "B-MISC"), ("Olímpicos", "I-MISC"), ("de", "O"),
                ("Paris", "B-LOC"), ("em", "O"), ("2024", "O"), (".", "O"),
            ],
        },

        // ===== CIÊNCIA E TECNOLOGIA =====
        AnnotatedSentence {
            text: "O Instituto Nacional de Pesquisas Espaciais lançou o satélite Amazônia-1 em órbita.",
            domain: "ciência",
            annotations: &[
                ("O", "O"), ("Instituto", "B-ORG"), ("Nacional", "I-ORG"), ("de", "I-ORG"),
                ("Pesquisas", "I-ORG"), ("Espaciais", "I-ORG"), ("lançou", "O"), ("o", "O"),
                ("satélite", "O"), ("Amazônia-1", "B-MISC"), ("em", "O"), ("órbita", "O"), (".", "O"),
            ],
        },
        AnnotatedSentence {
            text: "A Universidade de São Paulo é a melhor instituição de ensino superior da América Latina.",
            domain: "educação",
            annotations: &[
                ("A", "O"), ("Universidade", "B-ORG"), ("de", "I-ORG"), ("São", "I-ORG"), ("Paulo", "I-ORG"),
                ("é", "O"), ("a", "O"), ("melhor", "O"), ("instituição", "O"), ("de", "O"),
                ("ensino", "O"), ("superior", "O"), ("da", "O"), ("América", "B-LOC"), ("Latina", "I-LOC"), (".", "O"),
            ],
        },
        AnnotatedSentence {
            text: "A startup brasileira Nubank se tornou o maior banco digital do mundo com mais de 90 milhões de clientes.",
            domain: "tecnologia",
            annotations: &[
                ("A", "O"), ("startup", "O"), ("brasileira", "O"), ("Nubank", "B-ORG"),
                ("se", "O"), ("tornou", "O"), ("o", "O"), ("maior", "O"), ("banco", "O"),
                ("digital", "O"), ("do", "O"), ("mundo", "O"), ("com", "O"), ("mais", "O"),
                ("de", "O"), ("90", "O"), ("milhões", "O"), ("de", "O"), ("clientes", "O"), (".", "O"),
            ],
        },

        // ===== CULTURA =====
        AnnotatedSentence {
            text: "Jorge Amado foi um dos maiores escritores brasileiros, autor de Gabriela, Cravo e Canela.",
            domain: "cultura",
            annotations: &[
                ("Jorge", "B-PER"), ("Amado", "I-PER"), ("foi", "O"), ("um", "O"), ("dos", "O"),
                ("maiores", "O"), ("escritores", "O"), ("brasileiros", "O"), (",", "O"),
                ("autor", "O"), ("de", "O"), ("Gabriela", "B-MISC"), (",", "O"),
                ("Cravo", "I-MISC"), ("e", "I-MISC"), ("Canela", "I-MISC"), (".", "O"),
            ],
        },
        AnnotatedSentence {
            text: "Carmen Miranda representou o Brasil no cinema americano nas décadas de 1940 e 1950.",
            domain: "cultura",
            annotations: &[
                ("Carmen", "B-PER"), ("Miranda", "I-PER"), ("representou", "O"), ("o", "O"),
                ("Brasil", "B-LOC"), ("no", "O"), ("cinema", "O"), ("americano", "O"),
                ("nas", "O"), ("décadas", "O"), ("de", "O"), ("1940", "O"), ("e", "O"), ("1950", "O"), (".", "O"),
            ],
        },

        // ===== MEIO AMBIENTE =====
        AnnotatedSentence {
            text: "O desmatamento da Floresta Amazônica atingiu 11 mil km² em 2022, segundo o INPE.",
            domain: "meio ambiente",
            annotations: &[
                ("O", "O"), ("desmatamento", "O"), ("da", "O"), ("Floresta", "B-LOC"),
                ("Amazônica", "I-LOC"), ("atingiu", "O"), ("11", "O"), ("mil", "O"), ("km²", "O"),
                ("em", "O"), ("2022", "O"), (",", "O"), ("segundo", "O"), ("o", "O"), ("INPE", "B-ORG"), (".", "O"),
            ],
        },
        AnnotatedSentence {
            text: "O Rio São Francisco corta seis estados brasileiros e é vital para o Nordeste.",
            domain: "meio ambiente",
            annotations: &[
                ("O", "O"), ("Rio", "B-LOC"), ("São", "I-LOC"), ("Francisco", "I-LOC"),
                ("corta", "O"), ("seis", "O"), ("estados", "O"), ("brasileiros", "O"),
                ("e", "O"), ("é", "O"), ("vital", "O"), ("para", "O"), ("o", "O"), ("Nordeste", "B-LOC"), (".", "O"),
            ],
        },
        // ===== DESAMBIGUAÇÃO =====
        AnnotatedSentence {
            text: "Paris Hilton viajou para Paris na França para participar de um desfile de moda.",
            domain: "desambiguação",
            annotations: &[
                ("Paris", "B-PER"), ("Hilton", "I-PER"), ("viajou", "O"), ("para", "O"),
                ("Paris", "B-LOC"), ("na", "O"), ("França", "B-LOC"), ("para", "O"),
                ("participar", "O"), ("de", "O"), ("um", "O"), ("desfile", "O"), ("de", "O"), ("moda", "O"), (".", "O"),
            ],
        },
        AnnotatedSentence {
            text: "O Banco do Brasil emprestou dinheiro para seu João sentar no banco da praça.",
            domain: "desambiguação",
            annotations: &[
                ("O", "O"), ("Banco", "B-ORG"), ("do", "I-ORG"), ("Brasil", "I-ORG"),
                ("emprestou", "O"), ("dinheiro", "O"), ("para", "O"), ("seu", "O"),
                ("João", "B-PER"), ("sentar", "O"), ("no", "O"), ("banco", "O"),
                ("da", "O"), ("praça", "O"), (".", "O"),
            ],
        },
        AnnotatedSentence {
            text: "O Estado do Rio de Janeiro declarou estado de calamidade.",
            domain: "desambiguação",
            annotations: &[
                ("O", "O"), ("Estado", "B-ORG"), ("do", "I-ORG"),
                ("Rio", "I-ORG"), ("de", "I-ORG"), ("Janeiro", "I-ORG"), ("declarou", "O"),
                ("estado", "O"), ("de", "O"), ("calamidade", "O"), (".", "O"),
            ],
        },
    ]
}

/// Extrai gazetteers do corpus: conjuntos de entidades conhecidas por categoria
///
/// Varre todo o corpus de treinamento e constrói listas (sets) de nomes conhecidos.
/// Isso é usado para criar features binárias poderosas (ex: "está_no_gazetteer_de_pessoas?").
///
/// # Retorno
/// Tupla contendo vetores de strings para:
/// (Pessoas, Locais, Organizações, Miscelânea)
pub fn extract_gazetteers_from_corpus() -> (
    Vec<String>, // persons
    Vec<String>, // locations
    Vec<String>, // orgs
    Vec<String>, // misc
) {
    let corpus = get_corpus();
    let mut persons = std::collections::HashSet::new();
    let mut locations = std::collections::HashSet::new();
    let mut orgs = std::collections::HashSet::new();
    let mut misc = std::collections::HashSet::new();

    for sentence in &corpus {
        let mut entity_tokens: Vec<&str> = vec![];
        let mut current_type = "";

        for (word, tag) in sentence.annotations {
            match *tag {
                "B-PER" => {
                    if !entity_tokens.is_empty() {
                        let entity = entity_tokens.join(" ").to_lowercase();
                        match current_type {
                            "PER" => { persons.insert(entity); }
                            "LOC" => { locations.insert(entity); }
                            "ORG" => { orgs.insert(entity); }
                            "MISC" => { misc.insert(entity); }
                            _ => {}
                        }
                    }
                    entity_tokens = vec![word];
                    current_type = "PER";
                }
                "B-LOC" => {
                    if !entity_tokens.is_empty() {
                        let entity = entity_tokens.join(" ").to_lowercase();
                        match current_type {
                            "PER" => { persons.insert(entity); }
                            "LOC" => { locations.insert(entity); }
                            "ORG" => { orgs.insert(entity); }
                            "MISC" => { misc.insert(entity); }
                            _ => {}
                        }
                    }
                    entity_tokens = vec![word];
                    current_type = "LOC";
                }
                "B-ORG" => {
                    if !entity_tokens.is_empty() {
                        let entity = entity_tokens.join(" ").to_lowercase();
                        match current_type {
                            "PER" => { persons.insert(entity); }
                            "LOC" => { locations.insert(entity); }
                            "ORG" => { orgs.insert(entity); }
                            "MISC" => { misc.insert(entity); }
                            _ => {}
                        }
                    }
                    entity_tokens = vec![word];
                    current_type = "ORG";
                }
                "B-MISC" => {
                    if !entity_tokens.is_empty() {
                        let entity = entity_tokens.join(" ").to_lowercase();
                        match current_type {
                            "PER" => { persons.insert(entity); }
                            "LOC" => { locations.insert(entity); }
                            "ORG" => { orgs.insert(entity); }
                            "MISC" => { misc.insert(entity); }
                            _ => {}
                        }
                    }
                    entity_tokens = vec![word];
                    current_type = "MISC";
                }
                tag if tag.starts_with("I-") => {
                    entity_tokens.push(word);
                }
                _ => {
                    if !entity_tokens.is_empty() {
                        let entity = entity_tokens.join(" ").to_lowercase();
                        match current_type {
                            "PER" => { persons.insert(entity); }
                            "LOC" => { locations.insert(entity); }
                            "ORG" => { orgs.insert(entity); }
                            "MISC" => { misc.insert(entity); }
                            _ => {}
                        }
                        entity_tokens = vec![];
                        current_type = "";
                    }
                }
            }
        }
    }

    (
        persons.into_iter().collect(),
        locations.into_iter().collect(),
        orgs.into_iter().collect(),
        misc.into_iter().collect(),
    )
}

/// Textos de demonstração para a interface web
pub fn demo_texts() -> Vec<(&'static str, &'static str)> {
    vec![
        (
            "Saúde",
            "O Hospital Albert Einstein, localizado no Morumbi em São Paulo, é uma referência internacional em medicina de alta complexidade. Juntamente com o Instituto Butantan e a Fundação Oswaldo Cruz (Fiocruz), a instituição tem liderado pesquisas inovadoras no combate a doenças tropicais. A Agência Nacional de Vigilância Sanitária (Anvisa) aprovou recentemente novos protocolos clínicos densenvolvidos pela pesquisadora Margareth Dalcolmo para o tratamento de variantes da Covid-19.",
        ),
        (
            "História",
            "Em 7 de setembro de 1822, Dom Pedro I proclamou a Independência do Brasil às margens do Rio Ipiranga. Décadas mais tarde, a Princesa Isabel sancionou a Lei Áurea em 13 de maio de 1888, encerrando oficialmente o ciclo da escravidão no país. Figuras como Zumbi dos Palmares, líder do maior quilombo das Américas, e Tiradentes, mártir da Inconfidência Mineira, são celebrados como heróis nacionais que lutaram pela liberdade e justiça social.",
        ),
        (
            "Tecnologia",
            "A startup brasileira Nubank, fundada por David Vélez, Cristina Junqueira e Edward Wible, revolucionou o setor bancário na América Latina. Com sede em São Paulo, a empresa expandiu operações para o México e Colômbia, alcançando mais de 90 milhões de clientes. Recentemente, a Embraer anunciou uma parceria estratégica com a Boeing para o desenvolvimento de combustíveis sustentáveis de aviação, reforçando a posição do Brasil como líder em tecnologia aeroespacial.",
        ),
        (
            "Cultura",
            "A Semana de Arte Moderna de 1922, realizada no Theatro Municipal de São Paulo, contou com a participação de Mário de Andrade, Oswald de Andrade e Tarsila do Amaral. O evento marcou o início do Modernismo no Brasil, rompendo com o conservadorismo acadêmico. Na música, Heitor Villa-Lobos e, posteriormente, Carmen Miranda, levaram a identidade cultural brasileira para os palcos internacionais, consolidando o samba e a bossa nova como gêneros de exportação.",
        ),
        (
            "Desambiguação",
            "A socialite Paris Hilton viajou para Paris, capital da França, na última semana. Durante a viagem, ela sentou em um banco próximo à Torre Eiffel após autorizar saques em sua conta no Banco do Brasil. O porta-voz do Estado do Rio de Janeiro confirmou que o estado de calamidade pública impede o envio de representantes.",
        ),
        (
            "Tokenização",
            "A Sra. Silva (nascida em 15/03/1980) comprou U$5.000,00 na bolsa de N.Y. às 14h30min usando seu e-mail ana.silva@exemplo.com.br! O site www.financas.com reportou que as ações da Apple Inc. subiram 2,5%. E aí, será que a Bovespa (IBOV) vai acompanhar essa alta-frequência de mercado?",
        ),
        (
            "Esportes",
            "Neymar Jr. marcou dois gols pelo Al-Hilal no estádio King Fahd em Riad, na Arábia Saudita. A Confederação Brasileira de Futebol (CBF) convocou Vinícius Jr., do Real Madrid, e Endrick, também do Real Madrid, para a Copa América. O técnico Dorival Júnior declarou que o Maracanã será palco do próximo amistoso contra a Argentina de Lionel Messi.",
        ),
        (
            "Direito",
            "O Supremo Tribunal Federal (STF), sob a presidência do Ministro Luís Roberto Barroso, julgou a constitucionalidade da Emenda Constitucional nº 45. O Procurador-Geral da República, Paulo Gonet, apresentou parecer ao Tribunal Superior Eleitoral (TSE) em Brasília. A Ordem dos Advogados do Brasil (OAB) emitiu nota conjunta com o Conselho Nacional de Justiça (CNJ) sobre a reforma do Código Penal.",
        ),
        (
            "Economia",
            "O Banco Central do Brasil, presidido por Gabriel Galípolo, manteve a taxa Selic em 13,75%. O Fundo Monetário Internacional (FMI) revisou a previsão de crescimento do PIB brasileiro. A Petrobras anunciou investimentos de R$ 100 bilhões em parceria com a Shell e a TotalEnergies para exploração de petróleo na Bacia de Santos, litoral de São Paulo.",
        ),
        (
            "Ciência",
            "Pesquisadores do Instituto Nacional de Pesquisas Espaciais (INPE), em São José dos Campos, detectaram aumento no desmatamento da Amazônia usando satélites do programa CBERS, desenvolvido em parceria com a Agência Espacial Chinesa. A bióloga Natália Pasternak, do Instituto Questão de Ciência, publicou estudo na revista Nature sobre a eficácia de vacinas produzidas pelo Instituto Butantan em colaboração com a Universidade de Oxford.",
        ),
    ]
}
