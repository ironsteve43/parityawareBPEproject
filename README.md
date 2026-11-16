# O Custo Oculto da Paridade: Auditoria Estatística (Miller, 2024) e Impacto Econômico Líquido do Parity-Aware BPE

[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![Template: ACM sigconf](https://img.shields.io/badge/LaTeX-ACM%20sigconf-orange.svg)](https://www.overleaf.com/)
[![100% CPU / Colab Reproducible](https://img.shields.io/badge/Hardware-100%25%20CPU%20%2F%20Colab-green.svg)](#instruções-de-reprodução)
[![License: MIT](https://img.shields.io/badge/License-MIT-purple.svg)](LICENSE)

> **Artigos de Referência:**
> - Evan Miller. 2024. *Adding Error Bars to Evals: A Statistical Approach to Language Model Evaluations*. [arXiv:2411.00640](https://arxiv.org/pdf/2411.00640).
> - Negar Foroutan et al. 2026. *Parity-Aware Byte-Pair Encoding: Improving Cross-lingual Fairness in Tokenization*. [ACL 2026 / arXiv:2508.04796](https://aclanthology.org/2026.acl-long.342.pdf). Repositório oficial: [`swiss-ai/parity-aware-bpe`](https://github.com/swiss-ai/parity-aware-bpe).

---

## Equipe e Divisão de Responsabilidades

| Integrante | Papel | Seções no Artigo | Módulos Desenvolvidos | Artefatos Gerados |
| :--- | :--- | :--- | :--- | :--- |
| **Lucas de Oliveira Ferreira** | **Pessoa 1** (Inferência Estatística) | Seção 3.1 (Metodologia e Resultados de Significância Downstream) | `src/paired_eval.py` | **Gráfico 1** (`fig1_forest_plot.pdf/png`), Tabela pareada e clusterizada por idioma |
| **Mariana Assis Ramos** | **Pessoa 2** (Poder Estatístico & Falhas) | Seção 3.2 (Poder Estatístico & Casos de Falha), Seção 4 (Ameaças à Validade) | `src/power_and_failures.py` | **Gráfico 2** (`fig2_power_mde.pdf/png`), **Tabela 1** (Falhas de Tokenização BPE) |
| **João Vitor de Carvalho Silva** | **Pessoa 3** (Infra / Finanças) | Seção 1 (Intro), Seção 2 (Background), Seção 3.3 (Finanças), Seção 5 (Conclusão), Declarações | `src/cost_simulator.py`, `requirements.txt`, `README.md`, template Overleaf | **Gráfico 3** (`fig3_cost_simulation.pdf/png`), Tabelas de Break-Even e impacto em USD |

---

## Proposta e Resumo Executivo

O artigo *Parity-Aware BPE* (Foroutan et al., ACL 2026) sustenta que a substituição do critério guloso de merges pelo objetivo dinâmico ***fair-max*** equaliza as taxas de compressão entre idiomas (reduzindo a desigualdade medida pelo Gini em até $89\%$) sem degradar a acurácia dos modelos em benchmarks multilíngues, ao custo de uma perda marginal de compressão no inglês.

Este projeto coloca essa alegação à prova através de uma **auditoria experimental e econômica independente**, guiada pelo framework de inferência estatística em super-populações de Evan Miller (2024), estruturada em três frentes complementares que rodam **100% em CPU/Colab gratuito** (sem necessidade de pré-treinar LLMs):

```
                        AUDITORIA DO PARITY-AWARE BPE
                                      │
         ┌────────────────────────────┼────────────────────────────┐
         ▼                            ▼                            ▼
   PESSOA 1 (Lucas)             PESSOA 2 (Mari)              PESSOA 3 (João)
 Significância Downstream     Poder Estatístico (MDE)     Impacto Econômico Líquido
    (Miller Seções 2-4)         & Modos de Falha BPE        & Dinâmica de Break-Even
  Diferenças Pareadas (Eq. 7)    Curvas de MDE a 80% Poder     FLORES-200 x W3Techs Web
  Clusterização Questão (Eq. 8)  Cegueira a quedas de 1%-2%    GPT-4o, Claude 3.5, Llama
  Inflação de SE em 2.80x        Overhead de +62% a +330%     "Imposto do Inglês" supera
  0 de 22 línguas significativas  em Código, JSON e STEM       economia -> Break-Even 35.5%
```

---

## As Três Frentes da Auditoria

### 1. Auditoria de Significância Estatística (Pessoa 1 — Lucas)
- **O Problema:** A Tabela 2 de Foroutan et al. reporta médias com erros padrão isolados ($\hat{\mu} \pm \mathrm{SE}$), comparando barras não-pareadas ($\mathrm{SE}_{\mathrm{unpaired}} = \sqrt{\mathrm{SE}_A^2 + \mathrm{SE}_B^2}$, Eq. 5 de Miller). Essa abordagem introduz dois vícios metodológicos:
  1. *Omissão do Pareamento Questão-a-Questão:* como ambos os modelos respondem aos mesmos testes, há forte correlação positiva ($\rho \approx 0.60$), cuja inclusão reduz a variância do estimador ($\mathrm{SE}_{\mathrm{paired}}$, Eq. 7 de Miller).
  2. *Violação de Independência por Clusterização:* benchmarks traduzidos (*Belebele*, *XNLI*, *MMMLU*) avaliam a mesma pergunta fundamental em vários idiomas. Tratar traduções como independentes viola o TLC.
- **Implementação (`src/paired_eval.py`):** Reconstrução dos $n_{\mathrm{eff}}$ das 22 línguas (de $81$ no Coreano a $920$ no Tailandês) e aplicação do estimador de Erro Padrão Agrupado por Cluster ($\mathrm{SE}_{\mathrm{clustered}}$, Eq. 8 de Miller).
- **Descoberta:** A incerteza global aumenta em **$2.80\times$** sob clusterização ($\pm 0.57\% \rightarrow \pm 1.60\%$). Em todos os 22 idiomas, o zero está contido nos intervalos de confiança de 95% ($p > 0.05$). Os ganhos nominais (ex: Italiano $+2.19\%$, Alemão $+1.86\%$) e quedas (Persa $-3.65\%$) são ruído amostral indistinguível de zero.
- **Artefato:** Gráfico 1 (`figures/fig1_forest_plot.pdf` e `.png`).

### 2. Análise de Poder Estatístico & Modos de Falha (Pessoa 2 — Mari)
- **O Problema:** Provar que a diferença não tem significância ($p > 0.05$) não prova paridade se o teste for estatisticamente cego (baixo poder estatístico).
- **Implementação (`src/power_and_failures.py`):**
  1. *Fórmula de MDE (Miller Seção 5):* $\mathrm{MDE} = (z_{1-\alpha/2} + z_{1-\beta}) \cdot \mathrm{SE}_{\mathrm{clustered}}$ com $80\%$ de poder ($\alpha = 0.05$).
  2. *Execução do BPE Real:* Treinamento comparativo com pré-tokenização `Whitespace()` em domínios técnicos (código Python, constantes numéricas, JSON/API, compostos em alemão e terminologia STEM).
- **Descoberta:**
  - **$22$ de $22$ idiomas ($100\%$)** são sub-dimensionados para detectar quedas reais de $1.0\%$ ou $2.0\%$ ($\mathrm{MDE}$ médio de $\mathbf{\pm 9.88\%}$). Para detectar uma queda de $2\%$, os benchmarks precisariam ser **$26.8\times$ maiores** ($N_{\mathrm{req}} = 6.723$ questões agrupadas por língua).
  - O algoritmo *fair-max* quebra morfemas técnicos na cauda longa: fragmentação de $+62.5\%$ em sintaxe de código, $+94.7\%$ em constantes numéricas (destruindo alinhamento aritmético), $+126.1\%$ em payloads JSON e até $+330\%$ em terminologia científica STEM.
- **Artefatos:** Gráfico 2 (`figures/fig2_power_mde.pdf` e `.png`) e Tabela 1 de Casos de Falha.

### 3. Simulação Financeira Global & Ponto de Break-Even (Pessoa 3 — João)
- **O Problema:** As avaliações de equidade medem paridade de compressão (caracteres por token) em corpora equilibrados. Todavia, **APIs comerciais cobram por token** faturado. O que acontece com o bolso do usuário quando o inglês (que domina o tráfego) perde compressão em troca de ganhos em línguas de baixo tráfego?
- **Implementação (`src/cost_simulator.py`):**
  1. Tokenização no dataset paralelo FLORES-200 (22 idiomas correspondentes à Tabela 2 do paper).
  2. Ponderação pela distribuição empírica de conteúdo na web mundial (W3Techs 2024-2026: Inglês $49.80\%$, Espanhol $6.0\%$, Alemão $5.8\%$, ..., Tâmil $0.05\%$, Telugu $0.03\%$).
  3. Precificação real em USD para 1 Milhão de requisições típicas (prompt $+$ completion de 300 palavras):
     - **OpenAI GPT-4o:** $\$4.375$ / 1M tokens mistos
     - **Anthropic Claude 3.5 Sonnet:** $\$6.000$ / 1M tokens mistos
     - **Meta Llama 3.1 70B Hosted:** $\$0.800$ / 1M tokens mistos
- **Descoberta:**
  - O overhead de $+8.51\%$ tokens no inglês gera um **"Imposto sobre o Inglês" de $+\$81.03$** a cada 1M de requisições no GPT-4o.
  - Como as línguas beneficiadas possuem volume diminuto na web (Tâmil e Telugu somam menos de $0.1\%$), a **economia combinada de todas as 21 línguas não-inglesas é de apenas $-\$39.87$**.
  - O **Saldo Líquido Global é um PREJUÍZO de $+\$41.16$ por 1M de requisições** ($+\$56.43$ no Claude 3.5; $+\$6.35$ no Llama 3.1 70B).
  - Em escala corporativa (1 Bilhão de requisições/ano com GPT-4o), o PA-BPE impõe uma sobretaxa financeira líquida de **$+\$41,160.00$ USD/ano**.
  - **Ponto de Break-Even:** O método só quebra o equilíbrio financeiro se o tráfego em inglês cair abaixo de **$w_{\mathrm{en}}^* = 35.48\%$**. Na realidade da web ($50\%$) e de APIs de software ($70\%$--$80\%$), o PA-BPE atua invariavelmente como uma taxa econômica punitiva.
- **Artefato:** Gráfico 3 (`figures/fig3_cost_simulation.pdf` e `.png`).

---

## Síntese dos Resultados Experimentais

| Métrica Auditada | Classical BPE | Parity-Aware BPE (Hybrid) | Impacto / Veredito da Auditoria |
| :--- | :---: | :---: | :--- |
| **Diferença Global Downstream ($\Delta\mathrm{Acc}$)** | Baseline ($0.00\%$) | $+0.22\%$ | $\mathrm{SE}_{\mathrm{clustered}} = \pm 1.60\%$ ($z = 0.14, p = 0.89$) $\rightarrow$ **Puro ruído** |
| **Idiomas com Ganho Estatisticamente Significativo** | — | **0 de 22** ($0\%$) | $100\%$ dos CIs de 95\% cruzam o zero após correção de cluster |
| **Poder Estatístico para Detectar Queda de 1%** | — | **$6.2\%$** (médio) | MDE médio de $\pm 9.88\%$ $\rightarrow$ **Cego a regressões** |
| **Tamanho Amostral Necessário ($N_{\mathrm{req}}$ para 2%)** | $385$ questões | **$6.723$** questões | Déficit amostral de **$26.8\times$** frente aos benchmarks atuais |
| **Overhead de Tokens em Código Python** | 16 tokens | 26 tokens | **$+62.5\%$** em sintaxe e keywords |
| **Overhead de Tokens em Aritmética Numérica** | 19 tokens | 37 tokens | **$+94.7\%$** (estilhaça blocos de 3 dígitos) |
| **Overhead de Tokens em JSON Estruturado / APIs** | 23 tokens | 52 tokens | **$+126.1\%$** (infla latência e memória KV-cache) |
| **Overhead de Tokens em Inglês (FLORES-200)** | 27.5 tok/sent | 29.8 tok/sent | **$+8.51\%$** de sobretaxa estrutural |
| **Ganho de Compressão no Tâmil (FLORES-200)** | 66.8 tok/sent | 54.4 tok/sent | **$-18.50\%$** de economia em tokens |
| **Saldo Líquido em USD (GPT-4o por 1M Reqs)** | Baseline | **$+\$41.16$** (Prejuízo) | Imposto Inglês ($+\$81.03$) vs. Economia Outras ($-\$39.87$) |
| **Saldo Líquido em USD (Claude 3.5 por 1M Reqs)**| Baseline | **$+\$56.43$** (Prejuízo) | Imposto Inglês ($+\$111.12$) vs. Economia Outras ($-\$54.69$) |
| **Ponto de Break-Even de Tráfego em Inglês ($w_{\mathrm{en}}^*$)** | — | **$35.48\%$** | Prejuízo inevitável para tráfego web ($50\%$) ou enterprise ($75\%$) |

---

## Estrutura do Repositório

```
parityawareBPEproject/
├── .gitignore                      # Regras completas para Python, LaTeX e caches
├── requirements.txt                # Dependências científicas com versões fixadas
├── README.md                       # Documentação completa e instruções de reprodução
├── figures/                        # Gráficos prontos para publicação (PDF vetorial e PNG 300 DPI)
│   ├── fig1_forest_plot.pdf        # Gráfico 1 (Pessoa 1): Forest Plot naive vs. clustered SE
│   ├── fig1_forest_plot.png
│   ├── fig2_power_mde.pdf          # Gráfico 2 (Pessoa 2): Curva de Poder MDE vs. n_eff
│   ├── fig2_power_mde.png
│   ├── fig2b_sample_deficit_slide.png # Gráfico complementar para apresentação/slides
│   ├── fig3_cost_simulation.pdf    # Gráfico 3 (Pessoa 3): Imposto do Inglês & Curva de Break-Even
│   └── fig3_cost_simulation.png
├── results/                        # Tabelas CSV auditáveis (< 1 MB) geradas pelos scripts
│   ├── paired_significance_by_language.csv # Auditoria pareada/clusterizada dos 22 idiomas
│   ├── clustered_global_summary.csv        # Resumo global com inflação de 2.80x
│   ├── sensitivity_correlation_cluster.csv # Sweep de sensibilidade rho in [0.0, 0.8]
│   ├── mde_benchmark_audit.csv             # Auditoria de MDE e poder estatístico
│   ├── failure_cases.csv                   # Casos de falha reais palavra por palavra
│   ├── failure_modes_summary.csv           # Resumo quantitativo por domínio técnico
│   ├── cost_simulation_by_language.csv     # Impacto financeiro por idioma em USD
│   ├── breakeven_traffic_sweep.csv         # Sweep de tráfego em inglês de 0% a 100%
│   └── financial_executive_summary.csv     # Saldo líquido em GPT-4o, Claude e Llama
├── src/                            # Código-fonte Python modular e reproduzível em CPU
│   ├── paired_eval.py              # Pessoa 1: Inferência de Miller (Seções 2 a 4)
│   ├── power_and_failures.py       # Pessoa 2: MDE (Seção 5) e Tokenizador BPE Real
│   └── cost_simulator.py           # Pessoa 3: FLORES-200, W3Techs e Break-Even Comercial
└── paper/                          # Artigo científico em formato ACM sigconf (2 páginas)
    ├── main.tex                    # Documento mestre integrador
    ├── pessoa1_sections.tex        # Seção 3.1: Metodologia e Significância Estatística
    ├── pessoa2_power.tex           # Seção 3.2: Análise de Poder e Casos de Falha
    ├── pessoa2_limitations.tex     # Seção 4: Limitações e Ameaças à Validade
    ├── pessoa3_intro_bg.tex        # Seção 1 (Intro) e Seção 2 (Background)
    ├── pessoa3_financial.tex       # Seção 3.3: Impacto Financeiro Global e Break-Even
    ├── pessoa3_conclusion.tex      # Seção 5: Conclusão e Declarações Obrigatórias
    └── references.bib              # Bibliografia unificada em padrão BibTeX
```

---

## Instruções de Reprodução

Todo o pipeline roda **100% em CPU padrão** (Linux, Windows, macOS) ou no Google Colab gratuito em menos de 1 minuto.

### 1. Clonar o Repositório e Criar Ambiente Virtual

```bash
# Clonar o repositório
git clone https://github.com/ironsteve43/parityawareBPEproject.git
cd parityawareBPEproject

# Criar ambiente virtual Python (recomendado 3.10 ou 3.12)
python -m venv venv

# Ativar o ambiente virtual:
# No Linux/macOS:
source venv/bin/activate
# No Windows (PowerShell):
.\venv\Scripts\Activate.ps1
# No Windows (CMD):
.\venv\Scripts\activate.bat

# Instalar as dependências
pip install -r requirements.txt
```

### 2. Executar os Três Scripts da Auditoria

Você pode executar os scripts individualmente ou todos em sequência:

```bash
# Pessoa 1: Auditoria de Significância Downstream (Gera Fig 1 e CSVs)
python src/paired_eval.py

# Pessoa 2: Auditoria de Poder Estatístico MDE e BPE Real (Gera Fig 2 e Tabela 1)
python src/power_and_failures.py

# Pessoa 3: Simulação de Custo Financeiro Global e Break-Even (Gera Fig 3 e CSVs)
python src/cost_simulator.py
```

### 3. Execução em Linha Única (All-in-One)

```bash
python src/paired_eval.py && python src/power_and_failures.py && python src/cost_simulator.py
```

---

## Compilação no Overleaf (Template ACM sigconf)

O artigo foi formatado rigorosamente no padrão da ACM Conference Proceedings (`\documentclass[sigconf,nonacm]{acmart}`), ocupando 2 páginas de conteúdo técnico mais páginas extras para referências e declarações.

### Como Importar no Overleaf:
1. Compacte a pasta `paper/` e a pasta `figures/` em um arquivo `.zip` (ou vincule este repositório do GitHub diretamente à sua conta do Overleaf via *New Project $\rightarrow$ Import from GitHub*).
2. Certifique-se de que o compilador do Overleaf está configurado para **pdfLaTeX** (Menu superior esquerdo $\rightarrow$ *Compiler: pdfLaTeX*).
3. Defina `main.tex` como arquivo principal (*Main document: main.tex*).
4. Clique em **Recompile**. O PDF gerará perfeitamente com todas as figuras vetoriais e tabelas formatadas.

---

## Licença e Citação

Este projeto é disponibilizado sob a licença MIT. Para citar este trabalho:

```bibtex
@misc{silva2026custooculto,
  title        = {O Custo Oculto da Paridade: Auditoria Estat{\'\i}stica (Miller, 2024) e Impacto Econ{\^o}mico L{\'\i}quido do Parity-Aware BPE},
  author       = {Silva, Joao Vitor de Carvalho and Ferreira, Lucas de Oliveira and Ramos, Mariana Assis},
  year         = {2026},
  howpublished = {\url{https://github.com/ironsteve43/parityawareBPEproject}}
}
```
