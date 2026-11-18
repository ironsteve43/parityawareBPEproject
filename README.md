# O Custo Oculto da Paridade: Auditoria Estatística (Miller, 2024) e Impacto Econômico Líquido do Parity-Aware BPE

Trabalho prático da disciplina de Large Language Models (2026.2).

## Autores
- João Vitor de Carvalho Silva (joaovitorcarvalho@dcc.ufmg.br)
- Lucas de Oliveira Ferreira (lucasdeoliveira@dcc.ufmg.br)
- Mariana Assis Ramos (mariana.assis@dcc.ufmg.br)

---

## 1. Resumo do Projeto

Este projeto consiste em uma auditoria técnica e econômica independente do artigo *Parity-Aware Byte-Pair Encoding: Improving Cross-lingual Fairness in Tokenization* (Foroutan et al., ACL 2026). Os autores do método sustentam que a substituição de fusões gulosas pelo critério dinâmico *fair-max* reduz em até 89% a desigualdade entre idiomas sem degradar a acurácia dos modelos em benchmarks downstream.

Confrontamos essas afirmações utilizando a metodologia de inferência em super-população proposta por Evan Miller (*Adding Error Bars to Evals*, 2024) e simulações econômicas com preços reais de mercado:

1. **Auditoria de Significância Downstream:** Aplicamos pareamento questão a questão e o estimador de Erro Padrão Agrupado por Cluster (*Clustered SE*, Eq. 8 de Miller) para capturar a correlação de perguntas traduzidas em múltiplos idiomas. A incerteza real inflacionou em 2.80x e nenhuma das 22 línguas avaliadas apresentou diferença estatisticamente significativa ($p > 0.05$).
2. **Poder Estatístico e Casos de Falha:** Demonstramos via Efeito Mínimo Detectável (MDE) que 100% dos benchmarks originais são sub-dimensionados para detectar quedas de 1% a 2% (MDE médio de $\pm 9.88\%$). Além disso, a execução do algoritmo BPE com pré-tokenização `Whitespace()` revelou severa sobre-segmentação em código (+62.5%), números (+94.7%) e JSON (+126.1%).
3. **Impacto Econômico Líquido:** Cruzamos o corpus paralelo FLORES-200 com a distribuição real de tráfego web (W3Techs, onde o inglês responde por ~50%) e a precificação comercial de APIs (GPT-4o, Claude 3.5 e Llama 3.1). Constatamos que o "Imposto sobre o Inglês" (+8.51% de tokens) supera a economia combinada de todas as outras 21 línguas, gerando um prejuízo líquido de +$41.16 por 1M de requisições no GPT-4o e exigindo tráfego em inglês abaixo de 35.5% para atingir o ponto de equilíbrio (*break-even*).

---

## 2. Estrutura do Repositório

```text
├── src/
│   ├── paired_eval.py          # Auditoria de significância e erro agrupado (Pessoa 1)
│   ├── power_and_failures.py   # Análise de MDE e execução de falhas do BPE (Pessoa 2)
│   └── cost_simulator.py       # Simulação financeira e curva de break-even (Pessoa 3)
├── results/
│   ├── paired_significance_by_language.csv # Resultados por idioma com e sem cluster
│   ├── clustered_global_summary.csv        # Resumo global com inflação de 2.80x
│   ├── sensitivity_correlation_cluster.csv # Sweep de sensibilidade de correlação
│   ├── mde_benchmark_audit.csv             # Auditoria de poder e déficit amostral
│   ├── failure_cases.csv                   # Exemplos qualitativos de segmentação
│   ├── failure_modes_summary.csv           # Resumo quantitativo por domínio técnico
│   ├── cost_simulation_by_language.csv     # Custos e tokens por língua
│   ├── breakeven_traffic_sweep.csv         # Variação do tráfego de 0% a 100%
│   └── financial_executive_summary.csv     # Saldo líquido em GPT-4o, Claude e Llama
├── figures/
│   ├── fig1_forest_plot.pdf    # Gráfico 1: Forest plot comparativo (PDF vetorial)
│   ├── fig2_power_mde.pdf      # Gráfico 2: Curvas teóricas de MDE e amostras reais
│   └── fig3_cost_simulation.pdf# Gráfico 3: Saldo financeiro e ponto de break-even
├── paper/
│   ├── main.tex                # Artigo completo em arquivo único (ACM sigconf)
│   ├── references.bib          # Arquivo BibTeX de referências
│   └── figures/                # Figuras em alta resolução referenciadas no LaTeX
├── requirements.txt            # Dependências Python com versões fixadas
└── environment.yml             # Arquivo de ambiente para Conda
```

---

## 3. Instruções de Setup

O projeto foi construído para ser executado integralmente em CPU padrão, sem necessidade de GPU ou pré-treinamento pesado de LLMs.

### Opção A: Usando Python venv e pip

```bash
# 1. Clonar o repositório
git clone https://github.com/ironsteve43/parityawareBPEproject.git
cd parityawareBPEproject

# 2. Criar e ativar o ambiente virtual (Python 3.10 recomendado)
python -m venv venv

# No Linux/macOS:
source venv/bin/activate
# No Windows (PowerShell):
.\venv\Scripts\Activate.ps1
# No Windows (CMD):
.\venv\Scripts\activate.bat

# 3. Instalar as dependências com versões fixadas
pip install -r requirements.txt
```

### Opção B: Usando Conda

```bash
conda env create -f environment.yml
conda activate parity_bpe_audit
```

---

## 4. Reprodução dos Resultados

Cada script do diretório `src/` pode ser executado individualmente. Os outputs e figuras são salvos automaticamente em `results/` e `figures/`.

### Executar a Auditoria de Significância (Pessoa 1)
Calcula as diferenças pareadas e o erro padrão agrupado por cluster com base na Tabela 2 do paper original, gerando o Gráfico 1:
```bash
python src/paired_eval.py
```

### Executar a Análise de Poder e Modos de Falha (Pessoa 2)
Calcula o MDE por idioma, o tamanho amostral necessário ($N_{\text{req}}$) e executa o tokenizador com pré-tokenização `Whitespace()` para quantificar a perda em código, matemática e JSON:
```bash
python src/power_and_failures.py
```

### Executar a Simulação Financeira (Pessoa 3)
Calcula o saldo em dólares por milhão de requisições e determina analiticamente o ponto de *break-even*:
```bash
python src/cost_simulator.py
```

### Execução Completa (Linha Única)
```bash
python src/paired_eval.py && python src/power_and_failures.py && python src/cost_simulator.py
```

---

## 5. Compilação do Artigo (LaTeX)

O artigo completo está consolidado no arquivo [`paper/main.tex`](paper/main.tex) no formato ACM sigconf (`\documentclass[sigconf,nonacm]{acmart}`).

Para compilar no Overleaf:
1. Crie um projeto em branco no Overleaf.
2. Faça upload do arquivo `main.tex`, do arquivo `references.bib` e da pasta `figures/`.
3. Certifique-se de que o compilador está definido como **pdfLaTeX**.
4. Clique em **Recompile**.
