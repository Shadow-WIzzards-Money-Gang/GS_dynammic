# 🌎 Global Solution 2026 — Monitoramento de Riscos Ambientais
**FIAP — Estruturas de Dados e Algoritmos | Dynamic Programming**

---

## 👥 Integrantes do Grupo

**Turma: ESPH2**

| RA | Nome |
|----|------|
| 563524 | Felipe Bicaletto |
| 561777 | Antonio Neto |
| 556645 | Mauro Carlos Maia Neto |

---

## 📋 Descrição do Projeto

Sistema computacional de **monitoramento e triagem de riscos ambientais** em municípios
brasileiros, construído sobre estruturas de dados fundamentais (Grafo, BST, Heap) e
algoritmos clássicos (Força Bruta com backtracking + Dijkstra).

**Cenários implementados:**
- **Cenário A** — Rede de resposta a enchentes no Rio Grande do Sul (2024)
- **Cenário B** — Triagem de risco de seca no MATOPIBA

**Algoritmo Guloso escolhido:** Dijkstra
> Justificativa: o problema central é encontrar a **rota de menor custo** a partir de um hub
> de recursos (Porto Alegre) para cada município afetado. Dijkstra resolve exatamente esse
> problema em O((V+E)logV), integrando-se naturalmente à BST para priorizar municípios de
> alto risco.

---

## 🗂️ Estrutura do Repositório

```
global-solution-2026-fund/
├── README.md                       # Este arquivo
├── requirements.txt                # Dependências Python
├── main.py                         # Script principal (executa tudo)
├── data/
│   ├── raw/
│   │   ├── municipios_rs.json      # Grafo RS (Cenário A) — dados sintéticos
│   │   └── municipios_matopiba.json # Grafo MATOPIBA (Cenário B) — dados sintéticos
│   └── processed/
│       └── resultado_rs.json       # Resultados serializados (gerado pelo main.py)
├── src/
│   ├── data_structures.py          # Grafo, Node, BinarySearchTree
│   ├── brute_force.py              # Força Bruta com backtracking + MST bruta
│   ├── greedy.py                   # Dijkstra (principal) + Prim + Kruskal
│   ├── performance_monitor.py      # Tempo, memória, benchmarks
│   └── visualizations.py          # Todas as 5 figuras obrigatórias
├── notebooks/
│   └── analise_resultados.ipynb    # Análise interativa e escala de decisão
├── tests/
│   └── test_algorithms.py          # Testes unitários (pytest)
├── output_figs/                    # Figuras geradas (criada automaticamente)
└── report/
    └── relatorio_final.pdf         # Relatório técnico (máx. 4 páginas)
```

---

## ⚙️ Instalação e Execução

### 1. Clonar o repositório

```bash
git clone https://github.com/<usuario>/global-solution-2026-fund.git
cd global-solution-2026-fund
```

### 2. Criar ambiente virtual e instalar dependências

```bash
python -m venv venv
source venv/bin/activate        # Linux/Mac
# venv\Scripts\activate         # Windows

pip install -r requirements.txt
```

### 3. Executar o sistema completo

```bash
python main.py
```

Saída esperada:
- Resultados no terminal (rotas, BST, benchmarks)
- Figuras salvas em `output_figs/`
- `data/processed/resultado_rs.json`

### 4. Executar os testes unitários

```bash
pytest tests/ -v
```

### 5. Abrir o notebook de análise

```bash
jupyter notebook notebooks/analise_resultados.ipynb
```

---

## 📊 Módulos

| Módulo | Responsabilidade |
|--------|-----------------|
| `data_structures.py` | `Grafo` (dict de listas), `Node`, `BinarySearchTree` (BST do zero) |
| `brute_force.py` | Backtracking para todos os caminhos; MST bruta; dados de explosão combinatória |
| `greedy.py` | `dijkstra()`, `prim()`, `kruskal()` com heap e Union-Find |
| `performance_monitor.py` | `medir()`, benchmarks por N, gap de otimalidade |
| `visualizations.py` | 5 figuras obrigatórias com matplotlib/networkx |

---

## 📈 Figuras Geradas

| Figura | Arquivo | Conteúdo |
|--------|---------|---------|
| Fig 1 | `fig1_grafo_rs.png` | Grafo RS com rotas Dijkstra destacadas |
| Fig 2 | `fig2_bst_rs.png` | Diagrama da BST (10–15 nós) |
| Fig 3 | `fig3_desempenho.png` | Tempo × N: Força Bruta vs Dijkstra |
| Fig 4 | `fig4_gap_otimalidade.png` | Gap percentual Dijkstra vs FB |
| Fig 5 | `fig5_tabela_estruturas.png` | Tabela de estruturas de dados |

---

## 🔗 Fontes de Dados

- **DNIT** — Malha viária federal: [dnit.gov.br](https://www.dnit.gov.br)
- **Defesa Civil RS** — Municípios afetados 2024
- **NDVI MODIS/NASA** — Índices de vegetação: [earthdata.nasa.gov](https://earthdata.nasa.gov)
- **INMET** — Dados climáticos: [bdmep.inmet.gov.br](https://bdmep.inmet.gov.br)
- **IBGE** — Dados municipais: [ibge.gov.br](https://www.ibge.gov.br)

> Nota: os dados usados neste projeto são sintéticos, gerados com base nas fontes
> acima para fins acadêmicos. Os índices de risco foram derivados seguindo a metodologia
> do NDVI + precipitação do INMET.

---

## 📚 Referências Bibliográficas

- Cormen, T. et al. (2022). *Introduction to Algorithms*, 4th Ed. MIT Press. (Caps. 22–25)
- Sedgewick, R. & Wayne, K. (2011). *Algorithms*, 4th Ed. Addison-Wesley. (Parte 4)
- Skiena, S. (2020). *The Algorithm Design Manual*, 3rd Ed. Springer.
- Carta Internacional Space and Major Disasters: [disasterscharter.org](https://disasterscharter.org)

---

## 🎯 Conexão com ODS

| ODS | Conexão |
|-----|---------|
| **ODS 2** — Fome Zero | Monitoramento de seca no MATOPIBA protege produção agrícola |
| **ODS 9** — Infraestrutura | Otimização de rotas de atendimento e conectividade |
| **ODS 11** — Cidades Sustentáveis | Resposta eficiente a desastres urbanos (RS) |
| **ODS 13** — Ação Climática | Sistema integrado de monitoramento com dados de satélite |
