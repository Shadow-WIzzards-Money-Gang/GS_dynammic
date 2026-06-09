"""
visualizations.py
=================
Módulo de visualizações do sistema de monitoramento de riscos ambientais.

Figuras geradas:
  1. Grafo de municípios com arestas da MST/Dijkstra destacadas
  2. Diagrama da BST (10–15 nós)
  3. Gráfico comparativo: tempo × N (Força Bruta vs Dijkstra)
  4. Gráfico de gap de otimalidade
  5. Tabela de estruturas de dados
"""

from __future__ import annotations
import math
import os
from typing import Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")   # backend sem display (container headless)
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import networkx as nx
import numpy as np

from src.data_structures import (
    Grafo, BinarySearchTree, Node,
    vertice_id, vertice_nome, vertice_risco, vertice_custo
)
from src.greedy import ResultadoGuloso
from src.performance_monitor import Medicao

OUTPUT_DIR = "output_figs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

CORES = {
    "primario":    "#1a73e8",
    "secundario":  "#e8711a",
    "alto_risco":  "#d32f2f",
    "baixo_risco": "#2e7d32",
    "mst":         "#ff6f00",
    "fundo":       "#f8f9fa",
    "texto":       "#212121",
}


# ---------------------------------------------------------------------------
# Figura 1 — Grafo de municípios
# ---------------------------------------------------------------------------

def plotar_grafo(grafo: Grafo,
                 resultado_guloso: Optional[ResultadoGuloso] = None,
                 titulo: str = "Grafo de Municípios",
                 caminho_salvar: str = "") -> str:
    """
    Renderiza o grafo com:
    - Nós coloridos por índice de risco (vermelho = alto, verde = baixo)
    - Arestas da solução gulosa destacadas em laranja
    - Labels com nome e risco do município

    Retorna o caminho do arquivo salvo.
    """
    fig, ax = plt.subplots(figsize=(14, 9))
    fig.patch.set_facecolor(CORES["fundo"])
    ax.set_facecolor(CORES["fundo"])

    G = nx.Graph()
    for v in grafo.vertices():
        G.add_node(vertice_id(v), nome=vertice_nome(v), risco=vertice_risco(v))
    for u in grafo.ids_vertices():
        for v, peso in grafo.vizinhos(u):
            if u < v:
                G.add_edge(u, v, weight=peso)

    pos = nx.spring_layout(G, seed=42, k=2.5)

    # Cores dos nós por risco
    riscos = [G.nodes[n]["risco"] for n in G.nodes()]
    norm = plt.Normalize(vmin=min(riscos), vmax=max(riscos))
    cmap = plt.cm.RdYlGn_r
    node_colors = [cmap(norm(r)) for r in riscos]

    # Arestas normais e destacadas
    if resultado_guloso is not None and resultado_guloso.arestas_mst:
        arestas_dest = {(min(u, v), max(u, v)) for u, v, _ in resultado_guloso.arestas_mst}
    elif resultado_guloso is not None and resultado_guloso.caminho:
        cam = resultado_guloso.caminho
        arestas_dest = {(min(cam[i], cam[i+1]), max(cam[i], cam[i+1]))
                        for i in range(len(cam) - 1)}
    else:
        arestas_dest = set()

    arestas_normal = [(u, v) for u, v in G.edges()
                      if (min(u, v), max(u, v)) not in arestas_dest]
    arestas_highlight = [(u, v) for u, v in G.edges()
                         if (min(u, v), max(u, v)) in arestas_dest]

    nx.draw_networkx_edges(G, pos, edgelist=arestas_normal,
                           edge_color="#cccccc", width=1.2, ax=ax)
    nx.draw_networkx_edges(G, pos, edgelist=arestas_highlight,
                           edge_color=CORES["mst"], width=3.0,
                           style="solid", ax=ax)

    nx.draw_networkx_nodes(G, pos, node_color=node_colors,
                           node_size=700, ax=ax)

    labels = {n: f"{G.nodes[n]['nome'][:10]}\nr={G.nodes[n]['risco']:.2f}"
              for n in G.nodes()}
    nx.draw_networkx_labels(G, pos, labels, font_size=6.5,
                            font_color=CORES["texto"], ax=ax)

    pesos = nx.get_edge_attributes(G, "weight")
    pesos_fmt = {k: f"{v:.1f}h" for k, v in pesos.items()}
    nx.draw_networkx_edge_labels(G, pos, pesos_fmt, font_size=5.5,
                                 font_color="#555555", ax=ax)

    # Colorbar
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, shrink=0.7, pad=0.02)
    cbar.set_label("Índice de Risco", fontsize=10)

    # Rótulo dinâmico da legenda conforme o tipo de resultado passado
    if resultado_guloso is not None and resultado_guloso.arestas_mst:
        label_dest = f"MST ({resultado_guloso.algoritmo})"
    elif resultado_guloso is not None and resultado_guloso.caminho:
        label_dest = f"Caminho mínimo ({resultado_guloso.algoritmo})"
    else:
        label_dest = "Solução Gulosa"
    patch_dest = mpatches.Patch(color=CORES["mst"], label=label_dest)
    patch_norm = mpatches.Patch(color="#cccccc", label="Demais arestas")
    ax.legend(handles=[patch_dest, patch_norm], loc="lower left", fontsize=9)

    ax.set_title(titulo, fontsize=14, fontweight="bold", color=CORES["texto"], pad=15)
    ax.axis("off")

    fonte = "Fonte: Dados sintéticos baseados em DNIT + Defesa Civil RS"
    fig.text(0.01, 0.01, fonte, fontsize=7, color="#888888")
    fig.text(0.01, 0.98,
             "Interpretação: Nós em vermelho indicam municípios de alto risco ambiental. "
             "Arestas laranja representam as rotas otimizadas pelo Dijkstra a partir de Porto Alegre.",
             fontsize=7.5, color="#444444", wrap=True)

    caminho = caminho_salvar or os.path.join(OUTPUT_DIR, "fig1_grafo_municipios.png")
    fig.savefig(caminho, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return caminho


# ---------------------------------------------------------------------------
# Figura 2 — Diagrama da BST
# ---------------------------------------------------------------------------

def _posicoes_bst(no: Optional[Node],
                  x: float = 0.0, y: float = 0.0,
                  dx: float = 4.0,
                  pos: Optional[Dict] = None,
                  edges: Optional[List] = None) -> Tuple[Dict, List]:
    if pos is None:
        pos = {}
    if edges is None:
        edges = []
    if no is None:
        return pos, edges

    nid = vertice_id(no.vertice)
    pos[nid] = (x, y)

    if no.esquerda:
        edges.append((nid, vertice_id(no.esquerda.vertice)))
        _posicoes_bst(no.esquerda, x - dx, y - 1.5, dx / 2, pos, edges)
    if no.direita:
        edges.append((nid, vertice_id(no.direita.vertice)))
        _posicoes_bst(no.direita, x + dx, y - 1.5, dx / 2, pos, edges)
    return pos, edges


def _inserir_balanceado(bst_obj: BinarySearchTree, vertices: list) -> None:
    """Insere vértices em ordem middle-out para BST visualmente balanceada."""
    if not vertices:
        return
    meio = len(vertices) // 2
    bst_obj.inserir(vertices[meio])
    _inserir_balanceado(bst_obj, vertices[:meio])
    _inserir_balanceado(bst_obj, vertices[meio + 1:])


def plotar_bst(bst: BinarySearchTree,
               titulo: str = "BST — Municípios por Índice de Risco",
               caminho_salvar: str = "") -> str:
    """
    Renderiza o diagrama da BST com nós coloridos por risco.
    Limita a 15 nós para legibilidade.
    Usa inserção middle-out para evitar árvore degenerada.
    """
    # Pega os 15 primeiros em in-order (ordenados por risco)
    vertices = bst.percurso_in_order()[:15]

    # Inserção middle-out: evita BST degenerada (altura 15)
    # que ocorreria inserindo em ordem crescente
    bst_reduzida = BinarySearchTree()
    _inserir_balanceado(bst_reduzida, vertices)

    pos, edges = _posicoes_bst(bst_reduzida._raiz)

    fig, ax = plt.subplots(figsize=(14, 7))
    fig.patch.set_facecolor(CORES["fundo"])
    ax.set_facecolor(CORES["fundo"])

    # Arestas
    for (u, v) in edges:
        x1, y1 = pos[u]
        x2, y2 = pos[v]
        ax.plot([x1, x2], [y1, y2], color="#9e9e9e", lw=1.5, zorder=1)

    # Nós
    for vid, (x, y) in pos.items():
        risco = _buscar_risco_por_id(bst_reduzida._raiz, vid)
        nome_curto = _buscar_nome_por_id(bst_reduzida._raiz, vid)

        cor = plt.cm.RdYlGn_r(risco) if risco is not None else "#999999"
        circle = plt.Circle((x, y), 0.55, color=cor, zorder=2, ec="#333333", lw=1.2)
        ax.add_patch(circle)
        txt = f"r={risco:.2f}\n{nome_curto[:8]}" if risco is not None else str(vid)
        ax.text(x, y, txt, ha="center", va="center",
                fontsize=6.5, fontweight="bold", color="white", zorder=3)

    ax.set_xlim(min(x for x, _ in pos.values()) - 1.5,
                max(x for x, _ in pos.values()) + 1.5)
    ax.set_ylim(min(y for _, y in pos.values()) - 1.5,
                max(y for _, y in pos.values()) + 1.5)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_title(titulo, fontsize=13, fontweight="bold", color=CORES["texto"])

    fig.text(0.01, 0.02,
             "Fonte: BST construída sobre dados sintéticos de municípios RS.\n"
             "Interpretação: Nós à esquerda têm menor índice de risco; "
             "nós à direita, maior. A travessia in-order retorna municípios em "
             "ordem crescente de criticidade para priorização de atendimento.",
             fontsize=7.5, color="#444444")

    caminho = caminho_salvar or os.path.join(OUTPUT_DIR, "fig2_bst.png")
    fig.savefig(caminho, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return caminho


def _buscar_risco_por_id(no: Optional[Node], vid: int) -> Optional[float]:
    if no is None:
        return None
    if vertice_id(no.vertice) == vid:
        return vertice_risco(no.vertice)
    r = _buscar_risco_por_id(no.esquerda, vid)
    return r if r is not None else _buscar_risco_por_id(no.direita, vid)


def _buscar_nome_por_id(no: Optional[Node], vid: int) -> str:
    if no is None:
        return ""
    if vertice_id(no.vertice) == vid:
        return vertice_nome(no.vertice)
    r = _buscar_nome_por_id(no.esquerda, vid)
    return r if r else _buscar_nome_por_id(no.direita, vid)


# ---------------------------------------------------------------------------
# Figura 3 — Desempenho comparativo: tempo × N
# ---------------------------------------------------------------------------

def plotar_desempenho(medicoes_fb: List[Medicao],
                      medicoes_greedy: List[Medicao],
                      caminho_salvar: str = "") -> str:
    """
    Gráfico comparativo de tempo de execução × N para Força Bruta e Dijkstra.
    Inclui marcação do ponto de cruzamento (se houver).
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.patch.set_facecolor(CORES["fundo"])

    ns_fb = [m.n_vertices for m in medicoes_fb]
    ts_fb = [m.tempo_ms for m in medicoes_fb]
    ns_gd = [m.n_vertices for m in medicoes_greedy]
    ts_gd = [m.tempo_ms for m in medicoes_greedy]

    # --- Subgráfico 1: Tempo ---
    ax1 = axes[0]
    ax1.set_facecolor(CORES["fundo"])
    ax1.plot(ns_fb, ts_fb, "o-", color=CORES["alto_risco"],
             linewidth=2.2, markersize=7, label="Força Bruta O(V!)")
    ax1.plot(ns_gd, ts_gd, "s-", color=CORES["primario"],
             linewidth=2.2, markersize=7, label="Dijkstra O((V+E)logV)")

    ax1.set_xlabel("N (número de vértices)", fontsize=11)
    ax1.set_ylabel("Tempo de execução (ms)", fontsize=11)
    ax1.set_title("Tempo de Execução × N", fontsize=12, fontweight="bold")
    ax1.legend(fontsize=9)
    ax1.grid(True, alpha=0.3)

    # Anotação de cruzamento aproximado
    ax1.axvline(x=12, color="#888888", linestyle="--", alpha=0.6, linewidth=1)
    ax1.text(12.2, max(ts_fb) * 0.5, "Limite N=12\n(FB inviável)", fontsize=8,
             color="#666666")

    # --- Subgráfico 2: Memória ---
    ax2 = axes[1]
    ax2.set_facecolor(CORES["fundo"])
    mems_fb = [m.memoria_mb for m in medicoes_fb]
    mems_gd = [m.memoria_mb for m in medicoes_greedy]
    ax2.plot(ns_fb, mems_fb, "o-", color=CORES["alto_risco"],
             linewidth=2.2, markersize=7, label="Força Bruta")
    ax2.plot(ns_gd, mems_gd, "s-", color=CORES["primario"],
             linewidth=2.2, markersize=7, label="Dijkstra")

    ax2.set_xlabel("N (número de vértices)", fontsize=11)
    ax2.set_ylabel("Memória alocada (MB)", fontsize=11)
    ax2.set_title("Memória Alocada × N", fontsize=12, fontweight="bold")
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.3)

    fig.suptitle("Análise Comparativa de Desempenho — Força Bruta vs Dijkstra",
                 fontsize=13, fontweight="bold", y=1.01)

    fig.text(0.01, -0.04,
             "Fonte: Benchmarks executados com grafos sintéticos gerados aleatoriamente.\n"
             "Interpretação: A Força Bruta cresce fatorialmente (O(V!)), tornando-se inviável "
             "a partir de N≈10–12 vértices. O Dijkstra mantém crescimento polinomial mesmo "
             "para N=100, confirmando sua adequação para instâncias reais de municípios.",
             fontsize=8, color="#444444")

    caminho = caminho_salvar or os.path.join(OUTPUT_DIR, "fig3_desempenho.png")
    fig.savefig(caminho, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return caminho


# ---------------------------------------------------------------------------
# Figura 4 — Gap de otimalidade
# ---------------------------------------------------------------------------

def plotar_gap_otimalidade(gaps: Dict[int, float],
                           caminho_salvar: str = "") -> str:
    """
    Gráfico de gap percentual entre Força Bruta (ótimo global) e Dijkstra.
    Gap esperado ≈ 0% pois Dijkstra é ótimo para caminhos mínimos.
    """
    ns = sorted(gaps.keys())
    gs = [gaps[n] for n in ns]

    fig, ax = plt.subplots(figsize=(9, 5))
    fig.patch.set_facecolor(CORES["fundo"])
    ax.set_facecolor(CORES["fundo"])

    bars = ax.bar(ns, gs, color=CORES["primario"], alpha=0.8,
                  edgecolor="#333333", linewidth=0.8)

    for bar, g in zip(bars, gs):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.001,
                f"{g:.3f}%", ha="center", va="bottom", fontsize=9)

    ax.axhline(y=0, color=CORES["baixo_risco"], linestyle="--",
               linewidth=1.5, alpha=0.7, label="Gap ideal = 0%")

    ax.set_xlabel("N (número de vértices)", fontsize=11)
    ax.set_ylabel("Gap de Otimalidade (%)", fontsize=11)
    ax.set_title("Gap de Otimalidade: Dijkstra vs Força Bruta",
                 fontsize=12, fontweight="bold")
    ax.set_xticks(ns)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3, axis="y")

    fig.text(0.01, 0.01,
             "Fonte: Comparação entre Força Bruta (oráculo) e Dijkstra em grafos sintéticos.\n"
             "Interpretação: Gap ≈ 0% confirma que o Dijkstra encontra a solução ótima "
             "para caminhos mínimos, validando o algoritmo guloso para uso em instâncias reais.",
             fontsize=8, color="#444444")

    caminho = caminho_salvar or os.path.join(OUTPUT_DIR, "fig4_gap_otimalidade.png")
    fig.savefig(caminho, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return caminho


# ---------------------------------------------------------------------------
# Figura 6 — Explosão Combinatória da Força Bruta (seção 3.1)
# ---------------------------------------------------------------------------

def plotar_explosao_combinatoria(caminho_salvar: str = "") -> str:
    """
    Gráfico do crescimento do número de caminhos em função de N.
    Evidencia a explosão combinatória que torna a Força Bruta inviável para N > 12.

    Exibido em escala logarítmica no eixo Y para melhor visualização do crescimento.
    """
    from src.brute_force import dados_explosao_combinatoria

    n_range = list(range(3, 16))
    explosao = dados_explosao_combinatoria(n_range)
    ns = list(explosao.keys())
    cs = list(explosao.values())

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.patch.set_facecolor(CORES["fundo"])

    # --- Escala linear ---
    ax1 = axes[0]
    ax1.set_facecolor(CORES["fundo"])
    ax1.plot(ns, cs, "o-", color=CORES["alto_risco"], linewidth=2.5, markersize=8)
    ax1.fill_between(ns, cs, alpha=0.15, color=CORES["alto_risco"])
    ax1.axvline(x=12, color="#888888", linestyle="--", alpha=0.7, linewidth=1.5)
    ax1.text(12.15, max(cs) * 0.6,
             "N=12\n(limite\nprático)", fontsize=8.5, color="#555555")
    ax1.set_xlabel("N (número de vértices)", fontsize=11)
    ax1.set_ylabel("Número estimado de caminhos", fontsize=11)
    ax1.set_title("Escala Linear — Crescimento Fatorial", fontsize=12, fontweight="bold")
    ax1.grid(True, alpha=0.3)

    # Anota valores nos pontos chave
    for n, c in zip(ns, cs):
        if n in (3, 6, 9, 12, 15):
            ax1.annotate(f"{c:,}", (n, c), textcoords="offset points",
                         xytext=(0, 8), ha="center", fontsize=7.5,
                         color=CORES["alto_risco"])

    # --- Escala logarítmica ---
    ax2 = axes[1]
    ax2.set_facecolor(CORES["fundo"])
    ax2.semilogy(ns, cs, "o-", color=CORES["alto_risco"], linewidth=2.5, markersize=8,
                 label="Força Bruta O(V!)")

    # Linha de referência Dijkstra (crescimento ~polinomial simulado)
    import math as _math
    cs_dijk = [n * _math.log2(n + 1) * 5 for n in ns]
    ax2.semilogy(ns, cs_dijk, "s--", color=CORES["primario"], linewidth=2,
                 markersize=6, label="Dijkstra O((V+E)logV)")

    ax2.axvline(x=12, color="#888888", linestyle="--", alpha=0.7, linewidth=1.5)
    ax2.text(12.15, max(cs) * 0.05, "N=12", fontsize=8.5, color="#555555")
    ax2.set_xlabel("N (número de vértices)", fontsize=11)
    ax2.set_ylabel("Número de operações (escala log)", fontsize=11)
    ax2.set_title("Escala Log — Comparação de Crescimento", fontsize=12, fontweight="bold")
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.3, which="both")

    fig.suptitle("Explosão Combinatória da Força Bruta vs Dijkstra",
                 fontsize=13, fontweight="bold", y=1.01)

    fig.text(0.01, -0.06,
             "Fonte: Estimativa analítica do número de caminhos simples em grafo completo de N vértices.\n"
             "Interpretação: O número de caminhos cresce fatorialmente — para N=12 já ultrapassa 1 milhão.\n"
             "O Dijkstra mantém crescimento O((V+E)logV), tornando-se a única opção viável para N > 12.",
             fontsize=8, color="#444444")

    caminho = caminho_salvar or os.path.join(OUTPUT_DIR, "fig6_explosao_combinatoria.png")
    fig.savefig(caminho, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return caminho


# ---------------------------------------------------------------------------
# Figura 5 — Tabela de estruturas de dados
# ---------------------------------------------------------------------------

def plotar_tabela_estruturas(caminho_salvar: str = "") -> str:
    """Gera figura com a tabela de estruturas de dados do projeto."""
    dados = [
        ["Lista (list)",     "Adjacência do grafo",    "Lista de vizinhos por município",   "O(V+E) espaço"],
        ["Tupla (tuple)",    "Vértice imutável",        "5-tupla: (id, nome, risco, custo, pop)",  "O(1) acesso"],
        ["Dicionário (dict)","Adjacência ponderada",    "Mapeamento id → [(vizinho, peso)]", "O(1) médio lookup"],
        ["Conjunto (set)",   "Controle de visitados",   "BFS/DFS e fronteira Dijkstra",      "O(1) médio pertence"],
        ["Heap (heapq)",     "Fila de prioridade",      "Extrai mín no Dijkstra/Prim",       "O(log V) push/pop"],
        ["BST (classe)",     "Ranking por risco",       "Busca municípios críticos",         "O(h) busca"],
        ["Grafo (dict+list)","Rede de municípios",      "Modelagem de rotas e conexões",     "O(V+E) espaço"],
        ["deque",            "Fila BFS",                "Travessia em largura do grafo",     "O(1) append/popleft"],
    ]
    colunas = ["Estrutura", "Uso no Sistema", "Aplicação Concreta", "Complexidade"]

    fig, ax = plt.subplots(figsize=(14, 5))
    fig.patch.set_facecolor(CORES["fundo"])
    ax.axis("off")

    tabela = ax.table(
        cellText=dados,
        colLabels=colunas,
        cellLoc="left",
        loc="center",
        colWidths=[0.18, 0.22, 0.38, 0.22],
    )
    tabela.auto_set_font_size(False)
    tabela.set_fontsize(8.5)
    tabela.scale(1, 1.8)

    for (row, col), cell in tabela.get_celld().items():
        if row == 0:
            cell.set_facecolor(CORES["primario"])
            cell.set_text_props(color="white", fontweight="bold")
        elif row % 2 == 0:
            cell.set_facecolor("#e8f0fe")
        else:
            cell.set_facecolor("#ffffff")
        cell.set_edgecolor("#cccccc")

    ax.set_title("Tabela de Estruturas de Dados — Justificativa e Complexidade",
                 fontsize=12, fontweight="bold", pad=12, color=CORES["texto"])

    caminho = caminho_salvar or os.path.join(OUTPUT_DIR, "fig5_tabela_estruturas.png")
    fig.savefig(caminho, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return caminho
