"""
main.py
=======
Script principal do sistema de monitoramento de riscos ambientais.
Executa os dois cenários brasileiros, gera todas as figuras e exibe os resultados.

Uso: python main.py
"""

import os
import sys
import json
import math

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.data_structures import Grafo, BinarySearchTree
from src.brute_force import forca_bruta_caminhos, forca_bruta_mst
from src.greedy import dijkstra, prim, kruskal, reconstruir_caminho
from src.performance_monitor import PerformanceMonitor, gerar_grafo_sintetico
from src.visualizations import (
    plotar_grafo, plotar_bst, plotar_desempenho,
    plotar_gap_otimalidade, plotar_tabela_estruturas,
    plotar_explosao_combinatoria
)

os.makedirs("output_figs", exist_ok=True)
os.makedirs("data/processed", exist_ok=True)

LINHA = "=" * 65


def secao(titulo: str) -> None:
    print(f"\n{LINHA}")
    print(f"  {titulo}")
    print(LINHA)


# ---------------------------------------------------------------------------
# Cenário A — Enchentes RS
# ---------------------------------------------------------------------------

def executar_cenario_a() -> None:
    secao("CENÁRIO A — Rede de Resposta a Enchentes (RS)")

    g_rs = Grafo.from_json("data/raw/municipios_rs.json")
    print(f"Grafo RS carregado: {g_rs}")

    bst_rs = BinarySearchTree.from_grafo(g_rs)
    print(f"BST RS: {bst_rs}")
    print(f"  Árvore balanceada: {bst_rs.esta_balanceada()}")

    # Municípios de alto risco (>= 0.8)
    alto_risco = bst_rs.buscar(0.80, 1.0)
    print(f"\nMunicípios com risco >= 0.80:")
    for v in sorted(alto_risco, key=lambda x: -x[2]):
        print(f"  {v[1]:<25} risco={v[2]:.2f}  custo=R${v[3]:.0f}  pop={v[4]:,}")

    # Dijkstra a partir de Porto Alegre
    PORTO_ALEGRE = 4314902
    print(f"\nDijkstra a partir de Porto Alegre...")
    res_dijk = dijkstra(g_rs, PORTO_ALEGRE, bst=bst_rs, limiar_risco=0.80)
    print(f"  {res_dijk}")
    print(f"  Municípios de alto risco priorizados: {len(res_dijk.vertices_priorizados)}")

    # Exibe rotas para os 5 municípios mais críticos
    print("\n  Rotas mínimas (Porto Alegre → município crítico):")
    criticos = sorted(alto_risco, key=lambda x: -x[2])[:5]
    for v in criticos:
        vid = v[0]
        cam = reconstruir_caminho(res_dijk, vid)
        custo = res_dijk.custos_acumulados.get(vid, math.inf)
        nomes = [g_rs.get_vertice(c)[1] for c in cam if g_rs.get_vertice(c)]
        print(f"    {v[1]:<20} custo={custo:.2f}h  caminho: {' → '.join(nomes)}")

    # MST com Prim
    res_prim = prim(g_rs, PORTO_ALEGRE)
    print(f"\n  MST (Prim): custo_total={res_prim.custo_total:.2f}h, "
          f"arestas={len(res_prim.arestas_mst)}")

    # Força Bruta em subgrafo pequeno (N=8 primeiros vértices)
    ids_pequenos = g_rs.ids_vertices()[:8]
    ids_set = set(ids_pequenos)
    g_pequeno = Grafo()
    for vid in ids_pequenos:
        g_pequeno.adicionar_vertice(g_rs.get_vertice(vid))
    arestas_adicionadas = set()
    for vid in ids_pequenos:
        for viz, peso in g_rs.vizinhos(vid):
            if viz in ids_set:
                chave = (min(vid, viz), max(vid, viz))
                if chave not in arestas_adicionadas:
                    arestas_adicionadas.add(chave)
                    g_pequeno.adicionar_aresta(vid, viz, peso)

    print(f"\n  Força Bruta em subgrafo N={g_pequeno.num_vertices()}...")
    origem = ids_pequenos[0]
    destino = ids_pequenos[-1]
    res_fb = forca_bruta_caminhos(g_pequeno, origem, destino)
    print(f"  {res_fb}")
    print(f"  Melhor caminho FB: {res_fb.melhor_caminho} | custo={res_fb.melhor_custo:.2f}")

    # Validação: Dijkstra vs FB
    res_dijk_p = dijkstra(g_pequeno, origem)
    custo_dijk_p = res_dijk_p.custos_acumulados.get(destino, math.inf)
    gap = abs(custo_dijk_p - res_fb.melhor_custo) / max(res_fb.melhor_custo, 1e-9) * 100
    print(f"  Gap Dijkstra vs FB: {gap:.4f}% (esperado ≈ 0%)")

    # Salva resultado processado
    resultado = {
        "cenario": "A",
        "vertices": g_rs.num_vertices(),
        "arestas": g_rs.num_arestas(),
        "custo_mst_prim": res_prim.custo_total,
        "municipios_alto_risco": [v[1] for v in alto_risco]
    }
    with open("data/processed/resultado_rs.json", "w") as f:
        json.dump(resultado, f, indent=2, ensure_ascii=False)

    # Visualizações
    print("\n  Gerando figuras do Cenário A...")
    # Fig 1: usa MST (Prim) para destacar arestas — documento exige "arestas da MST destacadas"
    f1 = plotar_grafo(g_rs, res_prim,
                      titulo="Cenário A — Grafo RS com Árvore Geradora Mínima (Prim)",
                      caminho_salvar="output_figs/fig1_grafo_rs.png")
    print(f"  Fig 1 salva: {f1}")

    f2 = plotar_bst(bst_rs,
                    titulo="BST — Municípios RS por Índice de Risco (in-order = ordem de prioridade)",
                    caminho_salvar="output_figs/fig2_bst_rs.png")
    print(f"  Fig 2 salva: {f2}")

    # res_dijk e bst_rs disponíveis localmente para uso futuro se necessário


# ---------------------------------------------------------------------------
# Cenário B — MATOPIBA
# ---------------------------------------------------------------------------

def executar_cenario_b() -> None:
    secao("CENÁRIO B — Triagem de Seca no MATOPIBA")

    g_mt = Grafo.from_json("data/raw/municipios_matopiba.json")
    print(f"Grafo MATOPIBA carregado: {g_mt}")

    bst_mt = BinarySearchTree.from_grafo(g_mt)
    print(f"BST MATOPIBA: {bst_mt}")

    # In-order da BST → ordem de prioridade de atendimento
    print("\nOrdem de prioridade de atendimento (in-order BST):")
    for v in bst_mt.percurso_in_order():
        print(f"  {v[1]:<30} risco={v[2]:.2f}  custo_atend=R${v[3]:.0f}")

    # Kruskal para MST de cobertura mínima
    res_kruskal = kruskal(g_mt)
    print(f"\n  MST (Kruskal): custo_total={res_kruskal.custo_total:.2f}h, "
          f"arestas={len(res_kruskal.arestas_mst)}")

    # Dijkstra a partir de Balsas (hub regional)
    BALSAS = 2101400
    res_dijk_mt = dijkstra(g_mt, BALSAS, bst=bst_mt, limiar_risco=0.85)
    print(f"\n  Dijkstra (hub: Balsas): {res_dijk_mt}")

    # Visualização MATOPIBA
    plotar_grafo(g_mt, res_dijk_mt,
                 titulo="Cenário B — Grafo MATOPIBA com Rotas Dijkstra (Balsas como hub)",
                 caminho_salvar="output_figs/fig1b_grafo_matopiba.png")
    print("  Fig MATOPIBA salva: output_figs/fig1b_grafo_matopiba.png")


# ---------------------------------------------------------------------------
# Benchmarks e gráficos de desempenho
# ---------------------------------------------------------------------------

def executar_benchmarks() -> None:
    secao("BENCHMARKS — Análise de Desempenho e Escalabilidade")

    monitor = PerformanceMonitor()

    # Tamanhos para benchmark — conforme especificado no documento (seção 4)
    # N = 5, 8, 10, 12, 20, 50, 100
    n_fb = [5, 8, 10, 12]          # Força Bruta: apenas N <= 12
    n_greedy = [5, 8, 10, 12, 20, 50, 100]

    print("Executando benchmark Força Bruta (N = 5..12)...")
    meds_fb = monitor.benchmark_forca_bruta(n_fb, repeticoes=3)

    print("Executando benchmark Dijkstra (N = 5..100)...")
    meds_dijk = monitor.benchmark_dijkstra(n_greedy, repeticoes=3)

    print("\n" + monitor.tabela_resumo())

    # Gap de otimalidade
    print("\nCalculando gap de otimalidade...")
    gaps = monitor.calcular_gap_otimalidade([5, 8, 10, 12])
    print("  Gap Dijkstra vs Força Bruta:")
    for n, g in sorted(gaps.items()):
        print(f"    N={n:2d}: {g:.4f}%")

    # Figuras
    print("\n  Gerando figuras de desempenho...")
    f3 = plotar_desempenho(meds_fb, meds_dijk,
                           caminho_salvar="output_figs/fig3_desempenho.png")
    print(f"  Fig 3 salva: {f3}")

    f4 = plotar_gap_otimalidade(gaps,
                                caminho_salvar="output_figs/fig4_gap_otimalidade.png")
    print(f"  Fig 4 salva: {f4}")

    f5 = plotar_tabela_estruturas(caminho_salvar="output_figs/fig5_tabela_estruturas.png")
    print(f"  Fig 5 salva: {f5}")

    f6 = plotar_explosao_combinatoria(caminho_salvar="output_figs/fig6_explosao_combinatoria.png")
    print(f"  Fig 6 salva: {f6}")

    # Análise do ponto de inviabilidade
    secao("ANÁLISE: Ponto de Inviabilidade da Força Bruta")
    print("  N  |  Tempo FB (ms)  |  Tempo Dijkstra (ms)  |  Razão")
    print("  " + "-" * 55)
    dijk_dict = {m.n_vertices: m.tempo_ms for m in meds_dijk}
    for m in sorted(meds_fb, key=lambda x: x.n_vertices):
        t_dijk = dijk_dict.get(m.n_vertices)
        if t_dijk is None or t_dijk == 0:
            continue
        razao = m.tempo_ms / t_dijk
        print(f"  {m.n_vertices:2d} | {m.tempo_ms:13.4f}   | "
              f"{t_dijk:17.4f}       | {razao:6.1f}x")

    print("\n  Conclusão: A Força Bruta torna-se inviável empiricamente a partir")
    print("  de N ≈ 10–12 vértices, onde o tempo cresce fatorialmente (O(V!)).")
    print("  O Dijkstra mantém desempenho estável até N=100 e além.")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print(LINHA)
    print("  GLOBAL SOLUTION 2026 — Monitoramento de Riscos Ambientais")
    print("  FIAP — Estruturas de Dados e Algoritmos")
    print(LINHA)

    executar_cenario_a()
    executar_cenario_b()
    executar_benchmarks()

    print(f"\n{LINHA}")
    print("  Execução concluída. Figuras salvas em: output_figs/")
    print(LINHA)
