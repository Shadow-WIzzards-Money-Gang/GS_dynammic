"""
tests/test_algorithms.py
========================
Testes unitários automatizados (pytest) para o sistema de monitoramento.

Cobertura:
  - BinarySearchTree: insert, search, in_order, height, remove
  - Grafo: add_vertex, add_edge, bfs, dfs
  - Força Bruta: caminhos, MST bruta
  - Dijkstra: caminho mínimo, reconstrução
  - Prim / Kruskal: MST
  - Performance Monitor: benchmark executa sem erro
"""

import math
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
from src.data_structures import (
    Grafo, BinarySearchTree, criar_vertice,
    vertice_id, vertice_risco, vertice_nome
)
from src.brute_force import forca_bruta_caminhos, forca_bruta_mst
from src.greedy import dijkstra, prim, kruskal, reconstruir_caminho
from src.performance_monitor import gerar_grafo_sintetico, PerformanceMonitor


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def grafo_simples():
    """Grafo triangular simples: A--B--C--A com pesos diferentes."""
    g = Grafo()
    for args in [
        (1, "A", 0.5, 100.0, 10000),
        (2, "B", 0.8, 200.0, 20000),
        (3, "C", 0.3, 150.0, 15000),
    ]:
        g.adicionar_vertice(criar_vertice(*args))
    g.adicionar_aresta(1, 2, 1.0)
    g.adicionar_aresta(2, 3, 2.0)
    g.adicionar_aresta(1, 3, 4.0)
    return g


@pytest.fixture
def grafo_linear():
    """Grafo linear: 1--2--3--4--5 (caminho único)."""
    g = Grafo()
    for i in range(1, 6):
        g.adicionar_vertice(criar_vertice(i, f"V{i}", i * 0.1, 100.0, 1000))
    for i in range(1, 5):
        g.adicionar_aresta(i, i + 1, float(i))
    return g


@pytest.fixture
def bst_populada():
    """BST com 7 municípios de risco variado."""
    bst = BinarySearchTree()
    dados = [
        (1, "M1", 0.50, 100.0, 1000),
        (2, "M2", 0.75, 200.0, 2000),
        (3, "M3", 0.30, 150.0, 1500),
        (4, "M4", 0.90, 300.0, 3000),
        (5, "M5", 0.60, 180.0, 1800),
        (6, "M6", 0.20, 120.0, 1200),
        (7, "M7", 0.85, 250.0, 2500),
    ]
    for d in dados:
        bst.inserir(criar_vertice(*d))
    return bst


# ---------------------------------------------------------------------------
# Testes — Grafo
# ---------------------------------------------------------------------------

class TestGrafo:
    def test_adicionar_vertice(self, grafo_simples):
        assert grafo_simples.num_vertices() == 3

    def test_adicionar_aresta(self, grafo_simples):
        assert grafo_simples.num_arestas() == 3

    def test_vizinhos(self, grafo_simples):
        vizinhos_a = {v for v, _ in grafo_simples.vizinhos(1)}
        assert vizinhos_a == {2, 3}

    def test_vertice_invalido(self, grafo_simples):
        with pytest.raises(KeyError):
            grafo_simples.adicionar_aresta(1, 99, 1.0)

    def test_bfs(self, grafo_linear):
        ordem = grafo_linear.bfs(1)
        assert ordem[0] == 1
        assert set(ordem) == {1, 2, 3, 4, 5}

    def test_dfs(self, grafo_linear):
        ordem = grafo_linear.dfs(1)
        assert ordem[0] == 1
        assert set(ordem) == {1, 2, 3, 4, 5}

    def test_from_json(self, tmp_path):
        import json
        dados = {
            "vertices": [
                {"id": 1, "nome": "A", "indice_risco": 0.5,
                 "custo_atendimento": 100.0, "populacao": 1000}
            ],
            "arestas": []
        }
        arquivo = tmp_path / "test.json"
        arquivo.write_text(json.dumps(dados))
        g = Grafo.from_json(str(arquivo))
        assert g.num_vertices() == 1


# ---------------------------------------------------------------------------
# Testes — BinarySearchTree
# ---------------------------------------------------------------------------

class TestBST:
    def test_tamanho(self, bst_populada):
        assert len(bst_populada) == 7

    def test_altura_positiva(self, bst_populada):
        assert bst_populada.altura() >= 1

    def test_in_order_crescente(self, bst_populada):
        vertices = bst_populada.percurso_in_order()
        riscos = [vertice_risco(v) for v in vertices]
        assert riscos == sorted(riscos), "In-order deve retornar em ordem crescente"

    def test_busca_intervalo(self, bst_populada):
        resultados = bst_populada.buscar(0.7, 1.0)
        for v in resultados:
            assert 0.7 <= vertice_risco(v) <= 1.0

    def test_busca_vazia(self, bst_populada):
        resultados = bst_populada.buscar(0.95, 1.0)
        assert resultados == []

    def test_remocao_existente(self, bst_populada):
        removido = bst_populada.remover(4)  # id=4, risco=0.90
        assert removido is True
        assert len(bst_populada) == 6
        # Risco 0.90 não deve mais aparecer
        riscos = [vertice_risco(v) for v in bst_populada.percurso_in_order()]
        assert 0.90 not in riscos

    def test_remocao_inexistente(self, bst_populada):
        removido = bst_populada.remover(999)
        assert removido is False
        assert len(bst_populada) == 7

    def test_from_grafo(self, grafo_simples):
        bst = BinarySearchTree.from_grafo(grafo_simples)
        assert len(bst) == 3

    def test_iteracao(self, bst_populada):
        vertices = list(bst_populada)
        assert len(vertices) == 7


# ---------------------------------------------------------------------------
# Testes — Força Bruta
# ---------------------------------------------------------------------------

class TestForcaBruta:
    def test_caminhos_triangulo(self, grafo_simples):
        res = forca_bruta_caminhos(grafo_simples, 1, 3)
        # Dois caminhos: 1->3 (custo 4.0) e 1->2->3 (custo 3.0)
        assert res.num_caminhos_avaliados >= 2
        assert abs(res.melhor_custo - 3.0) < 1e-9

    def test_caminho_linear(self, grafo_linear):
        res = forca_bruta_caminhos(grafo_linear, 1, 5)
        assert res.num_caminhos_avaliados >= 1
        # Único caminho possível: 1-2-3-4-5, custo = 1+2+3+4 = 10
        assert abs(res.melhor_custo - 10.0) < 1e-9

    def test_contador_recursivo(self, grafo_simples):
        res = forca_bruta_caminhos(grafo_simples, 1, 3)
        assert res.num_chamadas_recursivas > 0

    def test_mst_bruta_triangulo(self, grafo_simples):
        arestas, custo = forca_bruta_mst(grafo_simples)
        # MST do triângulo: duas arestas de custo mínimo → 1.0 + 2.0 = 3.0
        assert abs(custo - 3.0) < 1e-9
        assert len(arestas) == 2   # V - 1 = 2


# ---------------------------------------------------------------------------
# Testes — Dijkstra
# ---------------------------------------------------------------------------

class TestDijkstra:
    def test_custo_minimo_triangulo(self, grafo_simples):
        res = dijkstra(grafo_simples, 1)
        # 1->3: direto=4.0, via 2=3.0 → mínimo=3.0
        assert abs(res.custos_acumulados[3] - 3.0) < 1e-9

    def test_custo_zero_na_origem(self, grafo_simples):
        res = dijkstra(grafo_simples, 1)
        assert res.custos_acumulados[1] == 0.0

    def test_reconstrucao_caminho(self, grafo_simples):
        res = dijkstra(grafo_simples, 1)
        cam = reconstruir_caminho(res, 3)
        assert cam[0] == 1
        assert cam[-1] == 3
        assert cam == [1, 2, 3]   # caminho mais curto

    def test_todos_alcancados(self, grafo_linear):
        res = dijkstra(grafo_linear, 1)
        assert set(res.custos_acumulados.keys()) == {1, 2, 3, 4, 5}

    def test_consistencia_com_fb(self, grafo_simples):
        """Dijkstra deve concordar com Força Bruta (gap = 0)."""
        fb = forca_bruta_caminhos(grafo_simples, 1, 3)
        dijk = dijkstra(grafo_simples, 1)
        custo_dijk = dijk.custos_acumulados.get(3, math.inf)
        assert abs(custo_dijk - fb.melhor_custo) < 1e-9

    def test_com_bst(self, grafo_simples):
        bst = BinarySearchTree.from_grafo(grafo_simples)
        res = dijkstra(grafo_simples, 1, bst=bst, limiar_risco=0.6)
        assert len(res.custos_acumulados) > 0


# ---------------------------------------------------------------------------
# Testes — Prim e Kruskal
# ---------------------------------------------------------------------------

class TestMST:
    def test_prim_num_arestas(self, grafo_simples):
        res = prim(grafo_simples, 1)
        assert len(res.arestas_mst) == 2   # V - 1

    def test_prim_custo(self, grafo_simples):
        res = prim(grafo_simples, 1)
        assert abs(res.custo_total - 3.0) < 1e-9

    def test_kruskal_num_arestas(self, grafo_simples):
        res = kruskal(grafo_simples)
        assert len(res.arestas_mst) == 2

    def test_kruskal_custo(self, grafo_simples):
        res = kruskal(grafo_simples)
        assert abs(res.custo_total - 3.0) < 1e-9

    def test_prim_kruskal_mesmo_custo(self, grafo_simples):
        res_p = prim(grafo_simples, 1)
        res_k = kruskal(grafo_simples)
        assert abs(res_p.custo_total - res_k.custo_total) < 1e-9


# ---------------------------------------------------------------------------
# Testes — Performance Monitor
# ---------------------------------------------------------------------------

class TestPerformanceMonitor:
    def test_gerar_grafo_sintetico(self):
        g = gerar_grafo_sintetico(10, seed=0)
        assert g.num_vertices() == 10
        assert g.num_arestas() >= 9   # pelo menos uma spanning tree

    def test_benchmark_dijkstra(self):
        monitor = PerformanceMonitor()
        medicoes = monitor.benchmark_dijkstra([5, 8], repeticoes=1)
        assert len(medicoes) == 2
        for m in medicoes:
            assert m.tempo_ms >= 0
            assert m.memoria_mb >= 0
            assert m.num_operacoes > 0

    def test_benchmark_forca_bruta(self):
        monitor = PerformanceMonitor()
        medicoes = monitor.benchmark_forca_bruta([5, 8], repeticoes=1)
        assert len(medicoes) == 2

    def test_gap_otimalidade_zero(self):
        monitor = PerformanceMonitor()
        gaps = monitor.calcular_gap_otimalidade([5, 8])
        for n, gap in gaps.items():
            assert gap < 1.0, f"Gap inesperado para N={n}: {gap:.4f}%"
