"""
performance_monitor.py
======================
Módulo de monitoramento de desempenho dos algoritmos.

Métricas coletadas:
  - Tempo de execução (ms) via time.perf_counter()
  - Memória alocada (MB) via tracemalloc
  - Número de operações elementares
  - Comparação escalabilidade: curva N x tempo
"""

from __future__ import annotations
import time
import tracemalloc
import math
import random
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Any, Optional

from src.data_structures import Grafo, BinarySearchTree, criar_vertice
from src.brute_force import forca_bruta_caminhos, ResultadoForcaBruta
from src.greedy import dijkstra, ResultadoGuloso


# ---------------------------------------------------------------------------
# Estrutura de uma medição
# ---------------------------------------------------------------------------

@dataclass
class Medicao:
    algoritmo: str
    n_vertices: int
    tempo_ms: float
    memoria_mb: float
    num_operacoes: int
    custo_solucao: float
    extra: Dict[str, Any] = field(default_factory=dict)

    def __repr__(self) -> str:
        return (f"Medicao({self.algoritmo}, N={self.n_vertices}, "
                f"t={self.tempo_ms:.3f}ms, "
                f"mem={self.memoria_mb:.4f}MB, "
                f"ops={self.num_operacoes})")


# ---------------------------------------------------------------------------
# Decorador / função de medição genérica
# ---------------------------------------------------------------------------

def medir(func: Callable, *args, **kwargs) -> tuple:
    """
    Executa func(*args, **kwargs) e retorna (resultado, tempo_ms, memoria_mb).
    """
    tracemalloc.start()
    t0 = time.perf_counter()
    resultado = func(*args, **kwargs)
    t1 = time.perf_counter()
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    tempo_ms = (t1 - t0) * 1000
    memoria_mb = peak / (1024 * 1024)
    return resultado, tempo_ms, memoria_mb


# ---------------------------------------------------------------------------
# Geração de grafos sintéticos para benchmark
# ---------------------------------------------------------------------------

def gerar_grafo_sintetico(n: int, seed: int = 42) -> Grafo:
    """
    Gera um grafo conexo ponderado aleatório com N vértices.
    Garante conectividade via spanning tree aleatória + arestas extras.
    """
    rng = random.Random(seed)
    g = Grafo()

    for i in range(n):
        v = criar_vertice(
            id_municipio=1000 + i,
            nome=f"Municipio_{i:04d}",
            indice_risco=round(rng.uniform(0.3, 0.99), 2),
            custo_atendimento=round(rng.uniform(200.0, 2000.0), 1),
            populacao=rng.randint(5000, 1500000)
        )
        g.adicionar_vertice(v)

    ids = g.ids_vertices()

    # Spanning tree aleatória para garantir conectividade
    rng.shuffle(ids)
    for i in range(1, n):
        j = rng.randint(0, i - 1)
        peso = round(rng.uniform(0.5, 8.0), 2)
        g.adicionar_aresta(ids[i], ids[j], peso)

    # Arestas extras (~30% de densidade adicional)
    extras = max(1, n // 3)
    for _ in range(extras):
        u, v = rng.sample(ids, 2)
        # Verifica se aresta já existe
        if v not in {nb for nb, _ in g.vizinhos(u)}:
            peso = round(rng.uniform(0.5, 8.0), 2)
            g.adicionar_aresta(u, v, peso)

    return g


# ---------------------------------------------------------------------------
# Benchmark principal
# ---------------------------------------------------------------------------

class PerformanceMonitor:
    """
    Executa benchmark dos algoritmos para diferentes tamanhos de instância
    e armazena os resultados para geração de gráficos.
    """

    def __init__(self) -> None:
        self.medicoes: List[Medicao] = []

    def benchmark_dijkstra(self, n_list: List[int],
                           repeticoes: int = 3) -> List[Medicao]:
        """Benchmark do Dijkstra para tamanhos N em n_list."""
        resultados: List[Medicao] = []
        for n in n_list:
            tempos, memorias, ops, custos = [], [], [], []
            for rep in range(repeticoes):
                g = gerar_grafo_sintetico(n, seed=42 + rep)
                bst = BinarySearchTree.from_grafo(g)
                origem = g.ids_vertices()[0]

                res, t, mem = medir(dijkstra, g, origem, bst)
                tempos.append(t)
                memorias.append(mem)
                ops.append(res.num_operacoes)
                custos.append(res.custo_total)

            m = Medicao(
                algoritmo="Dijkstra",
                n_vertices=n,
                tempo_ms=sum(tempos) / len(tempos),
                memoria_mb=sum(memorias) / len(memorias),
                num_operacoes=int(sum(ops) / len(ops)),
                custo_solucao=sum(custos) / len(custos)
            )
            self.medicoes.append(m)
            resultados.append(m)
        return resultados

    def benchmark_forca_bruta(self, n_list: List[int],
                              repeticoes: int = 3) -> List[Medicao]:
        """
        Benchmark da Força Bruta para tamanhos N em n_list.
        Atenção: não executar para N > 12 (explosão combinatória).
        """
        resultados: List[Medicao] = []
        for n in n_list:
            if n > 12:
                print(f"[AVISO] N={n} > 12: Força Bruta ignorada (explosão combinatória)")
                continue
            tempos, memorias, ops, custos = [], [], [], []
            for rep in range(repeticoes):
                g = gerar_grafo_sintetico(n, seed=42 + rep)
                ids = g.ids_vertices()
                origem, destino = ids[0], ids[-1]

                res, t, mem = medir(forca_bruta_caminhos, g, origem, destino)
                tempos.append(t)
                memorias.append(mem)
                ops.append(res.num_chamadas_recursivas)
                custos.append(res.melhor_custo if res.melhor_custo < math.inf else 0)

            m = Medicao(
                algoritmo="Forca Bruta",
                n_vertices=n,
                tempo_ms=sum(tempos) / len(tempos),
                memoria_mb=sum(memorias) / len(memorias),
                num_operacoes=int(sum(ops) / len(ops)),
                custo_solucao=sum(custos) / len(custos)
            )
            self.medicoes.append(m)
            resultados.append(m)
        return resultados

    def calcular_gap_otimalidade(
            self, n_list: List[int]) -> Dict[int, float]:
        """
        Calcula o gap percentual entre Força Bruta e Dijkstra para N <= 12.

        gap(%) = (custo_dijkstra - custo_fb) / custo_fb * 100

        O Dijkstra é ótimo para caminhos mínimos — o gap deve ser 0 ou próximo
        de 0, confirmando que a estratégia gulosa produz a solução ótima.
        """
        gaps: Dict[int, float] = {}
        for n in n_list:
            if n > 12:
                continue
            g = gerar_grafo_sintetico(n, seed=42)
            ids = g.ids_vertices()
            origem, destino = ids[0], ids[-1]

            res_fb = forca_bruta_caminhos(g, origem, destino)
            res_dijk = dijkstra(g, origem)

            custo_dijk = res_dijk.custos_acumulados.get(destino, math.inf)
            custo_fb = res_fb.melhor_custo

            if custo_fb > 0 and custo_fb < math.inf:
                gap = abs(custo_dijk - custo_fb) / custo_fb * 100
            else:
                gap = 0.0
            gaps[n] = round(gap, 4)
        return gaps

    def tabela_resumo(self) -> str:
        """Gera tabela textual de resumo das medições."""
        linhas = [
            f"{'Algoritmo':<15} {'N':>5} {'Tempo(ms)':>12} "
            f"{'Mem(MB)':>10} {'Ops':>10} {'Custo':>12}",
            "-" * 68
        ]
        for m in sorted(self.medicoes, key=lambda x: (x.algoritmo, x.n_vertices)):
            linhas.append(
                f"{m.algoritmo:<15} {m.n_vertices:>5} {m.tempo_ms:>12.4f} "
                f"{m.memoria_mb:>10.6f} {m.num_operacoes:>10} {m.custo_solucao:>12.2f}"
            )
        return "\n".join(linhas)
