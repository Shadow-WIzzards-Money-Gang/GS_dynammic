"""
brute_force.py
==============
Força Bruta — enumeração exaustiva de todos os caminhos entre origem e destino.

Estratégia: recursão com backtracking.
Restrição de uso: N <= 12 vértices (explosão combinatória).

Serve como oráculo de validação para o algoritmo Guloso.
"""

from __future__ import annotations
import math
from typing import List, Dict, Optional, Tuple
from src.data_structures import Grafo


# ---------------------------------------------------------------------------
# Resultado da busca exaustiva
# ---------------------------------------------------------------------------

class ResultadoForcaBruta:
    """Encapsula todos os resultados de uma execução da Força Bruta."""

    def __init__(self) -> None:
        self.melhor_caminho: List[int] = []
        self.melhor_custo: float = math.inf
        self.todos_caminhos: List[Tuple[List[int], float]] = []   # (caminho, custo)
        self.num_chamadas_recursivas: int = 0
        self.num_caminhos_avaliados: int = 0

    def registrar_caminho(self, caminho: List[int], custo: float) -> None:
        self.num_caminhos_avaliados += 1
        self.todos_caminhos.append((list(caminho), custo))
        if custo < self.melhor_custo:
            self.melhor_custo = custo
            self.melhor_caminho = list(caminho)

    def __repr__(self) -> str:
        return (f"ResultadoFB("
                f"melhor_custo={self.melhor_custo:.4f}, "
                f"caminhos_avaliados={self.num_caminhos_avaliados}, "
                f"chamadas_recursivas={self.num_chamadas_recursivas})")


# ---------------------------------------------------------------------------
# Força Bruta — todos os caminhos simples (sem repetição de vértice)
# ---------------------------------------------------------------------------

def forca_bruta_caminhos(grafo: Grafo, origem: int,
                         destino: int) -> ResultadoForcaBruta:
    """
    Enumera TODOS os caminhos simples de `origem` a `destino` no grafo.

    Algoritmo
    ---------
    DFS recursiva com backtracking:
    1. Adiciona vértice atual ao caminho corrente e ao conjunto de visitados.
    2. Se chegou ao destino: registra o caminho e seu custo total.
    3. Caso contrário, expande cada vizinho não visitado.
    4. Remove vértice do conjunto de visitados ao retornar (backtracking).

    Complexidade: O(V!) no pior caso.

    Parâmetros
    ----------
    grafo   : Grafo ponderado não-direcionado
    origem  : id do vértice de partida
    destino : id do vértice de chegada

    Retorno
    -------
    ResultadoForcaBruta com o melhor caminho e estatísticas completas.
    """
    resultado = ResultadoForcaBruta()
    visitados: set = set()
    caminho_atual: List[int] = []
    custo_atual: List[float] = [0.0]   # lista para mutabilidade dentro da recursão

    def _backtrack(u: int) -> None:
        resultado.num_chamadas_recursivas += 1
        visitados.add(u)
        caminho_atual.append(u)

        if u == destino:
            resultado.registrar_caminho(caminho_atual, custo_atual[0])
        else:
            for v, peso in sorted(grafo.vizinhos(u)):
                if v not in visitados:
                    custo_atual[0] += peso
                    _backtrack(v)
                    custo_atual[0] -= peso

        # Backtracking
        caminho_atual.pop()
        visitados.remove(u)

    _backtrack(origem)
    return resultado


# ---------------------------------------------------------------------------
# Força Bruta — todas as subárvores geradoras (MST bruta) para N ≤ 12
# ---------------------------------------------------------------------------

def forca_bruta_mst(grafo: Grafo) -> Tuple[List[Tuple[int, int, float]], float]:
    """
    Encontra a MST por enumeração exaustiva de subconjuntos de arestas.

    Estratégia
    ----------
    Itera sobre todos os subconjuntos de arestas do grafo.
    Para cada subconjunto com (V-1) arestas, verifica se forma uma árvore
    geradora (grafo conexo e acíclico). Mantém a de menor custo total.

    Complexidade: O(2^E * V)  — inviável para E > ~20.

    Retorno
    -------
    (lista de arestas da MST, custo total)
    """
    ids = grafo.ids_vertices()
    V = len(ids)

    # Coleta arestas únicas (evita duplicata de não-direcionado)
    arestas_set = set()
    arestas: List[Tuple[int, int, float]] = []
    for u in ids:
        for v, peso in grafo.vizinhos(u):
            chave = (min(u, v), max(u, v))
            if chave not in arestas_set:
                arestas_set.add(chave)
                arestas.append((u, v, peso))

    E = len(arestas)
    melhor_custo = math.inf
    melhor_arestas: List[Tuple[int, int, float]] = []

    # Enumera todos os subconjuntos de tamanho (V-1)
    for mask in range(1 << E):
        if bin(mask).count('1') != V - 1:
            continue
        selecionadas = [arestas[i] for i in range(E) if mask & (1 << i)]
        custo = sum(p for _, _, p in selecionadas)

        if custo >= melhor_custo:
            continue

        # Verifica conectividade via Union-Find simplificado
        if _e_arvore_geradora(ids, selecionadas):
            melhor_custo = custo
            melhor_arestas = selecionadas

    return melhor_arestas, melhor_custo


def _e_arvore_geradora(ids: List[int],
                       arestas: List[Tuple[int, int, float]]) -> bool:
    """Verifica se as arestas formam uma árvore geradora (conexa e acíclica)."""
    # Union-Find
    pai = {v: v for v in ids}

    def find(x: int) -> int:
        while pai[x] != x:
            pai[x] = pai[pai[x]]
            x = pai[x]
        return x

    def union(x: int, y: int) -> bool:
        rx, ry = find(x), find(y)
        if rx == ry:
            return False   # ciclo
        pai[rx] = ry
        return True

    for u, v, _ in arestas:
        if not union(u, v):
            return False   # ciclo detectado

    # Verifica conectividade: todos no mesmo componente?
    raiz = find(ids[0])
    return all(find(v) == raiz for v in ids)


# ---------------------------------------------------------------------------
# Crescimento combinatório — geração de dados para o gráfico
# ---------------------------------------------------------------------------

def contar_caminhos_para_n(n: int) -> int:
    """
    Estima o número máximo de caminhos simples em um grafo completo de N vértices.
    Usado para o gráfico de explosão combinatória.

    Para um grafo completo, o número de caminhos simples de s a t é:
    sum_{k=1}^{N-1} P(N-2, k-1) = sum_{k=0}^{N-2} (N-2)! / (N-2-k)!
    """
    total = 0
    for k in range(1, n):
        perm = 1
        for i in range(n - 2, n - 2 - (k - 1), -1):
            if i < 0:
                break
            perm *= max(i, 1)
        total += perm
    return total


def dados_explosao_combinatoria(n_range: List[int]) -> Dict[int, int]:
    """Retorna {N: numero_estimado_de_caminhos} para fins de visualização."""
    return {n: contar_caminhos_para_n(n) for n in n_range}
