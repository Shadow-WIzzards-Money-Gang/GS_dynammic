"""
greedy.py
=========
Algoritmos Gulosos para o sistema de monitoramento de riscos ambientais.

Algoritmo principal escolhido: Dijkstra — caminho mínimo de fonte única.

Justificativa da escolha
-------------------------
O Dijkstra foi escolhido em detrimento de Prim/Kruskal porque o cenário
principal (Enchentes RS) exige encontrar a ROTA DE MENOR CUSTO DE ATENDIMENTO
a partir de um hub central (Porto Alegre) para cada município afetado —
não uma árvore de cobertura mínima. O Dijkstra resolve exatamente esse problema
com complexidade O((V + E) log V) usando heap binário.

Implementações bonus: Prim (MST) e Kruskal (MST) para comparação.
"""

from __future__ import annotations
import heapq
import math
from typing import Dict, List, Optional, Tuple, Set
from src.data_structures import Grafo, BinarySearchTree, vertice_risco, vertice_id


# ---------------------------------------------------------------------------
# Resultado genérico de algoritmo guloso
# ---------------------------------------------------------------------------

class ResultadoGuloso:
    """Encapsula resultado e métricas de um algoritmo guloso."""

    def __init__(self, algoritmo: str) -> None:
        self.algoritmo = algoritmo
        self.custo_total: float = 0.0
        self.caminho: List[int] = []                          # para Dijkstra
        self.arestas_mst: List[Tuple[int, int, float]] = []  # para Prim/Kruskal
        self.custos_acumulados: Dict[int, float] = {}
        self.predecessores: Dict[int, Optional[int]] = {}
        self.num_operacoes: int = 0                           # inserções no heap / arestas relaxadas
        self.vertices_priorizados: List[int] = []             # municípios de alto risco priorizados

    def __repr__(self) -> str:
        return (f"ResultadoGuloso(algoritmo={self.algoritmo}, "
                f"custo_total={self.custo_total:.4f}, "
                f"operacoes={self.num_operacoes})")


# ---------------------------------------------------------------------------
# Dijkstra — caminho mínimo de fonte única
# ---------------------------------------------------------------------------

def dijkstra(grafo: Grafo, origem: int,
             bst: Optional[BinarySearchTree] = None,
             limiar_risco: float = 0.7) -> ResultadoGuloso:
    """
    Algoritmo de Dijkstra com integração à BST de risco.

    Estratégia gulosa
    -----------------
    A cada iteração, extrai o vértice de menor custo acumulado do heap (min-heap).
    Para cada vizinho, "relaxa" a aresta: se o novo custo for menor que o registrado,
    atualiza e insere no heap. A escolha local ótima (menor custo acumulado) garante
    a solução global ótima para grafos sem arestas negativas (prova: invariante do heap).

    Integração com BST
    ------------------
    Se uma BST for fornecida, os municípios com índice_risco >= limiar_risco são
    priorizados: seus custos iniciais recebem um bônus de -0.01 no heap para
    desempate, garantindo que municípios de alto risco sejam visitados primeiro
    quando o custo acumulado for igual.

    Parâmetros
    ----------
    grafo         : Grafo ponderado
    origem        : id do vértice de partida (hub de recursos)
    bst           : BST opcional para priorizar municípios de alto risco
    limiar_risco  : municípios com risco >= limiar são priorizados

    Complexidade: O((V + E) log V)

    Retorno
    -------
    ResultadoGuloso com custos mínimos a todos os vértices alcançáveis.
    """
    resultado = ResultadoGuloso("Dijkstra")

    # Consulta BST: identifica municípios de alto risco
    alto_risco: Set[int] = set()
    if bst is not None:
        for v in bst.buscar(limiar_risco, 1.0):
            alto_risco.add(vertice_id(v))
    resultado.vertices_priorizados = list(alto_risco)

    # Inicialização
    dist: Dict[int, float] = {v: math.inf for v in grafo.ids_vertices()}
    pred: Dict[int, Optional[int]] = {v: None for v in grafo.ids_vertices()}
    dist[origem] = 0.0

    # Heap: (custo_acumulado, id_vertice)
    # Municípios de alto risco recebem desempate de -0.001
    heap: List[Tuple[float, int]] = [(0.0, origem)]

    visitados: Set[int] = set()

    while heap:
        custo_u, u = heapq.heappop(heap)
        resultado.num_operacoes += 1

        if u in visitados:
            continue
        visitados.add(u)

        for v, peso in grafo.vizinhos(u):
            if v in visitados:
                continue
            novo_custo = dist[u] + peso
            # Bônus de desempate para municípios de alto risco
            desempate = -0.001 if v in alto_risco else 0.0
            if novo_custo < dist[v]:
                dist[v] = novo_custo
                pred[v] = u
                resultado.num_operacoes += 1
                heapq.heappush(heap, (novo_custo + desempate, v))

    resultado.custos_acumulados = {k: v for k, v in dist.items() if v < math.inf}
    resultado.predecessores = pred
    resultado.custo_total = sum(d for d in dist.values() if d < math.inf)
    return resultado


def reconstruir_caminho(resultado: ResultadoGuloso, destino: int) -> List[int]:
    """Reconstrói o caminho mínimo da origem ao destino a partir dos predecessores."""
    caminho: List[int] = []
    atual: Optional[int] = destino
    while atual is not None:
        caminho.append(atual)
        atual = resultado.predecessores.get(atual)
    caminho.reverse()
    return caminho


# ---------------------------------------------------------------------------
# Prim — Árvore Geradora Mínima (MST)
# ---------------------------------------------------------------------------

def prim(grafo: Grafo, raiz: int) -> ResultadoGuloso:
    """
    Algoritmo de Prim para MST.

    Estratégia gulosa
    -----------------
    Mantém um conjunto de vértices já na árvore (in_tree).
    A cada passo, escolhe a aresta de MENOR PESO que conecta um vértice
    da árvore a um vértice fora dela — decisão local ótima.
    Usa min-heap para selecionar eficientemente a aresta mínima.

    Complexidade: O((V + E) log V)
    """
    resultado = ResultadoGuloso("Prim")
    ids = grafo.ids_vertices()

    chave: Dict[int, float] = {v: math.inf for v in ids}
    pai: Dict[int, Optional[int]] = {v: None for v in ids}
    in_tree: Set[int] = set()

    chave[raiz] = 0.0
    heap: List[Tuple[float, int]] = [(0.0, raiz)]

    while heap:
        custo, u = heapq.heappop(heap)
        resultado.num_operacoes += 1

        if u in in_tree:
            continue
        in_tree.add(u)

        if pai[u] is not None:
            resultado.arestas_mst.append((pai[u], u, custo))
            resultado.custo_total += custo

        for v, peso in grafo.vizinhos(u):
            if v not in in_tree and peso < chave[v]:
                chave[v] = peso
                pai[v] = u
                resultado.num_operacoes += 1
                heapq.heappush(heap, (peso, v))

    resultado.predecessores = pai
    return resultado


# ---------------------------------------------------------------------------
# Kruskal — Árvore Geradora Mínima via ordenação de arestas
# ---------------------------------------------------------------------------

def kruskal(grafo: Grafo) -> ResultadoGuloso:
    """
    Algoritmo de Kruskal para MST.

    Estratégia gulosa
    -----------------
    Ordena TODAS as arestas por peso (crescente).
    Adiciona cada aresta se ela não criar ciclo (verificado com Union-Find).
    Decisão local ótima: sempre escolhe a aresta mais barata disponível.

    Complexidade: O(E log E)   (dominado pela ordenação)

    Comparação com Prim
    -------------------
    Kruskal é mais eficiente para grafos ESPARSOS (E ~ V).
    Prim é mais eficiente para grafos DENSOS (E ~ V^2) com heap de Fibonacci.
    Na prática, para redes municipais (esparsas), Kruskal e Prim são equivalentes.
    """
    resultado = ResultadoGuloso("Kruskal")
    ids = grafo.ids_vertices()

    # Coleta arestas únicas e ordena por peso
    arestas_set: set = set()
    arestas: List[Tuple[float, int, int]] = []
    for u in ids:
        for v, peso in grafo.vizinhos(u):
            chave = (min(u, v), max(u, v))
            if chave not in arestas_set:
                arestas_set.add(chave)
                arestas.append((peso, u, v))
    arestas.sort()   # O(E log E)

    # Union-Find
    pai: Dict[int, int] = {v: v for v in ids}
    rank: Dict[int, int] = {v: 0 for v in ids}

    def find(x: int) -> int:
        while pai[x] != x:
            pai[x] = pai[pai[x]]
            x = pai[x]
        return x

    def union(x: int, y: int) -> bool:
        rx, ry = find(x), find(y)
        if rx == ry:
            return False
        if rank[rx] < rank[ry]:
            rx, ry = ry, rx
        pai[ry] = rx
        if rank[rx] == rank[ry]:
            rank[rx] += 1
        return True

    for peso, u, v in arestas:
        resultado.num_operacoes += 1
        if union(u, v):
            resultado.arestas_mst.append((u, v, peso))
            resultado.custo_total += peso
            if len(resultado.arestas_mst) == len(ids) - 1:
                break   # MST completa

    return resultado
