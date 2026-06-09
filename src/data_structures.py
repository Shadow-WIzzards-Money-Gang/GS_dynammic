"""
data_structures.py
==================
Implementação das estruturas de dados do sistema de monitoramento de riscos ambientais.

Estruturas implementadas:
  - Grafo ponderado (dicionário de listas de adjacência)
  - Árvore Binária de Busca (BST) para municípios por índice de risco
  - Funções auxiliares para manipulação de vértices e arestas
"""

from __future__ import annotations
import json
from collections import deque
from typing import Optional, List, Tuple, Dict, Iterator


# ---------------------------------------------------------------------------
# Tipo canônico de vértice: tupla imutável
# (id_municipio, nome, indice_risco, custo_atendimento, populacao)
# ---------------------------------------------------------------------------
Vertice = Tuple[int, str, float, float, int]


def criar_vertice(id_municipio: int, nome: str, indice_risco: float,
                  custo_atendimento: float, populacao: int) -> Vertice:
    """Cria e retorna um vértice imutável (tupla) de município."""
    return (id_municipio, nome, float(indice_risco), float(custo_atendimento), int(populacao))


def vertice_id(v: Vertice) -> int:          return v[0]
def vertice_nome(v: Vertice) -> str:        return v[1]
def vertice_risco(v: Vertice) -> float:     return v[2]
def vertice_custo(v: Vertice) -> float:     return v[3]
def vertice_populacao(v: Vertice) -> int:   return v[4]


# ---------------------------------------------------------------------------
# Grafo — dicionário de listas de adjacência
# Justificativa: lista de adjacência tem complexidade de espaço O(V + E),
# adequada para grafos esparsos (típico em redes municipais).
# Matriz de adjacência exigiria O(V^2), desperdiçando memória.
# ---------------------------------------------------------------------------

class Grafo:
    """
    Grafo ponderado não-direcionado representado como dicionário de adjacência.

    Atributos internos
    ------------------
    _vertices : dict[int, Vertice]
        Mapeia id_municipio → tupla de atributos
    _adj : dict[int, list[tuple[int, float]]]
        Mapeia id_municipio → [(vizinho, peso), ...]
    """

    def __init__(self) -> None:
        self._vertices: Dict[int, Vertice] = {}
        self._adj: Dict[int, List[Tuple[int, float]]] = {}

    # ------------------------------------------------------------------
    # Inserção
    # ------------------------------------------------------------------

    def adicionar_vertice(self, v: Vertice) -> None:
        """Adiciona vértice ao grafo; ignora duplicatas."""
        vid = vertice_id(v)
        if vid not in self._vertices:
            self._vertices[vid] = v
            self._adj[vid] = []

    def adicionar_aresta(self, u: int, v: int, peso: float) -> None:
        """Adiciona aresta não-direcionada (u, v) com peso."""
        if u not in self._adj or v not in self._adj:
            raise KeyError(f"Vértice {u} ou {v} não existe no grafo.")
        self._adj[u].append((v, peso))
        self._adj[v].append((u, peso))

    # ------------------------------------------------------------------
    # Consultas
    # ------------------------------------------------------------------

    def vizinhos(self, u: int) -> List[Tuple[int, float]]:
        """Retorna lista de (vizinho, peso) para o vértice u."""
        return self._adj.get(u, [])

    def get_vertice(self, vid: int) -> Optional[Vertice]:
        return self._vertices.get(vid)

    def vertices(self) -> List[Vertice]:
        return list(self._vertices.values())

    def ids_vertices(self) -> List[int]:
        return list(self._vertices.keys())

    def num_vertices(self) -> int:
        return len(self._vertices)

    def num_arestas(self) -> int:
        return sum(len(adj) for adj in self._adj.values()) // 2

    # ------------------------------------------------------------------
    # Travessias
    # ------------------------------------------------------------------

    def bfs(self, origem: int) -> List[int]:
        """BFS a partir da origem; retorna lista de ids visitados em ordem."""
        visitados: set = set()
        fila: deque = deque([origem])
        visitados.add(origem)
        ordem: List[int] = []
        while fila:
            u = fila.popleft()
            ordem.append(u)
            for v, _ in sorted(self._adj[u]):          # ordena para determinismo
                if v not in visitados:
                    visitados.add(v)
                    fila.append(v)
        return ordem

    def dfs(self, origem: int) -> List[int]:
        """DFS iterativa a partir da origem; retorna lista de ids visitados."""
        visitados: set = set()
        pilha: List[int] = [origem]
        ordem: List[int] = []
        while pilha:
            u = pilha.pop()
            if u not in visitados:
                visitados.add(u)
                ordem.append(u)
                for v, _ in sorted(self._adj[u], reverse=True):
                    if v not in visitados:
                        pilha.append(v)
        return ordem

    # ------------------------------------------------------------------
    # Persistência
    # ------------------------------------------------------------------

    def to_dict(self) -> dict:
        return {
            "vertices": [list(v) for v in self._vertices.values()],
            "adj": {str(k): val for k, val in self._adj.items()}
        }

    @classmethod
    def from_json(cls, caminho: str) -> "Grafo":
        """Carrega grafo a partir de arquivo JSON no formato do projeto."""
        with open(caminho, "r", encoding="utf-8") as f:
            dados = json.load(f)
        g = cls()
        for m in dados["vertices"]:
            v = criar_vertice(m["id"], m["nome"], m["indice_risco"],
                              m["custo_atendimento"], m["populacao"])
            g.adicionar_vertice(v)
        for e in dados["arestas"]:
            g.adicionar_aresta(e["u"], e["v"], e["peso"])
        return g

    def __repr__(self) -> str:
        return f"Grafo(V={self.num_vertices()}, E={self.num_arestas()})"


# ---------------------------------------------------------------------------
# Árvore Binária de Busca — BST por índice_risco
# ---------------------------------------------------------------------------

class Node:
    """
    Nó da BST.

    Chave de ordenação: indice_risco (float).
    Payload: tupla Vertice completa.
    """
    __slots__ = ("vertice", "risco", "esquerda", "direita")

    def __init__(self, vertice: Vertice) -> None:
        self.vertice: Vertice = vertice
        self.risco: float = vertice_risco(vertice)
        self.esquerda: Optional[Node] = None
        self.direita: Optional[Node] = None

    def __repr__(self) -> str:
        return f"Node(id={vertice_id(self.vertice)}, risco={self.risco:.2f})"


class BinarySearchTree:
    """
    Árvore Binária de Busca (BST) de municípios por índice de risco.

    Propriedade invariante: risco_esquerda < risco_pai <= risco_direita
    (duplicatas vão para a subárvore direita)

    Complexidade
    ------------
    Inserção   : O(h)    — h = altura da árvore
    Busca range: O(h + k) — k = número de resultados
    In-order   : O(n)
    Altura      : O(n)
    Remoção     : O(h)
    """

    def __init__(self) -> None:
        self._raiz: Optional[Node] = None
        self._tamanho: int = 0

    # ------------------------------------------------------------------
    # Inserção
    # ------------------------------------------------------------------

    def inserir(self, vertice: Vertice) -> None:
        """Insere um município mantendo a propriedade BST."""
        self._raiz = self._inserir_rec(self._raiz, vertice)
        self._tamanho += 1

    def _inserir_rec(self, no: Optional[Node], vertice: Vertice) -> Node:
        if no is None:
            return Node(vertice)
        risco = vertice_risco(vertice)
        if risco < no.risco:
            no.esquerda = self._inserir_rec(no.esquerda, vertice)
        else:
            no.direita = self._inserir_rec(no.direita, vertice)
        return no

    # ------------------------------------------------------------------
    # Busca por intervalo
    # ------------------------------------------------------------------

    def buscar(self, r_min: float, r_max: float) -> List[Vertice]:
        """
        Retorna todos os municípios com índice de risco em [r_min, r_max].
        Complexidade: O(h + k), onde k é o número de resultados.
        """
        resultado: List[Vertice] = []
        self._buscar_rec(self._raiz, r_min, r_max, resultado)
        return resultado

    def _buscar_rec(self, no: Optional[Node], r_min: float,
                    r_max: float, resultado: List[Vertice]) -> None:
        if no is None:
            return
        if r_min < no.risco:
            self._buscar_rec(no.esquerda, r_min, r_max, resultado)
        if r_min <= no.risco <= r_max:
            resultado.append(no.vertice)
        if no.risco <= r_max:
            self._buscar_rec(no.direita, r_min, r_max, resultado)

    # ------------------------------------------------------------------
    # Percurso in-order (crescente)
    # ------------------------------------------------------------------

    def percurso_in_order(self) -> List[Vertice]:
        """Retorna municípios em ordem crescente de risco."""
        resultado: List[Vertice] = []
        self._in_order_rec(self._raiz, resultado)
        return resultado

    def _in_order_rec(self, no: Optional[Node], resultado: List[Vertice]) -> None:
        if no is None:
            return
        self._in_order_rec(no.esquerda, resultado)
        resultado.append(no.vertice)
        self._in_order_rec(no.direita, resultado)

    # ------------------------------------------------------------------
    # Altura
    # ------------------------------------------------------------------

    def altura(self) -> int:
        """Calcula a altura da árvore (0 para árvore vazia)."""
        return self._altura_rec(self._raiz)

    def _altura_rec(self, no: Optional[Node]) -> int:
        if no is None:
            return 0
        return 1 + max(self._altura_rec(no.esquerda), self._altura_rec(no.direita))

    def esta_balanceada(self) -> bool:
        """Retorna True se |altura_esq - altura_dir| <= 1 em todo nó."""
        return self._balanceada_rec(self._raiz) != -1

    def _balanceada_rec(self, no: Optional[Node]) -> int:
        if no is None:
            return 0
        le = self._balanceada_rec(no.esquerda)
        if le == -1:
            return -1
        ld = self._balanceada_rec(no.direita)
        if ld == -1:
            return -1
        if abs(le - ld) > 1:
            return -1
        return 1 + max(le, ld)

    # ------------------------------------------------------------------
    # Remoção
    # ------------------------------------------------------------------

    def remover(self, id_municipio: int) -> bool:
        """
        Remove o nó com o município de id_municipio informado.
        Retorna True se removido, False se não encontrado.
        """
        self._raiz, removido = self._remover_por_id(self._raiz, id_municipio)
        if removido:
            self._tamanho -= 1
        return removido

    def _remover_por_id(self, no: Optional[Node],
                        target_id: int) -> Tuple[Optional[Node], bool]:
        if no is None:
            return None, False

        if vertice_id(no.vertice) == target_id:
            # Nó encontrado — três casos
            if no.esquerda is None:
                return no.direita, True
            if no.direita is None:
                return no.esquerda, True
            # Dois filhos: substitui pelo sucessor in-order (mínimo da subárvore direita)
            sucessor = self._minimo(no.direita)
            no.vertice = sucessor.vertice
            no.risco = sucessor.risco
            no.direita, _ = self._remover_por_id(no.direita, vertice_id(sucessor.vertice))
            return no, True

        # Busca nas duas subárvores (sem saber o risco, percorre as duas)
        no.esquerda, rem_esq = self._remover_por_id(no.esquerda, target_id)
        if rem_esq:
            return no, True
        no.direita, rem_dir = self._remover_por_id(no.direita, target_id)
        return no, rem_dir

    def _minimo(self, no: Node) -> Node:
        while no.esquerda is not None:
            no = no.esquerda
        return no

    # ------------------------------------------------------------------
    # Utilitários
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return self._tamanho

    def __iter__(self) -> Iterator[Vertice]:
        return iter(self.percurso_in_order())

    @classmethod
    def from_grafo(cls, grafo: Grafo) -> "BinarySearchTree":
        """Constrói uma BST a partir de todos os vértices do grafo."""
        bst = cls()
        for v in grafo.vertices():
            bst.inserir(v)
        return bst

    def __repr__(self) -> str:
        return f"BinarySearchTree(n={self._tamanho}, altura={self.altura()})"
