"""Implementation of TensorNetwork structure."""

from __future__ import annotations
from typing import (
    Any,
    Callable,
    Text,
    Tuple,
    List,
    Sequence,
    Iterable,
    Set,
    Dict,
    TypedDict,
    Optional,
    Hashable,
)
from copy import copy
import pickle
from itertools import chain
from collections import Counter

import opt_einsum as oe
import torch
from networkx import MultiGraph, kamada_kawai_layout
import plotly.io as pio

from .utils import prod
from .visualizers import vis2d, vis3d


__all__ = ["Node", "Nodes", "Edge", "Edges", "TensorNetwork"]


Tensor = Any


# Class for nodes
class NodeCore(TypedDict):
    """Base class for nodes."""

    legs: Sequence[Hashable]


class Node(NodeCore, total=False):
    """Class of nodes."""

    tensor: Tensor
    shape: Sequence[int]
    tags: Set[Hashable]


Nodes = Dict[Hashable, Node]

# nodess: Nodes = {
#     "a": {"shape": (5, 2), "legs": [1, 3]},
#     "b": {"shape": (5, 2), "legs": [2, 3]},
# }


# Class for edges
class EdgeCore(TypedDict):
    """Base class for edges."""

    nodes: Set[Tuple[Hashable, int]]


class Edge(EdgeCore, total=False):
    """Class of nodes."""

    dimension: int
    tags: Set[Hashable]


Edges = Dict[Hashable, Edge]

# edgess: Edges = {
#     1: {"nodes": {("a", 0)}, "dimension": 5},
#     2: {"nodes": {("b", 0)}, "dimension": 5},
#     3: {"nodes": {("a", 1), ("b", 1)}, "dimension": 2},
# }


# Tensor network class
class TensorNetwork(object):
    def __init__(
        self,
        nodes: Optional[Nodes] = None,
        edges: Optional[Edges] = None,
        output_edges: Optional[Sequence[Hashable]] = None,
    ):
        """Implement a TensorNetwork structure.

        Args:
            nodes : Nodes of the tensor network.
            edges: Edges of the tensor network.
            output_edges: Output (dangling) edges of the tensor network.

        Examples:
            A tensor train made up of 3 tensors:
            >>> m = 3  # output dimension
            >>> d = 5  # rank dimension

            >>> nodes = [
                    {"name": "a", "shape": (m, d), "legs": [1, 5]},
                    {"name": "b", "shape": (m, d, d), "legs": [2, 5, 6]},
                    {"name": "c", "shape": (m, d), "legs": [3, 6]},
                ]
            >>> output_edges = [1, 2, 3]

            >>> tensor_net = TensorNetwork(nodes=nodes, output_edges=output_edges)
            >>> tensor_net.edges[5]["tags"].add("left")
            >>> tensor_net.edges[6]["tags"].add("right")

            >>> tensor_net.nodes
            >>> tensor_net.edges
            >>> tensor_net.output_edges

        """
        super().__init__()

        if nodes is None:
            nodes = {}

        if edges is None:
            edges = {}

        if output_edges is None:
            output_edges = []

        self.nodes = nodes
        self.edges = edges
        self.output_edges = output_edges

        self._make_shapes()

        if not edges:
            self._make_edges()

        self._make_tags()

    def _make_shapes(self):
        for k, v in self.nodes.items():
            if "tensor" in v:
                if "shape" in v:
                    assert v["shape"] == v["tensor"].shape
                else:
                    self.nodes[k]["shape"] = v["tensor"].shape
            else:
                assert "shape" in v

    def _make_edges(self):
        self.edges = {}
        for k, v in self.nodes.items():
            for i, e in enumerate(v["legs"]):
                if "shape" in v:
                    dim = v["shape"][i]
                else:
                    dim = v["tensor"].shape[i]

                self.edges.setdefault(
                    e, {"nodes": set(), "dimension": dim, "tags": {e}},
                )
                self.edges[e]["nodes"].add((k, i))

    def _make_tags(self):
        for k, v in self.nodes.items():
            if "tags" not in v.keys():
                self.nodes[k]["tags"] = {k}
            else:
                self.nodes[k]["tags"].add(k)

    @property
    def tensors(self) -> Iterable[Tensor]:
        return (node.get("tensor") for node in self.nodes.values())

    @property
    def shapes(self) -> Iterable[Tensor]:
        return (node.get("shape") for node in self.nodes.values())

    @property
    def legs(self) -> Iterable[Sequence[Hashable]]:
        return (node["legs"] for node in self.nodes.values())

    @property
    def output_shape(self) -> Sequence[Hashable]:
        return tuple(
            self.edges[edge]["dimension"] for edge in self.output_edges
        )

    @property
    def num_nodes(self) -> int:
        return len(self.nodes)

    @property
    def compression(self):
        # degrees of freedom of the input tensor
        df_max = prod(self.output_shape)
        # degrees of freedom of the tensor network
        df_tn = sum(prod(shape) for shape in self.shapes)
        compression = df_max / df_tn
        return compression

    # @property
    # def gain(self) -> int:
    #     E = prod(e["dimension"] for e in self.edges.values())
    #     return (1 / E) ** (1 / self.num_nodes)

    def copy(self) -> TensorNetwork:
        nodes: Nodes = {}
        for k, v in self.nodes.items():
            nodes[k] = dict.fromkeys(v)
            for m, u in v.items():
                nodes[k][m] = u if m == "tensor" else copy(u)

        edges = pickle.loads(pickle.dumps(self.edges, -1))
        output_edges = copy(self.output_edges)
        tn = TensorNetwork(nodes=nodes, edges=edges, output_edges=output_edges)
        return tn

    def tags(self, which: Text = "all") -> Set:

        if which not in {"edges", "nodes", "all"}:
            raise ValueError("'which' is not specified properly.")

        tg: Set = set()
        if which in {"nodes", "all"}:
            for vn in self.nodes.values():
                tg.update(vn["tags"])

        if which in {"edges", "all"}:
            for ve in self.edges.values():
                tg.update(ve["tags"])

        return tg

    def filter_by_tags(
        self, tags: Iterable[Hashable] = [], which: Text = "all"
    ) -> List[Set]:

        if which not in {"edges", "nodes", "all"}:
            raise ValueError("'which' is not specified properly.")

        res: Dict[Hashable, Set] = {tg: set() for tg in tags}
        if which in {"nodes", "all"}:
            for kn, vn in self.nodes.items():
                for tg in vn["tags"]:
                    if tg in res:
                        res[tg].add(kn)

        if which in {"edges", "all"}:
            for ke, ve in self.edges.items():
                for tg in ve["tags"]:
                    if tg in res:
                        res[tg].add(ke)

        return list(res.values())

    def popnode(self, node: Hashable) -> TensorNetwork:
        removed_node = self.nodes.pop(node)
        legs = list(removed_node["legs"])
        leg_freq_dict = Counter(legs)

        linking_legs_indicator = [True] * len(legs)
        for i, e in enumerate(removed_node["legs"]):
            neighboring_nodes = self.edges[e]["nodes"]
            if len(neighboring_nodes) == 1:  # output legs
                # update edges
                del self.edges[e]

                # update output edges
                self.output_edges.remove(e)

                # update node legs
                linking_legs_indicator[i] = False

            elif (
                leg_freq_dict[e] > 1 or e in self.output_edges
            ):  # buffered legs
                # form an identity matrix as a buffer
                buff_tensor = torch.eye(
                    removed_node["shape"][i],
                    out=torch.empty_like(removed_node["tensor"]),
                )
                buff_node = f"Node{e}_{node}[{i}]"
                buff_leg = hash(f"Edge{e}_{node}[{i}]") & ((1 << 64) - 1)

                # update nodes
                self.nodes[buff_node] = {
                    "tensor": buff_tensor,
                    "legs": [e, buff_leg],
                    "tags": {buff_node, "buffer"},
                }

                # update edges
                self.edges[buff_leg] = {
                    "nodes": {(buff_node, 1)},
                    "dimension": buff_tensor.shape[1],
                    "tags": {buff_leg, "buffer"},
                }
                self.edges[e]["nodes"].remove((node, i))
                self.edges[e]["nodes"].add((buff_node, 0))

                # update output edges
                self.output_edges.append(buff_leg)

                # update node legs
                legs[i] = buff_leg

            else:  # regular legs
                # update edges
                self.edges[e]["nodes"].remove((node, i))

                # update output edges
                self.output_edges.append(e)

        removed_node["linking_legs_indicator"] = linking_legs_indicator
        tn = TensorNetwork(
            nodes={node: removed_node}, output_edges=removed_node["legs"]
        )
        return tn

    def popedge(self, edge: Hashable) -> Edges:
        edge_value = self.edges.pop(edge)
        for node, _ in edge_value["nodes"]:
            legs = []
            shape = []
            for e, s in zip(
                self.nodes[node]["legs"], self.nodes[node]["shape"]
            ):
                if e != edge:
                    legs.append(e)
                    shape.append(s)

            self.nodes[node]["legs"] = legs
            self.nodes[node]["shape"] = shape

        self.output_edges = [e for e in self.output_edges if e != edge]
        self._make_edges()
        return {edge: edge_value}

    def join(
        self,
        obj: TensorNetwork,
        linking_edges: Optional[
            Tuple[Sequence[Hashable], Sequence[Hashable]]
        ] = None,
        inplace=False,
    ) -> TensorNetwork:

        if inplace:
            tn = self
        else:
            tn = self.copy()

        if linking_edges is None:
            shared_edges = set(tn.output_edges) & set(obj.output_edges)
            shared_edges_list = list(shared_edges)
            linking_edges = (shared_edges_list, shared_edges_list)

        all_linking_edges = set(linking_edges[0]).union(linking_edges[1])
        obj_output_or_linking_edges = set(obj.output_edges).union(
            all_linking_edges
        )
        output_edges_1 = [
            e for e in tn.output_edges if e not in obj_output_or_linking_edges
        ]
        output_edges_2 = [
            e for e in obj.output_edges if e not in all_linking_edges
        ]
        tn.output_edges = output_edges_1 + output_edges_2

        edges_hash_table = dict(zip(linking_edges[1], linking_edges[0]))
        obj_temp = obj.copy()
        for k, v in edges_hash_table.items():
            e = tn.edges[v]
            del tn.edges[v]
            e["nodes"].update(obj.edges[k]["nodes"])
            tn.edges[v] = e
            for node, _ in obj.edges[k]["nodes"]:
                obj_temp.nodes[node]["legs"] = [
                    edges_hash_table.get(e, e)
                    for e in obj_temp.nodes[node]["legs"]
                ]

            del obj_temp.edges[k]

        tn.nodes.update(obj_temp.nodes)
        tn.edges.update(obj_temp.edges)

        return tn

    def get_subnet(
        self, nodes: Iterable[Hashable] = [], tags: bool = True
    ) -> TensorNetwork:
        if tags:
            nodes_set = set().union(
                *self.filter_by_tags(tags=nodes, which="nodes")
            )
        else:
            nodes_set = set(nodes)

        nodes_dic: Nodes = {}
        output_edges: List[Hashable] = []
        for node in nodes_set:
            nodes_dic[node] = self.nodes[node]

        for k, v in self.edges.items():
            tp = tuple(n in nodes_set for n, _ in v["nodes"])
            if any(tp) and ((not all(tp)) or (k in self.output_edges)):
                output_edges.append(k)

        tn = TensorNetwork(nodes=nodes_dic, output_edges=output_edges)
        return tn

    def rename_nodes(self, name_map: Dict[Hashable, Hashable]):
        for n_old, n_new in name_map.items():
            self.nodes[n_new] = self.nodes[n_old]
            self.nodes[n_new]["tags"].add(n_new)
            self.nodes[n_new]["tags"].discard(n_old)
            del self.nodes[n_old]
            for i, edge in enumerate(self.nodes[n_new]["legs"]):
                self.edges[edge]["nodes"].add((n_new, i))
                self.edges[edge]["nodes"].remove((n_old, i))

    def rename_edges(self, name_map: Dict[Hashable, Hashable]):
        for e_old, e_new in name_map.items():
            self.edges[e_new] = self.edges[e_old]
            self.edges[e_new]["tags"].add(e_new)
            self.edges[e_new]["tags"].discard(e_old)
            for i, e in enumerate(self.output_edges):
                if e == e_old:
                    self.output_edges[i] = e_new

            del self.edges[e_old]
            for node, mode in self.edges[e_new]["nodes"]:
                self.nodes[node]["legs"][mode] = e_new

    def set_con_info(self, **kwargs):
        args = chain(*zip(self.shapes, self.legs))
        _, con_info = oe.contract_path(*args, shapes=True, **kwargs)
        self.con_info = con_info

    def contract(self, **kwargs):
        if hasattr(self, "con_info") and hasattr(self.con_info, "path"):
            kwargs.setdefault("optimize", self.con_info.path)

        args = chain(*zip(self.tensors, self.legs))
        out = oe.contract(*args, **kwargs)
        return out

    def contract_expression(self, **kwargs):
        args = [*chain(*zip(self.shapes, self.legs)), self.output_edges]
        _, con_info = oe.contract_path(*args, shapes=True, **kwargs)
        kwargs["optimize"] = con_info.path
        con_expr = oe.contract_expression(
            con_info.eq, *con_info.shapes, **kwargs
        )
        return self._expression_decorator(con_expr), con_info

    def _expression_decorator(self, expr):
        def wrapper(*args, **kwargs):
            tensors_dict = dict(*args, **kwargs)
            tensors = [tensors_dict[k] for k in self.nodes]
            return expr(*tensors)

        return wrapper

    def to_graph(self):
        vertices = []
        for kn, vn in self.nodes.items():
            vertices.append(
                (kn, {"legs": vn["legs"], "tags": vn["tags"], "size": 1})
            )

        output_counter = 0
        junction_counter = 0
        # junctions = []
        # outputs = []
        edges = []
        for ke, ve in self.edges.items():
            edge_nodes = [n for n, _ in ve["nodes"]]
            edge_attr = {
                "dimension": ve["dimension"],
                "tags": ve["tags"],
                "type": set(),
            }
            if ke in self.output_edges:
                while True:
                    output = "out" + str(output_counter)
                    output_counter += 1
                    if output not in self.nodes:
                        vertices.append(
                            (
                                output,
                                {
                                    "legs": [ke],
                                    "tags": {output},
                                    "type": {"output"},
                                    "size": 0,
                                },
                            )
                        )
                        edge_attr["type"].add("output")
                        break
                edge_nodes.append(output)

            if len(edge_nodes) > 2:
                while True:
                    junction = "-".join([*edge_nodes, str(junction_counter)])
                    junction_counter += 1
                    if junction not in self.nodes:
                        vertices.append(
                            (
                                junction,
                                {
                                    "legs": [ke] * len(edge_nodes),
                                    "tags": {junction},
                                    "type": {"junction"},
                                    "size": 0,
                                },
                            )
                        )
                        edge_attr["type"].add("hyperedge")
                        break

                for e in edge_nodes:
                    edges.append((e, junction, edge_attr))
            else:
                edges.append((*edge_nodes, edge_attr))

        g = MultiGraph()
        g.add_nodes_from(vertices)
        g.add_edges_from(edges)

        # n_vertices = len(vertices)
        # g = Graph(n_vertices)
        # for vertix, vertix_attr in vertices:
        #     g.add_vertex(vertix, **vertix_attr)

        for v0, v1, edge_attr in edges:
            g.add_edge(v0, v1, **edge_attr)

        return g

    def visualize(self, dim: int = 2, graph_layout: Callable = None, **kwargs):
        if graph_layout is None:

            def fun(g, dim):
                return kamada_kawai_layout(g, dim=dim)

            graph_layout = fun
            kwargs = {}

        g = self.to_graph()
        pos = graph_layout(g, dim=dim, **kwargs)
        if dim == 2:
            fig = vis2d(g, pos)
        elif dim == 3:
            fig = vis3d(g, pos)
        else:
            ValueError("`dim` must be 2 or 3.")

        pio.show(fig)
        return fig, g
