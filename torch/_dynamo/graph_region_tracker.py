import math
from collections import defaultdict, deque
from typing import Any, Callable, Deque, Dict, List, Optional, Set, Tuple

import torch.fx


Node = torch.fx.Node
Region = List[Node]
IdenticalNodes = List[Node]


# This is typical BFS with the caveat
# that a node's children need to be explicitly
# added with the add_children() method
# The flow is yield a node and check if it's valid for all regions
# if not valid, discard and continue onto the next node
class BfsRegionIter:
    def __init__(self, origin: Node) -> None:
        self._cur_node: Tuple[Optional[str], Optional[Node]] = (None, origin)
        self._queue: Deque[Tuple[Optional[str], Optional[Node]]] = deque()

    @staticmethod
    def create(origin: Node) -> "BfsRegionIter":
        it = BfsRegionIter(origin)
        it.add_children(origin)
        return it

    def next(self) -> Tuple[Optional[str], Optional[Node]]:
        ret_node = self._cur_node
        if not self._queue:
            self._cur_node = (None, None)
        else:
            self._cur_node = self._queue.popleft()
        return ret_node

    def peek(self) -> Tuple[Optional[str], Optional[Node]]:
        return self._cur_node

    def add_children(self, node: Node) -> None:
        arg: Any
        for arg in node.args:
            if isinstance(arg, Node):
                self._queue.append((None, arg))

        key: str
        kwarg: Any
        for key, kwarg in node.kwargs.items():
            if isinstance(kwarg, Node):
                self._queue.append((key, kwarg))


class GraphRegionTracker:
    def __init__(self) -> None:
        self.loc_to_duplicates: Dict[str, IdenticalNodes] = defaultdict(list)
        self.node_to_duplicates: Dict[Node, IdenticalNodes] = {}

    @staticmethod
    def _get_loc_str(filename: str, lineno: int) -> str:
        return f"{filename}:{lineno}"

    def track_node(self, filename: str, lineno: int, node: Node) -> None:
        loc_str = self._get_loc_str(filename, lineno)
        duplicates = self.loc_to_duplicates[loc_str]
        duplicates.append(node)
        self.node_to_duplicates[node] = duplicates

    def has_same_loc(self, n0: Node, n1: Node) -> bool:
        return (
            n0 in self.node_to_duplicates
            and n1 in self.node_to_duplicates
            and self.node_to_duplicates[n0] == self.node_to_duplicates[n1]
        )

    def get_identical_regions(self, graph: torch.fx.Graph) -> List[List[Region]]:
        topological_ranking = {node: i for i, node in enumerate(graph.nodes)}
        group_ranking = {}
        region_groups = []

        # Create region groups; a region group is a group
        # of regions that are all identical. In this initial state
        # each region in the group is a single node, and we discard
        # groups that are only a single region.
        # We track the topological ranking to start with groups later in the graph
        # the reason for this is that we will necessarily create the largest groups first.
        for group in self.loc_to_duplicates.values():
            if len(group) > 1:
                region_group = []
                min_rank = math.inf
                for node in group:
                    min_rank = min(min_rank, topological_ranking[node])
                    region_group.append([node])

                region_groups.append(region_group)
                group_ranking[id(region_group)] = min_rank

        region_groups.sort(key=lambda g: -group_ranking[id(g)])

        # We start from regions later in the graph and expand them earlier
        # as a result, we will create the largest regions first and they won't
        # overlap.
        seen_nodes: Set[Node] = set()
        for region_group in region_groups:
            fully_expand_region_group(region_group, seen_nodes, self.has_same_loc)

        return [
            region_group for region_group in region_groups if len(region_group[0]) > 1
        ]

    def __str__(self) -> str:
        return f"GraphRegionTracker(loc_to_duplicates={self.loc_to_duplicates}, node_to_duplicates={self.node_to_duplicates})"


def fully_expand_region_group(
    regions: List[Region],
    seen_nodes: Set[Node],
    has_same_loc_fn: Callable[[Node, Node], bool],
) -> None:
    # All regions should start with 1 node
    assert all(len(region) == 1 for region in regions)
    region_iters = []
    for region in regions:
        (origin,) = region  # Only works for 1 element sets
        region_iters.append(BfsRegionIter.create(origin))

    nodes_to_add: List[Node] = []

    # we already have the origin node in each region
    for region_it in region_iters:
        _, node = region_it.next()
        assert node
        region_it.add_children(node)

    # arg_name is set for kwargs, None for args
    current_arg_name, current_node = region_iters[0].next()
    assert current_node is not None
    seen_nodes.add(current_node)
    # Loop incrementally adding new nodes to each region
    # regions are only expanded if the node to add is valid
    # for ALL regions
    while current_node:
        add_node = True
        nodes_to_add.clear()
        nodes_to_add.append(current_node)
        for region_it in region_iters[1:]:
            arg_name, node = region_it.next()

            if node:
                add_node &= (
                    current_arg_name == arg_name
                    and node not in seen_nodes
                    and has_same_loc_fn(node, current_node)
                )
                nodes_to_add.append(node)
                seen_nodes.add(node)
            else:
                add_node = False

        if add_node:
            for region, region_it, node in zip(regions, region_iters, nodes_to_add):
                region.append(node)
                region_it.add_children(node)

        current_arg_name, current_node = region_iters[0].next()

    # Ensure regions are sorted in topological order
    for region in regions:
        region.reverse()
