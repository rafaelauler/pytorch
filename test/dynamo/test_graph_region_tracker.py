import torch
import torch.fx
from torch._dynamo.graph_region_tracker import GraphRegionTracker
from torch._dynamo.test_case import TestCase


def extract_graph(fn, *args, **kwargs):
    gm = None

    def extract_graph_backend(_gm, *args, **kwargs):
        nonlocal gm
        gm = _gm
        return _gm

    torch.compile(backend=extract_graph_backend)(fn)(*args, **kwargs)
    return gm.graph


def get_nodes_by_name(graph, names):
    nodes = []
    for node in graph.nodes:
        if node.name in names:
            nodes.append(node)

    return nodes


class GraphRegionTrackerTests(TestCase):
    def test_no_single_node_regions(self):
        def inner_fn(x):
            return x + 1

        def fn(x):
            o0 = inner_fn(x)
            o1 = inner_fn(x)
            o2 = inner_fn(x)
            return o0 + o1 + o2

        graph = extract_graph(fn, torch.ones(10, 10))
        region_tracker = GraphRegionTracker()

    def test_mismatched_arg_shapes(self):
        pass

    def test_mismatched_global_state(self):
        pass

    def test_no_cycles_introduced(self):
        pass

    def test_deeply_nested_args(self):
        pass

    def test_overlapping_regions(self):
        pass

    def test_non_node_args(self):
        pass

    def test_different_input_args_for_region(self):
        pass

    def test_tensor_only_outputs(self):
        pass


if __name__ == "__main__":
    from torch._dynamo.test_case import run_tests

    run_tests()
