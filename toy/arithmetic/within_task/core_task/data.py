"""Primitive libraries, graph conditions, data partitions, and integrity checks."""

from __future__ import absolute_import

from collections import namedtuple

import numpy as np


DEFAULT_Q = 16
DEFAULT_M = 8
DEFAULT_N = 8

Example = namedtuple("Example", "example_id a_index b_index x y")


class PrimitiveLibrary(object):
    """A reproducibly sampled collection of A and B permutations."""

    def __init__(self, a_permutations, b_permutations, seed):
        self.a_permutations = tuple(np.asarray(p, dtype=np.int64) for p in a_permutations)
        self.b_permutations = tuple(np.asarray(p, dtype=np.int64) for p in b_permutations)
        self.seed = int(seed)
        if not self.a_permutations or not self.b_permutations:
            raise ValueError("a primitive library must contain A and B permutations")
        self.q = int(self.a_permutations[0].shape[0])
        self.m = len(self.a_permutations)
        self.n = len(self.b_permutations)


class ExperimentData(object):
    """Model-facing examples plus a physically separate diagnostic latent store."""

    def __init__(self, library, observed_edges, observed_examples,
                 heldout_examples, diagnostic_latents, graph_name):
        self.library = library
        self.observed_edges = tuple(sorted(observed_edges))
        self.observed_examples = tuple(observed_examples)
        self.heldout_examples = tuple(heldout_examples)
        self.diagnostic_latents = dict(diagnostic_latents)
        self.graph_name = str(graph_name)


def sample_primitive_library(seed, q=DEFAULT_Q, m=DEFAULT_M, n=DEFAULT_N):
    """Sample independent uniform random permutations using only ``seed``."""

    rng = np.random.RandomState(int(seed))
    a_permutations = [rng.permutation(q) for _ in range(m)]
    b_permutations = [rng.permutation(q) for _ in range(n)]
    return PrimitiveLibrary(a_permutations, b_permutations, seed)


def connected_cycle_edges(size=8):
    """Return E_conn = {(i,i), (i,i+1 mod size)}."""

    return tuple(sorted(
        set((i, j) for i in range(size) for j in (i, (i + 1) % size))
    ))


def disconnected_cycle_edges(size=8):
    """Return two matched degree-two cycles on equal A/B partitions."""

    if size % 2 != 0 or size < 4:
        raise ValueError("disconnected cycles require an even size of at least four")
    group_size = size // 2
    edges = set()
    for start in (0, group_size):
        for local_i in range(group_size):
            i = start + local_i
            edges.add((i, start + local_i))
            edges.add((i, start + ((local_i + 1) % group_size)))
    return tuple(sorted(edges))


def edge_distances(observed_edges, m=DEFAULT_M, n=DEFAULT_N):
    """Shortest bipartite path length from each A_i to each B_j."""

    adjacency = {}
    for i in range(m):
        adjacency[("a", i)] = []
    for j in range(n):
        adjacency[("b", j)] = []
    for i, j in observed_edges:
        adjacency[("a", i)].append(("b", j))
        adjacency[("b", j)].append(("a", i))

    distances = {}
    for i in range(m):
        start = ("a", i)
        queue = [(start, 0)]
        visited = set([start])
        cursor = 0
        while cursor < len(queue):
            node, distance = queue[cursor]
            cursor += 1
            if node[0] == "b":
                distances[(i, node[1])] = distance
            for neighbor in adjacency[node]:
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append((neighbor, distance + 1))
    return distances


def compose(left, right):
    """Return ``left o right`` for permutations stored as output-by-input maps."""

    left = np.asarray(left, dtype=np.int64)
    right = np.asarray(right, dtype=np.int64)
    return left[right]


def inverse(permutation):
    """Invert a permutation stored as an input-to-output array."""

    permutation = np.asarray(permutation, dtype=np.int64)
    result = np.empty_like(permutation)
    result[permutation] = np.arange(permutation.shape[0], dtype=np.int64)
    return result


def permutation_matrix(permutation):
    """Create M[output, input] such that M @ one_hot(input) is one_hot(output)."""

    permutation = np.asarray(permutation, dtype=np.int64)
    q = permutation.shape[0]
    matrix = np.zeros((q, q), dtype=np.int64)
    matrix[permutation, np.arange(q)] = 1
    return matrix


def composite(library, a_index, b_index):
    return compose(
        library.b_permutations[b_index],
        library.a_permutations[a_index],
    )


def build_experiment_data(library, observed_edges, graph_name):
    """Build exhaustive edge/input examples and keep z out of example records."""

    observed_edge_set = set(tuple(edge) for edge in observed_edges)
    observed_examples = []
    heldout_examples = []
    diagnostic_latents = {}

    for i in range(library.m):
        for j in range(library.n):
            outputs = composite(library, i, j)
            for x in range(library.q):
                example_id = "a{}_b{}_x{}".format(i, j, x)
                example = Example(example_id, i, j, x, int(outputs[x]))
                diagnostic_latents[example_id] = int(library.a_permutations[i][x])
                if (i, j) in observed_edge_set:
                    observed_examples.append(example)
                else:
                    heldout_examples.append(example)

    return ExperimentData(
        library=library,
        observed_edges=observed_edge_set,
        observed_examples=observed_examples,
        heldout_examples=heldout_examples,
        diagnostic_latents=diagnostic_latents,
        graph_name=graph_name,
    )


def _is_permutation(values, q):
    values = np.asarray(values)
    return values.shape == (q,) and np.array_equal(np.sort(values), np.arange(q))


def validate_experiment_data(data):
    """Run the required pre-training integrity checks, raising on any failure."""

    library = data.library
    observed_edges = set(data.observed_edges)
    all_edges = set((i, j) for i in range(library.m) for j in range(library.n))
    missing_edges = all_edges - observed_edges

    checks = {}
    checks["all_primitives_bijective"] = all(
        _is_permutation(p, library.q)
        for p in library.a_permutations + library.b_permutations
    )

    observed_by_edge = dict((edge, set()) for edge in observed_edges)
    for example in data.observed_examples:
        observed_by_edge[(example.a_index, example.b_index)].add(example.x)
    checks["observed_edges_have_all_inputs"] = all(
        xs == set(range(library.q)) for xs in observed_by_edge.values()
    )

    training_edges = set(
        (example.a_index, example.b_index) for example in data.observed_examples
    )
    checks["no_missing_edge_in_training"] = not bool(training_edges & missing_edges)

    all_examples = data.observed_examples + data.heldout_examples
    checks["outputs_match_composition"] = all(
        example.y == int(composite(
            library, example.a_index, example.b_index
        )[example.x])
        for example in all_examples
    )

    checks["partition_sizes_correct"] = (
        len(data.observed_examples) == len(observed_edges) * library.q
        and len(data.heldout_examples) == len(missing_edges) * library.q
    )
    if library.q == 16 and library.m == 8 and library.n == 8 and len(observed_edges) == 16:
        checks["spec_split_is_256_observed_768_heldout"] = (
            len(data.observed_examples) == 256
            and len(data.heldout_examples) == 768
        )

    checks["latent_excluded_from_model_examples"] = (
        Example._fields == ("example_id", "a_index", "b_index", "x", "y")
        and set(data.diagnostic_latents) == set(e.example_id for e in all_examples)
    )
    ids = [example.example_id for example in all_examples]
    checks["example_ids_unique_and_fixed"] = len(ids) == len(set(ids))

    failed = sorted(name for name, passed in checks.items() if not passed)
    if failed:
        raise AssertionError("data integrity checks failed: {}".format(", ".join(failed)))
    return checks
