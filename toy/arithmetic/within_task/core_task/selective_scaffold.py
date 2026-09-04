"""Training-example selections for selective maximal scaffolding.

Each observed example joins one encoder-side primitive fact ``(A_i, x) -> z``
to one decoder-side primitive fact ``(B_j, z) -> y``.  This module constructs
sets that either cover those facts exactly once or deliberately leave gaps.
"""

from __future__ import absolute_import

from collections import defaultdict

import numpy as np


def fact_nodes(example, diagnostic_latents):
    """Return the encoder and decoder primitive facts taught by an example."""

    z_value = int(diagnostic_latents[example.example_id])
    return (
        ("a", int(example.a_index), int(example.x)),
        ("b", int(example.b_index), z_value),
    )


def _component_matching_options(examples, diagnostic_latents):
    """Enumerate the two perfect matchings in every degree-two component."""

    node_edges = defaultdict(list)
    for index, example in enumerate(examples):
        for node in fact_nodes(example, diagnostic_latents):
            node_edges[node].append(index)
    bad = [node for node, edges in node_edges.items() if len(edges) != 2]
    if bad:
        raise ValueError("fact graph is not degree two")

    unseen = set(node_edges)
    components = []
    while unseen:
        start = min(unseen)
        queue = [start]
        nodes = set([start])
        edge_indices = set()
        while queue:
            node = queue.pop()
            for edge_index in node_edges[node]:
                edge_indices.add(edge_index)
                for neighbor in fact_nodes(
                        examples[edge_index], diagnostic_latents):
                    if neighbor not in nodes:
                        nodes.add(neighbor)
                        queue.append(neighbor)
        unseen.difference_update(nodes)

        a_nodes = sorted(node for node in nodes if node[0] == "a")
        if len(a_nodes) * 2 != len(nodes):
            raise ValueError("fact component is not balanced")

        options = []
        root = a_nodes[0]
        for forced_edge in sorted(node_edges[root]):
            matched_a = set()
            matched_b = set()
            selected = []

            def choose(edge_index):
                a_node, b_node = fact_nodes(
                    examples[edge_index], diagnostic_latents
                )
                if a_node in matched_a or b_node in matched_b:
                    return False
                matched_a.add(a_node)
                matched_b.add(b_node)
                selected.append(edge_index)
                return True

            if not choose(forced_edge):
                raise AssertionError("forced matching edge was unavailable")
            while len(matched_a) < len(a_nodes):
                available = []
                for a_node in a_nodes:
                    if a_node in matched_a:
                        continue
                    choices = [
                        edge_index for edge_index in node_edges[a_node]
                        if fact_nodes(
                            examples[edge_index], diagnostic_latents
                        )[1] not in matched_b
                    ]
                    if len(choices) == 1:
                        available.append((a_node, choices[0]))
                if not available:
                    raise ValueError("could not complete fact matching")
                unused_node, edge_index = min(available)
                if not choose(edge_index):
                    raise AssertionError("forced continuation was unavailable")
            options.append(tuple(sorted(selected)))
        if options[0] == options[1]:
            raise ValueError("fact component does not have two matchings")
        components.append(tuple(options))
    return components


def distributed_fact_cover(examples, diagnostic_latents):
    """Choose a perfect fact cover using as many observed task edges as possible.

    The fact graph is a union of even cycles.  Every component has two
    alternating perfect matchings.  A small dynamic program over the 16 task
    edge identities chooses orientations that maximize task-edge diversity.
    """

    examples = tuple(examples)
    task_edges = sorted(set(
        (int(example.a_index), int(example.b_index)) for example in examples
    ))
    edge_bit = dict((edge, 1 << index) for index, edge in enumerate(task_edges))
    states = {0: ()}
    for options in _component_matching_options(examples, diagnostic_latents):
        next_states = {}
        for prior_mask, prior_selection in states.items():
            for option in options:
                option_mask = 0
                for example_index in option:
                    example = examples[example_index]
                    option_mask |= edge_bit[
                        (int(example.a_index), int(example.b_index))
                    ]
                mask = prior_mask | option_mask
                selection = prior_selection + tuple(option)
                incumbent = next_states.get(mask)
                if incumbent is None or selection < incumbent:
                    next_states[mask] = selection
        states = next_states
    best_mask, best_selection = min(
        states.items(),
        key=lambda item: (-bin(int(item[0])).count("1"), item[1]),
    )
    unused_best_mask = best_mask
    return tuple(sorted(examples[index].example_id for index in best_selection))


def select_example_ids(data, strategy, seed=None):
    """Construct a declared selective-scaffolding candidate set."""

    examples = tuple(data.observed_examples)
    strategy = str(strategy)
    if strategy == "all":
        selected = [example.example_id for example in examples]
    elif strategy == "diagonal_fact_cover":
        selected = [
            example.example_id for example in examples
            if int(example.b_index) == int(example.a_index)
        ]
    elif strategy == "offdiagonal_fact_cover":
        selected = [
            example.example_id for example in examples
            if int(example.b_index) == (
                int(example.a_index) + 1
            ) % data.library.n
        ]
    elif strategy == "distributed_fact_cover":
        selected = distributed_fact_cover(
            examples, data.diagnostic_latents
        )
    elif strategy == "latent_half_rectangles":
        selected = [
            example.example_id for example in examples
            if int(data.diagnostic_latents[example.example_id])
            < data.library.q // 2
        ]
    elif strategy == "random_half":
        if seed is None:
            raise ValueError("random_half requires a seed")
        rng = np.random.RandomState(int(seed))
        indices = rng.choice(len(examples), len(examples) // 2, replace=False)
        selected = [examples[index].example_id for index in sorted(indices)]
    elif strategy == "input_half":
        selected = [
            example.example_id for example in examples
            if int(example.x) < data.library.q // 2
        ]
    else:
        raise ValueError("unknown selection strategy: {}".format(strategy))
    return tuple(sorted(selected))


def selection_statistics(data, selected_ids):
    """Report fact coverage and task-edge diversity for a candidate set."""

    selected_set = set(selected_ids)
    example_by_id = dict(
        (example.example_id, example) for example in data.observed_examples
    )
    unknown = selected_set - set(example_by_id)
    if unknown:
        raise ValueError("selection contains unknown example IDs")
    a_counts = defaultdict(int)
    b_counts = defaultdict(int)
    edge_counts = defaultdict(int)
    for example_id in selected_set:
        example = example_by_id[example_id]
        a_node, b_node = fact_nodes(example, data.diagnostic_latents)
        a_counts[a_node] += 1
        b_counts[b_node] += 1
        edge_counts[(int(example.a_index), int(example.b_index))] += 1

    all_a = [
        ("a", i, x)
        for i in range(data.library.m) for x in range(data.library.q)
    ]
    all_b = [
        ("b", j, z)
        for j in range(data.library.n) for z in range(data.library.q)
    ]

    def histogram(nodes, counts):
        result = defaultdict(int)
        for node in nodes:
            result[int(counts[node])] += 1
        return dict((str(key), value) for key, value in sorted(result.items()))

    return {
        "selected_example_count": len(selected_set),
        "selected_example_ids": sorted(selected_set),
        "a_fact_count": len(all_a),
        "b_fact_count": len(all_b),
        "covered_a_fact_count": sum(a_counts[node] > 0 for node in all_a),
        "covered_b_fact_count": sum(b_counts[node] > 0 for node in all_b),
        "exact_fact_cover": (
            all(a_counts[node] == 1 for node in all_a)
            and all(b_counts[node] == 1 for node in all_b)
        ),
        "a_fact_multiplicity_histogram": histogram(all_a, a_counts),
        "b_fact_multiplicity_histogram": histogram(all_b, b_counts),
        "observed_task_edges_covered": len(edge_counts),
        "per_task_edge_selected_examples": dict(
            ("a{}_b{}".format(*edge), count)
            for edge, count in sorted(edge_counts.items())
        ),
    }
