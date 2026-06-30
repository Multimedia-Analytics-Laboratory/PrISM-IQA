import itertools
from collections import deque

import torch

from label_schema import (
    ATTR_GROUP_ID,
    QUESTION_COLS,
    QUESTION_GROUPS,
    SUBSUMPTION_EDGES_BY_NAME,
)


NUM_LEVELS = 4
NUM_BINARY = NUM_LEVELS - 1

DEFAULT_GROUP_SIZES = [len(group) for group in QUESTION_GROUPS]


def _levels_to_cum_bits(levels: torch.Tensor) -> torch.Tensor:
    """
    levels: (S, M) with entries in {0, 1, 2, 3}
    returns: (S, 3M) cumulative binary states
    """
    thresholds = torch.arange(NUM_BINARY, dtype=levels.dtype).view(1, 1, NUM_BINARY)
    bits = (levels.unsqueeze(-1) > thresholds).to(torch.float32)
    return bits.reshape(levels.shape[0], -1)


def _build_mutex_group_states(group_size: int):
    """
    For one mutex group, legal local states are:
    - all zero
    - choose exactly one attribute and assign it one of the 3 positive levels
    """
    num_states = 1 + group_size * NUM_BINARY
    levels = torch.zeros(num_states, group_size, dtype=torch.long)

    state_id = 1
    for attr_idx in range(group_size):
        for level in range(1, NUM_LEVELS):
            levels[state_id, attr_idx] = level
            state_id += 1

    bits = _levels_to_cum_bits(levels)
    return bits, levels


def _build_group_attr_indices():
    group_attr_indices = []
    offset = 0
    for group in QUESTION_GROUPS:
        size = len(group)
        group_attr_indices.append(list(range(offset, offset + size)))
        offset += size
    return group_attr_indices


def _attr_indices_to_bit_indices(attr_indices):
    bit_indices = []
    for attr_idx in attr_indices:
        start = attr_idx * NUM_BINARY
        bit_indices.extend(range(start, start + NUM_BINARY))
    return bit_indices


def _validate_subsumption_edges():
    label_to_attr_idx = {label: idx for idx, label in enumerate(QUESTION_COLS)}
    validated_edges = []

    for src_name, src_thr, dst_name, dst_thr in SUBSUMPTION_EDGES_BY_NAME:
        if src_name not in label_to_attr_idx:
            raise ValueError(f"unknown subsumption source label: {src_name}")
        if dst_name not in label_to_attr_idx:
            raise ValueError(f"unknown subsumption destination label: {dst_name}")
        if not (1 <= int(src_thr) < NUM_LEVELS):
            raise ValueError(f"invalid source threshold @{src_thr} for {src_name}")
        if not (1 <= int(dst_thr) < NUM_LEVELS):
            raise ValueError(f"invalid destination threshold @{dst_thr} for {dst_name}")

        src_attr_idx = label_to_attr_idx[src_name]
        dst_attr_idx = label_to_attr_idx[dst_name]
        validated_edges.append({
            "src_name": src_name,
            "src_attr_idx": src_attr_idx,
            "src_group_idx": ATTR_GROUP_ID[src_attr_idx],
            "src_threshold": int(src_thr),
            "dst_name": dst_name,
            "dst_attr_idx": dst_attr_idx,
            "dst_group_idx": ATTR_GROUP_ID[dst_attr_idx],
            "dst_threshold": int(dst_thr),
        })

    return validated_edges


def _build_connected_components(num_groups, edges):
    adjacency = [set() for _ in range(num_groups)]
    for edge in edges:
        src_group = edge["src_group_idx"]
        dst_group = edge["dst_group_idx"]
        adjacency[src_group].add(dst_group)
        adjacency[dst_group].add(src_group)

    components = []
    visited = [False] * num_groups
    for start in range(num_groups):
        if visited[start]:
            continue

        queue = deque([start])
        visited[start] = True
        component = []

        while queue:
            group_idx = queue.popleft()
            component.append(group_idx)
            for neighbor in sorted(adjacency[group_idx]):
                if not visited[neighbor]:
                    visited[neighbor] = True
                    queue.append(neighbor)

        components.append(sorted(component))

    return components


def _satisfies_subsumption(levels_row: torch.Tensor, local_edges):
    for src_pos, src_thr, dst_pos, dst_thr in local_edges:
        if levels_row[src_pos] >= src_thr and levels_row[dst_pos] < dst_thr:
            return False
    return True


def _enumerate_component_states(component_group_ids, group_levels, group_attr_indices, edges):
    attr_indices = []
    for group_idx in component_group_ids:
        attr_indices.extend(group_attr_indices[group_idx])

    local_pos_by_attr_idx = {attr_idx: pos for pos, attr_idx in enumerate(attr_indices)}
    component_edge_specs = []
    component_group_id_set = set(component_group_ids)
    for edge in edges:
        if edge["src_group_idx"] in component_group_id_set and edge["dst_group_idx"] in component_group_id_set:
            component_edge_specs.append((
                local_pos_by_attr_idx[edge["src_attr_idx"]],
                edge["src_threshold"],
                local_pos_by_attr_idx[edge["dst_attr_idx"]],
                edge["dst_threshold"],
            ))

    per_group_level_tables = [group_levels[group_idx] for group_idx in component_group_ids]
    state_index_ranges = [range(level_table.shape[0]) for level_table in per_group_level_tables]

    valid_rows = []
    for state_ids in itertools.product(*state_index_ranges):
        candidate = torch.cat(
            [per_group_level_tables[group_offset][state_id] for group_offset, state_id in enumerate(state_ids)],
            dim=0,
        )
        if _satisfies_subsumption(candidate, component_edge_specs):
            valid_rows.append(candidate)

    if not valid_rows:
        labels = [QUESTION_COLS[attr_idx] for attr_idx in attr_indices]
        raise ValueError(f"component has no legal states after subsumption filtering: {labels}")

    local_levels = torch.stack(valid_rows, dim=0)
    local_states = _levels_to_cum_bits(local_levels)
    bit_indices = _attr_indices_to_bit_indices(attr_indices)
    return attr_indices, bit_indices, local_states, local_levels


def build_factorized_legal_states():
    group_bits = []
    group_levels = []
    for group in QUESTION_GROUPS:
        bits, levels = _build_mutex_group_states(len(group))
        group_bits.append(bits)
        group_levels.append(levels)

    del group_bits  # only the level tables are needed to enumerate merged components

    subsumption_edges = _validate_subsumption_edges()
    group_attr_indices = _build_group_attr_indices()
    components = _build_connected_components(len(QUESTION_GROUPS), subsumption_edges)

    local_states = []
    local_levels = []
    component_attr_indices = []
    component_bit_indices = []
    component_group_ids = []
    component_labels = []
    state_sizes = []

    for component in components:
        attr_indices, bit_indices, bits, levels = _enumerate_component_states(
            component,
            group_levels,
            group_attr_indices,
            subsumption_edges,
        )
        local_states.append(bits)
        local_levels.append(levels)
        component_attr_indices.append(attr_indices)
        component_bit_indices.append(bit_indices)
        component_group_ids.append(component)
        component_labels.append([QUESTION_COLS[attr_idx] for attr_idx in attr_indices])
        state_sizes.append(bits.shape[0])

    return {
        "mode": "factorized",
        "num_levels": NUM_LEVELS,
        "num_binary": NUM_BINARY,
        "num_attrs": len(QUESTION_COLS),
        "num_bits": len(QUESTION_COLS) * NUM_BINARY,
        "state_sizes": state_sizes,
        "component_attr_indices": component_attr_indices,
        "component_bit_indices": component_bit_indices,
        "component_group_ids": component_group_ids,
        "component_labels": component_labels,
        "local_states": local_states,
        "local_levels": local_levels,
        "mutex_attr_group_id": list(ATTR_GROUP_ID),
    }


def list_legal_states(group_sizes=None):
    if group_sizes is not None and list(group_sizes) != DEFAULT_GROUP_SIZES:
        raise ValueError(
            "custom group_sizes are not supported in v7 because factorization is driven by "
            "the fixed label schema plus named subsumption edges."
        )
    return build_factorized_legal_states()
