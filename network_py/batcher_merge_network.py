import math
from collections import defaultdict
from typing import List, Tuple, Dict, Set

class BatcherMergeNetwork:
    """
    Implements the recursive truncated merge (Algorithm 2) as a sorting network.

    This class builds the network structure for merging two sorted lists of wire indices.
    It determines the necessary comparators and the depth required for the merge.

    Attributes:
        depth (int): The number of layers in the merge network.
        comparators_by_layer (Dict[int, Set[Tuple[int, int]]]): A mapping from
            a layer index to the set of comparators at that layer.
    """
    def __init__(self, x_indices: List[int], y_indices: List[int], k: int):
        """
        Initializes and builds the merge network.

        Args:
            x_indices: A list of wire indices representing the first sorted input.
            y_indices: A list of wire indices representing the second sorted input.
            k: The number of smallest elements to select in the output.
        """
        self.d1 = len(x_indices)
        self.d2 = len(y_indices)
        self.k = k
        self.comparators_by_layer: Dict[int, Set[Tuple[int, int]]] = defaultdict(set)
        
        self.perm, self.depth = self._build_recursive_merge(
            x_indices, y_indices, k, start_layer=0
        )

    def _truncate(self, indices: List[int], k: int) -> List[int]:
        """(Helper) Truncates a list of indices to k elements."""
        return indices[:min(len(indices), k)]

    def _build_recursive_merge(
        self, x: List[int], y: List[int], k: int, start_layer: int
    ) -> Tuple[List[int], int]:
        """
        Recursively implements Algorithm 2 to build the network.
        """
        d1, d2 = len(x), len(y)

        if d1 * d2 == 0:
            return self._truncate(x + y, k), 0
        if d1 * d2 == 1:
            comparator = (x[0], y[0])
            self.comparators_by_layer[start_layer].add(comparator)
            return self._truncate([x[0], y[0]], k), 1

        even_x, odd_x = x[0::2], x[1::2]
        even_y, odd_y = y[0::2], y[1::2]

        k_v = math.floor(k / 2) + 1
        k_w = math.floor(k / 2)

        v_indices, v_depth = self._build_recursive_merge(even_x, even_y, k_v, start_layer)
        w_indices, w_depth = self._build_recursive_merge(odd_x, odd_y, k_w, start_layer)
        recursive_depth = max(v_depth, w_depth)

        z_indices = []
        v_ptr, w_ptr = 0, 0
        while v_ptr < len(v_indices) or w_ptr < len(w_indices):
            if v_ptr < len(v_indices):
                z_indices.append(v_indices[v_ptr])
                v_ptr += 1
            if w_ptr < len(w_indices):
                z_indices.append(w_indices[w_ptr])
                w_ptr += 1

        final_comp_layer = start_layer + recursive_depth
        
        for i in range(1, len(z_indices), 2):
            if i + 1 < len(z_indices):
                idx1 = z_indices[i]
                idx2 = z_indices[i+1]
                comparator = (idx1, idx2)
                self.comparators_by_layer[final_comp_layer].add(comparator)

        total_depth = recursive_depth + 1
        return self._truncate(z_indices, k), total_depth

    def __str__(self) -> str:
        """Provides a human-readable summary of the merge network."""
        total_comparisons = sum(len(c) for c in self.comparators_by_layer.values())
        summary = ["="*40, f"Merge Network (d1={self.d1}, d2={self.d2}, k={self.k})", "-"*40]
        summary.append(f"Depth: {self.depth}")
        summary.append(f"Total Comparators: {total_comparisons}")
        summary.append("Layers:")
        if not self.comparators_by_layer:
            summary.append("  - No comparators in this network.")
        else:
            sorted_layers = sorted(self.comparators_by_layer.keys())
            for layer in sorted_layers:
                sorted_comps = sorted(list(self.comparators_by_layer[layer]))
                summary.append(f"  - Layer {layer}: {sorted_comps}")
        summary.append("="*40)
        return "\n".join(summary)

