import math
from collections import defaultdict
from typing import List, Tuple, Dict, Set

class BitonicNetwork:
    """
    Implements a full bitonic sorting network.

    This class builds the network structure for a given number of inputs 'N'.
    It handles non-power-of-two inputs by padding to the next power of two.
    The network is constructed to sort in ascending order.
    
    Attributes:
        N (int): The number of inputs the network is designed for.
        depth (int): The total number of parallel layers in the network.
        comparators_by_layer (Dict[int, Set[Tuple[int, int]]]): A mapping from
            a layer index to the set of comparators at that layer.
    """
    def __init__(self, N: int, k: int):
        """
        Initializes and builds the bitonic sorting network.

        Args:
            N: The total number of input wires.
        """
        self.N = N
        self.k = k
        self.comparators_by_layer: Dict[int, Set[Tuple[int, int]]] = defaultdict(set)
        
        # Bitonic sort requires a power-of-two number of inputs.
        # We find the next power of two and build the network for that size.
        if self.N > 0:
            self.padded_N = 1 << (N - 1).bit_length()
        else:
            self.padded_N = 0

        initial_indices = list(range(self.padded_N))
        
        if not initial_indices:
            self.depth = 0
        else:
            # The top-level call builds a network that sorts in ascending order (up=True).
            self.depth, self.perm = self._build_recursive_sort(initial_indices, 0, k, up=True)


    def _build_recursive_sort(self, indices: List[int], start_layer: int, k: int, up: bool) -> tuple[int, list[int]]:
        """
        Recursively builds the sorting part of the network, which creates a bitonic sequence.
        
        Args:
            indices: The list of wire indices for this sub-network.
            start_layer: The layer index to start building from.
            up: The desired sort direction (True for ascending, False for descending).
        
        Returns:
            The depth of the constructed sub-network.
        """
        d = len(indices)
        if d <= 1:
            return (0, indices)

        # Split and recursively build two smaller sorters in opposite directions.
        mid_point = d // 2
        first_half, second_half = indices[:mid_point], indices[mid_point:]
        
        # The first half is sorted ascending, the second descending to create a bitonic sequence.
        depth1, perm1 = self._build_recursive_sort(first_half, start_layer, k, up=True)
        depth2, perm2 = self._build_recursive_sort(second_half, start_layer, k, up=False)
        recursive_depth = max(depth1, depth2)

        # After the recursive calls, we have a bitonic sequence over all 'indices'.
        # Now, we build the bitonic merger to sort this sequence.
        merge_start_layer = start_layer + recursive_depth
        merge_depth, perm = self._build_bitonic_merge(perm1 + perm2, merge_start_layer, k, up)
        
        return recursive_depth + merge_depth, perm

    def _build_bitonic_merge(self, indices: List[int], start_layer: int, k: int, up: bool) -> Tuple[int, list[int]]:
        """
        Recursively builds the merge part, which sorts a bitonic sequence.

        Args:
            indices: The bitonic sequence of wire indices to sort.
            start_layer: The layer index to start building from.
            up: The sort direction for the final output.
        
        Returns:
            The depth of the merger.
        """
        d = len(indices)
        if d <= 1:
            return 0, indices
        
        # Add the parallel comparison layer.
        step = d // 2
        for i in range(step):
            lhs, rhs = indices[i], indices[i + step]
            if not up:
                lhs, rhs = rhs, lhs
            self.comparators_by_layer[start_layer].add((lhs, rhs))

        if d == self.N:
            return 1, indices[:k] if up else indices[-k:]

        # Recursively build mergers for the two halves.
        mid_point = d // 2
        first_half, second_half = indices[:mid_point], indices[mid_point:]
        
        if k <= d // 2:
            good_half = first_half if up else second_half
            depth, perm = self._build_bitonic_merge(good_half, start_layer + 1, min(k, len(good_half)), up)
            depth = 1 + depth
        else:
            depth1, perm1 = self._build_bitonic_merge(first_half, start_layer + 1, min(k, len(first_half)), up)
            depth2, perm2 = self._build_bitonic_merge(second_half, start_layer + 1, min(k, len(second_half)), up)
            depth, perm = 1 + max(depth1, depth2), perm1 + perm2
        return depth, perm

    def __str__(self) -> str:
        """Provides a human-readable summary of the bitonic network."""
        total_comparisons = sum(len(c) for c in self.comparators_by_layer.values())
        summary = ["="*40, f"Bitonic Network (N={self.N}, Padded_N={self.padded_N})", "-"*40]
        summary.append(f"Depth: {self.depth}")
        summary.append(f"Total Comparators: {total_comparisons}")
        summary.append("Layers:")
        if not self.comparators_by_layer:
            summary.append("  - No comparators in this network.")
        else:
            sorted_layers = sorted(self.comparators_by_layer.keys())
            for layer in sorted_layers:
                sorted_comps = sorted(list(self.comparators_by_layer[layer]))
                summary.append(f"  - Layer {layer}: {len(sorted_comps)}")
        summary.append("="*40)
        return "\n".join(summary)