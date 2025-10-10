import math
from collections import defaultdict
from typing import List, Tuple, Dict, Set

# We import the MergeNetwork component we built in the previous step.
from batcher_merge_network import BatcherMergeNetwork

class BatcherNetwork:
    """
    Implements the recursive truncated odd-even merge sort.
    
    Includes one optimization:
    1. Alekseev's merge for the final top-level combination step.
    """
    def __init__(self, N: int, k: int):
        """
        Initializes and builds the full sorting network.

        Args:
            N: The total number of input wires.
            k: The number of smallest elements to find.
        """
        self.N = N
        self.k = k
        self.comparators_by_layer: Dict[int, Set[Tuple[int, int]]] = defaultdict(set)
        
        initial_indices = list(range(N))
        # The initial call is marked as the top level to trigger Alekseev's merge.
        if k <= N - k:
            # Get a Min(k) network
            self.perm, self.depth = self._build_recursive_sort(initial_indices, k, 0, is_top_level=True)
        else:
            # Call with N - k
            perm, self.depth = self._build_recursive_sort(initial_indices, N - k, 0, is_top_level=True)
            # Invert all comparators so it becomes a Max(N - k) network
            for layer in self.comparators_by_layer:
                swapped = set((b, a) for (a, b) in self.comparators_by_layer[layer])
                self.comparators_by_layer[layer] = swapped
            # Complement the Max(N - k) outputs, thus obtaining a Min(k) network
            self.perm = [i for i in range(N) if i not in set(perm)]

    def _chunk_size(self, d: int, k: int) -> int:
        """(Helper) Implements the ChunkSize function from Algorithm 3."""
        if k <= 0: return math.ceil(d / 2)
        mu = 1 << (k - 1).bit_length() if k > 1 else 1
        if d <= mu:
            return math.ceil(d / 2)
        else:
            return mu * math.ceil(d / (2 * mu))

    def _alekseev_merge(self, v_perm: List[int], w_perm: List[int], k: int, start_layer: int) -> Tuple[List[int], int]:
        """
        (Optimization) A simple, non-sorting merge for the final step.
        """
        n, m = len(v_perm), len(w_perm)
        assert n <= k and m <= k
        comps, res_perm = [], []

        for i in range(k):
            if i < n and k - i - 1 < m:
                res_perm.append(v_perm[i])
                comps.append((v_perm[i], w_perm[k - i - 1]))
            elif i < n and k - i - 1 >= m:
                res_perm.append(v_perm[i])
            elif i >= n and k - i - 1 < m:
                res_perm.append(w_perm[k - i - 1])

        self.comparators_by_layer[start_layer].update(comps)
        return res_perm, 1

    def _build_recursive_sort(self, x_indices: List[int], k: int, start_layer: int, is_top_level: bool = False) -> Tuple[List[int], int]:
        """
        Recursively implements Algorithm 3.
        """
        if k == 0:
            return [], 0

        d = len(x_indices)
        if d <= 1:
            return x_indices, 0

        chunk_idx = self._chunk_size(d, k)
        chunk1, chunk2 = x_indices[:chunk_idx], x_indices[chunk_idx:]

        # Recursively build the sub-networks.
        v_perm, v_depth = self._build_recursive_sort(chunk1, k, start_layer)
        w_perm, w_depth = self._build_recursive_sort(chunk2, k, start_layer)

        recursive_depth = max(v_depth, w_depth)
        merge_start_layer = start_layer + recursive_depth
        
        if is_top_level:
            # Top-level merge is always Alekseev's.
            final_perm, merge_depth = self._alekseev_merge(v_perm, w_perm, k, merge_start_layer)
        else:
            # Internal steps use the standard MergeNetwork.
            merge_net = BatcherMergeNetwork(v_perm, w_perm, k)
            
            for layer, comparators in merge_net.comparators_by_layer.items():
                self.comparators_by_layer[merge_start_layer + layer].update(comparators)
            
            final_perm = merge_net.perm
            merge_depth = merge_net.depth
            
        total_depth = recursive_depth + merge_depth
        return final_perm, total_depth

    def __str__(self) -> str:
        """Provides a human-readable summary of the sorting network."""
        total_comparisons = sum(len(c) for c in self.comparators_by_layer.values())
        summary = ["="*40, f"Sorting Network (N={self.N}, k={self.k})", "-"*40]
        summary.append(f"Depth: {self.depth}")
        summary.append(f"Total Comparators: {total_comparisons}")
        #summary.append(f"Final Permutation (Top {len(self.perm)} wires): {self.perm}")
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

