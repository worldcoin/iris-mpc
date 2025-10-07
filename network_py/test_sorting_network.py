import unittest
import random
from sorting_network import SortingNetwork
from typing import List

def simulate_sort(network: SortingNetwork, input_data: List[int]) -> List[int]:
    """Simulates the full sorting network on an unsorted list of data."""
    # Start with a direct copy of the input data
    sim_data = list(input_data)
    
    # Apply comparators layer by layer
    for layer_idx in sorted(network.comparators_by_layer.keys()):
        for i, j in network.comparators_by_layer[layer_idx]:
            # Ensure indices from comparator are valid for the sim_data list
            if i < len(sim_data) and j < len(sim_data):
                if sim_data[i] > sim_data[j]:
                    sim_data[i], sim_data[j] = sim_data[j], sim_data[i]
            else:
                # This would indicate a bug in network generation, but we handle it defensively.
                raise IndexError(f"Comparator indices ({i}, {j}) out of bounds for data of length {len(sim_data)}")

    return sim_data

class TestSortingNetwork(unittest.TestCase):
    """A strong test suite for the complete SortingNetwork."""

    def run_sort_test_case(self, N, k, case_name):
        """Helper function to run a single test case for the sorting network."""
        network = SortingNetwork(N, k)
        
        # Generate random, unsorted data
        input_data = random.sample(range(N * 3), N)
        
        # Simulate the network's effect on the data
        output_data = simulate_sort(network, input_data)
        
        # Determine the correct answer
        expected_result = sorted(input_data)[:k]
        
        # Get the actual result by picking from the output wires specified by the network's permutation
        # The final permutation tells us where the top K elements reside. We should sort this subset
        # to have a consistent order for comparison.
        actual_result = sorted([output_data[i] for i in network.perm])

        self.assertEqual(actual_result, expected_result, f"Failed case: {case_name}\n{network}")

    def test_small_sort(self):
        """Tests a basic N, K scenario."""
        self.run_sort_test_case(N=8, k=4, case_name="Small Sort (8, 4)")
        self.run_sort_test_case(N=16, k=3, case_name="Paper Example (16, 3)")

    def test_full_sort(self):
        """Tests a full sort where K = N."""
        self.run_sort_test_case(N=8, k=8, case_name="Full Sort (8, 8)")

    def test_non_power_of_two(self):
        """Tests an input size that is not a power of two."""
        self.run_sort_test_case(N=7, k=3, case_name="Non-Power-of-Two (7, 3)")

    def test_edge_cases(self):
        """Tests edge cases for N and K."""
        self.run_sort_test_case(N=10, k=1, case_name="Edge Case (k=1)")
        self.run_sort_test_case(N=2, k=2, case_name="Edge Case (2, 2)")
        self.run_sort_test_case(N=1, k=1, case_name="Edge Case (1, 1)")

    def test_randomized_larger_sorts(self):
        """Runs a series of tests with randomly generated larger inputs."""
        num_tests = 1000
        max_n = 50  # Networks get very large, so keep N reasonable
        print(f"\nRunning {num_tests} randomized larger sort test cases...")

        for i in range(num_tests):
            N = random.randint(2, max_n)
            k = random.randint(1, N)
            case_name = f"Random Sort #{i+1} (N={N}, K={k})"
            
            with self.subTest(case_name=case_name):
                self.run_sort_test_case(N, k, case_name)
        print("✅ PASSED: All randomized sort tests completed successfully.")


if __name__ == '__main__':
    unittest.main(argv=['first-arg-is-ignored'], exit=False)

