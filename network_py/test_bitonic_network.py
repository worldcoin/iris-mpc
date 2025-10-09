import unittest
import random
from bitonic_network import BitonicNetwork
from typing import List

def simulate_sort(network: BitonicNetwork, input_data: List[int]) -> List[int]:
    """
    Simulates the full bitonic sorting network on an unsorted list of data.
    Handles padding for non-power-of-two inputs.
    """
    N = len(input_data)
    padded_N = network.padded_N

    # Create a list of the padded size, filling extra spots with infinity.
    sim_data = [float('inf')] * padded_N
    for i in range(N):
        sim_data[i] = input_data[i]
    
    # Apply comparators layer by layer
    for layer_idx in sorted(network.comparators_by_layer.keys()):
        for i, j in network.comparators_by_layer[layer_idx]:
            if sim_data[i] > sim_data[j]:
                sim_data[i], sim_data[j] = sim_data[j], sim_data[i]

    # Return only the initial N elements, discarding the padded infinity values.
    return sim_data[:N]

class TestBitonicNetwork(unittest.TestCase):
    """A strong test suite for the complete BitonicNetwork."""

    def run_sort_test_case(self, N, k, case_name):
        """Helper function to run a single test case for the sorting network."""
        if N == 0:
            self.assertEqual([], [])
            return
        
        network = BitonicNetwork(N, k)
        
        # Generate random, unsorted data
        input_data = random.sample(range(N * 3), N)
        
        # Simulate the network's effect on the data
        output_data = simulate_sort(network, input_data)
        output_data = [output_data[network.perm[i]] for i in range(k)]
        output_data = sorted(output_data)
        # Determine the correct answer
        expected_result = sorted(input_data)[:k]
        
        self.assertEqual(output_data, expected_result, f"Failed case: {case_name}\n{network}")

    def test_power_of_two(self):
        """Tests input sizes that are powers of two."""
        self.run_sort_test_case(N=2, k = 1, case_name="Power of Two (N=2)")
        self.run_sort_test_case(N=4, k = 2,case_name="Power of Two (N=4)")
        self.run_sort_test_case(N=8, k = 8, case_name="Power of Two (N=8)")
        self.run_sort_test_case(N=16, k = 4, case_name="Power of Two (N=16)")

    def test_edge_cases(self):
        """Tests edge cases for N."""
        self.run_sort_test_case(N=1, k = 1, case_name="Edge Case (N=1)")
        self.run_sort_test_case(N=0, k = 0, case_name="Edge Case (N=0)")

    def test_randomized_larger_sorts(self):
        """Runs a series of tests with randomly generated larger inputs."""
        num_tests = 1000
        max_log = 10  # Bitonic networks are efficient, so we can test larger N.
        print(f"\nRunning {num_tests} randomized larger bitonic sort test cases...")

        for i in range(num_tests):
            n_log = random.randint(1, max_log)
            N = 2 ** n_log
            k = 2 ** random.randint(0, n_log)
            case_name = f"Random Sort #{i+1} (N={N}, k = {k}))"
            
            with self.subTest(case_name=case_name):
                self.run_sort_test_case(N, k, case_name)
        print("✅ PASSED: All randomized bitonic sort tests completed successfully.")

if __name__ == '__main__':
    unittest.main(argv=['first-arg-is-ignored'], exit=False)