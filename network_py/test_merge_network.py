import unittest
import random
from merge_network import MergeNetwork
from typing import List

def simulate_merge(network: MergeNetwork, x_data: List[int], y_data: List[int]) -> List[int]:
    """
    Simulates the merge network on actual sorted data lists.
    
    Args:
        network: The constructed MergeNetwork object.
        x_data: The first list of sorted numbers.
        y_data: The second list of sorted numbers.

    Returns:
        The final state of the data after all comparisons have been applied.
    """
    d1 = len(x_data)
    d2 = len(y_data)
    num_wires = d1 + d2
    
    # If there's nothing to merge, return the initial data.
    if num_wires == 0:
        return []
        
    sim_data = [float('inf')] * num_wires 
    
    # Place the actual data at the positions corresponding to the original wire indices.
    # This assumes the network was built with indices range(d1) and range(d1, d1+d2).
    for i in range(d1):
        sim_data[i] = x_data[i]
    for i in range(d2):
        sim_data[d1 + i] = y_data[i]
        
    sorted_layers = sorted(network.comparators_by_layer.keys())
    for layer_idx in sorted_layers:
        for i, j in network.comparators_by_layer[layer_idx]:
            if sim_data[i] > sim_data[j]:
                sim_data[i], sim_data[j] = sim_data[j], sim_data[i]
                
    return sim_data

class TestMergeNetwork(unittest.TestCase):
    
    def run_test_case(self, x_data, y_data, k, case_name):
        """Helper function to run a single test case and assert correctness."""
        d1, d2 = len(x_data), len(y_data)
        x_indices = list(range(d1))
        y_indices = list(range(d1, d1 + d2))
        
        network = MergeNetwork(x_indices, y_indices, k)
        
        output_data = simulate_merge(network, x_data, y_data)
        
        expected_result = sorted(x_data + y_data)[:k]
        
        # The network must place the k smallest elements into the first k positions.
        # Since the merge network's output should be sorted, we compare directly.
        result_top_k = [output_data[i] for i in network.perm]

        # if expected_result != result_top_k:

        self.assertEqual(result_top_k, expected_result, f"Failed case: {case_name, x_data, y_data}\n{network.__str__()}")
        # Commenting out the print statement to avoid clutter during randomized tests.
        # print(f"✅ PASSED: {case_name} (N1={d1}, N2={d2}, K={k})")

    def test_full_merge_even(self):
        """Tests a basic, non-truncated merge of two equal-sized lists."""
        self.run_test_case([10, 30], [20, 40], 4, "Full Merge (2+2)")
        self.run_test_case([1, 5, 8, 9], [2, 3, 6, 7], 8, "Full Merge (4+4)")

    def test_full_merge_uneven(self):
        """Tests merging lists of unequal size."""
        self.run_test_case([10, 30, 50], [20, 40], 5, "Full Merge (3+2)")
        self.run_test_case([2], [1, 3, 4, 5], 5, "Full Merge (1+4)")

    def test_truncated_merge(self):
        """Tests when k is smaller than the total number of elements."""
        self.run_test_case([10, 40, 50, 60], [20, 30, 70, 80], 3, "Truncated Merge (k=3)")
        self.run_test_case([1, 8, 9], [2, 3, 4], 1, "Truncated Merge (k=1)")

    def test_edge_cases(self):
        """Tests edge cases like empty inputs."""
        self.run_test_case([10, 20], [], 2, "Edge Case (y is empty)")
        self.run_test_case([], [30, 40], 2, "Edge Case (x is empty)")
        self.run_test_case([10], [5], 2, "Edge Case (1+1)")
        self.run_test_case([8], [1, 10], 2, "")
        self.run_test_case([8], [1, 6, 10], 2, "")

    def test_randomized_larger_cases(self):
        """Runs a series of tests with randomly generated larger inputs."""
        num_tests = 1000
        max_list_size = 10
        print(f"\nRunning {num_tests} randomized larger test cases...")


        for i in range(num_tests):
            d1 = random.randint(0, max_list_size)
            d2 = random.randint(0, max_list_size)
            
            # Ensure at least one list has elements for a meaningful test
            if d1 == 0 and d2 == 0:
                d1 = random.randint(1, max_list_size)

            total_size = d1 + d2
            k = random.randint(1, total_size)

            # Generate two distinct, sorted lists of random numbers
            pool = range(total_size * 3)
            full_sample = random.sample(pool, total_size)
            x_data = sorted(full_sample[:d1])
            y_data = sorted(full_sample[d1:])
            
            case_name = f"Random Test #{i+1} (N1={d1}, N2={d2}, K={k})"
            
            # Use subTest to report failures on a per-case basis
            with self.subTest(case_name=case_name):
                self.run_test_case(x_data, y_data, k, case_name)
        print("✅ PASSED: All randomized tests completed successfully.")

if __name__ == '__main__':
    unittest.main(argv=['first-arg-is-ignored'], exit=False)

