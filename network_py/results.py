from bitonic_network import BitonicNetwork
from sorting_network import SortingNetwork

def is_power_of_two(x):
    return x > 0 and (x & (x - 1)) == 0

print(f"{'n':>6} {'k':>6} {'bitonic':>30} {'batcher':>30}")
print("-" * 74)
for (n, k) in [(512, 1), (512, 4), (512, 64), (512, 256), (768, 320), (500, 320), (350, 320)]:
    if is_power_of_two(n) and is_power_of_two(k):
        bitonic = BitonicNetwork(n, k)
        bitonic_comparators = sum(len(v) for v in bitonic.comparators_by_layer.values())
        bitonic_depth = len(bitonic.comparators_by_layer)
        bitonic_str = str((bitonic_comparators, bitonic_depth))
    else:
        bitonic_str = "N/A"
    batcher = SortingNetwork(n, k)
    batcher_comparators = sum(len(v) for v in batcher.comparators_by_layer.values())
    batcher_depth = len(batcher.comparators_by_layer)
    print(f"{n:6} {k:6} {bitonic_str:>30} {str((batcher_comparators, batcher_depth)):>30}")
