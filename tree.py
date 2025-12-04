import math
import sys

# --- COST FUNCTION CONSTANTS (from the provided image) ---
C0 = 172
C_LOG = 15
C_SQ = 115.5625
C_N2_SPECIAL = 94.375

def cost_of_single_node(n, m):
    """
    Calculates f(n): Cost of a single tournament node with fan-in 'n'.
    n=1 is defined as a zero-cost pass-through (pass-through node performs no computation).
    """
    if n == 1:
        return 0.0
    elif n == 2:
        return calculate_cost_n2_special(m)
    elif n > 2:
        return calculate_total_cost(n, m)
    else:
        return float('inf')
    

def calculate_cost_n2_special(m):
    """
    Computes the total cost for the special case where fan-in n=2,
    using the simplified, combined formula provided by the user.

    Formula (where n in the formula is read as m):
    172 + 52 * m + 2712 * ceil(m/64)

    Args:
        m (int): The number of parallel nodes (instances).

    Returns:
        float: The total cost for n=2.
    """
    # 172 + 52 * m + 2712 * ceil(m/64)
    cost = 172 + 52 * m + 2712 * math.ceil(m / 64)
    return cost

def calculate_total_cost(n, m):
    """
    Computes the total cost for a single layer based on the fan-in (n) and 
    the number of parallel nodes (m).

    - If n=1 (pass-through), cost is 0.
    - If n=2, uses the dedicated combined formula.
    - If n >= 3, computes the sum of the four original component formulas.

    Args:
        n (int): The number of children (fan-in) per node.
        m (int): The number of parallel nodes (instances).

    Returns:
        float: The sum of the cost components.
    """
    if n == 1:
        # Pass-through node (zero cost)
        return 0.0
    
    if n == 2:
        # Special case: use the combined formula
        return calculate_cost_n2_special(m)

    # --- Case: n >= 3 (General Component Sum) ---
    
    # 1. oblivious_cross-compare: 
    # Use direct formula for n_choose_2: n * (n - 1) / 2
    n_choose_2 = n * (n - 1) // 2
    
    # Term 1: 3 * (44 + 4 * n_choose_2 * m + 904 * ceil(n_choose_2 * m / 64))
    cost_1_term = n_choose_2 * m
    cost_1 = 3 * (44 + 
                  4 * cost_1_term + 
                  904 * math.ceil(cost_1_term / 64))

    # 2. binary AND tree: (Only calculated for n >= 3)
    R = math.ceil(math.log2(n))
    # R will be >= 2 for n >= 3, but setting a floor of 1 defensively.
    R = max(1, R) 
            
    summation_term = 0
    # Summation from i=1 up to R (number of rounds/levels)
    for i in range(1, R + 1):
        # Term: ceil(n/2^i * n * m / 64)
        term = (n / math.pow(2, i)) * (n * m / 64)
        summation_term += math.ceil(term)

    # Term 2: 3 * (5 * R + 8 * summation_term)
    cost_2 = 3 * (5 * R + 8 * summation_term)

    # 3. bit_inject_ot_2round:
    # Term 3: 25 + 16 * n * m
    cost_3 = 25 + 16 * n * m

    # 4. conditionally_select_distance:
    # Term 4: 3 * (5 + 8 * n * m)
    cost_4 = 3 * (5 + 8 * n * m)

    total_cost = cost_1 + cost_2 + cost_3 + cost_4
    return total_cost

def solve_optimal_total_cost_tree(N, D, nr_trees):
    """
    Finds the optimal tree structure minimizing the SUM of bandwidths over all layers 
    for N inputs and max depth D, using a two-level DP.
    
    Level 1 (Inner DP): BandwidthDP[k][m] - Minimum cost for a single layer k -> m.
    Level 2 (Outer DP): DP_Sum[k][d] - Minimum total sum of costs for a tree k, depth d.
    """
    if N < 2 or D < 1:
        return "Invalid input: N must be >= 2 and D must be >= 1.", None

    # --- INNER DP: BandwidthDP[k][m] ---
    BandwidthDP = [[float('inf')] * (N + 1) for _ in range(N + 1)]
    FanInConfig = [[None] * (N + 1) for _ in range(N + 1)]
    
    # Base Cases for Inner DP
    BandwidthDP[0][0] = 0.0
    FanInConfig[0][0] = {} 
        
    # Recurrence for Inner DP (Finds min cost partition k -> m)
    for k in range(1, N + 1): 
        for m in range(1, k + 1): 
            # n ranges from 1 up to the max size that still allows m-1 nodes of size >= 1
            for n in range(1, k + 1):
                for f in range(1, m + 1):
                    if f * n > k or m - f < 0:
                        break
                
                    cost_n = cost_of_single_node(n, f * nr_trees)
                    prev_cost = BandwidthDP[k - n * f][m - f]
                    
                    if prev_cost != float('inf'):
                        current_cost = cost_n + prev_cost
                        if current_cost < BandwidthDP[k][m]:
                            BandwidthDP[k][m] = current_cost
                            prev_config = FanInConfig[k - n * f][m - f]
                            new_config = prev_config.copy()
                            new_config[n] = new_config.get(n, 0) + f
                            FanInConfig[k][m] = new_config

    # --- OUTER DP (SUM): DP_Sum[k][d] ---
    DP_Sum = [[float('inf')] * (D + 1) for _ in range(N + 1)]
    RootConfig = [[-1] * (D + 1) for _ in range(N + 1)]

    # Base Cases: d = 1
    for k in range(2, N + 1):
        DP_Sum[k][1] = BandwidthDP[k][1]
        RootConfig[k][1] = 1 

    # Recurrence: d = 2 to D (Minimizing Sum of Bandwidths)
    for d in range(2, D + 1):
        for k in range(2, N + 1):
            # Iterate over all possible number of outputs 'm' for the root layer (1 <= m < k)
            for m in range(1, k):
                
                B_current_layer = BandwidthDP[k][m]
                
                if B_current_layer == float('inf') or DP_Sum[m][d - 1] == float('inf'):
                    continue
                
                # Minimizing SUM: Current Layer Cost + Subtree Total Cost
                total_cost = B_current_layer + DP_Sum[m][d - 1]

                if total_cost < DP_Sum[k][d]:
                    DP_Sum[k][d] = total_cost
                    RootConfig[k][d] = m

    # --- Find Optimal Result and Reconstruction (Structure unchanged) ---
    min_total_cost = float('inf')
    optimal_depth = -1
    
    for d in range(1, D + 1):
        if DP_Sum[N][d] < min_total_cost:
            min_total_cost = DP_Sum[N][d]
            optimal_depth = d
            
    if optimal_depth == -1:
        return f"No valid tree structure for N={N} found with depth <= {D}.", None

    def reconstruct_tree(current_N, current_D):
        if current_D == 0: return None
        
        optimal_m = RootConfig[current_N][current_D]
        if optimal_m == -1: 
             raise ValueError(f"Tree reconstruction error: Missing optimal m for N={current_N}, D={current_D}.")

        config = FanInConfig[current_N][optimal_m]
        B_layer = BandwidthDP[current_N][optimal_m]
        
        layer_structure = {
            "depth": current_D,
            "inputs": current_N,
            "outputs": optimal_m,
            "bandwidth": B_layer,
            "fan_in_configuration": config
        }
        
        if optimal_m > 1:
             layer_structure["subtree_structure"] = reconstruct_tree(optimal_m, current_D - 1)
        else:
             layer_structure["subtree_structure"] = None

        return layer_structure

    optimal_tree_structure = reconstruct_tree(N, optimal_depth)
    
    return {
        "N_Inputs": N,
        "D_Max_Depth": D,
        "Optimal_Depth": optimal_depth,
        "Min_Total_Bandwidth": min_total_cost,
        "Tree_Structure": optimal_tree_structure
    }, optimal_tree_structure

def display_tree(structure, level=0):
    """Prints the tree structure in a human-readable format."""
    if structure is None:
        return
    indent = "  " * level
    print(f"{indent}**Layer {structure['depth']}** (Inputs: {structure['inputs']}, Outputs: {structure['outputs']}):")
    print(f"{indent}  Bandwidth (Cost): **{structure['bandwidth']:.2f}**")
    config_str = ", ".join(f"{count} nodes with fan-in {n}" 
                           for n, count in sorted(structure['fan_in_configuration'].items()))
    print(f"{indent}  Fan-in Configuration: {config_str}")
    if structure['subtree_structure']:
        print(f"{indent}  -> Outputs ({structure['outputs']}) feed into the subtrees below:")
        display_tree(structure['subtree_structure'], level + 1)

if __name__ == "__main__":
    if len(sys.argv) != 4:
        print("Usage: python optimal_tree_solver.py <N_inputs> <D_max_depth>")
        sys.exit(1)
    
    try:
        N = int(sys.argv[1])
        D = int(sys.argv[2])
        T = int(sys.argv[3])
    except ValueError:
        print("Error: N_inputs and D_max_depth must be valid integers.")
        sys.exit(1)

    # --- Execute and Display ---
    try:
        results, tree_structure = solve_optimal_total_cost_tree(N, D, T)

        if tree_structure:
            print(f"## 🌳 Optimal Total Bandwidth Tree Structure (N={N}, D<={D})")
            print("---")
            print(f"**Total Inputs (N):** {results['N_Inputs']}")
            print(f"**Max Allowed Depth (D):** {results['D_Max_Depth']}")
            print(f"**Optimal Tree Depth:** {results['Optimal_Depth']}")
            print(f"**Minimum Total Bandwidth (Sum):** {results['Min_Total_Bandwidth']:.2f}")
            print("\n### Tree Layer Details (Top-Down: Layer D is the root)")
            display_tree(tree_structure)
        else:
            print(f"Result: {results}")

    except Exception as e:
        print(f"An error occurred during computation: {e}")