def main():
    test_sharing()
    # test_gr_inv()
    # test_gr_prod()
    # test_lagrange()
    # compute_lagrange_coeffs()
    # GR_symbolic_inverse_formula()
    print("done")

########################################

def test_sharing():
    # degree of interpolating polynomials
    k = 2

    # Exceptional sequence in Galois ring
    # First point in list is index of shared sequence
    base_points = (
        GR.zero(),
        GR.one(),
        GR((0, 1, 0, 0)),
        GR((1, 1, 0, 0)),
        GR((1, 0, 1, 0)),
        GR((0, 1, 1, 0)),
    )

    # Random vectors to compute inner products over
    Coeffs = GR.coeff_ring
    N = 10*GR.extension_degree # total length 40
    u = list(Coeffs.random_element() for _ in range(N))
    v = list(Coeffs.random_element() for _ in range(N))

    plaintext_ip = sum(ui*vi for ui, vi in zip(u, v))

    ### Protocol 11, Step 1 ###

    # Translate vectors to corresponding Galois ring elements in monomial basis
    u_monom = encode_vec_symmetric(u)
    v_monom = encode_vec_symmetric(v)

    # Set up parties of MPC
    party_base_points = base_points[1:]
    parties = tuple(InnerProductParty(x) for x in party_base_points)

    ### Protocol 11, Step 2 ###

    # Share Galois ring elements to parties
    for ui in u_monom:
        ui_shares = share_gr_elt(ui, k+1, base_points)
        for p, sh in zip(parties, ui_shares):
            p.u_shares.append(sh)
    for vi in v_monom:
        vi_shares = share_gr_elt(vi, k+1, base_points)
        for p, sh in zip(parties, vi_shares):
            p.v_shares.append(sh)

    # Print all parties and shares
    for p in parties:
        party_idx = party_base_points.index(p.base_point)
        print(f"Party {party_idx+1}")
        print(f"Eval point ({p.base_point})")
        print("U shares:")
        for sh in p.u_shares:
            print(f"  {sh}")
        print("V shares:")
        for sh in p.v_shares:
            print(f"  {sh}")
        print()

    ### Protocol 11, Steps 3-6 ###

    # Parties compute shares of inner product
    ip_shares = tuple(
        p.compute_inner_product_share(party_base_points) for p in parties
    )
    print("Additive IP shares:", *ip_shares)

    # Sum (additive) inner product shares to compute final inner product
    mpc_ip = sum(ip_shares)

    print(f"Shares sum to plaintext inner product: {mpc_ip == plaintext_ip}")
    assert(mpc_ip == plaintext_ip)

def encode_vec_symmetric(u):
    """
    Encode a list of values as Galois ring elements batching into blocks and
    interpreting the block as a vector in terms of the precomputed symmetric
    encoding basis.
    """
    assert(GR.extension_degree == 4)
    return list(map(
        lambda u_block: change_of_basis_A_to_monom*GR(u_block),
        batched(u, 4)
    ))

def share_gr_elt(elt, deg, base_points):
    """
    Produce secret shares of degree `deg` sharing of elt over `base_points`.

    Takes place in the monomial basis.
    """
    vals = (elt,) + tuple(GR.random_element() for _ in range(1, deg))
    shares = tuple(legrange_interp(base_points[:deg], p, vals) for p in base_points[1:])
    return shares

class InnerProductParty:
    """
    Class representing the data and operation of an MPC party for the Galois
    ring inner product protocol.
    """

    def __init__(self, base_point):
        self.base_point = base_point
        self.u_shares = list()
        self.v_shares = list()

    def compute_inner_product_share(self, base_points):
        ### Protocol 11, Step 3
        base_point_idx = base_points.index(self.base_point)
        shamir_coeff = legrange_coeffs(base_points, GR.zero())[base_point_idx]

        ### Protocol 11, Step 4
        u_premult = (GR.prod_monom(shamir_coeff, val) for val in self.u_shares)
        u_premult_B = (change_of_basis_monom_to_B * ui for ui in u_premult)

        ### Protocol 11, Step 5
        products = list(val_u.dot_product(val_v) for val_u, val_v in zip(u_premult_B, self.v_shares))

        ### Protocol 11, Step 6
        output_share = sum(products)

        return output_share

########################################

class GaloisRing4:
    extension_degree = 4

    def __init__(self, modulus_bits):
        self.extension_degree = GaloisRing4.extension_degree
        self.modulus_bits = modulus_bits
        self.coeff_ring = IntegerModRing(2**modulus_bits)
        self.matrix_space = MatrixSpace(self.coeff_ring, self.extension_degree)
        self.element_space = self.matrix_space.column_space()

    def __call__(self, vec):
        return self.element_space(vec)

    def prod_monom(self, a, b):
        """
        Return the product of two Galois ring elements represented in the
        monomial basis.
        """
        a0, a1, a2, a3 = a
        b0, b1, b2, b3 = b

        # multiplication formula derived directly from ring definition
        # monomial basis to monomial basis
        return self.element_space((
            a3*b1 + a2*b2 + a1*b3 + a0*b0,
            a3*b2 + a2*b3 + a3*b1 + a2*b2 + a1*b3 + a1*b0 + a0*b1,
            a3*b3 + a3*b2 + a2*b3 + a2*b0 + a1*b1 + a0*b2,
            a3*b3 + a3*b0 + a2*b1 + a1*b2 + a0*b3,
        ))

    def inv_monom(self, a):
        """
        Return the inverse of a Galois ring element represented in the
        monomial basis.
        """
        if (all(ai % 2 == 0 for ai in a)):
            raise ZeroDivisionError("inverse of %s does not exist" % a)

        a0, a1, a2, a3 = a

        # see: GR_symbolic_inverse_formula() for symbolic computation of these formulas
        b0 = (a0^3 + a1^3 - a0*a2^2 + a2^3 + (3*a0 - 2*a1 + a2)*a3^2 + a3^3 - (3*a0*a1 - a1^2)*a2 + (3*a0^2 - 2*a0*a1 - 3*a1*a2 + a2^2)*a3)/(a0^4 + a0*a1^3 - a1^4 - 2*a0^2*a2^2 + (a0 - a1)*a2^3 + a2^4 + (a0 - a1 + a2)*a3^3 - a3^4 + (3*a0^2 - 5*a0*a1 + 2*a1^2 + 4*a0*a2)*a3^2 - (3*a0^2*a1 - 4*a0*a1^2)*a2 + (3*a0^3 - 4*a0^2*a1 + (a0 - 4*a1)*a2^2 - 3*(a0*a1 - a1^2)*a2)*a3)
        b1 = -(a0^2*a1 - (a0 - a1)*a2^2 + a3^3 + (a0*a1 - a1^2 - 2*a0*a2)*a3)/(a0^4 + a0*a1^3 - a1^4 - 2*a0^2*a2^2 + (a0 - a1)*a2^3 + a2^4 + (a0 - a1 + a2)*a3^3 - a3^4 + (3*a0^2 - 5*a0*a1 + 2*a1^2 + 4*a0*a2)*a3^2 - (3*a0^2*a1 - 4*a0*a1^2)*a2 + (3*a0^3 - 4*a0^2*a1 + (a0 - 4*a1)*a2^2 - 3*(a0*a1 - a1^2)*a2)*a3)
        b2 = (a0*a1^2 - a0^2*a2 + a2^3 - (a0 + 2*a1)*a2*a3 + a0*a3^2 + a3^3)/(a0^4 + a0*a1^3 - a1^4 - 2*a0^2*a2^2 + (a0 - a1)*a2^3 + a2^4 + (a0 - a1 + a2)*a3^3 - a3^4 + (3*a0^2 - 5*a0*a1 + 2*a1^2 + 4*a0*a2)*a3^2 - (3*a0^2*a1 - 4*a0*a1^2)*a2 + (3*a0^3 - 4*a0^2*a1 + (a0 - 4*a1)*a2^2 - 3*(a0*a1 - a1^2)*a2)*a3)
        b3 = -(a1^3 - 2*a0*a1*a2 + a2^3 + (2*a0 - a1)*a3^2 + a3^3 + (a0^2 - 3*a1*a2 + a2^2)*a3)/(a0^4 + a0*a1^3 - a1^4 - 2*a0^2*a2^2 + (a0 - a1)*a2^3 + a2^4 + (a0 - a1 + a2)*a3^3 - a3^4 + (3*a0^2 - 5*a0*a1 + 2*a1^2 + 4*a0*a2)*a3^2 - (3*a0^2*a1 - 4*a0*a1^2)*a2 + (3*a0^3 - 4*a0^2*a1 + (a0 - 4*a1)*a2^2 - 3*(a0*a1 - a1^2)*a2)*a3)

        return self.element_space((b0, b1, b2, b3))

    def zero(self):
        return self.element_space.zero()

    def one(self):
        return self.element_space((1, 0, 0, 0))

    def random_element(self):
        return self.element_space.random_element()

# symmetric encoding matrix for up to 64 bits
M64 = MatrixSpace(Integers(2**64), 4, 4)
S64 = M64([
    [                    1,                    0,                    0,  0 ],
    [  3721680401061044962,                    1,                    0,  0 ],
    [  8289148916157705379,  3107565179763450474,                    1,  0 ],
    [ 11432970333055501229,  7744929693175761652, 17484058852862123699,  1 ],
])

########################################

GR = GaloisRing4(16)

S = GR.matrix_space(S64)
change_of_basis_A_to_monom = S
change_of_basis_monom_to_B = (S*S.transpose()).inverse()

########################################

def legrange_interp(base_points, z, values):
    """
    Compute the value of the interpolating polynomial in GR[x] with
    evaluations `values` at points `base_points`, computed at `z`.  For
    proper functioning, the ring elements in `base_points` should form
    an "exceptional sequence", meaning any pair of distinct elements
    should be invertible in GR.
    """
    coeffs = legrange_coeffs(base_points, z)
    val = GR.zero()
    for coeff, y in zip(coeffs, values):
        val += GR.prod_monom(coeff, y)
    return val

def legrange_coeffs(base_points, z):
    """
    Returns a sequence of ring coefficients to multiply function evaluations
    by to compute the polynomial interpolation at a fixed evaluation point.

    `base_points` is the sequence of distinct points for the interpolation
    `eval` is the evaluation point for the interpolation.
    """
    coeffs = list()
    for i, p in enumerate(base_points):
        coeff = GR.one()
        for j, q in enumerate(base_points):
            if j != i:
                coeff = GR.prod_monom(coeff, z - q)
                coeff = GR.prod_monom(coeff, GR.inv_monom(p - q))
        coeffs.append(coeff)
    return coeffs

def batched(iterable, n):
    # batched('ABCDEFG', 3) --> ABC DEF G
    from itertools import islice
    if n < 1:
        raise ValueError('n must be at least one')
    it = iter(iterable)
    while batch := tuple(islice(it, n)):
        yield batch

########################################

def compute_lagrange_coeffs():
    base_points = (
        GR.one(),
        GR((0, 1, 0, 0)),
        GR((1, 1, 0 ,0)),
    )

    coeffs = legrange_coeffs(base_points, GR.zero())
    print("Base points:")
    for p in base_points:
        print(p)
    print()
    print("Lagrange coefficients for evaluation at 0:")
    for c in coeffs:
        print(c)

def GR_symbolic_inverse_formula():
    """
    Use Sage symbolic ring functionality to compute the formula for
    inversion in the Galois ring GR = (Z/2^kZ)[x] / (x^4 - x - 1).
    """

    a = SR.var('a', 4)
    a0, a1, a2, a3 = a

    b = SR.var('b', 4)
    b0, b1, b2, b3 = b

    eqs = [
        a3*b3 + a3*b0 + a2*b1 + a1*b2 + a0*b3 == 0,
        a3*b3 + a3*b2 + a2*b3 + a2*b0 + a1*b1 + a0*b2 == 0,
        a3*b2 + a2*b3 + a3*b1 + a2*b2 + a1*b3 + a1*b0 + a0*b1 == 0,
        a3*b1 + a2*b2 + a1*b3 + a0*b0 == 1,
    ]

    s, = solve(eqs, b, solution_dict=True)

    print("Symbolic formulas for inverses in Galois ring GR(2^k, 4)...\n")

    for key, value in s.items():
        print(f"{key}:")
        print(value)
        print()

########################################

def test_gr_inv():
    a = GR((1, 1, 0, 0))
    a_inv = GR((2, -1, 1, -1))
    assert(GR.inv_monom(a) == a_inv)

def test_gr_prod():
    a = GR((7, 0, 3, 0))
    b = GR((1, 1, 0, 1))
    ab = GR((7, 10, 6, 10))
    assert(GR.prod_monom(a, b) == ab)

def test_lagrange():
    # degree two polynomial
    x0 = GR.zero()
    y0 = GR((3, 7, 1, 0))

    x1 = GR.one()
    y1 = GR((2, 2, 2, 0))

    x2 = GR((0, 1, 0, 0))
    y2 = GR.zero()

    x3 = GR((1, 1, 0, 0))
    y3 = legrange_interp((x0, x1, x2), x3, (y0, y1, y2))

    x4 = GR((1, 0, 1, 0))
    y4 = legrange_interp((x0, x1, x2), x4, (y0, y1, y2))

    x5 = GR((0, 1, 1, 0))
    y5 = legrange_interp((x0, x1, x2), x5, (y0, y1, y2))

    z0 = legrange_interp((x1, x2, x3), x0, (y1, y2, y3))
    assert(y0 == z0)

    z4 = legrange_interp((x0, x1, x3), x4, (y0, y1, y3))
    assert(y4 == z4)

    z5 = legrange_interp((x0, x1, x2, x3, x4), x5, (y0, y1, y2, y3, y4))
    assert(y5 == z5)

if __name__ == '__main__':
    sys.exit(main())
