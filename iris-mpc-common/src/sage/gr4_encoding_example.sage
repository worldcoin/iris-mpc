### COMPUTED BASIS MATRICES ###

# 16-bit encoding
M16 = MatrixSpace(Integers(2**16), 4, 4)
V16 = M16.column_space()
S16 = M16([
    [     1,     0,     0,     0 ],
    [ 58082,     1,     0,     0 ],
    [ 60579, 25194,     1,     0 ],
    [ 17325, 51956, 57011,     1 ],
])
# TODO this one is wrong for the new matrices...
_lin_comb_16 = V16([
    1,
    50642,
    57413,
    17471
]) # only for reference, not required explicitly by protocol

# 64-bit encoding; matrix can be used for any lower bit count b by taking the
# entry-wise remainders mod 2^b
M64 = MatrixSpace(Integers(2**64), 4, 4)
V64 = M64.column_space()
S64 = M64([
    [                    1,                    0,                    0,  0 ],
    [  3721680401061044962,                    1,                    0,  0 ],
    [  8289148916157705379,  3107565179763450474,                    1,  0 ],
    [ 11432970333055501229,  7744929693175761652, 17484058852862123699,  1 ],
])
# TODO this one is wrong for the new matrices...
_lin_comb_64 = V64([
    1,
    13808594730259957202,
    12520909974947291205,
    12689163005896508479
]) # only for reference, not required explicitly by protocol
assert(M16(S64) == S16)
assert(V16(_lin_comb_64) == _lin_comb_16)


### DEMONSTRATION ###

# Change-of-basis matrix
S = S16

# Dimension 4 vector space (free module actually) for Galois ring elements
V = V16

# Specific linear combination appearing "under the hood" in the construction
_lin_comb = _lin_comb_16


# We will be working in the usual Galois ring R = (ZZ/2^mZZ)[x] / (x^4 - x - 1)

def prod_R_monom(a, b):
    """
    Return the product of two elements of R, represented as coefficient
    vectors in the monomial basis.
    """
    a0, a1, a2, a3 = a
    b0, b1, b2, b3 = b
    R = a0.base_ring()

    return vector(R, [
        a3*b1 + a2*b2 + a1*b3 + a0*b0,
        a3*b2 + a2*b3 + a3*b1 + a2*b2 + a1*b3 + a1*b0 + a0*b1,
        a3*b3 + a3*b2 + a2*b3 + a2*b0 + a1*b1 + a0*b2,
        a3*b3 + a3*b0 + a2*b1 + a1*b2 + a0*b3,
    ])


# Three bases of R are involved in our computations. They are:
# 1. The monomial basis
# 2. The basis A with change of basis matrix S to the monomial basis
# 3. The basis B with change of basis matrix S*S.transpose() to the monomial basis

change_of_basis_A_to_monom = S
change_of_basis_monom_to_B = (S*S.transpose()).inverse()
print(change_of_basis_A_to_monom)
print()
print(change_of_basis_A_to_monom.inverse())
print()
print(change_of_basis_monom_to_B)
print()

# - Vectors are first encoded as elements of R represented in basis A
# - For vectors so encoded, their dot product can be computed as the
#   first coordinate of their R-product when represented in basis B
# - This first product coordinate can be efficiently computed concretely by
#   representing one element in the monomial basis and the other in basis B,
#   and then taking the dot product of these vector representations

# The following demonstrates several ways of computing the inner product of
# two vectors using elements of R encoded in the same way, in terms of basis
# A, by applying appropriate basis changes.

# Initial vectors in basis A
u_A = V([1,1,1,1])
v_A = V.random_element()

# 1. Plaintext dot product of input vectors
ip1 = u_A.dot_product(v_A)

# 2. Specific linear combination of product coefficients in the monomial basis
u_monom = change_of_basis_A_to_monom * u_A
print()
print(u_A)
print(u_monom)
print()
v_monom = change_of_basis_A_to_monom * v_A
uv_monom = prod_R_monom(u_monom, v_monom)
ip2 = _lin_comb.dot_product(uv_monom)
# TODO fix the linear combinations to make this work
# assert(ip2 == ip1)

# 3. First coordinate of GR4 product in basis B
uv_B = change_of_basis_monom_to_B * uv_monom
ip3 = uv_B[0]
assert(ip3 == ip1)

# 4. First coordinate of GR4 product in basis B, computed as the dot product
# of vectors represented in the monomial basis and in basis B, respectively
v_B = change_of_basis_monom_to_B * v_monom
ip4 = u_monom.dot_product(v_B)
assert(ip4 == ip1)

print("All assertions passed.")


# For secret sharing, one should be able to convert the ring elements to the
# monomial basis to produce the secret shares for different parties as usual,
# and then "query" shares can remain in the monomial basis, while "database"
# shares are converted to basis B.  Then the dot product of query and
# database shares gives the analog of the "constant coefficient in the
# monomial basis" used in the original protocol, but here the MPC parties are
# able to produce the database shares themselves since it is accomplished by
# just a change of basis from the query shares.
