#!/usr/bin/env python3
import math

# ---------------------------------------------------------
# Finite-Resolution Locking of the Fine-Structure Constant
#
# Deterministic finite computation.
# No empirical fitting. No Monte Carlo sampling.
# All quantities are computed from finite sums and fixed constants.
#
# Running this file prints a single number: alpha^{-1}.
# ---------------------------------------------------------

def walk_counts_2d(m):
    # Integer counts for the 2D nearest-neighbor random walk (total weight = 4^m).
    counts = {(0, 0): 1}
    for _ in range(m):
        nxt = {}
        for (x, y), c in counts.items():
            nxt[(x + 1, y)] = nxt.get((x + 1, y), 0) + c
            nxt[(x - 1, y)] = nxt.get((x - 1, y), 0) + c
            nxt[(x, y + 1)] = nxt.get((x, y + 1), 0) + c
            nxt[(x, y - 1)] = nxt.get((x, y - 1), 0) + c
        counts = nxt
    return counts

def E_R_discrete(m):
    # Finite-sum evaluation of E[R_m] for the discrete walk.
    counts = walk_counts_2d(m)
    denom = 4 ** m
    s = 0.0
    for (x, y), c in counts.items():
        s += (c / denom) * math.sqrt(x * x + y * y)
    return s

def E_R_gaussian(m):
    # Continuum reference with Var = m/2: E[R~] = sqrt(m*pi)/2.
    return math.sqrt(m * math.pi) / 2.0

def P_full(m, k):
    # Exact coupon-collector full-coverage probability (finite sum).
    total = 0.0
    for j in range(k + 1):
        # Compute C(k,j) without external libraries.
        cj = 1
        for t in range(1, j + 1):
            cj = cj * (k - (t - 1)) // t
        total += ((-1.0) ** j) * cj * ((k - j) / k) ** m
    return total

def m_star(k, threshold=0.5, m_max=500):
    # Finite-resolution sweep depth: first m reaching the 1-bit boundary.
    for m in range(1, m_max + 1):
        if P_full(m, k) >= threshold:
            return m
    raise RuntimeError("m* not found")

def alpha0(n, Nu):
    # Principal finite-resolution term (no adjustable parameters).
    return (2 * n + 1) ** 2 / (8.0 * math.pi * (Nu ** 2) * (math.log(2.0) ** 2))

def main():
    # Core discrete inputs of the construction (theoretical, not tuning knobs).
    d = 3        # spatial dimension
    k = 2 * d    # number of reversible update directions
    Nu = 2 ** k  # total update channels
    n = d * d    # structural dimension (local endomorphisms)
    m = m_star(k, threshold=0.5)  # finite-resolution sweep depth

    # Mandatory finite-resolution residual (geometric correction).
    c_free = E_R_discrete(m) / E_R_gaussian(m)
    Delta_m = 1.0 / (12.0 * m * m)
    c_eff = c_free * math.sqrt(1.0 - Delta_m)

    # Assembly.
    K = 1.0 / (2.0 * (math.pi ** 2.5))
    a0 = alpha0(n, Nu)
    alpha_inv = (1.0 / a0) + K * c_eff

    # Reduced significant digits for compact output.
    print(f"{alpha_inv:.12f}")

if __name__ == "__main__":
    main()
