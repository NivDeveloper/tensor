# boltzmann — Bose-Einstein condensation of a gluon gas

The thesis numerics (`Numerics.nb`) ported to this library: an overpopulated
gluon gas relaxes under the **full 2→2 Boltzmann collision integral**, sheds
its excess into a condensate, and settles onto a Bose-Einstein distribution.
The port is validated against the live Mathematica kernel at two levels: the
building blocks — grids, kernel matrices (nonzero count exactly), `gas[F0]`,
`cond[F0]`, the time derivative — agree to ~1e-15, and the ENTIRE evolution
to t = 60 (the outer loop mirrors `runEvolution`'s chunking, ~10k adaptive
steps) reproduces the Mathematica trajectory to ~1e-12 in every cell of f,
in n_c, and in the accumulated time itself — the two integrators accept and
reject the same steps.

## The collision term is a contraction

The notebook builds a sparse `M × M³` kernel matrix and contracts it each
step against flattened outer products of the statistical factors
(`b = 1 + f`):

```mathematica
ℏgas[f_] := ℏw (b (Kw . Flatten[Outer[Times, b, f, f]]) -
             f (Kw . Flatten[Outer[Times, f, b, b]]))
```

An earlier version of this example had to replace this with its small-angle
(Fokker-Planck) limit, because `Kw . Flatten[Outer[…]]` needs two things the
library then lacked: rank-raising operands and an index summed across them.
`fold<>` (Expr/Fold.h) provides exactly that — and energy
conservation `E_i + E_j = E_k + E_l` makes the fourth index **affine**
(`l = i+j−k`), which collapses the sparse `M × M³` matrix (99% zeros that
only encode the delta function) into a dense rank-3 kernel:

```cpp
const Grid gas = eval(fold<j, k>(
    W[i, j, k] * hw[i] *
    (b[i] * b[j] * f[k] * f[zero(i + j - k)] -
     f[i] * f[j] * b[k] * b[zero(i + j - k)])));
```

Three library rules carry the physics here:

- **The kernel carries the mask AND the weights** — `W` is zero exactly
  where the notebook's SparseArray had no entry, and the trapezoid
  weights `ω_j ω_k ω_l` multiply into the fill just as `setupNumerics`
  folds them into `Kw`. So the masked terms vanish whatever the off-grid
  `f[…]` read resolves to (`wrap` and `clamp` give the same numbers to
  the last bit), and the mask also carries what a boundary policy cannot:
  the `i ≠ k`, `i ≠ l` exclusions.
- **The summand is the full elementwise grammar**, so gain − loss is ONE
  node: `W` (the large operand) streams once per step, not twice.
- **Free indices broadcast** — `hw[i]` and `b[i]` ride along the output
  index inside the sum.

The condensate channels (`l = i−k` and `l = i+j`, one leg in the
condensate) are rank-2 kernels sharing one `Σ_j`, so both live in one node:

```cpp
const Grid cond = eval(fold<j>(
    hc[i] *
    (M2[i, j] * (b[i] * f[j] * f[zero(i - j - 1_c)] -
                 f[i] * b[j] * b[zero(i - j - 1_c)]) +
     2.0 * M3[i, j] * (b[i] * b[j] * f[zero(i + j + 1_c)] -
                       f[i] * f[j] * b[zero(i + j + 1_c)]))));
```

(The `±1_c` offsets are the 1-based ↔ 0-based shift of the notebook's index
conventions under `E_i = m + iΔE`; a subscript constant must be
compile-time, hence the `_c` spelling.)

## What else maps where

| notebook | here |
| --- | --- |
| `w[i,j,k,l,μ]` — ArcSin/ArcCosh branches + NIntegrate band | plain C++ in the rank-3 index fill (built once) |
| `ℳw2`/`ℳw3` condensate matrix elements | plain C++ in the rank-2 fills |
| validity masks `1 ≤ l ≤ M && i ≠ k && i ≠ l` | zeros in the kernel fills |
| trapezoid weights `ω[j,k,l] = ω_j ω_k ω_l` | `edge_w` factors in the kernel fills, as the notebook folds them into `Kw` |
| `ℱ[f,nc] = ℏgas[f] + nc ℏcond[f]` | `eval(gas + nc * cond)` — the condensate back-reacts on the gas |
| `𝒩[f,nc] = −(nc ΔE/2π²)(Ε Κ).ℏcond[f]` | `-nc/(2π²) * integrate(E * K * cond)` — a fold |
| `RKF23` embedded 2(3) pair | ported line for line; the error norm is `fold<ops::Max>(math::Abs(…))` |
| `Total[…]`, densities, entropy | `fold<ops::Add>` folds of fused expressions |

Parameters are the notebook's own run (`setupNumerics`): M = 32, m = 0.2,
μ = 1.4, Λ = 4.7, α = 0.3, χ = 2 (overpopulated → condenses), weak
screening — the notebook's strong-screening kernels are declared dummy
placeholders, so the weak set is the physics.

## Reading the output

```
       t      n_gas        n_c      total    entropy
0.000000   0.034405   0.000010   0.034415   0.039187
4.003896   0.022716   0.015979   0.038695   0.060323
...
60.020896   0.019736   0.019034   0.038770   0.061138

Bose transform over 32 populated cells:  R² = 0.99996
  T = 0.4702,   μ_chem = 0.1694
number balance over the run: 1.27e-01 (relative)
```

- **The condensate forms**: `n_c` grows from 1e-5 to ~0.019 — half the
  particles end up in the zero mode, and the exchange settles as the gas
  reaches its stationary state.
- **The entropy rises monotonically** and saturates — the notebook's
  headline sanity check.
- **R² = 0.99996**: the Bose transform `ln((1+f)/f)` is straight — the gas
  has genuinely relaxed to Bose-Einstein with temperature 1/slope.
- **The number balance is NOT machine-zero**, and that is faithful: the
  thesis discretization (trapezoid weights + kinematic masks) does not
  conserve particle number exactly — the Mathematica kernel's own `gas[F0]`
  injects ~2.8% of n_gas per unit time at the initial condition, and the
  drift accumulates during the violent early relaxation, then stops once
  near equilibrium (`total` is constant from t ≈ 12 on). The old
  Fokker-Planck version of this example conserved to 1e-15 because its
  finite-volume scheme was built for that; the full kernel trades this for
  the real matrix elements. Run-to-run, the number is a property of the
  scheme, not of this port — the port matches the notebook's derivative to
  fifteen digits.

## Build & run

CPU-only, ~0.5 s at `-O3` — and faster than the notebook it ports: the
same machine runs Mathematica's evolution (MKL sparse dots over
precomputed outer products) in ~5.2 s against ~0.5 s here, with a
per-derivative cost of 173 µs vs 16 µs (the clamped contraction loop
nest — bench/README.md), and the kernel-matrix build drops from 2.2 s to
under a millisecond. The ≈10k-step evolution is three 32³ contractions
per step:

```sh
# after a CMake build (make, from the repo root):
./build/examples/boltzmann/boltzmann
# or directly, no CMake:
g++-16 -std=c++26 -freflection -O3 -I../../include boltzmann.cpp && ./a.out
```

`M`, `χ`, `α` and `mu` are the interesting knobs (all `constexpr` at the
top). The GPU path for contractions exists (`eval(dev, …)` lowers both
kernel shapes since docs/contraction-plan.md stage 2), but this example
stays CPU: at M = 32 a step is microseconds, and the round-trip upload of
the 256 KB kernel per eval would dominate — device residency
(docs/device-residency.md) is the missing piece, not the kernel.
