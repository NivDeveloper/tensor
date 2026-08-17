# bgk — test-particle Boltzmann relaxation in 3-D

Two discs of test particles fly at each other, collide off-axis, and
thermalize into a single Maxwellian. Positions and momenta are both
three-component — the particles move in 3-D and the cells are a C³ grid.
A step is free streaming, then per spatial cell: measure the equilibrium
parameters, histogram the distribution, relax that histogram toward the
Maxwellian, resample momenta from it, and shift-and-scale the samples so
the collision invariants survive. No hand-managed buffer, no one-hot,
no hand-written kernel.

```
vx distribution, 8192 test particles in 4x4x4 cells (the histogram is
illustrative; the program itself prints the four lines below it)

        before                                          after
-1.000  #############################################     #########
-0.867  ##############################################    ########
-0.733  ################################                  ###########
-0.600  ###########                                       ################
-0.467  #                                                 ####################
-0.333                                                    ########################
-0.200                                                    #############################
-0.067                                                    ##############################
 0.067                                                    ############################
 0.200                                                    #########################
 0.333  #                                                 ######################
 0.467  #########                                         #################
 0.600  ###############################                   ##########
 0.733  ################################################  ########
 0.867  #############################################     #########

T     0.2341
mu    2.0947
sigma 0.4869   (equipartition 0.4869)
drift  momentum -1.24e-07   energy 9.22e-07
```

## The idea: every cell quantity is a deposit

Every kinetic particle code needs to bin particles into cells, reduce
within each cell, and read the result back onto each particle. The usual
implementation is a scatter with atomics — one compute kernel to accumulate
sums, another to finalize, a third to gather.

x, y and z are an **axis, not three variables** — positions and momenta are
`[N, 3]`, so free streaming is one line for all three, and so is every cell
moment. Where an axis has to be singled out, a constant subscript does it.

A particle's cell is one number:

```cpp
Cells cells(const Vecs &pos) {
    auto a = go(Fmin(Fmax(Floor((pos + 0.5f) * f32(C)), 0.0f), f32(C - 1)));
    return go((a[i, 0_c] * f32(C) + a[i, 1_c]) * f32(C) + a[i, 2_c]);
}
```

and from there every cell quantity is a deposit at that number:

```cpp
auto count = go(scatter<i>(clamp<CC>(cid[i]), 1.0f));       // [CC]
auto p     = go(scatter<i>(clamp<CC>(cid[i]), mom[i, n]));  // [CC,3] — n survives
auto E     = go(scatter<i>(clamp<CC>(cid[i]), sq[i]));
```

while reading a cell value back onto each particle is the same number as a
subscript:

```cpp
auto next = go(b[clamp(cid[i])] * q[i, n] + shift[clamp(cid[i]), n]);
```

`cid` holds floats and the policies floor them, so there is no cast anywhere.
A three-coordinate grid reads exactly as well — `q[clamp(cx[i]), clamp(cy[i]),
clamp(cz[i]), n]`, and the library is happy either way — but the flat id
leaves one cell index where the code would otherwise carry three, in every
expression downstream. It is also faster than that sounds, because each
scatter loses two destination coordinate buffers and each gather two
subscript computations (best of three runs):

| | 3 coords, CPU | flat, CPU | 3 coords, GPU | flat, GPU |
| --- | ---: | ---: | ---: | ---: |
| N=8192, C=4 | 1.72 s | 1.51 s | 2.81 s | 2.51 s |
| N=32768, C=8 | 0.70 s | 0.57 s | 5.83 s | 4.68 s |
| N=131072, C=8 | 1.60 s | 1.07 s | 20.8 s | 14.2 s |

The gap widens with the particle count on both devices, which is what a
per-contribution cost should do. (The GPU columns predate the sliced float
scatter below and are an order of magnitude slower than the same runs
today; the comparison between the two spellings is what they are here for,
and the three-coordinate variant no longer exists to re-measure.)

**There is no one-hot tensor**: earlier versions carried an `[N, C, C, C]`
occupancy tensor — 2 MB rebuilt every step — purely because the library could
not say "deposit at this particle's cell". Both halves of that are now the
grammar, and the CPU step went from 11.7 s to 1.5 s per thousand steps.

Clamping the axes is not ceremony. `Floor((pos + 0.5f) * C)` produces exactly
`C` whenever `pos + 0.5f` rounds up to `1.0f` in single precision — measured,
**640 of 8192** positions do. Against the one-hot those particles matched no
cell and silently dropped out of the collision, momentum never updated;
folding them into the edge cell took the energy drift from −7.7e−03 to around
1e−06. The `clamp` on every deposit and every read-back is mandatory on top of
that, whatever `cid` happens to hold: a runtime coordinate is never in range
by construction, so the library makes you say what happens off the end.

The momentum histogram deposits at two coordinates — the cell, then the bin —
and is spelt with an **integral** value:

```cpp
auto hist = go(scatter<i>(clamp<CC>(cid[i]), clamp<B>(bin[i, n]), 1u));
```

A count is an integer, and it matters: an integral scatter deposits through
atomics, which have no cell-count bound, while a floating-point one goes
through per-thread groupshared bins and is bounded by them. Arithmetic
promotes downstream, so nothing else changes.

## The physics: the relaxation-time approximation

1. **Free stream** — every component, periodic on [−½, ½).
2. **Equilibrium parameters** — per cell: population *n*, bulk velocity
   **u**, and the fluctuation energy *S* about that bulk flow.
   Equipartition gives `T = S/3`, and for a classical gas the chemical
   potential is fixed by the other two, `n = e^{μ/T}(2πT)^{3/2}`, so
   `μ = T·ln(n / (2πT)^{3/2})`.
3. **Histogram** the peculiar velocity **c** = **p** − **u**, per component
   per cell.
4. **Relax.** The RTA collision operator is `∂f/∂t = −(f − f_eq)/τ`. Over a
   substep `f_eq` is *fixed* — the moments it is built from are exactly
   what the operator conserves — so the linear ODE integrates **exactly**,
   not to first order:

   > `f(t + Δt) = f_eq + (f − f_eq)·e^{−Δt/τ}`

   The relaxed histogram is a convex blend of the measured one and the
   Maxwellian, with weight `α = e^{−Δt/τ}`.
5. **Resample** by inverse transform. The cumulative distribution IS a
   prefix sum, so it is a scan along the bin axis — it used to be a
   contraction against a triangular mask, `B²` multiply-adds for what `B`
   additions give, because the library had no scan:
   ```cpp
   auto cdf = go(scan<ops::Add, m>(relaxed[j, n, m]) * c.inv[j]);
   ```
   The bin index is then *how many CDF entries the draw clears* — which is
   a fold. Reading the CDF back onto each particle happens INSIDE that
   fold, so the `[N, 3, B]` it would otherwise materialize — 288 bytes per
   particle — never exists:
   ```cpp
   const auto hit = go(fold<m>(1.0f * (u1[i, n] > cdf[clamp(cid[i]), n, m])));
   ```
   Spelling it as one expression rather than two is worth 1.4× on both
   devices, and the numbers are unchanged to the last bit.

   The scan is worth another **1.25×** of the whole step at 1M particles on
   a 40³ grid (26.96 → 21.56 ms), also bit for bit. One detail is load
   bearing and easy to lose: `relaxed` is spelt with the Maxwellian term
   FIRST so its free indices come out `j, n, m`, which makes the CDF's bins
   contiguous for the read-back above. Written the other way round the scan
   is a 7% LOSS — the line it feeds is the hottest in the step, and its
   reads go from stride 1 to stride 3.
6. **Shift and scale.** Resampling conserves nothing, so restore both
   invariants with `q' = b(q − q̄) + a`, one shared `b` across the three
   components. Momentum gives `a = u` directly. For energy, writing
   `S = (1/n)Σᵢ Σ_c (qᶜᵢ − q̄ᶜ)²`:

   > `Σᵢ|q'ᵢ|² = b²nS + 2b Σ_c aᶜ Σᵢ(qᶜᵢ − q̄ᶜ) + n|u|²`

   the middle term is identically zero, so matching the target
   `n(S_pre + |u|²)` gives **`b = sqrt(S_pre / S_post)`**. The shared `b` is
   essential: a per-component `b_c` would force each variance back to its
   old value and undo the relaxation.

The final width is the check. With the beams stopped, all the directed
energy has become thermal and shared over three components, so σ should
equal `sqrt(E₀/3N)` — measured 0.4869 against 0.4869.

### Two numerical details that matter

**Compute the fluctuation about the mean, never as `E[|p|²] − |u|²`.** In a
beam cell those are 0.708 and 0.640, so the subtraction throws away most of
the significant digits before anything else happens, and whatever error the
two sums carry arrives in `b` magnified by that cancellation. Folding
`|p − u|²` directly cut the energy drift 7×. A fold accumulates in float64
since ACC-G9 (docs/numerical-contract.md), which is why the drift is now
9.2e-07 rather than the 3.1e-06 this README used to print — but that is the
error term shrinking, not the cancellation going away. It is a property of
the formula, and the formula is still the wrong way to spell it.

**A cell with fewer than two particles has no fluctuation to measure** —
`S_post` collapses, the shift annihilates `q − q̄`, and the cell silently
drops its share of the energy. Those cells skip the collision, the same
guard every DSMC code carries.

## Running on the GPU

Configure with `-DTENSOR_ENABLE_GPU=ON` and the whole step loop runs as
compute shaders — scatters included. The expressions are unchanged; only
where they evaluate differs:

```cpp
auto go(const auto &e) {
    if (device) return eval(*device, e);
    return eval(e);
}
```

Two things stay on the CPU. `map<f>` is CPU-only without the
kernel-translation opt-in, so the lifted helpers became native vocabulary
(the periodic wrap is `v - Floor(v + 0.5f)`); and a rank-0 fold is capped at
4096 elements, so the momentum and energy diagnostics run outside the loop.

Measured ms per step, minimum of three runs of 200 steps, shader
compilation excluded (it is a one-time ~1.8 s, and counting it is what made
an earlier version of this table call the GPU slower than it is):

| | cpu/1 | cpu/8 | gpu | gpu ÷ cpu/8 |
| --- | ---: | ---: | ---: | ---: |
| N=8192, C=4 | 1.87 | 1.53 | **0.63** | 0.41 |
| N=8192, C=8 | 2.16 | 1.86 | **0.61** | 0.33 |
| N=8192, C=16 | 4.64 | 2.10 | **0.74** | 0.35 |
| N=32768, C=8 | 7.59 | 2.80 | **1.63** | 0.58 |
| N=32768, C=16 | 10.07 | 2.97 | **2.72** | 0.92 |
| N=131072, C=8 | 29.26 | 5.03 | 5.49 | 1.09 |
| N=524288, C=8 | 116.48 | 19.03 | 30.66 | 1.61 |

The device wins outright up to ~32768 particles and loses at the largest
sizes. (The table predates two later changes — the fused CDF read-back
above and the host-side Philox multiply — which take the CPU column down
by about 1.6× and the GPU column by 1.4×; the shape of the comparison is
unchanged, and the crossover moves toward the CPU.) Where the GPU loses,
the five float scatters are what it loses on: 73% of the step at
N=524288, C=8.

### What the scatters used to cost

A float scatter cannot use atomics: Slang emits one happily, but it lowers
under a capability the device does not enable. Every accumulator is
therefore owned privately by one thread, and the first implementation put
all of them in ONE workgroup's groupshared — so the thread count was set by
the cell count, `w · cells · 4 ≤ 16 KB`:

| scatter | cells | threads then | now |
| --- | ---: | ---: | ---: |
| `count`, `E` at C=4 | 64 | 64 | 64 × 1 group |
| `p`, `Q` at C=4 | 192 | 16 | 64 × 3 groups |
| `count`, `E` at C=8 | 512 | 8 | 64 × 8 groups |
| `p`, `Q` at C=8 | 1536 | **2** | 64 × 24 groups |
| `p`, `Q` at C=16 | 12288 | **would not compile** | 64 × 192 groups |

Two threads is not a GPU, and 12288 cells had no answer at all — one
thread's bins alone were 49 KB. `scatter_grid` now keeps the full workgroup
and slices the OUTPUT instead: a group owns 64 cells, scans every
contribution for the ones it owns, and merges its 64 private bins. The
redundant scanning is the price of having no float atomic, and parallelism
absorbs it. What that was worth here, same expressions, same device:

| | gpu before | gpu after | |
| --- | ---: | ---: | ---: |
| N=8192, C=8 | 4.28 | 0.61 | 7.1× |
| N=32768, C=8 | 16.34 | 1.63 | 10.0× |
| N=131072, C=8 | 62.70 | 5.49 | 11.4× |
| N=524288, C=8 | 303.63 | 30.66 | 9.9× |
| any C=16 | *would not build* | 0.74–43.6 | — |

Per expression at N=524288, C=8: `p` went from 125.4 ms to 7.9, and the
five scatters together from 310.6 ms (93% of the step) to 26.1.

C=2 and C=4 gained 1.2–3.0× as well, and at C=2 nothing is sliced — that
part is purely hoisting the destination into a local, which stopped the
emitter rendering the whole `clamp(floor(load))` chain twice per
contribution. Design record: `docs/scatter-multigroup-plan.md`. The
remaining case is few cells at large N, where one slice is one workgroup:
chunking the contributions across groups (with a merge pass) is measured at
5.2× on that shape and not yet implemented.

## Against the hand-written version

vklib's `examples/boltzmann3d` is the same method, as five compute shaders
(`FreeStream`, `StatsScatter`, `StatsFinalize`, `Scatter`, `Correct`) over
~220 lines of kernel code, plus explicit buffer allocation, uploads,
dispatch sizing and `ctx.Zero` calls between passes. It uses atomic scatter
for the binning, which is O(N) and scales to a million particles on a 40³
grid.

This version is O(N) in the binning too — a deposit goes straight to the
particle's own cell, so a finer grid costs it nothing. What still scales
with the bin count is the inverse transform: reading the CDF back onto each
particle and searching it are both O(N × B). What the spelling buys is that
the method reads as its own equations, and that one number — the particle's
cell — serves the binning, the reduction and the read-back alike.

Two details the original needs that are omitted here: a random
per-iteration grid offset (with a fixed grid, particles sitting exactly on
cell faces flip between cells and produce visible streaks), and a
Henyey–Greenstein anisotropic scattering angle, where this uses the
isotropic `g = 0` case.

## Running

```sh
./build/examples/bgk/bgk
```

or directly:

```sh
g++-16 -std=c++26 -freflection -O3 -I../../include bgk.cpp && ./a.out
```

Build with optimization — `-O0` is dramatically slower. `N`, `C` and
`steps` at the top of the file are the knobs; a step costs O(N × C³).

## Looking at the result

The four printed lines are aggregates; the distribution is the interesting
object. `Npy.h` writes it where matplotlib can pick it up — two lines in the
example, none of them plotting:

```cpp
#include <Tensor/Npy.h>
// … after the loop: one file per struct field, named after the field
npy::savez("state.npz", State{Pos, Mom});
npy::save("temp.npy", c.T);   // or one tensor on its own
```

```python
import numpy as np, matplotlib.pyplot as plt
d = np.load("state.npz")                        # keys "Pos" and "Mom" — the
mom = d["Mom"]                                  # C++ field names, via reflection
plt.hist(mom[:, 0], bins=80, density=True)      # the relaxed Maxwellian
plt.hist(np.linalg.norm(mom, axis=1), bins=80)  # or the speed distribution
plt.show()
```

The keys, shapes and dtypes all come from the C++ types — `State`'s field
names and `Tensor<f32, N, 3>`'s arguments — so nothing is declared twice on
either side. See `docs/vocabulary.md` under "Sharing data with other tools".
