# Examples

One directory per example. Each is **standalone**: a single program in
`<name>/<name>.cpp` with its own `README.md` explaining what it
demonstrates, and a two-line `CMakeLists.txt` — no shared code between
examples, no shared build magic.

| example | shows |
| --- | --- |
| [`basic/`](basic/) | the library in one sitting: lazy expressions, `map<f>`, the index model (folds, gather, scatter), `eval`, compile-time introspection, GPU shader emission |
| [`histogram/`](histogram/) | a histogram start to finish: `rng::` data, range via two folds, `bins` + `drop` + `scatter` as the one-line binning step, storage-free bin centres, and `npy::savez` out to matplotlib |
| [`pendulum/`](pendulum/) | explicit time-stepping of a pendulum ensemble (symplectic Euler): fixed-shape state tensors, `math::Sin` as the nonlinearity, an `eval`-driven integrator loop, and compile-time formula/GPU introspection |
| [`boltzmann/`](boltzmann/) | Bose-Einstein condensation of a gluon gas: index-fill grids, the 2→2 collision term as one `fold<j,k>` whose off-grid reads are zero per read, an `eval`-driven RKF23 integrator, and a Bose fit whose normal equations are folds over a 0/1 weight — entropy monotone, relaxation to Bose-Einstein measured (R² ≈ 0.99996), number balanced to the trapezoid rule |
| [`mlp/`](mlp/) | a two-layer perceptron trained by hand-written backprop: with a batch axis every layer and every gradient is ONE fused `fold<>` (matmul + bias + activation in a single eval), activations are `math::Tanh`, losses are folds, the target itself is a contraction — MSE drops five orders, train R² = 1.0000 |
| [`bgk/`](bgk/) | test-particle Boltzmann relaxation in 3-D: two discs collide off-axis and thermalize into one Maxwellian. Each particle's cell is ONE number, so every cell moment is a `scatter` at it and every read-back a `gather` by it — no one-hot anywhere; `rng::` sampling for the collisions, momentum and energy conserved to 1e-6, and the whole loop also runs on the GPU |
| [`gpu/`](gpu/) | `eval(dev, expr)` end to end (needs `-DTENSOR_ENABLE_GPU=ON`): opening a device, heat diffusion via a decorated-subscript Laplacian stencil with one memoized kernel, closed-form + CPU cross-checks, the executed Slang source |

## Building

Examples build as part of the tensor tree — they are on by default in a
top-level build (`TENSOR_BUILD_EXAMPLES=ON`) and need nothing beyond the
library's own requirement, GCC 16 with `-freflection`:

```sh
make                 # or: cmake -B build -DCMAKE_CXX_COMPILER=g++-16
                     #     cmake --build build
```

Each example lands next to its source directory in the build tree:

```sh
./build/examples/<name>/<name>       # e.g. ./build/examples/basic/basic
```

Every example also states a direct, CMake-free compile line in its
top-of-file comment — they are single files on purpose.

## Adding an example

1. `examples/<name>/` with three files:
   - `<name>.cpp` — one standalone program (put a direct compile line in
     the header comment);
   - `README.md` — what it demonstrates and what to look for in the
     output;
   - `CMakeLists.txt` — `add_executable(<name> <name>.cpp)` +
     `target_link_libraries(<name> PRIVATE tensor::tensor)`.
2. Register it in `examples/CMakeLists.txt` with one
   `add_subdirectory(<name>)` line.

An example with extra requirements keeps its gating and extra
dependencies inside its own `CMakeLists.txt` — the registration line in
the parent is always unconditional. [`gpu/`](gpu/) is the live instance:
it returns early unless `tensor::gpu` exists (a CPU-only build just
skips it) and links `tensor::gpu` + `gpud::auto_` itself, so it lands in
the GPU tree only: `make GPU=1` → `./build-gpu/examples/gpu/gpu`.
