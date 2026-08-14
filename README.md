# tensor

Header-only C++26 tensor expressions, built on reflection (P2996).

- Fixed-extent tensors, `mdspan`-backed.
- Lazy expression trees that fuse into **one pass** at `eval`.
- Index notation for stencils, contractions and reductions.
- A GPU compute shader generated from an expression's *type*.

Requires GCC 16+ (`-std=c++26 -freflection`).

## Install

Header-only, so pick whichever suits you:

```cmake
add_subdirectory(tensor)              # vendored
find_package(tensor 0.1 REQUIRED)     # installed
```

```cmake
FetchContent_Declare(tensor
    GIT_REPOSITORY https://github.com/NivDeveloper/tensor.git
    GIT_TAG        v0.1.0)
FetchContent_MakeAvailable(tensor)
```

Then one line, which carries the include path, C++26 and `-freflection` with it:

```cmake
target_link_libraries(app PRIVATE tensor::tensor)
```

Without CMake: `g++-16 -std=c++26 -freflection -Ipath/to/include`.

## Quick start

```cpp
#include <Tensor/Tensor.h>
using namespace tensor;
using namespace tensor::indices;

Tensor<float, 3, 4> A([](size_t r, size_t c) { return float(r + c); });
Tensor<float, 4>    x([](size_t k) { return float(k); });

auto y = eval(A[i, j] * 2.0f);   // fused elementwise → Tensor<float, 3, 4>
```

`eval` is the only way to evaluate. It always returns a new owning tensor —
`Tensor t = expr;` and `t = expr;` are deliberately deleted, each with a
message telling you to call `eval`.

## Constructing data

Everything in this section and the next needs `#include <Tensor/Gen.h>`.

```cpp
Tensor<float, 2, 3> A{1, 2, 3, 4, 5, 6};        // literal, row-major
Tensor<float, 4> B([](size_t k) { return k * k; });  // from the coordinates

auto x = eval(linspace<128>(0.0f, 1.0f));       // both endpoints exact
auto z = eval(fill<64, 64>(0.0f));
auto n = eval(iota<10>(1));                     // 1 … 10
```

`fill`, `iota` and `linspace` are **expressions**, not tensors, so they fuse
— `eval(math::Sin(linspace<64>(0.0f, tau)))` is one pass storing no ramp, and
on the GPU a generator costs a push constant or two instead of a buffer and
an upload.

## Random

Counter-based, so a sample is a pure function of its coordinate: identical at
any thread count, and `uniform` is bit-exact between CPU and GPU.

```cpp
seed(2026);                                     // optional — random otherwise
auto n = eval(mu + sigma * normal<float, 1024>());   // mu, sigma may be tensors
auto u = eval(a + uniform<float>());            // shape from a
auto t = eval(exponential<1024>(rate));         // and 8 more distributions
```

Anything needing rejection is a plain function taking an `Rng`:

```cpp
float gamma(Rng &r, float a) { for (;;) { /* … r.uniform() … */ } }
auto g = eval(sample<gamma, 1024>(2.5f));
```

Each cell draws from its own stream, so a rejection loop taking five draws in
one cell and two in the next still gives bit-identical results on any number
of threads. `sample<f>` is CPU-only; the named distributions run on the GPU.

## Index notation

Subscript with placeholders and an expression becomes a function of its free
indices. Those indices *are* its axes.

```cpp
auto outer = eval(x[i] * x[j]);          // outer product → 4x4
auto bcast = eval(A[i, j] / x[j]);       // broadcast along a shared index
auto mv    = eval(fold<j>(A[i, j] * x[j]));   // contraction → [14, 20, 26]
float norm = eval(math::Sqrt(fold(x * x)));   // full reduction → a value
```

A rank-0 result gives back the **value**, not a one-element tensor.

## Boundaries

Any read that can leave the grid must name what happens — there is no silent
default:

| spelling | at the edge |
|---|---|
| `u[wrap(i + 1_c)]` | periodic |
| `u[clamp(i + 1_c)]` | holds the edge value |
| `u[zero(i + 1_c)]` | reads zero |
| `u[pad(i + 1_c, v)]` | reads `v` |

One diffusion step, periodic:

```cpp
Tensor<float, 8> u([](size_t k) { return k == 4 ? 1.0f : 0.0f; });
auto step = eval(u[i] + 0.25f * (u[wrap(i - 1_c)] + u[wrap(i + 1_c)] - 2.0f * u[i]));
// 0.000  0.000  0.000  0.250  0.500  0.250  0.000  0.000
```

The `_c` suffix makes an offset a compile-time constant, which is what lets
the index map live in the type.

## Any function

`map<f>` lifts any plain function — any arity, no registration, scalars
broadcast:

```cpp
float my_fn(float a, float b) { return a * b + 1.0f; }
auto d = eval(map<my_fn>(A, A));
```

The `<cmath>` set is already there as `math::Sqrt(t)`, `math::Exp(t)`, and
~60 more — the same spelling works on scalars.

## Introspection

Expressions are introspectable from the **type alone**, no object needed:

```cpp
using E = decltype(A * 2.0f);
formula<E>();      // "(in0[i] * s0)"
gpu_source<E>();   // a complete Slang compute shader   (#include <Tensor/Gpu.h>)
```

## GPU

Opt in at configure time; the CPU path never sees it.

```cmake
cmake -B build -DTENSOR_ENABLE_GPU=ON -DCMAKE_CXX_COMPILER=g++-16
target_link_libraries(app PRIVATE tensor::gpu gpud::auto_)
```

```cpp
#include <Tensor/Gpu.h>
auto r = eval(dev, A[i, j] * 2.0f);   // same expression, on the GPU
```

The kernel is generated from the expression's type and compiled once per
type. Your application opens and owns the device; tensor only borrows it.
Needs a Vulkan ≥ 1.2 driver to run, nothing extra to build.

## Errors

Mistakes fail at compile time with a sentence, not a template dump:

```cpp
eval(u + u[wrap(i + 1_c)]);
// error: a plain tensor cannot meet an indexed read in one node —
// positional alignment would be ambiguous; subscript every tensor operand
```

## More

[`examples/`](examples/) — standalone and individually documented:
`basic` (the tour), `pendulum`, `mlp`, `boltzmann`, `gpu`.

Build with at least `-O1`: expression templates rely on the inliner, and
`-O0` is dramatically slower.

## License

MIT — see [LICENSE](LICENSE).
