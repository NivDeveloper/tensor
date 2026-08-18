# basic — the library in one sitting

A single `main` that walks the whole public surface, in the order you
would meet it as a user:

1. **Fill a tensor** — `Tensor<float, 4, 3>` is fixed-shape, mdspan-backed.
   Either hand it an index function —
   `Tensor<float, 4, 3> a([](size_t i, size_t j) { … });`, called once per
   cell with that cell's coordinates, one index per axis — or write the
   elements with `a[i, j] = …` (C++23 multidimensional subscript).
2. **Lazy expressions** — `auto c = eval(a + b * 2.0f);` builds a tree
   that computes nothing until `eval`, which then runs one fused
   elementwise pass. No temporaries, one loop.
3. **`map<f>`** — `eval(map<mix>(a, b, 0.5f) - a)` lifts the plain
   function `mix` into the expression with zero registration; scalars
   broadcast, and mapped nodes compose with operators freely.
4. **Index notation** — subscript an operand with placeholders and the
   tree is a function of its free indices: `eval(v[i] * v[j])` is the
   outer product, `eval(fold<j>(a[i, j] * v[j]))` a matrix-vector
   product, `eval(math::Sqrt(fold(v * v)))` a norm — a rank-0 result
   comes back as the **value**. A fold over a plain operand names axis
   numbers (`fold<ops::Max, 0>(a)`) rather than placeholders; spelling
   that one with placeholders is a lint. A fold op may also carry its own
   accumulator instead of reducing in the element type, which is how one
   pass answers two questions: `fold<ops::MinMax>` keeps both ends,
   `fold<ops::ArgMax>` a value and where it was, `ops::Welford` a mean and
   a variance. `Stats.h` names the common ones.
5. **Gather and scatter** — a subscript may be data (`v[clamp(cell[i])]`)
   and `scatter<i>(clamp<3>(cell[i]), load[i])` is its write-side twin,
   depositing into a cell chosen by data. Both must name what happens off
   the end, because a runtime coordinate is never in range by
   construction.
6. **`eval`** — `auto m = eval((a - b) / 4.0f);` materializes into a
   freshly *deduced* owning `Tensor<float, 4, 3>`. `eval` is the **only**
   way to evaluate — a tensor is neither constructible nor assignable
   from an expression, and both spellings are a compile error naming
   `eval`.
7. **Introspection from the type alone** — `formula<E>()` renders the
   expression as a string and `slots_of<E>()` counts its leaves, both
   computed at compile time from `decltype(a + b * 2.0f)`; no objects
   involved.
8. **Runtime traversal** — `for_each_leaf` visits the same tree at the
   value level, in the same DFS order the introspection numbered it.
9. **GPU emission** — `gpu_source<E>()` prints a complete, compilable
   Slang compute shader derived from the same expression type
   (`<Tensor/Gpu.h>` is the opt-in include; nothing runs on a GPU here).

## Build & run

```sh
# after a CMake build:
./build/examples/basic/basic
# or directly, no CMake:
g++-16 -std=c++26 -freflection -I../../include basic.cpp && ./a.out
```

Reading the output next to the source: the formula string, the leaf walk
and the shader's `pc.in0 / pc.in1 / pc.s0` all number leaves in the same
order — one tree, three renderings.
