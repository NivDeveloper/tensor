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
4. **`eval`** — `auto m = eval((a - b) / 4.0f);` materializes into a
   freshly *deduced* owning `Tensor<float, 4, 3>`. `eval` is the **only**
   way to evaluate — a tensor is neither constructible nor assignable
   from an expression, and both spellings are a compile error naming
   `eval`.
5. **Introspection from the type alone** — `formula<E>()` renders the
   expression as a string and `slots_of<E>()` counts its leaves, both
   computed at compile time from `decltype(a + b * 2.0f)`; no objects
   involved.
6. **Runtime traversal** — `for_each_leaf` visits the same tree at the
   value level, in the same DFS order the introspection numbered it.
7. **GPU emission** — `gpu_source<E>()` prints a complete, compilable
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
