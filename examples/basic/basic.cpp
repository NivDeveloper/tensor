// A quick tour of the tensor library. Build with CMake (the `basic` target)
// or directly:  g++-16 -std=c++26 -freflection -I../../include basic.cpp

#include <Tensor/Gen.h>  // fill/iota/linspace + sampling
#include <Tensor/Gpu.h>  // opt-in: GPU program construction
#include <Tensor/Math.h> // the elementwise math vocabulary
#include <Tensor/Tensor.h>

#include <iostream>

using namespace tensor;

// Any plain function drops into an expression via map<f> — no registration.
constexpr float mix(float x, float y, float t) { return x + (y - x) * t; }

int main() {
    // Two ways to fill a tensor: an index function — called once per cell
    // with that cell's coordinates, one index per axis — or element access.
    Tensor<float, 4, 3> a([](size_t i, size_t j) { return float(i * 3 + j); });
    Tensor<float, 4, 3> b;
    for (size_t r = 0; r < 4; ++r)
        for (size_t s = 0; s < 3; ++s)
            b[r, s] = float(10 + r);

    // A literal lists every element, row-major; the generators are lazy
    // LEAVES, so a ramp costs no storage and fuses into whatever uses it.
    Tensor<float, 2, 3> lit{1, 2, 3, 4, 5, 6};
    auto ramp = eval(gen::LinSpace<5>(0.0f, 1.0f)); // 0, .25, .5, .75, 1 exactly
    std::cout << "literal[1,2]  = " << (lit[1, 2]) << '\n';
    std::cout << "ramp ends     = " << ramp[0] << ", " << ramp[4] << '\n';

    // Sampling is counter-based: a value is a pure function of its cell, so
    // it is the same at any thread count and on the GPU. Random by default;
    // rng::Seed() pins a run.
    rng::Seed(2026);
    auto noise = eval(a + 0.5f * rng::Normal<float, 4, 3>());
    std::cout << "a + noise     = " << (noise[0, 0]) << '\n';

    // Lazy expressions: nothing computes until eval() — then one fused
    // elementwise pass. Operators and mapped functions compose freely, and
    // eval() materializes into a freshly deduced owning Tensor<float, 4, 3>.
    auto c = eval(a + b * 2.0f);
    auto blended = eval(map<mix>(a, b, 0.5f) - a);
    std::cout << "c[0, 0]       = " << (c[0, 0]) << '\n';
    std::cout << "blended[0, 0] = " << (blended[0, 0]) << '\n';

    // Subscript an operand with placeholders and the tree becomes a FUNCTION
    // OF ITS FREE INDICES — that one rule covers broadcast, outer products,
    // transposes, diagonals and stencils. `fold` is the only consumer of an
    // index, and a rank-0 result is the value, not a one-element tensor.
    using namespace tensor::indices;
    Tensor<float, 3> v{1, 2, 3};
    auto outer = eval(v[i] * v[j]);            // [3,3]
    auto av = eval(fold<j>(a[i, j] * v[j]));   // [4], matrix times vector
    float len = eval(math::Sqrt(fold(v * v))); // rank 0 → the VALUE

    // A fold over a PLAIN operand names axis numbers instead — and it has to:
    // spelling this one with placeholders is a lint, because every operand
    // would read the same bare indices and the subscripts say nothing.
    auto colmax = eval(fold<ops::Max, 0>(a)); // [3], down each column
    std::cout << "outer[2,0]    = " << (outer[2, 0]) << '\n'
              << "colmax[1]     = " << colmax[1] << '\n'
              << "(a*v)[0]      = " << av[0] << '\n'
              << "|v|           = " << len << '\n';

    // A subscript may also be DATA — that is a gather — and `scatter` is its
    // write-side twin, depositing into a cell chosen by data. Both must name
    // what happens off the end: a runtime coordinate is never in range by
    // construction, so there is no silent default.
    Tensor<int, 5> cell{2, 0, 2, 1, 9};
    Tensor<float, 5> load{1, 2, 3, 4, 5};
    auto picked = eval(v[clamp(cell[i])]);                      // [5]
    auto binned = eval(scatter<i>(clamp<3>(cell[i]), load[i])); // [3]
    std::cout << "gathered[4]   = " << picked[4] << "  (cell 9 clamped to 2)\n"
              << "scattered     = " << binned[0] << ", " << binned[1] << ", "
              << binned[2] << '\n';

    // Compile-time introspection, derived purely from the expression TYPE.
    using E = decltype(a + b * 2.0f);
    std::cout << "formula      = " << formula<E>() << '\n';
    std::cout << "leaves       = " << slots_of<E>().views << " tensors, "
              << slots_of<E>().scalars << " scalars\n";

    // Runtime traversal of the same tree, same DFS order.
    std::cout << "leaf values  =";
    for_each_leaf(a + b * 2.0f, [](const auto &leaf) {
        if constexpr (std::is_arithmetic_v<std::remove_cvref_t<decltype(leaf)>>)
            std::cout << " scalar(" << leaf << ")";
        else
            std::cout << " tensor@" << static_cast<const void *>(leaf.data);
    });
    std::cout << '\n';

    // And the same type again, lowered to a complete Slang compute shader.
    std::cout << "\n" << gpu_source<E>();
}
