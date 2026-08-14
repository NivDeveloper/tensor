// A quick tour of the tensor library. Build with CMake (the `basic` target)
// or directly:  g++-16 -std=c++26 -freflection -I../../include basic.cpp

#include <Tensor/Gen.h> // fill/iota/linspace + sampling
#include <Tensor/Gpu.h> // opt-in: GPU program construction
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
    for (size_t i = 0; i < 4; ++i)
        for (size_t j = 0; j < 3; ++j)
            b[i, j] = float(10 + i);

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
