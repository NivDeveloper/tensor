// A histogram, start to finish: sample data, find its range, deposit each
// value into the bin its VALUE picks, and hand the result to numpy. Build
// with CMake (the `histogram` target) or directly:
//   g++-16 -std=c++26 -freflection -I../../include histogram.cpp

#include <Tensor/Gen.h>   // rng:: sampling, gen::Iota for the bin centres
#include <Tensor/Math.h>  // bins — the value→bin map
#include <Tensor/Npy.h>   // the way out, to matplotlib
#include <Tensor/Stats.h> // stats::Histogram — this file, in one line

#include <iostream>
#include <utility>

using namespace tensor;
using namespace tensor::indices;

constexpr size_t N = 4096; // samples
constexpr size_t B = 40;   // bins

// One .npz entry per field, keyed by the FIELD NAME: in Python this is
// np.load("histogram.npz")["Counts"] with nothing declared twice.
struct Histogram {
    Tensor<unsigned, B> Counts;
    Tensor<float, B> Centres;
};

int main() {
    rng::Seed(2026); // pin the run; drop this line for fresh draws

    // The data: a product of two normals — sharply peaked, heavy-tailed,
    // and NOT a distribution the library names, which is the point of
    // histogramming it.
    auto x = eval(rng::Normal<float, N>() * rng::Normal<float, N>());

    // The range: ONE pass for both ends, through a structured accumulator
    // whose state carries them together.
    auto [lo, hi] = eval(fold<ops::MinMax>(x));

    // The histogram. bins<B>(x[i], lo, hi) is the value→bin map — which of
    // B equal bins over [lo, hi) holds x[i] — and scatter deposits a count
    // into that bin. drop discards out-of-range coordinates (numpy's
    // semantics); the maximum element sits exactly ON hi, so it floors to
    // bin B and is dropped — clamp<B> is the keep-everything spelling.
    auto counts = eval(scatter<i>(drop<B>(bins<B>(x[i], lo, hi)), 1u));

    // Bin centres, storage-free: the generator fuses into the arithmetic.
    float w = (hi - lo) / float(B);
    auto centres = eval(lo + gen::Iota<B>(0.5f) * w);

    // A text rendering; the real plot is one line of Python away.
    unsigned kept = 0, peak = 1;
    for (size_t b = 0; b < B; ++b) {
        kept += counts[b];
        peak = counts[b] > peak ? counts[b] : peak;
    }
    std::cout << "range [" << lo << ", " << hi << "], kept " << kept << " of "
              << N << " (the max itself lands ON hi and drops)\n";
    for (size_t b = 0; b < B; ++b) {
        std::cout.width(9);
        std::cout << centres[b] << " | ";
        for (unsigned s = 0; s < counts[b] * 50 / peak; ++s)
            std::cout << '#';
        std::cout << '\n';
    }

    // Out: np.load("histogram.npz") sees Counts and Centres, dtype and
    // shape derived from the types above.
    npy::savez("histogram.npz",
               Histogram{std::move(counts), std::move(centres)});
    std::cout << "wrote histogram.npz\n";

    // Everything above, named: stats::Histogram is exactly this program —
    // the MinMax pass, the deposit, the edges — and its {counts, edges}
    // struct saves the same way. Reach for it first; the spelling above is
    // what to write when a default does not fit.
    auto h = stats::Histogram<B>(x);
    unsigned named_total = 0;
    for (size_t b = 0; b < B; ++b)
        named_total += h.counts[b];
    std::cout << "stats::Histogram<" << B << ">(x) keeps " << named_total
              << " (clamp keeps the maximum too)\n";
}
