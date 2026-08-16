#pragma once

// Lowers an expression TYPE to a Slang compute shader. One helper per
// emitted section; gpu_program() is their concatenation.

#include "Meta.h"
#include "Sym.h"
#include "Tree.h"

// The sdw opt-in (map<f> GPU lowering): the generated header carries the
// translated blob + names. The ONE library header that names sdw.
#ifdef TENSOR_SDW_ENABLED
#include <sdw_functions.h>
#endif

#include <algorithm> // std::ranges::contains
#include <bit>
#include <cstddef>
#include <meta>
#include <string>
#include <string_view>
#include <type_traits>
#include <vector>

namespace tensor::detail {

// ── dialect ─────────────────────────────────────────────────────────────────
inline constexpr LeafStyle slang_style{"pc.in", ".data[", "]", "pc.s"};

// A value as a Slang literal: small exact integers as themselves, anything
// else as its exact bit pattern (a printed float would not round-trip).
template <typename T> consteval std::string gpu_literal(T v) {
    if constexpr (std::is_same_v<T, bool>) {
        return v ? "true" : "false";
    } else if constexpr (std::is_floating_point_v<T>) {
        // range FIRST: an out-of-range float→int conversion is UB
        if (v > T(-1024) && v < T(1024))
            if (const auto i = static_cast<long long>(v); T(i) == v)
                return (i < 0 ? "-" : "") + to_string(size_t(i < 0 ? -i : i)) +
                       ".0";
        if constexpr (sizeof(T) == 4)
            return "asfloat(0x" + to_hex(std::bit_cast<unsigned>(v), 8) + "u)";
        else {
            const auto b = std::bit_cast<unsigned long long>(v);
            return "asdouble(0x" + to_hex(unsigned(b), 8) + "u, 0x" +
                   to_hex(unsigned(b >> 32), 8) + "u)";
        }
    } else if (v < 0) {
        // negating the lowest value overflows: every negative as a pattern
        return std::string(gpu_type(^^T)) + "(0x" +
               to_hex(static_cast<unsigned long long>(v) &
                          ((1ull << (8 * sizeof(T))) - 1),
                      2 * sizeof(T)) +
               "u)";
    } else {
        return to_string(size_t(v));
    }
}

// ── tree census ─────────────────────────────────────────────────────────────

// Leaf element types, in render's (= for_each_leaf's) DFS order, plus the two
// shape facts the emitters would otherwise each re-derive with a walk of
// their own. Every whole-tree question this file asks is answered here, once.
struct Leaves {
    std::vector<std::meta::info> views;
    std::vector<std::meta::info> scalars;
    bool has_indexed = false;
    bool has_sampler = false;       // the tree draws: emit the random core
    bool has_stream = false;        // rng::Sample: not emissible at all
    std::meta::info fold_node = {}; // first fold in DFS order, or {}
    size_t nodes = 0;
};

consteval void collect(std::meta::info node, Leaves &l) {
    auto t = std::meta::dealias(node);
    ++l.nodes;
    if (is_broadcast_scalar_type(t))
        l.scalars.push_back(t);
    else if (is_generator_type(t)) {
        // A generator renders inline, so it claims no buffer and no ABI
        // slot — only the scalars its kind reads. A sampler's two are the
        // halves of its key, which are uint whatever it samples INTO.
        const auto k = gen_kind(t);
        const bool sampler = gen_is_sampler(k);
        l.has_sampler = l.has_sampler || (sampler && k != GenKind::Stream);
        l.has_stream = l.has_stream || k == GenKind::Stream;
        auto p = sampler ? ^^unsigned : alias_of(t, "type");
        for (size_t n = gen_params(k); n--;)
            l.scalars.push_back(p);
    } else if (is_indexed(t)) {
        l.has_indexed = true;
        const auto ms = maps_of(t);
        for (auto m : ms)
            if (map_padded(m)) {
                l.scalars.push_back(alias_of(t, "type"));
                break;
            }
        // Then each gathered axis's coordinate, in axis order — the order
        // for_each_leaf and the renderers both walk.
        const auto cs = indexed_coords_of(t);
        for (size_t k = 0; k < ms.size(); ++k)
            if (map_data(ms[k]))
                collect(cs[k], l);
        collect(indexed_operand_of(t), l);
    } else if (is_placed(t)) {
        // A scatter destination is transparent — its coordinate's leaves,
        // as for_each_leaf visits them. Without this branch Placed falls
        // through to the tensor-leaf arm below (it has a `type` alias but
        // no `op_type`) and claims a buffer slot it never fills.
        collect(placed_coord_of(t), l);
    } else if (auto op = op_of(t); op == std::meta::info{}) // tensor leaf
        // remove_cv: constness belongs to the view, not the buffer
        l.views.push_back(std::meta::remove_cv(alias_of(t, "type")));
    else {
        if (is_contraction(op) && l.fold_node == std::meta::info{})
            l.fold_node = t;
        for (auto c : children_of(t))
            collect(c, l);
    }
}

consteval Leaves leaves_of(std::meta::info expr) {
    Leaves l;
    collect(expr, l);
    return l;
}

consteval size_t element_count_of(std::meta::info expr) {
    size_t n = 1;
    auto d = std::meta::dealias(expr);
    if (is_indexed(d)) { // a lone subscripted leaf carries no alias
        auto p = free_plan(d);
        for (size_t a = 0; a < p.n; ++a)
            n *= p.ext[a];
        return n;
    }
    for (auto e : node_extents_of(expr))
        n *= e;
    return n;
}

// ── GPU evaluation support (Gpu/Eval.h) ─────────────────────────────────────
// Derived from the same census as the emitted program, so host and shader
// cannot disagree.

inline constexpr size_t workgroup_size = 64;

// Vulkan's GUARANTEED maxComputeSharedMemorySize. A dev machine typically
// reports far more (64 KB here), so gating on what it happens to allow
// would ship programs that fail pipeline creation elsewhere — the same
// cliff as OpCapability Int64 and shader double.
inline constexpr size_t groupshared_budget = 16u * 1024;

consteval size_t view_count_of(std::meta::info expr) {
    return leaves_of(std::meta::dealias(expr)).views.size();
}

// PC scalar-section bytes — the same natural-alignment rule eval's leaf
// walk packs by. Takes the census so a caller holding one pays no second
// walk.
consteval size_t scalar_bytes(const Leaves &l) {
    size_t off = 0;
    for (auto t : l.scalars) {
        size_t a = std::meta::alignment_of(t);
        off = (off + a - 1) & ~(a - 1);
        off += std::meta::size_of(t);
    }
    return off;
}

consteval size_t scalar_bytes_of(std::meta::info expr) {
    return scalar_bytes(leaves_of(std::meta::dealias(expr)));
}

// Vulkan's guaranteed push-constant minimum: 8-aligned scalars + one 8-byte
// address per buffer.
inline constexpr size_t push_budget = 128;

consteval size_t push_bytes_of(std::meta::info expr) {
    const auto l = leaves_of(std::meta::dealias(expr));
    return ((scalar_bytes(l) + 7) & ~size_t{7}) + 8 * (1 + l.views.size());
}

consteval std::string_view op_atomic(std::meta::info op) {
    auto anns = std::meta::annotations_of_with_type(op, ^^Atomic);
    return anns.empty()
               ? std::string_view{}
               : std::meta::extract<Atomic>(std::meta::constant_of(anns[0]))
                     .str;
}

// Integral element types only. Slang emits a float atomic happily, but it
// lowers to OpAtomicFAddEXT under a capability the device layer never enables
// — invalid use that some drivers accept, which is the worst way to be wrong.
consteval bool scatter_is_atomic(std::meta::info expr) {
    const auto t = std::meta::dealias(alias_of(expr, "type"));
    return std::meta::is_integral_type(t) &&
           !op_atomic(fold_op_of(op_of(std::meta::dealias(expr)))).empty();
}

// Path B's grid. No float atomic exists on this backend, so every
// accumulator is owned privately by one thread — W·S of them per group —
// and covering more cells than a slice means covering them in more groups,
// each re-reading the contributions to find the ones it owns. That
// redundancy is what buys full occupancy at any cell count; docs/
// scatter-multigroup-plan.md measures it.
struct ScatterGrid {
    size_t slice = 1;       // cells a group owns
    size_t cell_groups = 1; // slices covering the output
    constexpr size_t groups() const { return cell_groups; }
};

consteval ScatterGrid scatter_grid(size_t cells, size_t elem) {
    ScatterGrid g{};
    g.slice = groupshared_budget / (workgroup_size * elem);
    if (g.slice > cells)
        g.slice = cells;
    if (g.slice == 0)
        g.slice = 1; // an element wider than a lane's share: one cell each
    g.cell_groups = (cells + g.slice - 1) / g.slice;
    return g;
}

// ── backend capability ──────────────────────────────────────────────────────

consteval std::meta::info mapped_fn_of(std::meta::info op) {
    return std::meta::extract<std::meta::info>(
        std::meta::template_arguments_of(op)[0]);
}

// A mapped fn's marker: its first template argument is a reflected entity
// (an info value) — no structural op carries one.
consteval bool is_mapped_fn(std::meta::info op) {
    if (!std::meta::has_template_arguments(op))
        return false;
    auto args = std::meta::template_arguments_of(op);
    return !args.empty() && !std::meta::is_type(args[0]) &&
           std::meta::dealias(std::meta::type_of(args[0])) ==
               std::meta::dealias(^^std::meta::info);
}

consteval bool cpu_only(std::meta::info op) {
    return !std::meta::annotations_of_with_type(op, ^^CpuOnly).empty();
}
consteval std::meta::info first_cpu_only_op(std::meta::info node) {
    return first_op(node, cpu_only);
}

#ifdef TENSOR_SDW_ENABLED
// Keyed by identifier — the spelling both the blob and the call site use.
consteval bool sdw_translated(std::meta::info f) {
    if (!std::meta::has_identifier(f))
        return false;
    for (auto n : sdw::function_names)
        if (n == std::meta::identifier_of(f))
            return true;
    return false;
}
#endif

consteval bool fn_gpu_emissible([[maybe_unused]] std::meta::info op) {
#ifdef TENSOR_SDW_ENABLED
    return sdw_translated(mapped_fn_of(op));
#else
    return false;
#endif
}

consteval bool tree_gpu_emissible(std::meta::info node) {
    return first_op(node, [](std::meta::info op) consteval {
               return is_mapped_fn(op) && !fn_gpu_emissible(op);
           }) == std::meta::info{};
}

// Decides whether the program needs the fn_helpers section.
consteval bool tree_has_mapped_fn(std::meta::info node) {
    return first_op(node, is_mapped_fn) != std::meta::info{};
}

#ifdef TENSOR_SDW_ENABLED
// First map<f> function sdw did not translate — {} if all were.
consteval std::meta::info first_untranslated_fn(std::meta::info node) {
    auto op = first_op(node, [](std::meta::info o) consteval {
        return is_mapped_fn(o) && !sdw_translated(mapped_fn_of(o));
    });
    return op == std::meta::info{} ? op : mapped_fn_of(op);
}
#endif

// bool is out: Buf_bool is not host-shareable in SPIR-V. double is out
// because shader float64 is not portable — Metal has no double at all, so
// the SPIR-V compiles and the pipeline then fails to build.
consteval bool type_gpu_mappable(std::meta::info t) {
    t = std::meta::dealias(t);
    return t == (^^float) || t == (^^int) || t == (^^unsigned);
}

// First offending type — output first, then the leaves — or {} if all map.
// Takes the census so a caller holding one pays no second walk.
consteval std::meta::info first_unmappable_type(const Leaves &l,
                                                std::meta::info out_t) {
    if (!type_gpu_mappable(out_t))
        return out_t;
    for (auto t : l.views)
        if (!type_gpu_mappable(t))
            return t;
    for (auto t : l.scalars)
        if (!type_gpu_mappable(t))
            return t;
    return {};
}

consteval std::meta::info first_unmappable_type(std::meta::info expr) {
    expr = std::meta::dealias(expr);
    return first_unmappable_type(leaves_of(expr), alias_of(expr, "type"));
}

// ── eval diagnostics ────────────────────────────────────────────────────────

consteval std::string gpu_map_error(std::meta::info expr) {
    size_t v = 0, s = 0;
#ifdef TENSOR_SDW_ENABLED
    auto f = first_untranslated_fn(std::meta::dealias(expr));
    std::string who = f == std::meta::info{} ? std::string("a mapped function")
                      : std::meta::has_identifier(f)
                          ? "'" + std::string(std::meta::identifier_of(f)) + "'"
                          : std::string("an anonymous callable");
    return "gpu eval: " + who +
           " has no sdw translation - define it in namespace sdw in the "
           "kernels file given to sdw --emit-functions: " +
           render(std::meta::dealias(expr), formula_style, "i", v, s);
#else
    return "gpu eval: map<f> is CPU-only (function bodies are not capturable "
           "yet): " +
           render(std::meta::dealias(expr), formula_style, "i", v, s);
#endif
}

// rng::Sample<f> needs two things the shader has not got: the function body,
// and the per-cell stream to hand it. Keyed on the Stream LEAF rather than
// the op, so it holds however the function is spelt or translated.
consteval bool tree_gpu_streamless(std::meta::info expr) {
    return !leaves_of(std::meta::dealias(expr)).has_stream;
}

consteval std::string gpu_sample_error(std::meta::info expr) {
    return "gpu eval: rng::Sample<f> is CPU-only in " + type_name(expr) +
           " — the shader would need the function body and a per-cell "
           "stream, and neither is bridged; the named distributions "
           "(rng::Exponential, rng::Normal, …) do lower, or eval(expr) on the CPU.";
}

consteval std::string gpu_type_error(std::meta::info expr) {
    size_t v = 0, s = 0;
    const auto t = first_unmappable_type(expr);
    const bool f64 = std::meta::dealias(t) == (^^double);
    return "gpu eval: unsupported element type " + type_name(t) + " in " +
           render(std::meta::dealias(expr), formula_style, "i", v, s) +
           (f64 ? " - shader float64 is not portable (Metal has none); use "
                  "float, or eval(expr) on the CPU"
                : "");
}

consteval std::string gpu_cpu_only_error(std::meta::info expr) {
    size_t v = 0, s = 0;
    auto op = first_cpu_only_op(std::meta::dealias(expr));
    std::string who = op == std::meta::info{}
                          ? std::string("an op")
                          : "'" + std::string(symbol_of(op)) + "'";
    return "gpu eval: " + who +
           " has no Slang intrinsic and evaluates on the CPU only: " +
           render(std::meta::dealias(expr), formula_style, "i", v, s);
}

consteval std::string gpu_push_error(std::meta::info expr) {
    return "gpu eval: push-constant budget exceeded: " +
           to_string(push_bytes_of(expr)) +
           " bytes needed (8-aligned scalars + 8 per buffer), " +
           to_string(push_budget) + " available";
}

// ── program sections, in emission order ─────────────────────────────────────

// One `struct Buf_T { T data[1]; };` per distinct element type in the tree.
consteval std::string buffer_structs(const Leaves &l, std::meta::info out_t) {
    std::vector<std::meta::info> types{out_t};
    for (auto t : l.views)
        types.push_back(t);
    std::string s;
    std::vector<std::string_view> seen;
    for (auto t : types) {
        auto name = gpu_type(t);
        if (std::ranges::contains(seen, name))
            continue;
        seen.push_back(name);
        s += "struct Buf_";
        s += name;
        s += " { ";
        s += name;
        s += " data[1]; };\n";
    }
    return s;
}

// Philox-4x32-10, the same arithmetic detail/Rng.h runs on the host — so a
// sample is the same number on both sides, up to the float rounding of the
// transforms. 32-bit throughout on purpose: the 64-bit multiply would pull
// in OpCapability Int64, which not every driver offers. Emitted only where
// the tree actually samples, so every other program is unchanged.
inline constexpr std::string_view rng_helpers = R"(uint rng_mulhi(uint a, uint b) {
  uint a0 = a & 0xffffu, a1 = a >> 16, b0 = b & 0xffffu, b1 = b >> 16;
  uint p00 = a0 * b0, p01 = a0 * b1, p10 = a1 * b0, p11 = a1 * b1;
  uint mid = (p00 >> 16) + (p01 & 0xffffu) + (p10 & 0xffffu);
  return p11 + (p01 >> 16) + (p10 >> 16) + (mid >> 16);
}

uint4 rng_philox(uint4 c, uint k0, uint k1) {
  for (int r = 0; r < 10; ++r) {
    if (r > 0) { k0 += 0x9E3779B9u; k1 += 0xBB67AE85u; }
    uint hi0 = rng_mulhi(0xD2511F53u, c.x), lo0 = 0xD2511F53u * c.x;
    uint hi1 = rng_mulhi(0xCD9E8D57u, c.z), lo1 = 0xCD9E8D57u * c.z;
    c = uint4(hi1 ^ c.y ^ k0, lo1, hi0 ^ c.w ^ k1, lo0);
  }
  return c;
}

float rng_uniform(uint k0, uint k1, uint cell) {
  uint4 r = rng_philox(uint4(cell, 0u, 0u, 0u), k0, k1);
  return float(r.x >> 8) * 5.9604644775390625e-8f;
}

float rng_normal(uint k0, uint k1, uint cell) {
  uint4 r = rng_philox(uint4(cell, 0u, 0u, 0u), k0, k1);
  float u1 = 1.0f - float(r.x >> 8) * 5.9604644775390625e-8f;
  float u2 = float(r.z >> 8) * 5.9604644775390625e-8f;
  return sqrt(-2.0f * log(u1)) * cos(6.28318530717958647692f * u2);
}
)";

// The sdw blob iff the tree maps anything (Slang dead-strips), plus the
// random core iff it samples. A tree that does neither emits nothing here,
// so operator-only programs stay byte-identical in every configuration.
consteval std::string fn_helpers([[maybe_unused]] std::meta::info expr,
                                 const Leaves &l) {
    std::string out;
#ifdef TENSOR_SDW_ENABLED
    if (tree_has_mapped_fn(expr))
        out += sdw::functions_slang;
#endif
    if (l.has_sampler)
        out += rng_helpers;
    return out;
}

// Scalars, output pointer, input pointers — the member order eval's leaf
// walk mirrors when packing.
consteval std::string push_constants(const Leaves &l, std::meta::info out_t) {
    std::string s = "\nstruct PC {\n";
    for (size_t i = 0; i < l.scalars.size(); ++i) {
        s += "  ";
        s += gpu_type(l.scalars[i]);
        s += " s";
        append_number(s, i);
        s += ";\n";
    }
    s += "  Buf_";
    s += gpu_type(out_t);
    s += "* out_buf;\n";
    for (size_t i = 0; i < l.views.size(); ++i) {
        s += "  Buf_";
        s += gpu_type(l.views[i]);
        s += "* in";
        append_number(s, i);
        s += ";\n";
    }
    s += "};\n[[vk::push_constant]] PC pc;\n";
    return s;
}

// ── fold kernels ────────────────────────────────────────────────────────────
// A fold is root-only, so the emitter branches once; the child renders
// through the ordinary machinery at the composed index.

consteval std::string fold_step(std::string_view sym, const std::string &acc,
                                const std::string &x) {
    return identifier_like(sym) ? std::string(sym) + "(" + acc + ", " + x + ")"
                                : acc + " " + std::string(sym) + " " + x;
}

// fold_step's sibling: an atomic is a STATEMENT, so it cannot be spliced
// into `acc = …` the way an expression can.
consteval std::string fold_store(std::string_view atomic,
                                 const std::string &dst,
                                 const std::string &x) {
    return std::string(atomic) + "(" + dst + ", " + x + ");";
}

consteval size_t count_of(const FreePlan &p) {
    size_t n = 1;
    for (size_t a = 0; a < p.n; ++a)
        n *= p.ext[a];
    return n;
}

consteval std::string per_cell_prologue(size_t cells) {
    return "\n[shader(\"compute\")]\n[numthreads(" + to_string(workgroup_size) +
           ", 1, 1)]\n"
           "void main(uint3 tid : SV_DispatchThreadID) {\n"
           "  uint i = tid.x;\n"
           "  if (i >= " +
           to_string(cells) + ") return;\n";
}

// A caller holding the free plan already knows the count; the walk is for
// one that does not.
consteval std::string per_cell_prologue(std::meta::info expr) {
    return per_cell_prologue(element_count_of(expr));
}

// Both full folds: the caller's grid-stride accumulate, then a groupshared
// tree — one dispatch, one group.
consteval std::string single_group_kernel(const std::string &type,
                                          std::string_view sym,
                                          std::string_view identity,
                                          const std::string &accumulate) {
    return "\ngroupshared " + type + " sdata[" + to_string(workgroup_size) +
           "];\n"
           "\n[shader(\"compute\")]\n[numthreads(" +
           to_string(workgroup_size) +
           ", 1, 1)]\n"
           "void main(uint3 gid : SV_GroupThreadID) {\n"
           "  uint t = gid.x;\n"
           "  " +
           type + " acc = " + std::string(identity) + ";\n" + accumulate +
           "  sdata[t] = acc;\n"
           "  GroupMemoryBarrierWithGroupSync();\n"
           "  for (uint w = " +
           to_string(workgroup_size / 2) +
           "; w > 0; w >>= 1) {\n"
           "    if (t < w) sdata[t] = " +
           fold_step(sym, "sdata[t]", "sdata[t + w]") +
           ";\n"
           "    GroupMemoryBarrierWithGroupSync();\n"
           "  }\n"
           "  if (t == 0) pc.out_buf.data[0] = sdata[0];\n}\n";
}

// Past this a one-group fold is just slow; eval refuses and names the
// stage that lifts the limit.
inline constexpr size_t single_group_budget = workgroup_size * 64;

consteval size_t contract_fold_count(std::meta::info expr) {
    const auto op = op_of(expr);
    return contract_plan(std::meta::dealias(children_of(expr)[0]),
                         summed_of(op))
        .fold_count;
}

consteval size_t fold_size_of(std::meta::info expr) {
    return contract_fold_count(expr);
}

// The fold subtree below an epilogue (the whole tree when the root IS the
// fold), or {}.
consteval std::meta::info fold_node_of(std::meta::info node) {
    auto t = std::meta::dealias(node);
    if (is_broadcast_scalar_type(t))
        return {};
    if (is_indexed(t))
        return fold_node_of(indexed_operand_of(t));
    auto op = op_of(t);
    if (op == std::meta::info{})
        return {};
    if (is_contraction(op))
        return t;
    for (auto c : children_of(t))
        if (auto f = fold_node_of(c); f != std::meta::info{})
            return f;
    return {};
}

// The scan subtree below an epilogue (the whole tree when the root IS the
// scan), or {}. Its own twin of fold_node_of: a scan carries no `summed`,
// so is_contraction never answers for it.
consteval std::meta::info scan_node_of_tree(std::meta::info node) {
    auto t = std::meta::dealias(node);
    if (is_broadcast_scalar_type(t) || is_generator_type(t))
        return {};
    if (is_indexed(t))
        return scan_node_of_tree(indexed_operand_of(t));
    auto op = op_of(t);
    if (op == std::meta::info{})
        return {};
    if (is_scan_op(op))
        return t;
    for (auto c : children_of(t))
        if (auto f = scan_node_of_tree(c); f != std::meta::info{})
            return f;
    return {};
}

// ── contraction kernels ─────────────────────────────────────────────────────
// Coordinates are SIGNED int locals/loop variables — an affine subscript
// can go negative and its guard must be able to say so. The one place the
// dialects part ways, and only in the declarations.

consteval std::string contract_axis_kernel(std::meta::info expr,
                                           std::string_view identity,
                                           const std::vector<size_t> &order) {
    size_t v = 0, s = 0;
    const auto op = op_of(expr);
    const auto summand = std::meta::dealias(children_of(expr)[0]);
    const auto summed = summed_of(op);
    const auto census = id_census(summand);
    const auto sym = symbol_of(fold_op_of(op));
    const auto type = std::string(gpu_type(alias_of(expr, "type")));

    // The free ids in output order — eval<Order…> permutes the layout.
    std::vector<size_t> free_ids;
    for (auto id : census.order)
        if (!std::ranges::contains(summed, id))
            free_ids.push_back(id);
    if (!order.empty())
        free_ids = order;
    std::vector<size_t> out_ext;
    for (auto id : free_ids)
        out_ext.push_back(census.pinned(id));
    std::vector<std::string> coord(index_slots);
    std::string frees;
    size_t kept = 0;
    for (auto id : free_ids) {
        const auto name = "i" + to_string(kept);
        frees += "  int " + name + " = int(" + axis_coord("i", out_ext, kept) +
                 ");\n";
        coord[id] = name;
        ++kept;
    }
    std::string loops, indent = "  ";
    for (size_t d = 0; d < summed.size(); ++d) {
        const auto jn = "j" + to_string(d);
        coord[summed[d]] = jn;
        loops += indent + "for (int " + jn + " = 0; " + jn + " < " +
                 to_string(census.pinned(summed[d])) + "; ++" + jn + ")\n";
        indent += "  ";
    }
    const auto body = render_indexed(summand, slang_style, coord, v, s);
    return per_cell_prologue(expr) + frees + "  " + type +
           " acc = " + std::string(identity) + ";\n" + loops + indent +
           "acc = " + fold_step(sym, "acc", body) +
           ";\n"
           "  pc.out_buf.data[i] = acc;\n}\n";
}

// Full contraction: grid-stride over the summed space, the counter
// decomposing as the CPU loop does.
consteval std::string contract_all_kernel(std::meta::info expr,
                                          std::string_view identity) {
    size_t v = 0, s = 0;
    const auto op = op_of(expr);
    const auto summand = std::meta::dealias(children_of(expr)[0]);
    const auto summed = summed_of(op);
    const auto census = id_census(summand);
    const auto sym = symbol_of(fold_op_of(op));
    const auto type = std::string(gpu_type(alias_of(expr, "type")));

    std::vector<size_t> sext;
    for (auto a : summed)
        sext.push_back(census.pinned(a));
    std::vector<std::string> coord(index_slots);
    std::string decls;
    for (size_t d = 0; d < summed.size(); ++d) {
        const auto jn = "j" + to_string(d);
        coord[summed[d]] = jn;
        decls +=
            "    int " + jn + " = int(" + axis_coord("f", sext, d) + ");\n";
    }
    const auto body = render_indexed(summand, slang_style, coord, v, s);
    return single_group_kernel(
        type, sym, identity,
        "  for (uint f = t; f < " + to_string(contract_fold_count(expr)) +
            "; f += " + to_string(workgroup_size) + ") {\n" + decls +
            "    acc = " + fold_step(sym, "acc", body) + ";\n  }\n");
}

// ── epilogue kernels ────────────────────────────────────────────────────────
// The tree rendered with the fold subtree spelt as the accumulator; the
// summand renders AT the fold's DFS position, so leaf numbering stays in
// for_each_leaf lockstep. `inner` carries the loop variables.
consteval std::string render_epilogue(std::meta::info node,
                                      const LeafStyle &style,
                                      const std::vector<std::string> &coord,
                                      const std::vector<std::string> &inner,
                                      std::string &accumulate, size_t &views,
                                      size_t &scalars) {
    auto t = std::meta::dealias(node);
    if (is_broadcast_scalar_type(t))
        return std::string(style.scalar_prefix) + to_string(scalars++);
    if (is_indexed(t)) {
        auto operand = indexed_operand_of(t);
        auto fill = fill_spelling(t, style, scalars);
        auto data = render_data_coords(t, style, coord, views, scalars);
        auto ai =
            decorated_index(maps_of(t), node_extents_of(operand), coord, data);
        return guarded(ai.guards,
                       render(operand, style, ai.index, views, scalars), fill);
    }
    auto op = op_of(t);
    if (op != std::meta::info{} && is_contraction(op)) {
        accumulate = render_indexed(std::meta::dealias(children_of(t)[0]),
                                    style, inner, views, scalars);
        return "acc";
    }
    std::vector<std::string> child;
    for (auto c : children_of(t))
        child.push_back(render_epilogue(c, style, coord, inner, accumulate,
                                        views, scalars));
    return spell_node(op, child);
}

// The free/loop coordinate environments an epilogue (or ordered) kernel
// declares: free ids as int i0…, fold ids as loop variables j0….
struct KernelCoords {
    std::vector<std::string> coord, inner;
    std::string frees, loops, indent = "  ";
};

consteval KernelCoords
kernel_coords(const FreePlan &free, const std::vector<size_t> &summed,
              const std::vector<size_t> &summed_ext) {
    KernelCoords k;
    k.coord.resize(index_slots);
    std::vector<size_t> out_ext;
    for (size_t a = 0; a < free.n; ++a)
        out_ext.push_back(free.ext[a]);
    for (size_t a = 0; a < free.n; ++a) {
        const auto name = "i" + to_string(a);
        k.frees += "  int " + name + " = int(" +
                   axis_coord("i", out_ext, a) + ");\n";
        k.coord[free.id[a]] = name;
    }
    k.inner = k.coord;
    for (size_t d = 0; d < summed.size(); ++d) {
        const auto jn = "j" + to_string(d);
        k.inner[summed[d]] = jn;
        k.loops += k.indent + "for (int " + jn + " = 0; " + jn + " < " +
                   to_string(summed_ext[d]) + "; ++" + jn + ")\n";
        k.indent += "  ";
    }
    return k;
}

consteval std::string epilogue_axis_kernel(std::meta::info expr,
                                           std::meta::info fold,
                                           const FreePlan &free,
                                           std::string_view identity) {
    size_t v = 0, s = 0;
    const auto fop = op_of(fold);
    const auto summand = std::meta::dealias(children_of(fold)[0]);
    const auto summed = summed_of(fop);
    const auto census = id_census(summand);
    const auto sym = symbol_of(fold_op_of(fop));
    const auto ftype = std::string(gpu_type(alias_of(fold, "type")));
    std::vector<size_t> sext;
    for (auto a : summed)
        sext.push_back(census.pinned(a));
    auto k = kernel_coords(free, summed, sext);
    std::string accumulate;
    const auto store = render_epilogue(expr, slang_style, k.coord, k.inner,
                                       accumulate, v, s);
    return per_cell_prologue(expr) + k.frees + "  " + ftype +
           " acc = " + std::string(identity) + ";\n" + k.loops + k.indent +
           "acc = " + fold_step(sym, "acc", accumulate) +
           ";\n"
           "  pc.out_buf.data[i] = " +
           store + ";\n}\n";
}

consteval std::string epilogue_all_kernel(std::meta::info expr,
                                          std::meta::info fold,
                                          std::string_view identity) {
    size_t v = 0, s = 0;
    const auto fop = op_of(fold);
    const auto summand = std::meta::dealias(children_of(fold)[0]);
    const auto summed = summed_of(fop);
    const auto census = id_census(summand);
    const auto sym = symbol_of(fold_op_of(fop));
    const auto ftype = std::string(gpu_type(alias_of(fold, "type")));
    std::vector<size_t> sext;
    for (auto a : summed)
        sext.push_back(census.pinned(a));
    std::vector<std::string> coord(index_slots), inner(index_slots);
    std::string decls;
    for (size_t d = 0; d < summed.size(); ++d) {
        const auto jn = "j" + to_string(d);
        inner[summed[d]] = jn;
        decls +=
            "    int " + jn + " = int(" + axis_coord("f", sext, d) + ");\n";
    }
    std::string accumulate;
    const auto store = render_epilogue(expr, slang_style, coord, inner,
                                       accumulate, v, s);
    const auto count = to_string(contract_fold_count(fold));
    return "\ngroupshared " + ftype + " sdata[" + to_string(workgroup_size) +
           "];\n"
           "\n[shader(\"compute\")]\n[numthreads(" +
           to_string(workgroup_size) +
           ", 1, 1)]\n"
           "void main(uint3 gid : SV_GroupThreadID) {\n"
           "  uint t = gid.x;\n"
           "  " +
           ftype + " acc = " + std::string(identity) + ";\n" +
           "  for (uint f = t; f < " + count +
           "; f += " + to_string(workgroup_size) + ") {\n" + decls +
           "    acc = " + fold_step(sym, "acc", accumulate) +
           ";\n  }\n"
           "  sdata[t] = acc;\n"
           "  GroupMemoryBarrierWithGroupSync();\n"
           "  for (uint w = " +
           to_string(workgroup_size / 2) +
           "; w > 0; w >>= 1) {\n"
           "    if (t < w) sdata[t] = " +
           fold_step(sym, "sdata[t]", "sdata[t + w]") +
           ";\n"
           "    GroupMemoryBarrierWithGroupSync();\n"
           "  }\n"
           "  if (t == 0) {\n"
           "    acc = sdata[0];\n"
           "    pc.out_buf.data[0] = " +
           store +
           ";\n  }\n}\n";
}

// ── the scan kernel ─────────────────────────────────────────────────────────
// A scan keeps its index, so the parallel domain is the ROWS — the output
// space minus the scanned axis — and each thread walks its own row in a
// register. One dispatch, no groupshared, no scratch: the simplest kernel
// in this file, because the accumulator never crosses a thread.

// The tree above the scan, with the scan itself spelt as the accumulator.
// Its summand renders AT the scan's own DFS position, so leaf numbering
// stays in for_each_leaf lockstep — the epilogue's leaves are subscripted
// reads in the same coordinate environment.
consteval std::string render_over_scan(std::meta::info node,
                                       const std::vector<std::string> &coord,
                                       size_t &views, size_t &scalars,
                                       std::string &summand) {
    const auto t = std::meta::dealias(node);
    if (!is_broadcast_scalar_type(t) && !is_generator_type(t))
        if (const auto op = op_of(t); op != std::meta::info{}) {
            if (is_scan_op(op)) {
                summand = render_indexed(std::meta::dealias(children_of(t)[0]),
                                         slang_style, coord, views, scalars);
                return "acc";
            }
            std::vector<std::string> child;
            for (auto c : children_of(t))
                child.push_back(
                    render_over_scan(c, coord, views, scalars, summand));
            return spell_node(op, child);
        }
    return render_indexed(t, slang_style, coord, views, scalars);
}

consteval std::string scan_kernel(std::meta::info expr,
                                  std::string_view identity,
                                  std::meta::info node, const FreePlan &free) {
    const auto t = std::meta::dealias(node);
    const auto op = op_of(t);
    const size_t sid = scanned_of(op);
    const auto type = std::string(gpu_type(alias_of(t, "type")));
    const auto sym = symbol_of(fold_op_of(op));

    size_t axis = 0;
    for (size_t a = 0; a < free.n; ++a)
        if (free.id[a] == sid)
            axis = a;

    // Row-major strides over the whole output, so a scanned axis anywhere
    // writes where it should.
    std::vector<size_t> stride(free.n);
    for (size_t a = free.n, acc = 1; a-- > 0;) {
        stride[a] = acc;
        acc *= free.ext[a];
    }

    // The rows: every free axis but the scanned one, in plan order.
    std::vector<size_t> rext, rpos;
    for (size_t a = 0; a < free.n; ++a)
        if (a != axis) {
            rext.push_back(free.ext[a]);
            rpos.push_back(a);
        }
    size_t rows = 1;
    for (size_t e : rext)
        rows *= e;

    // The environment: the row coordinates decompose the dispatch index,
    // the scanned one is the loop variable. SIGNED ints, per the rule every
    // index-bearing kernel here follows.
    std::vector<std::string> coord(index_slots);
    std::string decls, base;
    for (size_t d = 0; d < rext.size(); ++d) {
        const auto name = "i" + to_string(rpos[d]);
        coord[free.id[rpos[d]]] = name;
        decls += "    int " + name + " = int(" + axis_coord("r", rext, d) +
                 ");\n";
        base += (base.empty() ? "" : " + ") + name + " * " +
                to_string(stride[rpos[d]]);
    }
    const auto sname = "s" + to_string(axis);
    coord[sid] = sname;
    // The store index, with the elisions the shape guarantees: a unit stride
    // needs no multiply, and a single-row scan has no base at all.
    const std::string step =
        stride[axis] == 1 ? sname : sname + " * " + to_string(stride[axis]);
    const std::string at = base.empty() ? step : base + " + " + step;

    size_t views = 0, scalars = 0;
    std::string summand;
    const auto stored = render_over_scan(std::meta::dealias(expr), coord, views,
                                         scalars, summand);

    return "\n[shader(\"compute\")]\n[numthreads(" + to_string(workgroup_size) +
           ", 1, 1)]\nvoid main(uint3 tid : SV_DispatchThreadID) {\n"
           "  uint r = tid.x;\n  if (r < " +
           to_string(rows) + ") {\n" + decls + "    " + type +
           " acc = " + std::string(identity) + ";\n    for (int " + sname +
           " = 0; " + sname + " < " + to_string(free.ext[axis]) + "; ++" +
           sname + ") {\n      acc = " + fold_step(sym, "acc", summand) +
           ";\n      pc.out_buf.data[" + at + "] = " + stored +
           ";\n    }\n  }\n}\n";
}

// ── the scatter kernel ──────────────────────────────────────────────────────
// The one kernel whose parallel domain is the INPUT: a thread per
// contribution (a consumed index crossed with a surviving one). Contributions
// differing only in a surviving index land in different cells, never contend.
//
// Privatized into a groupshared bin when the output fits, then one atomic per
// (group, cell) — contention drops from O(contributions/cells) to
// O(groups/cells). Nothing here writes a cell it did not first read, so the
// host pre-fills with the identity. The bounds check is an `if`, never an
// early return: a returning thread would skip the barriers below it.
// What a contribution costs to place: the coordinate declarations, the
// destination cell, the drop guards, and the value. Rendered at the scatter's
// own DFS position so the leaf numbering stays in for_each_leaf lockstep,
// which is what lets an epilogue render around it.
struct ScatterBody {
    std::string decls, cell, guards, value;
};

consteval ScatterBody scatter_body(std::meta::info t, size_t &views,
                                   size_t &scalars) {
    const auto lay = scatter_layout(t);
    const auto kids = children_of(t);
    ScatterBody b;

    // The environment: a contribution index splits into its consumed part
    // and its surviving part, each decomposed row-major.
    std::vector<std::string> coord(index_slots);
    const std::string cf =
        lay.surv_size == 1 ? "f" : "(f / " + to_string(lay.surv_size) + ")";
    std::vector<size_t> sext(lay.sum_ext.begin(),
                             lay.sum_ext.begin() + lay.n_sum);
    for (size_t d = 0; d < lay.n_sum; ++d) {
        auto name = "i" + to_string(d);
        coord[lay.sum_id[d]] = name;
        b.decls += "    int " + name + " = int(" + axis_coord(cf, sext, d) +
                   ");\n";
    }
    std::vector<size_t> vext(lay.surv_ext.begin(),
                             lay.surv_ext.begin() + lay.n_surv);
    const std::string vf = "(f % " + to_string(lay.surv_size) + ")";
    for (size_t d = 0; d < lay.n_surv; ++d) {
        auto name = "n" + to_string(lay.surv_id[d]);
        coord[lay.surv_id[d]] = name;
        b.decls += "    int " + name + " = int(" + axis_coord(vf, vext, d) +
                   ");\n";
    }

    // The destination: each axis resolved by its own policy, then composed
    // row-major and offset by the surviving cell.
    std::string dest;
    for (size_t k = 0; k < lay.n_dest; ++k) {
        const auto ext = to_string(lay.dest_ext[k]);
        auto x = data_seed(placed_coord_of(kids[k]),
                           render_indexed(placed_coord_of(kids[k]), slang_style,
                                          coord, views, scalars));
        switch (placed_policy_of(kids[k])) {
        case Place::Wrap:
            x = "((" + x + ") % " + ext + " + " + ext + ") % " + ext;
            break;
        case Place::Clamp:
            x = "(" + x + " < 0 ? 0 : " + x + " >= " + ext + " ? " + ext +
                " - 1 : " + x + ")";
            break;
        case Place::Drop:
            b.guards += (b.guards.empty() ? "" : " && ") + x + " >= 0 && " +
                        x + " < " + ext;
            break;
        }
        dest = dest.empty() ? x : "(" + dest + ") * " + ext + " + " + x;
    }
    b.value = render_indexed(std::meta::dealias(kids.back()), slang_style,
                             coord, views, scalars);
    b.cell = lay.surv_size == 1
                 ? "(" + dest + ")"
                 : "(" + dest + ") * " + to_string(lay.surv_size) + " + " + vf;
    if (lay.n_dest == 0)
        b.cell = vf;
    return b;
}

// The tree above a scatter, with the scatter itself spelt as the merged
// accumulator. Its own leaves render AT that position, so the positional ABI
// is the same walk for_each_leaf makes. The epilogue is elementwise over the
// finished shape (an indexed one is a lint), so everything else is a plain
// read at the output cell.
consteval std::string render_over_scatter(std::meta::info node,
                                          const std::string &idx,
                                          size_t &views, size_t &scalars,
                                          ScatterBody &body) {
    const auto t = std::meta::dealias(node);
    // op_of only asks a class about its members, so the non-class leaves —
    // a broadcast scalar is the bare element type — are settled first.
    if (!is_broadcast_scalar_type(t) && !is_generator_type(t)) {
        if (const auto op = op_of(t); op != std::meta::info{}) {
            if (is_placed_op(op)) {
                body = scatter_body(t, views, scalars);
                return "a";
            }
            std::vector<std::string> child;
            for (auto c : children_of(t))
                child.push_back(
                    render_over_scatter(c, idx, views, scalars, body));
            return spell_node(op, child);
        }
    }
    return render_plain(t, slang_style, idx, views, scalars);
}

consteval std::string scatter_kernel(std::meta::info expr,
                                     std::string_view identity,
                                     std::meta::info node = {}) {
    const auto t = node == std::meta::info{} ? std::meta::dealias(expr)
                                             : std::meta::dealias(node);
    const bool epilogue = t != std::meta::dealias(expr);
    const auto lay = scatter_layout(t);
    const auto type = std::string(gpu_type(alias_of(t, "type")));
    // scatter_is_atomic, not the bare annotation: the element type gate is
    // what keeps a FLOAT atomic out, and Slang would emit one happily.
    // An epilogue takes the privatized path whatever the type: it applies
    // once per output cell, and only that path has a place to apply it.
    const auto atom = scatter_is_atomic(t) && !epilogue
                          ? std::string(op_atomic(fold_op_of(op_of(t))))
                          : "";
    const size_t cells = lay.output_cells(), contrib = lay.contributions();
    const bool priv =
        cells * std::meta::size_of(alias_of(t, "type")) <= groupshared_budget;
    const auto sym = symbol_of(fold_op_of(op_of(t)));
    const std::string id{identity};

    const auto g =
        scatter_grid(cells, std::meta::size_of(alias_of(t, "type")));
    const bool sliced = atom.empty() && g.cell_groups > 1;
    const std::string out_cell = sliced ? "base + c" : "c";

    size_t views = 0, scalars = 0;
    ScatterBody b;
    // ONE walk, in DFS order: the epilogue's own leaves are numbered around
    // the scatter's, exactly as for_each_leaf visits them.
    const std::string stored =
        epilogue ? render_over_scatter(std::meta::dealias(expr), out_cell,
                                       views, scalars, b)
                 : (b = scatter_body(t, views, scalars), std::string("a"));
    const std::string &decls = b.decls, &guards = b.guards, &value = b.value,
                      &cell = b.cell;

    // ── path B: no atomic, so a lane may only touch bins it owns. The
    // output is sliced across groups and each group's W private bins merge
    // in ascending thread order — deterministic, and every output cell is
    // written, so it needs no pre-fill.
    if (atom.empty()) {
        const std::string ws = to_string(workgroup_size);
        const std::string sl = to_string(g.slice);
        // One slice owns everything: no base, no membership test, and the
        // store covers the output exactly.
        const std::string idx = sliced ? "d - base" : "d";
        std::string test = sliced ? "d >= base && d < base + " + sl : "";
        if (!guards.empty())
            test += (test.empty() ? "" : " && ") + guards;

        std::string deposit = "bin[t][" + idx + "] = " +
                              fold_step(sym, "bin[t][" + idx + "]", value) +
                              ";";
        std::string body = decls + "    int d = int(" + cell + ");\n    " +
                           (test.empty() ? deposit
                                         : "if (" + test + ") " + deposit) +
                           "\n";

        // The tail slice is partial whenever the slice does not divide the
        // output; every other store covers its slice exactly. A partial
        // tail implies a sliced grid, so `base` is always in scope here.
        const bool partial = cells % g.slice != 0;
        const std::string ind = partial ? "      " : "    ";
        const std::string store =
            ind + type + " a = " + id + ";\n" + ind + "for (uint u = 0; u < " +
            ws + "; ++u) a = " + fold_step(sym, "a", "bin[u][c]") + ";\n" +
            ind + "pc.out_buf.data[" + out_cell + "] = " + stored + ";\n";

        std::string s = "\ngroupshared " + type + " bin[" + ws + "][" + sl +
                        "];\n\n[shader(\"compute\")]\n[numthreads(" + ws +
                        ", 1, 1)]\nvoid main(" +
                        (sliced ? "uint3 gid : SV_GroupID, " : "") +
                        "uint3 lid : SV_GroupThreadID) {\n  uint t = lid.x;\n";
        if (sliced)
            s += "  int base = int(gid.x) * " + sl + ";\n";
        s += "  for (uint c = 0; c < " + sl + "; ++c) bin[t][c] = " + id +
             ";\n  GroupMemoryBarrierWithGroupSync();\n"
             "  for (uint f = t; f < " +
             to_string(contrib) + "; f += " + ws + ") {\n" + body +
             "  }\n  GroupMemoryBarrierWithGroupSync();\n"
             "  for (uint c = t; c < " +
             sl + "; c += " + ws + ") {\n";
        s += partial ? "    if (base + int(c) < " + to_string(cells) + ") {\n" +
                           store + "    }\n"
                     : store;
        s += "  }\n}\n";
        return s;
    }

    // ── path A: atomics, many workgroups, privatized when the bin fits.
    const std::string store = fold_store(
        atom, priv ? "bin[" + cell + "]" : "pc.out_buf.data[" + cell + "]",
        value);
    std::string body = decls + "    " +
                       (guards.empty() ? store
                                       : "if (" + guards + ") " + store) +
                       "\n";

    std::string out;
    if (priv)
        out += "\ngroupshared " + type + " bin[" + to_string(cells) + "];\n";
    out += "\n[shader(\"compute\")]\n[numthreads(" + to_string(workgroup_size) +
           ", 1, 1)]\n"
           "void main(uint3 tid : SV_DispatchThreadID, uint3 lid : "
           "SV_GroupThreadID) {\n"
           "  uint f = tid.x;\n";
    if (priv)
        out += "  for (uint c = lid.x; c < " + to_string(cells) +
               "; c += " + to_string(workgroup_size) + ") bin[c] = " + id +
               ";\n"
               "  GroupMemoryBarrierWithGroupSync();\n";
    out += "  if (f < " + to_string(contrib) + ") {\n" + body + "  }\n";
    if (priv)
        out += "  GroupMemoryBarrierWithGroupSync();\n"
               "  for (uint c = lid.x; c < " +
               to_string(cells) + "; c += " + to_string(workgroup_size) +
               ") " + fold_store(atom, "pc.out_buf.data[c]", "bin[c]") + "\n";
    out += "}\n";
    return out;
}

// The census answers both shape questions, so nothing here re-walks the tree.
consteval std::string kernel(std::meta::info expr, std::string_view identity,
                             const Leaves &l,
                             const std::vector<size_t> &order = {}) {
    const auto op = op_of(expr);
    const auto d = std::meta::dealias(expr);
    // ONE id census for the whole assembly. rank0 and whichever branch runs
    // below both want the free plan; computing it on first use keeps a
    // fold-free elementwise tree, which needs none, paying nothing.
    FreePlan free{};
    bool planned = false;
    auto free_of = [&]() -> const FreePlan & {
        if (!planned) {
            free = free_plan(d);
            if (!order.empty())
                free = ordered_plan(free, order);
            planned = true;
        }
        return free;
    };
    const bool rank0 = is_indexed(d) // a lone subscripted leaf has no alias
                           ? free_of().n == 0
                           : node_extents_of(expr).empty();
    // A scan first: it is neither a contraction nor elementwise, so every
    // test below would mis-route it — and the per-cell kernel at the bottom
    // would render meaningless code rather than fail, the way the
    // epilogue-over-scatter defect did.
    if (const auto sn = scan_node_of_tree(expr); sn != std::meta::info{})
        return scan_kernel(expr, identity, sn, free_of());
    // Before the contraction test: is_contraction is TRUE for a scatter
    // (it carries `summed`), so it would otherwise emit a fold kernel over
    // its first destination.
    if (op != std::meta::info{} && is_placed_op(op))
        return scatter_kernel(expr, identity);
    // An epilogue above a scatter is the same kernel: the elementwise ops
    // wrap the merged accumulator at the store, which is the one place each
    // output cell is written once — where the CPU applies it too.
    if (l.fold_node != std::meta::info{} && is_placed_op(op_of(l.fold_node)))
        return scatter_kernel(expr, identity, l.fold_node);
    if (op != std::meta::info{} && is_contraction(op))
        return rank0 ? contract_all_kernel(expr, identity)
                     : contract_axis_kernel(expr, identity, order);
    if (l.fold_node != std::meta::info{} &&
        !is_placed_op(op_of(l.fold_node))) {
        // The epilogue: elementwise above one fold, the store wraps acc.
        return free_of().n == 0
                   ? epilogue_all_kernel(expr, l.fold_node, identity)
                   : epilogue_axis_kernel(expr, l.fold_node, free_of(),
                                          identity);
    }
    size_t v = 0, s = 0;
    // is_indexed: element_count_of would take an id census of its own, and
    // free_of() has already taken one for exactly this tree.
    std::string out = per_cell_prologue(
        is_indexed(d) ? count_of(free_of()) : element_count_of(expr));
    if (l.has_indexed) {
        // Index-bearing, no fold: the per-cell kernel over the (possibly
        // ordered) free environment. Its coordinates are SIGNED int locals
        // for the reason the fold kernels declare theirs that way — an
        // affine subscript can go negative, and every boundary policy is
        // written to detect that. Off the unsigned dispatch index the
        // detection cannot fire: i - 1 at i == 0 is 4294967295, which
        // wrap only maps correctly when the extent divides 2^32, clamp
        // reads as past the END rather than before the start, and zero's
        // guard can never be false.
        const auto &fp = free_of();
        std::vector<size_t> out_ext;
        for (size_t a = 0; a < fp.n; ++a)
            out_ext.push_back(fp.ext[a]);
        auto k = kernel_coords(fp, {}, {});
        DirectRead direct{"i", {}, out_ext};
        for (size_t a = 0; a < fp.n; ++a)
            direct.id.push_back(fp.id[a]);
        out += k.frees;
        out += "  pc.out_buf.data[i] = ";
        out += render_indexed(std::meta::dealias(expr), slang_style, k.coord,
                              v, s, direct);
    } else {
        out += "  pc.out_buf.data[i] = ";
        out += render_plain(expr, slang_style, "i", v, s);
    }
    out += ";\n}\n";
    return out;
}

// ── assembly ────────────────────────────────────────────────────────────────
// `identity` arrives as a finished literal: a consteval function cannot
// splice its own parameters, so fold_identity<E>() spells it upstream.
//
// The program is the shader and nothing else. Annotating it with the source
// formula meant rendering the whole tree a SECOND time, in a second dialect,
// on every compile — and `formula<E>()` already spells exactly that string
// for a caller who wants it.
consteval std::string gpu_program(std::meta::info expr,
                                  std::string_view identity = "",
                                  const std::vector<size_t> &order = {}) {
    expr = std::meta::dealias(expr);
    auto leaves = leaves_of(expr);
    auto out_t = alias_of(expr, "type");
    std::string s = buffer_structs(leaves, out_t);
    s += fn_helpers(expr, leaves);
    s += push_constants(leaves, out_t);
    s += kernel(expr, identity, leaves, order);
    return s;
}

// ── gpu_source support ──────────────────────────────────────────────────────

template <typename E> consteval std::string fold_identity() {
    using D = std::remove_cvref_t<E>;
    // A scan needs its accumulator seeded the same way, and carries no
    // `summed`, so fold_node_of never finds it.
    constexpr auto sn = scan_node_of_tree(std::meta::dealias(^^D));
    if constexpr (sn != std::meta::info{}) {
        using S = typename[:sn:];
        return gpu_literal(
            S::op_type::op::template identity<typename S::type>());
    } else {
        constexpr auto fn = fold_node_of(std::meta::dealias(^^D));
        if constexpr (fn != std::meta::info{}) {
            using F = typename[:fn:];
            return gpu_literal(
                F::op_type::op::template identity<typename F::type>());
        }
        return "";
    }
}

// One thread owns a whole row's accumulator, so a scan with few rows and a
// long axis runs very nearly serially. Past this it is refused rather than
// emitted slow, naming the kernel that would fix it — the two-level scan
// inside one workgroup, which nothing needs until sort-by-cell exists.
inline constexpr size_t scan_serial_budget = 4096;

template <typename E> consteval bool scan_fits_row_per_thread() {
    using D = std::remove_cvref_t<E>;
    constexpr auto sn = scan_node_of_tree(std::meta::dealias(^^D));
    if constexpr (sn != std::meta::info{}) {
        using S = typename[:sn:];
        const auto free = free_plan(std::meta::dealias(^^D));
        size_t rows = 1, ext = 1;
        for (size_t a = 0; a < free.n; ++a)
            (free.id[a] == S::op_type::scanned ? ext : rows) *= free.ext[a];
        return rows >= workgroup_size || ext <= scan_serial_budget;
    }
    return true;
}

consteval std::string gpu_scan_rows_error(std::meta::info expr) {
    const auto sn = scan_node_of_tree(std::meta::dealias(expr));
    const auto free = free_plan(std::meta::dealias(expr));
    size_t rows = 1, ext = 1;
    for (size_t a = 0; a < free.n; ++a)
        (free.id[a] == scanned_of(op_of(sn)) ? ext : rows) *= free.ext[a];
    return "gpu eval: this scan has only " + to_string(rows) +
           " rows to spread over threads and " + to_string(ext) +
           " elements along the scanned axis, so one thread would walk almost "
           "all of it — the two-level single-workgroup scan that fixes this "
           "is not implemented (docs/scan-plan.md); scan a shorter axis, or "
           "eval(expr) on the CPU";
}

// One workgroup carries a full fold up to the budget; past it, eval refuses
// and names what lifts the limit rather than emitting a slow kernel. The
// gate reads the FOLD subtree — an epilogue above it changes nothing.
template <typename E> consteval bool fold_fits_one_group() {
    using D = std::remove_cvref_t<E>;
    constexpr auto fn = fold_node_of(std::meta::dealias(^^D));
    if constexpr (fn != std::meta::info{}) {
        using F = typename[:fn:];
        return F::rank != 0 || fold_size_of(fn) <= single_group_budget;
    }
    return true;
}

consteval std::string gpu_fold_size_error(std::meta::info expr) {
    return "gpu eval: a full fold over " + to_string(fold_size_of(expr)) +
           " elements exceeds what one workgroup should carry (" +
           to_string(single_group_budget) +
           ") — split-K across workgroups is not implemented; fold an axis "
           "instead, or eval(expr) on the CPU";
}

// ── the gates, from one census ──────────────────────────────────────────────

// Every whole-tree question gpu_source and eval(dev, ·) ask, answered from a
// SINGLE census. Booleans and sizes only: std::meta::info is consteval-only,
// so a variable template holding one would not be a valid constant. Which op
// or type offends is recovered by the message builders, and those run only in
// a discarded branch — on the failure path, where a re-walk costs nothing.
struct GpuGates {
    bool cpu_only = false;   // an op has no Slang intrinsic
    bool bad_type = false;   // an element type does not map
    bool emissible = true;   // every map<f> can lower
    bool streamless = true;  // no rng::Sample stream leaf
    bool fold_fits = true;
    bool scan_fits = true;
    size_t views = 0;
    size_t scalar_bytes = 0;
    size_t push_bytes = 0;
};

// A template, not a plain consteval function: fold_fits/scan_fits splice the
// fold and scan nodes, and a consteval function cannot splice its parameters.
template <typename E> consteval GpuGates compute_gpu_gates() {
    using D = std::remove_cvref_t<E>;
    const auto l = leaves_of(std::meta::dealias(^^D)); // the one census
    GpuGates g;
    g.cpu_only = first_cpu_only_op(^^D) != std::meta::info{};
    g.emissible = tree_gpu_emissible(^^D);
    g.streamless = !l.has_stream;
    g.bad_type =
        first_unmappable_type(l, alias_of(std::meta::dealias(^^D), "type")) !=
        std::meta::info{};
    g.views = l.views.size();
    g.scalar_bytes = scalar_bytes(l);
    g.push_bytes = ((g.scalar_bytes + 7) & ~size_t{7}) + 8 * (1 + g.views);
    g.fold_fits = fold_fits_one_group<D>();
    g.scan_fits = scan_fits_row_per_thread<D>();
    return g;
}

// Memoized per expression type, so the census is taken once however many
// times the gates, the ABI sizes and the push budget are consulted.
template <typename E>
inline constexpr GpuGates gpu_gates = compute_gpu_gates<E>();

} // namespace tensor::detail
