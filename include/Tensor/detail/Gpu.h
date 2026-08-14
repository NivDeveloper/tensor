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
        for (auto m : maps_of(t))
            if (map_padded(m)) {
                l.scalars.push_back(alias_of(t, "type"));
                break;
            }
        collect(indexed_operand_of(t), l);
    }
    else if (auto op = op_of(t); op == std::meta::info{}) // tensor leaf
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

consteval size_t view_count_of(std::meta::info expr) {
    return leaves_of(std::meta::dealias(expr)).views.size();
}

// PC scalar-section bytes — the same natural-alignment rule eval's leaf
// walk packs by.
consteval size_t scalar_bytes_of(std::meta::info expr) {
    size_t off = 0;
    for (auto t : leaves_of(std::meta::dealias(expr)).scalars) {
        size_t a = std::meta::alignment_of(t);
        off = (off + a - 1) & ~(a - 1);
        off += std::meta::size_of(t);
    }
    return off;
}

// Vulkan's guaranteed push-constant minimum: 8-aligned scalars + one 8-byte
// address per buffer.
inline constexpr size_t push_budget = 128;

consteval size_t push_bytes_of(std::meta::info expr) {
    return ((scalar_bytes_of(expr) + 7) & ~size_t{7}) +
           8 * (1 + view_count_of(expr));
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
consteval std::meta::info first_unmappable_type(std::meta::info expr) {
    expr = std::meta::dealias(expr);
    if (auto out = alias_of(expr, "type"); !type_gpu_mappable(out))
        return out;
    auto l = leaves_of(expr);
    for (auto t : l.views)
        if (!type_gpu_mappable(t))
            return t;
    for (auto t : l.scalars)
        if (!type_gpu_mappable(t))
            return t;
    return {};
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

// The comment exists to be read. Past this many nodes the formula is one
// unreadable multi-kilobyte line, and rendering the tree a second time to
// produce it is a third of the program's whole constant-evaluation budget —
// so a large tree gets its size and a pointer to formula<E>() instead.
inline constexpr size_t formula_comment_budget = 64;

consteval std::string formula_comment(std::meta::info expr, const Leaves &l) {
    std::string out = "// ";
    if (l.nodes > formula_comment_budget) {
        append_number(out, l.nodes);
        out += "-node expression; formula omitted — formula<E>() spells it\n";
        return out;
    }
    size_t v = 0, s = 0;
    out += l.has_indexed ? render(expr, formula_style, "i", v, s)
                         : render_plain(expr, formula_style, "i", v, s);
    out += "\n";
    return out;
}

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

consteval std::string per_cell_prologue(std::meta::info expr) {
    return "\n[shader(\"compute\")]\n[numthreads(" + to_string(workgroup_size) +
           ", 1, 1)]\n"
           "void main(uint3 tid : SV_DispatchThreadID) {\n"
           "  uint i = tid.x;\n"
           "  if (i >= " +
           to_string(element_count_of(expr)) + ") return;\n";
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
    const auto scan = index_scan(summand);
    const auto sym = symbol_of(fold_op_of(op));
    const auto type = std::string(gpu_type(alias_of(expr, "type")));

    // The free ids in output order — eval<Order…> permutes the layout.
    std::vector<size_t> free_ids;
    for (auto id : scan.order)
        if (!std::ranges::contains(summed, id))
            free_ids.push_back(id);
    if (!order.empty())
        free_ids = order;
    std::vector<size_t> out_ext;
    for (auto id : free_ids)
        out_ext.push_back(scan.pinned(id));
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
                 to_string(scan.pinned(summed[d])) + "; ++" + jn + ")\n";
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
    const auto scan = index_scan(summand);
    const auto sym = symbol_of(fold_op_of(op));
    const auto type = std::string(gpu_type(alias_of(expr, "type")));

    std::vector<size_t> sext;
    for (auto a : summed)
        sext.push_back(scan.pinned(a));
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
        auto ai =
            decorated_index(maps_of(t), node_extents_of(operand), coord);
        auto fill = fill_spelling(t, style, scalars);
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
    const auto scan = index_scan(summand);
    const auto sym = symbol_of(fold_op_of(fop));
    const auto ftype = std::string(gpu_type(alias_of(fold, "type")));
    std::vector<size_t> sext;
    for (auto a : summed)
        sext.push_back(scan.pinned(a));
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
    const auto scan = index_scan(summand);
    const auto sym = symbol_of(fold_op_of(fop));
    const auto ftype = std::string(gpu_type(alias_of(fold, "type")));
    std::vector<size_t> sext;
    for (auto a : summed)
        sext.push_back(scan.pinned(a));
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

// The census answers both shape questions, so nothing here re-walks the tree.
consteval std::string kernel(std::meta::info expr, std::string_view identity,
                             const Leaves &l,
                             const std::vector<size_t> &order = {}) {
    const auto op = op_of(expr);
    const bool rank0 = [&] {
        auto d = std::meta::dealias(expr);
        if (is_indexed(d)) // a lone subscripted leaf carries no alias
            return free_plan(d).n == 0;
        return node_extents_of(expr).empty();
    }();
    if (op != std::meta::info{} && is_contraction(op))
        return rank0 ? contract_all_kernel(expr, identity)
                     : contract_axis_kernel(expr, identity, order);
    if (l.fold_node != std::meta::info{}) {
        // The epilogue: elementwise above one fold, the store wraps acc.
        auto free = free_plan(std::meta::dealias(expr));
        if (!order.empty())
            free = ordered_plan(free, order);
        return free.n == 0
                   ? epilogue_all_kernel(expr, l.fold_node, identity)
                   : epilogue_axis_kernel(expr, l.fold_node, free, identity);
    }
    size_t v = 0, s = 0;
    std::string out = per_cell_prologue(expr);
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
        auto free = free_plan(std::meta::dealias(expr));
        if (!order.empty())
            free = ordered_plan(free, order);
        std::vector<size_t> out_ext;
        for (size_t a = 0; a < free.n; ++a)
            out_ext.push_back(free.ext[a]);
        auto k = kernel_coords(free, {}, {});
        DirectRead direct{"i", {}, out_ext};
        for (size_t a = 0; a < free.n; ++a)
            direct.id.push_back(free.id[a]);
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
consteval std::string gpu_program(std::meta::info expr,
                                  std::string_view identity = "",
                                  const std::vector<size_t> &order = {}) {
    expr = std::meta::dealias(expr);
    auto leaves = leaves_of(expr);
    auto out_t = alias_of(expr, "type");
    std::string s = formula_comment(expr, leaves);
    s += buffer_structs(leaves, out_t);
    s += fn_helpers(expr, leaves);
    s += push_constants(leaves, out_t);
    s += kernel(expr, identity, leaves, order);
    return s;
}

// ── gpu_source support ──────────────────────────────────────────────────────

template <typename E> consteval std::string fold_identity() {
    using D = std::remove_cvref_t<E>;
    constexpr auto fn = fold_node_of(std::meta::dealias(^^D));
    if constexpr (fn != std::meta::info{}) {
        using F = typename[:fn:];
        return gpu_literal(
            F::op_type::op::template identity<typename F::type>());
    }
    return "";
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

} // namespace tensor::detail
