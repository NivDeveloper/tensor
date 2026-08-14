#pragma once

// eval(dev, expr): compile, upload, dispatch, read back — CPU eval's result
// semantics, one round trip. The ONE header that names gpud; everything
// crosses through gpud::Device's virtual interface.
//
// Reached through <Tensor/Gpu.h>, which includes this under the opt-in —
// users include the surface, never this file. Including it directly still
// works and still names the flag when the opt-in is missing.
//
// The guard precedes every #include so the failure names the flag.
#ifndef TENSOR_GPU_ENABLED
#error                                                                         \
    "GPU evaluation is opt-in: configure with -DTENSOR_ENABLE_GPU=ON and link tensor::gpu"
#endif

#include "../Gpu.h"
#include "../Tensor.h"

#include <gpud/Device.h>

#include <array>
#include <cstddef>
#include <cstring>
#include <memory>
#include <meta>
#include <span>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <utility>

namespace tensor {

// The one DeviceStorage implementation: a gpud buffer parked on the device
// that filled it. download() is gpud's ordered read, so a stale host syncs
// after any chain of dispatches.
struct GpuStorage final : DeviceStorage {
    gpud::Device *dev;
    gpud::Buffer buf;
    GpuStorage(gpud::Device *d, gpud::Buffer b)
        : dev(d), buf(std::move(b)) {}
    void download(void *dst, size_t bytes) override {
        dev->read(buf, dst, bytes);
    }
};

template <auto... Order, AnyExpr E>
eval_return_t<E, Order...> eval(gpud::Device &dev, const E &e) {
    using D = std::remove_cvref_t<E>;
    using Result = eval_return_t<E, Order...>;

    constexpr bool scalar_result = detail::rank_of<D>() == 0;
    static_assert(sizeof...(Order) == 0 || detail::index_bearing_v<D>,
                  detail::order_on_plain_error());
    static_assert(
        (detail::is_placeholder_v<std::remove_cvref_t<decltype(Order)>> &&
         ...),
        "eval's order takes the index placeholders (i, j, ...)");
    static constexpr auto order_check = [] {
        if constexpr (sizeof...(Order) == 0)
            return detail::index_slots;
        else
            return detail::order_mismatch(
                detail::free_plan(std::meta::dealias(^^D)),
                {detail::order_id<Order>()...});
    }();
    static_assert(order_check == detail::index_slots,
                  detail::order_error(order_check));

    // The tree gates fire inside gpu_source; the push budget is about
    // BINDING the program, so it lives here.
    static_assert(detail::push_bytes_of(^^D) <= detail::push_budget,
                  detail::gpu_push_error(^^D));

    if (const auto d = dev.dialect(); d != "slang-vulkan" && d != "mock")
        throw std::runtime_error(
            "tensor: gpu eval emits Slang, but the device dialect is \"" +
            std::string(d) + '"');

    // one stable source address per expression type: gpud's identity-keyed
    // memoization makes the type system the kernel registry
    const gpud::Kernel &kernel = dev.compile(gpu_source<E, Order...>());

    Result out{};
    constexpr size_t out_bytes = [] {
        if constexpr (scalar_result)
            return sizeof(Result);
        else
            return Result::byte_size;
    }();
    gpud::Buffer out_buf = dev.alloc(out_bytes);

    // One walk fills the positional ABI, in the DFS order the shader
    // numbered the leaves. Pointer-equal leaves (a stencil's five reads of
    // u) share ONE upload — the same buffer bound into each of their slots;
    // the program and its slot count are untouched.
    constexpr size_t n_views = detail::view_count_of(^^D);
    constexpr size_t n_scalar_bytes = detail::scalar_bytes_of(^^D);
    std::array<gpud::Buffer, n_views> inputs;
    std::array<gpud::Buffer *, 1 + n_views> buffers;
    buffers[0] = &out_buf;
    std::array<std::pair<const void *, size_t>, n_views> seen;
    std::array<std::byte, n_scalar_bytes> scalars{}; // {}: padding stays 0
    size_t view = 0, off = 0;
    for_each_leaf(e, [&](const auto &leaf) {
        using L = std::remove_cvref_t<decltype(leaf)>;
        if constexpr (detail::is_broadcast_scalar_v<L>) {
            off = (off + alignof(L) - 1) & ~(alignof(L) - 1);
            std::memcpy(scalars.data() + off, &leaf, sizeof(L));
            off += sizeof(L);
        } else if constexpr (detail::is_generator_v<L>) {
            // No buffer: the kernel computes the value. What packs is what
            // the census declared — a sampler's key halves as uint, every
            // other kind's parameters in its own element type.
            auto pack = [&](const auto &v) {
                using P = std::remove_cvref_t<decltype(v)>;
                off = (off + alignof(P) - 1) & ~(alignof(P) - 1);
                std::memcpy(scalars.data() + off, &v, sizeof(P));
                off += sizeof(P);
            };
            if constexpr (detail::gen_is_sampler(L::kind)) {
                pack(unsigned(leaf.key));
                pack(unsigned(leaf.key >> 32));
            } else {
                const typename L::type params[]{leaf.a, leaf.b};
                for (size_t k = 0; k < detail::gen_params(L::kind); ++k)
                    pack(params[k]);
            }
        } else { // a tensor leaf: a view, or a bare Tensor as the root
            constexpr size_t bytes =
                detail::element_count_of(^^L) * sizeof(typename L::type);
            const auto v = [&] {
                if constexpr (requires { leaf.view(); })
                    return leaf.view();
                else
                    return leaf;
            }();
            gpud::Buffer *bound = nullptr;
            if (v.shadow) { // the leaf's Tensor may already be resident
                auto *g =
                    dynamic_cast<GpuStorage *>(v.shadow->storage.get());
                if (g && g->dev == &dev) {
                    bound = &g->buf; // cached on THIS device: bind
                } else if (v.shadow->storage) {
                    // resident elsewhere: settle the host copy through the
                    // OLD storage, then follow this device
                    if (!v.shadow->host_valid) {
                        v.shadow->storage->download(
                            const_cast<void *>(
                                static_cast<const void *>(v.data)),
                            bytes);
                        v.shadow->host_valid = true;
                    }
                    v.shadow->storage.reset();
                }
            }
            if (!bound)
                for (size_t q = 0; q < view; ++q)
                    if (seen[q].first == v.data && seen[q].second == bytes)
                        bound = buffers[1 + q];
            if (!bound) {
                if (v.shadow) { // cache the upload on the owning Tensor
                    auto g = std::make_unique<GpuStorage>(&dev,
                                                          dev.alloc(bytes));
                    dev.write(g->buf, v.data, bytes);
                    bound = &g->buf;
                    v.shadow->storage = std::move(g);
                } else { // a raw view: eval-local upload, freed at return
                    inputs[view] = dev.alloc(bytes);
                    dev.write(inputs[view], v.data, bytes);
                    bound = &inputs[view];
                }
            }
            seen[view] = {v.data, bytes};
            buffers[1 + view] = bound;
            ++view;
        }
    });

    const size_t groups = [] {
        if constexpr (scalar_result)
            return size_t{1}; // the tree kernel is one workgroup
        else
            return (Result::element_count + detail::workgroup_size - 1) /
                   detail::workgroup_size;
    }();
    dev.run(kernel, groups, {scalars.data(), scalars.size()}, buffers);
    if constexpr (scalar_result) {
        dev.read(out_buf, &out, out_bytes); // no object to park it on
    } else {
        // The result stays on the device; the host syncs on first access.
        // Dropping the eval-local buffers right after run() is safe under
        // gpud's lifetime contract: a Buffer may be destroyed while work
        // using it is still queued, and the backend keeps the memory
        // alive as long as anything queued still needs it.
        auto *slot = std::as_const(out).view().shadow;
        slot->storage =
            std::make_unique<GpuStorage>(&dev, std::move(out_buf));
        slot->host_valid = false;
    }
    return out;
}

} // namespace tensor
