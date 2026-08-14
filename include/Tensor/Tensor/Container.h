#pragma once

// Tensor member bodies: element access, views, spans, and the residency
// sync hooks.

#include "../Tensor.h"

#include <cstddef>
#include <mdspan>

namespace tensor {

template <typename T, size_t... Extents>
T &Tensor<T, Extents...>::operator[](Index<rank> idx) {
    return span()[idx];
}
template <typename T, size_t... Extents>
const T &Tensor<T, Extents...>::operator[](Index<rank> idx) const {
    return span()[idx];
}

template <typename T, size_t... Extents>
template <detail::SubscriptArg... Sub>
    requires(detail::SubscriptTerm<Sub> || ...)
constexpr auto Tensor<T, Extents...>::operator[](Sub... sub) const {
    return detail::make_indexed<TensorView<const T, Extents...>, extents_type,
                                Sub...>(view(), sub...);
}

template <typename T, size_t... Extents>
TensorView<T, Extents...> Tensor<T, Extents...>::view() {
    drop_shadow();
    return {data_.get(), &shadow_};
}
template <typename T, size_t... Extents>
TensorView<const T, Extents...> Tensor<T, Extents...>::view() const {
    return {data_.get(), &shadow_};
}

template <typename T, size_t... Extents>
typename Tensor<T, Extents...>::span_type Tensor<T, Extents...>::span() {
    drop_shadow();
    return span_type{data_.get()};
}
template <typename T, size_t... Extents>
std::mdspan<const T, typename Tensor<T, Extents...>::extents_type>
Tensor<T, Extents...>::span() const {
    sync_host();
    return std::mdspan<const T, extents_type>{data_.get()};
}

template <typename T, size_t... Extents> T *Tensor<T, Extents...>::data() {
    drop_shadow();
    return data_.get();
}
template <typename T, size_t... Extents>
const T *Tensor<T, Extents...>::data() const {
    sync_host();
    return data_.get();
}

template <typename T, size_t... Extents>
constexpr size_t Tensor<T, Extents...>::size() {
    return element_count;
}
template <typename T, size_t... Extents>
constexpr size_t Tensor<T, Extents...>::extent(size_t r) {
    return extents_type::static_extent(r);
}

template <typename T, size_t... Extents>
void Tensor<T, Extents...>::sync_host() const {
    if (shadow_.storage && !shadow_.host_valid) {
        shadow_.storage->download(data_.get(), byte_size);
        shadow_.host_valid = true;
    }
}
template <typename T, size_t... Extents>
void Tensor<T, Extents...>::drop_shadow() {
    sync_host();
    shadow_.storage.reset();
    shadow_.host_valid = true;
}

} // namespace tensor
