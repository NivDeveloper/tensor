#pragma once

// Generic consteval / reflection utilities; no tensor knowledge. The
// reflection layer — context-free helpers live in Utils.h below it.

#include "Utils.h"

#include <cstddef>
#include <meta>
#include <string>
#include <string_view>
#include <vector>

namespace tensor::detail {

// True iff `type` is a specialization of the class template `tmpl` — the
// generic replacement for one hand-written is_xxx<> trait per template.
consteval bool is_specialization_of(std::meta::info type,
                                    std::meta::info tmpl) {
    type = std::meta::dealias(type);
    return std::meta::has_template_arguments(type) &&
           std::meta::template_of(type) == tmpl;
}

// A nested type alias looked up by name, or {} if absent — protocol queries
// (op_of, element types, extents) are all this one primitive.
consteval std::meta::info alias_of(std::meta::info node,
                                   std::string_view name) {
    for (auto m :
         std::meta::members_of(node, std::meta::access_context::current()))
        if (std::meta::is_type_alias(m) && std::meta::has_identifier(m) &&
            std::meta::identifier_of(m) == name)
            return std::meta::dealias(m);
    return {};
}

// The display string of a (dealiased) type — how diagnostics name types.
consteval std::string type_name(std::meta::info t) {
    return std::string(std::meta::display_string_of(std::meta::dealias(t)));
}

// A member variable looked up by name — the sibling of alias_of for the
// protocols keyed on a static data member.
consteval bool has_static_member(std::meta::info type, std::string_view name) {
    for (auto m :
         std::meta::members_of(type, std::meta::access_context::current()))
        if (std::meta::is_variable(m) && std::meta::has_identifier(m) &&
            std::meta::identifier_of(m) == name)
            return true;
    return false;
}

// The template arguments of `t` from index `from` on, extracted as T values.
template <typename T>
consteval std::vector<T> args_of(std::meta::info t, size_t from) {
    auto args = std::meta::template_arguments_of(std::meta::dealias(t));
    std::vector<T> out;
    for (size_t i = from; i < args.size(); ++i)
        out.push_back(std::meta::extract<T>(args[i]));
    return out;
}

// The entity behind a reference NTTP — GCC 16 rejects ^^F on a template
// parameter, but Carrier<F>'s template argument IS the bound entity.
template <auto &F> struct Carrier {};
template <auto &F> consteval std::meta::info entity_of() {
    return std::meta::template_arguments_of(^^Carrier<F>)[0];
}

} // namespace tensor::detail
