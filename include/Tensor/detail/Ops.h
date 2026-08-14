#pragma once

// The ops lint — the sweep behind detail/OpsCheck.h's static_assert.

#include "Sym.h"

#include <meta>
#include <string>

namespace tensor::ops {} // the swept namespace, before any op exists

namespace tensor::detail {

consteval std::string ops_missing_symbol() {
    std::string missing;
    for (auto m :
         std::meta::members_of(^^ops, std::meta::access_context::current()))
        // is_type before is_class_type (throws on non-types); incomplete
        // types skipped (a surface entry header forward-declares ops)
        if (std::meta::is_type(m) && std::meta::is_class_type(m) &&
            std::meta::is_complete_type(m) &&
            std::meta::annotations_of_with_type(m, ^^Symbol).empty()) {
            missing += std::meta::identifier_of(m);
            missing += ' ';
        }
    return missing;
}

} // namespace tensor::detail
