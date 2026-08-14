// The ops lint's tripwire — deliberately NO #pragma once: every surface
// header that defines ops includes this last, and each inclusion re-sweeps
// the ops:: namespace as completed so far.

#include "Ops.h"

namespace tensor::detail {

static_assert(ops_missing_symbol().empty(),
              "ops missing a [[=detail::sym(\"...\")]] annotation: " +
                  ops_missing_symbol());

} // namespace tensor::detail
