#pragma once

// The math vocabulary — dual-use callable objects, std semantics:
//
//   auto k = eval(math::Sqrt(e * e - m * m));   // formula: "sqrt(…)"
//   double y = math::Sqrt(x);                   // plain std::sqrt
//   auto p = eval(math::Legendre(3u, ct));      // orders broadcast
//
// Real stays real (Sqrt(-1.0) is NaN); complex in → complex out where std
// has the overload, real-only elsewhere. A sym that is a Slang intrinsic
// lowers to the GPU with no opt-in; CpuOnly marks the rest and eval(dev,…)
// rejects them naming the op. Definitions live in Math/*.h, included at
// the bottom.

#include "Expr.h"

namespace tensor {

// The annotated op types (definitions in Math/Ops.h).
namespace ops {

struct Exp;
struct Exp2;
struct Expm1;
struct Log;
struct Log2;
struct Log10;
struct Log1p;
struct Pow;
struct Sqrt;
struct Cbrt;
struct Hypot;
struct Sin;
struct Cos;
struct Tan;
struct Asin;
struct Acos;
struct Atan;
struct Atan2;
struct Sinh;
struct Cosh;
struct Tanh;
struct Asinh;
struct Acosh;
struct Atanh;
struct Erf;
struct Erfc;
struct Tgamma;
struct Lgamma;
struct Floor;
struct Ceil;
struct Round;
struct Trunc;
struct Fmod;
struct Remainder;
struct Abs;
struct Copysign;
struct Fdim;
struct Fmin;
struct Fmax;
struct Fma;
struct Real;
struct Imag;
struct Arg;
struct Norm;
struct Conj;
struct Proj;
struct Beta;
struct Expint;
struct RiemannZeta;
struct CylBesselJ;
struct CylBesselI;
struct CylBesselK;
struct CylNeumann;
struct SphBessel;
struct SphNeumann;
struct Legendre;
struct AssocLegendre;
struct SphLegendre;
struct Hermite;
struct Laguerre;
struct AssocLaguerre;
struct Ellint1;
struct Ellint2;
struct Ellint3;
struct CompEllint1;
struct CompEllint2;
struct CompEllint3;

} // namespace ops

// The callable objects. A scalar argument list is a plain std call; a
// tensor operand builds the op's expression node — the ONE tensor
// spelling (map<math::…> is a compile_fail; map<f> lifts plain functions).
namespace math {

template <typename Op> struct Dual {
    using op = Op;
    template <typename... Cs> static constexpr auto operator()(Cs &&...cs);
};

inline constexpr Dual<ops::Exp> Exp{};
inline constexpr Dual<ops::Exp2> Exp2{};
inline constexpr Dual<ops::Expm1> Expm1{};
inline constexpr Dual<ops::Log> Log{};
inline constexpr Dual<ops::Log2> Log2{};
inline constexpr Dual<ops::Log10> Log10{};
inline constexpr Dual<ops::Log1p> Log1p{};
inline constexpr Dual<ops::Pow> Pow{};
inline constexpr Dual<ops::Sqrt> Sqrt{};
inline constexpr Dual<ops::Cbrt> Cbrt{};
inline constexpr Dual<ops::Hypot> Hypot{};
inline constexpr Dual<ops::Sin> Sin{};
inline constexpr Dual<ops::Cos> Cos{};
inline constexpr Dual<ops::Tan> Tan{};
inline constexpr Dual<ops::Asin> Asin{};
inline constexpr Dual<ops::Acos> Acos{};
inline constexpr Dual<ops::Atan> Atan{};
inline constexpr Dual<ops::Atan2> Atan2{};
inline constexpr Dual<ops::Sinh> Sinh{};
inline constexpr Dual<ops::Cosh> Cosh{};
inline constexpr Dual<ops::Tanh> Tanh{};
inline constexpr Dual<ops::Asinh> Asinh{};
inline constexpr Dual<ops::Acosh> Acosh{};
inline constexpr Dual<ops::Atanh> Atanh{};
inline constexpr Dual<ops::Erf> Erf{};
inline constexpr Dual<ops::Erfc> Erfc{};
inline constexpr Dual<ops::Tgamma> Tgamma{};
inline constexpr Dual<ops::Lgamma> Lgamma{};
inline constexpr Dual<ops::Floor> Floor{};
inline constexpr Dual<ops::Ceil> Ceil{};
inline constexpr Dual<ops::Round> Round{};
inline constexpr Dual<ops::Trunc> Trunc{};
inline constexpr Dual<ops::Fmod> Fmod{};
inline constexpr Dual<ops::Remainder> Remainder{};
inline constexpr Dual<ops::Abs> Abs{};
inline constexpr Dual<ops::Copysign> Copysign{};
inline constexpr Dual<ops::Fdim> Fdim{};
inline constexpr Dual<ops::Fmin> Fmin{};
inline constexpr Dual<ops::Fmax> Fmax{};
inline constexpr Dual<ops::Fma> Fma{};
inline constexpr Dual<ops::Real> Real{};
inline constexpr Dual<ops::Imag> Imag{};
inline constexpr Dual<ops::Arg> Arg{};
inline constexpr Dual<ops::Norm> Norm{};
inline constexpr Dual<ops::Conj> Conj{};
inline constexpr Dual<ops::Proj> Proj{};
inline constexpr Dual<ops::Beta> Beta{};
inline constexpr Dual<ops::Expint> Expint{};
inline constexpr Dual<ops::RiemannZeta> RiemannZeta{};
inline constexpr Dual<ops::CylBesselJ> CylBesselJ{};
inline constexpr Dual<ops::CylBesselI> CylBesselI{};
inline constexpr Dual<ops::CylBesselK> CylBesselK{};
inline constexpr Dual<ops::CylNeumann> CylNeumann{};
inline constexpr Dual<ops::SphBessel> SphBessel{};
inline constexpr Dual<ops::SphNeumann> SphNeumann{};
inline constexpr Dual<ops::Legendre> Legendre{};
inline constexpr Dual<ops::AssocLegendre> AssocLegendre{};
inline constexpr Dual<ops::SphLegendre> SphLegendre{};
inline constexpr Dual<ops::Hermite> Hermite{};
inline constexpr Dual<ops::Laguerre> Laguerre{};
inline constexpr Dual<ops::AssocLaguerre> AssocLaguerre{};
inline constexpr Dual<ops::Ellint1> Ellint1{};
inline constexpr Dual<ops::Ellint2> Ellint2{};
inline constexpr Dual<ops::Ellint3> Ellint3{};
inline constexpr Dual<ops::CompEllint1> CompEllint1{};
inline constexpr Dual<ops::CompEllint2> CompEllint2{};
inline constexpr Dual<ops::CompEllint3> CompEllint3{};

} // namespace math

// Which of NB equal bins over [lo, hi) holds each x — the world→grid map
// floor((x - lo) * (NB / (hi - lo))), composed from existing ops: no new
// node, no new leaf, and the scale is ONE scalar computed at build time.
// Floor, not trunc, so a value below lo is a NEGATIVE bin; the use site
// says what an out-of-range bin does — a write policy (drop<NB>/clamp<NB>/
// wrap<NB>) on a scatter destination, clamp(…) on a read. x == hi may land
// at NB or NB - 1 (one float rounding); under clamp that is the last bin
// either way.
template <size_t NB, Operand X, typename T>
constexpr auto bins(X &&x, const T &lo, const T &hi);

} // namespace tensor

// The definitions.
#include "Math/Dual.h" // IWYU pragma: export
#include "Math/Ops.h"  // IWYU pragma: export
#include "Math/Bins.h" // IWYU pragma: export

// The ops lint again: Expr.h's sweep saw these ops only forward-declared.
#include "detail/OpsCheck.h"
