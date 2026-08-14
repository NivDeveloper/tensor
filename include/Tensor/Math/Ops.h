#pragma once

// The annotated math op types — std semantics via `using std::f; return
// f(x);` (builtins and complex through std, user element types via ADL).

#include "../Math.h"

#include <cmath>
#include <complex>
#include <cstdlib>
#include <utility>

namespace tensor {

// clang-format off: annotations predate clang-format's parser
namespace ops {

// `using std::f; return f(x);` — builtins and complex via std, user
// element types via ADL.

// ── exponentials and logarithms ─────────────────────────────────────────────
struct [[=detail::sym("exp")]] Exp {
    static constexpr auto operator()(auto x) {
        using std::exp;
        return exp(x);
    }
};
struct [[=detail::sym("exp2")]] Exp2 {
    static constexpr auto operator()(auto x) {
        using std::exp2;
        return exp2(x);
    }
};
struct [[=detail::sym("expm1"), =detail::CpuOnly{}]] Expm1 {
    static constexpr auto operator()(auto x) {
        using std::expm1;
        return expm1(x);
    }
};
struct [[=detail::sym("log")]] Log {
    static constexpr auto operator()(auto x) {
        using std::log;
        return log(x);
    }
};
struct [[=detail::sym("log2")]] Log2 {
    static constexpr auto operator()(auto x) {
        using std::log2;
        return log2(x);
    }
};
struct [[=detail::sym("log10")]] Log10 {
    static constexpr auto operator()(auto x) {
        using std::log10;
        return log10(x);
    }
};
struct [[=detail::sym("log1p"), =detail::CpuOnly{}]] Log1p {
    static constexpr auto operator()(auto x) {
        using std::log1p;
        return log1p(x);
    }
};

// ── powers and roots ────────────────────────────────────────────────────────
struct [[=detail::sym("pow")]] Pow {
    static constexpr auto operator()(auto a, auto b) {
        using std::pow;
        return pow(a, b);
    }
};
struct [[=detail::sym("sqrt")]] Sqrt {
    static constexpr auto operator()(auto x) {
        using std::sqrt;
        return sqrt(x);
    }
};
struct [[=detail::sym("cbrt"), =detail::CpuOnly{}]] Cbrt {
    static constexpr auto operator()(auto x) {
        using std::cbrt;
        return cbrt(x);
    }
};
struct [[=detail::sym("hypot"), =detail::CpuOnly{}]] Hypot {
    static constexpr auto operator()(auto x, auto y) {
        using std::hypot;
        return hypot(x, y);
    }
    static constexpr auto operator()(auto x, auto y, auto z) {
        using std::hypot;
        return hypot(x, y, z);
    }
};

// ── trigonometric ───────────────────────────────────────────────────────────
struct [[=detail::sym("sin")]] Sin {
    static constexpr auto operator()(auto x) {
        using std::sin;
        return sin(x);
    }
};
struct [[=detail::sym("cos")]] Cos {
    static constexpr auto operator()(auto x) {
        using std::cos;
        return cos(x);
    }
};
struct [[=detail::sym("tan")]] Tan {
    static constexpr auto operator()(auto x) {
        using std::tan;
        return tan(x);
    }
};
struct [[=detail::sym("asin")]] Asin {
    static constexpr auto operator()(auto x) {
        using std::asin;
        return asin(x);
    }
};
struct [[=detail::sym("acos")]] Acos {
    static constexpr auto operator()(auto x) {
        using std::acos;
        return acos(x);
    }
};
struct [[=detail::sym("atan")]] Atan {
    static constexpr auto operator()(auto x) {
        using std::atan;
        return atan(x);
    }
};
struct [[=detail::sym("atan2")]] Atan2 {
    static constexpr auto operator()(auto a, auto b) {
        using std::atan2;
        return atan2(a, b);
    }
};

// ── hyperbolic ──────────────────────────────────────────────────────────────
struct [[=detail::sym("sinh")]] Sinh {
    static constexpr auto operator()(auto x) {
        using std::sinh;
        return sinh(x);
    }
};
struct [[=detail::sym("cosh")]] Cosh {
    static constexpr auto operator()(auto x) {
        using std::cosh;
        return cosh(x);
    }
};
struct [[=detail::sym("tanh")]] Tanh {
    static constexpr auto operator()(auto x) {
        using std::tanh;
        return tanh(x);
    }
};
struct [[=detail::sym("asinh")]] Asinh {
    static constexpr auto operator()(auto x) {
        using std::asinh;
        return asinh(x);
    }
};
struct [[=detail::sym("acosh")]] Acosh {
    static constexpr auto operator()(auto x) {
        using std::acosh;
        return acosh(x);
    }
};
struct [[=detail::sym("atanh")]] Atanh {
    static constexpr auto operator()(auto x) {
        using std::atanh;
        return atanh(x);
    }
};

// ── error functions and gamma ───────────────────────────────────────────────
struct [[=detail::sym("erf"), =detail::CpuOnly{}]] Erf {
    static constexpr auto operator()(auto x) {
        using std::erf;
        return erf(x);
    }
};
struct [[=detail::sym("erfc"), =detail::CpuOnly{}]] Erfc {
    static constexpr auto operator()(auto x) {
        using std::erfc;
        return erfc(x);
    }
};
struct [[=detail::sym("tgamma"), =detail::CpuOnly{}]] Tgamma {
    static constexpr auto operator()(auto x) {
        using std::tgamma;
        return tgamma(x);
    }
};
struct [[=detail::sym("lgamma"), =detail::CpuOnly{}]] Lgamma {
    static constexpr auto operator()(auto x) {
        using std::lgamma;
        return lgamma(x);
    }
};

// ── rounding and remainders ─────────────────────────────────────────────────
struct [[=detail::sym("floor")]] Floor {
    static constexpr auto operator()(auto x) {
        using std::floor;
        return floor(x);
    }
};
struct [[=detail::sym("ceil")]] Ceil {
    static constexpr auto operator()(auto x) {
        using std::ceil;
        return ceil(x);
    }
};
struct [[=detail::sym("round")]] Round {
    static constexpr auto operator()(auto x) {
        using std::round;
        return round(x);
    }
};
struct [[=detail::sym("trunc")]] Trunc {
    static constexpr auto operator()(auto x) {
        using std::trunc;
        return trunc(x);
    }
};
struct [[=detail::sym("fmod")]] Fmod {
    static constexpr auto operator()(auto a, auto b) {
        using std::fmod;
        return fmod(a, b);
    }
};
struct [[=detail::sym("remainder"), =detail::CpuOnly{}]] Remainder {
    static constexpr auto operator()(auto a, auto b) {
        using std::remainder;
        return remainder(a, b);
    }
};

// ── magnitude, sign, extrema ────────────────────────────────────────────────
struct [[=detail::sym("abs")]] Abs {
    static constexpr auto operator()(auto x) {
        using std::abs;
        return abs(x);
    }
};
struct [[=detail::sym("copysign")]] Copysign {
    static constexpr auto operator()(auto a, auto b) {
        using std::copysign;
        return copysign(a, b);
    }
};
struct [[=detail::sym("fdim")]] Fdim {
    static constexpr auto operator()(auto a, auto b) {
        using std::fdim;
        return fdim(a, b);
    }
};
struct [[=detail::sym("min")]] Fmin {
    static constexpr auto operator()(auto a, auto b) {
        using std::fmin;
        return fmin(a, b);
    }
};
struct [[=detail::sym("max")]] Fmax {
    static constexpr auto operator()(auto a, auto b) {
        using std::fmax;
        return fmax(a, b);
    }
};
struct [[=detail::sym("fma")]] Fma {
    static constexpr auto operator()(auto a, auto b, auto c) {
        using std::fma;
        return fma(a, b, c);
    }
};

// ── complex parts and companions (<complex>; CPU-only) ──────────────────────
// Total on reals via std's additional overloads; Conj/Proj return complex
// even then, as std does.
struct [[=detail::sym("real"), =detail::CpuOnly{}]] Real {
    static constexpr auto operator()(auto z) {
        using std::real;
        return real(z);
    }
};
struct [[=detail::sym("imag"), =detail::CpuOnly{}]] Imag {
    static constexpr auto operator()(auto z) {
        using std::imag;
        return imag(z);
    }
};
struct [[=detail::sym("arg"), =detail::CpuOnly{}]] Arg {
    static constexpr auto operator()(auto z) {
        using std::arg;
        return arg(z);
    }
};
struct [[=detail::sym("norm"), =detail::CpuOnly{}]] Norm {
    static constexpr auto operator()(auto z) {
        using std::norm;
        return norm(z);
    }
};
struct [[=detail::sym("conj"), =detail::CpuOnly{}]] Conj {
    static constexpr auto operator()(auto z) {
        using std::conj;
        return conj(z);
    }
};
struct [[=detail::sym("proj"), =detail::CpuOnly{}]] Proj {
    static constexpr auto operator()(auto z) {
        using std::proj;
        return proj(z);
    }
};

// ── the C++17 special functions (all CPU-only) ──────────────────────────────
struct [[=detail::sym("beta"), =detail::CpuOnly{}]] Beta {
    static constexpr auto operator()(auto a, auto b) {
        using std::beta;
        return beta(a, b);
    }
};
struct [[=detail::sym("expint"), =detail::CpuOnly{}]] Expint {
    static constexpr auto operator()(auto x) {
        using std::expint;
        return expint(x);
    }
};
struct [[=detail::sym("riemann_zeta"), =detail::CpuOnly{}]] RiemannZeta {
    static constexpr auto operator()(auto x) {
        using std::riemann_zeta;
        return riemann_zeta(x);
    }
};
struct [[=detail::sym("cyl_bessel_j"), =detail::CpuOnly{}]] CylBesselJ {
    static constexpr auto operator()(auto nu, auto x) {
        using std::cyl_bessel_j;
        return cyl_bessel_j(nu, x);
    }
};
struct [[=detail::sym("cyl_bessel_i"), =detail::CpuOnly{}]] CylBesselI {
    static constexpr auto operator()(auto nu, auto x) {
        using std::cyl_bessel_i;
        return cyl_bessel_i(nu, x);
    }
};
struct [[=detail::sym("cyl_bessel_k"), =detail::CpuOnly{}]] CylBesselK {
    static constexpr auto operator()(auto nu, auto x) {
        using std::cyl_bessel_k;
        return cyl_bessel_k(nu, x);
    }
};
struct [[=detail::sym("cyl_neumann"), =detail::CpuOnly{}]] CylNeumann {
    static constexpr auto operator()(auto nu, auto x) {
        using std::cyl_neumann;
        return cyl_neumann(nu, x);
    }
};
struct [[=detail::sym("sph_bessel"), =detail::CpuOnly{}]] SphBessel {
    static constexpr auto operator()(auto n, auto x) {
        using std::sph_bessel;
        return sph_bessel(n, x);
    }
};
struct [[=detail::sym("sph_neumann"), =detail::CpuOnly{}]] SphNeumann {
    static constexpr auto operator()(auto n, auto x) {
        using std::sph_neumann;
        return sph_neumann(n, x);
    }
};
struct [[=detail::sym("legendre"), =detail::CpuOnly{}]] Legendre {
    static constexpr auto operator()(auto l, auto x) {
        using std::legendre;
        return legendre(l, x);
    }
};
struct [[=detail::sym("assoc_legendre"), =detail::CpuOnly{}]] AssocLegendre {
    static constexpr auto operator()(auto l, auto m, auto x) {
        using std::assoc_legendre;
        return assoc_legendre(l, m, x);
    }
};
struct [[=detail::sym("sph_legendre"), =detail::CpuOnly{}]] SphLegendre {
    static constexpr auto operator()(auto l, auto m, auto theta) {
        using std::sph_legendre;
        return sph_legendre(l, m, theta);
    }
};
struct [[=detail::sym("hermite"), =detail::CpuOnly{}]] Hermite {
    static constexpr auto operator()(auto n, auto x) {
        using std::hermite;
        return hermite(n, x);
    }
};
struct [[=detail::sym("laguerre"), =detail::CpuOnly{}]] Laguerre {
    static constexpr auto operator()(auto n, auto x) {
        using std::laguerre;
        return laguerre(n, x);
    }
};
struct [[=detail::sym("assoc_laguerre"), =detail::CpuOnly{}]] AssocLaguerre {
    static constexpr auto operator()(auto n, auto m, auto x) {
        using std::assoc_laguerre;
        return assoc_laguerre(n, m, x);
    }
};
struct [[=detail::sym("ellint_1"), =detail::CpuOnly{}]] Ellint1 {
    static constexpr auto operator()(auto k, auto phi) {
        using std::ellint_1;
        return ellint_1(k, phi);
    }
};
struct [[=detail::sym("ellint_2"), =detail::CpuOnly{}]] Ellint2 {
    static constexpr auto operator()(auto k, auto phi) {
        using std::ellint_2;
        return ellint_2(k, phi);
    }
};
struct [[=detail::sym("ellint_3"), =detail::CpuOnly{}]] Ellint3 {
    static constexpr auto operator()(auto k, auto nu, auto phi) {
        using std::ellint_3;
        return ellint_3(k, nu, phi);
    }
};
struct [[=detail::sym("comp_ellint_1"), =detail::CpuOnly{}]] CompEllint1 {
    static constexpr auto operator()(auto k) {
        using std::comp_ellint_1;
        return comp_ellint_1(k);
    }
};
struct [[=detail::sym("comp_ellint_2"), =detail::CpuOnly{}]] CompEllint2 {
    static constexpr auto operator()(auto k) {
        using std::comp_ellint_2;
        return comp_ellint_2(k);
    }
};
struct [[=detail::sym("comp_ellint_3"), =detail::CpuOnly{}]] CompEllint3 {
    static constexpr auto operator()(auto k, auto nu) {
        using std::comp_ellint_3;
        return comp_ellint_3(k, nu);
    }
};

} // namespace ops
// clang-format on

} // namespace tensor
