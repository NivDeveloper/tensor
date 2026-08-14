# pendulum — explicit time-stepping with expressions

A small numerics program: integrate an ensemble of pendulums forward in
time and watch the nonlinearity show up. It's a realistic use of the
library — fixed-shape state tensors, elementwise physics fused into single
passes — and it exercises the whole expression surface along the way. Read
the ~60 lines top to bottom; this page is the guided tour.

### State as fixed-shape tensors

`theta` and `omega` (angle and angular velocity) are `Tensor<float, 5>` —
five pendulums integrated together, the count fixed in the type. There are
no per-pendulum objects and no loops over particles inside the physics;
every update is one elementwise expression over the whole ensemble.

### The nonlinearity is the math vocabulary

```cpp
omega = eval(omega - math::Sin(theta) * dt);
```

A pendulum's angular acceleration is `-sin(theta)` (natural units) — the
`sin` is exactly what makes it *not* a harmonic oscillator. `math::Sin`
is the math vocabulary (`Math.h`): on a tensor operand it builds
the expression node, on a scalar it is plain `std::sin` — one object,
one spelling for both worlds. A function the vocabulary doesn't cover
goes through `map<f>` as an ordinary function (the basic example shows
that form); the operator vocabulary itself is complete and closed.

### The integrator is a sequence of `eval`s

```cpp
for (int s = 0; s < steps; ++s) {
    omega = eval(omega - math::Sin(theta) * dt);        // kick
    theta = eval(theta + omega * dt);                   // drift
}
```

This is symplectic (semi-implicit) Euler: advance the velocity with the
force at the current angle, then the angle with the *new* velocity. Each
right-hand side builds a lazy tree and computes nothing; `eval` runs it as
one fused elementwise pass and returns a fresh owning tensor, which
move-assigns back into the state. `eval` is the only way to evaluate — a
tensor is neither constructible nor assignable from an expression — so a
time-stepping loop is naturally just evals.

### What the run shows (physics and numerics)

```
  theta0    theta(T)     dE/E
  0.1000     0.1000     0.0001
  0.5000     0.4980     0.0019
  1.0000     0.9299     0.0061
  2.0000     0.0465     0.0005
  3.0000    -2.7091     0.0009
```

Released from rest and integrated for one small-angle period, the
near-harmonic pendulum returns almost to its release angle; the wider
swings, whose nonlinear periods are longer, are caught mid-swing far from
where they started (the θ₀ = 3.0 pendulum has fallen through the bottom and
climbed the far side). Amplitude-dependent period is exactly what the
linear approximation misses. The `dE/E` column is the relative drift of a
per-pendulum energy (`eval(0.5f * omega*omega + (1.0f -
math::Cos(theta)))`, operators and the vocabulary composed): it stays
small and bounded, the reason to reach for a symplectic step.

### One tree, two more renderings

`formula<E>()` prints the velocity kick with the vocabulary op named by
its annotation — `(in0[i] - (sin(in1[i]) * s0))` — and `gpu_source<E>()`
lowers the same kick to a complete Slang compute shader: `sin` is a
Slang intrinsic, so even the nonlinear term runs on the GPU. With
`-DTENSOR_ENABLE_GPU=ON`, `eval(dev, kick)` (`<Tensor/Gpu/Eval.h>`)
compiles and runs exactly that source on a gpud device.

## Build & run

```sh
# after a CMake build (make, from the repo root):
./build/examples/pendulum/pendulum
# or directly, no CMake:
g++-16 -std=c++26 -freflection -I../../include pendulum.cpp && ./a.out
```
