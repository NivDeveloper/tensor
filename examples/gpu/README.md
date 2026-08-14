# gpu — stencil expressions on the GPU

The GPU path end to end, now with a real stencil: open a device, diffuse
heat across a doubly-periodic plate with a 5-point Laplacian written in
decorated reads, and check the numbers three ways — against the closed-form
decay of a Fourier mode, against the CPU running the *same expression*,
and by reading the kernel that actually ran. ~90 lines; this page is the
tour.

### The application owns the device

```cpp
auto dev = gpud::open_default();   // nullptr if no usable backend
```

tensor never opens, selects, or stores a device — it borrows a
`gpud::Device&` per `eval`. `open_default()` (from `gpud::auto_`) picks
the best compiled-in backend; `GPUD_BACKEND=vulkan|mock` insists on one,
`GPUD_LOG=1` explains a refusal. No device → the example prints why and
exits: the graceful-fallback policy is three lines of application code,
not library policy.

### The stencil: a Laplacian is a lambda over decorated reads

```cpp
auto lap = [](const auto &f) {
    return f[wrap(i - 1_c), j] + f[wrap(i + 1_c), j] +
           f[i, wrap(j - 1_c)] + f[i, wrap(j + 1_c)] - 4.0f * f[i, j];
};
```

A subscript names where each read lands — `f[i, wrap(j + 1_c)]` at cell
`(i, j)` reads `f[i, wrap(j+1_c)]`, wrapping at the edge.
The `wrap` decoration makes the plate periodic (`clamp` and `zero` are
the other boundary spellings; undecorated offset reads default to zero),
and any plain subexpression can be subscripted, so stencils compose
through the ordinary operators. A reusable stencil is just a lambda —
no registration, no special types.

### The physics: heat on a periodic plate

A 256×256 plate steps `u ← u + r·∇²u` at the stability limit
`r = 0.25`. The initial condition is a single Fourier mode
`sin(kx·i)·sin(ky·j)` — an exact eigenvector of the *discrete*
Laplacian, so every step scales it by exactly

```
λ = 1 − 4r·(sin²(π·mx/N) + sin²(π·my/N))
```

and after n steps the whole field is the initial mode times λⁿ — the GPU
output is checkable against pencil and paper, not just against the CPU.

### One expression type = one kernel

```cpp
for (int s = 0; s < steps; ++s) {
    u = eval(*dev, u[i, j] + r * lap(u));  // GPU
    w = eval(w[i, j] + r * lap(w));        // CPU — same spelling minus dev
}
```

The kernel is generated from the expression's *type* at compile time and
compiled by the device once — 200 iterations are 200 dispatches and a
single compile. Each `eval(dev, …)` is a full upload/compute/download
round trip; the six tensor leaves are all `u`, so they share one upload
bound into six slots (device-resident tensors, which would also skip the
re-upload across steps, are a roadmap item) — the honest cost model to
have in mind.

### What a run prints

```
256x256 periodic plate on "slang-vulkan", 200 steps at r = 0.25
mode (4, 3) decays by λ = 0.996238 per step; λ^200 = 0.470526

    i    j      u(0)    u(gpu)  analytic
    8    8    0.3928    0.1848    0.1848
   16   24    0.9808    0.4615    0.4615
   48   12   -0.7730   -0.3637   -0.3637
  100   50    0.1967    0.0926    0.0926
  200  150   -0.7063   -0.3323   -0.3323

max |gpu − cpu| after 200 steps: 0.0e+00
```

Two hundred steps in, the mode has decayed to 47% everywhere, matching
the closed form to display precision — and the GPU–CPU deviation is
exactly zero: both run the same operations from the same tree (where the
two compilers contract multiply-adds differently, the last bits can
drift; here they agree). The program ends by printing the complete Slang
kernel, where the stencil is visible as plain index arithmetic on the
neighbour reads:

```
pc.in1.data[((i / 256 + 255) % 256 * 256 + i % 256)]   // north, wrapped
```

That subscript is the same spelling `formula<Step>()` shows on the CPU
side — one index algebra, two dialects, generated from the type before
any device existed.

## Build & run

Needs the GPU path (skipped from CPU-only builds) and, at runtime, a
Vulkan ≥ 1.2 driver (macOS: `brew install molten-vk`) plus `slangc` on
the PATH:

```sh
make GPU=1                       # configures build-gpu with -DTENSOR_ENABLE_GPU=ON
./build-gpu/examples/gpu/gpu
```
