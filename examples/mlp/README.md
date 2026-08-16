# mlp — batched backprop as indexed folds

A two-layer perceptron (64 → 64 tanh → 1) fit to the single-index
regression target `y = sin(π c·x)` by full-batch gradient descent with
heavy-ball momentum — backprop written by hand, the way an einsum user
would spell it.

What it demonstrates:

- **With a batch axis, every backprop quantity is one fold.** The
  forward pass is `fold<k>(X[i,k] * W1[k,j])`; the first-layer
  gradient is the show-piece:

  ```cpp
  eval(fold<i>(X[i, k] * e[i] * w2[j] * (1.0 - H[i, j] * H[i, j])))
  ```

  — four factors, batch summed away, and the `[B, HID]` upstream-gradient
  intermediate that autograd frameworks materialize never exists.
- **Activations are the fold's epilogue.** `eval(Tanh(fold<k>(X[i,k] * W1[k,j])))` is ONE pass: the activation
  applies per output cell before the store — the index model's
  epilogue rule (one fold per tree, elementwise above it). It is the
  semantics of record and it is **currently the slower spelling**: the
  contraction interchange is reachable only when the fold is the tree's
  root, so this layer costs 1.04 ms where the two-eval split costs
  417 µs. That gap is the two-sweep epilogue on docs/roadmap.md, not
  anything about the example — which keeps the spelling the library
  means, rather than working around it.
- **The bias is the augmented constant-1 feature** (the classic trick) —
  the library broadcasts scalars, not rank-1-over-rank-2, so bias terms
  ride the input.
- Losses are folds (`fold(e * e)` — `ops::Add` is the default), momentum
  updates are plain elementwise evals, and the data and weights come from
  the counter-based samplers — no `<random>`, runs identically anywhere.
- **The target is the same shape as the model.** `y = sin(π c·x)` is
  `eval(Sin(pi * fold<j>(X[i, j] * c[j])))`, with `c` zeroed on the bias
  column by a mask over `gen::Iota<IN>` so that column plays no part. There
  is no hand-written loop anywhere in the file.
- `use_threads(4)` is worth ~15% here and changes no digit of the output:
  the contractions partition by output cell, so each accumulator is local.

What to look for in the output: the MSE falling ~5 orders of magnitude
over 1500 epochs and the final train-set fit, `R² = 1.0000` (256 samples
against 64×64 weights is the interpolation regime — the point is the
machinery, not generalization).

`bench/Mlp_bench.cpp` times `train_step` against a hand-written loop
nest of the same algebra — the workload is matmul-shaped. It reads 2.9×
the hand loops today for the epilogue reason above; split into two evals
the same step is within ~1.2× (bench/README.md carries the history and
the measurement).

```sh
# after a CMake build (make, from the repo root):
./build/examples/mlp/mlp
# or directly, no CMake:
g++-16 -std=c++26 -freflection -O3 -I../../include mlp.cpp && ./a.out
```
