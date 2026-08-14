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
  epilogue rule (one fold per tree, elementwise above it).
- **The bias is the augmented constant-1 feature** (the classic trick) —
  the library broadcasts scalars, not rank-1-over-rank-2, so bias terms
  ride the input.
- Losses are folds (`fold<ops::Add>(e * e)`), momentum updates are
  plain elementwise evals, and the data/weights are index fills from
  deterministic pseudo-noise — no `<random>`, runs identically anywhere.

What to look for in the output: the MSE falling ~5 orders of magnitude
over 1500 epochs and the final train-set fit, `R² = 1.0000` (256 samples
against 64×64 weights is the interpolation regime — the point is the
machinery, not generalization).

`bench/Mlp_bench.cpp` times `train_step` against a hand-written loop
nest of the same algebra — the workload is matmul-shaped, and since the
contraction interchange landed it runs within ~1.2× of the hand loops
(bench/README.md carries the history: 3.3× before).

```sh
# after a CMake build (make, from the repo root):
./build/examples/mlp/mlp
# or directly, no CMake:
g++-16 -std=c++26 -freflection -O3 -I../../include mlp.cpp && ./a.out
```
