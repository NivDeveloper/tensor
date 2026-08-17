# histogram — data to counts to numpy

The most common thing to do with a pile of numbers, start to finish:

1. **Sample** — `rng::Normal<float, N>() * rng::Normal<float, N>()`: a
   product of normals, sharply peaked and heavy-tailed, and not a
   distribution the library names — which is why it gets histogrammed.
2. **Range** — `eval(fold<ops::Min>(x))` / `fold<ops::Max>`: a fold with no
   index list folds the whole tensor to the VALUE.
3. **Bin** — the one-line histogram:
   ```cpp
   auto counts = eval(scatter<i>(drop<B>(bins<B>(x[i], lo, hi)), 1u));
   ```
   `bins<B>(x[i], lo, hi)` is the value→bin map — which of `B` equal bins
   over `[lo, hi)` holds each value — and `scatter` deposits a count into
   that bin. `drop` discards out-of-range coordinates (numpy's semantics);
   `clamp<B>` would pile them into the edge bins instead.
4. **Centres** — `lo + gen::Iota<B>(0.5f) * w`: storage-free, the generator
   fuses into the arithmetic.
5. **Out** — `npy::savez` writes a `.npz` keyed by the struct's FIELD
   NAMES, so Python needs nothing declared twice.

## What to look for in the output

`kept 4095 of 4096`: the maximum element sits exactly ON `hi`, floors to
bin `B`, and `drop` discards it — the one-rounding edge the `bins` docs
call out, made visible. The bar chart is the familiar bell of a
peaked-at-zero product distribution.

```
range [-6.20902, 6.84656], kept 4095 of 4096 (the max itself lands ON hi and drops)
      ...
-0.497204 | ###################
-0.170814 | ##############################################
 0.155575 | ##################################################
 0.481965 | ###################
      ...
wrote histogram.npz
```

Then, in Python:

```python
d = np.load("histogram.npz")
plt.stairs(d["Counts"], edges=None)   # or: plt.bar(d["Centres"], d["Counts"])
```

## Build & run

```sh
# from the repo root, as part of the tree
make && ./build/examples/histogram/histogram

# or directly, no CMake
g++-16 -std=c++26 -freflection -I../../include histogram.cpp -o histogram
./histogram
```

The program writes `histogram.npz` into the current directory.
