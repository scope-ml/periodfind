# CPU Backend

The CPU backend uses Rust with [Rayon](https://github.com/rayon-rs/rayon) for
multithreaded parallelism. Install with:

```bash
cd rust && maturin develop --release
```

## Period-Finding

All period-finding classes have the same API as their GPU counterparts.

::: periodfind.cpu.ConditionalEntropy

::: periodfind.cpu.AOV

::: periodfind.cpu.LombScargle

::: periodfind.cpu.FPW

::: periodfind.cpu.MultiCriterionAscent

## Feature Extraction

These classes are CPU-only (no GPU backend).

::: periodfind.cpu.FourierDecomposition

::: periodfind.cpu.EllipsoidalFit

::: periodfind.cpu.DmDt

::: periodfind.cpu.BasicStats

::: periodfind.cpu.RemoveHighCadence
