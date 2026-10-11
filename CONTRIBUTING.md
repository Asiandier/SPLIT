# Contributing

Thank you for considering a contribution to GPU_REML.

## Development Setup

```bash
python -m pip install -e ".[dev]"
```

Optional PGEN support:

```bash
python -m pip install -e ".[pgen]"
```

## Checks Before Submitting

Check the production checkout:

```bash
python -m pyflakes *.py
python -m pip wheel --no-deps --no-build-isolation --wheel-dir /tmp/gpu_reml_wheel .
```

Keep numerical regression tests and generated data in a separate validation
workspace. Run that suite against a snapshot of the candidate source (retaining
the package name `GPU_REML`), and record the source hashes with its results.
For example, from the validation snapshot containing its external `tests/`:

```bash
JAX_PLATFORMS=cpu python -m pytest -q tests
```

FP64-specific tests use scoped precision contexts; they must not change global
JAX precision at collection time. GPU tests can select CUDA explicitly through
`JAX_PLATFORMS=cuda,cpu` and `CUDA_VISIBLE_DEVICES`.

For changes touching genotype streaming, REML scoring, PCG, preconditioning,
or sparse paths, validate against a dense or analytically simple reference
before publishing.

## Coding Guidelines

- Preserve the linear-operator contract: `K @ V = Z @ (Z.T @ V) / m_eff`.
- Keep host-side streaming and traced JAX code separated. Avoid calling Python
  genotype-streaming functions inside `jax.jit`-traced regions.
- Prefer explicit validation for sample order, variant order, and component
  definitions.
- Runtime modules must serve a supported project pipeline or public model API.
  Keep study-specific batch runners, prior comparisons, benchmark utilities and
  oracle analyses in the study workspace outside this package. Keep numerical
  regression tests and their fixtures in the separate validation workspace.
- Do not commit benchmark outputs, logs, temporary genotype splits, or private
  data paths.

## Reporting Issues

Please include:

- command line or API call,
- input format and dimensions,
- JAX/JAXLIB/CUDA versions,
- GPU model and memory,
- full traceback or relevant log excerpt,
- a minimal reproducer when possible.
