# Sparse outer-iteration checkpoints

Add `--checkpoint` to `run_sparse_pipeline.py` or
`run_sparse_reml_pipeline.py`. The option is disabled by default and changes
no fitting or convergence settings. Run the same command again to resume.

Each native fit keeps `<out-prefix>.outer_checkpoint.npz`. The public
pipeline has separate checkpoints for validation fitting and final refitting.
Each completed REML–Lasso outer update replaces the preceding checkpoint;
the final covariance-aligned pair is also saved before effects and prediction
refinement are computed. A restart rebuilds genotype operators and device
workspaces, then continues after the last saved update. An interrupted Lasso
path, REML block or prediction refinement is repeated from its preceding
checkpoint; individual lambda values and refinement iterations are not saved.

The checkpoint preserves coefficients, variance components, candidate sets,
warm starts, information factors, preconditioner and convergence history.
JSON plus non-object NPY arrays preserve array precision and shared numerical
objects without pickle. A file lock prevents concurrent writers; streamed,
uncompressed writes use a temporary file, `fsync` and atomic replacement.
A failed write leaves the preceding complete checkpoint intact. Large
information factors can make a checkpoint several GiB; place its output on
a filesystem with sufficient space. Saving does not copy genotype matrices.

Source files, fitting settings, NumPy/JAX versions and input identities must
match. Small inputs and genotype sample/variant metadata use SHA-256; large
genotype bodies use resolved path, size and modification time. The execution
layout must also match. An incompatible checkpoint raises an error rather
than silently starting over. Keep the original source snapshot and inputs
unchanged when resuming. This is numerical-state recovery, not process-level
restoration; nondeterministic GPU kernels can still introduce roundoff changes.
