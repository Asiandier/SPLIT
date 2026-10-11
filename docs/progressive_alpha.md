# Progressive OLS alpha with background SS

`run_sparse_pipeline --progressive-alpha` runs the adopted prediction refinement
after native COHERIT selection. It enables background SS automatically.
Training, validation and prediction cohorts retain their usual independent roles;
the sample sizes are supplied by the keep files.

1. Fit the native training-sample weighted LASSO and information-corrected REML,
   selecting lambda with the existing validation predictive R² procedure.
2. Initialize grouped background SS from native conditional BLUP and fit its
   EM-VAMP posterior with native gamma and LASSO alpha fixed. Preserve the
   background constraints `pi_g * slab_variance_g = theta_g / M_g` and the existing
   no-hyperprior design; native variance components and noise remain fixed.
3. Freeze this background and gamma. Rank the support by decreasing absolute
   native LASSO alpha, breaking ties by source SNP index. Jointly refit each
   growing prefix with ordinary least squares while holding all other alpha
   coordinates fixed. The residual target is `y - C gamma - X beta_SS`.
   Include the original coefficients at prefix count zero.
4. Choose the count maximizing full validation-prediction corr², including the
   sparse, background and covariate scores. Scores within `1e-12` of the maximum
   prefer the smaller count. There is no alpha Q constraint.
5. Warm-refit native COHERIT on training plus validation at the frozen lambda
   ratio, and fit that cohort's own BLUP-initialized background SS. Transfer only
   the selected count. Use the combined cohort's own support, standardization
   and LASSO ranking, capping the count at its support size, and jointly OLS-refit
   that prefix with its background and gamma frozen.
6. Emit full prediction and effects using the new alpha, frozen SS and gamma.
   A prediction phenotype is never accepted or read.

For a prefix `T` within support `S`, the coefficient correction solves

```text
min_delta || y - C gamma - X beta_SS - X_S alpha_LASSO - X_T delta ||².
```

The FP64 kernel uses incremental Cholesky factorization and accepts dependent
columns. In a deficient prefix, it returns the minimum Euclidean norm correction
within that prefix. Outside-prefix coefficients remain exactly unchanged.

Example for a supplied four-group partition:

```bash
python -m GPU_REML.run_sparse_pipeline \
  --mode fixed --bed-prefix genotype --component-spec ld4.npz \
  --pheno-txt train.pheno --keep-path train.keep \
  --validation-pheno-txt validation.pheno --validation-keep-path validation.keep \
  --progressive-alpha --compute-effects \
  --prediction-bed-prefix genotype --prediction-keep-path test.keep \
  --out-prefix results/coherit
```

Add the ordinary covariate arguments when needed. The workflow supports the
existing BED/PGEN and fixed/adaptive component interfaces. The low-level command
uses `--progressive-alpha-select` for a validation-selection fit and
`--progressive-alpha-k N` for a frozen-lambda refit. These modes are exclusive.
Standalone `--background-ss` retains the SS-only workflow.

Each refinement emits `.progressive_alpha.npz`, `.progressive_alpha.json` and
`.progressive_alpha_path.tsv`. They record native/new alpha, source indices,
ranking, selected count, numerical diagnostics and fitted squared terms.
`.background_ss.npz` stores the fixed background. The usual sparse effect and
prediction tables use the updated alpha; prediction/effect metadata identifies
both refinements.

Native REML/CHIVE estimates, KKT certificates, `.selected_snps.tsv` LASSO effects
and reusable LASSO warm states describe the original covariance-aligned fit.
Prediction refinement does not redefine those inference results. Its training
squared terms and validation metrics are reported separately under
`progressive_alpha` in the summary.

Local frozen full-450 results, oracle diagnostics and numerical regression
inputs belong to the v4 study at
`STUDY/prediction_refinement/progressive_ss_450_20261010`.
Their physical storage is managed by the study rather than the code package.
The local `validation/progressive_ss_450` compatibility symlink allows
existing numerical regression commands to access that study folder.
Validate the kernel against the full reference with:

```bash
OPENBLAS_NUM_THREADS=1 python tests/validate_progressive_reference.py \
  --reference validation/progressive_ss_450
```

## Refining a completed simulation study

`python -m GPU_REML.run_progressive_refinement` reuses completed native 8k/10k
fits without rerunning LASSO or REML. It reads each successful attempt from the
study's `runs/<case>/coherit_ld4/result.json`, including attempts after retries.
Reused Random cases may instead be resolved through `random_result_reuse.tsv`;
the recorded source-result hash is checked. Held-out phenotype paths follow the
successful native fit's input directory, including reused study inputs.
The output directory supplies genotype caches under `cache/train8000` and
`cache/fit10000`: native-standardized SNP-major designs, normalization,
component membership, sample identity, and phenotype-independent sample Grams.
Caches are checked against the native partition, sample order, and independent
BED decoding before use. Their physical files may be linked from shared storage.

Run the stages in this order, supplying the same `--out-dir` each time:

```text
prepare --study-root STUDY --architectures high_ld low_ld
background --stage 8
select
background --stage 10
refit
predict
evaluate
```

Two GPUs may split either background stage using `CUDA_VISIBLE_DEVICES` and
`--worker 0 --workers 2` / `--worker 1 --workers 2`. Finished cases are retained;
unfinished constrained EM-VAMP messages can continue with conservative damping
and identical fixed-point tolerances. Full prediction uses batched FP64 BED
products, with independent host decoding and native prediction checks.
Packed genotypes and the compiled prediction kernel are reused across batches,
without constructing a full test genotype design.
Validation choices, final coefficients and predictions are frozen successively.
Only `evaluate` reads test phenotypes. Per-case effects and prediction arrays,
validation paths, squared fitted terms, paired metrics and architecture/setting
summaries are written separately from the original native fits.
