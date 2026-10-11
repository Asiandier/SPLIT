# Progressive OLS alpha with background SS

`run_sparse_pipeline --progressive-alpha` runs the adopted prediction refinement
after native COHERIT selection. It enables background SS automatically.
Training, validation and prediction cohorts retain their usual independent roles;
the sample sizes are supplied by the keep files.

1. Fit the native training-sample weighted LASSO and information-corrected REML,
   selecting lambda with the existing validation predictive R² procedure.
   Compute the native conditional Gaussian BLUP using the LASSO residual.
2. Freeze this conditional BLUP and native gamma. Rank the support by decreasing absolute
   native LASSO alpha, breaking ties by source SNP index. Jointly refit each
   growing prefix with ordinary least squares while holding all other alpha
   coordinates fixed. The residual target is `y - C gamma - X beta_BLUP`.
   Include the original coefficients at prefix count zero.
3. Choose the count maximizing full validation-prediction corr², including the
   sparse, background and covariate scores. Scores within `1e-12` of the maximum
   prefer the smaller count. There is no alpha Q constraint and no background
   SS fit on the training-only cohort.
4. Warm-refit native COHERIT on training plus validation at the frozen lambda
   ratio and compute that cohort's own native conditional BLUP. Transfer only
   the selected count. Use the combined cohort's own support, standardization
   and LASSO ranking, capping the count at its support size, and jointly OLS-refit
   that prefix with its conditional BLUP and native gamma frozen.
5. With the updated alpha fixed, fit grouped background SS to
   `y - C gamma - X alpha_OLS`, initialized from the existing native BLUP.
   Preserve `pi_g * slab_variance_g = theta_g / M_g` and the no-hyperprior design;
   native gamma, variance components and noise remain fixed. This is the only
   background SS fit in the full workflow; no new conditional BLUP solve is
   needed after OLS. Its linear module uses streamed genotype products, PCG
   and cached Lanczos trace quadrature without constructing a sample Gram matrix.
6. Emit full prediction and optional effect tables using the updated alpha,
   final SS background and native gamma. A prediction phenotype is never
   accepted or read. The validation choice of count uses conditional BLUP;
   it does not rerun SS for each candidate count.

For a prefix `T` within support `S`, the coefficient correction solves

```text
min_delta || y - C gamma - X beta_BLUP - X_S alpha_LASSO - X_T delta ||².
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
  --progressive-alpha \
  --prediction-bed-prefix genotype --prediction-keep-path test.keep \
  --out-prefix results/coherit
```

Prediction is enabled by `--prediction-bed-prefix` or
`--prediction-pgen-prefix`. Add `--prediction-keep-path` when selecting a
prediction cohort from a larger genotype file. `--compute-effects` separately
requests effect tables; prediction always computes its required effects in
memory. The command completes selection, combined-sample refitting and full
prediction in the same workflow.

Add the ordinary covariate arguments when needed. The workflow supports the
existing BED/PGEN and fixed/adaptive component interfaces. The low-level command
uses `--progressive-alpha-select` for a validation-selection fit and
`--progressive-alpha-k N` for a frozen-lambda refit. These modes are exclusive.
Selection skips SS even if `--background-ss` is supplied; a frozen-count refit
enables SS after OLS, including when `N=0`.
Standalone `--background-ss` retains the SS-only workflow.

Each refinement emits `.progressive_alpha.npz`, `.progressive_alpha.json` and
`.progressive_alpha_path.tsv`. They record native/new alpha, source indices,
ranking, selected count, numerical diagnostics and fitted squared terms.
`.background_ss.npz` stores the final SS posterior, native BLUP initialization,
conditioning alpha/support and SS training residual; it is emitted only by the
final combined-sample refit. The usual sparse effect and prediction tables use
the updated alpha and final background; metadata identifies both refinements
and their order.

Native REML/CHIVE estimates, KKT certificates, `.selected_snps.tsv` LASSO effects
and reusable LASSO warm states describe the original covariance-aligned fit.
Prediction refinement does not redefine those inference results. Under
`progressive_alpha`, `old_full_squared_term` and `new_full_squared_term` describe
the OLS step with conditional BLUP fixed. `final_full_squared_term` and
`final_training_mse` describe the complete prediction after the final SS update.
Squared terms are uncentered means of squared fitted values, not corr².
