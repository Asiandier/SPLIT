# Training-only AIC selection for fixed GRM groups

The public sparse runner supports `--selection-criterion aic` in fixed mode.
Pass the **entire fitting cohort** through `--pheno-txt` and `--keep-path`.
No validation phenotype, validation keep file, or combined-sample refit inputs
are accepted. The default selection criterion remains `validation`, with its
existing training/validation selection and final combined-sample refit.

```bash
python -m GPU_REML.run_sparse_pipeline --mode fixed --selection-criterion aic \
  --bed-prefix genotype --component-spec groups.npz \
  --pheno-txt fit.pheno --keep-path fit.keep --covar-txt covariates.txt \
  --out-prefix output/model --compute-effects --alpha-vamp
```

Prediction genotype options can be supplied as usual; no prediction phenotype
is accepted. AIC defaults to the native residual Lasso solver. Selecting the
BASIL backend explicitly is currently rejected for AIC. All common statistical
and numerical settings come from the same public runner used for validation.

## Model and selection

For the current covariance

$$
H=\sum_g\theta_g K_g+\theta_e I,
$$

each path point retains its penalized Lasso coefficients and profiles the
ordinary covariates C. The selection score, omitting terms constant within
the current covariance/path, is

$$
\mathrm{AIC}_\lambda\doteq
\min_\gamma (y-C\gamma-Z\widehat\alpha_\lambda)^T H^{-1}
              (y-C\gamma-Z\widehat\alpha_\lambda)
+2\operatorname{rank}([C,Z_{S_\lambda}]).
$$

The L1 penalty is not added to the likelihood. Background random effects are
integrated out, rather than counted as a separate parameter for every SNP.
Whitening by a positive-definite H preserves the selected column rank.
The rank is cached within the immutable fitting-cohort/design scope and uses
a full-rank certificate or SVD when columns are dependent.

After selecting alpha, the existing information-corrected sparse REML updates
the variance components. The next path reselects lambda at the updated
covariance; the existing final-alignment check still applies. This is a
training-AIC selection policy inside the native alternating algorithm. It is
not a claim of global joint maximum-likelihood/AIC optimization. In particular,
the omitted log-determinant changes with H, so absolute scores from different
outer iterations must not be compared.

The native nonimprovement-window length is applied to AIC with lower values
preferred. This is the same path-prefix computational heuristic used by the
validation workflow, not a guarantee of the optimum over an unevaluated grid.
Only full-genome-KKT certified points are eligible for selection.

## Verified, matrix-free likelihood calculation

Let Q be an orthonormal basis of C, Pi=I−QQᵀ,
A=Pi H Pi+QQᵀ and t=Pi(y−Z alpha). The profiled quadratic is tᵀA⁻¹t.
The callback reuses the residual solver's dual vector w, verifies its FP64
residual, and refines it only when necessary. For e=t−Aw,

$$
q_{\rm lower}=2t^Tw-w^TAw,\qquad
0\le t^TA^{-1}t-q_{\rm lower}
\le \|e\|^2/\min(\theta_e,1).
$$

The checked tolerance is min(requested PCG tolerance, 1e−5), matching the final
selected-information precision. The numerical quadratic error bound and any
extra solves are recorded. No sample-by-sample covariance matrix or held-out
prediction is required by the criterion callback.

## Outputs, resumption and scope

- `<prefix>.aic.json` contains the training-only path scores and outer trace.
  Validation R² fields are null; AIC values are never encoded as prediction R².
- `<prefix>.pipeline.json` uses `final_fit` and `refit_performed: false` for
  AIC. It explicitly reports zero validation samples. The validation workflow
  retains its existing `final_refit` output.
- The final prediction refinement, convergence certificates and native fallback
  policy are shared with the validation/refit workflow. Its internal legacy
  refinement-stage label `final_refit` denotes the final output stage; the AIC
  pipeline record states that no second cohort refit occurred.
- Selection criterion, inputs, numerical settings and implementation digest
  participate in the work-directory signature. A run cannot resume another
  criterion's endpoint silently.
- AIC currently supports fixed GRM groups. Adaptive partition selection retains
  its validation-based workflow.

This is a plug-in information criterion using the covariance learned by the
alternating fit. The fixed-covariance Lasso degrees-of-freedom result does not
by itself prove an exact risk estimate for the whole covariance/tuning search.
The local h² SE remains an approximation at a fixed selected branch; its
previous validation-based simulation coverage is not evidence of coverage
under AIC selection.

References: [Zou, Hastie and Tibshirani (2007)](https://arxiv.org/abs/0712.0881),
[Tibshirani and Taylor (2012)](https://arxiv.org/abs/1111.0653).
