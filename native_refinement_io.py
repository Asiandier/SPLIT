"""Read frozen native fits for post-fit progressive OLS/SS refinement."""
from pathlib import Path
import csv
import hashlib
import json
import os
import numpy as np
import polars as pl
from .data_utils import load_pheno_covar_aligned_with_transform, load_covar_aligned


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(8 << 20), b''):
            h.update(block)
    return h.hexdigest()


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name+f'.tmp.{os.getpid()}')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)+'\n')
    temporary.replace(path)


def read_json(path):
    return json.loads(Path(path).read_text())


def keep_ids(path):
    ids = [line.split()[1] for line in Path(path).read_text().splitlines() if line.strip()]
    if len(set(ids)) != len(ids):
        raise ValueError(f'Duplicate sample IDs: {path}')
    return ids


def table(path, ids=None, columns=None):
    data = pl.read_csv(path, separator='\t', schema_overrides={'iid': pl.String}, columns=columns)
    if ids is not None:
        names = data['iid'].to_list()
        if len(set(names)) != len(names) or set(names) != set(ids):
            raise ValueError(f'Sample IDs do not match: {path}')
        lookup = {iid: i for i, iid in enumerate(names)}
        data = data[[lookup[iid] for iid in ids]]
    return data


def config(case):
    pipe = read_json(case['prefix']+'.pipeline.json')
    if pipe['status'] != 'complete' or pipe['prediction_phenotype_used'] is not False:
        raise ValueError('Only complete native fits without test-phenotype usage are supported')
    return read_json(pipe['config'])


def discover(study, architectures, replicates):
    study = Path(study).resolve()
    reuse = {}
    reuse_manifest = study/'random_result_reuse.tsv'
    if reuse_manifest.is_file():
        with reuse_manifest.open() as handle:
            for row in csv.DictReader(handle, delimiter='\t'):
                if row['method'] != 'coherit_ld4':
                    continue
                cid = row['source_case_id']
                if cid in reuse:
                    raise ValueError(f'Duplicate reused native case: {cid}')
                reuse[cid] = row
    cases = []
    for rep in replicates:
        for setting in ['baseline', 'large_low', 'large_high']:
            for arch in architectures:
                cid = f'rep{rep:03d}_{arch}'+('' if setting == 'baseline' else '__'+setting)
                path = study/'runs'/cid/'coherit_ld4/result.json'
                if not path.is_file() and cid in reuse:
                    record = reuse[cid]
                    path = Path(record['source_result'])
                    if digest(path) != record['source_result_sha256']:
                        raise ValueError(f'Reused native result changed: {cid}')
                result = read_json(path)
                if result['state'] != 'complete':
                    raise ValueError(f'Incomplete native fit: {cid}')
                model = read_json(Path(result['attempt'])/'model.json')['model']
                prefix = model['native_summary'].removesuffix('.summary.json')
                # Use the successful fit's phenotype directory. Study layouts
                # differ, and reused cases may belong to an earlier study.
                cfg = config(dict(prefix=prefix))
                pheno = Path(cfg['inputs']['fit_pheno']['path']).with_name('test.pheno')
                if not pheno.is_file():
                    raise ValueError(f'Missing held-out phenotype path: {cid}')
                cases.append(dict(case_id=cid, replicate=rep, architecture=arch,
                                  setting=setting, prefix=prefix, original_result=str(path),
                                  test_pheno=str(pheno)))
    if not cases or len({case['case_id'] for case in cases}) != len(cases):
        raise ValueError('A nonempty set of unique native cases is required')
    return cases


def verify_inputs(cfg, stages, expected_component, hashes):
    inp = cfg['inputs']
    if inp['component_spec']['sha256'] != expected_component:
        raise ValueError('Component partition does not match genotype cache')
    for name in stages:
        record = inp[name]
        actual = hashes.setdefault(record['path'], None)
        if actual is None:
            actual = digest(record['path'])
            hashes[record['path']] = actual
        if actual != record['sha256']:
            raise ValueError(f'Native input changed: {name}')


def recover_gamma(covariates, nuisance_score):
    """Recover a unique fitted gamma without using validation responses."""
    c, score = np.asarray(covariates, float), np.asarray(nuisance_score, float)
    if (c.ndim != 2 or c.shape[0] == 0 or score.shape != (c.shape[0],)
            or not np.all(np.isfinite(c)) or not np.all(np.isfinite(score))):
        raise ValueError('Aligned finite covariates and nuisance score required')
    gamma, _, rank, _ = np.linalg.lstsq(c, score, rcond=None)
    if rank != c.shape[1]:
        raise ValueError('Native gamma recovery requires full-column-rank validation covariates')
    error = float(np.max(abs(c@gamma-score)))
    if error > 2e-7:
        raise ArithmeticError('Native 8k gamma recovery failed')
    return gamma, error


def read_snapshot(case, stage, x, metadata, expected_ids, validation_ids=None):
    cfg = config(case)
    inp, bed = cfg['inputs'], cfg['genotype']['prefix']
    prefix = case['prefix']+('.work/fixed/validation_fit/coherit' if stage == 8 else '')
    summary = read_json(prefix+'.summary.json')
    if (summary['sparse_output_schema_version'] != 11 or not summary['lasso_branch_valid']
            or summary['n_grms'] != len(metadata['normalizers'])):
        raise ValueError('Native sparse fit schema/partition mismatch')
    pheno = inp['pheno' if stage == 8 else 'fit_pheno']['path']
    keep = inp['train_keep' if stage == 8 else 'fit_keep']['path']
    y, c, ids, _, transform = load_pheno_covar_aligned_with_transform(
        bed+'.fam', pheno, inp['covar']['path'], keep_ids=keep_ids(keep))
    if ids != expected_ids or x.shape[1] != len(ids) or summary['n_samples'] != len(ids):
        raise ValueError('Native sample order does not match cached genotypes')
    sources = [prefix+'.summary.json', pheno, keep, inp['covar']['path']]
    errors = {}
    if stage == 8:
        state_path = Path(prefix).parent/'sparse_state.npz'
        with np.load(state_path) as state:
            support = state['marker_indices'].astype(int)
            alpha = state['selected_beta_snp'].astype(float)
        cv, vids, _ = load_covar_aligned(bed+'.fam', inp['covar']['path'],
            transform=transform, keep_ids=keep_ids(inp['validation_keep']['path']))
        if vids != validation_ids or not set(ids).isdisjoint(vids):
            raise ValueError('Training/validation cohort mismatch')
        native = table(prefix+'.sparse_prediction.tsv', vids)
        nuisance = native['lasso_nuisance_fixed_score'].to_numpy()
        # Recover fitted gamma from nuisance predictions; no validation y is used.
        gamma, error = recover_gamma(cv, nuisance)
        errors['gamma_recovery_max_error'] = error
        extra = dict(cv=cv, validation_ids=np.asarray(vids),
                     native_validation_background=native['lasso_background_blup'].to_numpy())
        sources += [str(state_path), prefix+'.sparse_prediction.tsv']
    else:
        fixed = table(prefix+'.fixed_effects.tsv')
        np.testing.assert_array_equal(fixed['covariate_index'].to_numpy(), np.arange(c.shape[1]))
        gamma = fixed['beta'].to_numpy()
        effects = table(prefix+'.sparse_effects.tsv', columns=[
            'source_snp_index', 'component_index', 'beta_sparse', 'beta_background'])
        np.testing.assert_array_equal(effects['source_snp_index'].to_numpy(), np.arange(len(x)))
        np.testing.assert_array_equal(effects['component_index'].to_numpy(), metadata['groups'])
        all_alpha = effects['beta_sparse'].to_numpy()
        support = np.flatnonzero(all_alpha)
        alpha = all_alpha[support]
        extra = dict(original_blup=effects['beta_background'].to_numpy())
        sources += [prefix+'.fixed_effects.tsv', prefix+'.sparse_effects.tsv']
    if len(support) != summary['support_size'] or len(set(support.tolist())) != len(support):
        raise ValueError('Support does not match native summary')
    scale_info = summary['input_phenotype_standardization']
    mean, scale = scale_info['mean'], scale_info['standard_deviation']
    y = ((y-np.float32(mean))/np.float32(scale)).astype(float)
    sparse_score = np.asarray(x[support], float).T@alpha
    theta = np.asarray(summary['var_components_lasso_ml'], float)
    return dict(support=support, alpha=alpha, gamma=gamma, y=y, c=c, ids=np.asarray(ids),
                residual=y-c@gamma-sparse_score, old_score=sparse_score, theta=theta,
                phenotype_mean=mean, phenotype_scale=scale, Q=summary['q_chive'], **extra), {
                    'sources': {path: digest(path) for path in sources},
                    'active_groups': np.flatnonzero(theta[:-1] > 0).tolist(), **errors}


def aligned_phenotype(path, ids):
    values = {}
    for line in Path(path).read_text().splitlines():
        fields = line.split()
        if fields:
            if fields[1] in values:
                raise ValueError('Duplicate phenotype IDs')
            values[fields[1]] = float(fields[2])
    if set(values) != set(ids):
        raise ValueError('Phenotype cohort does not match prediction cohort')
    y = np.asarray([values[iid] for iid in ids])
    if not np.all(np.isfinite(y)) or np.var(y) <= 0:
        raise ValueError('Phenotypes must be finite with positive variance')
    return y
