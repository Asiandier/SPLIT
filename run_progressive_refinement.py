"""Batch the adopted SS -> progressive OLS workflow over existing native fits.

Stages are explicit and resumable. prepare/background never read held-out
responses; select reads validation y only, and evaluate is the sole reader of
test y. Shared genotype caches must retain their original sample identities.
"""
import os
os.environ.setdefault('POLARS_MAX_THREADS', '4')
os.environ.setdefault('XLA_PYTHON_CLIENT_PREALLOCATE', 'false')
from pathlib import Path
import argparse
import gc
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict
import numpy as np
import polars as pl
from scipy.stats import t as student_t
from threadpoolctl import threadpool_limits
from . import native_refinement_io as io
from .progressive_alpha import fit_path, select_prefix


def emit(event, **kw):
    import json
    print(json.dumps(dict(event=event, **kw)), flush=True)


def code_hashes():
    directory = Path(__file__).parent
    return {name: io.digest(directory/name) for name in [
        'run_progressive_refinement.py', 'native_refinement_io.py', 'bed_products.py',
        'grouped_ss_vamp.py', 'background_ss.py', 'progressive_alpha.py']}


def loaded(path):
    with np.load(path) as data:
        return {k: data[k] for k in data.files}


def prepared(root):
    protocol = io.read_json(root/'protocol.json')
    manifest = io.read_json(root/'cache/manifest.json')
    if (manifest['status'] != 'ready' or manifest['protocol_sha256'] != io.digest(root/'protocol.json')
            or protocol['code_hashes'] != code_hashes()):
        raise ValueError('Changed protocol/code or incomplete native preparation')
    return protocol, manifest


def meta_and_cache(root, stage):
    cache = root/'cache'/('train8000' if stage == 8 else 'fit10000')
    return loaded(cache/f'metadata{stage}.npz'), cache


def baseline(root, case, stage, manifest):
    path = root/'results'/case['case_id']/f'baseline{stage}.npz'
    if io.digest(path) != manifest['cases'][case['case_id']][str(stage)]['snapshot_sha256']:
        raise ValueError('Changed native snapshot')
    return loaded(path)


def prepare(root, study, architectures, replicates):
    from .bed_products import packed_bed, decode_numpy
    root.mkdir(parents=True, exist_ok=True)
    for name in ['cache', 'results', 'logs']:
        (root/name).mkdir(exist_ok=True)
    cases = io.discover(study, architectures, replicates)
    meta8, cache8 = meta_and_cache(root, 8)
    meta10, cache10 = meta_and_cache(root, 10)
    ident8 = io.read_json(cache8/'cache_identity.json')
    ident10 = io.read_json(cache10/'genotype_identity.json')['identity']
    norm8 = io.read_json(cache8/'normalization.json')['identity']
    ids8, vids, ids10 = ident8['ids'], ident8['validation_ids'], ident10['ids']
    if (ids8 != norm8['training_ids'] or ids10 != norm8['fit_ids']
            or not set(ids8).isdisjoint(vids) or set(ids8)|set(vids) != set(ids10)
            or ident8['component_sha256'] != ident10['component_sha256']):
        raise ValueError('Shared genotype-cache cohorts/partition do not match')
    x8 = np.load(cache8/'x8.npy', mmap_mode='r')
    xv = np.load(cache8/'x_validation.npy', mmap_mode='r')
    x10 = np.load(cache10/'x10.npy', mmap_mode='r')
    p = len(meta10['groups'])
    if (x8.shape != (p, len(ids8)) or xv.shape != (p, len(vids))
            or x10.shape != (p, len(ids10))):
        raise ValueError('Shared genotype-cache dimensions do not match')
    np.testing.assert_array_equal(meta8['groups'], meta10['groups'])
    np.testing.assert_array_equal(meta8['normalizers'], np.bincount(meta8['groups']))
    np.testing.assert_array_equal(meta10['normalizers'], np.bincount(meta10['groups']))
    cfg0 = io.config(cases[0])
    bed = cfg0['genotype']['prefix']
    fam_ids = io.keep_ids(bed+'.fam')
    lookup = {name: i for i, name in enumerate(fam_ids)}
    packed = packed_bed(bed, p, len(fam_ids))
    probes = np.unique(np.linspace(0, p-1, 127, dtype=int))
    for design, ids, meta in [(x8, ids8, meta8), (xv, vids, meta8), (x10, ids10, meta10)]:
        chosen = np.linspace(0, len(ids)-1, 19, dtype=int)
        rows = [lookup[ids[i]] for i in chosen]
        expected = decode_numpy(np.asarray(packed[probes]), rows, meta['means'][probes], meta['inv'][probes])
        np.testing.assert_array_equal(design[np.ix_(probes, chosen)], expected)
    inputs_checked, records = {}, {}
    with np.load(cfg0['inputs']['component_spec']['path']) as spec:
        for g in range(len(meta10['normalizers'])):
            np.testing.assert_array_equal(np.sort(spec[f'arr_{g}']), np.flatnonzero(meta10['groups'] == g))
    protocol = dict(schema_version=1, study=str(Path(study).resolve()), cases=cases,
        p=p, bed=bed, fam_ids=fam_ids, train_ids=ids8, validation_ids=vids, fit_ids=ids10,
        n_train=len(ids8), n_validation=len(vids), n_refit=len(ids10),
        code_hashes=code_hashes(), algorithm='native grouped background SS then progressive joint OLS',
        Q_constraint=False, gamma_fixed=True, background_fixed_during_alpha=True,
        selection_metric='full validation corr_squared; smaller k within 1e-12',
        transfer='count only; independent native support and LASSO ranking at refit',
        background_initialization='native conditional BLUP; exact reconstruction if unavailable',
        background_pi_hyperprior=None, background_prior_constraint='pi_g*v_g=theta_g/M_g',
        component_sha256=ident10['component_sha256'], test_labels_used_for_selection=False,
        cache_decode_max_error=0., background_batch_size=32)
    if (root/'protocol.json').exists() and io.read_json(root/'protocol.json') != protocol:
        raise ValueError('Refusing to replace an existing different protocol')
    io.atomic_json(root/'protocol.json', protocol)
    began = time.monotonic()
    for i, case in enumerate(cases):
        cfg = io.config(case)
        if cfg['genotype']['prefix'] != bed:
            raise ValueError('All native fits must use the shared genotype')
        io.verify_inputs(cfg, ['pheno', 'validation_pheno', 'fit_pheno', 'train_keep',
            'validation_keep', 'fit_keep', 'covar', 'component_spec'],
            protocol['component_sha256'], inputs_checked)
        for extension, old in ident10['genotype']['files'].items():
            path = Path(bed+'.'+extension)
            if path.stat().st_size != old['size'] or path.stat().st_mtime_ns != old['mtime_ns']:
                raise ValueError('Cached genotype identity changed')
        directory = root/'results'/case['case_id']
        directory.mkdir(exist_ok=True)
        records[case['case_id']] = {}
        for stage, design, metadata, ids in [(8, x8, meta8, ids8), (10, x10, meta10, ids10)]:
            path = directory/f'baseline{stage}.npz'
            record_path = path.with_suffix('.json')
            if record_path.exists():
                record = io.read_json(record_path)
                if io.digest(path) != record['snapshot_sha256']:
                    raise ValueError('Changed existing native snapshot')
                for source, h in record['sources'].items():
                    if io.digest(source) != h:
                        raise ValueError('Changed native source')
            else:
                snapshot, record = io.read_snapshot(case, stage, design, metadata, ids, vids)
                if stage == 8:
                    native = io.table(case['prefix']+'.work/fixed/validation_fit/coherit.sparse_prediction.tsv', vids)
                    error = float(np.max(abs(np.asarray(xv[snapshot['support']], float).T@snapshot['alpha']
                                             -native['lasso_fixed_snp_score'].to_numpy())))
                    if error > 2e-6:
                        raise ArithmeticError('Native support or 8k genotype scaling mismatch')
                    record['validation_sparse_score_max_error'] = error
                np.savez(path, **snapshot)
                record.update(snapshot_sha256=io.digest(path), test_labels_read=False,
                              validation_labels_used_for_recovery=False)
                io.atomic_json(record_path, record)
            records[case['case_id']][str(stage)] = record
        if (i+1) % 10 == 0:
            emit('native_prepared', cases=i+1, total=len(cases), seconds=time.monotonic()-began)
    io.atomic_json(root/'cache/manifest.json', dict(status='ready', cases=records,
        protocol_sha256=io.digest(root/'protocol.json'), inputs_checked=inputs_checked,
        test_labels_read=False, validation_labels_read=False))
    emit('preparation_complete', cases=len(cases), seconds=time.monotonic()-began)


def background(root, stage, worker, workers, batch_size):
    import jax
    jax.config.update('jax_enable_x64', True)
    import jax.numpy as jnp
    from jax.scipy.sparse.linalg import cg
    from .bed_products import packed_bed, decode_resident
    from .background_ss import GroupedSSBackgroundContext, refine_lasso_background_ss
    from .grouped_ss_vamp import SampleSpectrum, VAMPConfig
    protocol, manifest = prepared(root)
    meta, cache = meta_and_cache(root, stage)
    ids = protocol['train_ids' if stage == 8 else 'fit_ids']
    lookup = {name: i for i, name in enumerate(protocol['fam_ids'])}
    packed = packed_bed(protocol['bed'], protocol['p'], len(protocol['fam_ids']))
    x = decode_resident(packed, [lookup[iid] for iid in ids], meta['means'], meta['inv'])
    context = GroupedSSBackgroundContext(x, meta['groups'], meta['normalizers'])
    groups = defaultdict(list)
    for case in protocol['cases']:
        active = tuple(manifest['cases'][case['case_id']][str(stage)]['active_groups'])
        groups[active].append(case)
    tasks = []
    for active in sorted(groups, key=lambda g: (-len(groups[g]), g)):
        rows = groups[active]
        tasks.extend(rows[i:i+batch_size] for i in range(0, len(rows), batch_size))
    assigned = [c for i, rows in enumerate(tasks) if i % workers == worker for c in rows]
    grams = jax.device_put(np.stack([np.load(cache/f'grm8_{g}.npy' if stage == 8
        else cache/f'grm{g}.npy') for g in range(len(meta['normalizers']))]))

    @jax.jit
    def inverse_many(grams, theta, residual):
        def one(t, r):
            def matvec(v):
                parts = jnp.stack([grams[g]@v for g in range(len(meta['normalizers']))])
                return jnp.sum(t[:-1, None]*parts, axis=0)+t[-1]*v
            z, _ = cg(matvec, r, tol=1e-10, maxiter=1000)
            return z, jnp.linalg.norm(matvec(z)-r)/jnp.linalg.norm(r)
        z, error = jax.vmap(one, in_axes=(0, 0))(theta.T, residual.T)
        return z.T, error

    tiers = [dict(max_iterations=300, damping=.2, em_damping=.05),
             dict(max_iterations=600, damping=.1, em_damping=.0025, em_acceleration_floor=1e-5),
             dict(max_iterations=1000, damping=.05, em_damping=.0005, em_acceleration_floor=1e-5),
             dict(max_iterations=1500, damping=.01, em_damping=.0001, em_acceleration_floor=1e-5)]
    for tier, options in enumerate(tiers):
        pending = defaultdict(list)
        for case in assigned:
            directory = root/'results'/case['case_id']
            info = directory/f'background{stage}.json'
            if info.exists() and io.read_json(info)['status'] == 'converged':
                if io.digest(directory/f'background{stage}.npz') != io.read_json(info)['posterior_sha256']:
                    raise ValueError('Changed converged SS posterior')
                continue
            fast = False
            state_file = directory/f'ss_state{stage}.npz'
            if tier and state_file.exists():
                pi = np.load(state_file)['pi']
                fast = bool(pi[pi > 0].min() < 1e-5)
            active = tuple(manifest['cases'][case['case_id']][str(stage)]['active_groups'])
            pending[(active, fast)].append(case)
        emit('ss_tier', stage=stage, worker=worker, tier=tier, remaining=sum(map(len, pending.values())))
        if not pending:
            break
        for (active, fast), cases in sorted(pending.items(), key=lambda pair: (-len(pair[1]), pair[0])):
            pattern = ''.join(map(str, active))
            if active and active not in context.spectra:
                name = f'spectrum8_{pattern}' if stage == 8 else f'vamp_spectrum_{pattern}'
                values_path, vectors_path = cache/(name+'_values.npy'), cache/(name+'_vectors.npy')
                if not values_path.exists():
                    gram = sum(meta['normalizers'][g]*grams[g] for g in active)
                    values, vectors = jnp.linalg.eigh((gram+gram.T)*.5)
                    values, vectors = map(np.asarray, (values, vectors))
                    probes = np.random.default_rng(41).normal(size=(len(ids), 2))
                    target = np.asarray(gram)@probes
                    error = float(np.linalg.norm(target-vectors@(values[:, None]*(vectors.T@probes)))/np.linalg.norm(target))
                    if error > 1e-9:
                        raise ArithmeticError('Eigensystem reconstruction failed')
                    # A single active pattern belongs to one task per worker only
                    # when small; atomic replacement also permits shared reuse.
                    for path, array in [(values_path, np.maximum(values, 0.)), (vectors_path, vectors)]:
                        temp = path.with_name(path.name+f'.tmp.{os.getpid()}.npy')
                        np.save(temp, array)
                        temp.replace(path)
                    io.atomic_json(cache/(name+'.json'), dict(active_groups=list(active),
                        reconstruction_relative_error=error, phenotypes_used=False))
                    emit('spectrum_ready', stage=stage, worker=worker, active=pattern, error=error)
                context.spectra[active] = SampleSpectrum(np.load(values_path), np.load(vectors_path),
                                                        int(np.isin(meta['groups'], active).sum()))
            for start in range(0, len(cases), batch_size):
                rows = cases[start:start+batch_size]
                originals = [baseline(root, case, stage, manifest) for case in rows]
                theta = np.column_stack([o['theta'] for o in originals])
                residual = np.column_stack([o['residual'] for o in originals])
                alpha = np.zeros((protocol['p'], len(rows)))
                for i, o in enumerate(originals):
                    alpha[o['support'], i] = o['alpha']
                    np.testing.assert_array_equal(originals[0]['c'], o['c'])
                state_files = [root/'results'/case['case_id']/f'ss_state{stage}.npz' for case in rows]
                state = None
                if tier and all(path.exists() for path in state_files):
                    saved = [loaded(path) for path in state_files]
                    state = dict(r1=np.column_stack([s['r1'] for s in saved]),
                        gamma1=np.asarray([float(s['gamma1']) for s in saved]),
                        pi=np.column_stack([s['pi'] for s in saved]),
                        stable=np.asarray([int(s['stable']) for s in saved]), done=np.zeros(len(rows), bool),
                        iterations=max(int(s['iterations']) for s in saved))
                    initial = np.zeros_like(alpha)
                    errors = np.asarray([io.read_json(root/'results'/case['case_id']/f'background{stage}.json')['initializer_relative_residual'] for case in rows])
                elif stage == 10:
                    initial = np.column_stack([o['original_blup'] for o in originals])
                    errors = np.zeros(len(rows))
                else:
                    z, errors = inverse_many(grams, jax.device_put(theta), jax.device_put(residual))
                    errors = np.asarray(errors)
                    if errors.max() > 2e-10:
                        raise ArithmeticError('Conditional BLUP reconstruction failed')
                    initial = np.asarray(context.products.transpose(z))*theta[:-1][meta['groups']]/meta['normalizers'][meta['groups'], None]
                cfg = VAMPConfig(**dict(options, **({'em_damping': .05} if fast else {})))
                trace = root/'logs'/f'ss{stage}_worker{worker}_tier{tier}_{rows[0]["case_id"]}.jsonl'
                emit('ss_batch_started', stage=stage, worker=worker, tier=tier,
                     cases=[case['case_id'] for case in rows], active=pattern)
                def callback(row):
                    import json
                    with trace.open('a') as handle:
                        handle.write(json.dumps(row)+'\n')
                    if row['iteration'] % 100 == 0:
                        emit('ss_progress', stage=stage, worker=worker, tier=tier,
                             iteration=row['iteration'], converged=sum(row['converged']),
                             total=len(rows), seconds=row['seconds'])
                result = refine_lasso_background_ss(context,
                    y=np.column_stack([o['y'] for o in originals]), alpha=alpha,
                    gamma=np.column_stack([o['gamma'] for o in originals]), covariates=originals[0]['c'],
                    blup=initial, variance_components=theta, config=cfg, callback=callback, initial_state=state)
                np.testing.assert_allclose(result.residual, residual, atol=2e-12, rtol=0)
                post = result.posterior
                np.testing.assert_allclose(post.pi*post.slab_variance,
                    theta[:-1]/meta['normalizers'][:, None], rtol=1e-10, atol=1e-15)
                for i, case in enumerate(rows):
                    directory = root/'results'/case['case_id']
                    path = directory/f'background{stage}.npz'
                    np.savez(path, beta=post.mean[:, i], pip=post.pip[:, i], pi=post.pi[:, i],
                             slab_variance=post.slab_variance[:, i], train_score=result.background[:, i])
                    info = dict(status=post.status[i], iterations=post.iterations, config=asdict(cfg),
                        posterior_sha256=io.digest(path), numerical_tier=tier, code_hashes=code_hashes(),
                        baseline_sha256=manifest['cases'][case['case_id']][str(stage)]['snapshot_sha256'],
                        initializer_relative_residual=float(errors[i]), initializer=(
                            'native exported conditional BLUP' if stage == 10 else 'exact conditional BLUP'),
                        gamma_fixed=True, alpha_fixed=True, validation_labels_used=False,
                        test_labels_used=False, trace=str(trace), seconds=post.seconds)
                    io.atomic_json(directory/f'background{stage}.json', info)
                    if post.status[i] != 'converged':
                        final = post.final_state
                        np.savez(state_files[i], r1=final['r1'][:, i], gamma1=final['gamma1'][i],
                                 pi=final['pi'][:, i], stable=final['stable'][i], iterations=final['iterations'])
                    elif state_files[i].exists():
                        state_files[i].unlink()
                emit('ss_batch_complete', stage=stage, worker=worker, tier=tier,
                     cases=len(rows), converged=post.status.count('converged'), seconds=post.seconds)
                del result, post, alpha, initial, originals
                gc.collect()
    statuses = {case['case_id']: io.read_json(root/'results'/case['case_id']/f'background{stage}.json')['status'] for case in assigned}
    io.atomic_json(root/'results'/f'ss{stage}_worker{worker}_status.json', dict(statuses=statuses,
        converged=sum(s == 'converged' for s in statuses.values()), total=len(assigned)))
    pending_ids = [cid for cid, s in statuses.items() if s != 'converged']
    if pending_ids:
        raise RuntimeError(f'Unconverged SS fits rejected: {pending_ids}')
    emit('ss_worker_complete', stage=stage, worker=worker, cases=len(assigned))


def posterior(root, case, stage, manifest):
    directory = root/'results'/case['case_id']
    info = io.read_json(directory/f'background{stage}.json')
    if (info['status'] != 'converged' or info['code_hashes'] != code_hashes()
            or info['baseline_sha256'] != manifest['cases'][case['case_id']][str(stage)]['snapshot_sha256']
            or info['posterior_sha256'] != io.digest(directory/f'background{stage}.npz')):
        raise ValueError('Unconverged or changed background')
    return loaded(directory/f'background{stage}.npz'), info


def select(root, jobs):
    import jax
    jax.config.update('jax_enable_x64', True)
    from .bed_products import packed_bed, PackedBedPredictor
    protocol, manifest = prepared(root)
    meta, cache = meta_and_cache(root, 8)
    x, xv = np.load(cache/'x8.npy', mmap_mode='r'), np.load(cache/'x_validation.npy', mmap_mode='r')
    lookup = {name: i for i, name in enumerate(protocol['fam_ids'])}
    rows = [lookup[iid] for iid in protocol['validation_ids']]
    packed = packed_bed(protocol['bed'], protocol['p'], len(protocol['fam_ids']))
    predictor = PackedBedPredictor(packed, rows, meta['means'], meta['inv'])
    scores = np.empty((len(rows), len(protocol['cases'])))
    for start in range(0, len(protocol['cases']), 48):
        cases = protocol['cases'][start:start+48]
        coefs = np.column_stack([posterior(root, case, 8, manifest)[0]['beta'] for case in cases])
        scores[:, start:start+len(cases)] = predictor(coefs)
        emit('validation_background_ready', cases=start+len(cases), total=len(protocol['cases']))
    np.save(root/'cache/validation_background.npy', scores)
    def one(index, case):
        directory = root/'results'/case['case_id']
        base = baseline(root, case, 8, manifest)
        bg, info = posterior(root, case, 8, manifest)
        xs, vs = np.asarray(x[base['support']], float).T, np.asarray(xv[base['support']], float).T
        r = base['y']-base['c']@base['gamma']-bg['train_score']
        path = fit_path(xs.T@xs/len(xs), xs.T@r/len(xs), base['alpha'], base['support'])
        yv = io.aligned_phenotype(io.config(case)['inputs']['validation_pheno']['path'], protocol['validation_ids'])
        mean, scale = float(base['phenotype_mean']), float(base['phenotype_scale'])
        fixed = mean+scale*(base['cv']@base['gamma']+scores[:, index])
        k, corr2, mse = select_prefix(path, scale*vs, fixed, yv)
        alpha_score = xs@path.coefficients[k]
        full = base['c']@base['gamma']+bg['train_score']+alpha_score
        row = dict(case_id=case['case_id'], architecture=case['architecture'], setting=case['setting'],
            replicate=case['replicate'], support=len(base['support']), selected_k=k,
            selected_rank=int(path.metrics[k, 1]), baseline_validation_corr2=float(corr2[0]),
            validation_corr2=float(corr2[k]), validation_gain=float(corr2[k]-corr2[0]),
            sparse_squared_term=float(np.mean(alpha_score**2)), full_squared_term=float(np.mean(full**2)),
            Q=float(base['Q']))
        np.savez(directory/'posterior8.npz', support=base['support'], alpha=path.coefficients[k],
            old_alpha=base['alpha'], order=path.order, selected_k=k, validation_corr2=corr2)
        pl.DataFrame(dict(k=np.arange(len(corr2)), validation_corr2=corr2, validation_mse=mse,
            sparse_squared_term=path.metrics[:, 3], train_mse_reduction=path.metrics[:, 2],
            normal_equation_error=path.metrics[:, 4])).write_csv(directory/'validation_path.tsv', separator='\t')
        io.atomic_json(directory/'fit8.json', dict(status='solved', metrics=row, diagnostics=path.diagnostics,
            background_sha256=info['posterior_sha256'], baseline_sha256=info['baseline_sha256'],
            validation_labels_used_for_selection=True, validation_labels_used_for_fitting=False,
            test_labels_used=False, Q_constraint=False, code_hashes=code_hashes(),
            validation_pheno_sha256=io.digest(io.config(case)['inputs']['validation_pheno']['path'])))
        return row
    records = []
    with threadpool_limits(limits=1), ThreadPoolExecutor(max_workers=jobs) as pool:
        for future in as_completed([pool.submit(one, i, case) for i, case in enumerate(protocol['cases'])]):
            records.append(future.result())
            if len(records) % 25 == 0:
                emit('validation_refits_ready', cases=len(records), total=len(protocol['cases']))
    pl.DataFrame(records).sort('case_id').write_csv(root/'results/validation_metrics.tsv', separator='\t')
    frozen = {case['case_id']: io.digest(root/'results'/case['case_id']/'fit8.json') for case in protocol['cases']}
    io.atomic_json(root/'results/validation_frozen.json', dict(status='frozen', fits=frozen,
        protocol_sha256=io.digest(root/'protocol.json'), test_labels_read=False))
    emit('validation_choices_frozen', cases=len(records))


def refit(root, jobs):
    protocol, manifest = prepared(root)
    frozen = io.read_json(root/'results/validation_frozen.json')
    if frozen['protocol_sha256'] != io.digest(root/'protocol.json'):
        raise ValueError('Changed frozen validation choices')
    _, cache = meta_and_cache(root, 10)
    x = np.load(cache/'x10.npy', mmap_mode='r')
    def one(case):
        directory = root/'results'/case['case_id']
        if io.digest(directory/'fit8.json') != frozen['fits'][case['case_id']]:
            raise ValueError('Changed validation choice')
        selected = int(io.read_json(directory/'fit8.json')['metrics']['selected_k'])
        base = baseline(root, case, 10, manifest)
        bg, info = posterior(root, case, 10, manifest)
        xs = np.asarray(x[base['support']], float).T
        r = base['y']-base['c']@base['gamma']-bg['train_score']
        k = min(selected, len(base['support']))
        path = fit_path(xs.T@xs/len(xs), xs.T@r/len(xs), base['alpha'], base['support'], max_k=k)
        alpha = path.coefficients[-1]
        np.testing.assert_array_equal(alpha[path.order[k:]], base['alpha'][path.order[k:]])
        raw_error = 0.
        if k:
            design = xs[:, path.order[:k]]
            delta = np.linalg.lstsq(design, r-xs@base['alpha'], rcond=1e-5)[0]
            raw_error = float(np.max(abs(design@delta-xs@(alpha-base['alpha']))))
            np.testing.assert_allclose(design@delta, xs@(alpha-base['alpha']), rtol=1e-7, atol=1e-8)
        score = xs@alpha
        fixed = base['c']@base['gamma']+bg['train_score']
        row = dict(case_id=case['case_id'], architecture=case['architecture'], setting=case['setting'],
            replicate=case['replicate'], support=len(base['support']), selected_k8=selected, selected_k=k,
            selected_rank=int(path.metrics[-1, 1]), old_sparse_squared_term=float(np.mean(base['old_score']**2)),
            new_sparse_squared_term=float(np.mean(score**2)), old_full_squared_term=float(np.mean((fixed+base['old_score'])**2)),
            new_full_squared_term=float(np.mean((fixed+score)**2)), Q=float(base['Q']),
            baseline_train_mse=float(np.mean((base['y']-fixed-base['old_score'])**2)),
            updated_train_mse=float(np.mean((base['y']-fixed-score)**2)), raw_selected_fit_error=raw_error)
        np.savez(directory/'alpha_effects.npz', support=base['support'], alpha=alpha, old_alpha=base['alpha'],
            order=path.order, selected_k=k, train_alpha_score=score, reestimated_snps=base['support'][path.order[:k]])
        io.atomic_json(directory/'fit10.json', dict(status='solved', metrics=row, diagnostics=path.diagnostics,
            fit8_sha256=frozen['fits'][case['case_id']], background_sha256=info['posterior_sha256'],
            baseline_sha256=info['baseline_sha256'], Q_constraint=False, background_fixed=True,
            gamma_fixed=True, test_labels_used=False, code_hashes=code_hashes()))
        return row
    records = []
    with threadpool_limits(limits=1), ThreadPoolExecutor(max_workers=jobs) as pool:
        for future in as_completed([pool.submit(one, case) for case in protocol['cases']]):
            records.append(future.result())
            if len(records) % 25 == 0:
                emit('combined_refits_ready', cases=len(records), total=len(protocol['cases']))
    pl.DataFrame(records).sort('case_id').write_csv(root/'results/training_metrics.tsv', separator='\t')
    fits = {case['case_id']: dict(fit_sha256=io.digest(root/'results'/case['case_id']/'fit10.json'),
        alpha_sha256=io.digest(root/'results'/case['case_id']/'alpha_effects.npz'),
        background_sha256=io.digest(root/'results'/case['case_id']/'background10.npz')) for case in protocol['cases']}
    io.atomic_json(root/'results/inference_frozen.json', dict(status='frozen', fits=fits,
        protocol_sha256=io.digest(root/'protocol.json'), code_hashes=code_hashes(),
        validation_frozen_sha256=io.digest(root/'results/validation_frozen.json'), test_labels_read=False))
    emit('inference_frozen', cases=len(records))


def predict(root):
    import jax
    jax.config.update('jax_enable_x64', True)
    from .bed_products import packed_bed, PackedBedPredictor, decode_numpy
    protocol, manifest = prepared(root)
    frozen = io.read_json(root/'results/inference_frozen.json')
    if (frozen['code_hashes'] != code_hashes() or frozen['protocol_sha256'] != io.digest(root/'protocol.json')
            or frozen['validation_frozen_sha256'] != io.digest(root/'results/validation_frozen.json')):
        raise ValueError('Changed inference freeze')
    meta, _ = meta_and_cache(root, 10)
    ids = io.table(protocol['cases'][0]['prefix']+'.sparse_prediction.tsv', columns=['iid'])['iid'].to_list()
    if not set(ids).isdisjoint(protocol['fit_ids']):
        raise ValueError('Refit/test samples overlap')
    lookup = {name: i for i, name in enumerate(protocol['fam_ids'])}
    rows = np.asarray([lookup[iid] for iid in ids])
    packed = packed_bed(protocol['bed'], protocol['p'], len(protocol['fam_ids']))
    predictor = PackedBedPredictor(packed, rows, meta['means'], meta['inv'])
    hashes, checks = {}, {}
    for start in range(0, len(protocol['cases']), 48):
        cases = protocol['cases'][start:start+48]
        all_coefs, originals = [], []
        for case in cases:
            directory = root/'results'/case['case_id']
            h = frozen['fits'][case['case_id']]
            if (io.digest(directory/'fit10.json') != h['fit_sha256']
                    or io.digest(directory/'alpha_effects.npz') != h['alpha_sha256']
                    or io.digest(directory/'background10.npz') != h['background_sha256']):
                raise ValueError('Changed fitted effects')
            post = loaded(directory/'alpha_effects.npz')
            base = baseline(root, case, 10, manifest)
            old, delta = np.zeros(protocol['p']), np.zeros(protocol['p'])
            old[post['support']] = post['old_alpha']
            delta[post['support']] = post['alpha']-post['old_alpha']
            bg = posterior(root, case, 10, manifest)[0]['beta']
            all_coefs.extend([bg, base['original_blup'], old, delta])
            originals.append(base)
        coef = np.column_stack(all_coefs)
        prediction = predictor(coef)
        chosen = np.linspace(0, len(ids)-1, 7, dtype=int)
        # Independent CPU decoding checks full contributions, including missing calls.
        expected = np.zeros((len(chosen), coef.shape[1]))
        for snp in range(0, protocol['p'], 4096):
            end = min(snp+4096, protocol['p'])
            tiny = decode_numpy(np.asarray(packed[snp:end]), rows[chosen], meta['means'][snp:end], meta['inv'][snp:end])
            expected += tiny.astype(float).T@coef[snp:end]
        np.testing.assert_allclose(prediction[chosen], expected, rtol=1e-10, atol=1e-11)
        for i, (case, base) in enumerate(zip(cases, originals)):
            native = io.table(case['prefix']+'.sparse_prediction.tsv', ids)
            ss_bg, native_bg, old_alpha, delta = prediction[:, i*4:(i+1)*4].T
            alpha_error = float(np.max(abs(old_alpha-native['lasso_fixed_snp_score'].to_numpy())))
            bg_error = float(np.max(abs(native_bg-native['lasso_background_blup'].to_numpy())))
            if alpha_error > 5e-5 or bg_error > 1e-4:
                raise ArithmeticError('Native test prediction reconstruction failed')
            nuisance = native['lasso_nuisance_fixed_score'].to_numpy()
            ss_full = nuisance+old_alpha+ss_bg
            scale, mean = float(base['phenotype_scale']), float(base['phenotype_mean'])
            path = root/'results'/case['case_id']/'test_predictions.npz'
            np.savez(path, ids=np.asarray(ids), native_full_raw=mean+scale*native['lasso_phenotype_prediction'].to_numpy(),
                baseline_full_raw=mean+scale*ss_full, updated_full_raw=mean+scale*(ss_full+delta),
                original_sparse_score=old_alpha, updated_sparse_score=old_alpha+delta,
                ss_background_score=ss_bg, nuisance_score=nuisance)
            hashes[case['case_id']] = io.digest(path)
            checks[case['case_id']] = dict(native_alpha_max_error=alpha_error, native_blup_max_error=bg_error)
        emit('test_predictions_ready', cases=start+len(cases), total=len(protocol['cases']), n_test=len(ids))
    io.atomic_json(root/'results/predictions_frozen.json', dict(status='frozen', prediction_hashes=hashes,
        inference_frozen_sha256=io.digest(root/'results/inference_frozen.json'),
        native_prediction_checks=checks, independent_decoder_passed=True,
        test_labels_read=False, n_test=len(ids)))


def score(y, p):
    mse = float(np.mean((y-p)**2))
    return dict(corr2=float(np.corrcoef(y, p)[0, 1]**2), mse=mse, r2=1-mse/float(np.var(y)))


def evaluate(root):
    protocol, _ = prepared(root)
    frozen = io.read_json(root/'results/predictions_frozen.json')
    if frozen['inference_frozen_sha256'] != io.digest(root/'results/inference_frozen.json'):
        raise ValueError('Changed frozen predictions')
    records = []
    for case in protocol['cases']:
        directory = root/'results'/case['case_id']
        path = directory/'test_predictions.npz'
        if io.digest(path) != frozen['prediction_hashes'][case['case_id']]:
            raise ValueError('Changed test predictions')
        pred = loaded(path)
        y = io.aligned_phenotype(case['test_pheno'], pred['ids'].tolist())
        native, old, new = [score(y, pred[key]) for key in ['native_full_raw', 'baseline_full_raw', 'updated_full_raw']]
        source = io.read_json(case['original_result'])
        np.testing.assert_allclose(native['corr2'], source['test_corr_squared'], rtol=0, atol=1e-12)
        row = io.read_json(directory/'fit10.json')['metrics']
        row.update(native_corr2=native['corr2'], baseline_corr2=old['corr2'], updated_corr2=new['corr2'],
            gain_corr2=new['corr2']-old['corr2'], gain_vs_native=new['corr2']-native['corr2'],
            ss_gain_vs_native=old['corr2']-native['corr2'], updated_mse=new['mse'], updated_r2=new['r2'])
        bayes_path = Path(case['original_result']).parents[1]/'bayesr/result.json'
        if bayes_path.exists() and io.read_json(bayes_path)['state'] == 'complete':
            row['bayesr_corr2'] = io.read_json(bayes_path)['test_corr_squared']
            row['gain_vs_bayesr'] = new['corr2']-row['bayesr_corr2']
        records.append(row)
    frame = pl.DataFrame(records)
    frame.sort('case_id').write_csv(root/'results/paired_metrics.tsv', separator='\t')
    def summary(rows, arch, setting='all'):
        result = dict(architecture=arch, setting=setting, cases=len(rows),
            wins=sum(r['gain_corr2'] > 1e-12 for r in rows), ties=sum(abs(r['gain_corr2']) <= 1e-12 for r in rows),
            losses=sum(r['gain_corr2'] < -1e-12 for r in rows), median_k=float(np.median([r['selected_k'] for r in rows])))
        fields = ['native_corr2', 'baseline_corr2', 'updated_corr2', 'gain_corr2', 'gain_vs_native',
            'ss_gain_vs_native', 'old_sparse_squared_term', 'new_sparse_squared_term',
            'old_full_squared_term', 'new_full_squared_term', 'Q', 'bayesr_corr2', 'gain_vs_bayesr']
        for field in fields:
            if all(field in row for row in rows):
                result[field] = float(np.mean([row[field] for row in rows]))
        by_rep = defaultdict(list)
        for row in rows:
            by_rep[row['replicate']].append(row['gain_corr2'])
        means = np.asarray([np.mean(v) for v in by_rep.values()])
        if len(means) > 1:
            width = float(student_t.ppf(.975, len(means)-1)*means.std(ddof=1)/np.sqrt(len(means)))
            result.update(gain_ci95_low=float(means.mean()-width), gain_ci95_high=float(means.mean()+width))
        return result
    arches = sorted(set(case['architecture'] for case in protocol['cases']))
    summaries = [summary([r for r in records if r['architecture'] == arch], arch) for arch in arches]
    summaries.append(summary(records, 'all'))
    allocations = [summary([r for r in records if r['architecture'] == arch and r['setting'] == setting], arch, setting)
                   for arch in arches for setting in ['baseline', 'large_low', 'large_high']
                   if any(r['architecture'] == arch and r['setting'] == setting for r in records)]
    pl.DataFrame(summaries).write_csv(root/'results/architecture_summary.tsv', separator='\t')
    pl.DataFrame(allocations).write_csv(root/'results/allocation_summary.tsv', separator='\t')
    io.atomic_json(root/'results/completion.json', dict(status='complete', cases=len(records),
        architectures=arches, n_train=protocol['n_train'], n_validation=protocol['n_validation'],
        n_refit=protocol['n_refit'], n_test=frozen['n_test'], all_backgrounds_converged=True,
        Q_constraint=False, test_labels_used_for_fitting_or_selection=False,
        paired_metrics_sha256=io.digest(root/'results/paired_metrics.tsv'),
        predictions_frozen_sha256=io.digest(root/'results/predictions_frozen.json'),
        protocol_sha256=io.digest(root/'protocol.json'), code_hashes=code_hashes()))
    emit('evaluation_complete', cases=len(records), architecture_summary=summaries)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['prepare', 'background', 'select', 'refit', 'predict', 'evaluate'])
    parser.add_argument('--out-dir', required=True, type=Path)
    parser.add_argument('--study-root', type=Path)
    parser.add_argument('--architectures', nargs='+', default=['high_ld', 'low_ld'])
    parser.add_argument('--replicates', type=int, nargs='+', default=list(range(50)))
    parser.add_argument('--stage', type=int, choices=[8, 10], default=8)
    parser.add_argument('--worker', type=int, default=0)
    parser.add_argument('--workers', type=int, default=1)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--cpu-workers', type=int, default=8)
    args = parser.parse_args()
    if (args.workers < 1 or not 0 <= args.worker < args.workers or args.batch_size < 1
            or args.cpu_workers < 1):
        parser.error('Invalid worker or batch settings')
    root = args.out_dir
    if args.action == 'prepare':
        if args.study_root is None:
            parser.error('prepare requires --study-root')
        prepare(root, args.study_root, args.architectures, args.replicates)
    elif args.action == 'background':
        background(root, args.stage, args.worker, args.workers, args.batch_size)
    elif args.action == 'select':
        select(root, args.cpu_workers)
    elif args.action == 'refit':
        refit(root, args.cpu_workers)
    elif args.action == 'predict':
        predict(root)
    else:
        evaluate(root)


if __name__ == '__main__':
    main()
