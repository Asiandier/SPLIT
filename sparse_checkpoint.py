"""Atomic, input-bound checkpoints at sparse outer-iteration boundaries.

Only numerical state is serialized, using JSON and non-object NPY arrays.
Genotypes and transient device workspaces are rebuilt; information factors,
PCG warm starts and the actual preconditioner survive a restart.
"""
from __future__ import annotations

import fcntl
import hashlib
import json
import logging
import math
import os
from pathlib import Path
import tempfile
import time
import zipfile

import jax
import jax.numpy as jnp
import numpy as np

from .precond import ProjectedCorePrecondConf
from .sparse_information import SparseMeanInformation
from .sparse_reml import SparseProjectedInformation

logger = logging.getLogger(__name__)
SCHEMA = 1

# Explicit restart contract. Transient candidate designs, operators and CUDA
# buffers are deliberately absent. Keep the final-alignment fallback as well.
STATE_NAMES = (
    'outer', 'theta', 'theta_lasso', 'support', 'candidate_cache',
    'previous_fixed_mean', 'previous_outer_h2', 'history', 'warm_screen',
    'warm_z_dict', 'warm_lasso_candidate', 'warm_lasso_beta_path',
    'sparse_path_performance', 'final_candidate', 'final_lasso',
    'variance_blocks_completed', 'outer_converged', 'outer_stop_reason',
    'final_alignment_pending', 'provisional_convergence',
    'final_alignment_completed', 'final_pair_available', 'final_pair_source',
    'final_alignment_warning', 'last_aligned_pair',
    'last_covariance_reml_diagnostics', 'final_information',
    'information_markers', 'mean_information', 'lasso_reml_stop_reason',
    'penalized_failure_reason', 'final_kkt_certificate',
    'iterative_validation_trace', 'lasso_warm_state_in_summary',
)


def _file_signature(path, *, large=False):
    p = Path(path).resolve(strict=True)
    stat = p.stat()
    out = dict(path=str(p), size=stat.st_size)
    if large:
        out['mtime_ns'] = stat.st_mtime_ns
    else:
        h = hashlib.sha256()
        with p.open('rb') as handle:
            for chunk in iter(lambda: handle.read(8 << 20), b''):
                h.update(chunk)
        out['sha256'] = h.hexdigest()
    return out


def run_identity(args, *, bed_list, prediction_bed_list):
    """Bind a checkpoint to code, parameters and both fitting/validation input.

    As in the public pipeline, large genotype bodies use size+mtime; their
    sample/variant metadata and all small fitting inputs use content hashes.
    """
    values = dict(vars(args))
    values.pop('checkpoint', None)
    values.pop('verbose', None)
    files = {}
    for name in ('pheno_txt', 'covar_txt', 'keep_path', 'component_spec',
                 'sparsity_validation_pheno_txt', 'prediction_covar_txt',
                 'prediction_keep_path', 'lasso_warm_state_in'):
        if values.get(name):
            files[name] = _file_signature(values[name])
    for prefix in sorted(set(bed_list + prediction_bed_list)):
        for suffix in ('.bed', '.bim', '.fam'):
            files[prefix+suffix] = _file_signature(prefix+suffix, large=suffix == '.bed')
    for name in ('pgen_prefix', 'prediction_pgen_prefix'):
        prefix = values.get(name)
        if prefix:
            for suffix in ('.pgen', '.pvar', '.pvar.zst', '.psam'):
                if Path(prefix+suffix).is_file():
                    files[prefix+suffix] = _file_signature(prefix+suffix, large=suffix == '.pgen')
    digest = hashlib.sha256()
    for p in sorted(Path(__file__).resolve().parent.iterdir()):
        if p.suffix in ('.py', '.cc', '.cu'):
            digest.update(p.name.encode()+b'\0')
            digest.update(hashlib.sha256(p.read_bytes()).digest())
    return dict(arguments=values, files=files, source_sha256=digest.hexdigest(),
                numpy_version=np.__version__, jax_version=jax.__version__,
                sparse_cublas_library=os.environ.get('SPLIT_CUBLAS_LIBRARY'))


class _Encoder:
    def __init__(self):
        self.nodes, self.arrays, self.memo = [], {}, {}

    def encode(self, value):
        if isinstance(value, np.generic):
            value = value.item()
        if value is None or isinstance(value, (str, bool, int)):
            return value
        if isinstance(value, float):
            return value if math.isfinite(value) else {'float': repr(value)}
        key = id(value)
        if key in self.memo:
            return {'ref': self.memo[key]}
        index = len(self.nodes)
        self.memo[key] = index
        self.nodes.append(None)
        if isinstance(value, (np.ndarray, jax.Array)):
            if np.dtype(value.dtype).hasobject:
                raise TypeError('Object arrays are not permitted in checkpoints')
            name = f'array_{index}'
            self.arrays[name] = value
            node = dict(type='array', name=name, device=isinstance(value, jax.Array))
        elif isinstance(value, dict):
            node = dict(type='dict', items=[(self.encode(k), self.encode(v)) for k, v in value.items()])
        elif isinstance(value, (tuple, list)):
            node = dict(type='tuple' if isinstance(value, tuple) else 'list',
                        items=[self.encode(v) for v in value])
        elif isinstance(value, SparseProjectedInformation):
            if value.q64 is not None:
                raise ValueError('Checkpoint boundary retains a live projected-likelihood cache')
            node = dict(type='projected_information', exact=self.encode(value.exact),
                        slq_workspace_bytes=value.slq_workspace_bytes)
        elif isinstance(value, SparseMeanInformation):
            attributes = dict(vars(value))
            if attributes['_state'] is not None:
                # Contains callable identities, not additional numerical state.
                attributes['_state'] = {k: v for k, v in attributes['_state'].items()
                                        if k != 'score_trace_cache'}
            node = dict(type='mean_information', attributes=self.encode(attributes))
        elif isinstance(value, ProjectedCorePrecondConf):
            node = dict(type='preconditioner', attributes=self.encode(dict(vars(value))))
        else:
            raise TypeError(f'Unsupported checkpoint value: {type(value).__name__}')
        self.nodes[index] = node
        # Retain temporary dictionaries until encoding ends: their IDs cannot
        # be recycled and confused with earlier nodes in the shared-state map.
        if not hasattr(self, '_objects'):
            self._objects = []
        self._objects.append(value)
        return {'ref': index}


def _decode(metadata, archive):
    cache = {}

    def decode(value):
        if not isinstance(value, dict):
            return value
        if 'float' in value:
            if value['float'] not in ('inf', '-inf', 'nan'):
                raise ValueError('Invalid checkpoint floating-point tag')
            return float(value['float'])
        index = value['ref']
        if index in cache:
            return cache[index]
        node = metadata['nodes'][index]
        kind = node['type']
        if kind == 'array':
            result = archive[node['name']]
            if node['device']:
                with jax.experimental.enable_x64():
                    result = jnp.asarray(result)
        elif kind == 'dict':
            result = {}
            cache[index] = result
            result.update((decode(k), decode(v)) for k, v in node['items'])
        elif kind in ('tuple', 'list'):
            result = [decode(v) for v in node['items']]
            if kind == 'tuple':
                result = tuple(result)
        elif kind == 'mean_information':
            result = SparseMeanInformation.__new__(SparseMeanInformation)
            result.__dict__.update(decode(node['attributes']))
            if (result.basis.shape != (result.n, result.rank)
                    or result.covar.shape[0] != result.n):
                raise ValueError('Invalid checkpoint information dimensions')
        elif kind == 'projected_information':
            result = SparseProjectedInformation(
                decode(node['exact']), slq_workspace_bytes=node.get('slq_workspace_bytes'))
        elif kind == 'preconditioner':
            result = ProjectedCorePrecondConf(**decode(node['attributes']))
        else:
            raise ValueError(f'Unknown checkpoint node: {kind}')
        cache[index] = result
        return result

    return decode(metadata['state'])


class OuterCheckpoint:
    """One durable generation per native fit; no pickle or untrusted code."""
    def __init__(self, path, identity):
        self.path = Path(path)
        self.identity = identity
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.lock = self.path.with_suffix(self.path.suffix+'.lock').open('a')
        try:
            fcntl.flock(self.lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BaseException:
            self.lock.close()
            raise
        # A killed writer cannot publish its temporary file. Remove only our
        # own stale temporaries, under the exclusive writer lock.
        for p in self.path.parent.glob('.'+self.path.name+'.*.tmp'):
            p.unlink()

    def load(self):
        if not self.path.exists():
            return None
        with np.load(self.path, allow_pickle=False) as archive:
            meta = json.loads(archive['metadata'].tobytes().decode())
            if meta['schema'] != SCHEMA or meta['identity'] != self.identity:
                raise ValueError('Checkpoint code, settings or inputs do not match this run')
            state = _decode(meta, archive)
        logger.info('[checkpoint] restored outer=%s stage=%s from %s',
                    state['outer'], state['checkpoint_stage'], self.path)
        return state

    def save(self, state):
        started = time.perf_counter()
        encoder = _Encoder()
        tree = encoder.encode(state)
        meta = dict(schema=SCHEMA, identity=self.identity, state=tree,
                    nodes=encoder.nodes, saved_at=time.time())
        temporary = None
        try:
            with tempfile.NamedTemporaryFile(dir=self.path.parent, prefix='.'+self.path.name+'.',
                                             suffix='.tmp', delete=False) as handle:
                temporary = Path(handle.name)
                # Uncompressed, streamed arrays avoid CPU compression and a
                # second whole-checkpoint memory buffer. ZIP provides per-array CRC.
                with zipfile.ZipFile(handle, 'w', compression=zipfile.ZIP_STORED, allowZip64=True) as archive:
                    values = dict(encoder.arrays, metadata=np.frombuffer(
                        json.dumps(meta, allow_nan=False, separators=(',', ':')).encode(), dtype=np.uint8))
                    for name, array in values.items():
                        with archive.open(name+'.npy', 'w', force_zip64=True) as entry:
                            np.lib.format.write_array(entry, np.asarray(array), allow_pickle=False)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary, self.path)
            directory = os.open(self.path.parent, os.O_RDONLY | os.O_DIRECTORY)
            try:
                os.fsync(directory)
            finally:
                os.close(directory)
        finally:
            if temporary is not None:
                temporary.unlink(missing_ok=True)
        logger.info('[checkpoint] saved outer=%s stage=%s bytes=%s time=%.2fs path=%s',
                    state['outer'], state['checkpoint_stage'], self.path.stat().st_size,
                    time.perf_counter()-started, self.path)

    def close(self):
        self.lock.close()


def capture_state(namespace, *, stage):
    state = {name: namespace[name] for name in STATE_NAMES}
    if stage == 'fit_complete':
        # The resumed pipeline enters final inference directly at this stage.
        # Only final_information is used there; candidate inverse columns and
        # the fallback pair serve the outer loop.
        state.update(warm_z_dict={}, last_aligned_pair=None, mean_information=None)
    state.update(checkpoint_stage=stage,
                 preconditioner=namespace['fitter'].precond_conf,
                 execution=dict(call_width=namespace['call_width'],
                     precond_rank=namespace['plan'].precond_rank,
                     ring_depth=namespace['plan'].ring_depth))
    return state
