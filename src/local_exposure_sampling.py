"""Past-only continuous exposure neighborhoods for warm-click comparisons."""
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import scipy.sparse as sp
from negative import pooled


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def graph(fit, item_count, user_count, basis, neighbors=128):
    """Use user/item exposures only. Self-neighbors are excluded."""
    nu, ni = user_count + 1, item_count + 1
    exposures = sp.coo_matrix((np.ones(len(fit['user']) * 10, np.float32),
                              (fit['user'].repeat(10), fit['items'].ravel())), shape=(nu, ni)).tocsr()
    exposures.data = np.sqrt(exposures.data)
    popularity = np.asarray(exposures.sum(0)).ravel()
    vectors = np.load(basis)['v'][:, 1:17] / np.sqrt(np.maximum(popularity[:, None], 1))
    profiles = np.asarray(exposures @ vectors, dtype=np.float32)
    profiles /= np.maximum(np.linalg.norm(profiles, axis=1, keepdims=True), 1e-8)
    active = np.flatnonzero(np.diff(exposures.indptr) > 0)
    k = min(neighbors, len(active) - 1)
    if k < 1:
        raise ValueError('At least two exposed users are required.')
    result = np.full((nu, k), -1, np.int32)
    candidates = profiles[active].T.copy()
    for start in range(0, len(active), 256):
        ids = active[start:start + 256]
        similarities = profiles[ids] @ candidates
        similarities[np.arange(len(ids)), np.arange(start, start + len(ids))] = -np.inf
        chosen = np.argpartition(-similarities, k - 1, axis=1)[:, :k]
        chosen_scores = np.take_along_axis(similarities, chosen, axis=1)
        for row in range(len(ids)):
            order = np.lexsort((active[chosen[row]], -chosen_scores[row]))
            result[ids[row]] = active[chosen[row, order]]
        if start % 10240 == 0:
            print({'neighbor_users_completed': start, 'total': len(active)}, flush=True)
    assert np.all(result[active] != active[:, None])
    order = np.argsort(fit['user'], kind='stable')
    items = fit['items'][order].ravel().astype(np.int32)
    pointers = np.r_[0, np.cumsum(np.bincount(fit['user'], minlength=nu) * 10)]
    assert (np.diff(pointers)[result[active]] > 0).all()
    return {'neighbors': result, 'user_items': items, 'user_ptr': pointers,
            'active_users': active, 'profiles': profiles}


def build(fit, ni, nu, cut, basis, cache, neighbors=128):
    folder = Path(cache)
    folder.mkdir(parents=True, exist_ok=True)
    original = pooled(fit, ni, nu, cut, basis=basis, cache=folder)
    path = folder / f'local_exposure_{cut}_{neighbors}.npz'
    signature = {'cut': cut, 'neighbors': neighbors, 'basis_sha256': sha(basis),
                 'source_sha256': sha(Path(__file__)),
                 'exposure_sha256': hashlib.sha256(fit['user'].tobytes() + fit['items'].tobytes()).hexdigest()}
    metadata = path.with_suffix('.json')
    if path.exists():
        report = json.loads(metadata.read_text())
        assert report['inputs'] == signature and sha(path) == report['sha256']
        local = dict(np.load(path))
    else:
        if metadata.exists():
            raise ValueError('Missing completed neighborhood archive.')
        started = time.time()
        local = graph(fit, ni, nu, basis, neighbors)
        temporary = path.with_suffix('.partial.npz')
        np.savez_compressed(temporary, **local)
        temporary.replace(path)
        report = {'inputs': signature, 'sha256': sha(path), 'seconds': time.time() - started,
                  'active_users': len(local['active_users']), 'self_neighbors_excluded': True,
                  'query_or_click_labels_used': False}
        metadata.write_text(json.dumps(report, indent=2) + '\n')
    return dict(original, **local)


def sample(pool, users, rng, mixture, n=9):
    """Both mixture arms consume identical RNG draws, including the zero control."""
    if not 0 <= mixture <= 1:
        raise ValueError('Invalid local mixture.')
    shape = (len(users), n)
    choose_neighbor, choose_item, choose_cluster, choose_mixture = [rng.random(shape) for _ in range(4)]
    neighbors = pool['neighbors'][users]
    assert (neighbors >= 0).all()
    selected = np.take_along_axis(neighbors, (choose_neighbor * neighbors.shape[1]).astype(np.int64), axis=1)
    ptr = pool['user_ptr']
    lengths = ptr[selected + 1] - ptr[selected]
    local = pool['user_items'][ptr[selected] + (choose_item * lengths).astype(np.int64)]
    groups = pool['labels'][users]
    ptr = pool['ptr']
    positions = ptr[groups, None] + (choose_cluster * (ptr[groups + 1] - ptr[groups])[:, None]).astype(np.int64)
    cluster = pool['pool'][positions]
    result = np.where(choose_mixture < mixture, local, cluster)
    assert result.shape == shape and (result > 0).all()
    return result
