"""Resolve relocated local models without rewriting immutable score provenance.

The generated, repository-relative map is optional. New configurations use
canonical per-task paths; archived reports/configs can retain their exact bytes.
"""
import json
from functools import lru_cache
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
MANIFEST = REPO / 'fomofd/tasks/model_locations.json'


@lru_cache(maxsize=4)
def _read_manifest(filename, modified_ns, size):
    data = json.loads(Path(filename).read_text())
    if data.get('schema_version') != 1:
        raise ValueError('Unknown model relocation manifest schema')
    return data


def resolve_model_path(value, base=None, *, manifest=None):
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = (Path(base) if base is not None else REPO) / path
    path = path.resolve()
    mapping = Path(manifest) if manifest is not None else MANIFEST
    if not mapping.is_file():
        return path
    st = mapping.stat()
    data = _read_manifest(str(mapping), st.st_mtime_ns, st.st_size)
    root = Path(data.get('repo_root', REPO)).resolve()
    try:
        relative = path.relative_to(root).as_posix()
    except ValueError:
        return path
    # Exact files override directory mappings, e.g. original PACE config.json.
    for old, new in sorted(data['relocations'].items(), key=lambda p: len(p[0]), reverse=True):
        if relative == old or relative.startswith(old + '/'):
            target = (root / (new + relative[len(old):])).resolve()
            if not target.is_relative_to(root / 'fomofd/tasks'):
                raise ValueError('Model relocation target must stay under fomofd/tasks')
            return target
    return path


def relocate_identity(value):
    """Compare score identities modulo recorded path-only model moves."""
    if isinstance(value, dict):
        result = {}
        for key, item in value.items():
            relocated = relocate_identity(key)
            if relocated in result and result[relocated] != relocate_identity(item):
                raise ValueError('Conflicting provenance after model relocation')
            result[relocated] = relocate_identity(item)
        return result
    if isinstance(value, (list, tuple)):
        return [relocate_identity(v) for v in value]
    if isinstance(value, str) and '/fomofd/tasks/' in value:
        return str(resolve_model_path(value, REPO / 'lerobot'))
    return value
