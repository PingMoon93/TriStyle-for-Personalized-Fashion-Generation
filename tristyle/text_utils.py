"""Small dependency helpers used by the standalone Text-V1.1 release."""
from __future__ import annotations
import hashlib
import json
import re
from pathlib import Path
import numpy as np


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def fingerprint(obj):
    return hashlib.sha256(json.dumps(obj, sort_keys=True).encode()).hexdigest()


def integer_id(value):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (str, int, np.integer)):
        raise ValueError(f'ID must be a nonnegative integer, got {value!r}')
    if not re.fullmatch(r'[0-9]+', str(value)):
        raise ValueError(f'Invalid ID: {value!r}')
    return int(value)


def trusted_npy(path):
    """Load only trusted local protocol assets; object arrays use Python pickle."""
    array = np.load(path, allow_pickle=True)
    return array.item() if array.shape == () else array
