import pytest

from engagecam.features import MemorySource
from engagecam.manifest import validate_manifest
from engagecam.pipeline import compute_features
from engagecam.synthetic import make_dataset


@pytest.fixture(scope="session")
def data():
    manifest, images = make_dataset(n_subjects=40, clips_per_subject=4, frames_per_clip=4, size=48, seed=2)
    return validate_manifest(manifest), images


@pytest.fixture(scope="session")
def manifest(data):
    return data[0]


@pytest.fixture(scope="session")
def source(data):
    return MemorySource(data[1])


@pytest.fixture(scope="session")
def feats(manifest, source):
    return compute_features(manifest, source)
