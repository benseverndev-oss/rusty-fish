import pathlib
import sys

import pytest

sys.path.insert(0, str(pathlib.Path(__file__).parent))
from train_nnue import _load_samples


def test_load_samples_rejects_mixed_schema_or_feature_dimension(tmp_path):
    path = tmp_path / "mixed.tsv"
    path.write_text("v1\t1\t0\t\nHalfKA\t1\t0\t\n", encoding="utf-8")

    with pytest.raises(ValueError, match="schema"):
        _load_samples(path, expected_schema="v1", input_dimension=768)
