"""Unit tests for data processing modules."""

import numpy as np

from src.data.preprocessing import preprocess, convert_lower_case, remove_punctuation
from src.data.dataset import SummaryDataset, summary_collate_fn


class TestPreprocessing:
    """Tests for text preprocessing functions."""

    def test_lowercase(self):
        result = convert_lower_case("Hello WORLD")
        assert str(result) == "hello world"

    def test_punctuation_removal(self):
        result = remove_punctuation(np.array("hello! world?"))
        assert "!" not in str(result)
        assert "?" not in str(result)

    def test_full_pipeline(self):
        result = preprocess("Hello, World! This is a TEST.")
        result_str = str(result)
        assert result_str == result_str.lower()


class TestSummaryDataset:
    """Tests for the SummaryDataset class."""

    def test_length(self):
        ds = SummaryDataset(
            sources=[["a", "b"], ["c"]],
            targets=[["x"], ["y"]],
            labels=[[1, 0], [1]],
        )
        assert len(ds) == 2

    def test_getitem(self):
        ds = SummaryDataset(
            sources=[["a", "b"]],
            targets=[["x"]],
            labels=[[1, 0]],
        )
        src, tgt, lbl = ds[0]
        assert src == ["a", "b"]
        assert tgt == ["x"]
        assert lbl == [1, 0]


class TestCollate:
    """Tests for the collate function."""

    def test_padding(self):
        batch = [
            (["a", "b", "c"], ["x"], [1, 0, 1]),
            (["d"], ["y", "z"], [0]),
        ]
        sources, targets, labels = summary_collate_fn(batch)
        assert len(sources[0]) == len(sources[1])  # Same length after padding
        assert len(targets[0]) == len(targets[1])
        assert len(labels[0]) == len(labels[1])
