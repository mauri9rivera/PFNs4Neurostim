"""Unit tests for diagnostics serialization in src/finetuning.py (C6).

Covers:
- load_diagnostics(): round-trip fidelity, key preservation, missing-file error
- CSV export: column naming via pd.json_normalize, row count per epoch

No GPU, no real data, no model loading — uses only synthetic _diagnostics_ dicts.
"""
import os
import pickle
import sys
from typing import Any

import numpy as np
import pandas as pd
import pytest

# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------

_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
_SRC_DIR = os.path.join(_PROJECT_ROOT, "src")

if _SRC_DIR not in sys.path:
    sys.path.insert(0, _SRC_DIR)


# ---------------------------------------------------------------------------
# Synthetic diagnostics helper
# ---------------------------------------------------------------------------

def _make_fake_diagnostics(n_epochs: int = 5) -> list[dict[str, Any]]:
    """Build a synthetic _diagnostics_ list matching GradientMonitoredRegressor output.

    Args:
        n_epochs: Number of per-epoch entries to generate.

    Returns:
        List of per-epoch dicts with the exact key structure produced by
        GradientMonitoredRegressor._log_epoch_evaluation().
    """
    layer_groups = ["transformer_encoder", "decoder_dict", "attention_dict"]
    cka_layers = [
        "transformer_encoder.layers.0",
        "transformer_encoder.layers.4",
        "transformer_encoder.layers.9",
        "transformer_encoder.layers.13",
        "transformer_encoder.layers.17",
    ]
    rng = np.random.RandomState(42)
    return [
        {
            "epoch": i,
            "grad_norm": {g: float(rng.rand()) for g in layer_groups},
            "grad_weight_ratio": {g: float(rng.rand()) for g in layer_groups},
            "update_to_param_ratio": {g: float(rng.rand()) for g in layer_groups},
            "weight_displacement": {g: float(rng.rand()) for g in layer_groups},
            "cosine_similarity": {g: float(rng.uniform(0.9, 1.0)) for g in layer_groups},
            "cka": {k: float(rng.uniform(0.8, 1.0)) for k in cka_layers},
        }
        for i in range(n_epochs)
    ]


# ---------------------------------------------------------------------------
# Lazy import
# ---------------------------------------------------------------------------

def _import_load_diagnostics():
    from finetuning import load_diagnostics
    return load_diagnostics


# ---------------------------------------------------------------------------
# TestLoadDiagnostics
# ---------------------------------------------------------------------------

class TestLoadDiagnostics:
    """Tests for load_diagnostics() round-trip correctness and error handling."""

    def test_roundtrip_restores_list_length(self, tmp_path):
        """Pickled diagnostics round-trip returns a list with the same length."""
        load_diagnostics = _import_load_diagnostics()
        n_epochs = 7
        diag = _make_fake_diagnostics(n_epochs)

        diag_dir = tmp_path / "diagnostics"
        diag_dir.mkdir()
        with open(diag_dir / "diagnostics.pkl", "wb") as f:
            pickle.dump(diag, f, protocol=pickle.HIGHEST_PROTOCOL)

        loaded = load_diagnostics(str(tmp_path))
        assert len(loaded) == n_epochs

    def test_roundtrip_preserves_epoch_keys(self, tmp_path):
        """Each round-tripped entry contains all expected top-level keys."""
        load_diagnostics = _import_load_diagnostics()
        diag = _make_fake_diagnostics(3)

        diag_dir = tmp_path / "diagnostics"
        diag_dir.mkdir()
        with open(diag_dir / "diagnostics.pkl", "wb") as f:
            pickle.dump(diag, f, protocol=pickle.HIGHEST_PROTOCOL)

        loaded = load_diagnostics(str(tmp_path))
        expected_keys = {
            "epoch", "grad_norm", "grad_weight_ratio",
            "update_to_param_ratio", "weight_displacement",
            "cosine_similarity", "cka",
        }
        for entry in loaded:
            assert expected_keys == set(entry.keys())

    def test_roundtrip_preserves_float_values(self, tmp_path):
        """Numeric values survive the pickle round-trip without drift."""
        load_diagnostics = _import_load_diagnostics()
        diag = _make_fake_diagnostics(4)

        diag_dir = tmp_path / "diagnostics"
        diag_dir.mkdir()
        with open(diag_dir / "diagnostics.pkl", "wb") as f:
            pickle.dump(diag, f, protocol=pickle.HIGHEST_PROTOCOL)

        loaded = load_diagnostics(str(tmp_path))
        for orig, rt in zip(diag, loaded):
            for layer in orig["grad_weight_ratio"]:
                assert orig["grad_weight_ratio"][layer] == rt["grad_weight_ratio"][layer]
            for layer in orig["cka"]:
                assert orig["cka"][layer] == rt["cka"][layer]

    def test_missing_file_raises_file_not_found_error(self, tmp_path):
        """FileNotFoundError is raised when diagnostics.pkl is absent, with path in message."""
        load_diagnostics = _import_load_diagnostics()
        with pytest.raises(FileNotFoundError, match="diagnostics.pkl"):
            load_diagnostics(str(tmp_path))


# ---------------------------------------------------------------------------
# TestDiagnosticsCSV
# ---------------------------------------------------------------------------

class TestDiagnosticsCSV:
    """Tests for the CSV export produced by pd.json_normalize on _diagnostics_."""

    def _write_csv(self, diag: list[dict[str, Any]], csv_path: str) -> None:
        """Replicate the exact CSV export logic from finetune_tabpfn()."""
        pd.json_normalize(diag, sep="__").to_csv(csv_path, index=False)

    def test_csv_row_count_equals_n_epochs(self, tmp_path):
        """CSV has exactly n_epochs data rows (one per epoch)."""
        n_epochs = 6
        diag = _make_fake_diagnostics(n_epochs)
        csv_path = str(tmp_path / "diagnostics.csv")
        self._write_csv(diag, csv_path)

        df = pd.read_csv(csv_path)
        assert len(df) == n_epochs

    def test_csv_contains_epoch_column_with_sequential_values(self, tmp_path):
        """CSV 'epoch' column contains sequential integers starting from 0."""
        n_epochs = 4
        diag = _make_fake_diagnostics(n_epochs)
        csv_path = str(tmp_path / "diagnostics.csv")
        self._write_csv(diag, csv_path)

        df = pd.read_csv(csv_path)
        assert "epoch" in df.columns
        assert list(df["epoch"]) == list(range(n_epochs))

    def test_csv_contains_grad_weight_ratio_columns(self, tmp_path):
        """CSV contains flattened grad_weight_ratio columns for each layer group."""
        diag = _make_fake_diagnostics(2)
        csv_path = str(tmp_path / "diagnostics.csv")
        self._write_csv(diag, csv_path)

        df = pd.read_csv(csv_path)
        for col in [
            "grad_weight_ratio__transformer_encoder",
            "grad_weight_ratio__decoder_dict",
            "grad_weight_ratio__attention_dict",
        ]:
            assert col in df.columns, f"Missing column: {col}"

    def test_csv_contains_cka_column_with_full_layer_name(self, tmp_path):
        """CSV contains CKA column with full dotted layer path name."""
        diag = _make_fake_diagnostics(2)
        csv_path = str(tmp_path / "diagnostics.csv")
        self._write_csv(diag, csv_path)

        df = pd.read_csv(csv_path)
        assert "cka__transformer_encoder.layers.17" in df.columns
