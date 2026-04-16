"""Unit tests for Step 1 -- Experiment Tag & Config System.

Covers:
- generate_experiment_tag() in src/utils/data_utils.py
- create_run_dir() in src/utils/data_utils.py
- _load_yaml_config() (standalone reimplementation tested in isolation)
- All YAML config files in configs/ parse correctly and contain required keys
- YAML-then-CLI merge logic

All tests are low-cost: no GPU, no model loading, no dataset I/O.
"""
import os
import sys
import tempfile
import types

import pytest
import yaml

# ---------------------------------------------------------------------------
# Path setup -- modules live under src/
# ---------------------------------------------------------------------------

_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
_SRC_DIR = os.path.join(_PROJECT_ROOT, "src")

if _SRC_DIR not in sys.path:
    sys.path.insert(0, _SRC_DIR)


# ---------------------------------------------------------------------------
# Lazy imports -- defer until test body so import errors surface per-test
# ---------------------------------------------------------------------------

def _import_generate_experiment_tag():
    from utils.data_utils import generate_experiment_tag
    return generate_experiment_tag


def _import_create_run_dir():
    from utils.data_utils import create_run_dir
    return create_run_dir


def _standalone_load_yaml_config(path: str) -> dict:
    """Standalone reimplementation matching both finetuning.py and id_ood_analysis.py."""
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Config file not found: {path}")
    with open(path) as f:
        cfg = yaml.safe_load(f)
    if not isinstance(cfg, dict):
        raise ValueError(
            f"Config file must be a YAML mapping, got {type(cfg).__name__}: {path}"
        )
    return cfg


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def base_config() -> dict:
    return {"epochs": 50, "lr": 1e-5, "n_augmentations": 25}


@pytest.fixture
def tmp_dir():
    with tempfile.TemporaryDirectory() as d:
        yield d


# ---------------------------------------------------------------------------
# generate_experiment_tag
# ---------------------------------------------------------------------------

class TestGenerateExperimentTag:

    def test_format_has_correct_prefix(self, base_config):
        gen = _import_generate_experiment_tag()
        tag = gen("nhp", "optimization", base_config)
        assert tag.startswith("nhp-optimization-")

    def test_hash_is_exactly_5_chars(self, base_config):
        gen = _import_generate_experiment_tag()
        tag = gen("nhp", "optimization", base_config)
        hash_part = tag[len("nhp-optimization-"):]
        assert len(hash_part) == 5

    def test_hash_is_lowercase_hex(self, base_config):
        gen = _import_generate_experiment_tag()
        tag = gen("nhp", "optimization", base_config)
        hash_part = tag[len("nhp-optimization-"):]
        assert all(c in "0123456789abcdef" for c in hash_part)

    def test_deterministic_same_config(self, base_config):
        gen = _import_generate_experiment_tag()
        assert gen("nhp", "optimization", base_config) == gen("nhp", "optimization", base_config)

    def test_deterministic_insertion_order_invariant(self):
        gen = _import_generate_experiment_tag()
        cfg_a = {"epochs": 50, "lr": 1e-5}
        cfg_b = {"lr": 1e-5, "epochs": 50}
        assert gen("nhp", "optimization", cfg_a) == gen("nhp", "optimization", cfg_b)

    def test_different_configs_produce_different_hashes(self):
        gen = _import_generate_experiment_tag()
        tag_a = gen("nhp", "optimization", {"epochs": 50})
        tag_b = gen("nhp", "optimization", {"epochs": 100})
        assert tag_a != tag_b

    def test_different_datasets_produce_different_tags(self, base_config):
        gen = _import_generate_experiment_tag()
        assert gen("nhp", "optimization", base_config) != gen("rat", "optimization", base_config)

    def test_different_families_produce_different_tags(self, base_config):
        gen = _import_generate_experiment_tag()
        assert gen("nhp", "optimization", base_config) != gen("nhp", "lora-ablation", base_config)

    def test_empty_config_accepted(self):
        gen = _import_generate_experiment_tag()
        tag = gen("rat", "optimization", {})
        assert tag.startswith("rat-optimization-")

    def test_hyphenated_family_tag_format(self):
        gen = _import_generate_experiment_tag()
        tag = gen("nhp", "lora-ablation", {"epochs": 50})
        assert tag.startswith("nhp-lora-ablation-")
        # Hash is the last 5 chars
        hash_part = tag.rsplit("-", 1)[-1]
        assert len(hash_part) == 5


# ---------------------------------------------------------------------------
# create_run_dir
# ---------------------------------------------------------------------------

class TestCreateRunDir:

    def test_with_tag_returns_expected_path(self, tmp_dir):
        create_run_dir = _import_create_run_dir()
        run_dir = create_run_dir("legacy_tag", base_dir=tmp_dir, tag="nhp-optimization-a3f9c")
        assert run_dir == os.path.join(tmp_dir, "nhp-optimization-a3f9c")

    def test_with_tag_directory_exists(self, tmp_dir):
        create_run_dir = _import_create_run_dir()
        run_dir = create_run_dir("legacy", base_dir=tmp_dir, tag="nhp-optimization-a3f9c")
        assert os.path.isdir(run_dir)

    def test_with_tag_all_subdirs_created(self, tmp_dir):
        create_run_dir = _import_create_run_dir()
        run_dir = create_run_dir("legacy", base_dir=tmp_dir, tag="my-tag")
        for sub in ("fitness", "fitness/emg_maps", "optimization",
                     "optimization/emg_maps", "results", "diagnostics"):
            assert os.path.isdir(os.path.join(run_dir, sub)), f"Missing subdir: {sub}"

    def test_without_tag_uses_exp_tag_prefix(self, tmp_dir):
        create_run_dir = _import_create_run_dir()
        run_dir = create_run_dir("legacy_exp", base_dir=tmp_dir, tag=None)
        assert os.path.basename(run_dir).startswith("legacy_exp_")

    def test_without_tag_timestamp_suffix_numeric(self, tmp_dir):
        create_run_dir = _import_create_run_dir()
        run_dir = create_run_dir("legacy_exp", base_dir=tmp_dir, tag=None)
        dir_name = os.path.basename(run_dir)
        timestamp_part = dir_name[len("legacy_exp_"):]
        assert timestamp_part.replace("_", "").isdigit()

    def test_without_tag_subdirs_created(self, tmp_dir):
        create_run_dir = _import_create_run_dir()
        run_dir = create_run_dir("legacy_exp", base_dir=tmp_dir, tag=None)
        for sub in ("fitness", "optimization", "results", "diagnostics"):
            assert os.path.isdir(os.path.join(run_dir, sub))

    def test_idempotent_second_call_does_not_raise(self, tmp_dir):
        create_run_dir = _import_create_run_dir()
        create_run_dir("x", base_dir=tmp_dir, tag="my-tag-00000")
        create_run_dir("x", base_dir=tmp_dir, tag="my-tag-00000")  # no error


# ---------------------------------------------------------------------------
# _load_yaml_config (standalone reimplementation)
# ---------------------------------------------------------------------------

class TestLoadYamlConfig:

    def test_valid_mapping_returns_dict(self, tmp_dir):
        path = os.path.join(tmp_dir, "cfg.yaml")
        with open(path, "w") as f:
            yaml.dump({"dataset": "nhp", "epochs": 50}, f)
        result = _standalone_load_yaml_config(path)
        assert isinstance(result, dict)
        assert result["dataset"] == "nhp"

    def test_missing_file_raises_file_not_found(self, tmp_dir):
        with pytest.raises(FileNotFoundError, match="Config file not found"):
            _standalone_load_yaml_config(os.path.join(tmp_dir, "missing.yaml"))

    def test_list_root_raises_value_error(self, tmp_dir):
        path = os.path.join(tmp_dir, "list.yaml")
        with open(path, "w") as f:
            f.write("- a\n- b\n")
        with pytest.raises(ValueError, match="YAML mapping"):
            _standalone_load_yaml_config(path)

    def test_nested_list_values_preserved(self, tmp_dir):
        path = os.path.join(tmp_dir, "nested.yaml")
        with open(path, "w") as f:
            f.write("mode:\n  - optimization\n  - fit\n")
        result = _standalone_load_yaml_config(path)
        assert result["mode"] == ["optimization", "fit"]

    def test_float_scientific_notation(self, tmp_dir):
        path = os.path.join(tmp_dir, "lr.yaml")
        with open(path, "w") as f:
            f.write("lr: 1.0e-5\n")
        result = _standalone_load_yaml_config(path)
        assert abs(result["lr"] - 1e-5) < 1e-12


# ---------------------------------------------------------------------------
# YAML config files -- parse and required keys
# ---------------------------------------------------------------------------

_CONFIGS_DIR = os.path.join(_PROJECT_ROOT, "configs")

# Step 1 finetuning configs (created by agent)
_FINETUNING_CONFIGS = [
    "nhp_optimization.yaml",
    "nhp_lora_ablation.yaml",
    "nhp_aug_sweep.yaml",
    "nhp_optimization_budget.yaml",
    "rat_optimization.yaml",
]

_FINETUNING_REQUIRED_KEYS = {
    "dataset", "mode", "epochs", "lr", "n_augmentations", "budget", "n_reps",
}

# Pre-existing config with slightly different structure
_PREEXISTING_CONFIGS = ["nhp_fit.yaml"]

_ID_OOD_CONFIGS = [
    "id_ood_gp.yaml",
    "id_ood_noise.yaml",
    "id_ood_prior_bag.yaml",
    "id_ood_all_priors.yaml",
]

_ID_OOD_REQUIRED_KEYS = {
    "datasets", "analyses", "prior_source", "device", "n_synthetic", "n_context", "seed",
}


@pytest.mark.parametrize("config_name", _FINETUNING_CONFIGS)
def test_finetuning_config_parses(config_name):
    path = os.path.join(_CONFIGS_DIR, config_name)
    assert os.path.isfile(path), f"Config file missing: {path}"
    with open(path) as f:
        cfg = yaml.safe_load(f)
    assert isinstance(cfg, dict)


@pytest.mark.parametrize("config_name", _FINETUNING_CONFIGS)
def test_finetuning_config_required_keys(config_name):
    with open(os.path.join(_CONFIGS_DIR, config_name)) as f:
        cfg = yaml.safe_load(f)
    missing = _FINETUNING_REQUIRED_KEYS - set(cfg.keys())
    assert not missing, f"{config_name} missing keys: {missing}"


@pytest.mark.parametrize("config_name", _FINETUNING_CONFIGS)
def test_finetuning_config_dataset_valid(config_name):
    with open(os.path.join(_CONFIGS_DIR, config_name)) as f:
        cfg = yaml.safe_load(f)
    assert cfg["dataset"] in {"nhp", "rat", "spinal"}


@pytest.mark.parametrize("config_name", _FINETUNING_CONFIGS)
def test_finetuning_config_budget_positive_int(config_name):
    with open(os.path.join(_CONFIGS_DIR, config_name)) as f:
        cfg = yaml.safe_load(f)
    assert isinstance(cfg["budget"], int) and cfg["budget"] > 0


@pytest.mark.parametrize("config_name", _ID_OOD_CONFIGS)
def test_id_ood_config_parses(config_name):
    path = os.path.join(_CONFIGS_DIR, config_name)
    assert os.path.isfile(path), f"Config file missing: {path}"
    with open(path) as f:
        cfg = yaml.safe_load(f)
    assert isinstance(cfg, dict)


@pytest.mark.parametrize("config_name", _ID_OOD_CONFIGS)
def test_id_ood_config_required_keys(config_name):
    with open(os.path.join(_CONFIGS_DIR, config_name)) as f:
        cfg = yaml.safe_load(f)
    missing = _ID_OOD_REQUIRED_KEYS - set(cfg.keys())
    assert not missing, f"{config_name} missing keys: {missing}"


@pytest.mark.parametrize("config_name", _ID_OOD_CONFIGS)
def test_id_ood_config_seed_is_int(config_name):
    with open(os.path.join(_CONFIGS_DIR, config_name)) as f:
        cfg = yaml.safe_load(f)
    assert isinstance(cfg["seed"], int)


# ---------------------------------------------------------------------------
# CLI arg merge logic (isolated from heavy deps)
# ---------------------------------------------------------------------------

class TestYamlCliMergeLogic:
    """Replicate the YAML-then-CLI override pattern from run_finetuning()."""

    def _simulate_merge(self, yaml_cfg: dict, cli_args: dict) -> dict:
        args = types.SimpleNamespace(**cli_args)
        _bool_flags = {"save", "diagnostics", "lora"}
        for key, value in yaml_cfg.items():
            if key in _bool_flags:
                if not getattr(args, key, False):
                    setattr(args, key, value)
            elif getattr(args, key, None) is None:
                setattr(args, key, value)
        return vars(args)

    def test_yaml_applied_when_cli_is_none(self):
        merged = self._simulate_merge({"epochs": 50}, {"epochs": None})
        assert merged["epochs"] == 50

    def test_cli_overrides_yaml_when_not_none(self):
        merged = self._simulate_merge({"epochs": 50}, {"epochs": 100})
        assert merged["epochs"] == 100

    def test_yaml_bool_applied_when_cli_is_false(self):
        merged = self._simulate_merge({"save": True}, {"save": False})
        assert merged["save"] is True

    def test_cli_bool_true_not_overridden(self):
        merged = self._simulate_merge({"save": False}, {"save": True})
        assert merged["save"] is True

    def test_multiple_keys_merged(self):
        merged = self._simulate_merge(
            {"epochs": 50, "lr": 1e-5, "dataset": "nhp"},
            {"epochs": None, "lr": None, "dataset": None},
        )
        assert merged == {"epochs": 50, "lr": 1e-5, "dataset": "nhp"}

    def test_partial_override(self):
        merged = self._simulate_merge(
            {"epochs": 50, "lr": 1e-5},
            {"epochs": 200, "lr": None},
        )
        assert merged["epochs"] == 200
        assert merged["lr"] == 1e-5
