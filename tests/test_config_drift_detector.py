"""Tests for the static configuration drift detector."""

from pathlib import Path

from tools.config_drift_detector import ConfigDriftDetector


ROOT = Path(__file__).resolve().parents[1]


def _write(path: Path, text: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return path


def _write_mapping_fixture(root: Path, lora_source: str, molora_source: str, default_yaml: str) -> None:
    _write(root / "ultralytics/cfg/default.yaml", default_yaml)
    _write(root / "ultralytics/utils/lora/config.py", lora_source)
    _write(root / "ultralytics/nn/peft/molora/config.py", molora_source)


def test_duplicate_yaml_keys_are_reported(tmp_path):
    config = _write(tmp_path / "duplicate.yaml", "alpha: 1\nalpha: 2\n")

    issues = ConfigDriftDetector(tmp_path).check_yaml_duplicates([config])

    assert [issue.code for issue in issues] == ["YAML_DUPLICATE_KEY"]
    assert issues[0].line == 2


def test_unmapped_dataclass_fields_are_reported(tmp_path):
    _write_mapping_fixture(
        tmp_path,
        """
from dataclasses import dataclass

@dataclass
class LoRAConfig:
    r: int = 0
    alpha: int = 16

    @classmethod
    def from_args(cls, args=None):
        mapping = {"r": "lora_r"}
        return cls()
""",
        """
from dataclasses import dataclass

@dataclass
class MoLoRAConfig:
    num_experts: int = 4

    @classmethod
    def from_args(cls, args=None):
        molora_mapping = {"num_experts": "molora_num_experts"}
        return cls()
""",
        "lora_r: 0\nmolora_num_experts: 4\n",
    )

    issues = ConfigDriftDetector(tmp_path).check_config_mappings()

    assert any(issue.code == "CONFIG_FIELD_UNMAPPED" and "alpha" in issue.message for issue in issues)


def test_mapped_cli_fields_must_exist_in_default_yaml(tmp_path):
    _write_mapping_fixture(
        tmp_path,
        """
from dataclasses import dataclass

@dataclass
class LoRAConfig:
    r: int = 0

    @classmethod
    def from_args(cls, args=None):
        mapping = {"r": "lora_r"}
        return cls()
""",
        """
from dataclasses import dataclass

@dataclass
class MoLoRAConfig:
    num_experts: int = 4

    @classmethod
    def from_args(cls, args=None):
        molora_mapping = {"num_experts": "molora_num_experts"}
        return cls()
""",
        "lora_r: 0\n",
    )

    issues = ConfigDriftDetector(tmp_path).check_config_mappings()

    assert any(
        issue.code == "CONFIG_MAPPING_DEFAULT_MISSING" and "molora_num_experts" in issue.message
        for issue in issues
    )


def test_mixture_resolver_defaults_must_match_public_defaults(tmp_path):
    _write(tmp_path / "ultralytics/cfg/default.yaml", "moe_router_z_loss: 0.1\n")
    _write(
        tmp_path / "ultralytics/nn/modules/moe/config.py",
        """
MIXTURE_DEFAULTS = {"moe": {"router_z_loss_coeff": 1.0}}
CLI_FIELDS = {"moe": {"router_z_loss_coeff": "moe_router_z_loss"}}
""",
    )

    issues = ConfigDriftDetector(tmp_path).check_mixture_runtime_contract()

    assert any(issue.code == "MIXTURE_DEFAULT_MISMATCH" for issue in issues)


def test_cli_type_registry_keys_must_exist_in_default_yaml(tmp_path):
    _write(tmp_path / "ultralytics/cfg/default.yaml", "epochs: 100\n")
    _write(
        tmp_path / "ultralytics/cfg/__init__.py",
        """
CFG_FLOAT_KEYS = frozenset({"missing_float"})
CFG_FRACTION_KEYS = frozenset()
CFG_INT_KEYS = frozenset({"epochs"})
CFG_BOOL_KEYS = frozenset()
""",
    )

    issues = ConfigDriftDetector(tmp_path).check_cli_type_registry()

    assert any(issue.code == "CLI_TYPE_KEY_MISSING" and "missing_float" in issue.message for issue in issues)


def _write_model_fixture(root: Path, module_name: str, args: list) -> None:
    _write(
        root / "ultralytics/nn/tasks.py",
        """
class Known:
    def __init__(self, c1, c2, enabled=False):
        pass

def parse_model():
    base_modules = frozenset({Known})
    repeat_modules = frozenset()
""",
    )
    _write(
        root / "ultralytics/cfg/models/master/test.yaml",
        f"backbone:\n  - [-1, 1, {module_name}, {args!r}]\nhead: []\n",
    )


def test_unknown_model_modules_are_reported(tmp_path):
    _write_model_fixture(tmp_path, "Missing", [16])

    issues = ConfigDriftDetector(tmp_path).check_master_model_configs()

    assert any(issue.code == "MODEL_MODULE_UNKNOWN" and "Missing" in issue.message for issue in issues)


def test_incompatible_model_argument_counts_are_reported(tmp_path):
    _write_model_fixture(tmp_path, "Known", [16, 1, 2])

    issues = ConfigDriftDetector(tmp_path).check_master_model_configs()

    assert any(issue.code == "MODEL_ARGS_INCOMPATIBLE" and "Known" in issue.message for issue in issues)


def test_repository_has_no_configuration_drift():
    issues = ConfigDriftDetector(ROOT).check_all()

    assert not issues, "\n".join(issue.format(ROOT) for issue in issues)
