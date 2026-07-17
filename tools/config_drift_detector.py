"""Detect structural drift across public config, runtime mappings, and model YAML files."""

from __future__ import annotations

import argparse
import ast
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable

import yaml


@dataclass(frozen=True)
class DriftIssue:
    """A stable, machine-readable configuration drift diagnostic."""

    code: str
    path: Path
    message: str
    line: int | None = None

    def format(self, root: Path | None = None) -> str:
        """Format the issue for terminal and CI logs."""
        path = self.path
        if root is not None:
            try:
                path = path.relative_to(root)
            except ValueError:
                pass
        location = f"{path}:{self.line}" if self.line else str(path)
        return f"[{self.code}] {location}: {self.message}"

    def to_dict(self, root: Path | None = None) -> dict[str, Any]:
        """Serialize the issue with a repository-relative path when possible."""
        data = asdict(self)
        path = self.path
        if root is not None:
            try:
                path = path.relative_to(root)
            except ValueError:
                pass
        data["path"] = str(path)
        return data


class _DuplicateKeyError(yaml.YAMLError):
    """Raised when a YAML mapping contains the same key more than once."""

    def __init__(self, key: Any, line: int):
        super().__init__(f"duplicate key {key!r}")
        self.key = key
        self.line = line


class _UniqueKeyLoader(yaml.SafeLoader):
    """Safe YAML loader that rejects duplicate mapping keys."""


def _construct_unique_mapping(loader: _UniqueKeyLoader, node: yaml.MappingNode, deep: bool = False) -> dict:
    mapping = {}
    for key_node, value_node in node.value:
        key = loader.construct_object(key_node, deep=deep)
        try:
            duplicate = key in mapping
        except TypeError as exc:
            raise yaml.constructor.ConstructorError(
                "while constructing a mapping",
                node.start_mark,
                "found an unhashable key",
                key_node.start_mark,
            ) from exc
        if duplicate:
            raise _DuplicateKeyError(key, key_node.start_mark.line + 1)
        mapping[key] = loader.construct_object(value_node, deep=deep)
    return mapping


_UniqueKeyLoader.add_constructor(yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG, _construct_unique_mapping)


@dataclass(frozen=True)
class _Signature:
    min_positional: int
    max_positional: int | None

    def accepts(self, count: int) -> bool:
        return count >= self.min_positional and (self.max_positional is None or count <= self.max_positional)


@dataclass
class _ClassInfo:
    signatures: list[_Signature]
    bases: set[str]


class ConfigDriftDetector:
    """Run fast, static configuration consistency checks for YOLO-Master."""

    DEFAULT_CFG = Path("ultralytics/cfg/default.yaml")
    LORA_CONFIG = Path("ultralytics/utils/lora/config.py")
    MOLORA_CONFIG = Path("ultralytics/nn/peft/molora/config.py")
    MIXTURE_CONFIG = Path("ultralytics/nn/modules/moe/config.py")
    CFG_INIT = Path("ultralytics/cfg/__init__.py")
    TASKS = Path("ultralytics/nn/tasks.py")
    MASTER_MODELS = Path("ultralytics/cfg/models/master")
    TYPE_REGISTRIES = ("CFG_FLOAT_KEYS", "CFG_FRACTION_KEYS", "CFG_INT_KEYS", "CFG_BOOL_KEYS")
    TORCH_MODULE_SIGNATURES = {"nn.Upsample": _Signature(0, 5)}

    def __init__(self, root: str | Path):
        self.root = Path(root).resolve()

    def check_all(self) -> list[DriftIssue]:
        """Run every drift check and return sorted diagnostics."""
        master_paths = sorted((self.root / self.MASTER_MODELS).rglob("*.yaml"))
        yaml_paths = [self.root / self.DEFAULT_CFG, *master_paths]
        issues = []
        issues.extend(self.check_yaml_duplicates(yaml_paths))
        issues.extend(self.check_config_mappings())
        issues.extend(self.check_mixture_runtime_contract())
        issues.extend(self.check_cli_type_registry())
        issues.extend(self.check_master_model_configs())
        return sorted(issues, key=lambda issue: (str(issue.path), issue.line or 0, issue.code, issue.message))

    def check_yaml_duplicates(self, paths: Iterable[Path]) -> list[DriftIssue]:
        """Reject duplicate keys and invalid YAML in public configuration files."""
        issues = []
        for raw_path in paths:
            path = self._absolute(raw_path)
            if not path.is_file():
                issues.append(DriftIssue("CONFIG_FILE_MISSING", path, "configuration file does not exist"))
                continue
            try:
                self._load_yaml(path)
            except _DuplicateKeyError as exc:
                issues.append(
                    DriftIssue("YAML_DUPLICATE_KEY", path, f"duplicate mapping key {exc.key!r}", line=exc.line)
                )
            except yaml.YAMLError as exc:
                mark = getattr(exc, "problem_mark", None)
                line = mark.line + 1 if mark is not None else None
                issues.append(DriftIssue("YAML_PARSE_ERROR", path, str(exc).splitlines()[0], line=line))
        return issues

    def check_config_mappings(self) -> list[DriftIssue]:
        """Compare LoRA/MoLoRA dataclass fields, from_args mappings, and public defaults."""
        default_path = self.root / self.DEFAULT_CFG
        defaults, issues = self._load_yaml_for_check(default_path)
        if defaults is None:
            return issues

        lora_path = self.root / self.LORA_CONFIG
        molora_path = self.root / self.MOLORA_CONFIG
        lora_tree, tree_issues = self._parse_python(lora_path)
        issues.extend(tree_issues)
        molora_tree, tree_issues = self._parse_python(molora_path)
        issues.extend(tree_issues)
        if lora_tree is None or molora_tree is None:
            return issues

        lora_fields, lora_mapping, lora_line = self._class_mapping(lora_tree, "LoRAConfig", "mapping")
        molora_fields, molora_mapping, molora_line = self._class_mapping(
            molora_tree, "MoLoRAConfig", "molora_mapping"
        )
        for path, class_name, mapping_name, fields, mapping, line in (
            (lora_path, "LoRAConfig", "mapping", lora_fields, lora_mapping, lora_line),
            (molora_path, "MoLoRAConfig", "molora_mapping", molora_fields, molora_mapping, molora_line),
        ):
            if not fields:
                issues.append(
                    DriftIssue(
                        "CONFIG_CLASS_UNREADABLE",
                        path,
                        f"{class_name} must declare annotated dataclass fields",
                        line=line,
                    )
                )
            if not mapping:
                issues.append(
                    DriftIssue(
                        "CONFIG_MAPPING_UNREADABLE",
                        path,
                        f"{class_name}.from_args must declare literal {mapping_name}",
                        line=line,
                    )
                )
        issues.extend(
            self._validate_mapping(
                lora_path,
                "LoRAConfig",
                lora_fields,
                lora_mapping,
                defaults,
                allowed_fields=lora_fields,
                line=lora_line,
            )
        )
        issues.extend(
            self._validate_mapping(
                molora_path,
                "MoLoRAConfig",
                molora_fields,
                molora_mapping,
                defaults,
                allowed_fields=molora_fields | lora_fields,
                line=molora_line,
            )
        )
        return issues

    def check_mixture_runtime_contract(self) -> list[DriftIssue]:
        """Ensure runtime resolver fields and defaults match the public CLI surface."""
        default_path = self.root / self.DEFAULT_CFG
        defaults, issues = self._load_yaml_for_check(default_path)
        if defaults is None:
            return issues

        path = self.root / self.MIXTURE_CONFIG
        tree, tree_issues = self._parse_python(path)
        issues.extend(tree_issues)
        if tree is None:
            return issues

        mixture_defaults = self._assigned_literal(tree, "MIXTURE_DEFAULTS")
        cli_fields = self._assigned_literal(tree, "CLI_FIELDS")
        if not isinstance(mixture_defaults, dict) or not isinstance(cli_fields, dict):
            issues.append(
                DriftIssue(
                    "MIXTURE_CONTRACT_UNREADABLE",
                    path,
                    "MIXTURE_DEFAULTS and CLI_FIELDS must be literal dictionaries",
                )
            )
            return issues

        for kind in sorted(set(mixture_defaults) | set(cli_fields)):
            runtime_fields = mixture_defaults.get(kind)
            mappings = cli_fields.get(kind)
            if not isinstance(runtime_fields, dict) or not isinstance(mappings, dict):
                issues.append(
                    DriftIssue("MIXTURE_KIND_INVALID", path, f"mixture kind {kind!r} must map to dictionaries")
                )
                continue
            for field in sorted(set(runtime_fields) - set(mappings)):
                issues.append(
                    DriftIssue("MIXTURE_FIELD_UNMAPPED", path, f"{kind}.{field} has no CLI_FIELDS mapping")
                )
            for field in sorted(set(mappings) - set(runtime_fields)):
                issues.append(
                    DriftIssue("MIXTURE_MAPPING_UNKNOWN_FIELD", path, f"{kind}.{field} has no runtime default")
                )
            for field in sorted(set(runtime_fields) & set(mappings)):
                cli_key = mappings[field]
                if cli_key not in defaults:
                    issues.append(
                        DriftIssue(
                            "MIXTURE_CLI_DEFAULT_MISSING",
                            path,
                            f"{kind}.{field} maps to missing default key {cli_key!r}",
                        )
                    )
                    continue
                runtime_value = runtime_fields[field]
                public_value = defaults[cli_key]
                if not self._same_value(runtime_value, public_value):
                    issues.append(
                        DriftIssue(
                            "MIXTURE_DEFAULT_MISMATCH",
                            path,
                            f"{kind}.{field}={runtime_value!r} differs from {cli_key}={public_value!r}",
                        )
                    )
        return issues

    def check_cli_type_registry(self) -> list[DriftIssue]:
        """Validate that CLI type registries reference public keys with compatible defaults."""
        default_path = self.root / self.DEFAULT_CFG
        defaults, issues = self._load_yaml_for_check(default_path)
        if defaults is None:
            return issues

        path = self.root / self.CFG_INIT
        tree, tree_issues = self._parse_python(path)
        issues.extend(tree_issues)
        if tree is None:
            return issues

        registries: dict[str, set[str]] = {}
        for name in self.TYPE_REGISTRIES:
            value = self._assigned_literal(tree, name)
            if not isinstance(value, (set, frozenset, list, tuple)):
                issues.append(DriftIssue("CLI_TYPE_REGISTRY_UNREADABLE", path, f"{name} must be a literal collection"))
                continue
            registries[name] = set(value)

        owners: dict[str, list[str]] = {}
        for registry, keys in registries.items():
            for key in keys:
                owners.setdefault(key, []).append(registry)
                if key not in defaults:
                    issues.append(
                        DriftIssue("CLI_TYPE_KEY_MISSING", path, f"{registry} references missing default key {key!r}")
                    )
                    continue
                value = defaults[key]
                if value is not None and not self._value_matches_registry(value, registry):
                    issues.append(
                        DriftIssue(
                            "CLI_TYPE_DEFAULT_MISMATCH",
                            path,
                            f"{key}={value!r} is incompatible with {registry}",
                        )
                    )
        for key, key_owners in sorted(owners.items()):
            if len(key_owners) > 1:
                issues.append(
                    DriftIssue(
                        "CLI_TYPE_KEY_OVERLAP",
                        path,
                        f"{key!r} appears in multiple type registries: {', '.join(sorted(key_owners))}",
                    )
                )
        return issues

    def check_master_model_configs(self) -> list[DriftIssue]:
        """Validate Master model layer records, module names, and constructor arity."""
        tasks_path = self.root / self.TASKS
        tasks_tree, issues = self._parse_python(tasks_path)
        if tasks_tree is None:
            return issues

        class_index, resolvable_names = self._build_class_index(tasks_tree)
        base_modules, repeat_modules, channel_append_modules = self._parse_model_groups(tasks_tree)
        known_names = resolvable_names | set(self.TORCH_MODULE_SIGNATURES)
        model_paths = sorted((self.root / self.MASTER_MODELS).rglob("*.yaml"))
        for path in model_paths:
            data, yaml_issues = self._load_yaml_for_check(path)
            issues.extend(yaml_issues)
            if data is None:
                continue
            if not isinstance(data, dict):
                issues.append(DriftIssue("MODEL_YAML_INVALID", path, "model YAML root must be a mapping"))
                continue
            for section in ("backbone", "head"):
                layers = data.get(section)
                if not isinstance(layers, list):
                    issues.append(DriftIssue("MODEL_SECTION_INVALID", path, f"{section} must be a list"))
                    continue
                for index, layer in enumerate(layers):
                    location = f"{section}[{index}]"
                    if not isinstance(layer, list) or len(layer) != 4:
                        issues.append(
                            DriftIssue(
                                "MODEL_LAYER_INVALID",
                                path,
                                f"{location} must be [from, repeats, module, args]",
                            )
                        )
                        continue
                    _, repeats, module_name, args = layer
                    if not isinstance(repeats, int) or isinstance(repeats, bool) or repeats < 1:
                        issues.append(
                            DriftIssue("MODEL_REPEAT_INVALID", path, f"{location} repeats must be a positive integer")
                        )
                    if not isinstance(module_name, str):
                        issues.append(DriftIssue("MODEL_MODULE_INVALID", path, f"{location} module must be a string"))
                        continue
                    if not isinstance(args, list):
                        issues.append(DriftIssue("MODEL_ARGS_INVALID", path, f"{location} args must be a list"))
                        continue
                    if module_name not in known_names:
                        issues.append(
                            DriftIssue(
                                "MODEL_MODULE_UNKNOWN",
                                path,
                                f"{location} references unknown module {module_name!r}",
                            )
                        )
                        continue
                    signatures = self._resolve_signatures(module_name, class_index)
                    torch_signature = self.TORCH_MODULE_SIGNATURES.get(module_name)
                    if torch_signature is not None:
                        signatures = [torch_signature]
                    if not signatures:
                        continue
                    effective_count = len(args)
                    if module_name in base_modules:
                        effective_count += 1  # parse_model prepends input channels.
                        if module_name in repeat_modules:
                            effective_count += 1  # parse_model inserts the scaled repeat count.
                    elif module_name in channel_append_modules:
                        effective_count += 1  # detection-style heads receive the input-channel list.
                    if not any(signature.accepts(effective_count) for signature in signatures):
                        expected = ", ".join(self._format_signature(signature) for signature in signatures)
                        issues.append(
                            DriftIssue(
                                "MODEL_ARGS_INCOMPATIBLE",
                                path,
                                f"{location} {module_name} receives {effective_count} positional args after parse_model; expected {expected}",
                            )
                        )
        return issues

    def _absolute(self, path: Path) -> Path:
        return path if path.is_absolute() else self.root / path

    @staticmethod
    def _load_yaml(path: Path) -> Any:
        with path.open(encoding="utf-8") as stream:
            return yaml.load(stream, Loader=_UniqueKeyLoader)

    def _load_yaml_for_check(self, path: Path) -> tuple[Any | None, list[DriftIssue]]:
        duplicate_issues = self.check_yaml_duplicates([path])
        if duplicate_issues:
            return None, duplicate_issues
        try:
            return self._load_yaml(path), []
        except OSError as exc:
            return None, [DriftIssue("CONFIG_READ_ERROR", path, str(exc))]

    @staticmethod
    def _parse_python(path: Path) -> tuple[ast.Module | None, list[DriftIssue]]:
        if not path.is_file():
            return None, [DriftIssue("PYTHON_FILE_MISSING", path, "Python source file does not exist")]
        try:
            return ast.parse(path.read_text(encoding="utf-8"), filename=str(path)), []
        except (OSError, SyntaxError) as exc:
            return None, [
                DriftIssue(
                    "PYTHON_PARSE_ERROR",
                    path,
                    getattr(exc, "msg", str(exc)),
                    line=getattr(exc, "lineno", None),
                )
            ]

    @staticmethod
    def _class_mapping(tree: ast.Module, class_name: str, mapping_name: str) -> tuple[set[str], dict, int | None]:
        class_node = next(
            (node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == class_name), None
        )
        if class_node is None:
            return set(), {}, None
        fields = {
            node.target.id
            for node in class_node.body
            if isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name)
        }
        for node in class_node.body:
            if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) or node.name != "from_args":
                continue
            for child in ast.walk(node):
                if isinstance(child, (ast.Assign, ast.AnnAssign)):
                    targets = child.targets if isinstance(child, ast.Assign) else [child.target]
                    if any(isinstance(target, ast.Name) and target.id == mapping_name for target in targets):
                        value = ConfigDriftDetector._literal_value(child.value)
                        return fields, value if isinstance(value, dict) else {}, child.lineno
        return fields, {}, class_node.lineno

    @staticmethod
    def _validate_mapping(
        path: Path,
        class_name: str,
        own_fields: set[str],
        mapping: dict,
        defaults: dict,
        allowed_fields: set[str],
        line: int | None,
    ) -> list[DriftIssue]:
        issues = []
        for field in sorted(own_fields - set(mapping)):
            issues.append(
                DriftIssue("CONFIG_FIELD_UNMAPPED", path, f"{class_name}.{field} has no from_args mapping", line=line)
            )
        for field in sorted(set(mapping) - allowed_fields):
            issues.append(
                DriftIssue(
                    "CONFIG_MAPPING_UNKNOWN_FIELD",
                    path,
                    f"{class_name} mapping references unknown field {field!r}",
                    line=line,
                )
            )
        targets: dict[str, list[str]] = {}
        for field, cli_key in mapping.items():
            if not isinstance(cli_key, str):
                issues.append(
                    DriftIssue(
                        "CONFIG_MAPPING_INVALID_TARGET",
                        path,
                        f"{class_name}.{field} maps to non-string target {cli_key!r}",
                        line=line,
                    )
                )
                continue
            targets.setdefault(cli_key, []).append(field)
            if cli_key not in defaults:
                issues.append(
                    DriftIssue(
                        "CONFIG_MAPPING_DEFAULT_MISSING",
                        path,
                        f"{class_name}.{field} maps to missing default key {cli_key!r}",
                        line=line,
                    )
                )
        for cli_key, fields in sorted(targets.items()):
            if len(fields) > 1:
                issues.append(
                    DriftIssue(
                        "CONFIG_MAPPING_DUPLICATE_TARGET",
                        path,
                        f"{class_name} fields {', '.join(sorted(fields))} all map to {cli_key!r}",
                        line=line,
                    )
                )
        return issues

    @staticmethod
    def _assigned_literal(tree: ast.Module, name: str) -> Any:
        for node in tree.body:
            if isinstance(node, ast.Assign):
                if any(isinstance(target, ast.Name) and target.id == name for target in node.targets):
                    return ConfigDriftDetector._literal_value(node.value)
            elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name) and node.target.id == name:
                return ConfigDriftDetector._literal_value(node.value)
        return None

    @staticmethod
    def _literal_value(node: ast.AST | None) -> Any:
        if node is None:
            return None
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id in {
            "dict",
            "frozenset",
            "list",
            "set",
            "tuple",
        }:
            if node.keywords:
                return None
            values = ast.literal_eval(node.args[0]) if node.args else ()
            return {"dict": dict, "frozenset": frozenset, "list": list, "set": set, "tuple": tuple}[
                node.func.id
            ](values)
        try:
            return ast.literal_eval(node)
        except (ValueError, TypeError):
            return None

    def _build_class_index(self, tasks_tree: ast.Module) -> tuple[dict[str, list[_ClassInfo]], set[str]]:
        index: dict[str, list[_ClassInfo]] = {}
        source_paths = [self.root / self.TASKS]
        modules_root = self.root / "ultralytics/nn/modules"
        if modules_root.is_dir():
            source_paths.extend(sorted(modules_root.rglob("*.py")))
        for path in source_paths:
            tree = tasks_tree if path == self.root / self.TASKS else self._parse_python(path)[0]
            if tree is None:
                continue
            for node in tree.body:
                if not isinstance(node, ast.ClassDef):
                    continue
                signatures = []
                init = next(
                    (child for child in node.body if isinstance(child, ast.FunctionDef) and child.name == "__init__"),
                    None,
                )
                if init is not None:
                    signatures.append(self._signature_from_function(init))
                bases = {self._node_name(base) for base in node.bases}
                index.setdefault(node.name, []).append(_ClassInfo(signatures, {base for base in bases if base}))
        imported_names = {
            alias.asname or alias.name
            for node in tasks_tree.body
            if isinstance(node, (ast.Import, ast.ImportFrom))
            for alias in node.names
        }
        locally_defined = {node.name for node in tasks_tree.body if isinstance(node, ast.ClassDef)}
        return index, imported_names | locally_defined

    @staticmethod
    def _signature_from_function(node: ast.FunctionDef) -> _Signature:
        positional = [*node.args.posonlyargs, *node.args.args]
        if positional and positional[0].arg in {"self", "cls"}:
            positional = positional[1:]
        minimum = len(positional) - len(node.args.defaults)
        maximum = None if node.args.vararg is not None else len(positional)
        return _Signature(minimum, maximum)

    @staticmethod
    def _parse_model_groups(tree: ast.Module) -> tuple[set[str], set[str], set[str]]:
        parse_model = next(
            (node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "parse_model"), None
        )
        if parse_model is None:
            return set(), set(), set()
        base_modules: set[str] = set()
        repeat_modules: set[str] = set()
        channel_append_modules: set[str] = set()
        for node in ast.walk(parse_model):
            if isinstance(node, (ast.Assign, ast.AnnAssign)):
                targets = node.targets if isinstance(node, ast.Assign) else [node.target]
                value = node.value
                for target in targets:
                    if isinstance(target, ast.Name) and target.id in {"base_modules", "repeat_modules"}:
                        names = ConfigDriftDetector._names_from_collection(value)
                        if target.id == "base_modules":
                            base_modules.update(names)
                        else:
                            repeat_modules.update(names)
            if isinstance(node, ast.If) and any(
                isinstance(child, ast.Call)
                and isinstance(child.func, ast.Attribute)
                and isinstance(child.func.value, ast.Name)
                and child.func.value.id == "args"
                and child.func.attr == "append"
                for statement in node.body
                for child in ast.walk(statement)
            ):
                channel_append_modules.update(ConfigDriftDetector._names_from_membership_test(node.test))
        return base_modules, repeat_modules, channel_append_modules

    @staticmethod
    def _names_from_collection(node: ast.AST) -> set[str]:
        if isinstance(node, ast.Call) and node.args:
            node = node.args[0]
        if not isinstance(node, (ast.Set, ast.List, ast.Tuple)):
            return set()
        return {name for item in node.elts if (name := ConfigDriftDetector._node_name(item))}

    @staticmethod
    def _names_from_membership_test(node: ast.AST) -> set[str]:
        for child in ast.walk(node):
            if not isinstance(child, ast.Compare) or len(child.ops) != 1 or not isinstance(child.ops[0], ast.In):
                continue
            if not isinstance(child.left, ast.Name) or child.left.id != "m":
                continue
            return ConfigDriftDetector._names_from_collection(child.comparators[0])
        return set()

    @staticmethod
    def _node_name(node: ast.AST) -> str:
        if isinstance(node, ast.Name):
            return node.id
        if isinstance(node, ast.Attribute):
            base = ConfigDriftDetector._node_name(node.value)
            return f"{base}.{node.attr}" if base else node.attr
        return ""

    def _resolve_signatures(
        self, name: str, index: dict[str, list[_ClassInfo]], seen: set[str] | None = None
    ) -> list[_Signature]:
        seen = set() if seen is None else set(seen)
        if name in seen:
            return []
        seen.add(name)
        signatures = []
        for info in index.get(name, []):
            signatures.extend(info.signatures)
            if not info.signatures:
                for base in info.bases:
                    signatures.extend(self._resolve_signatures(base.rsplit(".", 1)[-1], index, seen))
        return list(dict.fromkeys(signatures))

    @staticmethod
    def _same_value(left: Any, right: Any) -> bool:
        return type(left) is type(right) and left == right

    @staticmethod
    def _value_matches_registry(value: Any, registry: str) -> bool:
        if registry == "CFG_BOOL_KEYS":
            return isinstance(value, bool)
        if registry == "CFG_INT_KEYS":
            return isinstance(value, int) and not isinstance(value, bool)
        if registry in {"CFG_FLOAT_KEYS", "CFG_FRACTION_KEYS"}:
            valid = isinstance(value, (int, float)) and not isinstance(value, bool)
            return valid and (registry != "CFG_FRACTION_KEYS" or 0.0 <= value <= 1.0)
        return True

    @staticmethod
    def _format_signature(signature: _Signature) -> str:
        maximum = "unbounded" if signature.max_positional is None else str(signature.max_positional)
        return f"{signature.min_positional}..{maximum}"


def main(argv: list[str] | None = None) -> int:
    """Run the drift detector as a command-line CI gate."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--json", action="store_true", help="Emit machine-readable JSON diagnostics")
    args = parser.parse_args(argv)

    detector = ConfigDriftDetector(args.root)
    issues = detector.check_all()
    if args.json:
        print(json.dumps([issue.to_dict(detector.root) for issue in issues], indent=2, sort_keys=True))
    elif issues:
        print(f"Configuration drift check: FAIL ({len(issues)} issue(s))")
        for issue in issues:
            print(issue.format(detector.root))
    else:
        count = len(list((detector.root / detector.MASTER_MODELS).rglob("*.yaml")))
        print(f"Configuration drift check: PASS ({count} Master model configs)")
    return 1 if issues else 0


if __name__ == "__main__":
    raise SystemExit(main())
