"""Typed, validated loading of the YAML files in ``configs/``.

Each loader returns frozen dataclasses. Unknown keys, missing required keys, wrong types
and unsafe combinations raise :class:`ConfigError` with the offending key path.
"""

from __future__ import annotations

import re
from dataclasses import MISSING, asdict, dataclass, field, fields
from pathlib import Path
from typing import Any

import yaml

from decision_model.utils.paths import find_repo_root, resolve

CONFIG_DIR = "configs"
DATA_CONFIG = "data.yaml"
TRAINING_CONFIG = "training.yaml"
EVALUATION_CONFIG = "evaluation.yaml"
PUBLISHING_CONFIG = "publishing.yaml"

_HF_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*/[A-Za-z0-9][A-Za-z0-9._-]*$")
_REVISION = re.compile(r"^[0-9a-f]{40}$")
_PROFILE_KINDS = {"full", "smoke"}
SPLITS = ("train", "validation", "test")


class ConfigError(ValueError):
    """Raised for invalid or unsafe configuration."""


# --------------------------------------------------------------------------- helpers


def read_yaml(path: Path) -> dict[str, Any]:
    """Read a YAML mapping, with a clear error for missing files and syntax errors."""
    if not path.is_file():
        raise ConfigError(f"Config file not found: {path}")
    try:
        data = yaml.safe_load(path.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        raise ConfigError(f"Invalid YAML in {path}: {exc}") from exc
    if not isinstance(data, dict):
        raise ConfigError(f"{path} must contain a mapping at the top level")
    return data


def _section(data: dict[str, Any], key: str, where: str) -> dict[str, Any]:
    if key not in data:
        raise ConfigError(f"{where}: missing required section '{key}'")
    value = data[key]
    if not isinstance(value, dict):
        raise ConfigError(f"{where}.{key} must be a mapping")
    return value


def _build(cls: type, data: dict[str, Any], where: str) -> Any:
    """Instantiate a flat dataclass from ``data`` with type and key checks."""
    known = {f.name: f for f in fields(cls)}
    unknown = sorted(set(data) - set(known))
    if unknown:
        raise ConfigError(f"{where}: unknown key(s) {unknown}")
    kwargs: dict[str, Any] = {}
    for name, f in known.items():
        if name not in data:
            if f.default is MISSING and f.default_factory is MISSING:
                raise ConfigError(f"{where}: missing required key '{name}'")
            continue
        kwargs[name] = _check_type(data[name], f.type, f"{where}.{name}")
    return cls(**kwargs)


def _check_type(value: Any, annotation: Any, where: str) -> Any:
    """Check ``value`` against a simple annotation (a type or its string form)."""
    ann = annotation.__name__ if isinstance(annotation, type) else str(annotation)
    ann = ann.replace(" ", "")
    optional = ann.endswith("|None")
    base = ann.removesuffix("|None")
    if value is None:
        if optional:
            return None
        raise ConfigError(f"{where} must not be null")
    if base == "int":
        if isinstance(value, bool) or not isinstance(value, int):
            raise ConfigError(f"{where} must be an integer, got {value!r}")
    elif base == "float":
        if isinstance(value, bool) or not isinstance(value, int | float):
            raise ConfigError(f"{where} must be a number, got {value!r}")
        value = float(value)
    elif base == "bool":
        if not isinstance(value, bool):
            raise ConfigError(f"{where} must be true or false, got {value!r}")
    elif base == "str":
        if not isinstance(value, str) or not value.strip():
            raise ConfigError(f"{where} must be a non-empty string, got {value!r}")
    elif base == "list[str]" and (
        not isinstance(value, list) or not all(isinstance(v, str) and v for v in value)
    ):
        raise ConfigError(f"{where} must be a list of non-empty strings")
    return value


def _positive(value: int | float, where: str, allow_zero: bool = False) -> None:
    if value < 0 or (value == 0 and not allow_zero):
        raise ConfigError(f"{where} must be {'>= 0' if allow_zero else '> 0'}, got {value}")


def validate_hf_id(repo_id: str, where: str) -> None:
    """Validate a Hugging Face ``namespace/name`` identifier."""
    if not _HF_ID.match(repo_id) or ".." in repo_id or len(repo_id) > 96:
        raise ConfigError(f"{where}: '{repo_id}' is not a valid Hugging Face repo id")


def _validate_revision(revision: str | None, where: str) -> None:
    if revision is not None and not _REVISION.match(revision):
        raise ConfigError(f"{where}: revision must be a 40-character commit hash, got {revision!r}")


def _mapping_of_str(value: Any, where: str) -> dict[str, str]:
    if not isinstance(value, dict) or set(value) != set(SPLITS):
        raise ConfigError(f"{where} must map exactly {list(SPLITS)}")
    return {k: _check_type(v, str, f"{where}.{k}") for k, v in value.items()}


def _mapping_of_int(value: Any, where: str) -> dict[str, int]:
    if not isinstance(value, dict) or set(value) != set(SPLITS):
        raise ConfigError(f"{where} must map exactly {list(SPLITS)}")
    out = {k: _check_type(v, int, f"{where}.{k}") for k, v in value.items()}
    for k, v in out.items():
        _positive(v, f"{where}.{k}")
    return out


# --------------------------------------------------------------------------- data


@dataclass(frozen=True)
class DatasetSource:
    repo_id: str
    revision: str | None
    license: str | None
    files: dict[str, str]
    expected_rows: dict[str, int]


@dataclass(frozen=True)
class RequestStreamConfig:
    seed: int
    num_requests: int


@dataclass(frozen=True)
class SmokeDataConfig:
    dir: str
    train_size: int
    test_size: int


@dataclass(frozen=True)
class DataConfig:
    dataset: DatasetSource
    requests: RequestStreamConfig
    output_dir: Path
    smoke: SmokeDataConfig
    smoke_dir: Path


def parse_data_config(raw: dict[str, Any], root: Path) -> DataConfig:
    """Validate the contents of ``data.yaml``."""
    w = DATA_CONFIG
    if unknown := sorted(set(raw) - {"dataset", "training_requests", "output", "smoke"}):
        raise ConfigError(f"{w}: unknown section(s) {unknown}")
    ds = dict(_section(raw, "dataset", w))
    if unknown := sorted(set(ds) - {"repo_id", "revision", "license", "files", "expected_rows"}):
        raise ConfigError(f"{w}.dataset: unknown key(s) {unknown}")
    source = DatasetSource(
        repo_id=_check_type(ds.get("repo_id"), str, f"{w}.dataset.repo_id"),
        revision=_check_type(ds.get("revision"), str | None, f"{w}.dataset.revision"),
        license=_check_type(ds.get("license"), str | None, f"{w}.dataset.license"),
        files=_mapping_of_str(ds.get("files"), f"{w}.dataset.files"),
        expected_rows=_mapping_of_int(ds.get("expected_rows"), f"{w}.dataset.expected_rows"),
    )
    validate_hf_id(source.repo_id, f"{w}.dataset.repo_id")
    _validate_revision(source.revision, f"{w}.dataset.revision")
    stream = _build(
        RequestStreamConfig, _section(raw, "training_requests", w), f"{w}.training_requests"
    )
    _positive(stream.num_requests, f"{w}.training_requests.num_requests")
    if stream.num_requests > source.expected_rows["train"]:
        raise ConfigError(f"{w}.training_requests.num_requests exceeds the training rows")
    output = _section(raw, "output", w)
    out_dir = _check_type(output.get("dir"), str, f"{w}.output.dir")
    smoke = _build(SmokeDataConfig, _section(raw, "smoke", w), f"{w}.smoke")
    _positive(smoke.train_size, f"{w}.smoke.train_size")
    _positive(smoke.test_size, f"{w}.smoke.test_size")
    if smoke.train_size > stream.num_requests or smoke.test_size > source.expected_rows["test"]:
        raise ConfigError(f"{w}.smoke sizes cannot exceed the canonical sizes")
    cfg = DataConfig(source, stream, resolve(out_dir, root), smoke, resolve(smoke.dir, root))
    if cfg.smoke_dir.resolve() == cfg.output_dir.resolve():
        raise ConfigError(f"{w}: smoke.dir must differ from output.dir")
    return cfg


# --------------------------------------------------------------------------- training


@dataclass(frozen=True)
class ModelConfig:
    repo_id: str
    revision: str | None
    license: str | None = None


@dataclass(frozen=True)
class HeadConfig:
    hidden_size: int
    width: int
    routing_layers: int
    layers: int
    heads: int
    feedforward: int

    def to_dict(self) -> dict[str, int]:
        return asdict(self)


@dataclass(frozen=True)
class LoraConfig:
    rank: int
    alpha: float
    dropout: float
    keys: list[str]

    @property
    def scale(self) -> float:
        """MLX-LM multiplies the low-rank update by ``scale`` = alpha / rank."""
        return self.alpha / self.rank


@dataclass(frozen=True)
class TrainingProfile:
    """One complete, explicit training recipe."""

    name: str
    kind: str
    requests_file: str
    output_dir: str
    steps: int
    batch_size: int
    grad_accumulation_steps: int
    lr_lora: float
    lr_head: float
    weight_decay: float
    warmup_steps: int
    max_grad_norm: float
    grad_checkpoint: bool
    log_every: int
    save_every: int
    seed: int

    @property
    def examples_per_update(self) -> int:
        return self.batch_size * self.grad_accumulation_steps

    @property
    def requests_needed(self) -> int:
        return self.steps * self.examples_per_update

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class SafetyConfig:
    smoke_max_steps: int = 5
    min_free_disk_gb: float = 4.0


@dataclass(frozen=True)
class TrainingConfig:
    model: ModelConfig
    head: HeadConfig
    lora: LoraConfig
    max_length: int
    profiles: dict[str, TrainingProfile]
    safety: SafetyConfig

    def profile(self, name: str) -> TrainingProfile:
        if name not in self.profiles:
            raise ConfigError(
                f"Unknown training profile '{name}'. Available: {', '.join(sorted(self.profiles))}"
            )
        return self.profiles[name]


def validate_profile(p: TrainingProfile, safety: SafetyConfig) -> None:
    """Range checks and smoke-profile guard rails for a training profile."""
    w = f"{TRAINING_CONFIG}.profiles.{p.name}"
    if p.kind not in _PROFILE_KINDS:
        raise ConfigError(f"{w}.kind must be one of {sorted(_PROFILE_KINDS)}")
    for name in (
        "steps",
        "batch_size",
        "grad_accumulation_steps",
        "lr_lora",
        "lr_head",
        "max_grad_norm",
        "log_every",
    ):
        _positive(getattr(p, name), f"{w}.{name}")
    for name in ("weight_decay", "warmup_steps", "save_every"):
        _positive(getattr(p, name), f"{w}.{name}", allow_zero=True)
    if max(p.lr_lora, p.lr_head) > 1e-2:
        raise ConfigError(f"{w}: learning rates above 1e-2 are implausible here")
    if p.warmup_steps > p.steps:
        raise ConfigError(f"{w}.warmup_steps cannot exceed steps")
    if p.kind == "smoke" and p.steps > safety.smoke_max_steps:
        raise ConfigError(
            f"{w}: smoke profile has steps={p.steps} > safety.smoke_max_steps="
            f"{safety.smoke_max_steps}; a smoke run must stay tiny"
        )


def parse_training_config(raw: dict[str, Any]) -> TrainingConfig:
    """Validate the contents of ``training.yaml``."""
    w = TRAINING_CONFIG
    allowed = {"model", "head", "max_length", "lora", "profiles", "safety"}
    if unknown := sorted(set(raw) - allowed):
        raise ConfigError(f"{w}: unknown section(s) {unknown}")
    model = _build(ModelConfig, _section(raw, "model", w), f"{w}.model")
    validate_hf_id(model.repo_id, f"{w}.model.repo_id")
    _validate_revision(model.revision, f"{w}.model.revision")
    head = _build(HeadConfig, _section(raw, "head", w), f"{w}.head")
    for f in fields(HeadConfig):
        _positive(getattr(head, f.name), f"{w}.head.{f.name}")
    if head.width % head.heads:
        raise ConfigError(f"{w}.head.width must be divisible by head.heads")
    lora = _build(LoraConfig, _section(raw, "lora", w), f"{w}.lora")
    _positive(lora.rank, f"{w}.lora.rank")
    _positive(lora.alpha, f"{w}.lora.alpha")
    if not 0.0 <= lora.dropout < 1.0:
        raise ConfigError(f"{w}.lora.dropout must be in [0, 1)")
    if not lora.keys:
        raise ConfigError(f"{w}.lora.keys must not be empty")
    max_length = _check_type(raw.get("max_length"), int, f"{w}.max_length")
    _positive(max_length, f"{w}.max_length")
    safety = _build(SafetyConfig, raw.get("safety") or {}, f"{w}.safety")
    profiles_raw = _section(raw, "profiles", w)
    profiles = {}
    for name, praw in profiles_raw.items():
        if not isinstance(praw, dict):
            raise ConfigError(f"{w}.profiles.{name} must be a mapping")
        profile = _build(TrainingProfile, {**praw, "name": name}, f"{w}.profiles.{name}")
        validate_profile(profile, safety)
        profiles[name] = profile
    if "reference" not in profiles:
        raise ConfigError(f"{w}.profiles must keep the 'reference' profile")
    outputs = [p.output_dir for p in profiles.values()]
    if len(set(outputs)) != len(outputs):
        raise ConfigError(f"{w}: every profile needs its own output_dir")
    return TrainingConfig(model, head, lora, max_length, profiles, safety)


# --------------------------------------------------------------------------- evaluation


@dataclass(frozen=True)
class EvaluationConfig:
    test_file: Path
    smoke_test_file: Path
    output_dir: Path
    batch_size: int
    chat_max_tokens: int
    enable_thinking: bool
    max_examples: int | None
    smoke_max_examples: int


def parse_evaluation_config(raw: dict[str, Any], root: Path) -> EvaluationConfig:
    """Validate the contents of ``evaluation.yaml``."""
    w = EVALUATION_CONFIG
    allowed = {"test_file", "smoke_test_file", "output_dir", "decision", "chat_baseline", "limits"}
    if unknown := sorted(set(raw) - allowed):
        raise ConfigError(f"{w}: unknown key(s) {unknown}")
    dec = _section(raw, "decision", w)
    chat = _section(raw, "chat_baseline", w)
    lim = _section(raw, "limits", w)
    cfg = EvaluationConfig(
        test_file=resolve(_check_type(raw.get("test_file"), str, f"{w}.test_file"), root),
        smoke_test_file=resolve(
            _check_type(raw.get("smoke_test_file"), str, f"{w}.smoke_test_file"), root
        ),
        output_dir=resolve(_check_type(raw.get("output_dir"), str, f"{w}.output_dir"), root),
        batch_size=_check_type(dec.get("batch_size"), int, f"{w}.decision.batch_size"),
        chat_max_tokens=_check_type(chat.get("max_tokens"), int, f"{w}.chat_baseline.max_tokens"),
        enable_thinking=_check_type(
            chat.get("enable_thinking"), bool, f"{w}.chat_baseline.enable_thinking"
        ),
        max_examples=_check_type(lim.get("max_examples"), int | None, f"{w}.limits.max_examples"),
        smoke_max_examples=_check_type(
            lim.get("smoke_max_examples"), int, f"{w}.limits.smoke_max_examples"
        ),
    )
    for name in ("batch_size", "chat_max_tokens", "smoke_max_examples"):
        _positive(getattr(cfg, name), f"{w}.{name}")
    if cfg.max_examples is not None:
        _positive(cfg.max_examples, f"{w}.limits.max_examples")
    if cfg.smoke_max_examples > 32:
        raise ConfigError(f"{w}.limits.smoke_max_examples must stay small (<= 32)")
    return cfg


# --------------------------------------------------------------------------- publishing


@dataclass(frozen=True)
class PublishingConfig:
    repo_id: str | None
    private: bool
    mode: str
    run_dir: Path
    model_card_path: Path
    staging_dir: Path
    commit_message: str
    adapter_license: str
    include: list[str] = field(default_factory=list)


def parse_publishing_config(raw: dict[str, Any], root: Path) -> PublishingConfig:
    """Validate the contents of ``publishing.yaml``."""
    w = PUBLISHING_CONFIG
    allowed = {f.name for f in fields(PublishingConfig)}
    if unknown := sorted(set(raw) - allowed):
        raise ConfigError(f"{w}: unknown key(s) {unknown}")
    repo_id = _check_type(raw.get("repo_id"), str | None, f"{w}.repo_id")
    if repo_id is not None:
        validate_hf_id(repo_id, f"{w}.repo_id")
    mode = _check_type(raw.get("mode", "adapter"), str, f"{w}.mode")
    if mode != "adapter":
        raise ConfigError(f"{w}.mode: only 'adapter' publication is supported")
    include = _check_type(raw.get("include"), list[str], f"{w}.include")
    for name in include:
        if "/" in name or name.startswith("."):
            raise ConfigError(f"{w}.include: '{name}' must be a plain file name")
    return PublishingConfig(
        repo_id=repo_id,
        private=_check_type(raw.get("private", True), bool, f"{w}.private"),
        mode=mode,
        run_dir=resolve(_check_type(raw.get("run_dir"), str, f"{w}.run_dir"), root),
        model_card_path=resolve(
            _check_type(raw.get("model_card_path"), str, f"{w}.model_card_path"), root
        ),
        staging_dir=resolve(_check_type(raw.get("staging_dir"), str, f"{w}.staging_dir"), root),
        commit_message=_check_type(raw.get("commit_message"), str, f"{w}.commit_message"),
        adapter_license=_check_type(raw.get("adapter_license"), str, f"{w}.adapter_license"),
        include=include,
    )


# --------------------------------------------------------------------------- bundle


@dataclass(frozen=True)
class ProjectConfig:
    root: Path
    data: DataConfig
    training: TrainingConfig
    evaluation: EvaluationConfig
    publishing: PublishingConfig


def load_project_config(root: Path | None = None, config_dir: Path | None = None) -> ProjectConfig:
    """Load and validate all four configuration files."""
    root = root or find_repo_root()
    cdir = config_dir or (root / CONFIG_DIR)
    data = parse_data_config(read_yaml(cdir / DATA_CONFIG), root)
    training = parse_training_config(read_yaml(cdir / TRAINING_CONFIG))
    evaluation = parse_evaluation_config(read_yaml(cdir / EVALUATION_CONFIG), root)
    publishing = parse_publishing_config(read_yaml(cdir / PUBLISHING_CONFIG), root)
    return ProjectConfig(root, data, training, evaluation, publishing)
