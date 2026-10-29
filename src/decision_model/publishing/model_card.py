"""Generate the Hugging Face model card (README.md) from configs and recorded outputs.

Everything in the card comes from configuration files, the data manifest, the training
run manifest and saved evaluation records. Values that were never measured are written
as "not measured" / "pending"; nothing is estimated or filled in.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from decision_model.config import ProjectConfig
from decision_model.evaluation.evaluator import load_latest
from decision_model.publishing.huggingface import RUNTIME_NAME
from decision_model.task import INSTRUCTIONS, ROUTES
from decision_model.training.artifacts import RUN_MANIFEST, read_json
from decision_model.utils.paths import display

NOT_MEASURED = "not measured"
PENDING = "pending: not evaluated yet"
REQUIRED_SECTIONS = (
    "## Model overview",
    "## Model details",
    "## Intended use",
    "## Training data",
    "## Training procedure",
    "## Evaluation",
    "## How to use",
    "## Limitations and risks",
    "## License and attribution",
    "## Reproducibility",
)


def runtime_constant(name: str) -> str:
    """A string constant of ``runtime.py``, read from its source (importing it needs MLX)."""
    import ast

    tree = ast.parse((Path(__file__).resolve().parent.parent / "runtime.py").read_text())
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(
            isinstance(t, ast.Name) and t.id == name for t in node.targets
        ):
            return str(ast.literal_eval(node.value))
    raise KeyError(name)


class _Recorded(dict):
    """Manifest section where absent keys read as "not recorded" instead of failing."""

    def __missing__(self, key: str) -> str:
        return "not recorded"


@dataclass(frozen=True)
class CardInputs:
    project: ProjectConfig
    run_dir: Path
    repo_id: str | None
    run: dict[str, Any] | None
    data_manifest: dict[str, Any] | None
    chat_eval: dict[str, Any] | None
    decision_eval: dict[str, Any] | None


def training_status(run: dict[str, Any] | None) -> str:
    """``not trained`` | ``smoke test only`` | ``completed`` | ``failed`` | ``interrupted``."""
    if run is None:
        return "not trained"
    if run.get("profile", {}).get("kind") == "smoke" and run.get("status") == "completed":
        return "smoke test only"
    return str(run.get("status", "unknown"))


def collect_inputs(
    project: ProjectConfig, run_dir: Path | None = None, repo_id: str | None = None
) -> CardInputs:
    """Gather all recorded outputs the card is built from.

    The decision evaluation is only used when it was produced with this exact training
    run; the chat baseline only when it used the configured model revision.
    """
    pub = project.publishing
    run_dir = run_dir or pub.run_dir
    run = read_json(run_dir / RUN_MANIFEST)
    out_dir = project.evaluation.output_dir
    revision = project.training.model.revision
    chat = load_latest(out_dir, "base-chat")
    if chat and chat.get("model", {}).get("revision") != revision:
        chat = None
    decision = load_latest(out_dir, "decision")
    if decision and (run is None or (decision.get("run") or {}).get("run_id") != run.get("run_id")):
        decision = None
    return CardInputs(
        project,
        run_dir,
        repo_id or pub.repo_id,
        run,
        read_json(project.data.output_dir / "manifest.json"),
        chat,
        decision,
    )


def card_metadata(inputs: CardInputs) -> dict[str, Any]:
    """Hugging Face YAML front matter, derived from configuration only."""
    p = inputs.project
    return {
        "base_model": p.training.model.repo_id,
        "library_name": "mlx",
        "license": p.publishing.adapter_license,
        "datasets": [p.data.dataset.repo_id],
        "language": ["en"],
        "pipeline_tag": "text-classification",
        "tags": ["mlx", "lora", "decision-model", "complaint-routing", "qwen3_5"],
    }


def _fmt(value: Any, suffix: str = "") -> str:
    if value is None:
        return NOT_MEASURED
    if isinstance(value, dict):
        return ", ".join(f"{k} {v}" for k, v in value.items())
    if isinstance(value, int) and not isinstance(value, bool):
        return f"{value:,}{suffix}"
    return f"{value}{suffix}"


def _pct(value: float | None) -> str:
    return NOT_MEASURED if value is None else f"{100 * value:.1f}%"


def _eval_row(name: str, ev: dict[str, Any] | None) -> str:
    if ev is None:
        return f"| {name} | {PENDING} | – | – | – |"
    m = ev["metrics"]
    latency = "n/a" if m["mean_latency_ms"] is None else f"{m['mean_latency_ms']} ms"
    return (
        f"| {name} | {m['correct']} / {m['examples']} ({_pct(m['accuracy'])}) | "
        f"{m['invalid_answers']} | {latency} | {ev['evaluated_at'][:10]} |"
    )


def render_card(inputs: CardInputs) -> str:
    """Full README.md for the publication repository."""
    p = inputs.project
    t = p.training
    model = t.model
    ds = p.data.dataset
    run = inputs.run
    status = training_status(run)
    if run and run.get("profile"):
        profile = _Recorded(run["profile"])
        profile_name = str(profile["name"])
    else:
        profile = _Recorded(t.profile("m1_air_8gb").to_dict())
        profile_name = "m1_air_8gb (planned)"
    metrics = (run or {}).get("metrics", {})
    trainable = metrics.get("trainable_parameters") or {}
    env = (run or {}).get("environment", {})
    pkgs = env.get("packages", {})
    manifest = inputs.data_manifest or {}
    stream = manifest.get("training_stream") or {}
    repo = inputs.repo_id or "<your-namespace>/<repo>"
    route_list = ", ".join(f"`{r}`" for r in ROUTES)
    example_criteria = (
        "{\n"
        + "".join(f"                {json.dumps(k)}: {json.dumps(v)},\n" for k, v in ROUTES.items())
        + "            }"
    )

    sections: list[str] = []
    add = sections.append
    add("---\n" + yaml.safe_dump(card_metadata(inputs), sort_keys=False) + "---")
    add(
        "# Qwen3.5-0.8B complaint-routing decision model (MLX)\n\n"
        "A System 1 decision model: it reads a consumer complaint and a typed question with "
        "its allowed options **once** (one forward pass, nothing generated) and returns the "
        f"chosen financial product with a probability for every option. Built on "
        f"[`{model.repo_id}`](https://huggingface.co/{model.repo_id}) with a LoRA adapter and "
        "a small joint schema head, trained and run with Apple MLX.\n\n"
        f"**Training status: {status}.**"
        + (
            ""
            if status == "completed"
            else " The numbers below are only filled in from recorded runs; anything not yet run "
            "is marked as pending or not measured."
        )
    )

    add(
        "## Model overview\n\n"
        "| | |\n|---|---|\n"
        f"| Task | route a complaint to one of 9 products: {route_list} |\n"
        "| Output | typed JSON answer: choice, confidence, probability per allowed option |\n"
        f"| Base model | `{model.repo_id}` (revision `{model.revision}`), text model only |\n"
        "| Added parts | LoRA on every attention, MLP and linear-attention projection; joint "
        "schema head (option pooling, evidence routing, joint field decoding, one logit per "
        "option) |\n"
        "| Framework | MLX / MLX-LM (Apple Silicon, Metal) |\n"
        "| Head design | follows Cloudflare's Clef joint schema model (Apache-2.0), scaled down "
        "and text-only; independent implementation, not affiliated with Cloudflare |\n"
        f"| Project status | experimental / educational; training status: {status} |"
    )

    add(
        "## Model details\n\n"
        f"- **Base model:** `{model.repo_id}` at commit `{model.revision}`, license "
        f"`{model.license}`. MLX-LM loads its text model; the vision tower is not used.\n"
        "- **This repository is not a standalone model.** It holds a LoRA adapter "
        "(`adapters.safetensors`, `adapter_config.json`, MLX-LM format), the head "
        "(`head.safetensors`, `head_config.json`) and the self-contained loader "
        f"`{RUNTIME_NAME}`. All three are applied on top of the exact base model above.\n"
        f"- **Head:** {json.dumps(t.head.to_dict())}.\n"
        f"- **Trainable parameters:** LoRA {_fmt(trainable.get('lora'))}, head "
        f"{_fmt(trainable.get('head'))} (backbone {_fmt(trainable.get('total_backbone'))}).\n"
        "- **Weight format:** safetensors (MLX); base weights bf16, head float32.\n"
        f"- **Software:** mlx {_fmt(pkgs.get('mlx'))}, mlx-lm {_fmt(pkgs.get('mlx-lm'))}."
    )

    add(
        "## Intended use\n\n"
        "Routing short English consumer-finance complaints to the product team that handles "
        "them, with calibrated-looking probabilities that a workflow can threshold, for "
        "experiments and teaching about decision models versus generative answers. The "
        "request may offer any subset (at least two) of the 9 routes; the model only chooses "
        "among the offered ones.\n\n"
        "Not for: decisions about individuals (credit, collections, eligibility), other "
        "domains or label sets without retraining, or unattended routing where a wrong route "
        "harms a consumer. Keep a human review path for low-confidence answers."
    )

    add(
        "## Training data\n\n"
        f"- **Dataset:** [`{ds.repo_id}`](https://huggingface.co/datasets/{ds.repo_id}), "
        f"revision `{ds.revision}`, license `{ds.license}`: short complaints from the US CFPB "
        "Consumer Complaint Database (public data; personal details redacted as `XXXX` by "
        "the CFPB), labelled with 9 products, balanced train split.\n"
        f"- **Rows:** {_fmt(manifest.get('rows'))}.\n"
        f"- **Training stream:** {_fmt(stream.get('requests'))} requests drawn without "
        f"replacement from the train split with seed {p.data.requests.seed}; "
        f"{_pct(stream.get('share_with_option_subset'))} of them offer only a random subset "
        "of the routes (always including the right one), so the head has to read the options "
        "instead of memorising 9 fixed slots.\n"
        "- **Prompt layout:** a system line, `STATE:` with the request state as compact JSON, "
        "then `SCHEMA FIELDS:` with each question's id, type, instruction and every allowed "
        'option as `{"option_id": …, "description": …}`, then an empty think block and '
        "`JOINT SCHEMA DECISIONS:`. System line:\n\n"
        f"```text\n{runtime_constant('SYSTEM_PROMPT')}\n```\n\n"
        f"- **Question:** `{INSTRUCTIONS}`\n"
        "- **Target:** the index of the correct option; loss is cross-entropy over the "
        "offered options.\n"
        "- **Overlap:** no id or complaint text is shared between train, validation and test "
        f"({_fmt(manifest.get('overlap'))}).\n"
        "- **Limitations:** short complaints only, US consumer-finance vocabulary, CFPB "
        "product taxonomy mapped to 9 routes; labels reflect what consumers selected when "
        "filing, which is sometimes debatable."
    )

    add(
        "## Training procedure\n\n"
        "| Setting | Value |\n|---|---|\n"
        f"| Profile | `{profile_name}` |\n"
        f"| LoRA | rank {t.lora.rank}, alpha {t.lora.alpha:g} (scale {t.lora.scale:g}), dropout "
        f"{t.lora.dropout:g}, all 24 blocks |\n"
        f"| LoRA targets | {', '.join(f'`{k}`' for k in t.lora.keys)} |\n"
        f"| Updates | {profile['steps']} × {profile['batch_size']} requests × "
        f"{profile['grad_accumulation_steps']} accumulated |\n"
        f"| Optimizer | AdamW (bias correction), weight decay {profile['weight_decay']}; lr "
        f"{profile['lr_lora']} (LoRA) / {profile['lr_head']} (head) |\n"
        f"| Schedule | lr × min(1, (step+1)/{profile['warmup_steps']}) × ½(1+cos(π·step/"
        f"{profile['steps']})) |\n"
        f"| Gradient clipping | global norm {profile['max_grad_norm']} |\n"
        f"| Gradient checkpointing | {profile['grad_checkpoint']} |\n"
        f"| Seed | {profile['seed']} |\n"
        f"| Final training loss (mean of last 10 updates) | "
        f"{_fmt(metrics.get('final_loss_avg10'))} |\n"
        f"| Training wall time | {_fmt(metrics.get('train_wall_time_s'), ' s')} |\n"
        f"| Peak MLX memory | {_fmt(metrics.get('peak_memory_gb'), ' GB')} |\n"
        + (
            f"| Hardware | {env.get('chip')}, {env.get('memory_gb')} GB unified memory |\n"
            f"| OS / Python | macOS {env.get('os_version')} / Python {env.get('python')} |\n"
            if env
            else "| Hardware / OS | not recorded (no training run yet) |\n"
        )
        + f"| Run id | {_fmt((run or {}).get('run_id'))} |"
    )

    add(
        "## Evaluation\n\n"
        f"The {p.data.dataset.expected_rows['test']} test complaints of the dataset, never "
        "used for training, each offered all 9 routes.\n\n"
        "- **Decision model (System 1):** one forward pass; the answer is the option with the "
        "highest logit. Reported with expected calibration error of its confidences.\n"
        "- **Chat baseline (System 2):** the base model asked the same request as a chat "
        f"(greedy, thinking off, at most {p.evaluation.chat_max_tokens} new tokens); the "
        "answer is the `product` of the last JSON object in the reply. A reply without a "
        "valid route counts as wrong.\n\n"
        "| Model | Accuracy | Invalid answers | Mean latency | Date |\n"
        "|---|---|---|---|---|\n"
        + _eval_row(f"Base `{model.repo_id}`, chat", inputs.chat_eval)
        + "\n"
        + _eval_row("Decision model (this repository)", inputs.decision_eval)
        + "\n\n"
        "Accuracy is measured against the dataset's labels; a 95% interval at this size is "
        f"up to about ±{100 * 1.96 * (0.25 / p.data.dataset.expected_rows['test']) ** 0.5:.0f}"
        " percentage points. Latency is per complaint on the machine that ran the evaluation."
    )

    add(
        "## How to use\n\n"
        "Requires macOS on Apple Silicon.\n\n"
        '```bash\npip install "mlx-lm>=0.32.2,<0.33" huggingface_hub\n```\n\n'
        "```python\n"
        "import json, sys\n"
        "from huggingface_hub import snapshot_download\n\n"
        f'folder = snapshot_download("{repo}")\n'
        "sys.path.insert(0, folder)\n"
        f"from {RUNTIME_NAME.removesuffix('.py')} import decide, load_release\n\n"
        f'model, tokenizer = load_release("{model.repo_id}", "{model.revision}", folder)\n'
        "request = {\n"
        '    "state": {"complaint": "I wired $1,200 to a contractor; it left my account "\n'
        '                            "but never arrived and the bank will not trace it."},\n'
        '    "questions": {\n'
        '        "product": {\n'
        '            "type": "choice",\n'
        f'            "instructions": "{INSTRUCTIONS}",\n'
        f'            "criteria": {example_criteria},\n'
        "        }\n"
        "    },\n"
        "}\n"
        "print(json.dumps(decide(model, tokenizer, request), indent=2))\n"
        "```\n\n"
        "The response holds `answers.product.choice`, its `confidence` and the probability "
        "of every offered option; `usage.output_tokens` is 0 because nothing is generated. "
        "To restrict the choice, pass fewer criteria (at least two)."
    )

    add(
        "## Limitations and risks\n\n"
        "- Misroutes complaints that mention several products (a credit-card dispute that is "
        "also on a credit report) and complaints whose label is itself debatable.\n"
        "- Probabilities are a model output, not guarantees; check calibration on your own "
        "data before thresholding.\n"
        "- Trained on short English US complaints and these 9 routes; other wording, lengths, "
        "languages or taxonomies are out of distribution.\n"
        "- Only `choice` questions are supported by the published runtime.\n"
        "- The head was trained jointly with this adapter; neither works with another base "
        "model or revision.\n"
        "- Small model on consumer hardware: no production-readiness guarantees.\n"
        "- **Do not use it to make or justify decisions about individual consumers.**"
    )

    add(
        "## License and attribution\n\n"
        f"- **Weights in this repository:** released under `{p.publishing.adapter_license}` "
        "(as configured by the publisher).\n"
        f"- **Base model:** `{model.repo_id}`, license `{model.license}`.\n"
        f"- **Data:** `{ds.repo_id}`, license `{ds.license}`; derived from the US CFPB "
        "Consumer Complaint Database (public data).\n"
        "- **Head design:** follows Cloudflare's Clef joint schema model (Apache-2.0); this "
        "is an independent MLX implementation and is not affiliated with or endorsed by "
        "Cloudflare.\n"
        "- The code, data and model licenses are independent; check each before reuse."
    )

    add(
        "## Reproducibility\n\n"
        "The project that produced this model keeps every setting in `configs/` and drives "
        "everything through `make`:\n\n"
        "```bash\n"
        "make install\n"
        "make data-prepare        # splits + the ordered training-request stream\n"
        "make data-validate\n"
        "make evaluate-chat       # System 2 baseline\n"
        f"make train PROFILE={profile_name.split()[0]}\n"
        "make evaluate            # System 1 decision model\n"
        "make card\n"
        "make publish-dry-run\n"
        "```\n\n"
        f"Artifacts: run directory `{display(inputs.run_dir, p.root)}` with `{RUN_MANIFEST}` "
        "(configuration, versions, loss per update, timings), evaluation records under "
        f"`{display(p.evaluation.output_dir, p.root)}/`. Hardware: an Apple Silicon Mac; the "
        "`m1_air_8gb` profile targets an 8 GB M1 MacBook Air."
    )
    return "\n\n".join(sections) + "\n"


def write_card(inputs: CardInputs, output: Path) -> Path:
    """Write the card to ``output`` (refuses to write over a hand-written README)."""
    if output.name == "README.md" and output.parent.resolve() == inputs.project.root.resolve():
        raise ValueError("Refusing to overwrite the project's own README.md with a model card")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(render_card(inputs), encoding="utf-8")
    return output


def parse_front_matter(card: str) -> dict[str, Any]:
    """Parse the YAML metadata block of a card (raises ValueError when absent/invalid)."""
    if not card.startswith("---\n"):
        raise ValueError("card does not start with YAML front matter")
    end = card.find("\n---", 4)
    if end < 0:
        raise ValueError("unterminated YAML front matter")
    data = yaml.safe_load(card[4:end])
    if not isinstance(data, dict):
        raise ValueError("front matter is not a mapping")
    return data
