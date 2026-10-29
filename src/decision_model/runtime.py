"""Decision-model runtime (MLX): prompt encoding, joint schema head, typed answers.

This module is self-contained on purpose (it imports only ``mlx``, ``mlx_lm`` and the
standard library) so it can be shipped next to a published adapter and used without the
rest of this project.

How a decision is made:

1. One prompt holds the STATE and the SCHEMA: every question with every allowed option
   written out as text. The state comes first, so every option token has read it.
2. The language model reads the prompt once (one forward pass). Its vocabulary head is
   not used; nothing is generated.
3. The joint schema head reads the final hidden states. It pools the tokens of each
   question and each option, lets every option collect evidence from the whole prompt,
   decides the questions jointly, and produces one logit per allowed option.
4. A softmax per question gives probabilities; plain code writes the JSON answer, so the
   output is always valid and always one of the allowed options.

The design follows Cloudflare's Clef joint schema model (Apache-2.0), scaled down and
text-only; this implementation is independent and not affiliated with Cloudflare.
"""

from __future__ import annotations

import json
import math
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import mlx.core as mx
import mlx.nn as nn

SYSTEM_PROMPT = (
    "Read the complete state and schema. Decide every field jointly. Each answer "
    "must be exactly one of that field's allowed options."
)
PREFIX = "<|im_start|>system\n{system}<|im_end|>\n<|im_start|>user\nSTATE:\n"
SUFFIX = "\n<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\nJOINT SCHEMA DECISIONS:"
QUESTION_TYPES = {"noul": 0, "choice": 1, "score": 2}
DEFAULT_HEAD_CONFIG = {
    "hidden_size": 1024,
    "width": 512,
    "routing_layers": 1,
    "layers": 2,
    "heads": 8,
    "feedforward": 2048,
}
DEFAULT_MAX_LENGTH = 4096
HEAD_WEIGHTS = "head.safetensors"
HEAD_CONFIG = "head_config.json"

Tokenize = Callable[[str], list[int]]


# --------------------------------------------------------------------------- encoding


def render(value: Any) -> str:
    """Strings as they are; everything else as compact, key-sorted JSON."""
    if isinstance(value, str):
        return value
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"), sort_keys=True)


def choice_options(question: dict[str, Any]) -> list[tuple[str, Any]]:
    """(option id, description) pairs of a ``choice`` question, sorted by id."""
    if question.get("type") != "choice":
        raise ValueError(f"only 'choice' questions are supported, got {question.get('type')!r}")
    criteria = question.get("criteria")
    if not isinstance(criteria, dict) or len(criteria) < 2:
        raise ValueError("a choice question needs at least two criteria")
    return sorted((str(k), v) for k, v in criteria.items())


@dataclass(frozen=True)
class EncodedQuestion:
    question_id: str
    question_type: int
    question_span: tuple[int, int]  # token range of the instruction
    option_spans: tuple[tuple[int, int], ...]  # token range of each option's text
    option_ids: tuple[str, ...]


@dataclass(frozen=True)
class EncodedRecord:
    input_ids: tuple[int, ...]
    questions: tuple[EncodedQuestion, ...]


def encode_request(
    tokenize: Tokenize, request: dict[str, Any], max_length: int = DEFAULT_MAX_LENGTH
) -> EncodedRecord:
    """Token ids of the decision prompt plus the spans of every question and option.

    Every piece is tokenized separately and concatenated, so spans are exact. The state
    is truncated (never the schema) when the prompt would exceed ``max_length``.
    """
    schema: list[int] = tokenize("\n\nSCHEMA FIELDS:\n")
    pending = []
    for qi, (qid, q) in enumerate(request["questions"].items()):
        schema += tokenize(f"\nFIELD {qi + 1}\nID: {qid}\nTYPE: {q['type']}\nINSTRUCTION: ")
        q0 = len(schema)
        schema += tokenize(render(q.get("instructions") or qid))
        q1 = len(schema)
        schema += tokenize("\nALLOWED OPTIONS:\n")
        spans, ids = [], []
        for oi, (oid, desc) in enumerate(choice_options(q)):
            schema += tokenize(f"OPTION {oi + 1}: ")
            o0 = len(schema)
            option = {"option_id": oid, **({"description": desc} if desc is not None else {})}
            schema += tokenize(render(option))
            spans.append((o0, len(schema)))
            ids.append(oid)
            schema += tokenize("\n")
        schema += tokenize("END FIELD\n")
        pending.append((qid, QUESTION_TYPES[q["type"]], (q0, q1), spans, ids))
    prefix = tokenize(PREFIX.format(system=SYSTEM_PROMPT))
    suffix = tokenize(SUFFIX)
    budget = max_length - len(prefix) - len(schema) - len(suffix)
    if budget <= 0:
        raise ValueError(f"the schema alone needs more than max_length={max_length} tokens")
    state = tokenize(render(request["state"]))[:budget]
    offset = len(prefix) + len(state)

    def shift(span: tuple[int, int]) -> tuple[int, int]:
        return span[0] + offset, span[1] + offset

    questions = tuple(
        EncodedQuestion(qid, qtype, shift(qspan), tuple(shift(s) for s in spans), tuple(ids))
        for qid, qtype, qspan, spans, ids in pending
    )
    return EncodedRecord(tuple(prefix + state + schema + suffix), questions)


def tokenizer_fn(tokenizer: Any) -> Tokenize:
    """``text -> ids`` without special tokens, for an MLX-LM or Hugging Face tokenizer.

    MLX-LM's ``TokenizerWrapper`` forwards ``encode`` to the Hugging Face tokenizer.
    """
    return lambda text: list(tokenizer.encode(text, add_special_tokens=False))


def collate(records: list[EncodedRecord], pad_id: int) -> tuple[mx.array, mx.array]:
    """Right-padded ids and a boolean padding mask (True = padding).

    Right padding is safe for a causal model: real tokens never attend to later
    positions, and the head masks the padding explicitly.
    """
    n = max(len(r.input_ids) for r in records)
    ids = [list(r.input_ids) + [pad_id] * (n - len(r.input_ids)) for r in records]
    pad = [[False] * len(r.input_ids) + [True] * (n - len(r.input_ids)) for r in records]
    return mx.array(ids), mx.array(pad)


# --------------------------------------------------------------------------- head


def _additive_mask(pad: mx.array) -> mx.array:
    """(L,) padding flags → (1, 1, 1, L) additive attention mask."""
    return mx.where(pad, mx.array(-1e9, dtype=mx.float32), mx.array(0.0, dtype=mx.float32))[
        None, None, None, :
    ]


def _normalize(x: mx.array, eps: float = 1e-12) -> mx.array:
    return x / mx.maximum(mx.linalg.norm(x, axis=-1, keepdims=True), eps)


class Attention(nn.Module):
    """Multi-head attention with input/output projections (with bias) and explicit softmax.

    Written out instead of ``nn.MultiHeadAttention`` because MLX's fused attention kernel
    for very short query sequences (here: a handful of options or fields) has no gradient.
    """

    def __init__(self, dims: int, heads: int):
        super().__init__()
        self.heads = heads
        self.query_proj = nn.Linear(dims, dims)
        self.key_proj = nn.Linear(dims, dims)
        self.value_proj = nn.Linear(dims, dims)
        self.out_proj = nn.Linear(dims, dims)

    def __call__(
        self, queries: mx.array, keys: mx.array, values: mx.array, mask: mx.array | None = None
    ) -> mx.array:
        b, lq, d = queries.shape
        lk = keys.shape[1]
        h = self.heads
        q = self.query_proj(queries).reshape(b, lq, h, d // h).transpose(0, 2, 1, 3)
        k = self.key_proj(keys).reshape(b, lk, h, d // h).transpose(0, 2, 1, 3)
        v = self.value_proj(values).reshape(b, lk, h, d // h).transpose(0, 2, 1, 3)
        scores = (q @ k.transpose(0, 1, 3, 2)) * (1.0 / math.sqrt(d // h))
        if mask is not None:
            scores = scores + mask
        out = mx.softmax(scores, axis=-1) @ v
        return self.out_proj(out.transpose(0, 2, 1, 3).reshape(b, lq, d))


class FeedForward(nn.Module):
    """Linear → GELU (exact) → Linear."""

    def __init__(self, dims: int, hidden: int, out_dims: int | None = None):
        super().__init__()
        self.up = nn.Linear(dims, hidden)
        self.down = nn.Linear(hidden, dims if out_dims is None else out_dims)

    def __call__(self, x: mx.array) -> mx.array:
        return self.down(nn.gelu(self.up(x)))


class EvidenceRoutingLayer(nn.Module):
    """Each option (a query) attends over every prompt token (the memory) for evidence."""

    def __init__(self, width: int, heads: int, feedforward: int):
        super().__init__()
        self.query_norm = nn.LayerNorm(width)
        self.memory_norm = nn.LayerNorm(width)
        self.attention = Attention(width, heads)
        self.feedforward_norm = nn.LayerNorm(width)
        self.feedforward = FeedForward(width, feedforward)

    def __call__(self, queries: mx.array, memory: mx.array, mask: mx.array) -> mx.array:
        mem = self.memory_norm(memory)
        queries = queries + self.attention(self.query_norm(queries), mem, mem, mask)
        return queries + self.feedforward(self.feedforward_norm(queries))


class FieldDecoderLayer(nn.Module):
    """Pre-norm decoder block: fields attend to each other, then to the prompt."""

    def __init__(self, width: int, heads: int, feedforward: int):
        super().__init__()
        self.self_norm = nn.LayerNorm(width)
        self.self_attention = Attention(width, heads)
        self.cross_norm = nn.LayerNorm(width)
        self.cross_attention = Attention(width, heads)
        self.feedforward_norm = nn.LayerNorm(width)
        self.feedforward = FeedForward(width, feedforward)

    def __call__(self, x: mx.array, memory: mx.array, mask: mx.array) -> mx.array:
        h = self.self_norm(x)
        x = x + self.self_attention(h, h, h)
        x = x + self.cross_attention(self.cross_norm(x), memory, memory, mask)
        return x + self.feedforward(self.feedforward_norm(x))


class JointSchemaHead(nn.Module):
    """Option logits from the language model's hidden states (≈16M parameters)."""

    def __init__(
        self,
        hidden_size: int,
        width: int,
        routing_layers: int,
        layers: int,
        heads: int,
        feedforward: int,
    ):
        super().__init__()
        self.hidden_norm = nn.LayerNorm(hidden_size)
        self.memory_projection = nn.Linear(hidden_size, width, bias=False)
        self.question_projection = nn.Linear(hidden_size, width, bias=False)
        self.option_question_projection = nn.Linear(hidden_size, width, bias=False)
        self.global_projection = nn.Linear(hidden_size, width, bias=False)
        self.option_context_projection = nn.Linear(hidden_size, width, bias=False)
        self.option_lexical_projection = nn.Linear(hidden_size, width, bias=False)
        self.type_embedding = nn.Embedding(len(QUESTION_TYPES), width)
        self.evidence_layers = [
            EvidenceRoutingLayer(width, heads, feedforward) for _ in range(routing_layers)
        ]
        self.option_summary_norm = nn.LayerNorm(width)
        self.layers = [FieldDecoderLayer(width, heads, feedforward) for _ in range(layers)]
        self.field_norm = nn.LayerNorm(width)
        self.option_norm = nn.LayerNorm(width)
        self.residual_scorer = FeedForward(width * 4, width, 1)
        self.prior_logit_scale = mx.zeros(())
        self.joint_logit_scale = mx.zeros(())
        self.residual_gate = mx.zeros(())

    def record_logits(
        self, h: mx.array, ids: mx.array, pad: mx.array, rec: EncodedRecord, embedding: mx.array
    ) -> list[mx.array]:
        """One logit vector per question for one (unpadded-length) prompt."""
        h = self.hidden_norm(h.astype(mx.float32))
        mask = _additive_mask(pad)
        memory = self.memory_projection(h)[None]
        length = len(rec.input_ids)
        global_vector = h[length - 1]  # the last token has read the whole prompt

        spans = [q.question_span for q in rec.questions]
        spans += [s for q in rec.questions for s in q.option_spans]
        counts = [len(q.option_spans) for q in rec.questions]
        n_tokens = h.shape[0]
        rows = []
        for s0, s1 in spans:
            weights = [0.0] * n_tokens
            for t in range(s0, s1):
                weights[t] = 1.0 / (s1 - s0)
            rows.append(weights)
        pool = mx.array(rows, dtype=mx.float32)  # span means as one matmul
        nq = len(rec.questions)
        means = pool @ h
        lexical = pool[nq:] @ embedding[ids].astype(mx.float32)  # the options' own words
        qvec = means[:nq]
        bounds = [sum(counts[:i]) for i in range(1, len(counts))]
        contexts = mx.split(means[nq:], bounds) if bounds else [means[nq:]]
        lexicals = mx.split(lexical, bounds) if bounds else [lexical]
        types = mx.array([q.question_type for q in rec.questions])

        options = mx.concatenate(
            [
                self.option_context_projection(c)
                + self.option_lexical_projection(lx)
                + self.option_question_projection(qvec[i])[None]
                for i, (c, lx) in enumerate(zip(contexts, lexicals, strict=True))
            ]
        )[None]
        for layer in self.evidence_layers:
            options = layer(options, memory, mask)
        per_question = mx.split(options[0], bounds) if bounds else [options[0]]

        base = self.question_projection(qvec)
        summaries = mx.stack(
            [
                (mx.softmax(o @ f / math.sqrt(o.shape[-1]), axis=0)[:, None] * o).sum(axis=0)
                for f, o in zip(base, per_question, strict=True)
            ]
        )
        fields = (
            base
            + self.option_summary_norm(summaries)
            + self.global_projection(global_vector)[None]
            + self.type_embedding(types)
        )
        fields = fields[None]
        for decoder in self.layers:  # questions see each other and the prompt: decided jointly
            fields = decoder(fields, memory, mask)
        fields = self.field_norm(fields[0])

        prior_scale = mx.exp(mx.minimum(self.prior_logit_scale, math.log(100.0)))
        joint_scale = mx.exp(mx.minimum(self.joint_logit_scale, math.log(100.0)))
        logits = []
        for i, (field, lx, o) in enumerate(zip(fields, lexicals, per_question, strict=True)):
            anchor = _normalize(qvec[i] + global_vector)
            prior = prior_scale * (_normalize(lx) @ anchor)
            o = self.option_norm(o)
            f = mx.broadcast_to(field[None], o.shape)
            cosine = (_normalize(f) * _normalize(o)).sum(axis=-1)
            joint = joint_scale * cosine
            joint = joint + self.residual_scorer(
                mx.concatenate([f, o, f * o, mx.abs(f - o)], axis=-1)
            ).squeeze(-1)
            logits.append(prior + mx.sigmoid(self.residual_gate) * joint)
        return logits


class DecisionModel(nn.Module):
    """An MLX-LM causal LM (optionally with LoRA layers) plus the joint schema head."""

    def __init__(self, backbone: nn.Module, head: JointSchemaHead):
        super().__init__()
        self.backbone = backbone
        self.head = head

    def text_model(self) -> Any:
        lm: Any = getattr(self.backbone, "language_model", self.backbone)
        return lm.model

    def __call__(
        self, ids: mx.array, pad: mx.array, records: list[EncodedRecord]
    ) -> list[list[mx.array]]:
        text = self.text_model()
        hidden = text(ids)
        embedding = text.embed_tokens.weight
        return [
            self.head.record_logits(hidden[b], ids[b], pad[b], rec, embedding)
            for b, rec in enumerate(records)
        ]


# --------------------------------------------------------------------------- answers


def choice_answer(option_ids: tuple[str, ...], logits: mx.array) -> dict[str, Any]:
    """Typed answer for one choice question: the argmax plus every probability."""
    raw: Any = mx.softmax(logits.astype(mx.float32), axis=-1).tolist()
    probs = [float(p) for p in raw]
    table = dict(zip(option_ids, probs, strict=True))
    choice = max(option_ids, key=table.__getitem__)
    return {
        "type": "choice",
        "choice": choice,
        "confidence": round(table[choice], 4),
        "probabilities": {o: round(p, 4) for o, p in table.items()},
    }


def decide(
    model: DecisionModel,
    tokenizer: Any,
    request: dict[str, Any],
    max_length: int = DEFAULT_MAX_LENGTH,
) -> dict[str, Any]:
    """A ``{state, questions}`` request in, a ``{answers, usage}`` response out."""
    enc = encode_request(tokenizer_fn(tokenizer), request, max_length)
    ids, pad = collate([enc], tokenizer.pad_token_id)
    logits = model(ids, pad, [enc])[0]
    answers = {
        q.question_id: choice_answer(q.option_ids, lg)
        for q, lg in zip(enc.questions, logits, strict=True)
    }
    return {
        "model": request.get("model", "decision-model"),
        "answers": answers,
        "usage": {"input_tokens": len(enc.input_ids), "output_tokens": 0},
    }


def load_release(
    base_model: str, revision: str | None, folder: str | Path
) -> tuple[DecisionModel, Any]:
    """Base model + LoRA adapter + head from a release folder (as written by training)."""
    from mlx_lm import load

    folder = Path(folder)
    backbone, tokenizer = load(base_model, revision=revision, adapter_path=str(folder))[:2]
    config = json.loads((folder / HEAD_CONFIG).read_text())
    head = JointSchemaHead(**config)
    head.load_weights(str(folder / HEAD_WEIGHTS))
    model = DecisionModel(backbone, head)
    model.eval()
    return model, tokenizer
