"""Torch-free inference for OpenMed DistilBERT token-classification models (numpy + HF `tokenizers`).

Used where PyTorch cannot be installed (locked-down sandboxes).  Reads the model's safetensors directly
(bfloat16/float16/float32), runs the 6-layer DistilBERT encoder in float32 numpy, and aggregates word-level
entities with the 'first sub-token' rule (same as transformers' aggregation_strategy="first").
"""
from __future__ import annotations

import json, struct
from pathlib import Path

import numpy as np

_DT = {"F32": np.float32, "F16": np.float16, "BF16": np.uint16}


def load_safetensors(path: Path) -> dict:
    raw = Path(path).read_bytes()
    n = struct.unpack("<Q", raw[:8])[0]; hdr = json.loads(raw[8:8 + n]); base = 8 + n; out = {}
    for k, v in hdr.items():
        if k == "__metadata__":
            continue
        a, b = v["data_offsets"]; arr = np.frombuffer(raw[base + a: base + b], dtype=_DT[v["dtype"]])
        if v["dtype"] == "BF16":
            arr = (arr.astype(np.uint32) << 16).view(np.float32)
        out[k] = arr.astype(np.float32).reshape(v["shape"])
    return out


def _ln(x, g, b, eps=1e-12):
    m = x.mean(-1, keepdims=True); v = ((x - m) ** 2).mean(-1, keepdims=True)
    return (x - m) / np.sqrt(v + eps) * g + b


def _gelu(x):
    from scipy.special import erf
    return 0.5 * x * (1 + erf(x / np.sqrt(2)))


class NumpyDistilBertNER:
    def __init__(self, model_dir: str, max_len: int = 512):
        from tokenizers import Tokenizer
        d = Path(model_dir); cfg = json.loads((d / "config.json").read_text())
        self.id2label = {int(k): v for k, v in cfg["id2label"].items()}
        self.h, self.L = cfg["n_heads"], cfg["n_layers"]; self.max_len = max_len
        w = load_safetensors(d / "model.safetensors")
        p = "distilbert." if any(k.startswith("distilbert.") for k in w) else ""
        self.we = w[p + "embeddings.word_embeddings.weight"]; self.pe = w[p + "embeddings.position_embeddings.weight"]
        self.eln = (w[p + "embeddings.LayerNorm.weight"], w[p + "embeddings.LayerNorm.bias"])
        self.layers = []
        for i in range(self.L):
            q = f"{p}transformer.layer.{i}."
            g = lambda n: w[q + n]
            self.layers.append(dict(
                qkv_w=np.concatenate([g("attention.q_lin.weight"), g("attention.k_lin.weight"), g("attention.v_lin.weight")]).T.copy(),
                qkv_b=np.concatenate([g("attention.q_lin.bias"), g("attention.k_lin.bias"), g("attention.v_lin.bias")]),
                o_w=g("attention.out_lin.weight").T.copy(), o_b=g("attention.out_lin.bias"),
                ln1=(g("sa_layer_norm.weight"), g("sa_layer_norm.bias")),
                f1_w=g("ffn.lin1.weight").T.copy(), f1_b=g("ffn.lin1.bias"),
                f2_w=g("ffn.lin2.weight").T.copy(), f2_b=g("ffn.lin2.bias"),
                ln2=(g("output_layer_norm.weight"), g("output_layer_norm.bias"))))
        self.cw, self.cb = w["classifier.weight"].T.copy(), w["classifier.bias"]
        self.tok = Tokenizer.from_file(str(d / "tokenizer.json"))
        self.tok.no_padding(); self.tok.enable_truncation(max_len)

    def _forward(self, ids: np.ndarray) -> np.ndarray:
        T = len(ids); x = _ln(self.we[ids] + self.pe[:T], *self.eln); dh = x.shape[1] // self.h
        for ly in self.layers:
            qkv = x @ ly["qkv_w"] + ly["qkv_b"]; q, k, v = np.split(qkv, 3, axis=1)
            q = q.reshape(T, self.h, dh).transpose(1, 0, 2) / np.sqrt(dh)
            k = k.reshape(T, self.h, dh).transpose(1, 2, 0); v = v.reshape(T, self.h, dh).transpose(1, 0, 2)
            s = q @ k; s -= s.max(-1, keepdims=True); a = np.exp(s); a /= a.sum(-1, keepdims=True)
            ctx = (a @ v).transpose(1, 0, 2).reshape(T, -1)
            x = _ln(ctx @ ly["o_w"] + ly["o_b"] + x, *ly["ln1"])
            f = _gelu(x @ ly["f1_w"] + ly["f1_b"]) @ ly["f2_w"] + ly["f2_b"]
            x = _ln(f + x, *ly["ln2"])
        return x @ self.cw + self.cb

    def entities(self, text: str, threshold: float = .5) -> list[tuple[str, str]]:
        if not text or not text.strip():
            return []
        enc = self.tok.encode(text); ids = np.asarray(enc.ids)
        logits = self._forward(ids); pr = np.exp(logits - logits.max(-1, keepdims=True)); pr /= pr.sum(-1, keepdims=True)
        words = enc.word_ids; offs = enc.offsets; out = []; cur = None; seen = set()
        for t, wid in enumerate(words):
            if wid is None or wid in seen:
                continue                                   # first sub-token of each word only
            seen.add(wid); lab = self.id2label[int(pr[t].argmax())]; sc = float(pr[t].max())
            # word span = union of its sub-token offsets
            span = [o for o, ww in zip(offs, words) if ww == wid]; s0, s1 = span[0][0], span[-1][1]
            if lab == "O" or sc < threshold:
                if cur: out.append(cur); cur = None
                continue
            typ = lab.split("-", 1)[-1]
            if cur and (lab.startswith("I-") or not lab.startswith("B-")) and cur[0] == typ:
                cur = (typ, cur[1], s1)
            else:
                if cur: out.append(cur)
                cur = (typ, s0, s1)
        if cur:
            out.append(cur)
        return [(typ, text[a:b].lower().strip()) for typ, a, b in out]

    def entities_long(self, text: str, window: int = 1500, max_chars: int = 12000) -> list[tuple[str, str]]:
        t = (text or "")[:max_chars]; res = []
        for k in range(0, len(t), window):
            res += self.entities(t[k:k + window])
        return res
