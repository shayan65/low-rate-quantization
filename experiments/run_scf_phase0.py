"""Phase 0: a falsification gate for the self-consistent quantization direction.

Test A -- is the forward-only SCF loop redundant?
    In a feedforward stack the input statistics of layer L depend only on layers
    before it, so quantizing in topological order while propagating quantized
    activations should reach the fixed point in ONE pass.  We compare that
    one-pass `sequential` arm against our iterative `scf_forward` loop.  If they
    agree, the iteration buys nothing and the forward-only framing is dead
    (which is also what GPTAQ and CoreQ imply).

Test B -- is there a downstream residual worth solving?  THE REAL GATE.
    The per-layer second-order cost factorizes as
        sum_j  S_j * dw_j^T H^xx dw_j ,   S_j = E[(dL/dy_j)^2]
    where H^xx depends on UPSTREAM layers and S depends on DOWNSTREAM layers.
    Only the second makes the dependency graph cyclic and forces iteration.
    We measure how much S actually moves when the model is quantized, against
    the forward residual for the same layers.

    KILL CRITERION (predeclared): if the backward residual is the same order as
    the forward one (<= 2x it), the downstream coupling is negligible, iteration
    cannot help, and the SCF direction is abandoned.

Everything runs under a hard wall-clock cap.
"""

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

from run_scq_qwen import atomic_json, evaluate, find_layers, get_module, make_codec, paired_bootstrap
from scq import SCFConfig, calibration_nll, capture_inputs, run_scf, scf_summary


class Budget:
    def __init__(self, seconds: float):
        self.t0 = time.monotonic()
        self.cap = seconds

    def left(self) -> float:
        return self.cap - (time.monotonic() - self.t0)

    def check(self, stage: str, need: float = 0.0) -> bool:
        ok = self.left() > need
        if not ok:
            print(json.dumps({"skip": stage, "seconds_left": round(self.left(), 1)}), flush=True)
        return ok

    def stamp(self) -> float:
        return round(time.monotonic() - self.t0, 1)


@torch.no_grad()
def install(params, weights):
    for k, w in weights.items():
        params[k].copy_(w.to(params[k].dtype))


@torch.no_grad()
def restore(params, originals):
    for k, p in params.items():
        p.copy_(originals[k].to(p.dtype).cuda())


def capture_sensitivity(model, modules, windows, batch=4, device="cuda"):
    """S_l = E[(dL/dy_l)^2] per output channel, measured on the CURRENT weights.

    Only the target weights carry requires_grad, so autograd builds the graph
    through them and the module backward hooks see dL/dy without paying for a
    full-model gradient.
    """
    acc, cnt, handles = {}, {}, []

    def mk(name):
        def hook(_m, _gin, gout):
            g = gout[0].detach()
            flat = g.reshape(-1, g.shape[-1]).float()
            s = flat.square().sum(0).double()
            if name in acc:
                acc[name] += s
                cnt[name] += flat.shape[0]
            else:
                acc[name], cnt[name] = s, flat.shape[0]

        return hook

    saved = {}
    for name, mod in modules.items():
        saved[name] = mod.weight.requires_grad
        mod.weight.requires_grad_(True)
        handles.append(mod.register_full_backward_hook(mk(name)))
    try:
        for i in range(0, len(windows), batch):
            x = torch.tensor(windows[i : i + batch], device=device)
            logits = model(x[:, :-1], use_cache=False).logits.float()
            loss = F.cross_entropy(
                logits.reshape(-1, logits.shape[-1]), x[:, 1:].reshape(-1), reduction="mean"
            )
            model.zero_grad(set_to_none=True)
            loss.backward()
            del x, logits, loss
    finally:
        for h in handles:
            h.remove()
        for name, mod in modules.items():
            mod.weight.requires_grad_(saved[name])
        model.zero_grad(set_to_none=True)
    return {k: (acc[k] / max(cnt[k], 1)).float() for k in acc}


def rel_residual(a: dict, b: dict) -> dict:
    """Per-key ||a - b|| / ||b||, plus a pooled value."""
    out, num, den = {}, 0.0, 0.0
    for k in b:
        d = (a[k].double() - b[k].double()).norm().item()
        n = b[k].double().norm().item()
        out[k] = d / max(n, 1e-30)
        num += d**2
        den += n**2
    out["_pooled"] = math.sqrt(num) / max(math.sqrt(den), 1e-30)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="../models/Qwen3.5-0.8B-Base")
    ap.add_argument("--data", default="../data/wikitext2")
    ap.add_argument("--out", default="../results/scf_phase0_v1")
    ap.add_argument("--eval-blocks", type=int, default=256)
    ap.add_argument("--fit-windows", type=int, default=32)
    ap.add_argument("--guard-windows", type=int, default=32)
    ap.add_argument("--scf-iters", type=int, default=4)
    ap.add_argument("--codec", default="vq2d")
    ap.add_argument("--seconds", type=float, default=1200, help="hard wall-clock cap")
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    B = Budget(a.seconds)

    tok = AutoTokenizer.from_pretrained(a.model)
    train_ids = tok(
        "\n".join(pd.read_parquet(f"{a.data}/train.parquet").text.tolist()), add_special_tokens=False
    ).input_ids
    val_ids = tok(
        "\n".join(pd.read_parquet(f"{a.data}/validation.parquet").text.tolist()),
        add_special_tokens=False,
    ).input_ids
    rng = np.random.default_rng(20260921)
    starts = np.sort(
        rng.choice(len(train_ids) - 129, a.fit_windows + a.guard_windows, replace=False)
    ).tolist()
    fit = [train_ids[s : s + 129] for s in starts[: a.fit_windows]]
    guard = [train_ids[s : s + 129] for s in starts[a.fit_windows :]]
    blocks = [val_ids[i * 128 : i * 128 + 129] for i in range(a.eval_blocks)]
    blocks = [b for b in blocks if len(b) == 129]

    model = AutoModelForCausalLM.from_pretrained(
        a.model, dtype=torch.bfloat16, device_map="cuda"
    ).eval()
    for p in model.parameters():
        p.requires_grad_(False)
    layers = find_layers(model)
    params = {f"L{i}": get_module(model, i).weight for i in layers}
    modules = {f"L{i}": get_module(model, i) for i in layers}
    originals = {k: p.detach().clone().cpu() for k, p in params.items()}
    codec = make_codec(a.codec)
    oneshot_weights = None

    res = {
        "plan": {
            "model": a.model,
            "codec": a.codec,
            "layers": layers,
            "eval_blocks": len(blocks),
            "eval_targets": 128 * len(blocks),
            "fit_windows": a.fit_windows,
            "guard_windows": a.guard_windows,
            "cap_seconds": a.seconds,
            "kill_criterion": "abandon SCF if backward residual <= 2x forward residual",
        },
        "testA": {},
        "testB": {},
    }
    atomic_json(out / "results.json", res)
    print(json.dumps({"stage": "setup_done", "t": B.stamp(), "layers": len(layers)}), flush=True)

    # ---------------- Test A ----------------
    restore(params, originals)
    bf = evaluate(model, blocks)
    res["testA"]["bf16"] = {k: v for k, v in bf.items() if k != "block_nll"}
    blocks_bf = bf["block_nll"]
    print(json.dumps({"arm": "bf16", "nll": bf["nll"], "t": B.stamp()}), flush=True)

    def record(name, ev, extra=None):
        d, lo, hi = paired_bootstrap(ev["block_nll"], blocks_bf)
        rec = {k: v for k, v in ev.items() if k != "block_nll"}
        rec.update(delta_nll=ev["nll"] - bf["nll"], delta_ci=[lo, hi], **(extra or {}))
        res["testA"][name] = rec
        res["testA"][name]["block_nll"] = ev["block_nll"]
        atomic_json(out / "results.json", {**res, "testA": {
            k: {kk: vv for kk, vv in v.items() if kk != "block_nll"}
            for k, v in res["testA"].items()}})
        print(json.dumps({"arm": name, "nll": ev["nll"], "delta": rec["delta_nll"],
                          "ci": [lo, hi], "t": B.stamp()}), flush=True)

    # one-shot: all statistics from the BF16 model
    if B.check("one_shot", 120):
        restore(params, originals)
        h0 = capture_inputs(model, modules, fit)
        w = {}
        with torch.no_grad():
            for k in params:
                q, _, _ = codec(k, originals[k].cuda().float(), h0.get(k), {})
                w[k] = q
        install(params, w)
        record("one_shot", evaluate(model, blocks))
        oneshot_weights = w

    # sequential: topological order, statistics recomputed with earlier layers quantized
    if B.check("sequential", 150):
        restore(params, originals)
        t0 = time.monotonic()
        seq_state = {}
        with torch.no_grad():
            for k in params:  # dict preserves the layer order from find_layers
                h = capture_inputs(model, {k: modules[k]}, fit)
                q, st, _ = codec(k, originals[k].cuda().float(), h.get(k), {})
                params[k].copy_(q.to(params[k].dtype))
                seq_state[k] = st.get("pairing")
        record("sequential", evaluate(model, blocks),
               {"seconds_quantize": time.monotonic() - t0, "pairings": seq_state})

    # scf_forward: our iterative loop, disjoint guard set
    if B.check("scf_forward", 300):
        restore(params, originals)
        t0 = time.monotonic()
        cfg = SCFConfig(alpha=0.5, max_iters=a.scf_iters)
        _, hist = run_scf(model, params, modules, originals, codec, fit, cfg,
                          guard_windows=guard)
        record("scf_forward", evaluate(model, blocks),
               {"seconds_quantize": time.monotonic() - t0, "scf": scf_summary(hist),
                "scf_history": hist.iterations})

    # ---------------- Test B ----------------
    if oneshot_weights is not None and B.check("testB", 150):
        restore(params, originals)
        S_fp = capture_sensitivity(model, modules, fit)
        H_fp = capture_inputs(model, modules, fit)
        install(params, oneshot_weights)
        S_q = capture_sensitivity(model, modules, fit)
        H_q = capture_inputs(model, modules, fit)

        sres = rel_residual(S_q, S_fp)
        hres = rel_residual(H_q, H_fp)
        ratio = sres["_pooled"] / max(hres["_pooled"], 1e-30)
        res["testB"] = {
            "backward_residual_per_layer": {k: v for k, v in sres.items() if k != "_pooled"},
            "forward_residual_per_layer": {k: v for k, v in hres.items() if k != "_pooled"},
            "backward_pooled": sres["_pooled"],
            "forward_pooled": hres["_pooled"],
            "ratio_backward_over_forward": ratio,
            "verdict": "PROCEED" if ratio > 2.0 else "ABANDON",
            "note": "predeclared: abandon SCF if backward residual <= 2x forward residual",
        }
        print(json.dumps({"testB": {k: v for k, v in res["testB"].items()
                                    if "per_layer" not in k}, "t": B.stamp()}), flush=True)

    for v in res["testA"].values():
        v.pop("block_nll", None)
    res["seconds"] = B.stamp()
    atomic_json(out / "results.json", res)

    # ---------------- report ----------------
    L = ["# SCF Phase 0: falsification gate", "",
         f"Qwen3.5-0.8B, {a.codec} codec, all {len(layers)} DeltaNet input projections, "
         f"{128 * len(blocks):,} WikiText-2 validation targets. Wall clock {res['seconds']}s.", "",
         "## Test A: is the one-pass sequential arm as good as the iterative loop?", "",
         f"BF16 NLL {bf['nll']:.6f}.", "",
         "| Arm | NLL | ΔNLL vs BF16 | 95% CI |", "|---|---:|---:|:--|"]
    for k in ("one_shot", "sequential", "scf_forward"):
        if k in res["testA"]:
            r = res["testA"][k]
            L.append(f"| {k} | {r['nll']:.6f} | {r['delta_nll']:+.6f} "
                     f"| [{r['delta_ci'][0]:+.6f}, {r['delta_ci'][1]:+.6f}] |")
    if "sequential" in res["testA"] and "scf_forward" in res["testA"]:
        gap = res["testA"]["scf_forward"]["delta_nll"] - res["testA"]["sequential"]["delta_nll"]
        L += ["", f"`scf_forward` minus `sequential`: **{gap:+.6f}** NLL "
                  f"(negative means the iteration still adds something)."]
    if res["testB"]:
        b = res["testB"]
        L += ["", "## Test B: is there a downstream residual to solve?", "",
              "| Quantity | Pooled relative residual |", "|---|---:|",
              f"| forward, E[x^2] | {b['forward_pooled']:.4%} |",
              f"| backward, E[(dL/dy)^2] | {b['backward_pooled']:.4%} |",
              f"| **ratio** | **{b['ratio_backward_over_forward']:.2f}x** |", "",
              f"Predeclared kill criterion: abandon if ratio <= 2. **Verdict: {b['verdict']}.**"]
    (out / "summary.md").write_text("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
