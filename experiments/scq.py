"""Self-consistent weight quantization (SCQ), by analogy with the DFT SCF loop.

Motivation
----------
Every activation-aware post-training quantizer (GPTQ, AWQ, and every
activation-weighted variant in this repository) calibrates against activations
measured on the **full-precision** model.  Once many layers are actually
replaced, the inputs those layers receive are no longer the inputs the codec
was fitted to.  The calibration statistic and the deployed model therefore
disagree: the procedure is not self-consistent.

This is structurally the same problem Kohn-Sham DFT solves with a self-
consistent field iteration, and it admits the same fix.

    Kohn-Sham DFT                         Self-consistent quantization
    ------------------------------------  ------------------------------------
    electron density  n(r)                activation statistics  h = E[x^2]
    effective potential  v_eff[n]         per-layer weighted quantizer metric
    solve KS equations -> n'              re-quantize, re-measure -> h'
    total energy  E[n]                    calibration NLL of the quantized net
    density mixing (linear / Pulay)       statistics mixing (linear / Anderson)
    SCF convergence                       codebook-assignment churn -> 0
    charge sloshing                       assignment oscillation between codes

Fixed point: ``Q* = Quantize(h[Q*])`` -- the code is optimal for the
activations that that very code produces.

Guarded iteration
-----------------
Unlike DFT, the discrete assignment step has no variational guarantee, so a
plain fixed-point iteration can oscillate or drift to a worse solution.  We
therefore treat the calibration NLL as the total energy and *guard* every step:
an update that raises it is rejected and the mixing parameter is halved.  The
loop consequently cannot end worse than the conventional one-shot calibration
it starts from, which also makes it a safe drop-in on top of any codec.
"""

from __future__ import annotations

import json
import math
import time
from dataclasses import dataclass, field
from typing import Callable

import torch
import torch.nn.functional as F


@dataclass
class SCFConfig:
    alpha: float = 0.5  # linear mixing weight
    max_iters: int = 8
    churn_tol: float = 1e-3  # stop when < 0.1% of indices move
    residual_tol: float = 1e-3
    anderson_m: int = 0  # 0 = plain linear mixing; >0 enables Anderson/Pulay
    min_alpha: float = 0.03
    guard: bool = True


@dataclass
class SCFHistory:
    iterations: list = field(default_factory=list)

    def add(self, **kw) -> None:
        self.iterations.append(kw)
        print(json.dumps({"scf": kw}), flush=True)


@torch.no_grad()
def capture_inputs(
    model,
    modules: dict,
    windows: list[list[int]],
    batch: int = 4,
    device: str = "cuda",
) -> dict[str, torch.Tensor]:
    """Run the calibration windows and return E[x^2] per input channel per layer.

    The second moment is accumulated in float64 on the fly so that no full
    activation tensor is retained; this is what makes an SCF iteration cost one
    forward pass rather than a stored activation cache.
    """
    acc: dict[str, torch.Tensor] = {}
    cnt: dict[str, int] = {}
    handles = []

    def mk(name):
        def hook(_mod, args):
            x = args[0].detach()
            flat = x.reshape(-1, x.shape[-1]).float()
            s = flat.square().sum(0).double()
            if name in acc:
                acc[name] += s
                cnt[name] += flat.shape[0]
            else:
                acc[name] = s
                cnt[name] = flat.shape[0]

        return hook

    for name, mod in modules.items():
        handles.append(mod.register_forward_pre_hook(mk(name)))
    try:
        for i in range(0, len(windows), batch):
            x = torch.tensor(windows[i : i + batch], device=device)
            model(x[:, :-1], use_cache=False)
            del x
    finally:
        for h in handles:
            h.remove()
    return {k: (acc[k] / max(cnt[k], 1)).float() for k in acc}


@torch.no_grad()
def calibration_nll(model, windows: list[list[int]], batch: int = 4, device: str = "cuda") -> float:
    total = 0.0
    n = 0
    for i in range(0, len(windows), batch):
        x = torch.tensor(windows[i : i + batch], device=device)
        logits = model(x[:, :-1], use_cache=False).logits.float()
        y = x[:, 1:]
        total += F.cross_entropy(
            logits.reshape(-1, logits.shape[-1]), y.reshape(-1), reduction="sum"
        ).item()
        n += y.numel()
        del x, y, logits
    return total / n


class AndersonMixer:
    """Anderson (Pulay/DIIS) acceleration on the flattened statistics vector."""

    def __init__(self, m: int = 3):
        self.m = m
        self.xs: list[torch.Tensor] = []
        self.fs: list[torch.Tensor] = []

    def step(self, x: torch.Tensor, fx: torch.Tensor, alpha: float) -> torch.Tensor:
        r = fx - x
        self.xs.append(x.clone())
        self.fs.append(r.clone())
        if len(self.xs) > self.m + 1:
            self.xs.pop(0)
            self.fs.pop(0)
        k = len(self.xs)
        if k < 2:
            return x + alpha * r
        # Least-squares combination of past residuals (the DIIS subproblem).
        dF = torch.stack([self.fs[i + 1] - self.fs[i] for i in range(k - 1)], 1)
        dX = torch.stack([self.xs[i + 1] - self.xs[i] for i in range(k - 1)], 1)
        try:
            gamma = torch.linalg.lstsq(dF, r.unsqueeze(1)).solution.squeeze(1)
        except Exception:
            return x + alpha * r
        if not torch.isfinite(gamma).all():
            return x + alpha * r
        return x + alpha * r - (dX + alpha * dF) @ gamma


@torch.no_grad()
def run_scf(
    model,
    params: dict[str, torch.nn.Parameter],
    modules: dict[str, torch.nn.Module],
    originals: dict[str, torch.Tensor],
    quantize: Callable[[str, torch.Tensor, torch.Tensor | None, dict], tuple[torch.Tensor, dict, torch.Tensor]],
    cal_windows: list[list[int]],
    cfg: SCFConfig = SCFConfig(),
    device: str = "cuda",
    guard_windows: list[list[int]] | None = None,
) -> tuple[dict, SCFHistory]:
    """Run the guarded self-consistent field loop over a set of layers.

    ``guard_windows`` supplies the data the energy guard is evaluated on.  It
    should be **disjoint** from ``cal_windows``: the statistics are estimated on
    ``cal_windows``, so guarding on the same text lets the loop accept steps
    that only fit that text's noise.  Measured behaviour with a shared, small
    calibration set is exactly that -- calibration energy falls while held-out
    loss rises.  Defaults to ``cal_windows`` (the unguarded-in-effect setting)
    only for backward compatibility.

    ``quantize(name, w_fp32, h_or_None, state)`` must return
    ``(quantized_weight_fp32, new_state, flat_index_tensor)``.  ``state`` is
    carried between iterations so codebooks can warm start -- the discrete
    analogue of reusing the previous density as the next SCF guess.

    Returns ``(final_state_per_layer, history)``.  The model's parameters are
    left holding the accepted self-consistent quantization.
    """
    hist = SCFHistory()
    names = list(params)
    guard_windows = cal_windows if guard_windows is None else guard_windows

    def install(qs: dict[str, torch.Tensor]) -> None:
        for n in names:
            params[n].copy_(qs[n].to(params[n].dtype))

    def restore_bf16() -> None:
        for n in names:
            params[n].copy_(originals[n].to(params[n].dtype))

    # --- Iteration 0: the conventional one-shot calibration ------------------
    restore_bf16()
    t0 = time.monotonic()
    h = capture_inputs(model, modules, cal_windows, device=device)
    state: dict[str, dict] = {n: {} for n in names}
    qs: dict[str, torch.Tensor] = {}
    idx: dict[str, torch.Tensor] = {}
    for n in names:
        q, st, ix = quantize(n, originals[n].to(device).float(), h.get(n), state[n])
        qs[n], state[n], idx[n] = q, st, ix
    install(qs)
    energy = calibration_nll(model, guard_windows, device=device)
    hist.add(
        iter=0,
        energy=energy,
        alpha=cfg.alpha,
        churn=None,
        residual=None,
        accepted=True,
        seconds=time.monotonic() - t0,
        note="one-shot: activations from the BF16 model",
    )

    flat = lambda d: torch.cat([d[n].flatten().float() for n in names])  # noqa: E731
    h_vec = flat(h)
    mixer = AndersonMixer(cfg.anderson_m) if cfg.anderson_m > 0 else None
    alpha = cfg.alpha
    sizes = [h[n].numel() for n in names]

    for it in range(1, cfg.max_iters + 1):
        t0 = time.monotonic()
        # Measure the density this code actually produces (model currently holds qs).
        h_out = capture_inputs(model, modules, cal_windows, device=device)
        out_vec = flat(h_out)
        residual = (out_vec - h_vec).norm().item() / max(h_vec.norm().item(), 1e-12)

        new_vec = (
            mixer.step(h_vec, out_vec, alpha)
            if mixer is not None
            else h_vec + alpha * (out_vec - h_vec)
        ).clamp_min(0)

        parts = torch.split(new_vec, sizes)
        h_new = {n: parts[i].clone() for i, n in enumerate(names)}

        trial_qs, trial_state, trial_idx = {}, {}, {}
        for n in names:
            q, st, ix = quantize(n, originals[n].to(device).float(), h_new[n], dict(state[n]))
            trial_qs[n], trial_state[n], trial_idx[n] = q, st, ix
        install(trial_qs)
        trial_energy = calibration_nll(model, guard_windows, device=device)

        moved = sum((trial_idx[n] != idx[n]).sum().item() for n in names)
        total = sum(idx[n].numel() for n in names)
        churn = moved / max(total, 1)

        accept = (not cfg.guard) or (trial_energy <= energy)
        if accept:
            qs, state, idx, h_vec, energy = trial_qs, trial_state, trial_idx, new_vec, trial_energy
        else:
            # Energy went up: reject, damp harder. This is the discrete analogue
            # of reducing the mixing parameter when an SCF cycle overshoots.
            install(qs)
            alpha = max(alpha * 0.5, cfg.min_alpha)
            if mixer is not None:
                mixer = AndersonMixer(cfg.anderson_m)  # drop a bad history

        hist.add(
            iter=it,
            energy=trial_energy,
            best_energy=energy,
            alpha=alpha,
            churn=churn,
            residual=residual,
            accepted=bool(accept),
            seconds=time.monotonic() - t0,
        )
        if accept and (churn < cfg.churn_tol or residual < cfg.residual_tol):
            break
        if alpha <= cfg.min_alpha and not accept:
            break

    install(qs)
    return state, hist


def scf_summary(hist: SCFHistory) -> dict:
    it = hist.iterations
    accepted = [r for r in it if r.get("accepted")]
    return {
        "iterations": len(it) - 1,
        "one_shot_energy": it[0]["energy"],
        "final_energy": accepted[-1]["energy"] if accepted else it[0]["energy"],
        "energy_gain": it[0]["energy"] - (accepted[-1]["energy"] if accepted else it[0]["energy"]),
        "final_churn": it[-1].get("churn"),
        "final_residual": it[-1].get("residual"),
        "rejected_steps": sum(1 for r in it[1:] if not r.get("accepted")),
    }
