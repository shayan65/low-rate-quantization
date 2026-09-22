"""Pure-stdlib checks and summaries for the historical reconstruction sweep.

The old rows describe FP32 reconstruction experiments, not packed codecs or
optimal rate-distortion bounds. Matching L**d entries only matches ideal index
entropy; scales, codebooks, padding and executed precision were not accounted.
"""

import math
from collections import Counter, defaultdict

LEVELS = (3, 4, 8, 16)
DIMS = (1, 2, 4, 8)
MAX_K = 8192
DEFAULT_TENSORS = (
    ("Qwen3.8-27B-metadata", "model.language_model.layers.0.linear_attn.in_proj_qkv.weight"),
    ("Qwen3.8-27B-metadata", "model.language_model.layers.3.self_attn.q_proj.weight"),
    ("Qwen3.8-27B-metadata", "model.language_model.layers.3.mlp.up_proj.weight"),
    ("Qwen3.5-0.8B-Base", "model.language_model.layers.0.linear_attn.in_proj_qkv.weight"),
    ("Qwen3.5-0.8B-Base", "model.language_model.layers.8.linear_attn.in_proj_qkv.weight"),
    ("Qwen3.5-0.8B-Base", "model.language_model.layers.16.linear_attn.in_proj_qkv.weight"),
)


def row_key(row):
    return (row["model"], row["tensor"], row["rotation"], row["levels"], row["dim"])


def validate_rows(rows, expected_tensors=None):
    """Require the complete declared design, with unique finite observations.

    The default manifest prevents an entirely missing model/tensor from silently
    shrinking the tested hypothesis. Custom runs must pass their requested list.
    """
    identities = list(DEFAULT_TENSORS if expected_tensors is None else expected_tensors)
    errors = []
    if not identities or len(set(identities)) != len(identities):
        errors.append("The expected tensor manifest is empty or contains duplicates.")
    expected = {
        (model, tensor, rotation, levels, dim)
        for model, tensor in identities
        for rotation in ("none", "hadamard")
        for levels in LEVELS for dim in DIMS if levels ** dim <= MAX_K
    }
    counts = Counter()
    index = {}
    for i, row in enumerate(rows):
        try:
            key = row_key(row)
            counts[key] += 1
            product = row["product_grid_mse"]
            free = row["free_vq_mse"]
            if not math.isfinite(product) or product <= 0 or not math.isfinite(free) or free < 0:
                raise ValueError("MSE values must be finite, product > 0 and VQ >= 0")
            if row.get("codebook") != row["levels"] ** row["dim"]:
                raise ValueError("Codebook cardinality does not equal levels ** dim")
            if "vq_over_product" in row and not math.isclose(
                    row["vq_over_product"], free / product, rel_tol=1e-6, abs_tol=1e-9):
                raise ValueError("Stored VQ/product ratio disagrees with MSE values")
            index[key] = row
        except (KeyError, TypeError, ValueError, OverflowError) as exc:
            errors.append(f"Row {i}: {exc}")
    duplicates = sorted(key for key, count in counts.items() if count != 1)
    missing = sorted(expected - set(index))
    unexpected = sorted(set(index) - expected)
    # Each scalar product comparator must be identical across dimensions.
    products = defaultdict(list)
    for key, row in index.items():
        products[key[:4]].append(row["product_grid_mse"])
    for key, values in products.items():
        if any(not math.isclose(values[0], value, rel_tol=1e-9, abs_tol=1e-15)
               for value in values[1:]):
            errors.append(f"Inconsistent scalar product MSE across dimensions: {key}")
    coverage = {
        "complete": not (errors or duplicates or missing or unexpected),
        "expected_tensor_count": len(identities), "expected_row_count": len(expected),
        "actual_row_count": len(rows), "duplicates": duplicates, "missing": missing,
        "unexpected": unexpected, "errors": errors,
    }
    return index, identities, coverage


def evaluate_gates(rows, expected_tensors=None):
    """Recompute the historical decision rule without mixing models.

    H1 compares the best observed allowed dimension at each rate, so it is a
    resource-constrained configuration gate, not a fixed-dimension test of a
    general rate-dependence mechanism. Incomplete/duplicate data never passes.
    """
    index, identities, coverage = validate_rows(rows, expected_tensors)
    out = {"coverage": coverage, "H0_rotation_closes": [], "H1_margins": [],
           "H0_subsumed": False, "H1_passes": False,
           "comparison": "best observed allowed dimension per ideal index rate"}
    if not coverage["complete"]:
        out["verdict"] = "STOP: invalid or incomplete declared sweep; no gate decision"
        return out
    for model, tensor in sorted(identities):
        unrot = index[(model, tensor, "none", 3, 2)]
        rotated = index[(model, tensor, "hadamard", 3, 2)]
        gap = unrot["product_grid_mse"] - unrot["free_vq_mse"]
        closed = ((unrot["product_grid_mse"] - rotated["product_grid_mse"]) / gap
                  if gap > 0 else None)
        out["H0_rotation_closes"].append({
            "model": model, "tensor": tensor, "fraction_closed_by_rotation": closed,
            "positive_unrotated_vq_gap": gap > 0,
        })
        best = {}
        for levels in (3, 16):
            candidates = [index[(model, tensor, "hadamard", levels, dim)]
                          for dim in DIMS if dim > 1 and levels ** dim <= MAX_K]
            selected = min(candidates, key=lambda r: r["free_vq_mse"] / r["product_grid_mse"])
            best[levels] = (1 - selected["free_vq_mse"] / selected["product_grid_mse"],
                            selected["dim"])
        m3, m16 = best[3][0], best[16][0]
        out["H1_margins"].append({
            "model": model, "tensor": tensor, "margin_ternary": m3, "margin_4bit": m16,
            "dim_ternary": best[3][1], "dim_4bit": best[16][1],
            "ratio": m3 / m16 if m16 > 0 else None,
            "passes": m3 >= 0.15 and (m16 <= 0 or m3 >= 2 * m16),
        })
    out["H0_subsumed"] = all(
        x["fraction_closed_by_rotation"] is not None and x["fraction_closed_by_rotation"] >= 0.80
        for x in out["H0_rotation_closes"])
    out["H1_passes"] = all(x["passes"] for x in out["H1_margins"])
    undefined_gap = any(not x["positive_unrotated_vq_gap"] for x in out["H0_rotation_closes"])
    if undefined_gap:
        out["H1_passes"] = False
        out["verdict"] = "STOP: H0 has no positive unrotated VQ gap; comparison is undefined"
    elif out["H0_subsumed"]:
        out["verdict"] = "STOP: the tested configurations meet the H0 rotation-closure threshold"
    elif out["H1_passes"]:
        out["verdict"] = "PROCEED to a separately specified feasibility test; historical configuration gate passed"
    else:
        out["verdict"] = "STOP: the tested configurations failed the predeclared H1 gate"
    return out


def aggregate_rows(rows, expected_tensors=None):
    """Equal-tensor means, separately by model and pooled; never weight by size."""
    index, _, coverage = validate_rows(rows, expected_tensors)
    if not coverage["complete"]:
        return []
    grouped = defaultdict(list)
    for row in index.values():
        for scope in (row["model"], "ALL_MODELS"):
            grouped[(scope, row["rotation"], row["levels"], row["dim"])].append(row)
    return [{"model": key[0], "rotation": key[1], "levels": key[2], "dim": key[3],
             "tensor_count": len(items),
             "mean_vq_over_product": sum(r["free_vq_mse"] / r["product_grid_mse"] for r in items) / len(items),
             "mean_mse_reduction": sum(1 - r["free_vq_mse"] / r["product_grid_mse"] for r in items) / len(items)}
            for key, items in sorted(grouped.items())]


def render(rows, verdict, aggregates=None):
    identities = sorted({(r["model"], r["tensor"]) for r in rows})
    lines = ["# Corrected historical rate sweep", "",
             "These are measured weight-reconstruction errors, not language-model quality results.",
             "Ratios compare observed VQ MSE with the scalar product baseline at equal ideal",
             "index entropy. Historical FP32 scales/codebooks, packing, and metadata overhead",
             "were not measured as a deployable format. These fits are not optimality bounds.", ""]
    if not verdict["coverage"]["complete"]:
        lines += ["**Invalid/incomplete coverage: no gate may pass.**", "",
                  "```json", __import__("json").dumps(verdict["coverage"], indent=2), "```", ""]
    for model, tensor in identities:
        short = tensor.split("layers.")[-1].replace(".weight", "")
        lines += [f"## {model} / {short}", "",
                  "| Rotation | Ideal index bits/weight | Dim 2 | Dim 4 | Dim 8 |",
                  "|---|---:|---:|---:|---:|"]
        for rotation in ("none", "hadamard"):
            for levels in LEVELS:
                cells = []
                for dim in (2, 4, 8):
                    matches = [r for r in rows if row_key(r) == (model, tensor, rotation, levels, dim)]
                    cells.append(f"{matches[0]['free_vq_mse'] / matches[0]['product_grid_mse']:.6f}"
                                 if len(matches) == 1 else ("DUPLICATE" if matches else "—"))
                lines.append(f"| {rotation} | {math.log2(levels):.6f} | " + " | ".join(cells) + " |")
        lines.append("")
    if aggregates:
        lines += ["## Equal-tensor post-rotation mean MSE reductions", "",
                  "Models are separated below; ALL_MODELS gives each of the six tensors equal weight.", "",
                  "| Model | Levels | Dim | Tensors | MSE reduction |", "|---|---:|---:|---:|---:|"]
        for row in aggregates:
            if row["rotation"] == "hadamard" and row["dim"] > 1:
                lines.append(f"| {row['model']} | {row['levels']} | {row['dim']} | {row['tensor_count']} | {100 * row['mean_mse_reduction']:.4f}% |")
        lines.append("")
    lines += ["## Historical decision gate", "",
              "The H1 rule selects the best observed permitted dimension at each rate. It is not",
              "a fixed-dimension causal test and cannot falsify a general coding mechanism.", "",
              f"- Coverage complete: `{verdict['coverage']['complete']}`",
              f"- H0 rotation-closure threshold met on every tensor: `{verdict['H0_subsumed']}`",
              f"- H1 margin ≥15% and ≥2× four-bit margin on every tensor: `{verdict['H1_passes']}`", "",
              f"**{verdict['verdict']}**", ""]
    return "\n".join(lines)
