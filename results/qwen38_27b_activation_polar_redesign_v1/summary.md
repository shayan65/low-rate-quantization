# Qwen3.8-27B activation-weighted polar redesign gate

One frozen layer-11 attention Q matrix was replaced at a time; every other weight remained BF16. Sixteen real WikiText-2 **training** windows provided input-activation second moments. A new 8,192-target validation slice (blocks 224–287) was untouched by earlier screens. The redesigned polar quantizer selected between two real-coordinate pairings and two global offsets of a 5-bit phase grid; for each row, it optimized an FP32 magnitude scale and selected each 3-bit magnitude by an activation-weighted projection onto its quantized phase ray. Candidate selection minimized diagonal activation-weighted coefficient error, not validation NLL. Each polar or real-4 payload counted 31,506,433 bytes; AWQ-style real-4 counted 31,526,913 bytes with its extra channel scales.

| Method | Held-out NLL | Difference vs redesigned polar |
| --- | ---: | ---: |
| BF16 | 2.650441 | -0.000452 |
| Optimized row real-4 | 2.651218 | +0.000325 |
| Legacy polar 3+5 | 2.650755 | -0.000138 |
| Activation-weighted polar 3+5 | 2.650893 | 0 |
| Local AWQ-style real-4 | 2.651304 | +0.000411 |

Training calibration chose adjacent pairing and a half-bin phase offset (0.098175 rad). Its weighted coefficient-error proxy improved from 1.9195e-6 at zero offset to 1.7782e-6 at half-bin offset for adjacent pairing. **That did not translate into better task NLL than legacy polar.** The redesigned method beat the local AWQ-style and plain real-4 controls on this one slice, but the paired gaps were small. An exploratory bootstrap over eight contiguous groups of eight windows gave 95% intervals for redesigned-minus-control NLL of [-0.000983,+0.001203] versus legacy polar and [-0.001196,+0.000381] versus AWQ-style. Both include zero. These intervals are descriptive and do not establish generalization.

The whole gate took 249.5 seconds; each 8,192-target forward pass took about 45 seconds. The RTX 3090 was idle after completion. This is a **decoded-BF16, one-tensor quality screen**, not a packed-runtime, cumulative quantization, or full-validation result. The current activation-weighted proxy is not a demonstrated improvement. A useful follow-up would test output reconstruction with actual captured activations or a task-calibrated objective, then evaluate on multiple layers and disjoint datasets before spending GPU hours on full passes.
