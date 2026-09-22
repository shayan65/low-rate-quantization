# Fused packed polar matmul

```json
{
  "model": "models/Qwen3.5-0.8B-Base",
  "layer": 0,
  "shape": [
    6144,
    1024
  ],
  "pairing": "reverse_half",
  "compressed_bytes": 3170304,
  "bf16_bytes": 12582912,
  "rows": [
    {
      "tokens": 1,
      "fused_packed_ms": 0.04491495059684236,
      "resident_bf16_ms": 0.02324492546217061,
      "slowdown": 1.9322475638796135,
      "max_abs_error_vs_bf16_gemm": 0.007405757904052734,
      "rmse_vs_bf16_gemm": 0.0010993288597092032
    },
    {
      "tokens": 16,
      "fused_packed_ms": 0.26990168563203315,
      "resident_bf16_ms": 0.023436675767426366,
      "slowdown": 11.516210247152797,
      "max_abs_error_vs_bf16_gemm": 0.021434783935546875,
      "rmse_vs_bf16_gemm": 0.0012551230611279607
    },
    {
      "tokens": 128,
      "fused_packed_ms": 2.209598513536675,
      "resident_bf16_ms": 0.04237233283412126,
      "slowdown": 52.14719997095244,
      "max_abs_error_vs_bf16_gemm": 0.020121097564697266,
      "rmse_vs_bf16_gemm": 0.001252192771062255
    }
  ],
  "note": "Packed magnitude/phase decoding is fused into matmul; no full weight matrix is materialized."
}
```
