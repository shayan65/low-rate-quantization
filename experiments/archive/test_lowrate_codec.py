"""Small CPU tests. Torch-only fits are skipped when torch is unavailable."""
import importlib.util
import json
import math
from pathlib import Path
import tempfile
import unittest

import numpy as np

from lowrate_codec import (ArtifactWriter, CodecConfig, EncodedTensor, artifact_storage,
                          decode_numpy, decode_tensor, encode_tensor, fit_codebook,
                          pack_indices, pack_trits, prepare, read_encoded,
                          unpack_indices, unpack_trits)


class PackingTests(unittest.TestCase):
    def test_trit_roundtrip_group_padding_and_real_rate(self):
        rng = np.random.default_rng(9)
        for group in (64, 128):
            x = rng.integers(0, 3, 7 * group, dtype=np.uint8)
            payload = pack_trits(x, group)
            self.assertEqual(len(payload), 7 * math.ceil(group / 5))
            np.testing.assert_array_equal(x, unpack_trits(payload, len(x), group))
        self.assertEqual((len(pack_trits(np.zeros(128, dtype=np.uint8))) + 2) * 8 / 128, 1.75)

    def test_fixed_width_roundtrip_including_chunk_boundary(self):
        rng = np.random.default_rng(4)
        for n in (1, 7, 8, 32767, 32768, 32769):
            x = rng.integers(0, 6561, n, dtype=np.uint16)
            payload = pack_indices(x, 13)
            self.assertEqual(len(payload), math.ceil(13 * n / 8))
            np.testing.assert_array_equal(x, unpack_indices(payload, n, 13, 6561))

    def test_reject_malformed_payloads(self):
        with self.assertRaises(ValueError):
            unpack_indices(bytes([0, 0x80]), 1, 13)
        with self.assertRaises(ValueError):
            unpack_indices(pack_indices([7000], 13), 1, 13, 6561)
        with self.assertRaises(ValueError):
            unpack_trits(bytes([243]), 5, 5)
        payload = bytearray(pack_trits(np.zeros(128, dtype=np.uint8)))
        payload[-1] = 27  # Only three real trits in the final byte.
        with self.assertRaises(ValueError):
            unpack_trits(payload, 128)
        with self.assertRaises(ValueError):
            pack_trits([0, 1, 3], 3)


class ArtifactTests(unittest.TestCase):
    def test_literal_ternary_decode_and_all_bytes_accounted(self):
        with tempfile.TemporaryDirectory() as tmp:
            cfg = CodecConfig("ternary", rotate=False)
            indices = np.arange(256, dtype=np.uint16) % 3
            e = EncodedTensor((2, 128), np.array([0.5, 2], dtype=np.float16), indices)
            writer = ArtifactWriter(tmp, cfg)
            writer.add("weight", e)
            info = writer.finalize()
            expected = (indices.astype(np.float32) - 1).reshape(2, 128) * np.array([[0.5], [2]])
            np.testing.assert_array_equal(decode_numpy(tmp, "weight"), expected)
            self.assertEqual(info["total_bytes"], sum(p.stat().st_size for p in Path(tmp).iterdir()))
            self.assertEqual(info["component_bytes"]["indices"], 52)
            self.assertEqual(info["component_bytes"]["scales"], 4)
            self.assertGreater(info["bits_per_weight"], 1.75)  # Metadata is charged.

    def test_shared_dim8_codebook_stored_once_and_13bit_indices(self):
        with tempfile.TemporaryDirectory() as tmp:
            cfg = CodecConfig("vq", rotate=False)
            book = np.arange(6561 * 8, dtype=np.float32).reshape(6561, 8) / 65536
            book = book.astype(np.float16)
            writer = ArtifactWriter(tmp, cfg, book)
            index = np.arange(16, dtype=np.uint16) * 401
            for name in ("a", "b"):
                writer.add(name, EncodedTensor((1, 128), np.array([2], dtype=np.float16), index, codebook=book))
            info = writer.finalize()
            self.assertEqual(info["component_bytes"]["codebooks"], 6561 * 8 * 2)
            self.assertEqual(info["component_bytes"]["indices"], 26 * 2)
            np.testing.assert_array_equal(decode_numpy(tmp, "b"), book[index].astype(np.float32).reshape(1, 128) * 2)

    def test_per_tensor_scalar_books_and_zero_weights(self):
        with tempfile.TemporaryDirectory() as tmp:
            writer = ArtifactWriter(tmp, CodecConfig("scalar", rotate=False))
            for i, book in enumerate(([-0.6, 0, 0.8], [-1, 0, 0.3])):
                writer.add(str(i), EncodedTensor((1, 128), np.ones(1, dtype=np.float16), np.ones(128, dtype=np.uint16)),
                           codebook=np.array(book, dtype=np.float16).reshape(3, 1))
            info = writer.finalize()
            self.assertEqual(info["component_bytes"]["codebooks"], 12)
            np.testing.assert_array_equal(decode_numpy(tmp, "0"), np.zeros((1, 128)))

    def test_stored_signs_and_inverse_hadamard(self):
        with tempfile.TemporaryDirectory() as tmp:
            cfg = CodecConfig("ternary", group_size=8, block_size=8)
            signs = np.array([0, 1, 0, 0, 1, 0, 1, 1], dtype=np.uint8)
            index = np.array([0, 1, 2, 0, 1, 1, 2, 2], dtype=np.uint8)
            writer = ArtifactWriter(tmp, cfg)
            writer.add("x", EncodedTensor((1, 8), np.ones(1, dtype=np.float16), index, signs))
            writer.finalize()
            hadamard = np.array([[1 if (i & j).bit_count() % 2 == 0 else -1 for j in range(8)] for i in range(8)]) / math.sqrt(8)
            expected = ((index.astype(float) - 1) @ hadamard) * (signs.astype(float) * 2 - 1)
            np.testing.assert_allclose(decode_numpy(tmp, "x")[0], expected, atol=2e-7)

    def test_checksums_extra_files_and_no_path_traversal(self):
        with tempfile.TemporaryDirectory() as tmp:
            writer = ArtifactWriter(tmp, CodecConfig("ternary", rotate=False))
            writer.add("x", EncodedTensor((1, 128), np.ones(1, dtype=np.float16), np.ones(128, dtype=np.uint16)))
            writer.finalize()
            extra = Path(tmp) / "ignored.bin"
            extra.write_bytes(b"extra")
            with self.assertRaises(ValueError):
                artifact_storage(tmp)
            extra.unlink()
            manifest_path = Path(tmp) / "manifest.json"
            m = json.loads(manifest_path.read_text())
            m["tensors"]["x"]["indices"]["file"] = "../outside.bin"
            manifest_path.write_text(json.dumps(m))
            with self.assertRaises(ValueError):
                read_encoded(tmp, "x")
        with tempfile.TemporaryDirectory() as tmp:
            writer = ArtifactWriter(tmp, CodecConfig("ternary", rotate=False))
            writer.add("x", EncodedTensor((1, 128), np.ones(1, dtype=np.float16), np.ones(128, dtype=np.uint16)))
            writer.finalize()
            payload = Path(tmp) / "tensor-0000.indices.bin"
            payload.write_bytes(b"x" * payload.stat().st_size)
            with self.assertRaises(ValueError):
                decode_numpy(tmp, "x")


@unittest.skipUnless(importlib.util.find_spec("torch"), "torch unavailable")
class TorchFitTests(unittest.TestCase):
    def test_roundtrip_uses_rounded_metadata_and_original_basis(self):
        import torch
        torch.manual_seed(4)
        w = torch.randn(3, 128) * 0.12
        for kind in ("ternary", "scalar", "vq"):
            cfg = CodecConfig(kind, dim=2, block_size=128, max_points=256, iterations=3, memory_mb=0.01)
            p = prepare(w, cfg)
            book = fit_codebook([p], cfg)
            encoded = encode_tensor(p, cfg, book)
            with tempfile.TemporaryDirectory() as tmp:
                writer = ArtifactWriter(tmp, cfg, book)
                writer.add("x", encoded)
                writer.finalize()
                actual = decode_tensor(tmp, "x", dtype=torch.float32).numpy()
                np.testing.assert_allclose(actual, decode_numpy(tmp, "x"), atol=2e-7)
                self.assertTrue(np.isfinite(actual).all())
                self.assertLess(float(np.mean((actual - w.numpy()) ** 2)), float(w.square().mean()))
            self.assertEqual(p.scales.dtype, torch.float16)
            if book is not None:
                self.assertEqual(book.dtype, torch.float16)

    def test_zero_tiny_weights_and_budget_propagation(self):
        import torch
        for kind in ("ternary", "scalar", "vq"):
            cfg = CodecConfig(kind, dim=2, rotate=False, max_points=128, iterations=2)
            w = torch.zeros(2, 128)
            w[1, :] = 1e-10
            p = prepare(w, cfg)
            self.assertTrue(torch.isfinite(p.u).all())
            self.assertTrue((p.scales > 0).all())
            book = fit_codebook([p], cfg)
            e = encode_tensor(p, cfg, book)
            self.assertTrue(np.isfinite(e.scales).all())
        def stop():
            raise TimeoutError("budget exhausted")
        with self.assertRaises(TimeoutError):
            fit_codebook([p], cfg, stop)

    def test_weighted_scalar_fit_and_product_fallback(self):
        import torch
        from lowrate_codec import PreparedTensor, _sample, _scalar_fit
        # Strongly weighted groups must affect the fit; low-scale groups should
        # not dominate just because normalized samples are equally numerous.
        u = torch.tensor([[-1., 0., 0.2, 1.], [-1., -0.2, 0., 1.]])
        p = PreparedTensor(u, torch.tensor([100., 0.01]).half(), None, (2, 4))
        cfg = CodecConfig("vq", group_size=4, dim=2, rotate=False, max_points=4, iterations=3)
        points, weight = _sample([p], cfg)
        levels = _scalar_fit(points, weight, cfg)
        product = torch.cartesian_prod(levels.float(), levels.float())
        fitted = fit_codebook([p], cfg).float()
        def loss(book):
            distance = (points[:, None, :] - book[None, :, :]).square().sum(2)
            return (distance.min(1).values * weight).sum()
        self.assertLessEqual(loss(fitted).item(), loss(product).item() + 1e-4)


if __name__ == "__main__":
    unittest.main()
