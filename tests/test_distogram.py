import gc
import pickle
import numpy as np
import pytest
from pathlib import Path
from unittest import mock
from unittest.mock import MagicMock, patch, call
from colabfold.batch import (
    process_distogram_chunk,
    generate_intermediate_distograms_from_representations,
)


def _make_repr_file(tmp_path: Path, name: str, loop_type: str = "evoformer") -> Path:
    """Write a minimal pickled representations dict and return its Path."""
    data = {
        "pair": np.random.rand(64, 64, 128).astype(np.float32),
        "loop_type": loop_type,
    }
    p = tmp_path / name
    p.write_bytes(pickle.dumps(data))
    return p


def _fake_distogram_output():
    """Return a dict that mimics the DistogramHead __call__ output."""
    return {
        "logits": np.random.rand(64, 64, 64).astype(np.float32),
        "bin_edges": np.linspace(2.0, 22.0, 63).astype(np.float32),
    }


@pytest.fixture()
def mock_jax(monkeypatch):
    """Patch jax so no GPU is required and jit is a no-op."""
    jax_mod = MagicMock()
    jax_mod.devices.side_effect = lambda kind: (
        [MagicMock()]
        if kind == "cpu"
        else (_ for _ in ()).throw(RuntimeError("no GPU"))
    )
    jax_mod.device_put.side_effect = lambda arr, dev: arr  # passthrough
    jax_mod.random.PRNGKey.return_value = MagicMock()
    jax_mod.jit.side_effect = lambda fn: fn  # identity wrapper
    jax_mod.clear_caches = MagicMock()
    jax_mod.numpy = np  # use real numpy
    monkeypatch.setitem(__import__("sys").modules, "jax", jax_mod)
    monkeypatch.setitem(__import__("sys").modules, "jax.numpy", np)
    return jax_mod


@pytest.fixture()
def mock_hk(monkeypatch):
    """Patch haiku so hk.transform returns a fake transformed object."""
    hk_mod = MagicMock()

    def fake_transform(fn):
        transformed = MagicMock()
        # .apply(params, rng, *args) returns fake distogram output
        transformed.apply.side_effect = (
            lambda params, rng, *args, **kwargs: _fake_distogram_output()
        )
        return transformed

    hk_mod.transform.side_effect = fake_transform
    monkeypatch.setitem(__import__("sys").modules, "haiku", hk_mod)
    monkeypatch.setitem(__import__("sys").modules, "hk", hk_mod)
    return hk_mod


@pytest.fixture()
def mock_alphafold_modules(monkeypatch):
    """Patch alphafold.model.modules with a fake DistogramHead."""
    modules_mod = MagicMock()
    modules_mod.DistogramHead.return_value = MagicMock(
        __call__=lambda self, reps, batch, is_training: _fake_distogram_output()
    )
    monkeypatch.setitem(__import__("sys").modules, "alphafold", MagicMock())
    monkeypatch.setitem(__import__("sys").modules, "alphafold.model", MagicMock())
    monkeypatch.setitem(
        __import__("sys").modules, "alphafold.model.modules", modules_mod
    )
    return modules_mod


@pytest.fixture()
def fake_config():
    """Minimal config object that mirrors model_runner.config.model."""
    cfg = MagicMock()
    cfg.heads.distogram = MagicMock()
    cfg.global_config = MagicMock()
    return cfg


@pytest.fixture()
def fake_params():
    """Tiny fake trained-params dict (values are numpy scalars for speed)."""
    return {f"key_{i}": np.float32(i) for i in range(4)}


class TestProcessDistogramChunk:
    """Unit tests for the worker function."""

    def test_writes_npz_for_evoformer_files(
        self,
        tmp_path,
        mock_jax,
        mock_hk,
        mock_alphafold_modules,
        fake_config,
        fake_params,
    ):
        """One evoformer file → one .npz in the output dir."""
        out_dir = tmp_path / "distograms"
        out_dir.mkdir()
        f = _make_repr_file(
            tmp_path, "cycle_000_representations.pkl", loop_type="evoformer"
        )

        results = process_distogram_chunk([f], fake_params, fake_config, str(out_dir))

        npz_files = list(out_dir.glob("*_distogram.npz"))
        assert len(npz_files) == 1, "Expected exactly one .npz output file"
        assert results["success"] == 1
        assert results["error"] == 0
        assert results["skipped"] == 0

    def test_npz_contains_logits_and_bin_edges(
        self,
        tmp_path,
        mock_jax,
        mock_hk,
        mock_alphafold_modules,
        fake_config,
        fake_params,
    ):
        """Output .npz must have both 'logits' and 'bin_edges' arrays."""
        out_dir = tmp_path / "distograms"
        out_dir.mkdir()
        f = _make_repr_file(tmp_path, "cycle_001_representations.pkl")

        process_distogram_chunk([f], fake_params, fake_config, str(out_dir))

        npz_path = out_dir / "cycle_001_distogram.npz"
        assert npz_path.exists()
        data = np.load(npz_path)
        assert "logits" in data, "Missing 'logits' array in .npz"
        assert "bin_edges" in data, "Missing 'bin_edges' array in .npz"

    def test_multiple_evoformer_files_all_processed(
        self,
        tmp_path,
        mock_jax,
        mock_hk,
        mock_alphafold_modules,
        fake_config,
        fake_params,
    ):
        """All evoformer files in a chunk are processed in a single call."""
        out_dir = tmp_path / "out"
        out_dir.mkdir()
        files = [
            _make_repr_file(tmp_path, f"cycle_{i:03d}_representations.pkl")
            for i in range(5)
        ]

        results = process_distogram_chunk(files, fake_params, fake_config, str(out_dir))

        assert results["success"] == 5
        assert len(list(out_dir.glob("*_distogram.npz"))) == 5

    def test_extra_msa_files_are_skipped(
        self,
        tmp_path,
        mock_jax,
        mock_hk,
        mock_alphafold_modules,
        fake_config,
        fake_params,
    ):
        """Files with loop_type='extra_msa' must not produce output."""
        out_dir = tmp_path / "out"
        out_dir.mkdir()
        msa_file = _make_repr_file(
            tmp_path, "extra_000_representations.pkl", loop_type="extra_msa"
        )

        results = process_distogram_chunk(
            [msa_file], fake_params, fake_config, str(out_dir)
        )

        assert results["skipped"] == 1
        assert results["success"] == 0
        assert len(list(out_dir.glob("*_distogram.npz"))) == 0

    def test_mixed_loop_types_only_evoformer_written(
        self,
        tmp_path,
        mock_jax,
        mock_hk,
        mock_alphafold_modules,
        fake_config,
        fake_params,
    ):
        """A chunk with mixed loop types: only evoformer ones produce output."""
        out_dir = tmp_path / "out"
        out_dir.mkdir()
        evo = _make_repr_file(
            tmp_path, "evo_representations.pkl", loop_type="evoformer"
        )
        msa = _make_repr_file(
            tmp_path, "msa_representations.pkl", loop_type="extra_msa"
        )

        results = process_distogram_chunk(
            [evo, msa], fake_params, fake_config, str(out_dir)
        )

        assert results["success"] == 1
        assert results["skipped"] == 1
        assert len(list(out_dir.glob("*_distogram.npz"))) == 1

    def test_unknown_loop_type_is_processed(
        self,
        tmp_path,
        mock_jax,
        mock_hk,
        mock_alphafold_modules,
        fake_config,
        fake_params,
    ):
        """Files with loop_type='unknown' (default) should not be skipped."""
        out_dir = tmp_path / "out"
        out_dir.mkdir()
        f = _make_repr_file(tmp_path, "unk_representations.pkl", loop_type="unknown")

        results = process_distogram_chunk([f], fake_params, fake_config, str(out_dir))

        assert results["success"] == 1

    def test_output_filename_replaces_representations_with_distogram(
        self,
        tmp_path,
        mock_jax,
        mock_hk,
        mock_alphafold_modules,
        fake_config,
        fake_params,
    ):
        """*_representations.pkl → *_distogram.npz (exact name check)."""
        out_dir = tmp_path / "out"
        out_dir.mkdir()
        f = _make_repr_file(tmp_path, "layer_007_representations.pkl")

        process_distogram_chunk([f], fake_params, fake_config, str(out_dir))

        assert (out_dir / "layer_007_distogram.npz").exists()

    def test_corrupt_pickle_increments_error_count(
        self,
        tmp_path,
        mock_jax,
        mock_hk,
        mock_alphafold_modules,
        fake_config,
        fake_params,
    ):
        """A corrupt pickle must not crash the worker; error count increments."""
        out_dir = tmp_path / "out"
        out_dir.mkdir()
        bad_file = tmp_path / "bad_representations.pkl"
        bad_file.write_bytes(b"not a pickle")

        results = process_distogram_chunk(
            [bad_file], fake_params, fake_config, str(out_dir)
        )

        assert results["error"] == 1
        assert results["success"] == 0

    def test_error_in_one_file_does_not_abort_chunk(
        self,
        tmp_path,
        mock_jax,
        mock_hk,
        mock_alphafold_modules,
        fake_config,
        fake_params,
    ):
        """A failure in one file must not prevent processing of subsequent files."""
        out_dir = tmp_path / "out"
        out_dir.mkdir()
        bad = tmp_path / "bad_representations.pkl"
        bad.write_bytes(b"garbage")
        good = _make_repr_file(tmp_path, "good_representations.pkl")

        results = process_distogram_chunk(
            [bad, good], fake_params, fake_config, str(out_dir)
        )

        assert results["error"] == 1
        assert results["success"] == 1

    def test_falls_back_to_cpu_when_no_gpu(
        self,
        tmp_path,
        monkeypatch,
        mock_hk,
        mock_alphafold_modules,
        fake_config,
        fake_params,
    ):
        """When jax.devices('gpu') raises, the worker falls back to CPU silently."""
        jax_cpu = MagicMock()
        cpu_dev = MagicMock()
        jax_cpu.devices.side_effect = lambda kind: (
            [cpu_dev]
            if kind == "cpu"
            else (_ for _ in ()).throw(RuntimeError("no GPU"))
        )
        jax_cpu.device_put.side_effect = lambda arr, dev: arr
        jax_cpu.random.PRNGKey.return_value = MagicMock()
        jax_cpu.jit.side_effect = lambda fn: fn
        jax_cpu.clear_caches = MagicMock()
        jax_cpu.numpy = np
        monkeypatch.setitem(__import__("sys").modules, "jax", jax_cpu)

        out_dir = tmp_path / "out"
        out_dir.mkdir()
        f = _make_repr_file(tmp_path, "c_representations.pkl")

        results = process_distogram_chunk([f], fake_params, fake_config, str(out_dir))

        assert results["success"] == 1

    def test_jax_clear_caches_called_after_chunk(
        self,
        tmp_path,
        mock_jax,
        mock_hk,
        mock_alphafold_modules,
        fake_config,
        fake_params,
    ):
        """jax.clear_caches() must be called once per worker invocation."""
        out_dir = tmp_path / "out"
        out_dir.mkdir()
        f = _make_repr_file(tmp_path, "cache_representations.pkl")

        process_distogram_chunk([f], fake_params, fake_config, str(out_dir))

        mock_jax.clear_caches.assert_called_once()

    def test_empty_chunk_returns_zero_counts(
        self,
        tmp_path,
        mock_jax,
        mock_hk,
        mock_alphafold_modules,
        fake_config,
        fake_params,
    ):
        """An empty file list must return all-zero result counts."""
        out_dir = tmp_path / "out"
        out_dir.mkdir()

        results = process_distogram_chunk([], fake_params, fake_config, str(out_dir))

        assert results == {"success": 0, "skipped": 0, "error": 0}


class TestGenerateIntermediateDistograms:

    def test_returns_early_when_staging_dir_is_none(self):
        with patch("alphafold.model.modules.intermediate_structures_dir", None):
            result = generate_intermediate_distograms_from_representations(
                "job", [("m", MagicMock(), {})], "/tmp/out"
            )
            assert result is None

    def test_returns_early_when_staging_path_missing(self, tmp_path):
        non_existent = str(tmp_path / "does_not_exist")
        with patch("alphafold.model.modules.intermediate_structures_dir", non_existent):
            result = generate_intermediate_distograms_from_representations(
                "job", [("m", MagicMock(), {})], str(tmp_path / "out")
            )
            assert result is None

    def test_returns_early_when_no_repr_files(self, tmp_path):
        staging = tmp_path / "staging"
        staging.mkdir()
        with patch("alphafold.model.modules.intermediate_structures_dir", str(staging)):
            result = generate_intermediate_distograms_from_representations(
                "job", [("m", MagicMock(), {})], str(tmp_path / "out")
            )
            assert result is None

    def _make_model_runner_and_params(self, params=None):
        fake_runner = MagicMock()
        fake_runner.config.model = MagicMock()
        return [("model1", fake_runner, params or {})]

    def _mock_pool(self, mock_pool_cls, submit_side_effect=None):
        mock_pool = MagicMock()
        mock_pool.__enter__ = MagicMock(return_value=mock_pool)
        mock_pool.__exit__ = MagicMock(return_value=False)
        if submit_side_effect:
            mock_pool.submit.side_effect = submit_side_effect
        mock_pool_cls.return_value = mock_pool
        return mock_pool

    def test_output_dir_created(self, tmp_path):
        staging = tmp_path / "staging"
        staging.mkdir()
        _make_repr_file(staging, "c_000_representations.pkl")
        out_dir = tmp_path / "deep" / "nested" / "out"
        assert not out_dir.exists()

        with (
            patch("alphafold.model.modules.intermediate_structures_dir", str(staging)),
            patch("concurrent.futures.ProcessPoolExecutor") as mock_pool_cls,
            patch("time.sleep"),
        ):
            # pre-populate so progress loop exits immediately
            out_dir.mkdir(parents=True)
            (out_dir / "c_000_distogram.npz").write_bytes(b"")
            self._mock_pool(mock_pool_cls)

            generate_intermediate_distograms_from_representations(
                "job", self._make_model_runner_and_params(), str(out_dir)
            )

        assert out_dir.exists()

    def test_submits_tasks_for_each_chunk(self, tmp_path):
        staging = tmp_path / "staging"
        staging.mkdir()
        for i in range(7):
            _make_repr_file(staging, f"c_{i:03d}_representations.pkl")

        submit_calls = []

        with (
            patch("alphafold.model.modules.intermediate_structures_dir", str(staging)),
            patch("concurrent.futures.ProcessPoolExecutor") as mock_pool_cls,
            patch("time.sleep"),
        ):
            out_dir = tmp_path / "out"
            out_dir.mkdir()
            # pre-populate so progress loop exits
            for i in range(7):
                (out_dir / f"c_{i:03d}_distogram.npz").write_bytes(b"")

            self._mock_pool(
                mock_pool_cls,
                submit_side_effect=lambda fn, *a, **kw: submit_calls.append(a),
            )

            generate_intermediate_distograms_from_representations(
                "job", self._make_model_runner_and_params(), str(out_dir)
            )

        # 7 files, max_workers=4 → ceil(7/4)=2 chunks
        assert 1 <= len(submit_calls) <= 4

    def test_params_are_reformatted_before_dispatch(self, tmp_path):
        """Only distogram_head params are extracted; prefix is stripped."""
        staging = tmp_path / "staging"
        staging.mkdir()
        _make_repr_file(staging, "c_000_representations.pkl")

        prefix = "alphafold/alphafold_iteration/distogram_head/"
        raw_params = {
            f"{prefix}weights": np.float32(1.0),
            f"{prefix}bias": np.float32(0.5),
            "alphafold/alphafold_iteration/structure_module/weight": np.float32(9.0),
        }
        captured_params = []

        with (
            patch("alphafold.model.modules.intermediate_structures_dir", str(staging)),
            patch("concurrent.futures.ProcessPoolExecutor") as mock_pool_cls,
            patch("time.sleep"),
        ):
            out_dir = tmp_path / "out"
            out_dir.mkdir()
            (out_dir / "c_000_distogram.npz").write_bytes(b"")

            self._mock_pool(
                mock_pool_cls,
                submit_side_effect=lambda fn, chunk, params, *a, **kw: captured_params.append(
                    params
                ),
            )

            generate_intermediate_distograms_from_representations(
                "job", self._make_model_runner_and_params(raw_params), str(out_dir)
            )

        assert captured_params, "submit was never called"
        params = captured_params[0]
        # Non-distogram keys must be excluded
        assert all("structure_module" not in k for k in params)
        # Prefix must be stripped
        assert all("alphafold/alphafold_iteration/" not in k for k in params)
        # Distogram keys must be present with stripped prefix
        assert "distogram_head/weights" in params
        assert "distogram_head/bias" in params

    def test_total_expected_excludes_extra_msa_files(self, tmp_path):
        """Progress bar total should only count non-extra_msa files."""
        staging = tmp_path / "staging"
        staging.mkdir()
        _make_repr_file(staging, "c_000_representations.pkl")
        _make_repr_file(
            staging, "extra_msa_000_representations.pkl", loop_type="extra_msa"
        )

        out_dir = tmp_path / "out"
        out_dir.mkdir()
        # Only pre-populate the evoformer output — if total_expected were 2 this would hang
        (out_dir / "c_000_distogram.npz").write_bytes(b"")

        with (
            patch("alphafold.model.modules.intermediate_structures_dir", str(staging)),
            patch("concurrent.futures.ProcessPoolExecutor") as mock_pool_cls,
            patch("time.sleep"),
        ):
            self._mock_pool(mock_pool_cls)
            # If total_expected is correctly 1 this exits; if 2 it would loop forever
            generate_intermediate_distograms_from_representations(
                "job", self._make_model_runner_and_params(), str(out_dir)
            )

    def test_progress_loop_exits_when_files_match_expected(self, tmp_path):
        staging = tmp_path / "staging"
        staging.mkdir()
        out_dir = tmp_path / "out"
        out_dir.mkdir()

        n = 3
        for i in range(n):
            _make_repr_file(staging, f"c_{i}_representations.pkl")
            (out_dir / f"c_{i}_distogram.npz").write_bytes(b"")

        with (
            patch("alphafold.model.modules.intermediate_structures_dir", str(staging)),
            patch("concurrent.futures.ProcessPoolExecutor") as mock_pool_cls,
            patch("time.sleep"),
        ):
            self._mock_pool(mock_pool_cls)
            generate_intermediate_distograms_from_representations(
                "job", self._make_model_runner_and_params(), str(out_dir)
            )
