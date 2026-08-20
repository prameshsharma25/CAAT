import h5py
import numpy as np
import pytest

from alphafold.model import modules


@pytest.fixture(autouse=True)
def attention_state(tmp_path):
    """Give every test a fresh attention writer and always close its file."""
    modules.reset_attention_state()
    modules.attention_dir = str(tmp_path)
    modules._save_attention_compressed = True
    modules.evoformer_loop_counter = 0
    modules.is_triangle = True
    yield
    modules.reset_attention_state()
    modules.attention_dir = None
    modules._save_attention_compressed = False


def _write_head(seed, model, values):
    modules.set_seed_number(seed)
    modules.set_model_number(model)
    modules.set_recycle_number(0)
    modules.write_array_to_file(values)


def _dataset_names(h5_file):
    names = []
    h5_file.visititems(
        lambda name, obj: (
            names.append(name) if isinstance(obj, h5py.Dataset) else None
        )
    )
    return names


def test_compressed_attention_is_separate_across_seeds_and_models(tmp_path):
    expected = {}
    for seed in (0, 1):
        for model in (1, 2):
            values = np.full((8, 8), seed * 10 + model, dtype=np.float32)
            head = modules.attention_head_counter
            path = (
                f"seed_{seed:03d}/model_{model}/recycle_0/"
                f"extra_msa_evoformer_loop_1/head_{head}"
            )
            expected[path] = (values.astype(np.float16), seed, model, head)
            _write_head(seed, model, values)

    modules.close_hdf5_file()

    with h5py.File(tmp_path / "attention_heads_compressed.h5", "r") as h5_file:
        assert set(_dataset_names(h5_file)) == set(expected)
        for path, (values, seed, model, head) in expected.items():
            dataset = h5_file[path]
            np.testing.assert_array_equal(dataset[:], values)
            assert dataset.compression == "gzip"
            assert dataset.compression_opts == 4
            assert dataset.shuffle
            assert dataset.dtype == np.dtype(np.float16)
            assert dataset.attrs["seed_number"] == seed
            assert dataset.attrs["model_number"] == model
            assert dataset.attrs["global_index"] == head
            assert dataset.attrs["recycle_number"] == 0
            assert dataset.attrs["loop_type"] == "extra_msa"
            assert dataset.attrs["loop_number"] == 1
            np.testing.assert_array_equal(dataset.attrs["shape"], values.shape)
            assert dataset.attrs["n_res"] == values.shape[-1]


def test_reopening_file_preserves_previous_seed(tmp_path):
    seed_zero = np.zeros((4, 4), dtype=np.float32)
    _write_head(0, 1, seed_zero)
    modules.close_hdf5_file()

    seed_one = np.ones((4, 4), dtype=np.float32)
    _write_head(1, 1, seed_one)
    modules.close_hdf5_file()

    with h5py.File(tmp_path / "attention_heads_compressed.h5", "r") as h5_file:
        seed_zero_path = (
            "seed_000/model_1/recycle_0/extra_msa_evoformer_loop_1/head_0"
        )
        seed_one_path = (
            "seed_001/model_1/recycle_0/extra_msa_evoformer_loop_1/head_1"
        )
        assert set(_dataset_names(h5_file)) == {seed_zero_path, seed_one_path}
        np.testing.assert_array_equal(
            h5_file[seed_zero_path][:], seed_zero.astype(np.float16)
        )
        np.testing.assert_array_equal(
            h5_file[seed_one_path][:], seed_one.astype(np.float16)
        )
