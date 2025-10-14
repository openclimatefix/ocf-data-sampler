"""
Base class testing - SampleBase
"""
import numpy as np
import pytest
import torch

from ocf_data_sampler.torch_datasets.utils.torch_batch_utils import (
    batch_to_tensor,
    copy_batch_to_device,
)


def test_batch_to_tensor_nested():
    """Test nested dictionary conversion"""
    batch = {"outer": {"inner": np.array([1, 2, 3])}}
    tensor_batch = batch_to_tensor(batch)

    assert torch.equal(tensor_batch["outer"]["inner"], torch.tensor([1, 2, 3]))


def test_batch_to_tensor_mixed_types():
    """Test handling of mixed data types"""
    batch = {
        "tensor_data": np.array([1, 2, 3]),
        "string_data": "not_a_tensor",
        "nested": {
            "numbers": np.array([4, 5, 6]),
            "text": "still_not_a_tensor",
        },
    }
    tensor_batch = batch_to_tensor(batch)

    assert isinstance(tensor_batch["tensor_data"], torch.Tensor)
    assert isinstance(tensor_batch["string_data"], str)
    assert isinstance(tensor_batch["nested"]["numbers"], torch.Tensor)
    assert isinstance(tensor_batch["nested"]["text"], str)


def test_batch_to_tensor_different_dtypes():
    """Test conversion of arrays with different dtypes"""
    batch = {
        "float_data": np.array([1.0, 2.0, 3.0], dtype=np.float32),
        "int_data": np.array([1, 2, 3], dtype=np.int64),
        "bool_data": np.array([True, False, True], dtype=bool),
    }
    tensor_batch = batch_to_tensor(batch)

    assert isinstance(tensor_batch["bool_data"], torch.Tensor)
    assert tensor_batch["float_data"].dtype == torch.float32
    assert tensor_batch["int_data"].dtype == torch.int64
    assert tensor_batch["bool_data"].dtype == torch.bool


def test_batch_to_tensor_multidimensional():
    """Test conversion of multidimensional arrays"""
    batch = {
        "matrix": np.array([[1, 2], [3, 4]]),
        "tensor": np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]]),
    }
    tensor_batch = batch_to_tensor(batch)

    assert tensor_batch["matrix"].shape == (2, 2)
    assert tensor_batch["tensor"].shape == (2, 2, 2)
    assert torch.equal(tensor_batch["matrix"], torch.tensor([[1, 2], [3, 4]]))

@pytest.mark.skipif(not torch.cuda.is_available(), reason="Test requires GPU")
def test_copy_batch_to_device():
    """Test moving tensors to a different device"""
    device = torch.device("cuda:0")
    batch = {
        "tensor_data": torch.tensor([1, 2, 3]),
        "nested": {"matrix": torch.tensor([[1, 2], [3, 4]])},
        "non_tensor": "unchanged",
    }
    moved_batch = copy_batch_to_device(batch, device)

    assert moved_batch["tensor_data"].device == device
    assert moved_batch["nested"]["matrix"].device == device
    assert moved_batch["non_tensor"] == "unchanged"  # Non-tensors should remain unchanged
