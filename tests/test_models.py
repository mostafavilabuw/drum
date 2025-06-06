import anndata as ad
import numpy as np
import pandas as pd
import pytest
import torch

from drum.models.encoders import SeqEncoder
from drum.models.layers import ConvBlock
from drum.pp.basic import align_anndata


def test_conv_block():
    """Test the ConvBlock layer."""
    # Create ConvBlock with specified parameters
    in_channels = 4
    out_channels = 16
    kernel_size = 5
    conv_block = ConvBlock(in_channels, out_channels, kernel_size)

    # Check if the module has expected layers
    assert isinstance(conv_block[0], torch.nn.BatchNorm1d)
    assert isinstance(conv_block[1], torch.nn.GELU)
    assert isinstance(conv_block[2], torch.nn.Conv1d)

    # Create dummy input and check output shape
    batch_size = 8
    seq_len = 100
    x = torch.randn(batch_size, in_channels, seq_len)

    output = conv_block(x)

    # Output should have the specified number of channels and same sequence length with 'same' padding
    assert output.shape == (batch_size, out_channels, seq_len)


def test_seq_encoder():
    """Test the SeqEncoder module."""
    # Create a SeqEncoder
    in_channels = 4  # e.g., for DNA sequences (A, C, G, T)
    out_channels = 32
    seq_encoder = SeqEncoder(
        in_channels=in_channels,
        out_channels=out_channels,
        first_kernel_size=15,
        kernel_size=5,
        pooling_size=2,
        layers=3,
    )

    # Create dummy input (batch_size, channels, seq_len)
    batch_size = 4
    seq_len = 200
    x = torch.randn(batch_size, in_channels, seq_len)

    # Forward pass
    output = seq_encoder(x)

    # Check output shape - should reflect pooling operations
    # For 3 layers with pooling_size=2: seq_len / (2^3) = seq_len / 8
    expected_seq_len = seq_len // (2**3)
    assert output.shape == (batch_size, out_channels, expected_seq_len)


def test_align_anndata():
    """Test the align_anndata function for aligning two AnnData objects."""
    # Create two AnnData objects with partially overlapping observations
    # First AnnData
    obs1 = pd.DataFrame(index=["cell1", "cell2", "cell3", "cell4"])
    var1 = pd.DataFrame(index=["gene1", "gene2", "gene3"])
    X1 = np.random.rand(4, 3)
    adata1 = ad.AnnData(X=X1, obs=obs1, var=var1)

    # Second AnnData
    obs2 = pd.DataFrame(index=["cell2", "cell3", "cell4", "cell5"])
    var2 = pd.DataFrame(index=["geneA", "geneB"])
    X2 = np.random.rand(4, 2)
    adata2 = ad.AnnData(X=X2, obs=obs2, var=var2)

    # Align AnnData objects
    adata1_aligned, adata2_aligned = align_anndata(adata1, adata2)

    # Check shapes
    assert adata1_aligned.shape[0] == adata2_aligned.shape[0]  # Same number of observations
    assert adata1_aligned.shape[1] == 3  # Original number of variables in adata1
    assert adata2_aligned.shape[1] == 2  # Original number of variables in adata2

    # Check that only shared observations are kept
    expected_obs = ["cell2", "cell3", "cell4"]
    assert list(adata1_aligned.obs.index) == expected_obs
    assert list(adata2_aligned.obs.index) == expected_obs

    # Check that the observations are in the same order
    pd.testing.assert_index_equal(adata1_aligned.obs.index, adata2_aligned.obs.index)


def test_align_anndata_no_shared():
    """Test align_anndata with no shared observations."""
    # Create two AnnData objects with no overlapping observations
    obs1 = pd.DataFrame(index=["cell1", "cell2"])
    var1 = pd.DataFrame(index=["gene1", "gene2"])
    X1 = np.random.rand(2, 2)
    adata1 = ad.AnnData(X=X1, obs=obs1, var=var1)

    obs2 = pd.DataFrame(index=["cell3", "cell4"])
    var2 = pd.DataFrame(index=["geneA", "geneB"])
    X2 = np.random.rand(2, 2)
    adata2 = ad.AnnData(X=X2, obs=obs2, var=var2)

    # Should raise ValueError for no shared indices
    with pytest.raises(ValueError, match="No shared indices found"):
        align_anndata(adata1, adata2)
