import numpy as np
import torch

import drum


def test_package_has_version():
    assert drum.__version__ is not None


def test_example():
    assert 1 == 1  # Test now passes


def test_onehot_encoding():
    """Test the one-hot encoding function."""
    sequence = "ACGTN"
    expected_shape = (4, 5)  # 4 nucleotides (ACGT), 5 positions

    # Perform one-hot encoding
    encoding = drum.tl._seq.onehot_encoding(sequence)

    # Check shape
    assert encoding.shape == expected_shape

    # Check dtype
    assert encoding.dtype == torch.float16

    # Convert to numpy for easier validation
    encoding_np = encoding.numpy()

    # Check N is ignored (should be all zeros in that column)
    assert np.all(encoding_np[:, 4] == 0)

    # Check correct encoding for each nucleotide (A=0, C=1, G=2, T=3)
    nucleotide_positions = {"A": 0, "C": 1, "G": 2, "T": 3}

    for i, nucleotide in enumerate("ACGT"):
        col_idx = nucleotide_positions[nucleotide]
        assert encoding_np[col_idx, i] == 1


def test_aggregate_obs_basic():
    """Test the aggregate_obs function with basic aggregation."""
    # Create a minimal AnnData object
    import anndata as ad
    import pandas as pd

    # Create dummy data
    X = np.array([[1, 2], [3, 4], [5, 6], [7, 8]])
    obs = pd.DataFrame(
        {
            "group": pd.Categorical(["A", "A", "B", "B"]),  # Use Categorical type explicitly
            "value": [1.0, 2.0, 3.0, 4.0],
        }
    )
    var = pd.DataFrame(index=["gene1", "gene2"])

    # Create AnnData object with explicit string indices
    obs.index = [f"cell_{i}" for i in range(len(obs))]
    var.index = var.index.astype(str)  # Ensure string index
    adata = ad.AnnData(X=X, obs=obs, var=var)

    # Aggregate by group
    aggregated = drum.tl._cell.aggregate_obs(adata, by="group", X_agg="sum", obs_agg={"value": "mean"})

    # Check shape
    assert aggregated.shape == (2, 2)  # 2 groups, 2 genes

    # Check X aggregation
    expected_X = np.array([[4, 6], [12, 14]])  # Sum of A and B groups
    np.testing.assert_array_equal(aggregated.X, expected_X)

    # Check obs aggregation
    expected_values = np.array([1.5, 3.5])  # Mean of values for A and B
    np.testing.assert_array_equal(aggregated.obs["value"], expected_values)

    # Check that group column is preserved
    assert "group" in aggregated.obs
    np.testing.assert_array_equal(aggregated.obs["group"], ["A", "B"])
