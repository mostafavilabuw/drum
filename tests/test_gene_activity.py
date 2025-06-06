import anndata as ad
import numpy as np
import pandas as pd

from drum.tl._cell import calculate_adata_correlation, calculate_gene_activity, extend_coordinates


def test_extend_coordinates():
    """Test the extend_coordinates function."""
    # Create a test dataframe with gene coordinates
    genes_df = pd.DataFrame(
        {
            "chr": ["chr1", "chr2", "chr3"],
            "start": [1000, 2000, 3000],
            "end": [1500, 2500, 3500],
            "strand": ["+", "-", "+"],
        }
    )

    # Test with default parameters
    extended = extend_coordinates(genes_df, upstream=2000, downstream=0)

    # Check + strand genes (upstream is before start, downstream is after end)
    plus_genes = extended[extended["strand"] == "+"]
    assert plus_genes.loc[0, "start"] == 0  # 1000 - 2000 = -1000, but capped at 0
    assert plus_genes.loc[0, "end"] == 1500  # Original end is preserved
    assert plus_genes.loc[2, "start"] == 1000  # 3000 - 2000 = 1000
    assert plus_genes.loc[2, "end"] == 3500  # Original end is preserved

    # Check - strand genes (upstream is after end, downstream is before start)
    minus_genes = extended[extended["strand"] == "-"]
    assert minus_genes.loc[1, "start"] == 2000  # Original start is preserved
    assert minus_genes.loc[1, "end"] == 4500  # 2500 + 2000 = 4500 (because it's - strand, we add upstream to end)

    # Test with custom upstream/downstream
    extended = extend_coordinates(genes_df, upstream=1000, downstream=500)

    # Check + strand with custom parameters
    plus_genes = extended[extended["strand"] == "+"]
    assert plus_genes.loc[0, "start"] == 0  # 1000 - 1000 = 0
    assert plus_genes.loc[0, "end"] == 2000  # 1500 + 500 = 2000
    assert plus_genes.loc[2, "start"] == 2000  # 3000 - 1000 = 2000
    assert plus_genes.loc[2, "end"] == 4000  # 3500 + 500 = 4000

    # Check - strand with custom parameters
    minus_genes = extended[extended["strand"] == "-"]
    assert minus_genes.loc[1, "start"] == 1500  # 2000 - 500 (downstream) = 1500
    assert minus_genes.loc[1, "end"] == 3500  # 2500 + 1000 (upstream) = 3500

    # Test without strand information
    genes_no_strand = genes_df.drop(columns=["strand"])
    extended_no_strand = extend_coordinates(genes_no_strand, upstream=1000, downstream=500)

    # All genes should be extended in both directions regardless of strand
    assert extended_no_strand.loc[0, "start"] == 0  # 1000 - 1000 = 0
    assert extended_no_strand.loc[0, "end"] == 2000  # 1500 + 500 = 2000
    assert extended_no_strand.loc[1, "start"] == 1000  # 2000 - 1000 = 1000
    assert extended_no_strand.loc[1, "end"] == 3000  # 2500 + 500 = 3000
    assert extended_no_strand.loc[2, "start"] == 2000  # 3000 - 1000 = 2000
    assert extended_no_strand.loc[2, "end"] == 4000  # 3500 + 500 = 4000


def test_calculate_adata_correlation():
    """Test the calculate_adata_correlation function."""
    # Create two simple AnnData objects with the same features
    n_obs = 10
    n_vars = 5

    # Create random data with known correlation structure
    np.random.seed(42)
    X1 = np.random.rand(n_obs, n_vars)

    # Create X2 with perfect correlation to X1 for some features
    X2 = X1.copy()
    # Make one feature uncorrelated
    X2[:, 0] = np.random.rand(n_obs)
    # Make one feature negatively correlated
    X2[:, 1] = -X1[:, 1]
    # Leave others perfectly correlated

    # Create AnnData objects
    var_names = [f"feature_{i}" for i in range(n_vars)]
    obs_names = [f"obs_{i}" for i in range(n_obs)]

    adata1 = ad.AnnData(X=X1, var=pd.DataFrame(index=var_names), obs=pd.DataFrame(index=obs_names))
    adata2 = ad.AnnData(X=X2, var=pd.DataFrame(index=var_names), obs=pd.DataFrame(index=obs_names))

    # Calculate correlations
    corr_df = calculate_adata_correlation(adata1, adata2, method="pearson")

    # Check the shape of the result
    assert corr_df.shape[0] == n_vars
    assert "feature_id" in corr_df.columns
    assert "correlation" in corr_df.columns
    assert "p_value" in corr_df.columns
    assert "abs_correlation" in corr_df.columns

    # Check that features are sorted by absolute correlation
    assert corr_df["abs_correlation"].is_monotonic_decreasing

    # Check specific correlation values
    # Features 2, 3, 4 should have perfect positive correlation (1.0)
    perfect_corr_features = corr_df[corr_df["correlation"] > 0.99]["feature_id"].tolist()
    assert "feature_2" in perfect_corr_features
    assert "feature_3" in perfect_corr_features
    assert "feature_4" in perfect_corr_features

    # Feature 1 should have perfect negative correlation (-1.0)
    neg_corr_features = corr_df[corr_df["correlation"] < -0.99]["feature_id"].tolist()
    assert "feature_1" in neg_corr_features

    # Feature 0 should have lower correlation (close to 0)
    low_corr_features = corr_df[abs(corr_df["correlation"]) < 0.5]["feature_id"].tolist()
    assert "feature_0" in low_corr_features


def test_calculate_gene_activity_basic():
    """Test the calculate_gene_activity function with basic inputs."""
    # Create a simple ATAC-seq dataset
    n_obs = 5
    n_peaks = 10

    # Create sparse peak matrix (cells x peaks)
    X = np.ones((n_obs, n_peaks))

    # Create peak annotations with properly formatted indices
    var = pd.DataFrame(
        {
            "chr": ["chr1"] * 5 + ["chr2"] * 5,
            "start": [100, 500, 1000, 2000, 3000, 100, 500, 1000, 2000, 3000],
            "end": [200, 600, 1100, 2100, 3100, 200, 600, 1100, 2100, 3100],
        }
    )
    var.index = [f"peak_{i}" for i in range(n_peaks)]  # Explicitly set string indices

    # Create AnnData with string indices to avoid index transformation warnings
    obs = pd.DataFrame(index=[f"cell_{i}" for i in range(n_obs)])
    adata = ad.AnnData(X=X, var=var, obs=obs)

    # Create gene annotation dataframe
    gene_annotation = pd.DataFrame(
        {
            "gene": ["gene1", "gene2", "gene3"],
            "chr": ["chr1", "chr1", "chr2"],
            "start": [900, 2500, 900],
            "end": [1200, 2700, 1200],
            "strand": ["+", "-", "+"],
        }
    )
    gene_annotation.index = gene_annotation["gene"]  # Use gene names as indices

    # Calculate gene activity (non-inplace)
    result = calculate_gene_activity(
        adata,
        gene_annotation,
        extend_upstream=0,
        extend_downstream=100,
    )

    # Check that the result is an AnnData object with the right shape
    assert isinstance(result, ad.AnnData)
    assert result.shape[0] == n_obs  # Same number of observations
    assert result.shape[1] == len(gene_annotation)  # One column per gene

    # Verify metadata was stored
    assert "gene_activity_params" in result.uns
    assert result.uns["gene_activity_params"]["extend_upstream"] == 0
    assert result.uns["gene_activity_params"]["extend_downstream"] == 100

    # TODO: Check the values in the layer
