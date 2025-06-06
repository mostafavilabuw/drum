import re
from collections.abc import Mapping
from typing import Callable, Optional, Union

import anndata as ad
import genomepy
import numpy as np
import pandas as pd
import scipy
import sklearn
from anndata import AnnData
from pandas.api.types import CategoricalDtype
from sklearn.preprocessing import normalize
from tqdm import tqdm

Array = Union[np.ndarray, scipy.sparse.spmatrix]


def basic_tool(adata: AnnData) -> int:
    """Run a tool on the AnnData object.

    Parameters
    ----------
    adata
        The AnnData object to preprocess.

    Returns
    -------
    Some integer value.
    """
    print("Implement a tool to run on the AnnData object.")
    return 0


# Adopted from https://github.com/gao-lab/GLUE/
def lsi(adata: AnnData, n_components: int = 20, use_highly_variable: Optional[bool] = None, **kwargs) -> None:
    r"""
    LSI analysis (following the Seurat v3 approach)

    Parameters
    ----------
    adata
        Input dataset
    n_components
        Number of dimensions to use
    use_highly_variable
        Whether to use highly variable features only, stored in
        ``adata.var['highly_variable']``. By default uses them if they
        have been determined beforehand.
    **kwargs
        Additional keyword arguments are passed to
        :func:`sklearn.utils.extmath.randomized_svd`
    """
    if "random_state" not in kwargs:
        kwargs["random_state"] = 0  # Keep deterministic as the default behavior
    if use_highly_variable is None:
        use_highly_variable = "highly_variable" in adata.var
    adata_use = adata[:, adata.var["highly_variable"]] if use_highly_variable else adata
    X = tfidf(adata_use.X)
    X_norm = normalize(X, norm="l1")
    X_norm = np.log1p(X_norm * 1e4)
    X_lsi = sklearn.utils.extmath.randomized_svd(X_norm, n_components, **kwargs)[0]
    X_lsi -= X_lsi.mean(axis=1, keepdims=True)
    X_lsi /= X_lsi.std(axis=1, ddof=1, keepdims=True)
    adata.obsm["X_lsi"] = X_lsi


def tfidf(X: Array) -> Array:
    r"""
    TF-IDF normalization (following the Seurat v3 approach)

    Parameters
    ----------
    X
        Input matrix

    Returns
    -------
    X_tfidf
        TF-IDF normalized matrix
    """
    idf = X.shape[0] / X.sum(axis=0)
    if scipy.sparse.issparse(X):
        tf = X.multiply(1 / X.sum(axis=1))
        return tf.multiply(idf)
    else:
        tf = X / X.sum(axis=1, keepdims=True)
        return tf * idf


def aggregate_obs(
    adata: AnnData,
    by: str,
    X_agg: Optional[str] = "sum",
    obs_agg: Optional[Mapping[str, str]] = None,
    obsm_agg: Optional[Mapping[str, str]] = None,
    layers_agg: Optional[Mapping[str, str]] = None,
    verbose: bool = True,
) -> AnnData:
    r"""
    Aggregate obs in a given dataset by certain categories

    Parameters
    ----------
    adata
        Dataset to be aggregated
    by
        Specify a column in ``adata.obs`` used for aggregation,
        must be discrete.
    X_agg
        Aggregation function for ``adata.X``, must be one of
        ``{"sum", "mean", ``None``}``. Setting to ``None`` discards
        the ``adata.X`` matrix.
    obs_agg
        Aggregation methods for ``adata.obs``, indexed by obs columns,
        must be one of ``{"sum", "mean", "majority"}``, where ``"sum"``
        and ``"mean"`` are for continuous data, and ``"majority"`` is for
        discrete data. Fields not specified will be discarded. If not provided,
        will automatically use "majority" for categorical columns and "mean" for numeric columns.
    obsm_agg
        Aggregation methods for ``adata.obsm``, indexed by obsm keys,
        must be one of ``{"sum", "mean"}``. Fields not specified will be
        discarded.
    layers_agg
        Aggregation methods for ``adata.layers``, indexed by layer keys,
        must be one of ``{"sum", "mean"}``. Fields not specified will be
        discarded.
    verbose
        Whether to print information about the aggregation methods used.

    Returns
    -------
    aggregated
        Aggregated dataset
    """
    # Initialize aggregation dictionaries if not provided
    obs_agg = obs_agg or {}
    obsm_agg = obsm_agg or {}
    layers_agg = layers_agg or {}

    original_by = by

    # For obs columns not specified in obs_agg, automatically determine the method
    # Use "majority" for categorical columns and "mean" for numeric columns
    for col in adata.obs.columns:
        if col not in obs_agg and col != by:
            if isinstance(adata.obs[col].dtype, CategoricalDtype) or adata.obs[col].dtype == "object":
                obs_agg[col] = "majority"
            elif np.issubdtype(adata.obs[col].dtype, np.number):
                obs_agg[col] = "mean"

    # Print information about aggregation methods if verbose
    if verbose:
        print(f"Aggregating by '{by}'")
        if X_agg:
            print(f"Using '{X_agg}' aggregation for X matrix")

        if obs_agg:
            print("Observation aggregations:")
            print("  Using majority:", ", ".join(k for k, v in obs_agg.items() if v == "majority"))
            print("  Using mean:", ", ".join(k for k, v in obs_agg.items() if v == "mean"))
            print("  Using sum:", ", ".join(k for k, v in obs_agg.items() if v == "sum"))

        if obsm_agg:
            print("obsm aggregations:", ", ".join(f"{k}: {v}" for k, v in obsm_agg.items()))

        if layers_agg:
            print("layers aggregations:", ", ".join(f"{k}: {v}" for k, v in layers_agg.items()))

    by = adata.obs[by]
    agg_idx = pd.Index(by.cat.categories) if isinstance(by.dtype, CategoricalDtype) else pd.Index(np.unique(by))
    agg_sum = scipy.sparse.coo_matrix(
        (np.ones(adata.shape[0]), (agg_idx.get_indexer(by), np.arange(adata.shape[0])))
    ).tocsr()
    agg_mean = agg_sum.multiply(1 / agg_sum.sum(axis=1))

    agg_method = {
        "sum": lambda x: agg_sum @ x,
        "mean": lambda x: agg_mean @ x,
        "majority": lambda x: pd.crosstab(by, x).idxmax(axis=1).loc[agg_idx].to_numpy(),
    }

    X = agg_method[X_agg](adata.X) if X_agg and adata.X is not None else None
    obs = pd.DataFrame({k: agg_method[v](adata.obs[k]) for k, v in obs_agg.items()}, index=agg_idx.astype(str))

    obs[original_by] = obs.index

    obsm = {k: agg_method[v](adata.obsm[k]) for k, v in obsm_agg.items()}
    layers = {k: agg_method[v](adata.layers[k]) for k, v in layers_agg.items()}
    for c in obs:
        if isinstance(adata.obs[c].dtype, CategoricalDtype):
            obs[c] = pd.Categorical(obs[c], categories=adata.obs[c].cat.categories)
    return AnnData(
        X=X,
        obs=obs,
        var=adata.var,
        obsm=obsm,
        varm=adata.varm,
        layers=layers,
        uns=adata.uns,
        dtype=None if X is None else X.dtype,
    )


def annotation_gene_meta(adata: AnnData, genome_name, genome_dir) -> AnnData:
    r"""
    Annotate gene metadata with genome annotation, including chromosome, start, and end positions, and strand.

    Parameters
    ----------
    adata
        Input dataset
    genome_name
        Name of the genome
    genome_dir
        Directory where the genome is located

    Returns
    -------
    adata
        Annotated dataset

    """
    # check if the adata already has the all gene metadata
    if all(col in adata.var.columns for col in ["gene", "chr", "start", "end", "strand"]):
        adata.var = adata.var.set_index("gene", drop=False)
        print("Gene metadata already annotated with genome annotation.")
        return adata

    genome_annotation = genomepy.Annotation(genome_name, genome_dir)

    print(f"Loading genome annotation for {genome_name} from {genome_dir}...")

    print(genome_annotation.named_gtf.columns)

    annotation_df = genome_annotation.named_gtf

    # Filter for gene features
    annotation_df = annotation_df[annotation_df["feature"] == "gene"]

    annotation_df = annotation_df.reset_index()  # Reset index to get gene names in a separate column
    annotation_df = annotation_df.rename(columns={"gene_name": "gene", "seqname": "chr"})

    print("Gene metadata loaded from genome annotation.")

    # get the canonical row for each gene based on the longest length
    gene_ranges = annotation_df.loc[
        annotation_df.groupby("gene").apply(lambda df: (df["end"] - df["start"]).abs().idxmax()).values
    ].reset_index(drop=True)

    adata.var = adata.var.merge(
        gene_ranges,
        left_on="gene",
        right_on="gene",
        how="left",
    )
    adata.var = adata.var.set_index("gene", drop=False)

    print(f"Gene metadata annotated with {genome_name} genome annotation.")
    return adata


# TODO: improve the speed of this function
def calculate_gene_activity(
    atac_adata,
    gene_annotation,
    extend_upstream=2000,
    extend_downstream=0,
    use_tss=True,
    use_strand=True,
    exclude_gene_body=False,
    verbose=True,
):
    """
    Calculate gene activity scores based on ATAC-seq data and gene annotations.

    Parameters
    ----------
    atac_adata : AnnData
        AnnData object containing ATAC-seq peak data
    gene_annotation : pd.DataFrame
        DataFrame containing gene coordinates with columns 'chr', 'start', 'end', and optionally 'strand'
    extend_upstream : int, default=2000
        Number of base pairs to extend upstream of TSS
    extend_downstream : int, default=0
        Number of base pairs to extend downstream of TSS
    use_tss : bool, default=True
        Whether to use TSS (transcription start site) or gene body
    use_strand : bool, default=True
        Whether to use strand information for extending coordinates
    exclude_gene_body : bool, default=False
        If True, only considers the extended regions around the TSS and excludes the gene body itself
    verbose : bool, default=True
        Whether to show progress bar and print messages

    Returns
    -------
    AnnData or None
        AnnData object with gene activity scores or None if inplace=True
    """
    if not isinstance(atac_adata, AnnData):
        raise TypeError("atac_adata must be an AnnData object")

    if not isinstance(gene_annotation, pd.DataFrame):
        raise TypeError("gene_annotation must be a pandas DataFrame")

    required_columns = ["chr", "start", "end"]
    if use_strand:
        required_columns.append("strand")
    missing_columns = [col for col in required_columns if col not in gene_annotation.columns]
    if missing_columns:
        raise ValueError(f"gene_annotation is missing required columns: {', '.join(missing_columns)}")

    # Get the extended gene coordinates
    if exclude_gene_body and use_tss:
        # If excluding gene body, create TSS-focused regions without the gene body
        tss_df = gene_annotation.copy()

        if use_strand:
            # For + strand, TSS is at start position
            plus_strand = tss_df["strand"] == "+"
            # For - strand, TSS is at end position
            minus_strand = tss_df["strand"] == "-"

            # For + strand: extend upstream before start, downstream after start
            tss_df.loc[plus_strand, "end"] = tss_df.loc[plus_strand, "start"] + extend_downstream
            tss_df.loc[plus_strand, "start"] = np.maximum(0, tss_df.loc[plus_strand, "start"] - extend_upstream)

            # For - strand: extend upstream after end, downstream before end
            tss_df.loc[minus_strand, "start"] = np.maximum(0, tss_df.loc[minus_strand, "end"] - extend_downstream)
            tss_df.loc[minus_strand, "end"] = tss_df.loc[minus_strand, "end"] + extend_upstream
        else:
            # Without strand info, extend symmetrically around TSS (assumed to be at start)
            tss_df["end"] = tss_df["start"] + extend_downstream
            tss_df["start"] = np.maximum(0, tss_df["start"] - extend_upstream)

        extended_genes_df = tss_df
        if verbose:
            print("Using TSS-centered regions only (excluding gene body)")
    else:
        # Use regular extension including gene body
        extended_genes_df = extend_coordinates(gene_annotation, upstream=extend_upstream, downstream=extend_downstream)

    # Initialize gene activity matrix
    gene_activity_matrix = np.zeros((atac_adata.shape[0], extended_genes_df.shape[0]))

    # Display progress if verbose
    iterator = (
        tqdm(
            enumerate(extended_genes_df.iterrows()), total=extended_genes_df.shape[0], desc="Calculating gene activity"
        )
        if verbose
        else enumerate(extended_genes_df.iterrows())
    )

    for gene_idx, (gene_id, gene_info) in iterator:
        try:
            # Get gene coordinates
            gene_chrom = gene_info["chr"]
            gene_start = float(gene_info["start"])
            gene_end = float(gene_info["end"])

            # Find overlapping ATAC peaks using DataFrame filtering
            overlapping_peaks = atac_adata.var[
                (atac_adata.var["chr"] == gene_chrom)
                & (atac_adata.var["start"] <= gene_end)
                & (atac_adata.var["end"] >= gene_start)
            ]

            if len(overlapping_peaks) > 0:
                # Get all overlapping peaks at once and sum their signal
                peak_data = atac_adata[:, overlapping_peaks.index].X
                if scipy.sparse.issparse(peak_data):
                    gene_activity_matrix[:, gene_idx] = np.array(peak_data.sum(axis=1)).flatten()
                else:
                    gene_activity_matrix[:, gene_idx] = np.sum(peak_data, axis=1)

        except (ValueError, TypeError, KeyError) as e:
            if verbose:
                print(f"Error processing gene {gene_id}: {str(e)}")
            continue

    # Create new AnnData object with gene activity scores
    gene_activity_adata = ad.AnnData(X=gene_activity_matrix, obs=atac_adata.obs.copy(), var=extended_genes_df.copy())

    # Add metadata
    gene_activity_adata.uns["gene_activity_params"] = {
        "extend_upstream": extend_upstream,
        "extend_downstream": extend_downstream,
        "use_tss": use_tss,
        "use_strand": use_strand,
        "exclude_gene_body": exclude_gene_body,
    }

    return gene_activity_adata


def extend_coordinates(genes_df: pd.DataFrame, upstream: int = 2000, downstream: int = 0) -> pd.DataFrame:
    """
    Extend gene coordinates to include upstream and downstream regions around gene body.

    Parameters
    ----------
    genes_df : pd.DataFrame
        DataFrame containing gene annotations
    upstream : int, optional
        Number of base pairs to extend upstream of gene start
    downstream : int, optional
        Number of base pairs to extend downstream of gene end

    Returns
    -------
    pd.DataFrame
        DataFrame with extended gene coordinates
    """
    # Create a copy to avoid modifying the original
    extended_df = genes_df.copy()

    # Get strand information if available
    strand_col = "strand" if "strand" in genes_df.columns else None

    if strand_col:
        # For + strand genes: upstream is before start, downstream is after end
        plus_strand = extended_df[strand_col] == "+"
        extended_df.loc[plus_strand, "start"] = np.maximum(0, extended_df.loc[plus_strand, "start"] - upstream)
        extended_df.loc[plus_strand, "end"] = extended_df.loc[plus_strand, "end"] + downstream

        # For - strand genes: upstream is after end, downstream is before start
        minus_strand = extended_df[strand_col] == "-"
        extended_df.loc[minus_strand, "start"] = np.maximum(0, extended_df.loc[minus_strand, "start"] - downstream)
        extended_df.loc[minus_strand, "end"] = extended_df.loc[minus_strand, "end"] + upstream
    else:
        # If no strand information, extend in both directions
        extended_df["start"] = np.maximum(0, extended_df["start"] - upstream)
        extended_df["end"] = extended_df["end"] + downstream

    return extended_df


def calculate_adata_correlation(
    adata1: ad.AnnData,
    adata2: ad.AnnData,
    features: Optional[list[str]] = None,
    features_dim: str = "var",
    method: str = "pearson",
    min_variance: float = 1e-8,
    verbose: bool = True,
    custom_correlation_func: Optional[Callable] = None,
) -> pd.DataFrame:
    """
    Calculate correlation between features in two AnnData objects.

    Parameters
    ----------
    adata1 : ad.AnnData
        First AnnData object
    adata2 : ad.AnnData
        Second AnnData object
    features : list[str], optional
        Specific features to calculate correlation for.
        If None, uses the intersection of features from both objects.
    features_dim : str, optional
        Dimension to use for features ('var' or 'obs')
    method : str, optional
        Correlation method ('pearson', 'spearman', 'kendall', or 'custom')
    min_variance : float, optional
        Minimum variance required to calculate correlation
    verbose : bool, optional
        Whether to print progress information
    custom_correlation_func : Callable, optional
        Custom function for correlation calculation when method='custom'
        Should accept two arrays and return (correlation, p-value)

    Returns
    -------
    pd.DataFrame
        DataFrame with correlation values between features (NaN for features with insufficient variance)
    """
    # Determine feature names based on dimension
    if features_dim == "var":
        features1 = adata1.var_names
        features2 = adata2.var_names
    elif features_dim == "obs":
        features1 = adata1.obs_names
        features2 = adata2.obs_names
    else:
        raise ValueError(f"Invalid features_dim: {features_dim}. Must be 'var' or 'obs'")

    # Find common features or use provided list
    if features is None:
        # checking if features1 and features2 are the same
        if set(features1) == set(features2):
            common_features = features1
        else:
            # Use intersection of features from both datasets
            common_features = list(set(features1) & set(features2))
            if verbose:
                print("Warning: the two datasets have different features. Using intersection of features.")
                print(f"Number of common features: {len(common_features)}")
    else:
        # Verify all requested features exist in both datasets
        missing_in_1 = set(features) - set(features1)
        missing_in_2 = set(features) - set(features2)

        if missing_in_1 or missing_in_2:
            if verbose:
                if missing_in_1:
                    print(f"Warning: {len(missing_in_1)} features missing in first dataset")
                if missing_in_2:
                    print(f"Warning: {len(missing_in_2)} features missing in second dataset")

            # Use only features present in both datasets
            common_features = list(set(features) - missing_in_1 - missing_in_2)
        else:
            common_features = features

    if len(common_features) == 0:
        raise ValueError("No common features found between the datasets")

    if verbose:
        print(f"Calculating correlation for {len(common_features)} features")

    # Initialize result dictionary with all features (will populate with NaN for low variance features)
    results = {
        "feature_id": common_features,
        "correlation": [np.nan] * len(common_features),
        "p_value": [np.nan] * len(common_features),
    }

    # Extract data matrices and handle sparse matrices
    matrix1 = adata1.X.toarray() if scipy.sparse.issparse(adata1.X) else adata1.X
    matrix2 = adata2.X.toarray() if scipy.sparse.issparse(adata2.X) else adata2.X

    # Transpose matrices if working with obs dimension
    if features_dim == "obs":
        matrix1 = matrix1.T
        matrix2 = matrix2.T

    # Set the correlation function
    if method.lower() == "pearson":
        corr_func = scipy.stats.pearsonr
    elif method.lower() == "spearman":
        corr_func = scipy.stats.spearmanr
    elif method.lower() == "kendall":
        corr_func = scipy.stats.kendalltau
    elif method.lower() == "custom" and custom_correlation_func is not None:
        corr_func = custom_correlation_func
    else:
        raise ValueError(f"Unsupported correlation method: {method}")

    # Calculate correlation for each feature
    low_variance_count = 0
    for i, feature in enumerate(common_features):
        try:
            if features_dim == "var":
                idx1 = np.where(adata1.var_names == feature)[0][0]
                idx2 = np.where(adata2.var_names == feature)[0][0]
            else:  # obs dimension
                idx1 = np.where(adata1.obs_names == feature)[0][0]
                idx2 = np.where(adata2.obs_names == feature)[0][0]

            values1 = matrix1[:, idx1] if features_dim == "var" else matrix1[idx1, :]
            values2 = matrix2[:, idx2] if features_dim == "var" else matrix2[idx2, :]

            # Check if variance is too small - if so, leave as NaN without calculation
            if np.var(values1) < min_variance or np.var(values2) < min_variance:
                low_variance_count += 1
                # No need to explicitly set NaN as it's already initialized that way
                continue

            # Calculate correlation
            corr, p_val = corr_func(values1, values2)

            # Store results at their corresponding index in the pre-populated arrays
            results["correlation"][i] = corr
            results["p_value"][i] = p_val

        except (IndexError, ValueError) as e:
            if verbose:
                print(f"Error calculating correlation for feature {feature}: {str(e)}")
            # Already NaN by default
            continue

    if verbose and low_variance_count > 0:
        print(f"{low_variance_count} features have insufficient variance (set as NaN)")

    # Create result DataFrame
    correlation_df = pd.DataFrame(results)

    # Add absolute correlation column for easier sorting
    correlation_df["abs_correlation"] = np.abs(correlation_df["correlation"])

    return correlation_df


def extract_and_add_gene_attributes(adata, attribute_col="attribute", attributes_to_extract=None):
    """
    Extract specified attributes from GTF-style attribute strings and add them to AnnData.var.

    Args:
        adata: AnnData object containing gene attributes
        attribute_col: Name of the column containing attribute strings (default: 'attribute')
        attributes_to_extract: List of attributes to extract (default: ['gene_id', 'gene_type'])
                              If None, extracts gene_id and gene_type

    Returns
    -------
        The modified AnnData object with new attribute columns
    """
    # Use default list only when None is passed
    if attributes_to_extract is None:
        attributes_to_extract = ["gene_id", "gene_type"]

    # Ensure the attribute column exists
    if attribute_col not in adata.var.columns:
        raise ValueError(f"Column '{attribute_col}' not found in adata.var")

    # Compile regex patterns for efficiency
    patterns = {attr: re.compile(rf'{attr} "([^"]+)"') for attr in attributes_to_extract}

    # Initialize dictionaries to hold extracted values
    extracted_values = {attr: [] for attr in attributes_to_extract}

    # Parse each attribute string
    for attr_str in adata.var[attribute_col].values:
        for attr in attributes_to_extract:
            value = None
            if isinstance(attr_str, str):
                match = patterns[attr].search(attr_str)
                if match:
                    # Special case for gene_id: remove version number
                    if attr == "gene_id":
                        value = match.group(1).split(".")[0]
                    else:
                        value = match.group(1)
            extracted_values[attr].append(value)

    # Add extracted values to adata.var
    for attr in attributes_to_extract:
        adata.var[attr] = extracted_values[attr]

    return adata
