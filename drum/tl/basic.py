from collections.abc import Mapping
from typing import Optional, Union

import genomepy
import numpy as np
import pandas as pd
import scipy
import sklearn
from anndata import AnnData
from sklearn.preprocessing import normalize

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
        discrete data. Fields not specified will be discarded.
    obsm_agg
        Aggregation methods for ``adata.obsm``, indexed by obsm keys,
        must be one of ``{"sum", "mean"}``. Fields not specified will be
        discarded.
    layers_agg
        Aggregation methods for ``adata.layers``, indexed by layer keys,
        must be one of ``{"sum", "mean"}``. Fields not specified will be
        discarded.

    Returns
    -------
    aggregated
        Aggregated dataset
    """
    obs_agg = obs_agg or {}
    obsm_agg = obsm_agg or {}
    layers_agg = layers_agg or {}

    original_by = by

    by = adata.obs[by]
    agg_idx = pd.Index(by.cat.categories) if pd.api.types.is_categorical_dtype(by) else pd.Index(np.unique(by))
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
        if pd.api.types.is_categorical_dtype(adata.obs[c]):
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

    annotation_df = genome_annotation.named_gtf[["seqname", "start", "end", "strand"]]

    annotation_df = annotation_df.reset_index()  # Reset index to get gene names in a separate column
    annotation_df = annotation_df.rename(columns={"gene_name": "gene", "seqname": "chr"})

    # Group by gene to get the minimum start and maximum end for each gene
    gene_ranges = (
        annotation_df.groupby("gene")
        .agg(
            {
                "chr": "first",  # Assuming each gene is on a single chromosome
                "start": "min",  # Minimum start position
                "end": "max",  # Maximum end position
            }
        )
        .reset_index()
    )

    adata.var = adata.var.merge(
        gene_ranges,
        left_on="gene",
        right_on="gene",
        how="left",
    )
    adata.var = adata.var.set_index("gene", drop=False)

    print(f"Gene metadata annotated with {genome_name} genome annotation.")
    return adata
