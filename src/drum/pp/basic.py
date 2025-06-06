from anndata import AnnData


def basic_preproc(adata: AnnData) -> int:
    """Run a basic preprocessing on the AnnData object.

    Parameters
    ----------
    adata
        The AnnData object to preprocess.

    Returns
    -------
    Some integer value.
    """
    print("Implement a preprocessing function here.")
    return 0


def create_metacell(adata: AnnData) -> AnnData:
    """Create metacells from the AnnData object.

    Parameters
    ----------
    adata
        The AnnData object to preprocess.

    Returns
    -------
    An AnnData object with metacells.
    """
    return NotImplementedError


def align_anndata(adata1: AnnData, adata2: AnnData, verbose: bool = True) -> tuple[AnnData, AnnData]:
    """
    Align two AnnData objects by their indices, keeping only shared observations.

    This function is useful for multiome data where measurements from different
    modalities need to be matched by common identifiers.

    Parameters
    ----------
    adata1 : AnnData
        First AnnData object
    adata2 : AnnData
        Second AnnData object
    verbose : bool, default=True
        Whether to print information about the alignment process

    Returns
    -------
    tuple
        A tuple containing (adata1_aligned, adata2_aligned), where:
        - adata1_aligned : AnnData - First AnnData object with only shared observations
        - adata2_aligned : AnnData - Second AnnData object with only shared observations

    Raises
    ------
    ValueError
        If no shared indices are found between the two AnnData objects
    TypeError
        If inputs are not AnnData objects
    """
    # Validate input types
    if not isinstance(adata1, AnnData) or not isinstance(adata2, AnnData):
        raise TypeError("Both inputs must be AnnData objects")

    if verbose:
        print(f"Original shapes: {adata1.shape} and {adata2.shape}")

    # Get indices from both objects
    indices1 = adata1.obs.index
    indices2 = adata2.obs.index

    # Find shared indices
    shared_indices = indices1.intersection(indices2)

    if verbose:
        print(f"Found {len(shared_indices)} shared observations")

    # Check if we have any shared indices
    if len(shared_indices) == 0:
        raise ValueError("No shared indices found between the two AnnData objects")

    # Subset both AnnData objects to keep only shared indices
    adata1_aligned = adata1[adata1.obs.index.isin(shared_indices)].copy()
    adata2_aligned = adata2[adata2.obs.index.isin(shared_indices)].copy()

    # Check for duplicates
    dup1 = adata1_aligned.obs.index.duplicated().sum()
    dup2 = adata2_aligned.obs.index.duplicated().sum()

    if dup1 > 0 or dup2 > 0:
        if verbose:
            print(f"Warning: Found duplicates in aligned data: {dup1} in first object, {dup2} in second object")

        # Deduplicate by keeping first occurrence
        adata1_aligned = adata1_aligned[~adata1_aligned.obs.index.duplicated()].copy()
        adata2_aligned = adata2_aligned[~adata2_aligned.obs.index.duplicated()].copy()

        # Re-align after deduplication
        shared_indices = adata1_aligned.obs.index.intersection(adata2_aligned.obs.index)
        adata1_aligned = adata1_aligned[adata1_aligned.obs.index.isin(shared_indices)].copy()
        adata2_aligned = adata2_aligned[adata2_aligned.obs.index.isin(shared_indices)].copy()

    # Ensure both AnnData objects have the same observation ordering
    adata1_aligned = adata1_aligned[shared_indices].copy()
    adata2_aligned = adata2_aligned[shared_indices].copy()

    if verbose:
        print(f"Final aligned shapes: {adata1_aligned.shape} and {adata2_aligned.shape}")

    return adata1_aligned, adata2_aligned
