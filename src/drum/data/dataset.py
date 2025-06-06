import warnings
from pathlib import Path

# Make Union available for type hinting
from typing import Optional, Union

import anndata
import numpy as np
import pandas as pd
import torch  # Add torch import
from torch.utils.data import Dataset
from tqdm import tqdm

from drum.tl import onehot_encoding

from .genome import Genome


class BaseSeqDataset(Dataset):
    """
    Base class for sequence datasets providing common functionality.

    Args:
        seq_meta (pd.DataFrame): Metadata for the sequences. Must contain 'chr', 'start', 'end'.
                                 Optionally 'strand' (defaults to '+').
        seq_len (int): Length of the sequence to extract, centered around the TSS.
        genome_name (str): Name of the genome assembly (e.g., "hg38").
        genome_dir (Optional[Path]): Directory where the genome is located or should be installed.
        genome_provider (Optional[str]): Provider for genome download (e.g., "ucsc").
        install_genome (bool): Whether to install the genome if not found.
        verbose (bool): Whether to show verbose output and warnings.
        **kwargs: Additional arguments passed to the Genome constructor.
    """

    def __init__(
        self,
        seq_meta: pd.DataFrame,
        seq_len: int = 1314,
        genome_name: str = "hg38",
        genome_dir: Optional[Path] = None,
        genome_provider: Optional[str] = None,
        install_genome: bool = True,
        verbose: bool = True,
        **kwargs,
    ):
        self.seq_meta = seq_meta
        self.seq_len = int(seq_len)
        self.genome_name = genome_name
        self.genome_dir = genome_dir
        self.verbose = verbose

        self.genome = Genome(
            genome_name=genome_name,
            genome_dir=genome_dir,
            genome_provider=genome_provider,
            install_genome=install_genome,
            **kwargs,
        )

        self.seq_cache: dict[str, str] = {}
        self.seq_info_cache: dict[int, pd.Series] = {}

    def _tensor_convert(self, data, dtype=torch.float32):
        """Safely convert data to tensor, avoiding unnecessary conversions if already a tensor."""
        if isinstance(data, torch.Tensor):
            # If already a tensor, just ensure correct dtype
            return data.to(dtype=dtype)
        else:
            # Otherwise convert to tensor
            return torch.tensor(data, dtype=dtype)

    def _get_seq_info(self, idx: int) -> pd.Series:
        """Get sequence info from cache or fetch from metadata."""
        if idx in self.seq_info_cache:
            return self.seq_info_cache[idx]
        else:
            # Ensure required columns exist
            required_cols = ["chr", "start", "end"]
            if not all(col in self.seq_meta.columns for col in required_cols):
                raise ValueError(f"seq_meta must contain columns: {required_cols}")
            seq_info = self.seq_meta.iloc[idx]
            self.seq_info_cache[idx] = seq_info
            return seq_info

    def _get_centered_coordinates(self, seq_info: pd.Series) -> tuple[str, int, int, int]:
        """
        Calculate sequence coordinates centered around TSS, adjusted for chromosome boundaries.

        Args:
            seq_info (pd.Series): Row from seq_meta containing 'chr', 'start', 'end', and optionally 'strand'.

        Returns
        -------
            tuple[str, int, int, int]: Chromosome name, adjusted sequence start (1-based),
                                       adjusted sequence end (1-based, inclusive),
                                       actual length of the sequence interval.
        """
        chr = seq_info["chr"]
        # Get strand information, default to "+" if not provided
        strand = seq_info.get("strand", "+")

        # Determine TSS position based on strand
        if strand == "+":
            tss = int(seq_info["start"])
        else:  # For "-" strand or any other value, use end position as TSS
            tss = int(seq_info["end"])

        # Calculate sequence coordinates centered around TSS
        half_len = self.seq_len // 2
        seq_start = tss - half_len
        # End coordinate is exclusive in genome.get_seq, but we calculate inclusive here for clarity
        seq_end = tss + half_len + (self.seq_len % 2) - 1  # Make end inclusive for interval calculation

        # Adjust for chromosome boundaries
        chr_len = self.genome.chr_sizes(chr)
        # Ensure start is at least 1
        seq_start = max(1, seq_start)
        # Ensure end does not exceed chromosome length
        seq_end = min(seq_end, chr_len)

        # Recalculate length and adjust start/end if interval is shorter than seq_len due to boundaries
        current_len = seq_end - seq_start + 1
        if current_len < self.seq_len:
            # If near the start of the chromosome
            if seq_start == 1:
                seq_end = min(self.seq_len, chr_len)
            # If near the end of the chromosome
            elif seq_end == chr_len:
                seq_start = max(1, chr_len - self.seq_len + 1)

        # Final adjusted length
        adjusted_len = seq_end - seq_start + 1

        # Return 1-based coordinates (start, end inclusive)
        return chr, seq_start, seq_end, adjusted_len

    def _get_seq(self, seq_info: pd.Series, ablation: list[str] = None) -> np.ndarray:
        """Get one-hot encoded sequence from cache or fetch from genome."""
        if ablation is None:
            ablation = []

        if "seq" in ablation:
            return np.zeros((4, self.seq_len), dtype=np.float32)

        # Use the helper method to get coordinates
        chr, seq_start, seq_end, adjusted_len = self._get_centered_coordinates(seq_info)

        cache_key = f"{chr}:{seq_start}-{seq_end}"  # Cache key uses 1-based inclusive coords

        if cache_key in self.seq_cache:
            seq = self.seq_cache[cache_key]
        else:
            try:
                # genome.get_seq expects 1-based start, 1-based inclusive end
                seq = self.genome.get_seq(chr, seq_start, seq_end)
                self.seq_cache[cache_key] = seq
            except ValueError as e:
                if self.verbose:
                    warnings.warn(
                        f"Error retrieving sequence for {chr}:{seq_start}-{seq_end}: {str(e)}. Returning N's.",
                        UserWarning,
                        stacklevel=2,
                    )
                # Use adjusted_len calculated by _get_centered_coordinates
                seq = "N" * adjusted_len
            except (OSError, RuntimeError) as e:  # Catch other potential errors from genome provider/filesystem
                if self.verbose:
                    warnings.warn(
                        f"Unexpected error retrieving sequence for {chr}:{seq_start}-{seq_end}: {str(e)}. Returning N's.",
                        UserWarning,
                        stacklevel=2,
                    )
                seq = "N" * adjusted_len

        one_hot_seq = onehot_encoding(seq)

        # Ensure correct length (pad if necessary, e.g., if adjusted_len < self.seq_len)
        current_len_onehot = one_hot_seq.shape[1]
        if current_len_onehot < self.seq_len:
            # Pad with zeros (representing Ns) if the sequence fetched was shorter than requested seq_len
            padding_needed = self.seq_len - current_len_onehot
            # Determine padding distribution (can add sophisticated logic, e.g., center padding)
            # Simple right padding:

            padding = np.zeros((one_hot_seq.shape[0], padding_needed), dtype=np.float32)
            one_hot_seq = np.hstack((one_hot_seq, padding))
        elif current_len_onehot > self.seq_len:
            # This case should ideally not happen if _get_centered_coordinates is correct
            # Truncate if somehow longer (e.g. off-by-one in genome provider)
            one_hot_seq = one_hot_seq[:, : self.seq_len]

        return one_hot_seq

    def _precache_seq(self) -> None:
        """Pre-fetch and cache sequences for better performance."""
        if not self.verbose:
            print("Prefetching sequences...")
        iterator = range(len(self.seq_meta))
        if self.verbose:
            iterator = tqdm(iterator, desc="Prefetching sequences")

        for idx in iterator:
            seq_info = self._get_seq_info(idx)  # Use cached info getter
            try:
                # Call _get_seq to fetch and cache, ignore the return value
                self._get_seq(seq_info)
            except (ValueError, Exception) as e:  # Catch potential errors during fetching
                if self.verbose:
                    chr, start, end, _ = self._get_centered_coordinates(seq_info)
                    warnings.warn(
                        f"Error precaching sequence {idx} ({chr}:{start}-{end}): {str(e)}",
                        UserWarning,
                        stacklevel=2,
                    )
        if not self.verbose:
            print("Sequence prefetching done.")


class SeqDataset(BaseSeqDataset):
    """
    A PyTorch dataset for working with sequence data.

    Args:
        seq_meta (pandas.DataFrame): Metadata for the sequences.
        seq_len (int, optional): Length of the sequence. Defaults to 1314.
        genome_name (str, optional): Name of the genome. Defaults to "hg38".
        genome_dir (Path, optional): Directory where the genome is located. Defaults to None.
        genome_provider (str, optional): Provider of the genome. Defaults to None.
        install_genome (bool, optional): Whether to install the genome if not found. Defaults to True.
    """

    def __init__(
        self,
        seq_meta: pd.DataFrame,
        seq_len: int = 1314,
        genome_name: str = "hg38",
        genome_dir: Optional[Path] = None,
        genome_provider: Optional[str] = None,
        install_genome: bool = True,
        **kwargs,
    ):
        super().__init__(
            seq_meta=seq_meta,
            seq_len=seq_len,
            genome_name=genome_name,
            genome_dir=genome_dir,
            genome_provider=genome_provider,
            install_genome=install_genome,
            **kwargs,
        )

    def __len__(self) -> int:
        return len(self.seq_meta)

    def __getitem__(self, idx: int) -> torch.Tensor:
        seq_info = self._get_seq_info(idx)
        try:
            one_hot_seq = self._tensor_convert(self._get_seq(seq_info, self.ablation))
            return one_hot_seq
        except ValueError as e:
            warnings.warn(
                f"Error processing sequence at index {idx}: {str(e)}. Returning zeros.",
                UserWarning,
                stacklevel=2,
            )
            return torch.zeros((4, self.seq_len), dtype=torch.float32)


class SeqByChromatinContextDataset(BaseSeqDataset):
    """
    A PyTorch dataset for generating sequences and their corresponding chromatin context.

    Args:
        seq_meta (pd.DataFrame): Metadata for the sequences. Must contain 'chr', 'start', 'end', 'gene'.
                                 Optionally 'strand' (defaults to '+').
        rna_adata (anndata.AnnData): AnnData containing RNA information (cells x genes).
        atac_adata (anndata.AnnData): AnnData containing ATAC information (cells x peaks).
                                      atac_adata.var must contain 'chr', 'start', 'end'.
        seq_len (int, optional): Length of the sequence. Defaults to 1314.
        genome_name (str, optional): Name of the genome. Defaults to "hg38".
        genome_dir (Path, optional): Directory where the genome is located. Defaults to None.
        genome_provider (str, optional): Provider of the genome. Defaults to None.
        install_genome (bool, optional): Whether to install the genome if not found. Defaults to False.
        verbose (bool, optional): Whether to show verbose output. Defaults to True.
        ablation (list, optional): Features to ablate (e.g., 'seq', 'chromatin', 'gene_expression'). Defaults to [].
        include_mean_expression (bool, optional): Whether to include mean gene expression across all cells. Defaults to False.
        precache_seq (bool, optional): Whether to pre-fetch and cache all DNA sequences on initialization. Defaults to False.
        precache_peaks (bool, optional): Whether to pre-compute peak overlaps for all sequences on initialization. Defaults to False.
    """

    def __init__(
        self,
        seq_meta: pd.DataFrame,
        rna_adata: anndata.AnnData,
        atac_adata: anndata.AnnData,
        seq_len: int = 1314,
        genome_name: str = "hg38",
        genome_dir: Optional[Path] = None,
        genome_provider: Optional[str] = None,
        install_genome: bool = False,
        verbose: bool = True,
        ablation: Optional[list[str]] = None,
        include_mean_expression: bool = False,
        precache_seq: bool = False,  # Added parameter
        precache_peaks: bool = False,  # Added parameter
    ):
        super().__init__(
            seq_meta=seq_meta,
            seq_len=seq_len,
            genome_name=genome_name,
            genome_dir=genome_dir,
            genome_provider=genome_provider,
            install_genome=install_genome,
            verbose=verbose,
        )

        # Input validation
        if "gene" not in seq_meta.columns:
            raise ValueError("seq_meta must contain a 'gene' column.")
        required_atac_var_cols = ["chr", "start", "end"]
        if not all(col in atac_adata.var.columns for col in required_atac_var_cols):
            raise ValueError(f"atac_adata.var must contain columns: {required_atac_var_cols}")
        if len(rna_adata) != len(atac_adata):
            raise ValueError(
                f"The RNA ({len(rna_adata)} obs) and ATAC ({len(atac_adata)} obs) "
                "AnnData objects must have the same number of observations (cells)."
            )

        self.rna_adata = rna_adata
        self.atac_adata = atac_adata

        # Convert AnnData objects to memory and to dense matrices
        self.rna_adata = self.rna_adata.to_memory()
        self.atac_adata = self.atac_adata.to_memory()

        # Convert sparse matrices to dense
        if hasattr(self.rna_adata.X, "toarray"):
            self.rna_adata.X = self.rna_adata.X.toarray()
        elif hasattr(self.rna_adata.X, "todense"):
            self.rna_adata.X = np.asarray(self.rna_adata.X.todense())

        if hasattr(self.atac_adata.X, "toarray"):
            self.atac_adata.X = self.atac_adata.X.toarray()
        elif hasattr(self.atac_adata.X, "todense"):
            self.atac_adata.X = np.asarray(self.atac_adata.X.todense())

        self.ablation = ablation if ablation is not None else []
        self.include_mean_expression = include_mean_expression

        # Caches
        self.rna_adata_cache: dict[int, anndata.AnnData] = {}  # Caches single-cell slices
        self.atac_adata_cache: dict[int, anndata.AnnData] = {}  # Caches single-cell slices
        self.peak_region_cache: dict[tuple, Optional[dict]] = {}  # Caches overlapping peak info per sequence region
        self.gene_expression_cache: dict[tuple, np.float32] = {}  # Caches expression per (gene, cell_id)
        self.batch_cache: dict[tuple[int, int], tuple] = {}  # Caches final output tuple per (seq_idx, adata_idx)

        # Pre-computed indices for efficiency
        self.atac_by_chr: dict[str, pd.Index] = {
            chr_name: self.atac_adata.var.index[self.atac_adata.var["chr"] == chr_name]
            for chr_name in self.atac_adata.var["chr"].unique()
        }
        self.rna_gene_indices: dict[str, int] = {
            gene: self.rna_adata.var_names.get_loc(gene) for gene in self.rna_adata.var_names
        }

        # Add cache for mean expression if needed
        if self.include_mean_expression:
            self.mean_expr_cache: dict[str, np.float32] = {}

        # Optional pre-caching based on flags
        if precache_peaks and "chromatin" not in self.ablation:
            self._precache_peak_regions()

        if precache_seq and "seq" not in self.ablation:
            self._precache_seq()

    def _precache_peak_regions(self):
        """Pre-compute peak regions for all genes to speed up _get_chromatin_context"""
        if not self.verbose:
            print("Pre-computing peak regions...")
        iterator = self.seq_meta.iterrows()
        if self.verbose:
            iterator = tqdm(iterator, total=len(self.seq_meta), desc="Pre-computing peak regions")

        for _, seq_info in iterator:  # Removed unused idx
            # Use helper to get coordinates
            chr, seq_start, seq_end, adjusted_len = self._get_centered_coordinates(seq_info)

            # Skip if we don't have data for this chromosome in ATAC
            if chr not in self.atac_by_chr or self.atac_by_chr[chr].empty:
                # No warning here, as it's expected during precaching
                continue

            # Create cache key based on actual genomic coordinates
            cache_key = (chr, seq_start, seq_end)

            # Skip if already cached (e.g., duplicate regions in seq_meta)
            if cache_key in self.peak_region_cache:
                continue

            # Get pre-filtered indices for this chromosome
            chr_indices = self.atac_by_chr[chr]

            # Find regions overlapping the sequence interval [seq_start, seq_end]
            potential_regions = self.atac_adata.var.loc[chr_indices]
            # Overlap condition: peak_start < seq_end AND peak_end > seq_start
            # Note: Using 1-based inclusive coordinates consistent with _get_centered_coordinates
            region_in_range = potential_regions[
                (potential_regions["start"] <= seq_end) & (potential_regions["end"] >= seq_start)
            ]

            # Cache the region indices and coordinates relative to the sequence start
            if not region_in_range.empty:
                starts_rel = np.maximum(region_in_range["start"].values, seq_start) - seq_start
                ends_rel = (
                    np.minimum(region_in_range["end"].values, seq_end) - seq_start + 1
                )  # Make end exclusive for slicing
                region_data = {
                    "indices": region_in_range.index.tolist(),
                    "starts_rel": starts_rel,  # 0-based, relative to seq start
                    "ends_rel": ends_rel,  # 0-based, relative to seq start, exclusive
                    "lengths": region_in_range["end"].values
                    - region_in_range["start"].values
                    + 1,  # Original peak lengths
                }
                self.peak_region_cache[cache_key] = region_data
            else:
                self.peak_region_cache[cache_key] = None
        if not self.verbose:
            print("Peak region pre-computation complete.")

    def _get_chromatin_context(self, atac_adata_cell: anndata.AnnData, seq_info: pd.Series) -> np.ndarray:
        """Get chromatin context for a specific sequence and cell's ATAC data."""
        seq_name = seq_info["gene"]  # Keep for potential debugging messages

        # Use helper to get coordinates
        chr, seq_start, seq_end, adjusted_len = self._get_centered_coordinates(seq_info)

        # Initialize empty chromatin track with the potentially adjusted length
        chromatin_track = np.zeros((1, adjusted_len), dtype=np.float32)

        if "chromatin" in self.ablation:
            return chromatin_track

        # Create cache key for peak regions based on genomic coordinates
        cache_key = (chr, seq_start, seq_end)

        try:
            region_data = None
            # Check if we have pre-computed peak regions
            if cache_key in self.peak_region_cache:
                region_data = self.peak_region_cache[cache_key]
            else:
                # Compute on the fly if not precached or cache miss
                if chr in self.atac_by_chr and not self.atac_by_chr[chr].empty:
                    chr_indices = self.atac_by_chr[chr]
                    potential_regions = self.atac_adata.var.loc[chr_indices]
                    region_in_range = potential_regions[
                        (potential_regions["start"] <= seq_end) & (potential_regions["end"] >= seq_start)
                    ]

                    if not region_in_range.empty:
                        starts_rel = np.maximum(region_in_range["start"].values, seq_start) - seq_start
                        ends_rel = np.minimum(region_in_range["end"].values, seq_end) - seq_start + 1
                        region_data = {
                            "indices": region_in_range.index.tolist(),
                            "starts_rel": starts_rel,
                            "ends_rel": ends_rel,
                            "lengths": region_in_range["end"].values - region_in_range["start"].values + 1,
                        }
                        # Optionally cache the computed result here if desired, though might duplicate precaching effort
                        # self.peak_region_cache[cache_key] = region_data

            # If no regions overlap (either from cache or computed), return zero track
            if region_data is None:
                if self.verbose > 1:  # More detailed verbosity
                    warnings.warn(
                        f"No overlapping chromatin peaks found for {seq_name} in region {chr}:{seq_start}-{seq_end}",
                        UserWarning,
                        stacklevel=2,
                    )
                return chromatin_track

            # Extract ATAC values for the specific cell and overlapping peaks
            # atac_adata_cell is assumed to be a slice for a single cell (1 obs)
            values = atac_adata_cell[:, region_data["indices"]].X
            values = values.toarray().flatten() if hasattr(values, "toarray") else np.asarray(values).flatten()

            # Add peak values to the track, normalizing by original peak length
            for i in range(len(values)):
                peak_start_rel = int(region_data["starts_rel"][i])
                peak_end_rel = int(region_data["ends_rel"][i])  # Exclusive end
                peak_len = float(region_data["lengths"][i])

                # Ensure indices are within the bounds of the current track length
                peak_start_rel = max(0, peak_start_rel)
                peak_end_rel = min(chromatin_track.shape[1], peak_end_rel)

                if peak_end_rel > peak_start_rel and peak_len > 0:
                    # Normalize by the original peak length
                    normalized_value = values[i] / peak_len
                    chromatin_track[0, peak_start_rel:peak_end_rel] += normalized_value

        except (ValueError, IndexError, KeyError) as e:
            if self.verbose:
                warnings.warn(
                    f"Error processing chromatin context for {seq_name} ({chr}:{seq_start}-{seq_end}): {str(e)}",
                    UserWarning,
                    stacklevel=2,
                )
            # Return zero track in case of error, ensure correct length
            chromatin_track = np.zeros((1, adjusted_len), dtype=np.float32)

        # Only pad if necessary to reach full seq_len (valid case when sequence is at chromosome boundaries)
        if adjusted_len < self.seq_len:
            padding_needed = self.seq_len - adjusted_len
            padding = np.zeros((1, padding_needed), dtype=np.float32)
            chromatin_track = np.hstack((chromatin_track, padding))
        elif chromatin_track.shape[1] > self.seq_len:
            # This should never happen - raise a warning if it does
            warnings.warn(
                f"Chromatin track length ({chromatin_track.shape[1]}) unexpectedly exceeds seq_len ({self.seq_len}) "
                f"for {seq_name} ({chr}:{seq_start}-{seq_end}). This indicates a bug in coordinate calculations.",
                UserWarning,
                stacklevel=2,
            )

        return chromatin_track

    def _get_gene_expression(self, rna_adata_cell: anndata.AnnData, seq_info: pd.Series) -> np.float32:
        """Get gene expression for a specific gene from a single cell's RNA data."""
        seq_name = seq_info.get("gene")
        # Assuming rna_adata_cell is a view/slice for one cell, use its obs_names if available, else use a placeholder
        cell_id = rna_adata_cell.obs_names[0] if len(rna_adata_cell.obs_names) > 0 else id(rna_adata_cell)

        # Create cache key using gene name and unique cell identifier
        cache_key = (seq_name, cell_id)

        # Return from cache if available
        if cache_key in self.gene_expression_cache:
            return self.gene_expression_cache[cache_key]

        # Return zero if gene expression is in ablation
        if "gene_expression" in self.ablation:
            return np.float32(0)

        gene_expression = np.float32(0)  # Default value
        try:
            # Use pre-indexed gene locations for efficiency
            if seq_name not in self.rna_gene_indices:
                # Don't warn every time, maybe only once per gene? Or rely on mean expr warning.
                # if self.verbose:
                #     warnings.warn(f"Gene {seq_name} not found in RNA AnnData var_names.", UserWarning, stacklevel=2)
                # Keep returning 0, but don't cache the failure? Or cache 0? Caching 0 is fine.
                pass  # gene_expression remains 0
            else:
                gene_idx = self.rna_gene_indices[seq_name]
                # Extract the gene expression directly using the pre-calculated index from the single-cell slice
                # rna_adata_cell is assumed to have 1 observation
                value = rna_adata_cell.X[0, gene_idx]

                # Convert to float32 scalar
                if hasattr(value, "toarray"):  # Handle sparse matrix element
                    value = value.item()  # .item() is efficient for single element extraction
                gene_expression = np.float32(value)

        except (ValueError, IndexError) as e:
            if self.verbose:
                warnings.warn(
                    f"Error retrieving gene expression for {seq_name} in cell {cell_id}: {str(e)}. Returning 0.",
                    UserWarning,
                    stacklevel=2,
                )
            gene_expression = np.float32(0)  # Ensure return 0 on error

        # Cache the result
        self.gene_expression_cache[cache_key] = gene_expression
        return gene_expression

    # _get_mean_expression method remains largely the same, ensure it uses full rna_adata
    def _get_mean_expression(self, seq_info: pd.Series) -> np.float32:
        """Compute mean expression for the given gene across all cells."""
        gene = seq_info.get("gene")

        # Use cache if available
        if gene in self.mean_expr_cache:
            return self.mean_expr_cache[gene]

        if "gene_expression" in self.ablation:  # Check ablation list
            return np.float32(0)

        mean_val = np.float32(0)  # Default value
        if gene not in self.rna_gene_indices:  # Use precomputed index map for check
            if self.verbose:
                warnings.warn(
                    f"Gene {gene} not found in RNA AnnData for mean expression; returning 0.", UserWarning, stacklevel=2
                )
            # Cache the failure (0) to avoid repeated warnings
            self.mean_expr_cache[gene] = mean_val
            return mean_val

        try:
            # Extract vector for all cells using the full rna_adata
            # Use the precomputed index for efficiency
            gene_idx = self.rna_gene_indices[gene]
            vals = self.rna_adata.X[:, gene_idx]  # Efficient column slicing

            # Calculate mean, handling sparse data
            if hasattr(vals, "mean"):  # Use sparse matrix mean if available
                mean_val = np.float32(vals.mean())
            elif hasattr(vals, "toarray"):  # Fallback to dense conversion
                mean_val = np.float32(np.mean(vals.toarray()))
            else:  # Handle dense array
                mean_val = np.float32(np.mean(vals))

        except (IndexError, KeyError, ValueError) as e:  # Catch potential errors during calculation
            if self.verbose:
                warnings.warn(
                    f"Error computing mean expression for gene {gene}: {str(e)}. Returning 0.",
                    UserWarning,
                    stacklevel=2,
                )
            mean_val = np.float32(0)  # Ensure return 0 on error

        # Cache the result (including 0 for errors/missing genes)
        self.mean_expr_cache[gene] = mean_val
        return mean_val

    def __len__(self) -> int:
        """Return the total number of items in the dataset (sequences × cells)."""
        return len(self.seq_meta) * len(self.rna_adata)

    def __getitem__(
        self, idx: int
    ) -> Union[
        tuple[int, int, torch.Tensor, torch.Tensor, torch.Tensor],
        tuple[int, int, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
    ]:
        """
        Get an item from the dataset.

        Args:
            idx (int): The combined index (sequence_idx * num_cells + cell_idx).

        Returns
        -------
            Union[tuple[int, int, torch.Tensor, torch.Tensor, torch.Tensor], tuple[int, int, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]]:
            A tuple containing:
            - seq_idx: Index of the sequence in seq_meta.
            - adata_idx: Index of the cell in rna_adata/atac_adata.
            - one_hot_seq: One-hot encoded DNA sequence (4 x seq_len).
            - chromatin_context: Chromatin accessibility track (1 x seq_len).
            - gene_expression: Gene expression value for the specific gene and cell.
            - mean_gene_expression (optional): Mean expression of the gene across all cells,
                                               if include_mean_expression is True.
        """
        num_cells = len(self.rna_adata)
        if num_cells == 0:
            raise ValueError("rna_adata has zero observations (cells).")

        seq_idx = idx // num_cells
        adata_idx = idx % num_cells

        # Use simplified cache key
        batch_key = (seq_idx, adata_idx)

        # Check batch cache first
        if batch_key in self.batch_cache:
            return self.batch_cache[batch_key]

        seq_info = self._get_seq_info(seq_idx)

        # Get RNA and ATAC data slices for the specific cell (use cache)
        if adata_idx in self.rna_adata_cache:
            rna_adata_selected = self.rna_adata_cache[adata_idx]
            atac_adata_selected = self.atac_adata_cache[adata_idx]
        else:
            # Slicing AnnData creates views, which should be reasonably efficient.
            # Copying might be needed if modifications were intended, but not here.
            rna_adata_selected = self.rna_adata[adata_idx : adata_idx + 1]
            self.rna_adata_cache[adata_idx] = rna_adata_selected
            atac_adata_selected = self.atac_adata[adata_idx : adata_idx + 1]
            self.atac_adata_cache[adata_idx] = atac_adata_selected

        # Get sequence, chromatin context, and gene expression using the single-cell slices
        one_hot_seq = self._tensor_convert(self._get_seq(seq_info, self.ablation), dtype=torch.float32)
        chromatin_context = self._tensor_convert(
            self._get_chromatin_context(atac_adata_selected, seq_info), dtype=torch.float32
        )
        gene_expression = self._tensor_convert(
            self._get_gene_expression(rna_adata_selected, seq_info), dtype=torch.float32
        )

        # Conditionally get and append mean expression
        if self.include_mean_expression:
            mean_expr = self._tensor_convert(self._get_mean_expression(seq_info), dtype=torch.float32)
            result = (seq_idx, adata_idx, one_hot_seq, chromatin_context, gene_expression, mean_expr)
        else:
            result = (seq_idx, adata_idx, one_hot_seq, chromatin_context, gene_expression)

        # Cache the final result tuple with a size limit
        # Consider using a more sophisticated cache like LRU if memory becomes an issue
        if len(self.batch_cache) < 100000:  # Increased cache size slightly
            self.batch_cache[batch_key] = result

        return result


class SeqByGeneExpressionDataset(BaseSeqDataset):
    """
    A PyTorch dataset returning peak DNA sequence, the full gene expression vector for a cell, and the peak's count in that cell.

    Args:
        seq_meta (pd.DataFrame): Metadata for the sequences (peaks). Index must match atac_adata.var_names.
                                 Must contain 'chr', 'start', 'end'. Optionally 'strand'.
        rna_adata (anndata.AnnData): AnnData containing RNA information (cells x genes).
        atac_adata (anndata.AnnData): AnnData containing ATAC information (cells x peaks). Index must match seq_meta.
        seq_len (int, optional): Length of the sequence to extract around the peak center. Defaults to 1314.
        genome_name (str, optional): Name of the genome assembly. Defaults to "hg38".
        genome_dir (Optional[Path], optional): Directory containing the genome files. Defaults to None.
        genome_provider (Optional[str], optional): Provider for genome download. Defaults to None.
        install_genome (bool, optional): Whether to install the genome if not found. Defaults to False.
        verbose (bool, optional): Whether to print verbose messages. Defaults to True.
        ablation (Optional[list[str]], optional): Features to ablate (e.g., 'seq', 'gene_expression', 'peak_count'). Defaults to None.
        include_mean_accessibility (bool, optional): Whether to include mean peak accessibility across all cells. Defaults to False.
        precache_seq (bool, optional): Whether to pre-fetch and cache all DNA sequences on initialization. Defaults to False.
        **kwargs: Additional arguments passed to BaseSeqDataset.
    """

    def __init__(
        self,
        seq_meta: pd.DataFrame,
        rna_adata: anndata.AnnData,
        atac_adata: anndata.AnnData,
        seq_len: int = 1314,
        genome_name: str = "hg38",
        genome_dir: Optional[Path] = None,
        genome_provider: Optional[str] = None,
        install_genome: bool = False,
        verbose: bool = True,
        ablation: Optional[list[str]] = None,
        include_mean_accessibility: bool = False,
        precache_seq: bool = False,  # Added parameter
        **kwargs,
    ):
        # Ensure seq_meta has required columns for BaseSeqDataset coordinate calculation
        required_seq_meta_cols = ["chr", "start", "end"]
        if not all(col in seq_meta.columns for col in required_seq_meta_cols):
            raise ValueError(f"seq_meta must contain columns: {required_seq_meta_cols}")

        super().__init__(
            seq_meta=seq_meta,  # seq_meta here represents peaks
            seq_len=seq_len,
            genome_name=genome_name,
            genome_dir=genome_dir,
            genome_provider=genome_provider,
            install_genome=install_genome,
            verbose=verbose,
            **kwargs,
        )

        # Input validation
        if len(rna_adata) != len(atac_adata):
            raise ValueError(
                f"The RNA ({len(rna_adata)} obs) and ATAC ({len(atac_adata)} obs) "
                "AnnData objects must have the same number of observations (cells)."
            )
        if not seq_meta.index.equals(atac_adata.var_names):
            # Provide more info in the error message
            n_seq = len(seq_meta)
            n_atac_var = len(atac_adata.var_names)
            match_count = seq_meta.index.isin(atac_adata.var_names).sum()
            raise ValueError(
                f"The index of seq_meta ({n_seq} entries) must exactly match the var_names of atac_adata ({n_atac_var} entries). "
                f"Currently, only {match_count} indices match."
            )

        self.rna_adata = rna_adata
        self.atac_adata = atac_adata

        # Convert AnnData objects to memory and to dense matrices
        self.rna_adata = self.rna_adata.to_memory()
        self.atac_adata = self.atac_adata.to_memory()

        # convert to dense if sparse
        if hasattr(self.rna_adata.X, "toarray"):
            self.rna_adata.X = self.rna_adata.X.toarray()
        elif hasattr(self.rna_adata.X, "todense"):
            self.rna_adata.X = np.asarray(self.rna_adata.X.todense())

        if hasattr(self.atac_adata.X, "toarray"):
            self.atac_adata.X = self.atac_adata.X.toarray()
        elif hasattr(self.atac_adata.X, "todense"):
            self.atac_adata.X = np.asarray(self.atac_adata.X.todense())

        self.ablation = ablation if ablation is not None else []
        self.include_mean_accessibility = include_mean_accessibility

        # Pre-computed index for fast peak lookup in atac_adata
        # Use atac_adata.var_names which is guaranteed to match seq_meta.index after validation
        self.atac_peak_indices: dict[str, int] = {peak_name: i for i, peak_name in enumerate(self.atac_adata.var_names)}

        # Cache for mean accessibility
        if self.include_mean_accessibility:
            self.mean_access_cache: dict[int, np.float32] = {}  # Keyed by seq_idx (peak index)

        # Optional pre-caching
        if precache_seq and "seq" not in self.ablation:
            self._precache_seq()  # Base class method works fine here

    def _get_peak_counts(self, adata_idx: int, seq_idx: int) -> np.float32:
        """Get the ATAC count for a specific peak (seq_idx) and cell (adata_idx)."""
        if "peak_count" in self.ablation:
            return np.float32(0)

        peak_count = self.atac_adata.X[adata_idx, seq_idx]

        return peak_count

    def _get_gene_expression_vector(self, adata_idx: int) -> np.ndarray:
        """Get the full gene expression vector for a specific cell (adata_idx)."""
        num_genes = self.rna_adata.shape[1]
        if "gene_expression" in self.ablation:
            # Return a zero vector of the correct shape if gene expression is ablated
            return np.zeros(num_genes, dtype=np.float32)

        # print(self.rna_adata.X)
        gex_vector = self.rna_adata.X[adata_idx, :]

        return gex_vector

    def _get_mean_accessibility(self, seq_idx: int) -> np.float32:
        """Compute mean accessibility for the peak (seq_idx) across all cells."""
        # Use cache if available
        if seq_idx in self.mean_access_cache:
            return self.mean_access_cache[seq_idx]

        if "peak_count" in self.ablation:  # Check ablation list (tied to peak counts)
            return np.float32(0)

        mean_access = np.float32(0)  # Default value
        try:
            # Get peak identifier and its index in atac_adata
            peak_id = self.seq_meta.index[seq_idx]
            peak_adata_var_idx = self.atac_peak_indices[peak_id]

            # Use the full atac_adata to get values across all cells for this peak
            peak_values = self.atac_adata.X[:, peak_adata_var_idx]

            # Calculate mean, handling sparse data efficiently
            if hasattr(peak_values, "mean"):  # Use sparse matrix mean if available
                mean_access = np.float32(peak_values.mean())
            elif hasattr(peak_values, "toarray"):  # Fallback to dense conversion
                mean_access = np.float32(np.mean(peak_values.toarray()))
            else:  # Handle dense array
                mean_access = np.float32(np.mean(peak_values))

        except (IndexError, KeyError, ValueError) as e:
            peak_id_str = f"index {seq_idx}"
            try:
                peak_id_str = self.seq_meta.index[seq_idx]
            except IndexError:
                pass
            if self.verbose:
                warnings.warn(
                    f"Error computing mean accessibility for peak ({peak_id_str}): {str(e)}. Returning 0.",
                    UserWarning,
                    stacklevel=2,
                )
            mean_access = np.float32(0)  # Ensure return 0 on error

        # Cache the result (including 0 for errors)
        self.mean_access_cache[seq_idx] = mean_access
        return mean_access

    def __len__(self) -> int:
        """Return the total number of items in the dataset (sequences/peaks × cells)."""
        return len(self.seq_meta) * len(self.rna_adata)

    def __getitem__(
        self, idx: int
    ) -> Union[
        tuple[int, int, torch.Tensor, torch.Tensor, torch.Tensor],
        tuple[int, int, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
    ]:
        """
        Get an item from the dataset.

        Args:
            idx (int): The combined index (peak_idx * num_cells + cell_idx).

        Returns
        -------
            Union[tuple[int, int, torch.Tensor, torch.Tensor, torch.Tensor], tuple[int, int, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]]:
            A tuple containing:
            - seq_idx: Index of the peak in seq_meta/atac_adata.var.
            - adata_idx: Index of the cell in rna_adata/atac_adata.
            - one_hot_seq: One-hot encoded DNA sequence of the peak region (4 x seq_len).
            - gene_expression_vector: Full gene expression vector for the cell (num_genes,).
            - peak_count: ATAC-seq count for the specific peak and cell.
            - mean_accessibility (optional): Mean accessibility of the peak across all cells,
                                             if include_mean_accessibility is True.
        """
        num_cells = len(self.rna_adata)
        if num_cells == 0:
            raise ValueError("rna_adata has zero observations (cells).")
        num_peaks = len(self.seq_meta)
        if num_peaks == 0:
            raise ValueError("seq_meta has zero entries (peaks).")

        seq_idx = idx // num_cells  # Index of the peak
        adata_idx = idx % num_cells  # Index of the cell

        # seq_info represents the peak's metadata ('chr', 'start', 'end', etc.)
        seq_info = self._get_seq_info(seq_idx)

        # Get sequence centered around the peak coordinates
        one_hot_seq = self._tensor_convert(self._get_seq(seq_info, self.ablation), dtype=torch.float32)

        # Get peak count for this specific peak and cell
        peak_count = self._tensor_convert(self._get_peak_counts(adata_idx, seq_idx), dtype=torch.float32)

        # Get the full gene expression vector for this cell
        gene_expression_vector = self._tensor_convert(self._get_gene_expression_vector(adata_idx), dtype=torch.float32)

        # Conditionally get and append mean accessibility of the peak
        if self.include_mean_accessibility:
            mean_access = self._tensor_convert(self._get_mean_accessibility(seq_idx), dtype=torch.float32)
            result = (seq_idx, adata_idx, one_hot_seq, gene_expression_vector, peak_count, mean_access)
        else:
            result = (seq_idx, adata_idx, one_hot_seq, gene_expression_vector, peak_count)

        return result


class SeqMultiTaskGEXDataset(BaseSeqDataset):
    """
    A PyTorch dataset for multi-task modeling of sequence data and gene expression prediction for different cell states.

    Assumes adata contains expression across different states/conditions as observations.

    Args:
        seq_meta (pd.DataFrame): Metadata for the sequences (e.g., genes/promoters). Must contain 'chr', 'start', 'end', 'gene'.
                                 Optionally 'strand'. Index should correspond to genes.
        adata (anndata.AnnData): AnnData object where obs represent different cell states/conditions
                                 and var represent genes (matching 'gene' in seq_meta).
        seq_len (int, optional): Length of the sequence to extract. Defaults to 1314.
        genome_name (str, optional): Genome assembly name. Defaults to "hg38".
        genome_dir (Optional[Path], optional): Genome directory. Defaults to None.
        genome_provider (Optional[str], optional): Genome provider. Defaults to None.
        install_genome (bool, optional): Install genome if not found. Defaults to False.
        verbose (bool, optional): Verbose output. Defaults to True.
        precache_seq (bool, optional): Pre-cache sequences on init. Defaults to True for this dataset type.
    """

    def __init__(
        self,
        seq_meta: pd.DataFrame,
        adata: anndata.AnnData,
        seq_len: int = 1314,
        genome_name: str = "hg38",
        genome_dir: Optional[Path] = None,
        genome_provider: Optional[str] = None,
        install_genome: bool = False,
        verbose: bool = True,
        precache_seq: bool = True,  # Defaulting to True as seq is primary input
    ):
        # Ensure seq_meta has required columns
        required_seq_meta_cols = ["chr", "start", "end", "gene"]
        if not all(col in seq_meta.columns for col in required_seq_meta_cols):
            raise ValueError(f"seq_meta must contain columns: {required_seq_meta_cols}")

        super().__init__(
            seq_meta=seq_meta,
            seq_len=seq_len,
            genome_name=genome_name,
            genome_dir=genome_dir,
            genome_provider=genome_provider,
            install_genome=install_genome,
            verbose=verbose,
        )

        # Validate that genes in seq_meta exist in adata
        genes_in_seq_meta = set(seq_meta["gene"])
        genes_in_adata = set(adata.var_names)
        missing_genes = genes_in_seq_meta - genes_in_adata
        if missing_genes:
            warnings.warn(
                f"{len(missing_genes)} genes found in 'gene' column of seq_meta are missing from adata.var_names. "
                f"Expression for these genes will be returned as zeros. Missing examples: {list(missing_genes)[:5]}",
                UserWarning,
                stacklevel=2,
            )

        self.adata = adata
        self.num_states = len(adata)  # Number of cell states/conditions

        # Pre-compute gene indices in adata for efficiency
        self.adata_gene_indices: dict[str, int] = {
            gene: adata.var_names.get_loc(gene)
            for gene in adata.var_names
            if gene in genes_in_seq_meta  # Only map genes present in both
        }

        if precache_seq:
            self._precache_seq()

    def __len__(self) -> int:
        # Dataset length is the number of sequences (genes/promoters)
        return len(self.seq_meta)

    def _get_gene_expression_vector_all_states(self, seq_info: pd.Series) -> np.ndarray:
        """Get gene expression for a specific gene across all cell states/conditions in adata."""
        seq_name = seq_info.get("gene")  # Gene name associated with the sequence

        # Initialize zero vector for the case gene is not found or error occurs
        gene_expressions = np.zeros(self.num_states, dtype=np.float32)

        try:
            # Use pre-computed index if gene exists in adata
            if seq_name in self.adata_gene_indices:
                gene_idx = self.adata_gene_indices[seq_name]

                # Extract expression for the gene across all observations (states)
                values = self.adata.X[:, gene_idx]

                # Handle sparse matrices
                if hasattr(values, "toarray"):
                    gene_expressions = values.toarray().flatten().astype(np.float32)
                else:  # Handle dense array
                    gene_expressions = np.asarray(values).flatten().astype(np.float32)
            # else: gene remains missing, return zeros (already initialized)

        except (ValueError, IndexError, KeyError) as e:
            # This error shouldn't happen often due to pre-check and pre-computed indices
            if self.verbose:
                warnings.warn(
                    f"Error retrieving multi-state gene expression for {seq_name}: {str(e)}. Returning zeros.",
                    UserWarning,
                    stacklevel=2,
                )
            # Ensure return vector is zeros
            gene_expressions = np.zeros(self.num_states, dtype=np.float32)

        return gene_expressions

    def __getitem__(self, idx: int) -> tuple[int, torch.Tensor, torch.Tensor]:
        """
        Get an item from the dataset.

        Args:
            idx (int): The index of the sequence in seq_meta.

        Returns
        -------
            tuple[int, torch.Tensor, torch.Tensor]:
            A tuple containing:
            - idx: Index of the sequence.
            - one_hot_seq: One-hot encoded DNA sequence (4 x seq_len).
            - gene_expressions: Gene expression values for this gene across all cell states/conditions (num_states,).
        """
        seq_info = self._get_seq_info(idx)
        one_hot_seq = self._tensor_convert(
            self._get_seq(seq_info), dtype=torch.float32
        )  # No ablation concept here by default
        gene_expressions_vector = self._tensor_convert(
            self._get_gene_expression_vector_all_states(seq_info), dtype=torch.float32
        )

        return idx, one_hot_seq, gene_expressions_vector


class SeqMultiTaskATACDataset(BaseSeqDataset):
    """
    A PyTorch dataset for multi-task modeling of sequence data and chromatin accessibility prediction for different cell states.

    Assumes adata contains accessibility across different states/conditions as observations.

    Args:
        seq_meta (pd.DataFrame): Metadata for the sequences (e.g., peaks/regions). Must contain 'chr', 'start', 'end'.
                                 Optionally 'strand'. Index should correspond to peak identifiers.
        adata (anndata.AnnData): AnnData object where obs represent different cell states/conditions
                                 and var represent peaks (matching index of seq_meta).
        seq_len (int, optional): Length of the sequence to extract. Defaults to 1314.
        genome_name (str, optional): Genome assembly name. Defaults to "hg38".
        genome_dir (Optional[Path], optional): Genome directory. Defaults to None.
        genome_provider (Optional[str], optional): Genome provider. Defaults to None.
        install_genome (bool, optional): Install genome if not found. Defaults to False.
        verbose (bool, optional): Verbose output. Defaults to True.
        precache_seq (bool, optional): Pre-cache sequences on init. Defaults to True for this dataset type.
    """

    def __init__(
        self,
        seq_meta: pd.DataFrame,
        adata: anndata.AnnData,
        seq_len: int = 1314,
        genome_name: str = "hg38",
        genome_dir: Optional[Path] = None,
        genome_provider: Optional[str] = None,
        install_genome: bool = False,
        verbose: bool = True,
        precache_seq: bool = True,  # Defaulting to True as seq is primary input
    ):
        # Ensure seq_meta has required columns
        required_seq_meta_cols = ["chr", "start", "end"]
        if not all(col in seq_meta.columns for col in required_seq_meta_cols):
            raise ValueError(f"seq_meta must contain columns: {required_seq_meta_cols}")

        super().__init__(
            seq_meta=seq_meta,
            seq_len=seq_len,
            genome_name=genome_name,
            genome_dir=genome_dir,
            genome_provider=genome_provider,
            install_genome=install_genome,
            verbose=verbose,
        )

        # Validate that peak indices in seq_meta exist in adata
        peaks_in_seq_meta = set(seq_meta.index)
        peaks_in_adata = set(adata.var_names)
        missing_peaks = peaks_in_seq_meta - peaks_in_adata
        if missing_peaks:
            warnings.warn(
                f"{len(missing_peaks)} peaks found in seq_meta index are missing from adata.var_names. "
                f"Accessibility for these peaks will be returned as zeros. Missing examples: {list(missing_peaks)[:5]}",
                UserWarning,
                stacklevel=2,
            )

        self.adata = adata
        self.num_states = len(adata)  # Number of cell states/conditions

        # Pre-compute peak indices in adata for efficiency
        self.adata_peak_indices: dict[str, int] = {
            peak: adata.var_names.get_loc(peak)
            for peak in adata.var_names
            if peak in peaks_in_seq_meta  # Only map peaks present in both
        }

        if precache_seq:
            self._precache_seq()

    def __len__(self) -> int:
        # Dataset length is the number of sequences (peaks/regions)
        return len(self.seq_meta)

    def _get_accessibility_vector_all_states(self, seq_idx: int) -> np.ndarray:
        """Get accessibility for a specific peak across all cell states/conditions in adata."""
        # Get peak ID from the seq_meta index
        try:
            peak_id = self.seq_meta.index[seq_idx]
        except IndexError:
            if self.verbose:
                warnings.warn(
                    f"Peak index {seq_idx} out of bounds for seq_meta with {len(self.seq_meta)} entries. Returning zeros.",
                    UserWarning,
                    stacklevel=2,
                )
            return np.zeros(self.num_states, dtype=np.float32)

        # Initialize zero vector for the case peak is not found or error occurs
        accessibility_values = np.zeros(self.num_states, dtype=np.float32)

        try:
            # Use pre-computed index if peak exists in adata
            if peak_id in self.adata_peak_indices:
                peak_idx = self.adata_peak_indices[peak_id]

                # Extract accessibility for the peak across all observations (states)
                values = self.adata.X[:, peak_idx]

                # Handle sparse matrices
                if hasattr(values, "toarray"):
                    accessibility_values = values.toarray().flatten().astype(np.float32)
                else:  # Handle dense array
                    accessibility_values = np.asarray(values).flatten().astype(np.float32)
            # else: peak remains missing, return zeros (already initialized)

        except (ValueError, IndexError, KeyError) as e:
            # This error shouldn't happen often due to pre-check and pre-computed indices
            if self.verbose:
                warnings.warn(
                    f"Error retrieving multi-state accessibility for peak {peak_id}: {str(e)}. Returning zeros.",
                    UserWarning,
                    stacklevel=2,
                )
            # Ensure return vector is zeros
            accessibility_values = np.zeros(self.num_states, dtype=np.float32)

        return accessibility_values

    def __getitem__(self, idx: int) -> tuple[int, torch.Tensor, torch.Tensor]:
        """
        Get an item from the dataset.

        Args:
            idx (int): The index of the sequence in seq_meta.

        Returns
        -------
            tuple[int, torch.Tensor, torch.Tensor]:
            A tuple containing:
            - idx: Index of the sequence.
            - one_hot_seq: One-hot encoded DNA sequence (4 x seq_len).
            - accessibility_values: Accessibility values for this peak across all cell states/conditions (num_states,).
        """
        seq_info = self._get_seq_info(idx)
        one_hot_seq = self._tensor_convert(
            self._get_seq(seq_info), dtype=torch.float32
        )  # No ablation concept here by default
        accessibility_vector = self._tensor_convert(self._get_accessibility_vector_all_states(idx), dtype=torch.float32)

        return idx, one_hot_seq, accessibility_vector
