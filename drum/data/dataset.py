import warnings
from pathlib import Path
from typing import Optional

import anndata
import numpy as np
import pandas as pd
from torch.utils.data import Dataset
from tqdm import tqdm

from drum.tl import onehot_encoding

from .genome import Genome


class SeqDataset(Dataset):
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
        seq_meta,
        seq_len: int = 1314,
        genome_name: str = "hg38",
        genome_dir: Optional[Path] = None,
        genome_provider: Optional[str] = None,
        install_genome: bool = True,
        **kwargs,
    ):
        self.genome = Genome(
            genome_name=genome_name,
            genome_dir=genome_dir,
            genome_provider=genome_provider,
            install_genome=install_genome,
            **kwargs,
        )
        self.seq_meta = seq_meta
        self.seq_len = seq_len

    def __len__(self):
        return len(self.seq_meta)

    def __getitem__(self, idx):
        seq_info = self.seq_meta.iloc[idx]

        chr = seq_info["chr"]
        region_mid = (seq_info["start"] + seq_info["end"]) // 2
        seq_start = region_mid - self.seq_len // 2
        seq_end = seq_start + self.seq_len

        seq = self.genome.get_seq(chr, seq_start, seq_end)
        one_hot_seq = onehot_encoding(seq)
        return one_hot_seq


class SeqByChromatinContextDataset(Dataset):
    """
    A PyTorch dataset for generating sequences and their corresponding chromatin context.

    Args:
        seq_meta (pd.DataFrame): Metadata for the sequences.
        adata (anndata.AnnData):  anndata containing chromatin information.
        seq_len (int, optional): Length of the sequence. Defaults to 1314.

    Returns
    -------
        tuple: A tuple containing the sequence index, adata index, one-hot encoded sequence, and chromatin context.
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
        ablation: list = [],
    ):
        self.rna_adata = rna_adata
        self.atac_adata = atac_adata
        # check if the rna_adata and atac_adata have the same number of observations

        if len(self.rna_adata) != len(self.atac_adata):
            raise ValueError("The RNA and ATAC AnnData objects must have the same number of observations.")
        self.verbose = verbose
        self.seq_meta = seq_meta
        self.seq_len = int(seq_len)
        self.genome_name = genome_name
        self.genome_dir = genome_dir
        self.ablation = ablation
        self.genome = Genome(
            genome_name=genome_name,
            genome_dir=genome_dir,
            genome_provider=genome_provider,
            install_genome=install_genome,
        )
        self.seq_cache = {}
        self.rna_adata_cache = {}
        self.atac_adata_cache = {}
        self.seq_info_cache = {}

        self._precache_seq()

    def __len__(self):
        return len(self.seq_meta) * len(self.rna_adata)

    def _get_seq(self, seq_info):
        if 'seq' in self.ablation:
            # return a zero sequence
            return np.zeros((4, self.seq_len), dtype=np.float32)

        # FIXME: considering the range of chr and seq_start, seq_end
        chr = seq_info["chr"]

        # TODO: update with the strand information

        region_mid = int((seq_info["start"] + seq_info["end"]) // 2)
        seq_start = int(region_mid - self.seq_len // 2)
        seq_end = int(seq_start + self.seq_len)

        # print(self.genome.chr_sizes(chr))
        chr_len = self.genome.chr_sizes(chr)

        # Adjust seq_start and seq_end if they go out of bounds
        if seq_start < 1:  # Adjust for 1-based coordinate system
            seq_start = 1
            seq_end = min(self.seq_len, chr_len)
        elif seq_end > chr_len:
            seq_end = chr_len
            seq_start = max(1, chr_len - self.seq_len + 1)

        cache_key = f"{chr}:{seq_start}-{seq_end}"

        if cache_key in self.seq_cache:
            # Retrieve the sequence from cache if it exists
            seq = self.seq_cache[cache_key]
        else:
            # If not in cache, fetch the sequence and store it in the cache
            seq = self.genome.get_seq(chr, seq_start, seq_end - 1)
            self.seq_cache[cache_key] = seq  # Store the sequence in cache

        one_hot_seq = onehot_encoding(seq)

        # Check the length of the one-hot encoded sequence and pad if necessary
        current_len = one_hot_seq.shape[1]
        if current_len < self.seq_len:
            # Padding with zeros along the sequence length dimension
            padding = np.zeros((one_hot_seq.shape[0], self.seq_len - current_len))
            one_hot_seq = np.hstack((one_hot_seq, padding))

        return one_hot_seq

    def _precache_seq(self):
        print("Prefetching sequences...")
        for idx in tqdm(range(len(self.seq_meta))):
            seq_info = self.seq_meta.iloc[idx]
            self._get_seq(seq_info)
        print("Prefetching done.")

    def _get_chromatin_context(self, atac_adata, seq_info):
        seq_name = seq_info["gene"]
        chr = seq_info["chr"]
        region_mid = int((seq_info["start"] + seq_info["end"]) // 2)
        seq_start = int(region_mid - self.seq_len // 2)
        seq_end = int(seq_start + self.seq_len)

        region_in_range = atac_adata.var[
            (atac_adata.var["chr"] == chr) & (atac_adata.var["start"] < seq_end) & (atac_adata.var["end"] > seq_start)
        ]

        chromatin_track = np.zeros((1, int(seq_end - seq_start)), dtype=np.float32)

        if 'chromatin' in self.ablation:
            return chromatin_track
        
        if (len(region_in_range) == 0) and self.verbose:
            warnings.warn(
                f"No chromatin context found for {seq_name} within the range {chr}:{seq_start}-{seq_end}",
                UserWarning,
                stacklevel=2,
            )

        # Extract the entire data for the required indices at once
        values = atac_adata[:, region_in_range.index].X[0, :]
        values = values.toarray().flatten()

        for row, value in zip(region_in_range.itertuples(), values):
            peak_start = row.start
            peak_end = row.end

            peak_start_adj = int(max(peak_start, seq_start) - seq_start)
            peak_end_adj = int(min(peak_end, seq_end) - seq_start)

            chromatin_track[0, peak_start_adj:peak_end_adj] += value / (peak_end - peak_start)

        return chromatin_track

    def _get_gene_expression(self, rna_adata, seq_info):
        seq_name = seq_info.get("gene")
        if seq_name not in rna_adata.var_names:
            raise ValueError(f"Gene {seq_name} not found in RNA AnnData object.")

        # Extract the gene expression for the required gene
        gene_expression = rna_adata[:, seq_name].X[0, 0]
        # make sure it's float32
        gene_expression = np.array(gene_expression, dtype=np.float32)

        return gene_expression

    def __getitem__(self, idx):
        seq_idx = idx // len(self.rna_adata)
        adata_idx = idx % len(self.rna_adata)

        if seq_idx in self.seq_info_cache:
            # Retrieve the seq_info from cache if it exists
            seq_info = self.seq_info_cache[seq_idx]
        else:
            # If not in cache, fetch the seq_info and store it in the cache
            seq_info = self.seq_meta.iloc[seq_idx]
            self.seq_info_cache[seq_idx] = seq_info

        if adata_idx in self.rna_adata_cache:
            # Retrieve the adata from cache if it exists
            rna_adata_selected = self.rna_adata_cache[adata_idx]
            atac_adata_selected = self.atac_adata_cache[adata_idx]
        else:
            # If not in cache, fetch the adata and store it in the cache
            rna_adata_selected = self.rna_adata[adata_idx]
            self.rna_adata_cache[adata_idx] = rna_adata_selected

            atac_adata_selected = self.atac_adata[adata_idx]
            self.atac_adata_cache[adata_idx] = atac_adata_selected

        chromatin_context = self._get_chromatin_context(atac_adata_selected, seq_info)

        gene_expression = self._get_gene_expression(rna_adata_selected, seq_info)

        one_hot_seq = self._get_seq(seq_info)

        return seq_idx, adata_idx, one_hot_seq, chromatin_context, gene_expression


class SeqMultiTaskGEXDataset(Dataset):
    """
    A PyTorch dataset for multi-task modeling of sequence data and gene expression prediction for different cell states.

    Args:
        seq_meta (pd.DataFrame): Metadata for the sequences.
        adata (anndata.AnnData): AnnData object containing gene expression data across multiple cell states.
        seq_len (int, optional): Length of the sequence. Defaults to 1314.

    Returns
    -------
        tuple: A tuple containing the sequence index, one-hot encoded sequence, and gene expressions for all cell states.
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
    ):
        self.adata = adata
        self.verbose = verbose
        self.seq_meta = seq_meta
        self.seq_len = int(seq_len)
        self.genome_name = genome_name
        self.genome_dir = genome_dir
        self.genome = Genome(
            genome_name=genome_name,
            genome_dir=genome_dir,
            genome_provider=genome_provider,
            install_genome=install_genome,
        )
        self.seq_cache = {}
        self.seq_info_cache = {}

        self._precache_seq()

    def __len__(self):
        return len(self.seq_meta)

    def _get_seq(self, seq_info):
        chr = seq_info["chr"]
        region_mid = int((seq_info["start"] + seq_info["end"]) // 2)
        seq_start = int(region_mid - self.seq_len // 2)
        seq_end = int(seq_start + self.seq_len)

        chr_len = self.genome.chr_sizes(chr)

        # Adjust seq_start and seq_end if they go out of bounds
        if seq_start < 1:
            seq_start = 1
            seq_end = min(self.seq_len, chr_len)
        elif seq_end > chr_len:
            seq_end = chr_len
            seq_start = max(1, chr_len - self.seq_len + 1)

        cache_key = f"{chr}:{seq_start}-{seq_end}"

        if cache_key in self.seq_cache:
            seq = self.seq_cache[cache_key]
        else:
            seq = self.genome.get_seq(chr, seq_start, seq_end - 1)
            self.seq_cache[cache_key] = seq

        one_hot_seq = onehot_encoding(seq)

        # Ensure the sequence length matches the desired length
        current_len = one_hot_seq.shape[1]
        if current_len < self.seq_len:
            padding = np.zeros((one_hot_seq.shape[0], self.seq_len - current_len))
            one_hot_seq = np.hstack((one_hot_seq, padding))

        return one_hot_seq

    def _precache_seq(self):
        print("Prefetching sequences...")
        for idx in tqdm(range(len(self.seq_meta))):
            seq_info = self.seq_meta.iloc[idx]
            self._get_seq(seq_info)
        print("Prefetching done.")

    def _get_gene_expression(self, adata, seq_info):
        seq_name = seq_info["gene"]

        if seq_name not in adata.var_names:
            raise ValueError(f"Gene {seq_name} not found in AnnData object.")

        # Extract gene expression for all cell states for the given gene
        gene_expression = adata[:, seq_name].X[:,0].toarray().flatten()

        gene_expression = np.array(gene_expression, dtype=np.float32)

        return gene_expression

    def __getitem__(self, idx):
        seq_info = self.seq_meta.iloc[idx]

        if idx in self.seq_info_cache:
            seq_info = self.seq_info_cache[idx]
        else:
            seq_info = self.seq_meta.iloc[idx]
            self.seq_info_cache[idx] = seq_info

        one_hot_seq = self._get_seq(seq_info)
        gene_expressions = self._get_gene_expression(self.adata, seq_info)

        return idx, one_hot_seq, gene_expressions
