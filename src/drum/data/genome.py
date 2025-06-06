from pathlib import Path
from typing import Optional

import genomepy


class Genome:
    """
    Represents a genome object.

    Args:
        genome_name (str): The name of the genome. Default is "hg38".
        genome_dir (Optional[Path]): The directory where the genome is stored. If not provided, it will be set to the default directory.
        genome_provider (Optional[str]): The provider of the genome.
        install_genome (bool): Whether to install the genome. Default is True.

    Attributes
    ----------
        genome_name (str): The name of the genome.
        genome_dir (Path): The directory where the genome is stored.
        genome_provider (str): The provider of the genome.
        genome: The genome object.

    Methods
    -------
        get_seq: Retrieves the sequence from the genome.
        chr_sizes: Returns the size of a specific chromosome.
    """

    def __init__(
        self,
        genome_name: str = "hg38",
        genome_dir: Optional[Path] = None,
        genome_provider: Optional[str] = None,
        install_genome: bool = True,
        **kwargs,
    ):
        self.genome_name = genome_name
        self.genome_dir = genome_dir if genome_dir else Path.home() / "data" / "genomes"
        self.genome_provider = genome_provider

        try:
            if install_genome:
                self.genome = genomepy.install_genome(genome_name, genome_provider, genomes_dir=genome_dir, **kwargs)
            else:
                self.genome = genomepy.Genome(genome_name, genomes_dir=genome_dir)
        except (OSError, ValueError, RuntimeError) as e:
            raise RuntimeError(f"Failed to initialize genome {genome_name}: {str(e)}") from e

    def get_seq(self, chr: str, start: int, end: int) -> str:
        """
        Retrieves the sequence from the genome.

        Args:
            chr (str): The chromosome name.
            start (int): The start position of the sequence.
            end (int): The end position of the sequence.

        Returns
        -------
            str: The sequence.

        Raises
        ------
            ValueError: If the chromosome doesn't exist or if the coordinates are invalid.
        """
        if chr not in self.genome.sizes:
            raise ValueError(f"Chromosome {chr} not found in genome {self.genome_name}")

        if start < 0 or end < start:
            raise ValueError(f"Invalid coordinates: start={start}, end={end}")

        chr_size = self.genome.sizes[chr]
        if start > chr_size:
            raise ValueError(f"Start position {start} exceeds chromosome {chr} size {chr_size}")

        try:
            seq = str(self.genome.get_seq(chr, start, end))
            return seq.upper()
        except (ValueError, IndexError, KeyError) as e:
            raise ValueError(f"Failed to retrieve sequence for {chr}:{start}-{end}: {str(e)}") from e

    def chr_sizes(self, chr: str) -> int:
        """
        Returns the size of a specific chromosome.

        Args:
            chr (str): The chromosome name.

        Returns
        -------
            int: The size of the chromosome.

        Raises
        ------
            ValueError: If the chromosome doesn't exist.
        """
        if chr not in self.genome.sizes:
            raise ValueError(f"Chromosome {chr} not found in genome {self.genome_name}")

        return self.genome.sizes[chr]

    def get_all_chr_sizes(self) -> dict[str, int]:
        """
        Returns sizes of all chromosomes in the genome.

        Returns
        -------
            dict[str, int]: Dictionary with chromosome names as keys and sizes as values.
        """
        return self.genome.sizes
