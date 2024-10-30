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

        if install_genome:
            self.genome = genomepy.install_genome(genome_name, genome_provider, genomes_dir=genome_dir, **kwargs)
        else:
            self.genome = genomepy.Genome(genome_name, genomes_dir=genome_dir)

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

        """
        seq = str(self.genome.get_seq(chr, start, end))
        return seq.upper()

    def chr_sizes(self, chr: str) -> int:
        """
        Returns the chromosome sizes of the genome.

        Returns
        -------
            dict: The chromosome sizes.

        """
        return self.genome.sizes[chr]
