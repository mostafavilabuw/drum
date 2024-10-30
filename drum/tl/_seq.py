import numba
import numpy
import torch


# credit to https://github.com/jmschrei/tangermeme/
@numba.njit("void(int8[:, :], int8[:], int8[:])")
def _fast_one_hot_encode(X_ohe, seq, mapping):
    """An internal function for quickly converting bytes to one-hot indexes."""
    for i in range(len(seq)):
        idx = mapping[seq[i]]
        if idx == -1:
            continue

        if idx == -2:
            raise ValueError("Encountered character that is not in " + "`alphabet` or in `ignore`.")

        X_ohe[i, idx] = 1


def onehot_encoding(sequence, alphabet="ACGT", dtype=torch.float16, ignore="N", desc=None, verbose=False, **kwargs):
    """Converts a string or list of characters into a one-hot encoding.

    This function will take in either a string or a list and convert it into a
    one-hot encoding. If the input is a string, each character is assumed to be
    a different symbol, e.g. 'ACGT' is assumed to be a sequence of four
    characters. If the input is a list, the elements can be any size.

    Although this function will be used here primarily to convert nucleotide
    sequences into one-hot encoding with an alphabet of size 4, in principle
    this function can be used for any types of sequences.

    Parameters
    ----------
    sequence : str or list
            The sequence to convert to a one-hot encoding.

    alphabet : set or tuple or list
            A pre-defined alphabet where the ordering of the symbols is the same
            as the index into the returned tensor, i.e., for the alphabet ['A', 'B']
            the returned tensor will have a 1 at index 0 if the character was 'A'.
            Characters outside the alphabet are ignored and none of the indexes are
            set to 1. Default is ['A', 'C', 'G', 'T'].

    dtype : str or torch.dtype, optional
            The data type of the returned encoding. Default is int8.

    ignore: list, optional
            A list of characters to ignore in the sequence, meaning that no bits
            are set to 1 in the returned one-hot encoding. Put another way, the
            sum across characters is equal to 1 for all positions except those
            where the original sequence is in this list. Default is ['N'].


    Returns
    -------
    ohe : numpy.ndarray
            A binary matrix of shape (alphabet_size, sequence_length) where
            alphabet_size is the number of unique elements in the sequence and
            sequence_length is the length of the input sequence.
    """
    for char in ignore:
        if char in alphabet:
            raise ValueError(f"Character {char} in the alphabet " + "and also in the list of ignored characters.")

    if isinstance(alphabet, list):
        alphabet = "".join(alphabet)

    if isinstance(ignore, list):
        ignore = "".join(ignore)

    seq_idxs = numpy.frombuffer(bytearray(sequence, "utf8"), dtype=numpy.int8)
    alpha_idxs = numpy.frombuffer(bytearray(alphabet, "utf8"), dtype=numpy.int8)
    ignore_idxs = numpy.frombuffer(bytearray(ignore, "utf8"), dtype=numpy.int8)

    one_hot_mapping = numpy.zeros(256, dtype=numpy.int8) - 2
    for i, idx in enumerate(alpha_idxs):
        one_hot_mapping[idx] = i

    for idx in ignore_idxs:
        one_hot_mapping[idx] = -1

    n, m = len(sequence), len(alphabet)

    one_hot_encoding = numpy.zeros((n, m), dtype=numpy.int8)
    _fast_one_hot_encode(one_hot_encoding, seq_idxs, one_hot_mapping)
    return torch.from_numpy(one_hot_encoding).type(dtype).T
