from collections import abc


def make_divisible(value, divisor, min_value=None, min_ratio=0.9):
    """Make divisible function.

    This function rounds the channel number down to the nearest value that can
    be divisible by the divisor.

    Parameters
    ----------
    value : int
        The original channel number.
    divisor : int
        The divisor to fully divide the channel number.
    min_value : int, optional
        The minimum value of the output channel.
        Default: None, means that the minimum value equal to the divisor.
    min_ratio : float, optional
        The minimum ratio of the rounded channel
        number to the original channel number. Default: 0.9.
    Returns
    -------
    int
        The modified output channel number
    """

    if min_value is None:
        min_value = divisor
    new_value = max(min_value, int(value + divisor / 2) // divisor * divisor)
    if new_value < min_ratio * value:
        new_value += divisor
    return new_value


def is_seq_of(seq, expected_type, seq_type=None):
    """Check whether it is a sequence of some type.

    Parameters
    ----------
    seq : Sequence
        The sequence to be checked.
    expected_type : type
        Expected type of sequence items.
    seq_type : type
        Expected sequence type.

    Returns
    -------
    out : bool
        Whether the sequence is valid.
    """
    if seq_type is None:
        expect_seq_type = abc.Sequence
    else:
        assert isinstance(seq_type, type)
        expect_seq_type = seq_type
    if not isinstance(seq, expect_seq_type):
        return False
    for item in seq:
        if not isinstance(item, expected_type):
            return False
    return True