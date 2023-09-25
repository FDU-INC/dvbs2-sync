"""
A brief implementation for multicoset sampling scheme
"""

import numpy as np


def sample(
    signal,
    time: float,
    bandwidth: float,
    nBands: int,
    nChannels: int,
    offsets: list[int] | None = None,
    scheme="continuous",
):
    """
    #### parameters:

    - `signal`: input signals. When `scheme` is 'continuous', `signal` is a
      function of time, Reals to Complex. When `scheme` is 'discrete', `signal`
      is a numpy.ndarray containing discrete signals, whose sampling frequency
      is indicated by `bandwidth`
    - `time`: sampling time duration, only used when `scheme` is 'continuous'
    - `bandwidth`: bandwidth W of the input signal. When `scheme` is
      'contiuous', `bandwidth` is viewed as the sampling frequency of the input
      signal
    - `nBands`: Number of bands L. The cognitive radio system divides the
      shared spectrum into L orthogonal channels, so the bandwidth of each
      channel is B = W/L. The sampling interval for each band is `nBands` times
      input discrete signal sampling interval
    - `nChannels`: number of parallel channels for sampling
    - `offsets`: offsets of each sampling channel, integers
    - `scheme`: 'continuous' by default. When set as 'discrete', use
      `bandwidth` to indicate the sampling frequency of the input signal

    #### returns:

    1. A numpy.ndarray. Each row is the sub-nyquist sampled signal sequence
    with time interval L/W and time offset c_i/W, where {c_i, i = 1, 2, ...}
    are non-negative integers
    2. offsets
    """
    if nChannels > nBands:
        raise ValueError("nBands should be greater than or equal to nChannels")

    if offsets is not None:
        if len(offsets) < nChannels:
            raise ValueError(
                "length of offsets should not be less than sampling channels"
            )
        elif np.max(offsets) >= nBands or np.min(offsets) < 0:
            raise ValueError("offset value should be in [0, L-1]")

        offsets = list(offsets[:nChannels])
    else:
        # construct offsets randomly
        rng = np.random.RandomState(43)
        pool = np.arange(0, nBands, 1)
        # np.random.shuffle(pool)
        rng.shuffle(pool)
        offsets = list(pool[0:nChannels])

    samplingRes = np.zeros((1))
    if scheme == "continuous":
        T = 1 / bandwidth
        Ts = T * nBands
        samplingRes = np.zeros((nChannels, int(np.floor(time / Ts))), dtype=np.cdouble)
        for i in range(nChannels):
            for j in range(len(samplingRes[i])):
                samplingRes[i][j] = signal(j * Ts + offsets[i] * T)

    elif scheme == "discrete":
        col = int(np.floor(len(signal) / nBands))
        samplingRes = np.zeros((nChannels, col), dtype=np.cdouble)
        for i in range(nChannels):
            samplingRes[i] = signal[offsets[i] :: nBands][:col]
    else:
        raise NotImplementedError("scheme not supported")

    return samplingRes, offsets


def dft(signals: np.ndarray, offsets: list[int], nBands: int):
    """
    Transforms signals sampled by multicoset method into frequency domain

    #### returns

    1.Multicoset sampling results in frequency domain, say, Y
    2. The measurement matrix A with shape(nChannels, nBands)
    """
    # DTFT * Ts = DTF, so the coeffecient 1/LT will be eleminated
    # To get matrix Y, we need to use the coeffecients to time the fft results.

    Y_aux = np.matrix(offsets).T * np.matrix(np.arange(0, signals.shape[1], 1))
    YCoeff = np.exp(-2j * np.pi / (nBands * signals.shape[1]) * Y_aux)

    # amplitudes got by numpy.fft.fft are N times larger than actual amplitudes
    # in frequency domain. We don't shrink here considering float point
    # arithmetic errors
    Y = np.matrix(np.multiply(YCoeff, np.fft.fft(signals)))

    A_aux = np.matrix(offsets).T * np.matrix(np.arange(0, nBands, 1))
    A = np.matrix(np.exp(2j * np.pi * A_aux / nBands))

    # to compensate the amplitudes caused by numpy.fft.fft(signals)
    A = signals.shape[1] * A
    return Y, A


def get_correlation(A: np.matrix) -> float:
    corr = A.H * A

    for i in range(corr.shape[0]):
        corr[i, i] = 0

    return np.max(np.abs(corr)) / (np.linalg.norm(A) ** 2)
