import numpy as np


class SqrtRaisedCos:
    def __init__(self, fsymb: float, rolloff: float):
        self.f = fsymb
        self.alpha = rolloff

    def __call__(self, t) -> float:
        return self.time(t)

    def time(self, t) -> float:
        res = 0
        if t == 0:
            # res = (1 + self.alpha * (4 / np.pi - 1)) * self.f
            res = 1 - self.alpha + 4 * self.alpha / np.pi
        elif t == 1 / (4 * self.alpha * self.f) or t == -1 / (4 * self.alpha * self.f):
            res = (
                self.alpha
                # * self.f
                / np.sqrt(2)
                * (
                    (1 + 2 / np.pi) * np.sin(np.pi / (4 * self.alpha))
                    + (1 - 2 / np.pi) * np.cos(np.pi / (4 * self.alpha))
                )
            )
        else:
            term1 = np.sin(np.pi * self.f * t * (1 - self.alpha))
            term2 = (
                4
                * self.alpha
                * self.f
                * t
                * np.cos(np.pi * self.f * t * (1 + self.alpha))
            )
            term3 = np.pi * t * self.f * (1 - (4 * self.alpha * t * self.f) ** 2)

            res = (term1 + term2) / term3

        return res

    def fre(self, f: float) -> float:
        unsqrt = 0.0
        f_cmp = np.abs(f) * 2 / self.f
        if f_cmp <= (1 - self.alpha):
            unsqrt = 1 / self.f
        elif f_cmp > (1 - self.alpha) and f_cmp <= (1 - self.alpha):
            unsqrt = (
                1
                / (2 * self.f)
                * (1 - np.sin(np.pi * (f - self.f / 2) / (self.alpha * self.f)))
            )

        return np.sqrt(unsqrt)


class SqrtRaisedCosFilter:
    def __init__(
        self, fsymb: float, rolloff: float, fsamp: float, window_sz_in_symbols: int = 10
    ):
        self.sqrt_raised_cos = SqrtRaisedCos(fsymb, rolloff)
        self.window_sz = window_sz_in_symbols * int(fsamp / fsymb + 0.5)
        self.__fsymb = fsymb
        self.__fsamp = fsamp
        self.__rolloff = rolloff

        parity = self.window_sz % 2
        pulse_len = self.window_sz + 1 - parity
        self.__pulse_len = pulse_len
        self.__pulse = np.zeros(pulse_len, dtype=np.cdouble)

        # len(self.__pulse) is 2 * half + 1
        half = int(len(self.__pulse) / 2)
        self.__pulse = np.array(
            [self.sqrt_raised_cos((i - half) / fsamp) for i in range(pulse_len)]
        )

    def filter(self, input: np.ndarray) -> np.ndarray:
        return np.convolve(input, self.__pulse, "same")

    def filter_matlab(self, input: np.ndarray) -> np.ndarray:
        signal = np.convolve(input, self.__pulse, "full")
        return signal[: len(input)]

    def pulse_shape(self, symbols: np.ndarray, fc: float):
        sps = int(self.fsamp / self.fsymb)
        signal = np.zeros(len(symbols) * sps, dtype=np.cdouble)
        signal[::sps] = symbols
        signal = np.convolve(signal, self.__pulse, "full")

        start = int((self.__pulse_len - sps) / 2)
        signal = signal[start : (start + len(symbols) * sps)]
        sig_shifted = np.multiply(
            signal, np.exp(2j * np.pi * fc / self.fsamp * np.arange(len(signal)))
        )
        return sig_shifted

    @property
    def fsymb(self):
        return self.__fsymb

    @property
    def fsamp(self):
        return self.__fsamp

    @property
    def rolloff(self):
        return self.__rolloff

    @property
    def pulse(self):
        return self.__pulse

    @property
    def pulse_len(self):
        return self.__pulse_len


class SqrtRaisedCosFilterMat(SqrtRaisedCosFilter):
    """
    behaves the same as comm.RaisedCosineReceiveFilter in MATLAB
    """

    def __init__(
        self, fsymb: float, rolloff: float, fsamp: float, window_sz_in_symbols: int = 10
    ):
        super().__init__(fsymb, rolloff, fsamp, window_sz_in_symbols)
        self.__pulse = super().pulse
        self.__pulse = self.__pulse / np.linalg.norm(self.__pulse, 2)
        self.__pulse_len = super().pulse_len
        self.gain = 1.0
        self.reset()

    def reset(self):
        self.buffer = np.zeros(self.__pulse_len, dtype=np.cdouble)

    def set_gain(self, gain: float):
        self.gain = gain

    def __call__(self, x: np.ndarray[int, np.dtype[np.cdouble]] | np.cdouble):
        x = np.atleast_1d(np.array(x, dtype=np.cdouble))
        assert len(x.shape) == 1

        unfiltered = np.concatenate((self.buffer, x), axis=-1)
        filtered = self.gain * np.convolve(unfiltered, self.__pulse, "full")
        buf_len = self.__pulse_len
        res = filtered[buf_len : buf_len + len(x)]
        self.buffer = unfiltered[-buf_len:]

        return res

    @property
    def pulse(self):
        return self.__pulse


def load_matlab(path: str, ratio=1.0):
    def rep(s):
        return complex(s.decode().replace("i", "j"))

    s = np.loadtxt(
        path,
        dtype=np.cdouble,
        converters={0: rep},
    )

    sz = int(np.floor(ratio * len(s)))

    return np.array(s[0:sz])


def addNoise(rawS: np.ndarray, noise=0.0, random_seed=1):
    if noise == 0.0:
        # print("No noise signals added, SNR=inf")
        return rawS, np.inf

    # print("Adding noise")
    sz = len(rawS)
    sPsqrt = np.linalg.norm(rawS)

    # n = np.random.normal(size=(sz), scale=np.sqrt(noise/2)) + \
    #         1j * np.random.normal(size=(sz), scale=np.sqrt(noise/2))

    rng = np.random.RandomState(random_seed)
    n = rng.normal(size=(sz), scale=np.sqrt(noise / 2)) + 1j * rng.normal(
        size=(sz), scale=np.sqrt(noise / 2)
    )

    nPsqrt = np.linalg.norm(n)
    snr = 20 * np.log(sPsqrt / nPsqrt)
    # print("SNR:", snr)

    return rawS + n, snr


def awgn(pure: np.ndarray, snr: float | None, random_seed=1):
    if snr is None:
        return pure

    sz = len(pure)
    # sig_p_sqrt = np.linalg.norm(pure) / np.sqrt(sz)
    sig_p_sqrt = np.sqrt(np.mean(np.abs(pure) ** 2))
    noise_scale = sig_p_sqrt / (10 ** (snr / 20)) / np.sqrt(2)

    rng = np.random.RandomState(random_seed)
    n = rng.normal(size=(sz), scale=1.0) + 1j * rng.normal(
        size=(sz),
        scale=1.0,
    )
    n = n * noise_scale

    return pure + n


def sinc_filter(
    signal: np.ndarray,
    fsamp: float,
    f_center: float,
    bandwidth: float,
    window_sz: int = 10,
) -> np.ndarray:
    """
    Given the multiband signal, filter out the signal of the desired band. This
    is done by multipling a time varing signal to shift the frequency, and
    convolving with sinc function

    #### parameters:

    - `signal`: The input multiband signal in `time` domain
    - `fsamp`: the sampling frequency of `signal`
    - `f_center`: the carrier frequency of the target band, that is to say, the
      central frequency of the target subband signal
    - `bandwidth`: the bandwidth of the target subband signal. In the context
      of sinc filter, this is also the symbol rate
    - `window_sz`: sinc function is restricted within a limited window. This
      parameter indicates the number of symbols the window can hold

    #### returns:
    1. A numpy.ndarray representing the filtered signal in time domain. The
    sampling frequency is still `fsamp`
    """
    fsymb = bandwidth

    window_sz = window_sz * int(fsamp / fsymb + 0.5)
    parity = window_sz % 2
    pulse = np.zeros(window_sz + 1 - parity, dtype=np.cdouble)

    half = int(len(pulse) / 2)
    pulse = np.sinc(fsymb * (np.arange(0, len(pulse), 1) - half) / fsamp)

    sig_shifted = np.multiply(
        signal, np.exp(-2j * np.pi * f_center / fsamp * np.arange(len(signal)))
    )

    return fsymb / fsamp * np.convolve(sig_shifted, pulse, "same")


def sinc_pulse_shape(
    symbols: np.ndarray,
    fsamp: float,
    fcenter: float,
    bandwidth: float,
    window_sz: int = 10,
):
    """
    Use sinc function to turn complex symbols into inband complex signal. Sinc
    function is convolved with `symbols`. The time interval of output complex
    array is determined by `fsamp`.

    ### parameters:
    - `symbols`: input complex symbols, for example, QPSK, 8PSK
    - `fsamp`: the sampling rate of output signal, since the output signal is
      in discrete form
    - `fcenter`: the frequency center of output signal. It is a little bit like
      up-convertion, but this function simply shifts baseband signal to
      `fcenter`, so it is more like inband
    - `bandwidth`: the bandwidth of output signal. For this sinc pulse shaping,
      bandwidth is equal to symbol rate
    - `window_sz`: default 10. length of sinc function represented in array
      form in terms of symbol length.
    """
    fsymb = bandwidth
    # samples per symbol
    sps = int(fsamp / fsymb + 0.5)

    window_sz = min(window_sz * sps, len(symbols) * sps)
    parity = window_sz % 2
    # make sure pulse_len is odd, so the pulse array is symmetric and contains
    # peek
    pulse_len = window_sz + 1 - parity
    # sinc pulse whose bandwidth and symbol rate is fsymb
    pulse = np.sinc((np.arange(0, pulse_len) - int(pulse_len / 2)) * fsymb / fsamp)

    # upsample
    signal = np.zeros(len(symbols) * sps, dtype=np.cdouble)
    signal[::sps] = symbols
    # convolved signal has length of int(pulse_len / 2) * 2 + len(symbols) * sps
    signal = fsymb / fsamp * np.convolve(signal, pulse, "full")

    # the peek of the first symbol is at len(pulse) / 2, and the pulse
    # representing the first symbol is centered at the peek with length of sps
    start = int((pulse_len - sps) / 2)
    signal = signal[start : (start + len(symbols) * sps)]
    # shift to fcenter
    sig_shifted = np.multiply(
        signal, np.exp(2j * np.pi * fcenter / fsamp * np.arange(len(signal)))
    )

    return sig_shifted


def sinc_demodulate(
    signal: np.ndarray,
    fsamp: float,
    fcenter: float,
    sps: int,
):
    sig_shifted = np.multiply(
        signal, np.exp(-2j * np.pi * fcenter / fsamp * np.arange(len(signal)))
    )
    pulse = np.sinc((np.arange(0, sps) - int(sps / 2)) / sps)
    demod = [
        np.sum(np.multiply(pulse, sig_shifted[i * sps : (i + 1) * sps]))
        for i in range(int(len(signal) / sps))
    ]
    return np.array(demod)


def sqrt_cos_filter(
    signal: np.ndarray, fsamp: float, f_center: float, bandwidth: float, rolloff: float
):
    """
    Given the multiband signal, use square-root raised cos filter to filter out
    the signal which has carrier frequency `f_center` and bandwidth `bandwidth`
    The whole procedure is in time domain. The output filtered discrete signal
    has the same sampling frequency as the input wide multiband signal

    #### parameters:

    - `signal`: the input wide multiband signal in time domain
    - `fsamp`: the sampling frequency of `signal`
    - `f_center`: the carrier frequency of the target band, that is to say, the
                central frequency of target subband signal
    - `bandwidth`: the bandwidth of the target subband signal
    - `rolloff`: roll-off factor of SqrtRaisedCosFilter

    #### returns:
    1. A numpy.ndarray representing the filtered signal in time domain. The
    sampling frequency is still `fsamp`
    """
    fsymb = bandwidth / (1 + rolloff)

    srcos_filter = SqrtRaisedCosFilter(fsymb=fsymb, rolloff=rolloff, fsamp=fsamp)

    sig_shifted = np.zeros(len(signal), dtype=np.cdouble)
    sig_shifted = np.multiply(
        signal, np.exp(-2j * np.pi * f_center / fsamp * np.arange(len(signal)))
    )

    # the sample rate is still fsamp
    sig_filtered = srcos_filter.filter(sig_shifted)
    return sig_filtered
