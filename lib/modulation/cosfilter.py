"""
Transmit filter for pulse shaping
"""
import numpy as np


class RaisedCos:
    def __init__(self, fsymb: float, rolloff: float):
        self.f = fsymb
        self.alpha = rolloff

    def time(self, t: float) -> float:
        term1 = np.sinc(t * self.f)
        term2 = np.cos(self.alpha * np.pi * t * self.f) / (
            1 - 4 * (self.alpha * t * self.f) ** 2
        )

        return term1 * term2


class SqrtRaisedCos:
    def __init__(self, fsymb: float, rolloff: float):
        self.f = fsymb
        self.alpha = rolloff

    def time(self, t: float) -> float:
        if t == 0:
            return (1 + self.alpha * (4 / np.pi - 1)) * self.f
        elif t == 1 / (4 * self.alpha * self.f) or t == -1 / (4 * self.alpha * self.f):
            return (
                self.alpha
                * self.f
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
            term3 = np.pi * t * (1 - (4 * self.alpha * t * self.f) ** 2)

            return (term1 + term2) / term3

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
        self,
        fsymb: float,
        rolloff: float,
        samples_per_symbol: int,
        window_sz_in_symbols: int = 10,
    ):
        self.sqrt_raised_cos = SqrtRaisedCos(fsymb, rolloff)
        self.sps = samples_per_symbol
        self.window_sz = window_sz_in_symbols * samples_per_symbol

        parity = self.window_sz % 2
        self.__pulse = np.zeros(self.window_sz + 1 - parity, dtype=np.cdouble)

        # len(self.__pulse) is 2 * half + 1
        half = int(len(self.__pulse) / 2)
        fsamp = self.sps * fsymb
        for i in range(len(self.__pulse)):
            self.__pulse[i] = self.sqrt_raised_cos.time((i - half) / fsamp)

    def filter(self, input: np.ndarray) -> np.ndarray:
        len_pulse = len(self.__pulse)
        res_ext = np.zeros((len(input) + len_pulse), dtype=np.cdouble)

        for i in range(len(input)):
            res_ext[i : i + len_pulse] += input[i] * self.__pulse

        half = int(len_pulse / 2)
        return res_ext[half:-half]

    def debug_plot_window(self):
        import matplotlib.pyplot as plt

        plt.plot(self.__pulse.real)
        plt.title("window pulse")
        plt.show()


def upsample(symbols: np.ndarray, samples_per_symbol: int) -> np.ndarray:
    res = np.zeros((len(symbols) * samples_per_symbol), dtype=np.cdouble)
    res[::samples_per_symbol] = symbols
    return res


if __name__ == "__main__":
    """
    ideally, Roll-off factor has nothing to do with constellation map's EVM
    """
    FSYMB = 512
    SPS = 4
    SYMB_LEN = 1000
    FSAMP = FSYMB * SPS

    qpsk = np.zeros(4, dtype=np.cdouble)
    qpsk[0] = 1 + 1j
    qpsk[1] = -1 + 1j
    qpsk[2] = -1 - 1j
    qpsk[3] = 1 - 1j

    raisedcos = RaisedCos(fsymb=FSYMB, rolloff=0.35)
    signals = np.zeros(SYMB_LEN * SPS, dtype=np.cdouble)

    pulse_basic = np.zeros(2 * SYMB_LEN * SPS + 1, dtype=np.cdouble)
    for i in range(len(pulse_basic)):
        pulse_basic[i] = raisedcos.time((i - SYMB_LEN * SPS) / FSAMP)

    for n in range(SYMB_LEN):
        symbol = qpsk[np.random.randint(4)]
        signals += (
            symbol * np.roll(pulse_basic, n * SPS)[SYMB_LEN * SPS : 2 * SYMB_LEN * SPS]
        )

    import matplotlib.pyplot as plt

    plt.plot(signals.real)
    plt.plot(signals.imag)
    plt.title("signals' real part and imag part")
    plt.show()

    plt.plot(signals.real[::SPS], signals.imag[::SPS], ".")
    plt.title("constellation map")
    plt.show()

    sigF = np.fft.fft(signals)
    plt.plot(np.abs(sigF) / len(sigF))
    plt.title("Spectrum amplitude")
    plt.show()

    plt.plot(sigF.real / len(sigF))
    plt.plot(sigF.imag / len(sigF))
    plt.title("spectrum's real and imag part")
    plt.show()

    plt.plot(np.abs(np.fft.fft(signals**2)) / len(signals))
    plt.title("spectrum 2-power")
    plt.show()

    plt.plot(np.abs(np.fft.fft(signals**4)) / len(signals))
    plt.title("spectrum 4-power")
    plt.show()

    plt.plot(np.abs(np.fft.fft(signals**8)) / len(signals))
    plt.title("spectrum 8-power")
    plt.show()

    cos_signals = np.zeros(2 * SYMB_LEN * SPS, dtype=np.cdouble)
    for i in range(SYMB_LEN * SPS):
        cos_signals[i] = raisedcos.time(-i / FSAMP)
        cos_signals[-i] = raisedcos.time(i / FSAMP)

    cos_signals_f = np.fft.fft(cos_signals) / len(cos_signals)
    plt.plot(cos_signals_f.real)
    plt.plot(cos_signals_f.imag)
    plt.title("raised cos spectrum's real and imag")
    plt.show()
