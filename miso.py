import lib.signals as S
import lib.dvbs2.physical as phy
from params import MisoParams

import logging
import numpy as np
from typing import Optional
import matplotlib.pyplot as plt

CNDarray = np.ndarray[int, np.dtype[np.cdouble]]

N_MC = 10

LOW_AMP = -1
HIGH_AMP = 1
LOW_SNR = -15
HIGH_SNR = 5
OFFSET = 1


# used by pure sender
INVALID_LEN = 5


class PureSender(MisoParams):
    def __init__(self, dummy_path: str, data_path: str):
        super().__init__()
        self.dummy = S.load_matlab(dummy_path)[INVALID_LEN*miso_params.sps:]
        self.dummy_len = len(self.dummy)
        self.data = S.load_matlab(data_path)[INVALID_LEN*miso_params.sps:][:self.data_symb_len * self.sps]
        self.sig_1 = np.zeros((2 * self.dummy_len + self.data_symb_len * self.sps), dtype=np.cdouble)
        self.sig_2 = np.zeros((2 * self.dummy_len + self.data_symb_len * self.sps), dtype=np.cdouble)

        self.sig_1[:self.dummy_len] = self.dummy
        self.sig_1[-self.data_symb_len * self.sps:] = self.data
        self.sig_2[self.dummy_len: 2 * self.dummy_len] = self.dummy
        self.sig_2[-self.data_symb_len * self.sps:] = self.data


class AlamoutiSender(PureSender):
    def __init__(self, dummy_path: str, data_path: str, filter: S.SqrtRaisedCosFilter):
        super().__init__(dummy_path, data_path)
        self.filter = filter
        assert self.sps == int(filter.fsamp / filter.fsymb)
        data_symb = self.filter.filter(self.data)[::self.sps]
        parity = len(data_symb) % 2
        if parity == 1:
            # repeat the last symbol
            data_symb = np.concatenate((data_symb, [data_symb[-1]]))

        data_symb1 = np.zeros_like(data_symb, dtype=np.cdouble)
        data_symb1[0::2] = data_symb[0::2]
        data_symb1[1::2] = -data_symb[1::2].conj()
        if parity == 1:
            data_symb1 = data_symb1[:-1]
        sig_data_1 = self.filter.pulse_shape(data_symb1, 0.0)

        self.sig_1[-self.data_symb_len * self.sps:] = sig_data_1

        data_symb2 = np.zeros_like(data_symb, dtype=np.cdouble)
        data_symb2[0::2] = data_symb[1::2]
        data_symb2[1::2] = data_symb[0::2].conj()
        if parity == 1:
            data_symb2 = data_symb2[:-1]
        sig_data_2 = self.filter.pulse_shape(data_symb2, 0.0)
        self.sig_2[-self.data_symb_len * self.sps:] = sig_data_2


class MISOChannel():
    def __init__(self, amps: list[complex], offsets: list[int], freq_offs: list[float], dummy_len: int, fsamp: float) -> None:
        assert len(amps) == 2 and len(offsets) == 2 and len(freq_offs) == 2
        self.n_sig = 2
        self.amps = amps
        self.offsets = np.array(offsets, dtype=int)
        self.offsets = self.offsets - self.offsets.min()
        self._max_off = self.offsets.max()
        self.freq_offs = freq_offs
        self.dummy_len = dummy_len
        self.fsamp = fsamp

    def combine(self, sig: list[CNDarray], is_freq_off: bool = False) -> CNDarray:
        assert len(sig) == self.n_sig
        sig_len = len(sig[0])
        for i in range(self.n_sig):
            assert len(sig[i]) == sig_len

        max_off = self._max_off
        output = np.zeros((sig_len + max_off), dtype=np.cdouble)

        out1 = self.amps[0] * sig[0]
        if is_freq_off:
            out1 *= np.exp(2j * np.pi * np.arange(sig_len) * self.freq_offs[0] / self.fsamp)
        output[self.offsets[0]: self.offsets[0] + sig_len] += out1

        out2 = self.amps[1] * sig[1]
        if is_freq_off:
            out2[self.dummy_len:] *= np.exp(2j * np.pi * np.arange(sig_len - self.dummy_len) * self.freq_offs[1] / self.fsamp)
        output[self.offsets[1]: self.offsets[1] + sig_len] += out2

        output = output[:sig_len]
        return output

class Receiver(MisoParams):
    def __init__(self, dummy_path: str) -> None:
        super().__init__()
        self.filter = S.SqrtRaisedCosFilter(
            fsymb=self.fsymb,
            fsamp=self.fsamp,
            rolloff=self.rolloff,
        )
        self.__noise_sniffed: bool = False
        self.sig: Optional[CNDarray] = None
        # TODO: software engineering
        self.dummy: CNDarray = S.load_matlab(dummy_path)
        self.dummy = self.filter.filter(self.dummy)[INVALID_LEN*self.sps:]
        self.dummy /= np.mean(np.abs(self.dummy[::self.sps]))
        self.dummy_len: int = len(self.dummy)
        self.dummy_symb_len: int = int(self.dummy_len / self.sps)

        self.data = None
        self.pilot_indices = None

        self.delta_t = 0

    def sniff_noise(self, noise_sig: CNDarray) -> None:
        self.n_power = np.mean(np.abs(noise_sig[:self.dummy_len] ** 2))
        self.__noise_sniffed = True

    def receive(self, sig: CNDarray, fc: float) -> None:
        if not self.__noise_sniffed:
            # logging.warning("did not sniff noise, setting n_power to 0.01")
            self.n_power = 0.01
        self.sig = sig
        id = self.__coarse_start()
        if id is None:
            raise ValueError("did not find start of packet")

        # this copies self.sig
        sig_bb = self.sig[id:] * 1.0
        sig_bb *= np.exp(-2j * np.pi * np.arange(len(sig_bb)) * fc / self.fsamp)
        sig_filtered = self.filter.filter(sig_bb)
        id = self.__find_dummy(sig_filtered)
        id2 = self.__find_dummy(sig_filtered[id + self.sps * (self.dummy_symb_len - 1):])
        self.delta_t = id2 - self.sps
        if abs(id2 - self.sps) > self.sps / 4:
            logging.warning("offset greater than T_s / 4, refuse to accept")
            return

        self.__channel_est(sig_filtered, id)

    def compensate_alamouti(self) -> Optional[CNDarray]:
        if self.data is None or self.h is None:
            return None

        parity = len(self.data) % 2
        y = self.data * 1.0
        h1 = self.h1t * 1.0
        h2 = self.h2t * 1.0
        if parity == 1:
            # the last symbol is special
            y = y[:-1]
            h1 = h1[:-1]
            h2 = h2[:-1]

        y[1::2] = y[1::2].conj()
        n_pairs = int(len(y) / 2)
        y = y.reshape((n_pairs, 2, 1))
        H = np.zeros((n_pairs, 2, 2), dtype=np.cdouble)
        H[:, 0, 0] = h1[::2]
        H[:, 0, 1] = h2[::2]
        H[:, 1, 0] = h2[1::2].conj()
        H[:, 1, 1] = -h1[1::2].conj()
        self.H = H
        try:
            H_inv = np.linalg.inv(H)
        except np.linalg.LinAlgError:
            logging.warning("singular matrix H")
            return None

        x = np.matmul(H_inv, y)
        x = x.flatten()
        if parity == 1:
            last_symb = self.data[-1] / (self.h1t[-1] + self.h2t[-1])
            x = np.concatenate((x, [last_symb]))

        return x

    def phase_track_alamouti(self):
        if self.data is None:
            return

        if self.pilot_indices is None:
            data_len = len(self.data)
            n_pilots = int(np.floor(data_len / (phy.SLOT_LEN * phy.PILOT_SLOT_GAP)))
            n_pilots -= int(np.floor(n_pilots * phy.PILOT_LEN / (phy.SLOT_LEN * phy.PILOT_SLOT_GAP) + 0.5))
            self.pilot_indices = phy.get_pilot_indices(n_pilots)
            self.n_pilots = n_pilots
            self.pilots_ref = phy.get_pilot_seq(n_pilots, scramb_id=0)
            self.pilots_ref = self.pilots_ref.reshape((n_pilots, int(phy.PILOT_LEN / 2), 2))

        y_pilots = self.data[self.pilot_indices].reshape(self.pilots_ref.shape)
        breakpoint()

    def __coarse_start(self) -> Optional[int]:
        """
        find the start of a frame at sampling level
        """
        if self.sig is None:
            return None

        # use energy window to coarsely find start
        buffer = self.sig[:self.dummy_len]
        power = np.mean(np.abs(buffer) ** 2)
        for i in range(len(self.sig) - self.dummy_len):
            if power > 2 * self.n_power:
                return i
            power -= np.abs(buffer[0]) ** 2 / self.dummy_len
            buffer = np.concatenate((buffer[1:], [self.sig[i + self.dummy_len]]))
            power += np.abs(buffer[-1]) ** 2 / self.dummy_len

        return None

    def __find_dummy(self, sig_filtered: CNDarray) -> int:
        """
        find the start of dummy frame at sampling level
        input `sig_filtered` should be coarsely synchronized
        """
        c_min = np.inf
        id_max = 0
        ref_diff = self.dummy[::self.sps][1:] / self.dummy[::self.sps][:-1]
        ref_diff = ref_diff[1:] / ref_diff[:-1]
        for i in range(self.sps):
            id = phy.find_header_v2(sig_filtered[i:i + 1 * self.dummy_len:self.sps].reshape(-1, 1))

            id = id[0]
            symbs = sig_filtered[id * self.sps + i: id * self.sps + i + self.dummy_len : self.sps]
            diff = symbs[1:] / symbs[:-1]
            diff = diff[1:] / diff[:-1]

            c = np.mean(np.abs(ref_diff - diff))
            if c < c_min:
                c_min = c
                id_max = id * self.sps + i

        return id_max

    def __channel_est(self, sig_filtered: CNDarray, id: int) -> None:
        """
        input: `symbs` denote symbols starting from the first dummy frame

        calculate the cfo, phase error, amplitude, then get the equivalent h on
        data frame
        """
        dummy1 = sig_filtered[id: id + self.dummy_len: self.sps]
        dummy1 *= self.dummy[::self.sps].conj()
        dummy2 = sig_filtered[id + self.dummy_len + self.delta_t : id + 2 * self.dummy_len + self.delta_t : self.sps]
        dummy2 *= self.dummy[::self.sps].conj()

        cfo1 = self.__cfo(dummy1)
        cfo2 = self.__cfo(dummy2)
        # cfo1 = 0.011 * BANDWIDTH
        # cfo2 = 0.011 * BANDWIDTH
        print("cfo:", cfo1, cfo2)

        dummy1 *= np.exp(-2j * np.pi * np.arange(len(dummy1)) * cfo1 / self.fsymb)
        dummy2 *= np.exp(-2j * np.pi * np.arange(len(dummy2)) * cfo2 / self.fsymb)

        h1 = (dummy1[5:-5]).mean()
        h2 = (dummy2[5:-5]).mean()
        # h1 = np.exp(-0.7j)
        # h2 = 1
        print("(h1, h2):", h1, h2)
        print("(phase):", np.angle(h1), np.angle(h2))

        data = sig_filtered[id + 2 * self.dummy_len::self.sps][:self.data_symb_len]
        self.data = data * 1.0

        carrier = 2j * np.pi * np.arange(len(data)) / self.fsymb
        ph1 = np.exp(4j * np.pi * cfo1 * self.dummy_len / self.fsamp)
        ph2 = np.exp(2j * np.pi * cfo2 * self.dummy_len / self.fsamp)
        self.h1t = h1 * ph1 * np.exp(cfo1 * carrier)
        self.h2t = h2 * ph2 * np.exp(cfo2 * carrier)
        self.h = self.h1t + self.h2t

    def __cfo(self, dummy: CNDarray) -> float:
        """
        use dummy frame symbols to calculate carrier frequency offset
        """
        # this must be real
        # p = dummy[5:-5]
        # omegas = np.zeros(3, dtype=float)
        # for i in range(1, 4):
        #     diff = (p[i:] / p[:-i]).mean()
        #     om = np.angle(diff) / i
        #     omegas[i - 1] = om
        #
        # omega = omegas.mean()
        # return omega * self.fsymb / 2 / np.pi
        angles = np.unwrap(np.angle(dummy[5:-5]).real)
        n_half = int(len(angles) / 2)
        diff = angles[n_half : 2 * n_half] - angles[:n_half]
        cfo = diff.mean() * self.fsymb / 2 / np.pi / n_half
        return cfo


def canonical_awgn(pure: CNDarray, dummy_len: int, snr: float) -> CNDarray:
    sig_dummy = pure[:2 * dummy_len]
    noise_scale = S.awgn_scale(sig_dummy, snr)

    sz = len(pure)
    rng = np.random
    n = rng.normal(size=(sz), scale=1.0) + 1j * rng.normal(
        size=(sz),
        scale=1.0,
    )
    n = n * noise_scale

    return pure + n


def to_digits(x: CNDarray):
    return 2 * (x.real > 0) + (x.imag > 0)


if __name__ == "__main__":
    miso_params = MisoParams()
    filter = S.SqrtRaisedCosFilter(
        fsymb=miso_params.fsymb,
        fsamp=miso_params.fsamp,
        rolloff=miso_params.rolloff,
    )

    ps = PureSender("./data/scrambleDvbs2x2pktsDummy.csv", "./data/scrambleDvbs2x2pktsQPSK.csv")
    ala_sender = AlamoutiSender("./data/scrambleDvbs2x2pktsDummy.csv", "./data/scrambleDvbs2x2pktsQPSK.csv", filter)
    channel = MISOChannel(
        amps = [np.exp(-0.7j), 1],
        # amps = [1, 1],
        offsets = [0, 0],
        freq_offs = [0.011 * miso_params.bandwidth, 0.019 * miso_params.bandwidth],
        # freq_offs = [0, 0],
        dummy_len=ps.dummy_len,
        fsamp=miso_params.fsamp
    )

    sig_pure = channel.combine([ps.sig_1, ps.sig_2], True)
    fc = 300e6
    sig_pure *= np.exp(2j * np.pi * np.arange(len(sig_pure)) * fc / miso_params.fsamp)

    gt = to_digits(filter.filter(ps.sig_1[2 * ps.dummy_len:])[::miso_params.sps])

    receiver = Receiver(dummy_path="./data/scrambleDvbs2x2pktsDummy.csv")
    res = {}
    for _ in range(N_MC):
        for snr in [5, 10, 20, 30, 50, 100, 200]:
            # sig = S.awgn(sig_pure, snr=snr)
            sig = canonical_awgn(sig_pure, ps.dummy_len, snr=100)
            receiver.receive(sig, fc)
            data = receiver.data
            h = data / filter.filter(ps.sig_1[2 * ps.dummy_len:])[::miso_params.sps]

            # data_ideal = data / h
            data_est = data * receiver.h.conj()
            # data_est = data / h
            data_est = data_est / np.abs(data_est)
            plt.plot(data_est.real, data_est.imag, ".")
            plt.show()
            
            # ideal = to_digits(data_ideal)
            est = to_digits(data_est)
            # print("ideal SER:", (ideal != gt).sum() / len(ideal))
            ser = (est != gt).sum() / len(est)
            if res.get(snr) is None:
                res[snr] = []
            res[snr].append(ser)

    for k, v in res.items():
        print("res:", k, np.array(v).mean())
