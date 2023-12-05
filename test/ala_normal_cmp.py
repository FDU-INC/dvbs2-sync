import numpy as np
import matplotlib.pyplot as plt

from miso import MISOChannel, PureSender
from miso import AlamoutiSender
from miso import Receiver
from miso import canonical_awgn
from params import MisoParams
import lib.signals as S

CNDarray = np.ndarray[int, np.dtype[np.cdouble]]
N_MC = 30


def to_digits(x: CNDarray):
    bits = np.zeros(2 * len(x), dtype=int)
    bits[0::2] = 1 * (x.real > 0)
    bits[1::2] = 1 * (x.imag > 0)
    return bits

def main():
    miso_params = MisoParams()
    filter = S.SqrtRaisedCosFilter(
        fsymb=miso_params.fsymb,
        fsamp=miso_params.fsamp,
        rolloff=miso_params.rolloff,
    )

    ala_sender = AlamoutiSender(
        "./data/scrambleDvbs2x2pktsDummy.csv",
        "./data/scrambleDvbs2x2pktsQPSK.csv",
        filter,
    )
    ps = PureSender(
        "./data/scrambleDvbs2x2pktsDummy.csv", "./data/scrambleDvbs2x2pktsQPSK.csv"
    )
    channel = MISOChannel(
        amps=[np.exp(-0.7j), 1],
        # amps = [1, 1],
        offsets=[0, 0],
        freq_offs=[0.011 * miso_params.bandwidth, 0.019 * miso_params.bandwidth],
        # freq_offs = [0, 0],
        dummy_len=ala_sender.dummy_len,
        fsamp=miso_params.fsamp,
    )

    sig_pure = channel.combine([ala_sender.sig_1, ala_sender.sig_2], True)
    fc = 300e6
    # fc = 0
    sig_pure *= np.exp(2j * np.pi * np.arange(len(sig_pure)) * fc / miso_params.fsamp)

    bers = np.zeros((11))
    for snr in np.arange(0, 11, 1):
        for m in np.arange(N_MC):
            receiver = Receiver(dummy_path="./data/scrambleDvbs2x2pktsDummy.csv")
            sig = canonical_awgn(sig_pure, ala_sender.dummy_len, snr=snr)
            receiver.receive(sig, fc)
            data_est = receiver.compensate_alamouti()

            if data_est is None:
                print("demod error")
                continue

            gt_symbs = filter.filter(ps.sig_2)[
                2 * ala_sender.dummy_len :: miso_params.sps
            ]
            gt = to_digits(gt_symbs)
            pred = to_digits(data_est)
            error_bits = (gt != pred).sum()
            bers[snr] = error_bits / len(gt) / (m + 1) + bers[snr] * m / (m + 1)
        print("BER:")
        print(bers)

    bers = np.zeros((11))
    for snr in np.arange(0, 11, 1):
        for m in np.arange(N_MC):
            sig = S.awgn(ps.sig_2, snr=snr)
            data_est = filter.filter(sig)[2 * ala_sender.dummy_len :: miso_params.sps]

            if data_est is None:
                print("demod error")
                continue

            gt_symbs = filter.filter(ps.sig_2)[
                2 * ala_sender.dummy_len :: miso_params.sps
            ]
            gt = to_digits(gt_symbs[:-1000])
            pred = to_digits(data_est[:-1000])
            error_bits = (gt != pred).sum()
            bers[snr] = error_bits / len(gt) / (m + 1) + bers[snr] * m / (m + 1)
        print("BER:")
        print(bers)

    plt.plot(np.arange(0, 11, 1), bers, label="normal")
    plt.xlabel("SNR (dB)")
    plt.ylabel("BER")
    plt.yscale("log")
    plt.legend()
    plt.show()

if __name__ == "__main__":
    main()
