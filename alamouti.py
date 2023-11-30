import numpy as np
import matplotlib.pyplot as plt

from miso import MISOChannel
from miso import AlamoutiSender
from miso import Receiver
from miso import canonical_awgn
from params import MisoParams
import lib.signals as S

if __name__ == "__main__":
    miso_params = MisoParams()
    filter = S.SqrtRaisedCosFilter(
        fsymb=miso_params.fsymb,
        fsamp=miso_params.fsamp,
        rolloff=miso_params.rolloff,
    )

    ala_sender = AlamoutiSender("./data/scrambleDvbs2x2pktsDummy.csv", "./data/scrambleDvbs2x2pktsQPSK.csv", filter)
    channel = MISOChannel(
        amps = [np.exp(-0.7j), 1],
        # amps = [1, 1],
        offsets = [0, 0],
        freq_offs = [0.011 * miso_params.bandwidth, 0.019 * miso_params.bandwidth],
        # freq_offs = [0, 0],
        dummy_len=ala_sender.dummy_len,
        fsamp=miso_params.fsamp
    )

    sig_pure = channel.combine([ala_sender.sig_1, ala_sender.sig_2], True)
    # fc = 300e6
    fc = 0
    sig_pure *= np.exp(2j * np.pi * np.arange(len(sig_pure)) * fc / miso_params.fsamp)

    receiver = Receiver(dummy_path="./data/scrambleDvbs2x2pktsDummy.csv")
    sig = canonical_awgn(sig_pure, ala_sender.dummy_len, snr=0)
    sig_bb = sig * np.exp(-2j * np.pi * np.arange(len(sig_pure)) * fc / miso_params.fsamp)
    receiver.receive(sig, fc)
    data_est = receiver.compensate_alamouti()
    receiver.phase_track_alamouti()

    if data_est is None:
        print("demod error")
        exit()

    plt.plot(data_est.real, data_est.imag, ".")
    plt.show()
