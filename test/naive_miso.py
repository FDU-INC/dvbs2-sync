from miso import (
    MisoParams,
    PureSender,
    MISOChannel,
    Receiver,
    to_digits,
    canonical_awgn,
)
import lib.signals as S

import numpy as np

N_MC = 10

if __name__ == "__main__":
    miso_params = MisoParams()
    filter = S.SqrtRaisedCosFilter(
        fsymb=miso_params.fsymb,
        fsamp=miso_params.fsamp,
        rolloff=miso_params.rolloff,
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
        dummy_len=ps.dummy_len,
        fsamp=miso_params.fsamp,
    )

    sig_pure = channel.combine([ps.sig_1, ps.sig_2], True)
    fc = 300e6
    sig_pure *= np.exp(2j * np.pi * np.arange(len(sig_pure)) * fc / miso_params.fsamp)

    gt = to_digits(filter.filter(ps.sig_1[2 * ps.dummy_len :])[:: miso_params.sps])

    receiver = Receiver(dummy_path="./data/scrambleDvbs2x2pktsDummy.csv")
    res = {}
    for _ in range(N_MC):
        for snr in [5, 10, 20, 30, 50, 100, 200]:
            sig = canonical_awgn(sig_pure, ps.dummy_len, snr=snr)
            receiver.receive(sig, fc)
            data = receiver.data
            if data is None:
                continue

            h = data / filter.filter(ps.sig_1[2 * ps.dummy_len :])[:: miso_params.sps]

            data_est = data * receiver.h.conj()
            data_est = data_est / np.abs(data_est)

            est = to_digits(data_est)
            ser = (est != gt).sum() / len(est)
            if res.get(snr) is None:
                res[snr] = []
            res[snr].append(ser)

    for k, v in res.items():
        print("res:", k, np.array(v).mean())
