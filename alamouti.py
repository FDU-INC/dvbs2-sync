from miso import *
from params import MisoParams

if __name__ == "__main__":
    miso_params = MisoParams()
    filter = S.SqrtRaisedCosFilter(
        fsymb=miso_params.fsymb,
        fsamp=miso_params.fsamp,
        rolloff=miso_params.rolloff,
    )

    ala_sender = AlamoutiSender("./data/scrambleDvbs2x2pktsDummy.csv", "./data/scrambleDvbs2x2pktsQPSK.csv", filter)
    channel = MISOChannel(
        # amps = [np.exp(-0.7j), 1],
        amps = [1, 1],
        offsets = [0, 0],
        # freq_offs = [0.011 * BANDWIDTH, 0.019 * BANDWIDTH],
        freq_offs = [0, 0],
        dummy_len=ala_sender.dummy_len,
        fsamp=miso_params.fsamp
    )

    sig_pure = channel.combine([ala_sender.sig_1, ala_sender.sig_2], True)
    fc = 300e6
    sig_pure *= np.exp(2j * np.pi * np.arange(len(sig_pure)) * fc / miso_params.fsamp)

    receiver = Receiver(sps=miso_params.sps, bandwidth=miso_params.bandwidth, rolloff=miso_params.rolloff, dummy_path="./data/scrambleDvbs2x2pktsDummy.csv")
    sig = canonical_awgn(sig_pure, ala_sender.dummy_len, snr=50)
    sig_bb = sig * np.exp(-2j * np.pi * np.arange(len(sig_pure)) * fc / miso_params.fsamp)
    receiver.receive(sig, fc)
    data_est = receiver.compensate_alamouti()
    receiver.phase_track_alamouti()

    if data_est is None:
        print("demod error")
        exit()

    plt.plot(data_est.real, data_est.imag, ".")
    plt.show()
