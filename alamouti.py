from miso import *

if __name__ == "__main__":
    filter = S.SqrtRaisedCosFilter(
        fsymb=FSYMB,
        fsamp=FSAMP,
        rolloff=ROLLOFF,
    )

    ala_sender = AlamoutiSender("./data/scrambleDvbs2x2pktsDummy.csv", "./data/scrambleDvbs2x2pktsQPSK.csv", filter)
    channel = MISOChannel(
        amps = [np.exp(-0.7j), 1],
        # amps = [1, 1],
        offsets = [0, 0],
        freq_offs = [0.011 * BANDWIDTH, 0.019 * BANDWIDTH],
        # freq_offs = [0, 0],
        dummy_len=ala_sender.dummy_len,
        fsamp=FSAMP
    )

    sig_pure = channel.combine([ala_sender.sig_1, ala_sender.sig_2], True)
    fc = 300e6
    sig_pure *= np.exp(2j * np.pi * np.arange(len(sig_pure)) * fc / FSAMP)

    receiver = Receiver(sps=SPS, bandwidth=BANDWIDTH, rolloff=ROLLOFF, dummy_path="./data/scrambleDvbs2x2pktsDummy.csv")
    sig = canonical_awgn(sig_pure, ala_sender.dummy_len, snr=5)
    sig_bb = sig * np.exp(-2j * np.pi * np.arange(len(sig_pure)) * fc / FSAMP)
    dummy = filter.filter(sig_bb[:ala_sender.dummy_len])[::SPS]
    plt.plot(dummy.real, dummy.imag, ".")
    plt.show()
    receiver.receive(sig, fc)
    data_est = receiver.compensate_alamouti()

    if data_est is None:
        print("demod error")
        exit()

    plt.plot(data_est.real, data_est.imag, ".")
    plt.show()
