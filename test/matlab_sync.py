import params
import lib.signals as S
from synchronize import Synchronizer

import numpy as np

if __name__ == "__main__":
    signal = S.load_matlab("./data/scrambleDvbs2x2pktsQPSK.csv")
    print(len(signal))
    signal = np.tile(signal, 26)
    simParams = params.SimParams()
    sps = 4
    simParams.sps = sps
    rxParams = params.RxParams(xFecFrameSize=32400)
    syncer = Synchronizer(simParams, rxParams)

    syncIn = signal * np.exp(1j * np.pi * np.arange(len(signal)) * 0.11)
    syncIn = syncIn.reshape((len(syncIn), 1))

    import matplotlib.pyplot as plt
    for out in syncer(rxIn=syncIn):
        print()
        print("frameCount:", syncer.rxParams.frameCount)
        plt.plot(out.real, out.imag, ".")
        plt.show()
