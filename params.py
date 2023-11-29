import numpy as np

import lib.dvbs2.physical as phy

CNDarray = np.ndarray[int, np.dtype[np.cdouble]]

class CfgDVBS2:
    def __init__(self) -> None:
        self.StreamFormat = "TS"
        self.FECFrame = "normal"
        self.MODCOD = 18
        self.DFL = 42960
        self.ScalingMethod = "Unit average power"
        self.RolloffFactor = 0.35
        self.HasPilots = True
        self.SamplesPerSymbol = 4

cfgDVBS2 = CfgDVBS2()


class SimParams:
    def __init__(self) -> None:
        self.sps = cfgDVBS2.SamplesPerSymbol             # Samples per symbol
        self.numFrames = 1                              # Number of frames to be processed
        self.chanBW = 36e6                               # Channel bandwidth in Hertz
        self.cfo = 3e6                                   # Carrier frequency offset in Hertz
        self.sco = 5                                     # Sampling clock offset in parts
                                                            # per million
        self.phNoiseLevel = "Low"         # Phase noise level provided as
                                                            # 'Low', 'Medium', or 'High'
        self.EsNodB = 20                                 # Energy per symbol to noise ratio
        self.isPSK = False
                                                            # in decibels
simParams = SimParams()

class RxParams:
    def __init__(self, xFecFrameSize:int = 16200) -> None:
        # -  Input data size to fill one baseband frame
        self.inputFrameSize = None

        # -  PL data frame size
        self.xFecFrameSize = xFecFrameSize

        # -  User packet length
        self.UPL = None

        # -  PL frame size
        numPilots = int(np.floor(xFecFrameSize / (90 * 16)))
        self.plFrameSize = xFecFrameSize + 90 + 36 * numPilots

        # -  Counter to update PL frame number
        self.frameCount = 1

        # -  Number of pilot blocks
        self.numPilots = None

        # -  Pilot indices position in PL frame
        self.pilotInd: np.ndarray = np.array([])

        # -  PL scrambled reference pilots used in transmission
        self.refPilots = None

        # -  Buffer to store coarse frequency compensated output
        self.cfBuffer: CNDarray = np.array([])

        # -  Buffer to store fine frequency compensated output
        self.ffBuffer: CNDarray = np.array([])

        # -  A vector storing the phase estimates made on pilots blocks in a
        # frame
        self.pilotEst = np.array([])

        # -  State variable for carrier phase estimation
        self.prevPhaseEst: float = 0.0

        # -  State variable to store auto correlation value usedin fine
        # frequency error estimation 
        self.fineFreqCorrVal = np.cdouble(0)

        # -  Frame start index
        self.syncIndex = None

        # -  Modulation order
        self.modOrder = None

        # -  LDPC code rate
        self.codeRate = None

        # -  LDPC codeword length
        self.cwLen = None

        self.carrSyncLoopBW = 1e-2 * 0.023
        self.symbSyncLoopBW = 8e-3
        self.symbSyncLock = 6
        self.frameSyncLock = 1
        self.coarseFreqLock = 3
        self.fineFreqLock = 6
        self.hasFinePhaseCompensation = False
        self.finePhaseSyncLoopBW = 3.5e-6
        self.initialTimeFreqSync = self.symbSyncLock + self.frameSyncLock + self.coarseFreqLock
        self.totalSyncFrames = self.initialTimeFreqSync + self.fineFreqLock
        self.syncIndex = 1


class MisoParams:
    def __init__(self) -> None:
        self.sps = 4
        self.bandwidth = 200e3
        self.rolloff = 0.35
        self.fsymb = self.bandwidth / (1 + self.rolloff)
        self.fsamp = self.fsymb * self.sps
        self.xfec_len = 32400
        self.n_pilots = 22
        self.data_symb_len = self.xfec_len + phy.PL_HEADER_LEN + self.n_pilots * phy.PILOT_LEN
