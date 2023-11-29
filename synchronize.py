import numpy as np
import lib.signals as S
import lib.dvbs2.physical as phy
from lib.dvbs2.physical import PL_HEADER_LEN, PILOT_LEN, PILOT_SLOT_GAP, SLOT_LEN
import params

# rxParams = params.RxParams()
# simParams = params.SimParams()

# complex numpy.ndarray
CNDarray = np.ndarray[int, np.dtype[np.cdouble]]


class SymbolSynchronizer:
    """
    input x is column vector

    Ref: MATLAB R2022a comm.SymbolSynchronizer
    """

    def __init__(self, kp: float, normalizedLoopBandwidth: float = 8e-3) -> None:
        self.normlizedLoogBandwidth = normalizedLoopBandwidth
        self.damplingFactor = 1 / np.sqrt(2)
        self.detectorGain = kp
        self.sps = 2
        self.alpha = 0.5
        self.maxOutputExpansionFactor = [11, 10]

        a = 0.5
        self.interpFilterCoeff = np.array(
            [
                [0, 0, 1, 0],
                [-a, 1 + a, -(1 - a), -a],
                [a, -a, -a, a],
            ]
        )

        self.isFirst = True
        self.__initLoopFilterGains()

    def __initLoopFilterGains(self) -> None:
        """
        Derive proportional gain (K1) and integrator gain (K2) in the loop
        filter. Refer to (C.56) & (C.57) on page 736 in Rice's book.
        """
        zeta = self.damplingFactor
        bnts = self.normlizedLoogBandwidth
        kp = self.detectorGain
        k0 = -1
        theta = bnts / self.sps / (zeta + 0.25 / zeta)
        d = (1 + 2 * zeta * theta + theta * theta) * k0 * kp
        self.proportionalGain = 4 * zeta * theta / d
        self.integratorGain = 4 * theta * theta / d

    def reset(self):
        self.isFirst = True

    def __reset(self, inputLen: int) -> None:
        self.inputFrameLen = inputLen
        self.maxOutputFrameLen = np.ceil(
            self.inputFrameLen
            * self.maxOutputExpansionFactor[0]
            / (self.maxOutputExpansionFactor[1] * self.sps)
        )
        self.maxOutputFrameLen = int(self.maxOutputFrameLen)

        self.loopFilterState = 0.0
        self.loopPreviousInput = 0.0
        self.strobe = False
        self.numStrobe = 0
        self.strobeHistory = np.zeros((self.sps), dtype=bool)
        self.mu = 0.0
        self.NCOCounter = 0
        self.timingError = np.zeros(self.inputFrameLen)
        self.interpFilterState = np.zeros((3, 1), dtype=np.cdouble)
        self.TEDBuffer = np.zeros((self.sps), dtype=np.cdouble)
        self.symbolHolder = np.zeros((self.maxOutputFrameLen, 1), dtype=np.cdouble)

        self.isFirst = False

    def __call__(self, x: CNDarray) -> CNDarray:
        if self.isFirst:
            self.__reset(len(x))

        y, overflowFlag = self.__symbolSyncCodegen(x)
        if overflowFlag:
            print("warning: SymbolSynchronizer: symbol dropping")

        return y

    def __symbolSyncCodegen(self, x: CNDarray):
        self.numStrobe = 0
        symbolHolderLength = self.maxOutputFrameLen
        overflowFlag = False

        for sampleIdx in range(self.inputFrameLen):
            if (self.numStrobe == symbolHolderLength) and self.strobe:
                overflowFlag = True
                break

            self.numStrobe += self.strobe
            self.timingError[sampleIdx] = self.mu
            intOut = self.__interpFilter(x[sampleIdx])
            if self.strobe:
                self.symbolHolder[self.numStrobe - 1] = intOut

            if self.numStrobe > symbolHolderLength:
                overflowFlag = True
                break

            e = self.__GardnerTED(intOut)

            s = np.sum(self.strobeHistory[1:]) + self.strobe
            if s == 0:
                pass
            elif s == 1:
                self.TEDBuffer = np.concatenate((self.TEDBuffer[1:], [intOut]))
            else:
                self.TEDBuffer = np.concatenate((self.TEDBuffer[2:], [0, intOut]))

            v = self.__loopFilter(e)
            self.__interpControl(v)

        y = self.symbolHolder[: self.numStrobe, 0]

        return y, overflowFlag

    def __interpFilter(self, x: CNDarray) -> np.cdouble:
        """
        x: (1, 1)
        """
        xSeq: CNDarray = np.vstack((x, self.interpFilterState))

        self.interpFilterState = xSeq[:3]
        y = (
            (self.interpFilterCoeff @ xSeq) * np.array([[1, self.mu, self.mu**2]]).T
        ).sum()

        return y

    def __GardnerTED(self, x: np.cdouble) -> float:
        e = 0
        if self.strobe and np.all(not self.strobeHistory[1:]):
            midSample = self.TEDBuffer[int(self.sps / 2)]
            e = (midSample * (self.TEDBuffer[0] - x).conj()).real

        return e

    def __loopFilter(self, e: float) -> float:
        loopFiltOut = self.loopPreviousInput + self.loopFilterState
        self.loopFilterState = loopFiltOut
        self.loopPreviousInput = e * self.integratorGain
        v = e * self.proportionalGain + loopFiltOut
        return v

    def __interpControl(self, v: float) -> None:
        w = v + 1.0 / self.sps
        self.strobeHistory = np.concatenate((self.strobeHistory[1:], [self.strobe]))
        self.strobe = self.NCOCounter < w
        if self.strobe:
            self.mu = self.NCOCounter / w

        self.NCOCounter = (self.NCOCounter - w) % 1


class TimeFreqSynchronizer:
    def __init__(self, simParams: params.SimParams, rxParams: params.RxParams) -> None:
        self.sps = simParams.sps
        self.PLScrambingIndex = 0
        assert self.sps % 2 == 0

        self.bandwidth: float = 36e6
        self.carrSyncLoopBW: float = rxParams.carrSyncLoopBW
        self.dataFrameSize: int = rxParams.xFecFrameSize
        self.frameSyncAveragingFrames: int = rxParams.frameSyncLock
        self.rolloff: float = 0.35
        self.fsymb: float = self.bandwidth / (1 + self.rolloff)
        self.fsamp: float = self.fsymb * self.sps
        self.symbSyncLoopBW: float = rxParams.symbSyncLoopBW
        self.symbSyncTransitFrames: int = rxParams.symbSyncLock
        self.isPSK = simParams.isPSK

        self.frameCount = 1

        self.__setup()
        self.isFirst = True

    def __setup(self):
        self.integratorGain = 4 * self.carrSyncLoopBW
        self.digitalSynthesizerGain = -1
        self.__srcosFilter = S.SqrtRaisedCosFilterMat(
            fsymb=self.fsymb,
            rolloff=self.rolloff,
            fsamp=self.fsamp,
            window_sz_in_symbols=20,
        )

        self.__srcosFilter.set_gain(self.__srcosFilter.pulse.sum())

        self.symbolSync = SymbolSynchronizer(
            kp=4 * np.sin(np.pi * self.rolloff / 2) / (1 - self.rolloff**2 / 4),
            normalizedLoopBandwidth=self.symbSyncLoopBW,
        )

        numPilots = int(np.floor(self.dataFrameSize / (SLOT_LEN * PILOT_SLOT_GAP)))
        if self.dataFrameSize % (SLOT_LEN * PILOT_SLOT_GAP) == 0:
            numPilots -= 1

        self.pilotIndices = phy.get_pilot_indices(numPilots)
        self.numPilots = numPilots

        rn = phy.getScramblingSequence(self.PLScrambingIndex)
        self.pilotSeq = (
            (1 + 1j)
            / np.sqrt(2)
            * np.exp(1j * np.pi / 2 * rn)[self.pilotIndices - PL_HEADER_LEN]
        )
        self.pilotSeq = self.pilotSeq.reshape((len(self.pilotSeq), 1))
        self.fullFrameSize = self.dataFrameSize + PL_HEADER_LEN + numPilots * PILOT_LEN

    def reset(self):
        self.isFirst = True

    def __reset(self):
        self.loopFilterState = 0.0
        self.integFilterState = 0.0
        self.DDSPreviousInput = 0.0
        self.phase = np.zeros((self.sps, 1))
        self.previousSample = 0.0 + 0.0j
        self.previousSample2 = 0.0 + 0.0j
        self.freqError = 0.0
        self.frameCount = 0
        self.FSBuffer = np.zeros((180, 1), dtype=np.cdouble)
        self.previousFrameLength = 0
        self.possibleSyncIndices = -1 * np.ones((2, 10), dtype=int)
        self.syncIndex = 0
        self.symbolSync.reset()

        self.isFirst = False

    def recFilter(self, x: CNDarray) -> CNDarray:
        filtered: CNDarray = self.__srcosFilter(x.flatten())
        filtered = filtered[:: int(self.sps / 2)]
        filtered = filtered.reshape((len(filtered), 1))
        return filtered

    def __call__(self, input: CNDarray, freqLock: bool = False):
        if self.isFirst:
            self.__reset()

        loopFiltState = self.loopFilterState
        integFiltState = self.integFilterState
        DDSPreviousInp = self.DDSPreviousInput
        prevSample = self.previousSample
        prevSample2 = self.previousSample2
        freqError = self.freqError
        sps = self.sps
        pIndices = self.pilotIndices
        fBuffer = self.FSBuffer
        symbolCount = 0
        possSyncInd = self.possibleSyncIndices
        if freqLock:
            self.integratorGain = self.integratorGain / 10

        output = np.zeros(
            (int(input.shape[0] / sps) + self.numPilots, 1), dtype=input.dtype
        )

        timeSyncStat = self.frameCount > self.symbSyncTransitFrames
        frameSyncStat = (
            self.frameCount > self.symbSyncTransitFrames + self.frameSyncAveragingFrames
        )

        syncInd = 1
        if frameSyncStat:
            syncInd = self.syncIndex + self.fullFrameSize - self.previousFrameLength

        winLen = []
        for k in range(int(len(input) / sps)):
            pilotInd = []
            winLen = np.arange(k * sps, (k + 1) * sps)
            filtIn = input[winLen] * np.exp(1j * self.phase)
            filtOut = self.recFilter(filtIn)
            tSyncOut = self.symbolSync(filtOut)
            # if self.isPSK:
            #     tSyncOut = np.exp(1j * np.angle(tSyncOut))

            for n in range(len(tSyncOut)):
                symbolCount += 1
                if not timeSyncStat:
                    continue

                if not frameSyncStat:
                    # frame synchronization
                    if symbolCount > 180:
                        fBuffer = np.vstack((fBuffer, tSyncOut[n]))
                    else:
                        fBuffer[symbolCount - 1] = tSyncOut[n]
                    if symbolCount == 180:
                        tempInd = phy.find_header_v2(fBuffer) + 1
                        for z in range(len(tempInd)):
                            p1 = np.argwhere(possSyncInd[0, :] == tempInd[z])
                            p2 = np.argwhere(possSyncInd[0, :] == tempInd[z] - 1)
                            if len(p1) != 0:
                                p1 = p1[0][0]
                                possSyncInd[1, p1] = possSyncInd[1, p1] + 1
                            elif len(p2) != 0:
                                p2 = p2[0][0]
                                possSyncInd[0, p2] = tempInd[z]
                                possSyncInd[1, p2] = possSyncInd[1, p2] + 1
                            else:
                                p3 = np.where(possSyncInd[1, :] == -1)[0][0]
                                possSyncInd[0, p3] = tempInd[z]
                                possSyncInd[1, p3] = possSyncInd[1, p3] + 1

                        self.FSBuffer = np.zeros((180, 1), dtype=np.cdouble)
                        if (
                            self.frameCount
                            == self.symbSyncTransitFrames
                            + self.frameSyncAveragingFrames
                        ):
                            index = np.argmax(possSyncInd[1, :])
                            syncInd = possSyncInd[0, index]
                else:
                    # pIndices is 1 less than the MATLAB side
                    pilotInd = np.argwhere(
                        symbolCount == (pIndices + syncInd)
                    ).flatten()
                    idx = pilotInd % PILOT_LEN
                    if len(pilotInd) != 0 and idx >= 3:
                        freqError = np.imag(
                            prevSample
                            * np.conj(prevSample2)
                            * np.conj(self.pilotSeq[pilotInd - 1])
                            * self.pilotSeq[pilotInd - 3]
                        )

                if symbolCount > len(output):
                    print("frameCount", self.frameCount)
                    temp = np.zeros((2 * len(output), 1), dtype=output.dtype)
                    temp[: len(output)] = output
                    output = temp
                output[symbolCount - 1] = tSyncOut[n]

            loopFiltOut = freqError * self.integratorGain + loopFiltState
            loopFiltState = loopFiltOut
            for p in range(sps):
                DDSOut = DDSPreviousInp + integFiltState
                integFiltState = DDSOut
                DDSPreviousInp = loopFiltState / sps
                self.phase[p] = self.digitalSynthesizerGain * DDSOut

            for n in range(len(tSyncOut)):
                if not timeSyncStat:
                    output[symbolCount - 1] = tSyncOut[n]
                elif frameSyncStat and len(pilotInd) != 0:
                    if symbolCount - (pIndices[0] + syncInd) >= 2:
                        prevSample2 = output[pIndices[pilotInd] - 3 + syncInd]

                    prevSample = output[pIndices[pilotInd] + syncInd - 1]

        output = output[:symbolCount]
        self.loopFilterState = loopFiltState
        self.integFilterState = integFiltState
        self.previousSample = prevSample
        self.previousSample2 = prevSample2
        self.DDSPreviousInput = DDSPreviousInp
        self.frameCount += 1
        self.freqError = freqError
        self.previousFrameLength = symbolCount
        self.possibleSyncIndices = possSyncInd
        self.syncIndex = syncInd

        return output, syncInd


class Synchronizer:
    # TODO: finePhaseSync
    def __init__(self, simParams: params.SimParams, rxParams: params.RxParams) -> None:
        self.simParams = simParams
        self.rxParams = rxParams
        self.coarseSyncer = TimeFreqSynchronizer(simParams, rxParams)
        self.rxParams.pilotInd = self.coarseSyncer.pilotIndices
        self.symSyncOutLen = np.zeros((rxParams.initialTimeFreqSync, 1))

    def __call__(self, rxIn: CNDarray):
        stIdx: int = 0
        resSymb: int = 0
        print("len(rxIn):", len(rxIn))
        while stIdx < len(rxIn):
            print("stIdx:", stIdx, ", len(rxIn):", len(rxIn))
            endIdx = stIdx + self.rxParams.plFrameSize * self.simParams.sps
            isLastFrame = endIdx > len(rxIn)
            if isLastFrame:
                endIdx = len(rxIn)
            rxData = rxIn[stIdx:endIdx]

            coarseFreqLock = (
                self.rxParams.frameCount >= self.rxParams.initialTimeFreqSync
            )

            syncIn = rxData
            if isLastFrame:
                resSymb = self.rxParams.plFrameSize - len(self.rxParams.cfBuffer)
                resSampCnt = resSymb * self.simParams.sps - len(rxData)
                if resSampCnt >= 0:
                    syncIn = np.concatenate((rxData, np.zeros((resSampCnt, 1))))
                else:
                    syncIn = rxData[: resSymb * self.simParams.sps]

            coarseFreqSyncOut, syncIndex = self.coarseSyncer(syncIn, coarseFreqLock)
            if self.rxParams.frameCount <= self.rxParams.initialTimeFreqSync:
                print("len(coarseFreqSyncOut):", len(coarseFreqSyncOut))
                self.symSyncOutLen[self.rxParams.frameCount - 1] = len(
                    coarseFreqSyncOut
                )
                diff = np.diff(self.symSyncOutLen[: self.rxParams.frameCount], axis=0)
                if np.any(np.abs(diff) > 5):
                    error_msg = (
                        "Symbol timing synchronization failed. The loop will not converge. No frame will be recovered. "
                        "Update the symbSyncLoopBW parameter according to the EsNo setting for proper loop convergence."
                    )
                    raise ValueError(error_msg)

            self.rxParams.syncIndex = syncIndex
            print("syncIndex:", syncIndex)

            fineFreqIn = (
                np.concatenate(
                    (
                        self.rxParams.cfBuffer,
                        coarseFreqSyncOut[: self.rxParams.syncIndex - 1],
                    )
                )
                if self.rxParams.cfBuffer.size
                else coarseFreqSyncOut[: self.rxParams.syncIndex - 1]
            )
            print("len(fineFreqIn):", len(fineFreqIn))
            if isLastFrame:
                resCnt = resSymb - len(coarseFreqSyncOut)
                if resCnt <= 0:
                    fineFreqIn = np.concatenate(
                        (self.rxParams.cfBuffer, coarseFreqSyncOut[:resSymb])
                    )
                else:
                    fineFreqIn = np.concatenate(
                        (
                            self.rxParams.cfBuffer,
                            coarseFreqSyncOut,
                            np.zeros((resCnt, 1), dtype=np.cdouble),
                        )
                    )

            if (self.rxParams.frameCount > self.rxParams.initialTimeFreqSync + 1) and (
                self.rxParams.frameCount <= self.rxParams.totalSyncFrames + 1
            ):
                self.rxParams.fineFreqCorrVal = fineFreqEst(
                    fineFreqIn[self.rxParams.pilotInd],
                    self.coarseSyncer.numPilots,
                    self.coarseSyncer.pilotSeq,
                    self.rxParams.fineFreqCorrVal,
                )

            # fineFreqLock
            syncOut: CNDarray = fineFreqIn
            if self.rxParams.frameCount > self.rxParams.totalSyncFrames:
                syncOut = self.__fineFreq(fineFreqIn, isLastFrame)

            self.rxParams.cfBuffer = coarseFreqSyncOut[self.rxParams.syncIndex - 1 :]
            self.rxParams.syncIndex = syncIndex
            self.rxParams.frameCount += 1

            stIdx = endIdx
            yield syncOut

    def __fineFreq(self, fineFreqIn: CNDarray, isLastFrame: bool) -> CNDarray:
        # Normalize the frequency estimate by the input symbol rate freqEst =
        # angle(R)/(pi*(N+1)), where N (18) is the number of elements used to
        # compute the mean of auto correlation (R) in HelperDVBS2FineFreqEst.
        # 19 / 2 comes from (1 + 18) / 2
        freqEst = np.angle(self.rxParams.fineFreqCorrVal) / (np.pi * 19)

        # Generate the symbol indices using frameCount and plFrameSize.
        # Subtract 2 from the rxParams.frameCount because the buffer used to
        # get one PL frame introduces a delay of one to the count.
        ind = np.arange(
            (self.rxParams.frameCount - 2) * self.rxParams.plFrameSize,
            (self.rxParams.frameCount - 1) * self.rxParams.plFrameSize,
        )
        phErr = np.exp(-1j * 2 * np.pi * freqEst * ind).reshape(-1, 1)
        fineFreqOut = fineFreqIn * phErr

        # Estimate the phase error estimation by using the HelperDVBS2PhaseEst
        # helper function.
        phEstRes, self.rxParams.prevPhaseEst = phaseEst(
            fineFreqOut[self.rxParams.pilotInd],
            self.coarseSyncer.pilotSeq,
            self.rxParams.prevPhaseEst,
        )

        if len(self.rxParams.pilotEst) != 0:
            phEstRes = np.unwrap(np.vstack((self.rxParams.pilotEst, phEstRes)), axis=0)[
                len(phEstRes) :
            ]

        # Compensate for the residual frequency and phase offset by using the
        # HelperDVBS2PhaseCompensate helper function. Use two frames for
        # initial phase error estimation. Starting with the second frame, use
        # the phase error estimates from the previous frame and the current
        # frame in compensation. Add 3 to the frame count comparison to account
        # for delays: One frame due to rxParams.cfBuffer delay and two frames
        # used for phase error estimate.
        syncOut = np.array([], dtype=np.cdouble)
        if self.rxParams.frameCount >= self.rxParams.totalSyncFrames + 3:
            coarsePhaseCompOut = phaseCompensate(
                self.rxParams.ffBuffer,
                self.rxParams.pilotEst,
                self.rxParams.pilotInd,
                phEstRes[1],
            )
            syncOut = coarsePhaseCompOut

        self.rxParams.ffBuffer = fineFreqOut
        self.rxParams.pilotEst = phEstRes

        if isLastFrame:
            pilotBlkFreq = PILOT_SLOT_GAP * SLOT_LEN + PILOT_LEN
            avgSlope = np.diff(phEstRes[1:]).mean()
            chunkLen = (
                self.rxParams.plFrameSize
                - self.rxParams.pilotInd[-1]
                + self.rxParams.pilotInd[PILOT_LEN - 1]
            )
            estEndPh = phEstRes[-1] + avgSlope * chunkLen / pilotBlkFreq
            coarsePhaseCompOut1 = phaseCompensate(
                self.rxParams.ffBuffer,
                self.rxParams.pilotEst,
                self.rxParams.pilotInd,
                estEndPh,
            )
            syncOut = np.concatenate((syncOut, coarsePhaseCompOut1))

        return syncOut


def fineFreqEst(
    rxPilots: CNDarray, numPilotBlks: int, refPilots: CNDarray, R: np.cdouble
) -> np.cdouble:
    """
    R is added with
    $sum_{m=1}^{18} e^{j omega m}$
    """
    Lp = PILOT_LEN
    # number of elements used to compute the auto correlation over each pilot block
    N = int(PILOT_LEN / 2)

    pilotBlock = rxPilots.reshape(numPilotBlks, Lp).T
    refBlock = refPilots.reshape(numPilotBlks, Lp).T

    for m in range(1, N + 1):
        rm = (
            pilotBlock[m:]
            * refBlock[m:].conj()
            * pilotBlock[:-m].conj()
            * refBlock[:-m]
        )
        R += rm.mean(axis=0).sum()

    return R


def phaseEst(rxPilots: CNDarray, refPilots: CNDarray, prevPhErrEst: float):
    numBlks = int(len(rxPilots) / PILOT_LEN)

    stIdx = 0
    phErrEst = np.zeros((numBlks + 1, 1))
    phErrEst[0] = prevPhErrEst

    for idx in range(numBlks):
        endIdx = stIdx + PILOT_LEN
        winLen = range(stIdx, endIdx)
        buffer = rxPilots[winLen]
        ref = refPilots[winLen]
        # carrier phase error calculation
        phTemp = float(np.angle(np.sum(buffer * ref.conj())))
        # Unwrapping the phase error using the phase estimate made on the
        # previous pilot block
        phErrEst[idx + 1] = prevPhErrEst + np.unwrap([phTemp - prevPhErrEst])
        prevPhErrEst = phErrEst[idx + 1]
        stIdx = endIdx

    phErrEst = np.unwrap(phErrEst, axis=0)
    return phErrEst, prevPhErrEst


def phaseCompensate(
    rxData: CNDarray, phEst: np.ndarray, pilotInd: np.ndarray, arg4: float
) -> CNDarray:
    syncOut = np.zeros((rxData.shape), dtype=np.cdouble)
    pilotBlkFreq = PILOT_LEN + SLOT_LEN * PILOT_SLOT_GAP

    # length of remaining symbols without pending pilots
    chunkLLen = len(rxData) - pilotInd[-1] - 1
    # length of symbols between the last pilots of the previous frame and the
    # first pilot of the current frame
    # which is also chunkLLen + PILOT_LEN
    chunk1Len = chunkLLen + pilotInd[PILOT_LEN - 1] + 1

    # TODO: in matlab, (phEst[1] - phEst[0]) * np.arange is element-wise, be cautious
    phData = (
        phEst[0]
        + (phEst[1] - phEst[0]) * np.arange(chunkLLen + 1, chunk1Len + 1) / chunk1Len
    )
    phData = phData.reshape(-1, 1)
    syncOut[: pilotBlkFreq + PL_HEADER_LEN] = rxData[
        : pilotBlkFreq + PL_HEADER_LEN
    ] * np.exp(-1j * phData)

    endIdx = pilotInd[PILOT_LEN - 1]
    stIdx = endIdx + 1

    numBlks = int(len(pilotInd) / PILOT_LEN)
    for idx in range(1, numBlks):
        stIdx = endIdx + 1
        endIdx = pilotInd[(idx + 1) * PILOT_LEN - 1]

        # Interpolation of phase estimate on data using the phase estimates
        # computed on preceding and succeeding pilot blocks
        phData = (
            phEst[idx]
            + (phEst[idx + 1] - phEst[idx])
            * np.arange(1, pilotBlkFreq + 1)
            / pilotBlkFreq
        )
        phData = phData.reshape(-1, 1)
        syncOut[stIdx : endIdx + 1] = rxData[stIdx : endIdx + 1] * np.exp(
            -1j * phData[:]
        )

    phData = phEst[-1] + (arg4 - phEst[-1]) * np.arange(1, chunkLLen + 1) / chunk1Len
    phData = phData.reshape(-1, 1)
    syncOut[pilotInd[-1] + 1 :] = rxData[pilotInd[-1] + 1 :] * np.exp(-1j * phData[:])
    return syncOut


if __name__ == "__main__":
    signal = S.load_matlab("./data/scrambleDvbs2x2pktsQPSK.csv")
    print(len(signal))
    signal = np.tile(signal, 26)
    simParams = params.SimParams()
    sps = 4
    simParams.sps = sps
    rxParams = params.RxParams(xFecFrameSize=32400)
    syncer = Synchronizer(simParams, rxParams)

    # stIdx = 0
    # endIdx = stIdx + rxParams.plFrameSize * simParams.sps
    # syncIn = signal[stIdx:endIdx]
    syncIn = signal * np.exp(1j * np.pi * np.arange(len(signal)) * 0.11)

    syncIn = syncIn.reshape((len(syncIn), 1))
    import matplotlib.pyplot as plt

    for out in syncer(rxIn=syncIn):
        print()
        print("frameCount:", syncer.rxParams.frameCount)
        plt.plot(out.real, out.imag, ".")
        plt.show()
