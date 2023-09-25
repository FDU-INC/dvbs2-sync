# physical layer
import numpy as np
import sys
import ctypes
import os

PL_HEADER_LEN = 90
SOF_LEN = 26
PILOT_LEN = 36
SLOT_LEN = 90
PILOT_SLOT_GAP = 16


def getScramblingSequence(scramIdx: int = 0) -> np.ndarray:
    """
    Get the physical layer scrambling sequence

    ##### parameters

    - `scramIdx`: the physical layer scrambling index described in ETSI EN 302
      307-2 Section 5.5.4 Table 19e. By default,`scramIdx` = 0

    ##### returns

    Integer valued scrambling sequence Rn (Rn assuming values 0, 1, 2, 3).
    Physical layer symbols should be multiplied by exp(j Rn(i) pi/2), where `n`
    is determined by `scramIdx`
    """
    if scramIdx > 6 or scramIdx < 0:
        raise ValueError("`scramIdx` should be in [0, 6]")

    dirname = os.path.dirname(__file__)
    libscramb = ctypes.WinDLL(dirname + "/libscramb.dll")

    libscramb.malloc_scramble_seq.restype = ctypes.POINTER(ctypes.c_int8)
    seq_ptr = libscramb.malloc_scramble_seq(scramIdx)
    rn = np.fromiter(seq_ptr, dtype=np.int8, count=libscramb.get_seq_len())

    libscramb.free_scramble_seq(seq_ptr)

    return rn


def find_header(symbols: np.ndarray) -> int:
    """
    a poor implementation for finding the starting index of PLHEADER in the
    input symbols. If cannot find the PLHEADER, -1 is returned
    """
    sofcodes = [0, 3, 2, 1, 0, 1, 2, 3, 0, 3, 0, 1, 2, 1, 2, 3, 2, 1, 2, 1, 0, 1, 0, 1, 2, 1]

    def s2c(s: np.cdouble):
        if s.real > 0 and s.imag > 0:
            return 0
        elif s.real < 0 and s.imag > 0:
            return 1
        elif s.real < 0 and s.imag < 0:
            return 2
        elif s.real > 0 and s.imag < 0:
            return 3
        else:
            # on the axis, can't judge
            return 4

    length = len(symbols)
    head = 0
    tail = 0
    while tail < length and (tail - head) < 26:
        # find the beginning index
        while head < length and s2c(symbols[head]) != sofcodes[0]:
            head += 1
            tail += 1

        # match SOF part
        i = 0
        while i < 26 and tail < length and s2c(symbols[tail]) == sofcodes[i]:
            tail += 1
            i += 1

        # didn't find SOF in this match trial
        if i < 26:
            head += 1
            tail = head

    # didn't find SOF in the whole input symbols
    if (tail - head) < 26:
        return -1

    return head


def find_header_v2(symbols: np.ndarray):
    ref_plsc = 1j*np.array([-1,1,1,-1,-1,-1,1,-1,-1,1,1,1,1,1,-1,-1,-1,-1,1,1,-1,1,1,-1,1,-1,1,-1,1,1,-1,-1])
    ref_sof = 1j*np.array([1,1,1,1,-1,-1,-1,-1,1,-1,-1,-1,1,-1,-1,1,1,-1,1,1,-1,1,-1,-1,1])
    ref_plsc = ref_plsc.reshape(-1, 1)
    ref_sof = ref_sof.reshape(-1, 1)
    win_len = PL_HEADER_LEN
    n = len(symbols)
    corr = np.zeros(n - win_len + 1, dtype=np.cdouble)
    for k in range(n - win_len + 1):
        buff = symbols[k : k + win_len]
        diff = buff[:-1] * np.conjugate(buff[1:])
        c_sof = (diff[:25] * np.conjugate(ref_sof)).sum()
        c_plsc = (diff[26::2] * np.conjugate(ref_plsc)).sum()
        corr[k] = np.abs(c_sof + c_plsc)

    max_val = np.max(corr)
    sync_id = np.where(corr / max_val > 0.75)[0][:2]
    return sync_id


def descramble(symbols: np.ndarray, scramb_id: int = 0) -> np.ndarray:
    """
    Use the physical layer scrambling sequence to descramble the waveform signal
    It is assumed that the first item of input symbols is the start of PLHEADER

    ##### parameters

    - `symbols`: the input dvb-s2(x) symbols to be descrambled. The first 90
    symbols represent the physical layer header
    - `scramb_id`: the physical layer scrambling index described in ETSI EN 302
      307-2 Section 5.5.4 Table 19e. By default,`scramb_id` = 0

    ##### returns

    a numpy.ndarray of the descrambled symbols
    """
    rn = getScramblingSequence(scramb_id)
    symbols[PL_HEADER_LEN:] = np.multiply(
        symbols[PL_HEADER_LEN:],
        np.exp(-1j * np.pi / 2 * rn[: (len(symbols) - PL_HEADER_LEN)]),
    )
    return symbols


def to_symbols(signal: np.ndarray, sps: int, scramIdx: int = 0) -> np.ndarray:
    """
    Find PLHEADER from `signal`, then descramble symbols

    ##### parameters

    - `signal`: discrete dvdb-s2(x) signal
    - `sps`: samples per symbol for the input `signal`
    - `scramIdx`: the physical layer scrambling index described in ETSI EN 302
      307-2 Section 5.5.4 Table 19e. By default,`scramIdx` = 0

    ##### returns

    - numpy.ndarray: The descrambled dvb-s2(x) complex symbols.
    """
    symbols = signal[::sps]

    head = find_header(symbols)
    if head >= 0:
        print("find PLHEADER at {}-th symbol".format(head))
        head_2 = find_header(symbols[(head + PL_HEADER_LEN) :])
        print("find PLHEADER {} symbols after the last header".format(head_2))
        symbols = descramble(symbols[head:], scramIdx)
    else:
        print("[In to_symbols()]: Failed to find PLHEADER", file=sys.stderr)

    return symbols


def __demod_plheader(header: np.ndarray):
    """
    Turns the PLHEADER in pi/2 BPSK modulated symbols form into binary sequence
    form. Length of `header` must be equal to PL_HEADER_LEN, which is, 90

    ##### parameters:

    - `header`: PLHEADER in symbols form, modulated by pi/2 BPSK. Its length
      should be equal to PL_HEADER_LEN(90)

    ##### returns:

    - numpy.ndarray: PLHEADER in binary sequence form, including SOF and
      encoded PLS code
    - bool: is this modulation using DVB-S2X format
    """
    if len(header) != PL_HEADER_LEN:
        raise ValueError("header length should be {}".format(PL_HEADER_LEN))

    header_bits = np.zeros(len(header), dtype=np.int8)

    def pi2bpsk_to_bits(symbols, length):
        bits = np.zeros(length, dtype=np.int8)
        for i in range(length):
            if i % 2 == 0:
                if symbols[i].real > 0 and symbols[i].imag > 0:
                    bits[i] = 0
                elif symbols[i].real < 0 and symbols[i].imag < 0:
                    bits[i] = 1
                else:
                    # invalid constellation point
                    bits[i] = -1
            else:
                if symbols[i].real < 0 and symbols[i].imag > 0:
                    bits[i] = 0
                elif symbols[i].real > 0 and symbols[i].imag < 0:
                    bits[i] = 1
                else:
                    # invalid constellation point
                    bits[i] = -1

        return bits

    header_bits[:SOF_LEN] = pi2bpsk_to_bits(header, SOF_LEN)

    plscode = pi2bpsk_to_bits(header[SOF_LEN:], PL_HEADER_LEN - SOF_LEN)
    isX = False
    if plscode.sum() < 0:
        plscode = pi2bpsk_to_bits(-1j * header[SOF_LEN:], PL_HEADER_LEN - SOF_LEN)
        isX = True

    header_bits[SOF_LEN:] = plscode
    return header_bits, isX


__pls_scramble_seq = np.array(
    [ 0, 1, 1, 1, 0, 0, 0, 1, 1, 0, 0, 1, 1, 1, 0, 1, 1, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 1, 0, 0, 1, 0, 1, 0, 1, 0, 0, 1, 1, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 1, 1, 0, 1, 1, 1, 1, 1, 1, 0, 1, 0, ]
)
__G = np.array(
    [
        [ 1, 0, 0, 1, 0, 0, 0, 0, 1, 0, 1, 0, 1, 1, 0, 0, 0, 0, 1, 0, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1, 0, 1, ],
        [ 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, ],
        [ 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, ],
        [ 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1, ],
        [ 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, ],
        [ 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, ],
        [ 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, ],
    ]
)
__encoded_pls_words = None
__encoded_pls_words_x = None


def __decode_pls(encoded_pls: np.ndarray, isX: bool):
    """
    Decodes the 64 bits long PLS code to an integer in the range of [0, 255],
    using maximum likelihood method

    ##### parameters:

    - `plscode`: 64 bits long binary sequence of encoded PLS code
    - `isX`: does the PLFRAME use DVB-S2X specific format

    ##### returns:

    - int: decoded decimal PLS code
    """
    if len(encoded_pls) != PL_HEADER_LEN - SOF_LEN:
        raise ValueError("length of input encoded plscode should be 64")

    N_POSSIBLE_CODES = 64
    WORD_LEN = 32

    global __G
    global __encoded_pls_words
    global __encoded_pls_words_x
    if __encoded_pls_words is None:
        __encoded_pls_words = np.zeros((N_POSSIBLE_CODES, WORD_LEN), dtype=np.int8)
        __encoded_pls_words_x = np.zeros((N_POSSIBLE_CODES, WORD_LEN), dtype=np.int8)
        for n in range(N_POSSIBLE_CODES):
            # the first bit b0 is determined by `isX`, the last bit b7 is
            # determined by the complementarity of the odd and even words
            plscode = np.array(list(format(n, "06b")), dtype=np.int8)
            __encoded_pls_words[n] = np.dot(plscode, __G[1:, :]) % 2
            __encoded_pls_words_x[n] = (__encoded_pls_words[n] + __G[0, :]) % 2

        # such that all code words have the same magnitude
        __encoded_pls_words = 2 * __encoded_pls_words - 1
        __encoded_pls_words_x = 2 * __encoded_pls_words_x - 1

    descrambled_pls = __pls_scramble_seq ^ encoded_pls
    soft_pls = (2 * descrambled_pls - 1)[::2]
    soft_pls_cmpl = (2 * descrambled_pls - 1)[1::2]
    b7 = 0
    if np.linalg.norm(soft_pls + soft_pls_cmpl) < np.linalg.norm(
        soft_pls - soft_pls_cmpl
    ):
        b7 = 1

    # maximum likelihood estimation
    correlation = np.zeros((N_POSSIBLE_CODES))
    b0 = 0
    if isX:
        correlation = np.dot(soft_pls, __encoded_pls_words_x.T)
        b0 = 1
    else:
        correlation = np.dot(soft_pls, __encoded_pls_words_x.T)

    code = b0 * 2 * N_POSSIBLE_CODES + (np.argmax(correlation) << 1) + b7

    return code


def get_plscode(plframe: np.ndarray):
    """
    get PLS code from input PLFRAME
    """
    header_bits, isX = __demod_plheader(plframe[:PL_HEADER_LEN])
    return __decode_pls(header_bits[SOF_LEN:], isX)


if __name__ == "__main__":
    rn = getScramblingSequence(0)
    print(rn[:20])
    rn = getScramblingSequence(1)
    print(rn[:20])
    rn = getScramblingSequence(2)
    print(rn[:20])
    # pls code = 138
    plscode = __decode_pls(
        np.array([0, 1, 0, 0, 1, 1, 0, 1, 0, 1, 1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 1, 1, 1, 1, 0, 0, 1, 1, 1, 0, 0, 1, 1, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 1, 1, 0, 1, 1, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 1, 0, 0, 1]),
        isX=True,
    )
    print(plscode)
