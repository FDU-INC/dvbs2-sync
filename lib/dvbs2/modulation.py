import numpy as np
import ctypes
import os

from sklearn.cluster import KMeans

__APSK_16_26_45 = None
__QPSK = None


def get_16apsk_26_45():
    """
    Get constellation points of 16APSK 26/45, the radius of outter ring is 1
    """
    global __APSK_16_26_45
    if __APSK_16_26_45 is None:
        gamma = 3.7
        cplx = np.array(
            [
                np.exp(1j * np.pi / 4),
                np.exp(-1j * np.pi / 4),
                np.exp(3j * np.pi / 4),
                np.exp(5j * np.pi / 4),
                np.exp(1j * np.pi / 12),
                np.exp(-1j * np.pi / 12),
                np.exp(11j * np.pi / 12),
                np.exp(13j * np.pi / 12),
                np.exp(5j * np.pi / 12),
                np.exp(-5j * np.pi / 12),
                np.exp(7j * np.pi / 12),
                np.exp(-7j * np.pi / 12),
                np.exp(1j * np.pi / 4) / gamma,
                np.exp(-1j * np.pi / 4) / gamma,
                np.exp(3j * np.pi / 4) / gamma,
                np.exp(5j * np.pi / 4) / gamma,
            ]
        )
        __APSK_16_26_45 = np.vstack((cplx.real, cplx.imag)).T

    return __APSK_16_26_45


def get_qpsk():
    """
    Get constellation points of QPSK, radius of the ring is 1
    """
    global __QPSK
    if __QPSK is None:
        cplx = np.array(
            [
                np.exp(1j * np.pi / 4),
                np.exp(-1j * np.pi / 4),
                np.exp(3j * np.pi / 4),
                np.exp(5j * np.pi / 4),
            ]
        )
        __QPSK = np.vstack((cplx.real, cplx.imag)).T

    return __QPSK


def from_symbols_to_16apsk_26_45(payload: np.ndarray):
    n_mod = 16
    pl = payload / np.max(np.abs(payload))
    pl = np.vstack((pl.real, pl.imag)).T

    demodulator = KMeans(n_clusters=n_mod, n_init=1, init=get_16apsk_26_45()).fit(
        get_16apsk_26_45()
    )

    digital_pl = np.array(demodulator.predict(pl), dtype=np.int8)

    bits_string = ""
    for n in digital_pl:
        bits_string += np.binary_repr(n, width=4)

    return np.array(list(bits_string), dtype=np.int8)


def from_symbols_to_qpsk(pl: np.ndarray):
    n_mod = 4
    pl = pl / np.max(np.abs(pl))
    pl = np.vstack((pl.real, pl.imag)).T
    pl = np.array(pl, dtype=float)

    demodulator = KMeans(n_clusters=n_mod, n_init=1, init=get_qpsk()).fit(get_qpsk())

    digital_pl = np.array(demodulator.predict(pl), dtype=np.int8)

    bits_string = ""
    for n in digital_pl:
        bits_string += np.binary_repr(n, width=2)

    return np.array(list(bits_string), dtype=np.int8)


def from_symbols_to_qpsk_ffi(payload: np.ndarray):
    dirname = os.path.dirname(__file__)
    libdemod = ctypes.WinDLL(dirname + "/libdemod.dll")
    libdemod.demod_qpsk.restype = None
    libdemod.demod_qpsk.argtypes = [
        np.ctypeslib.ndpointer(ctypes.c_double),
        np.ctypeslib.ndpointer(ctypes.c_double),
        ctypes.c_size_t,
        np.ctypeslib.ndpointer(ctypes.c_int8),
    ]

    output = np.zeros((2 * payload.shape[0]), dtype=np.int8)
    real = np.array(payload.real)
    imag = np.array(payload.imag)
    libdemod.demod_qpsk(real, imag, payload.shape[0], output)
    return output


def from_symbols_to_8psk_ffi(payload: np.ndarray):
    dirname = os.path.dirname(__file__)
    libdemod = ctypes.WinDLL(dirname + "/libdemod.dll")
    libdemod.demod_8psk.restype = None
    libdemod.demod_8psk.argtypes = [
        np.ctypeslib.ndpointer(ctypes.c_double),
        np.ctypeslib.ndpointer(ctypes.c_double),
        ctypes.c_size_t,
        np.ctypeslib.ndpointer(ctypes.c_int8),
    ]

    output = np.zeros((3 * payload.shape[0]), dtype=np.int8)
    real = np.array(payload.real)
    imag = np.array(payload.imag)
    libdemod.demod_8psk(real, imag, payload.shape[0], output)
    return output


if __name__ == "__main__":
    maps = get_16apsk_26_45()
    maps = maps[:, 0] + 1j * maps[:, 1]
    print(maps)
    print(from_symbols_to_16apsk_26_45(maps))

    maps = get_qpsk()
    maps = maps[:, 0] + 1j * maps[:, 1]
    print(maps)
    print(from_symbols_to_qpsk(maps))
    print(from_symbols_to_qpsk_ffi(maps))
