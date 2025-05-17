import numpy as np

FLOYD_STEINBERG = np.array([
    [0, 0, 7],
    [3, 5, 1]
], dtype=np.float16) / 16

STEVENSON_ARCE = np.array([
    [ 0,  0,  0,  0,  0, 32,  0],
    [12,  0, 26,  0, 30,  0, 16],
    [ 0, 12,  0, 26,  0, 12,  0],
    [ 5,  0, 12,  0, 12,  0,  5]
], dtype=np.float16) / 200

BURKES = np.array([
    [0, 0, 0, 8, 4],
    [2, 4, 8, 4, 2]
], dtype=np.float16) / 32

SIERRA = np.array([
    [0, 0, 0, 5, 3],
    [2, 4, 5, 4, 2],
    [0, 2, 3, 2, 0]
], dtype=np.float16) / 32

STUCKI = np.array([
    [0, 0, 0, 8, 4],
    [2, 4, 8, 4, 2],
    [1, 2, 4, 2, 1]
], dtype=np.float16) / 42

JARVIS_JUDICE_NINKE = np.array([
    [0, 0, 0, 7, 5],
    [3, 5, 7, 5, 3],
    [1, 3, 5, 3, 1]
], dtype=np.float16) / 48

CUSTOM = np.array([
    [0, 0, 4],
    [4, 4, 4]
], dtype=np.float16) / 16

array_lookup = {
    "floyd_steinberg": FLOYD_STEINBERG,
    "stevenson_arce": STEVENSON_ARCE,
    "burkes": BURKES,
    "sierra": SIERRA,
    "stucki": STUCKI,
    "jarvis_judice_ninke": JARVIS_JUDICE_NINKE,
    "custom": CUSTOM,
}

def get_matrices() -> dict[str, np.ndarray]:
    """
    Retorna todos os arrays disponiveis.
    """
    return array_lookup

if __name__ == "__main__":
    # teste
    for arr in array_lookup:
        print(f"{arr}: {np.sum(array_lookup[arr])}")