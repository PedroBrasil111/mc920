import numpy as np

FLOYD_STEINBERG = np.array([
    [0, 0, 7],
    [3, 5, 1]
], dtype=np.float32) / 16

STEVENSON_ARCE = np.array([
    [ 0,  0,  0,  0,  0, 32,  0],
    [12,  0, 26,  0, 30,  0, 16],
    [ 0, 12,  0, 26,  0, 12,  0],
    [ 5,  0, 12,  0, 12,  0,  5]
], dtype=np.float32) / 200

BURKES = np.array([
    [0, 0, 0, 8, 4],
    [2, 4, 8, 4, 2]
], dtype=np.float32) / 32

SIERRA = np.array([
    [0, 0, 0, 5, 3],
    [2, 4, 5, 4, 2],
    [0, 2, 3, 2, 0]
], dtype=np.float32) / 32

STUCKI = np.array([
    [0, 0, 0, 8, 4],
    [2, 4, 8, 4, 2],
    [1, 2, 4, 2, 1]
], dtype=np.float32) / 42

JARVIS_JUDICE_NINKE = np.array([
    [0, 0, 0, 7, 5],
    [3, 5, 7, 5, 3],
    [1, 3, 5, 3, 1]
], dtype=np.float32) / 48

array_lookup = {
    "floyd_steinberg": FLOYD_STEINBERG,
    "stevenson_arce": STEVENSON_ARCE,
    "burkes": BURKES,
    "sierra": SIERRA,
    "stucki": STUCKI,
    "jarvis_judice_ninke": JARVIS_JUDICE_NINKE
}

def get_array(array_name: str) -> np.ndarray:
    """
    Retorna o array correspondente ao nome fornecido.
    """
    lowered = array_name.lower()
    if lowered in array_lookup:
        return np.array(array_lookup[lowered], dtype=np.float32)
    raise ValueError(f"Unknown array name: {array_name}")

def get_all_arrays() -> dict[str, np.ndarray]:
    """
    Retorna todos os arrays disponiveis.
    """
    return array_lookup

if __name__ == "__main__":
    # teste
    for arr in array_lookup:
        print(f"{arr}: {np.sum(array_lookup[arr])}")