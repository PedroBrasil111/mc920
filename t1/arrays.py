import numpy as np

SEPIA = [
    [0.393, 0.769, 0.189],
    [0.349, 0.686, 0.168],
    [0.272, 0.534, 0.131]
]

RGBTOGRAY = [
    [0.2989, 0.5870, 0.1140],
]

h1 = np.array(
    [[0, 0, -1, 0, 0],
     [0, -1, -2, -1, 0],
     [-1, -2, 16, -2, -1],
     [0, -1, -2, -1, 0],
     [0, 0, -1, 0, 0]], dtype=np.float32
)

h2 = np.array(
    [[1, 4, 6, 4, 1],
     [4, 16, 24, 16, 4],
     [6, 24, 36, 24, 6],
     [4, 16, 24, 16, 4],
     [1, 4, 6, 4, 1]], dtype=np.float32
) / 256

h3 = np.array(
    [[-1, 0, 1],
     [-2, 0, 2],
     [-1, 0, 1]], dtype=np.float32
)

h4 = np.array(
    [[-1, -2, -1],
     [0, 0, 0],
     [1, 2, 1]], dtype=np.float32
)

h5 = np.array(
    [[-1, -1, -1],
     [-1, 8, -1],
     [-1, -1, -1]], dtype=np.float32
)

h6 = np.array(
    [[1, 1, 1],
     [1, 1, 1],
     [1, 1, 1]], dtype=np.float32
) / 9

h7 = np.array(
    [[-1, -1, 2],
     [-1, 2, -1],
     [2, -1, -1]], dtype=np.float32
)

h8 = np.array(
    [[2, -1, -1],
     [-1, 2, -1],
     [-1, -1, 2]], dtype=np.float32
)

h9 = np.identity(9, dtype=np.float32) / 9

h10 = np.array(
    [[-1, -1, -1, -1, -1],
     [-1, 2, 2, 2, -1],
     [-1, 2, 8, 2, -1],
     [-1, 2, 2, 2, -1],
     [-1, -1, -1, -1, -1]], dtype=np.float32
) / 8

h11 = np.array(
    [[-1, -1, 0],
     [-1, 0, 1],
     [0, 1, 1]], dtype=np.float32
)

array_lookup = {
    "sepia": SEPIA,
    "rgbtogray": RGBTOGRAY,
    "h1": h1,
    "h2": h2,
    "h3": h3,
    "h4": h4,
    "h5": h5,
    "h6": h6,
    "h7": h7,
    "h8": h8,
    "h9": h9,
    "h10": h10,
    "h11": h11,
}

def get_array(array_name: str) -> np.ndarray:
    if array_name.lower() in array_lookup:
        return np.array(array_lookup[array_name.lower()], dtype=np.float32)
    else:
        raise ValueError(f"Unknown kernel name: {array_name}")

def get_kernels() -> dict[str, np.ndarray]:
    return {name: get_array(name) for name in array_lookup.keys() if name[0] == "h"}