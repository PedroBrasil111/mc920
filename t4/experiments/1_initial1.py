import cv2 as cv
import numpy as np

THRESHOLD = 5e-2

rgb_img = cv.imread('images/ellipse.png')

# Em mask, fundo tem valor 1 e objetos 0
# A imagem binaria bin_img tem fundo branco (255) e objetos pretos (0)
mask = np.all(rgb_img == 255, axis=2).astype(np.uint8)
bin_img = 255 * mask

# Calcula bordas (vizinhanca-4)
neighbor_kernel = np.array([[0, 1, 0],
                            [1, 0, 1],
                            [0, 1, 0]], dtype=np.uint8)
neighbor_count = cv.filter2D(mask, -1, neighbor_kernel)
bin_border = np.where((mask == 1) & (neighbor_count < 4), 0, 1).astype(np.uint8)
border_img = bin_border * 255

# Encontra componentes
num_labels, labels = cv.connectedComponents(bin_border, connectivity=4)

def find_contour(label_mask, label):
    component_mask = (label_mask == label).astype(np.uint8)
    contours, _ = cv.findContours(component_mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    if len(contours) == 0:
        return None
    return contours[0]

def calculate_area(contour):
    return np.sum(contour)

def find_convex_hull(contour, img_shape):
    hull = cv.convexHull(contour)

    # Cria uma mascara para calcular area do fecho convexo
    hull_mask = np.zeros(img_shape, dtype=np.uint8)
    cv.drawContours(hull_mask, [hull], -1, 1, thickness=cv.FILLED)
    area = np.sum(hull_mask)

    return hull, area

def segment_inside_mask(p1, p2, label_mask, label):
    temp = np.zeros_like(label_mask, dtype=np.uint8)
    cv.line(temp, tuple(p1), tuple(p2), 255, 1) 
    line_pixels = (temp == 255)
    return np.all(label_mask[line_pixels] == label)

def find_minor_axis_length(mask, major_axis, label, step=1):
    p1, p2 = major_axis
    major_vec = p2 - p1
    major_vec = major_vec / np.linalg.norm(major_vec)
    perp_vec = np.array([-major_vec[1], major_vec[0]])  # rotate 90°

    height, width = mask.shape
    max_len = 0
    best_line = None

    # sample along the major axis
    num_samples = int(np.linalg.norm(p2 - p1))
    for i in range(0, num_samples + 1, step):
        base = p1 + i * major_vec
        base = base.astype(int)
        if not (0 <= base[1] < height and 0 <= base[0] < width):
            continue
        if mask[base[1], base[0]] != label:
            continue

        # scan in + and - perpendicular direction
        length = 0
        for direction in [-1, 1]:
            t = 0
            while True:
                pt = base + direction * t * perp_vec
                x, y = int(round(pt[0])), int(round(pt[1]))
                if 0 <= x < width and 0 <= y < height and mask[y, x] == label:
                    t += 1
                else:
                    break
            length += t

        if length > max_len:
            max_len = length
            best_line = (base - perp_vec * length // 2, base + perp_vec * length // 2)

    return max_len, best_line


def calculate_eccentricity(contour, label_mask, label):
    if len(contour) < 2:
        return 0

    # --- Find major axis ---
    major = (None, None)
    max_dist = 0
    for i in range(len(contour)):
        p1 = contour[i][0]
        for j in range(i + 1, len(contour)):
            p2 = contour[j][0]
            if not segment_inside_mask(p1, p2, label_mask, label):
                continue
            dist = np.linalg.norm(p1 - p2)
            if dist > max_dist:
                max_dist = dist
                major = (p1, p2)

    if major[0] is None:
        return None, None, float("inf")

    p1, p2 = major
    major_vec = p2 - p1
    major_vec_normalized = major_vec / np.linalg.norm(major_vec)
    major_axis_len = np.linalg.norm(major_vec)

    # --- Find minor axis (perpendicular to major) ---
    minor = (None, None)
    minor_axis_len = 0
    min_dot_product = float("inf")
    acutal_min_dot_product = float("inf")
    for i in range(len(contour)):
        pi = contour[i][0]
        for j in range(i + 1, len(contour)):
            pj = contour[j][0]
            vec = pj - pi
            vec_norm = np.linalg.norm(vec)
            if vec_norm == 0:
                continue
            vec_normalized = vec / vec_norm
            dot_product = np.abs(np.dot(major_vec_normalized, vec_normalized))

            if dot_product < acutal_min_dot_product:
                acutal_min_dot_product = dot_product

            if dot_product < min_dot_product:  # nearly perpendicular
                if not segment_inside_mask(pi, pj, label_mask, label):
                    continue
                if vec_norm > minor_axis_len:
                    minor_axis_len = vec_norm
                    minor = (pi, pj)
                    min_dot_product = dot_product

    print(f"Min dot product: {min_dot_product}, Actual min dot product: {acutal_min_dot_product}")

    if minor_axis_len == 0:
        return major, None, float("inf")

    return major, minor, major_axis_len / minor_axis_len

def calculate_perimeter(contour):
    return cv.arcLength(contour, True)

def add_line(img, p1, p2, color):
    cv.line(img, tuple(p1), tuple(p2), color, 1)

def add_label_and_color(label_img, label, color):
    label_img[labels == label] = color
    y, x = np.where(labels == i)
    y_center, x_center = int(np.mean(y)), int(np.mean(x))
    cv.putText(label_img, str(i - 2), (x_center, y_center), cv.FONT_HERSHEY_SIMPLEX, 0.3, (0, 0, 0), 1, cv.LINE_8)

def add_contour(img, contour, color):
    cv.drawContours(img, [contour], -1, color, 1)

label_colors = [(200, 200, 0), (0, 200, 200), (200, 0, 200), (200, 255, 0),]
label_img = np.full(rgb_img.shape, (255, 255, 255), dtype=np.uint8)
hull_img = rgb_img.copy()

info = {}

for i in range(2, num_labels):
    contour = find_contour(labels, i)
    # Calcula as propriedades da regiao
    hull, hull_area = find_convex_hull(contour, rgb_img.shape[:2])
    area = np.sum(np.where(labels == i, 1, 0))
    major, minor, eccentricity = calculate_eccentricity(contour, labels, i)
    info[i] = {
        'area': area,
        'perimeter': calculate_perimeter(contour),
        'eccentricity': eccentricity,
        'solidity': area / hull_area
    }
    # Desenha as regioes e contornos nas imagens
    add_label_and_color(label_img, i, label_colors[i % len(label_colors)])
    add_line(rgb_img, major[0], major[1], (0, 0, 0))
    add_line(rgb_img, minor[0], minor[1], (0, 0, 0))
    add_contour(rgb_img, contour, (0, 0, 0))
    # add_contour(rgb_img, )
    add_contour(hull_img, hull, (0, 0, 255))

print(f"numero de regioes: {num_labels - 2}")
for label, metrics in info.items():
    print(f"Regiao {label - 2}:\tarea: {metrics['area']}\tperimetro: {metrics['perimeter']:.6f}\texcentricidade: {metrics['eccentricity']:.6f}\tsolidez: {metrics['solidity']:.6f}")

windows = {
    'Original': rgb_img,
    'Binaria': bin_img,
    'Bordas': border_img,
    'Fecho': hull_img,
    'Labels': label_img
}
w, h = rgb_img.shape[1], rgb_img.shape[0]
offset_x, offset_y = 50, 50
columns = 3
for i, (title, image) in enumerate(windows.items()):
    cv.imshow(title, image)
    x = (i * (w + offset_x)) % (columns * (w + offset_x))
    y = (i // columns) * (h + offset_y)
    cv.moveWindow(title, x, y)
cv.waitKey(0)
cv.destroyAllWindows()