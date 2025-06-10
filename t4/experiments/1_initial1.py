import matplotlib
matplotlib.use('Agg') # Fix para erro de backend do matplotlib

import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

def calc_histogram(areas: np.ndarray) -> np.ndarray:
    """
    Plots a histogram of object areas using fixed bins.
    Returns it as an OpenCV RGB image.
    """
    areas = np.asarray(areas)
    bin_edges = [np.min(areas), 1500, 3000, np.max(areas) + 1]

    # Desenha o histograma
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.hist(areas, bins=bin_edges, color="blue", edgecolor="black")
    ax.set_xlabel("Área")
    ax.set_ylabel("Número de Objetos")
    ax.set_xlim(0, np.max(areas) + 500)
    ax.set_xticks(range(0, np.max(areas) + 500, 500))

    # Converte para imagem
    fig.canvas.draw()
    w, h = fig.canvas.get_width_height()
    buf = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8).reshape((h, w, 4))
    img = cv.cvtColor(buf, cv.COLOR_RGBA2BGR)
    plt.close(fig)

    return img

#rgb_img = cv.imread('images/ellipse.png')
rgb_img = cv.imread('images/objetos3.png')

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

def find_convex_hull(contour, img_shape):
    hull = cv.convexHull(contour)

    # Cria uma mascara para calcular area do fecho convexo
    hull_mask = np.zeros(img_shape, dtype=np.uint8)
    cv.drawContours(hull_mask, [hull], -1, 1, thickness=cv.FILLED)
    area = np.sum(hull_mask)

    return hull, area

def line_inside_mask(p1, p2, label_mask, label):
    temp = np.zeros_like(label_mask, dtype=np.uint8)
    cv.line(temp, tuple(p1), tuple(p2), 255, 1) 
    line_pixels = (temp == 255)
    return np.all(label_mask[line_pixels] == label)

def find_minor_axis(mask, major_axis, label, step=1):
    height, width = mask.shape

    # Vetor de suporte sobre o eixo maior
    p1, p2 = major_axis
    major_length = np.linalg.norm(p2 - p1)
    unit_vec = (p2 - p1) / major_length # Vetor unitario apontando de p1 para p2
    perp_vec = np.array([-unit_vec[1], unit_vec[0]])
    cross_vector = perp_vec * major_length / 2  # Vetor perpendicular ao eixo maior e com metade do tamanho

    # Inicializacao das variaveis
    max_dist = 0
    minor = (None, None)
    prev_base = None

    # Itera sobre os pontos da reta com passo step
    for i in np.arange(0, major_length + step, step):
        # Ponto sobre o eixo maior a partir do qual a reta perpendicular sera checada
        line_origin = (p1 + i * unit_vec).astype(int)
        if np.all(prev_base == line_origin):
            continue # Evita calculo desnecessario se ponto ja tiver sido checado
        prev_base = line_origin

        # Calcula os pontos extremos da reta candidata
        pts = (line_origin - cross_vector, line_origin + cross_vector)
        pts = np.clip(pts, [0, 0], [width - 1, height - 1]).astype(np.uint32)

        # Imagem com 1 nas posicoes da reta candidata
        line_mask = np.zeros_like(mask, dtype=np.uint8)
        add_line(line_mask, pts[0], pts[1], color=1)

        # Apenas os pontos da reta dentro do objeto
        pts_inside = np.nonzero(line_mask & (mask == label))
        if len(pts_inside[0]) < 2:
            continue # Em pontos de borda, pode ocorrer de nao ter overlap 

        # Primeiro e ultimo pontos dentro do objeto
        idx1, idx2 = np.argmax(pts_inside[0]), np.argmin(pts_inside[0])
        if pts_inside[0][idx1] == pts_inside[0][idx2]: # Se reta for vertical
            idx1, idx2 = np.argmax(pts_inside[1]), np.argmin(pts_inside[1])
        m1 = np.array([pts_inside[1][idx1], pts_inside[0][idx1]])
        m2 = np.array([pts_inside[1][idx2], pts_inside[0][idx2]])

        # Atualiza se for maior
        dist = np.linalg.norm(m1 - m2)
        if dist > max_dist:
            max_dist = dist
            minor = (m1, m2)
    return minor, max_dist

def calculate_eccentricity(contour, label_mask, label):
    major, major_axis_len = find_major_axis(contour, label_mask, label)
    minor, minor_axis_len = find_minor_axis(label_mask, major, label)
    return major, minor, major_axis_len / minor_axis_len

def find_major_axis(contour, label_mask, label):
    # Inicializacao das variavies
    major = (None, None)
    max_dist = 0

    # Verifica todos os pares de pontos do contorno
    # Se a reta entre eles for maior que a maior ja vista,
    # e estiver dentro do objeto, atualiza o eixo maior
    for i in range(len(contour) - 1):
        p1 = contour[i][0]

        contour_pts = contour[i+1:, 0, :]
        dists = np.linalg.norm(contour_pts - p1, axis=1)

        # Checa apenas os pontos que podem ter distancia maior
        # comecando pela maior distancia
        contour_pts = contour_pts[dists > max_dist]
        dists = dists[dists > max_dist]
        sorted_indices = np.argsort(-dists)
        for idx in sorted_indices:
            p2, dist = contour_pts[idx], dists[idx]
            if line_inside_mask(p1, p2, label_mask, label):
                max_dist = dist
                major = (p1, p2)
                break

    return major, max_dist

def calculate_perimeter(contour):
    return cv.arcLength(contour, closed=True)

def add_line(img, p1, p2, color):
    cv.line(img, tuple(p1), tuple(p2), color, thickness=1)

def add_label_and_color(label_img, labels, label, color):
    # Color the region
    label_img[labels == label] = color

    # Find coordinates of the label
    y_coords, x_coords = np.where(labels == label)

    # Center of mass
    x_center = int(np.mean(x_coords))
    y_center = int(np.mean(y_coords))

    # Text parameters
    text = str(label - 2)
    font = cv.FONT_HERSHEY_PLAIN
    font_scale = 1
    thickness = 1

    # Measure text size
    (text_width, text_height), baseline = cv.getTextSize(text, font, font_scale, thickness)

    # Top-left corner of text so that it's centered
    x_text = x_center - text_width // 2
    y_text = y_center + text_height // 2

    # Create a mask with the text
    text_mask = np.zeros(labels.shape, dtype=np.uint8)
    cv.putText(text_mask, text, (x_text, y_text), font, font_scale, 255, thickness, cv.LINE_8)

    # Check if the text is fully inside the object
    text_inside = np.all((text_mask == 0) | (labels == label))

    #if text_inside:
        # Draw text on image
    cv.putText(label_img, text, (x_text, y_text), font, font_scale, (0, 0, 0), thickness, cv.LINE_8)
    #else:
        #x_text, y_text = np.max(x_coords), np.min(y_coords)
        #while labels[y_text][x_text] != label:
        #    y_text += 1
        #    x_text -= 1
        #cv.putText(label_img, text, (x_text, y_text), font, 0.6, (0, 0, 0), thickness, cv.LINE_8)

def add_contour(img, contour, color):
    cv.drawContours(img, [contour], -1, color, 1)

label_colors = [(200, 200, 0), (0, 200, 200), (200, 0, 200), (200, 255, 0),]
label_img = np.full(rgb_img.shape, (255, 255, 255), dtype=np.uint8)
hull_img = rgb_img.copy()

info = {}

for i in range(2, num_labels):
    # Contorno e fecho convexo do objeto
    contour = find_contour(labels, i)
    hull, hull_area = find_convex_hull(contour, rgb_img.shape[:2])
    # Calcula as propriedades
    area = np.sum(np.where(labels == i, 1, 0))
    major, minor, eccentricity = calculate_eccentricity(contour, labels, i)
    info[i] = {
        "area": area,
        "perimetro": f"{calculate_perimeter(contour):.6f}",
        "excentricidade": f"{eccentricity:.6f}",
        "solidez": f"{area / hull_area:.6f}"
    }
    # Desenha as informacoes nas imagens
    add_label_and_color(label_img, labels, i, label_colors[i % len(label_colors)])
    add_line(rgb_img, major[0], major[1], (0, 0, 0))
    add_line(rgb_img, minor[0], minor[1], (0, 0, 0))
    add_contour(rgb_img, contour, (0, 0, 0))
    add_contour(hull_img, hull, (0, 0, 255))

print(f"numero de regioes: {num_labels - 2}")
for label, metrics in info.items():
    str_metrics = "\t".join([f"{m}: {v}" for m, v in metrics.items()])
    print(f"Regiao {label - 2}:\t{str_metrics}")

areas = np.array([info[label]["area"] for label in info.keys()])
bin_edges = [1500, 3000]  # → 0: <1500, 1: 1500–2999, 2: ≥3000
bins = np.digitize(areas, bin_edges)
bin_counts = [np.sum(bins == i) for i in range(len(bin_edges) + 1)]
str_bins = ["pequenas", "medias", "grandes"]
for bin in range(len(bin_counts)):
    print(f"numero de regioes {str_bins[bin]}:\t{bin_counts[bin]}")

windows = {
    "Eixos principais": rgb_img,
    "Binaria": bin_img,
    "Bordas": border_img,
    "Labels": label_img,
    "Fecho convexo": hull_img,
    "Histograma": calc_histogram(areas),
}
w, h = rgb_img.shape[1], rgb_img.shape[0]
screen_w, screen_h = 1440, 1080
offset_x, offset_y = 50, 50
columns = screen_w // (w + offset_x) - 1
for i, (title, image) in enumerate(windows.items()):
    cv.imshow(title, image)
    x = (i % columns) * (w + offset_x)
    y = (i // columns) * (h + offset_y)
    cv.moveWindow(title, x, y)
cv.waitKey(0)
cv.destroyAllWindows()