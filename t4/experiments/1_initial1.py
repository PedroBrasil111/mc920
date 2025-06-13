import matplotlib
matplotlib.use('Agg') # Fix para erro de backend do matplotlib

import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

#rgb_img = cv.imread('images/ellipse.png')
#rgb_img = cv.imread('images/objetos3.png')

def area_histogram(areas: np.ndarray) -> tuple[list[int], np.ndarray]:
    """
    Retorna a imagem do histograma de areas dos objetos.
    """
    # Define os bins
    mi, ma = np.min(areas), np.max(areas)
    bin_edges = [mi if mi < 1500 else 0, 1500, 3000, ma if ma > 3000 else 4500]

    # Calcula e desenha o histograma
    fig, ax = plt.subplots(figsize=(5, 4))
    hist = ax.hist(areas, bins=bin_edges, color="blue", edgecolor="black")
    ax.set_xlabel("Área")
    ax.set_ylabel("Número de Objetos")
    ax.set_xlim(0, 4500)
    xticklabels = [str(x) for x in bin_edges]

    # Adiciona "..." se limite do ultimo bin for maior que 4500
    if ma > 4500:
        bin_edges = bin_edges[:-1] + [7500/2, 4500]
        xticklabels = xticklabels[:-1] + ["...", str(ma)]
    ax.set_xticks(bin_edges)
    ax.set_xticklabels(xticklabels)

    # Converte para imagem
    fig.canvas.draw()
    w, h = fig.canvas.get_width_height()
    buf = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8).reshape((h, w, 4))
    img = cv.cvtColor(buf, cv.COLOR_RGBA2BGR)
    plt.close(fig)

    return [int(val) for val in hist[0]], img

def build_mask(img: np.ndarray) -> np.ndarray:
    """
    Dada uma imagem RGB com fundo branco,
    Retorna uma mascara binaria onde os objetos sao representados por 1 e o fundo por 0
    """
    return 1 - np.all(img == 255, axis=2).astype(np.uint8)

def borders(bin_obj):
    # Calculo das bordas usando convolucao
    # Um ponto de borda tem menos de 4 vizinhos em vizinhanca-4
    neighbor_kernel = np.array([[0, 1, 0],
                                [1, 0, 1],
                                [0, 1, 0]], dtype=np.uint8)
    neighbor_count = cv.filter2D(bin_obj, -1, neighbor_kernel, borderType=cv.BORDER_REPLICATE)
    bin_border = np.where((bin_obj == 1) & (neighbor_count < 4), 0, 1).astype(np.uint8)
    return bin_border

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

def line_inside_component(p1, p2, label_mask, label):
    line_img = np.zeros_like(label_mask, dtype=np.uint8)
    draw_line(line_img, p1, p2, 255)
    return np.all(label_mask[line_img == 255] == label)

def find_minor_axis(label_mask, major_axis, label, step=1):
    height, width = label_mask.shape

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
        line_mask = np.zeros_like(label_mask, dtype=np.uint8)
        draw_line(line_mask, pts[0], pts[1], color=1)

        # Apenas os pontos da reta dentro do objeto
        pts_inside = np.nonzero(line_mask & (label_mask == label))
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

def calculate_axis_eccentricity(contour, label_mask, label):
    major, major_axis_len = find_major_axis(contour, label_mask, label)
    minor, minor_axis_len = find_minor_axis(label_mask, major, label)
    return major, minor, np.sqrt(1 - minor_axis_len**2 / major_axis_len**2)

def calculate_ellipse_eccentricity(contour):
    """
    Teste 
    """
    if len(contour) < 5:
        return None, np.nan # minimo de 5 pontos

    # Fita a elipse e obtem os eixos menor e maior
    ellipse = cv.fitEllipse(contour)
    (major_axis, minor_axis) = ellipse[1]
    a, b = max(major_axis, minor_axis), min(major_axis, minor_axis)
    eccentricity = np.sqrt(1 - (b ** 2) / (a ** 2)) if a != 0 else np.inf
    return ellipse, eccentricity

def find_major_axis(contour, label_mask, label):
    # Inicializacao das variavies
    major = (None, None)
    max_dist = 0

    # Verifica todos os pares de pontos do contorno
    # para achar o eixo maior (maior segmento interno)
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
            if line_inside_component(p1, p2, label_mask, label):
                max_dist = dist
                major = (p1, p2)
                break

    return major, max_dist

def calculate_perimeter(contour):
    return cv.arcLength(contour, closed=True)

def draw_line(img, p1, p2, color):
    cv.line(img, tuple(p1), tuple(p2), color, thickness=1)

def draw_label_with_color(label_img, labels, label, color):
    # Adiciona cor
    label_img[labels == label] = color

    # Adiciona texto no centro de massa do objeto
    y_coords, x_coords = np.where(labels == label)
    x_center = int(np.mean(x_coords))
    y_center = int(np.mean(y_coords))
    text = str(label)
    font = cv.FONT_HERSHEY_PLAIN
    font_scale = 1
    thickness = 1
    (text_width, text_height), _ = cv.getTextSize(text, font, font_scale, thickness)
    x_text = x_center - text_width // 2
    y_text = y_center + text_height // 2
    cv.putText(label_img, text, (x_text, y_text), font, font_scale, (0, 0, 0), thickness, cv.LINE_8)

def draw_contour(img, contour, color):
    cv.drawContours(img, [contour], -1, color, 1)

def show_images(img_dict, screen_w, screen_h):
    h, w = img_dict[list(img_dict.keys())[0]].shape[:2]
    n = len(img_dict)
    cols = screen_w // w
    rows = int(np.ceil(n / cols))
    offset_x = (screen_w % w) // (cols - 1) if cols > 1 else 0
    offset_y = (screen_h % h) // (rows - 1) if rows > 1 else 0
    max_y = screen_h - h - offset_y
    print(f"Screen size: {screen_w}x{screen_h}, Image size: {w}x{h}, Offset: {offset_x}x{offset_y}, Rows: {rows}, Cols: {cols}")
    for i, (title, img) in enumerate(img_dict.items()):
        cv.imshow(title, img)
        x = (i % cols) * (w + offset_x)
        y = (i // cols) * (h + offset_y)
        if y > max_y:
            y = max_y
        print(x, y)
        cv.moveWindow(title, x, y)
    cv.waitKey(0)

def main():
    rgb_img = cv.imread('images/objetos3.png')
    mask = build_mask(rgb_img)
    bin_img = mask * 255

    bin_border = borders(mask)
    border_img = bin_border * 255

    # Encontra componentes
    num_labels, labels = cv.connectedComponents(1 - mask, connectivity=8)
    labels -= 1 # Ajusta labels para comecar de 0
    
    label_colors = [(198, 196, 152), (198, 139, 153), (170, 198, 153), (241, 220, 98),]
    label_img = np.full(rgb_img.shape, (255, 255, 255), dtype=np.uint8)
    hull_img = rgb_img.copy()

    info = {}

    for i in range(num_labels - 1):
        # Contorno e fecho convexo do objeto
        contour = find_contour(labels, i)
        hull, hull_area = find_convex_hull(contour, rgb_img.shape[:2])

        # Calcula as propriedades
        area = np.sum(np.where(labels == i, 1, 0))
        major, minor, e1 = calculate_axis_eccentricity(contour, labels, i)
        ellipse, e2 = calculate_ellipse_eccentricity(contour)
        info[i] = {
            "area": area,
            "perimetro": f"{calculate_perimeter(contour):.6f}",
            "excentricidade": f"{e1:.6f} || {e2:<8.6f}",
            "solidez": f"{area / hull_area:.6f}",
        }

        # Desenha as informacoes nas imagens
        draw_label_with_color(label_img, labels, i, label_colors[i % len(label_colors)])
        draw_line(rgb_img, major[0], major[1], (0, 0, 0))
        draw_line(rgb_img, minor[0], minor[1], (0, 0, 0))
        draw_contour(hull_img, hull, (0, 0, 0))
        if ellipse:
            cv.ellipse(rgb_img, ellipse, (0, 0, 0), 1)

    # Imprime as informacoes de cada regiao
    print(f"numero de regioes: {num_labels - 2}")
    for label, metrics in info.items():
        str_metrics = "\t".join([f"{m}: {v:<8}" for m, v in metrics.items()])
        print(f"Regiao {label - 2}:\t{str_metrics}")

    # Calcula o histograma de areas
    areas = np.array([info[label]["area"] for label in info.keys()])
    hist, hist_img = area_histogram(areas)
    
    # Imprime regioes por tamanho
    str_bins = ["pequenas", "medias", "grandes"]
    for bin in range(len(str_bins)):
        print(f"numero de regioes {str_bins[bin]}:\t{hist[bin]}")

    # Exibe as imagens
    windows = {
        "Eixos principais e elipse": rgb_img,
        "Histograma": hist_img,
        "Binaria": bin_img,
        "Bordas": border_img,
        "Labels": label_img,
        "Fecho convexo": hull_img,
    }
    show_images(windows, 1440, 1080)
    return
    w, h = rgb_img.shape[1], rgb_img.shape[0]
    screen_w, screen_h = 1440, 1080
    offset_x, offset_y = 50, 50
    columns = screen_w // (w + offset_x) - 1
    for i, (title, image) in enumerate(windows.items()):
        cv.imshow(title, image)
        x = (i % columns) * (w + offset_x)
        if i < columns:
            y = 0
        else:
            y = ((i // columns) % (screen_h - (h + offset_y))) * (h + offset_y)
        cv.moveWindow(title, x, y)
    cv.waitKey(0)
    cv.destroyAllWindows()

if __name__ == "__main__":
    main()