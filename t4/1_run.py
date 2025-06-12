from argparse import ArgumentParser
from measurement_methods import *
from helper_functions import (
    get_image_path, save_images, display_images,
)

def handle_args(args):
    """
    Trata os argumentos da linha de comando.
    """
    if not args.image:
        args.image = "objetos3.png"
    else:
        if not os.path.exists(get_image_path(args.image)):
            raise ValueError(f"Image not found: {args.image}")
    if not (args.save or args.display):
        raise ValueError("No action specified. Use -s to save images and/or -d to display them.")

def handle_results(
        results: dict[str, ndarray],
        exercise: str,
        save: bool,
        display: bool
    ) -> bool:
    """
    Salva e/ou exibe as imagens resultantes.
    """
    if save:
        save_images(results, exercise)
        print("\033[92mResults saved\033[0m")
    if display:
        print("\033[93mDisplaying results. Press any key to continue or 'q' to exit.\033[0m")
        return display_images(results, single=True)
    return True

def run(args):
    rgb_img = cv.imread(get_image_path(args.image), cv.IMREAD_COLOR)

    # Imagem binaria
    mask = build_mask(rgb_img)
    bin_img = mask * 255
    # Bordas
    bin_border = find_borders(mask)
    border_img = bin_border * 255
    # Encontra componentes (fundo: label 0)
    num_labels, labels = cv.connectedComponents(1 - mask, connectivity=8)
    # Ajusta labels dos objetos para comecar de 0
    labels -= 1
    num_labels -= 1

    # Imagems nas quais serao desenhadas as informacoes
    label_img = np.full(rgb_img.shape, (255, 255, 255), dtype=np.uint8)
    hull_img = rgb_img.copy()
    label_colors = [(198, 196, 152), (198, 139, 153), (170, 198, 153), (241, 220, 98),]
    info = {}

    # Calculo das propriedades para cada label
    for i in range(num_labels):
        contour = find_contour(labels, i)
        hull, hull_area = find_convex_hull(contour, rgb_img.shape[:2])
        area = calculate_area(labels, i)
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
    print(f"numero de regioes: {num_labels}")
    for label, metrics in info.items():
        str_metrics = "\t".join([f"{m}: {v:<8}" for m, v in metrics.items()])
        print(f"Regiao {label}:\t{str_metrics}")

    # Calcula o histograma de areas e imprime os resultados
    areas = np.array([info[label]["area"] for label in info.keys()])
    hist, hist_img = area_histogram(areas)
    str_bins = ["pequenas", "medias", "grandes"]
    for bin in range(len(str_bins)):
        print(f"numero de regioes {str_bins[bin]}:\t{hist[bin]}")

    # Exibe as imagens
    results = {
        "binaria": bin_img,
        "labels": label_img,
        "bordas": border_img,
        "fechos convexos": hull_img,
        "eixos principais e elipses": rgb_img,
        "histograma": hist_img,
    }
    handle_results(results, "01", args.save, args.display)

if __name__ == "__main__":
    args = ArgumentParser("Image measurements of objects")
    args.add_argument("-i", "--image", type=str, help="List of image names (with extension) - default: objetos3.png")
    args.add_argument("-s", "--save", action="store_true", help="Save the images")
    args.add_argument("-d", "--display", action="store_true", help="Display the images")
    args = args.parse_args()

    # Trata os argumentos da linha de comando
    try:
        handle_args(args)
    except ValueError as e:
        print(f"\033[91mError: {e}\033[0m")
        exit(1)

    # Executa o processamento
    run(args)