import argparse
import cv2 as cv
from arrays import get_array, get_kernels
from helper_functions import get_image_path, save_images, display_images
import numpy as np
import os

### 01
def esboco_lapis(image_name: str = "watch.png") -> dict[str, np.ndarray]:
    image = cv.imread(get_image_path(image_name), cv.IMREAD_COLOR)

    # Valores fixos para o filtro gaussiano
    ksize = (21, 21)
    sigma = 3.5

    # Aplica o efeito de esboco a lapis
    gray_image = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    blurred_image = cv.GaussianBlur(gray_image, ksize, sigma)
    result = cv.divide(gray_image, blurred_image, scale=255).astype(np.uint8)

    return {"esboco_lapis": result}


### 02
def ajuste_brilho(image_name: str = "baboon_monocromatica.png") -> dict[str, np.ndarray]:
    image = cv.imread(get_image_path(image_name), cv.IMREAD_GRAYSCALE)
    results = {} # guarda os resultados

    # Normalizacao para [0, 1]
    image = np.divide(image, 255)

    # Ajuste de brilho com diferentes valores de alpha
    alpha_list = [0.5, 2.5, 4.5,]
    for i, alpha in enumerate(alpha_list):
        result = np.floor(np.power(image, 1/alpha) * 255).astype(np.uint8)
        results[f"ajuste_brilho_{str(alpha).replace('.', '_')}"] = result
    return results


### 03
def mosaico(image_name: str = "baboon_monocromatica.png") -> dict[str, np.ndarray]:
    image = cv.imread(get_image_path(image_name), cv.IMREAD_GRAYSCALE)

    # Mapeamento dos blocos (subtraindo 1 para ficar de 0 a 15)
    mapping = {
        0: 5,  1: 10,  2:  12, 3:  2,
        4: 7,  5: 15,  6:  0,  7:  8,
        8: 11, 9: 13,  10: 1,  11: 6,
        12: 3, 13: 14, 14: 9,  15: 4,
    }

    # Calcula as dimensões dos blocos
    nrows, ncols = image.shape
    block_height = nrows // 4
    block_width  = ncols // 4

    # Aplica o efeito de mosaico
    mosaic = np.zeros((nrows, ncols), dtype=np.uint8) # mosaico inicializado com 0's
    for new_block, prev_block in mapping.items():
        # Calcula origens (ponto superior esquerdo) dos blocos
        prev_origin_row, prev_origin_col = (block_height * (prev_block // 4), block_width * (prev_block % 4))
        new_origin_row, new_origin_col = (block_height * (new_block // 4), block_width * (new_block % 4))
        # Copia o bloco da posicao anterior para a nova posicao
        mosaic[new_origin_row:(new_origin_row+block_height), new_origin_col:(new_origin_col+block_width)] = \
            image[prev_origin_row:(prev_origin_row+block_height), prev_origin_col:(prev_origin_col+block_width)]

    return {"mosaico": mosaic}


### 04
def alteracao_cores(image_name: str = "watch.png") -> dict[str, np.ndarray]:
    image = cv.imread(get_image_path(image_name), cv.IMREAD_COLOR_RGB)

    # Transformacao de cores
    transf_matrix = get_array("sepia") # matriz de transformacao
    transf = np.dot(image, transf_matrix.T) # aplica a transformacao
    result = cv.normalize(transf, None, 0, 255, cv.NORM_MINMAX).astype(np.uint8) # normaliza para [0, 255]
    cv.cvtColor(result, cv.COLOR_RGB2BGR, dst=result) # cv2 usa BGR

    return {"alteracao_cores": result}


### 05
def transformacao_imagens_coloridas(image_name: str = "watch.png") -> dict[str, np.ndarray]:
    image = cv.imread(get_image_path(image_name), cv.IMREAD_COLOR)
    results = {} # guarda os resultados

    # Exercicio a)
    transf_matrix = get_array("sepia") # matriz de transformacao
    transf = np.dot(image, transf_matrix.T) # aplica a transformacao
    transf[transf > 255] = 255 # limita a [0, 255]
    result = cv.cvtColor(transf.astype(np.uint8), cv.COLOR_RGB2BGR) # cv2 usa BGR
    results["alteracao_cores"] = result

    # Exercicio b)
    transf_matrix = get_array("rgbtogray") # matriz de transformacao
    result = np.dot(image, transf_matrix.T).astype(np.uint8) # aplica a transformacao
    results["transformacao_banda"] = result

    return results


### 06
def plano_bits(image_name: str = "baboon_monocromatica.png") -> dict[str, np.ndarray]:
    image = cv.imread(get_image_path(image_name), cv.IMREAD_GRAYSCALE)
    results = {} # guarda os resultados

    # Calcula cada plano de bit
    plain_list = [0, 1, 2, 3, 4, 5, 6, 7]
    for plain in plain_list:
        result = np.bitwise_and(image, 0b1 << plain) # image & (1 << plain)
        result[result > 0] = 255 # converte de 2**plain para 255
        results[f"plano_bit_{plain}"] = result
    
    return results


### 07
def combinacao_imagens(image_name_1: str = "baboon_monocromatica.png", image_name_2: str = "butterfly.png") -> dict[str, np.ndarray]:
    image_1 = cv.imread(get_image_path(image_name_1), cv.IMREAD_GRAYSCALE)
    image_2 = cv.imread(get_image_path(image_name_2), cv.IMREAD_GRAYSCALE)
    results = {} # guarda os resultados

    weights = [
        (0.2, 0.8),
        (0.5, 0.5),
        (0.8, 0.2),
    ]
    # Aplica a combinacao de imagens
    for i, (w1, w2) in enumerate(weights):
        result = (np.floor(w1*image_1 + w2*image_2)).astype(np.uint8)
        results[f"combinacao_{i+1}"] = result

    return results


### 08
def transformacao_intensidade(image_name: str = "city.png") -> dict[str, np.ndarray]:
    image = cv.imread(get_image_path(image_name), cv.IMREAD_GRAYSCALE)
    nrows, ncols = image.shape

    # Item b)
    negativo = np.subtract(255, image)

    # Item c)
    transformada = np.clip(image, 100, 200)
    cv.normalize(transformada, transformada, 0, 255, cv.NORM_MINMAX) # normaliza para [0, 255] inplace

    # Item d)
    linhas_pares_invertidas = image.copy()
    linhas_pares_invertidas[::2] = np.fliplr(linhas_pares_invertidas[::2])

    # Item e)
    reflexao_linhas = image.copy()
    if nrows & 1 == 0: # numero par de linhas
        reflexao_linhas[nrows//2:] = np.flipud(reflexao_linhas[:nrows//2])
    else: # numero impar (parte flipada comeca uma linha na frente)
        reflexao_linhas[nrows//2 + 1:] = np.flipud(reflexao_linhas[:nrows//2])

    # Item f)
    reflexao_vertical = np.flipud(image)

    results = {
        "negativo": negativo,
        "transformada": transformada,
        "linhas_pares_invertidas": linhas_pares_invertidas,
        "reflexao_linhas": reflexao_linhas,
        "espelhamento_vertical": reflexao_vertical
    }

    return results


### 09
def quantizacao_imagens(image_name: str = "baboon_monocromatica.png") -> dict[str, np.ndarray]:
    image = cv.imread(get_image_path(image_name), cv.IMREAD_GRAYSCALE)
    results = {} # guarda os resultados

    # Aplica a quantizacao
    levels_list = [128, 32, 16, 8, 4, 3, 2]
    for levels in levels_list:
        # Quantizacao uniforme
        delta = 256 / levels
        result = np.floor(image / delta) * delta
        # Normalizacao e conversao para uint8
        cv.normalize(result, result, 0, 255, cv.NORM_MINMAX) # inplace
        results[f"quantizacao_{levels}"] = result.astype(np.uint8)

    return results


### 10
def filtragem_imagens(image_name: str = "waterfall_cinza.png") -> dict[str, np.ndarray]:
    image = cv.imread(get_image_path(image_name), cv.IMREAD_GRAYSCALE)
    kernel_dict = get_kernels() # chaves: h1 até h11
    results = {} # guarda os resultados

    # Realiza padding de 4 pixels com replicacao da borda
    pad_r = pad_c = 4
    padded = cv.copyMakeBorder(
        image, pad_r, pad_r, pad_c, pad_c,
        cv.BORDER_REPLICATE
        )

    # Realiza a convolucao com cada filtro
    for kernel_name, kernel in kernel_dict.items():
        kernel = cv.flip(kernel, -1) # inversao horizontal e vertical
        result = cv.filter2D(padded, -1, kernel) # convolucao (ddepth = -1 mantem o tipo de dado)
        result = result[pad_r:-pad_r, pad_c:-pad_c] # remove o padding
        results[kernel_name] = result.astype(np.uint8)

    # Combina os resultados das aplicacoes de h3 e h4
    img3_squared = np.power(results["h3"], 2, dtype=np.uint16)
    img4_squared = np.power(results["h4"], 2, dtype=np.uint16)
    combination  = np.sqrt(img3_squared + img4_squared)
    results["combinacao"] = combination.astype(np.uint8)

    return results

def process_and_handle_exercise(
        exercise_function: callable,
        exercise_number: int,
        image_names: list[str],
        save: bool,
        display: bool
        ) -> None:
    """
    Processa a imagem usando a funcao exercise_function e lida com o resultado.
    Se o resultado for um dicionario, salva e/ou exibe as imagens.
    Se o resultado for None, encerra.
    """
    # Verifica quais imagens existem
    if image_names:
        for image_name in image_names:
            if not os.path.isfile(get_image_path(image_name)):
                print(f"\033[91mImage {image_name} not found.\033[0m") # vermelho
                image_names.remove(image_name)
                if not image_names:  # encerra se nenhuma imagem for encontrada
                    print("\033[91mNo valid images found.\033[0m") # vermelho
                    return
    # Caso especial para o exercicio 7, que combina duas imagens
    if image_names and exercise_number == 7:
        for i in range(0, len(image_names), 2):
            image_name_1 = image_names[i]
            if i + 1 < len(image_names):
                image_name_2 = image_names[i + 1]
                result = exercise_function(image_name_1, image_name_2)
            else:
                result = exercise_function(image_name_1)
            if not handle_result(result, exercise_number, save, display):
                return
    # Exercicios que requerem uma imagens
    else: 
        if image_names:
            for image_name in image_names:
                result = exercise_function(image_name)
                if not handle_result(result, exercise_number, save, display):
                    return
        else:  # usa a imagem padrao se nenhuma for fornecida
            result = exercise_function()
            if not handle_result(result, exercise_number, save, display):
                return

def handle_result(
        result: dict[str, np.ndarray],
        exercise_number: int,
        save: bool,
        display: bool
        ) -> bool:
    """
    Lida com o resultado da funcao de exercicio.
    Se o resultado for um dicionario, salva e/ou exibe as imagens.
    Se o resultado for None, encerra.
    """
    if result:
        if save:
            save_images(result, exercise_number)
        if display:
            return display_images(result) # retorna falso se o usuario pressionar 'n'
    return True

def main(args: argparse.Namespace) -> None:
    """
    Funcao principal que processa os argumentos e executa os exercicios.
    """
    exercises = {
        1: esboco_lapis,
        2: ajuste_brilho,
        3: mosaico,
        4: alteracao_cores,
        5: transformacao_imagens_coloridas,
        6: plano_bits,
        7: combinacao_imagens,
        8: transformacao_intensidade,
        9: quantizacao_imagens,
        10: filtragem_imagens,
    }
    if not (args.d or args.s): # acao nao especificada
        print("\033[91mNo action specified. Use -s to save images and/or -d to display them.\033[0m") # vermelho
        return
    
    if not args.e: # exercicios fora do intervalo
        print("\033[91mAllowed exercises are 1-10.\033[0m") # vermelho

    for exercise_number in args.e:
        print(f"\033[94mProcessing exercise {exercise_number}...\033[0m") # azul
        try:
            process_and_handle_exercise(
                exercises[exercise_number],
                exercise_number,
                args.i,
                args.s,
                args.d
            )
        except Exception as e: # programa nao encerra quando um exercicio falha
            print(f"\033[91mError processing exercise {exercise_number}: {e}\033[0m") # vermelho
        print("\033[92mDone\033[0m") # verde

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Process images with various exercises.')
    parser.add_argument('-i', nargs='+', type=str, help='List of image names (or all)')
    parser.add_argument('-e', nargs='+', type=int, help='List of exercise numbers (1-10) - default: all')
    parser.add_argument('-s', action='store_true', help='Save processed images')
    parser.add_argument('-d', action='store_true', help='Display processed images')
    args = parser.parse_args()

    # Todas as imagens
    if args.i and args.i[0].lower() == 'all':
        args.i = [file for file in os.listdir(get_image_path('')) if not file.startswith('.')]
    # Todos os exercicios por padrao, e limita entrada a 1-10
    args.e = range(1, 11) if not args.e else [ex for ex in args.e if 1 <= ex <= 10]

    main(args)