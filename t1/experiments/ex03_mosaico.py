import cv2 as cv
import os
import numpy as np
import helper_functions as F

image_path = F.get_image_path("baboon_monocromatica.png")
image = cv.imread(image_path, cv.IMREAD_GRAYSCALE)

nrows, ncols = image.shape
### Blocos devem ter dimensoes multiplas de 4
if nrows % 4 != 0 or ncols % 4 != 0:
    nrows = (nrows // 4) * 4
    ncols = (ncols // 4) * 4
    cv.resize(image, (ncols, nrows), interpolation=cv.INTER_AREA)
block_height = nrows // 4
block_width  = ncols // 4

mapping = {
    0: 5,  1: 10,  2:  12, 3:  2,
    4: 7,  5: 15,  6:  0,  7:  8,
    8: 11, 9: 13,  10: 1,  11: 6,
    12: 3, 13: 14, 14: 9,  15: 4,
}

mosaic = np.empty((nrows, ncols), dtype=np.uint8)
for new_block, prev_block in mapping.items():
    prev_origin_row, prev_origin_col = (block_height * (prev_block // 4), block_width * (prev_block % 4))
    new_origin_row, new_origin_col = (block_height * (new_block // 4), block_width * (new_block % 4))
    mosaic[new_origin_row:(new_origin_row+block_height), new_origin_col:(new_origin_col+block_width)] = \
        image[prev_origin_row:(prev_origin_row+block_height), prev_origin_col:(prev_origin_col+block_width)]

cv.imshow("Original", image)
cv.imshow("Mosaico", mosaic)
cv.waitKey(0)

""""
\subsubsection{Implementação inicial}
O método escolhido para representar as trocas de blocos em código foi a criação de um dicionário no qual os pares chave-valor equivalem ao mapeamento "bloco novo-bloco antigo":

\definecolor{LightGray}{gray}{0.9}
\begin{minted}[
    frame=lines,
    framesep=2mm,
    baselinestretch=1.2,
    bgcolor=LightGray,
    fontsize=\footnotesize,
    linenos
]{python}
mapping = {
    0: 5,  1: 10,  2:  12, 3:  2,
    4: 7,  5: 15,  6:  0,  7:  8,
    8: 11, 9: 13,  10: 1,  11: 6,
    12: 3, 13: 14, 14: 9,  15: 4,
}
\end{minted}

Note que foi subtraído 1 da posição dos blocos na imagem nesse mapeamento, a fim de que seja possível calcular a posição da origem (pixel superior esquerdo) de cada bloco a partir das seguintes relações:
\[
r = h_b*(i // 4) \text{ , }
c = w_b*(i \% 4)
\]

Onde $i$ é o índice do bloco (iniciado em 0), $h_b$ e $w_b$ são a altura e a largura dos blocos, respectivamente, e $a = (r,c)$ é a posição da origem do bloco. Utilizou-se // para representar a divisão exata (que retorna apenas o quociente) e \% para representar o resto da divisão.

Dessa forma, a posição do ponto inferior direito do bloco é $b = (r+h_b, c+w_b)$. Conhecendo $a$ e $b$, é possível obter um bloco rapidamente em código realizando um \textit{slicing} da imagem. Assim, o resultado final pode ser atingido a partir da iteração sobre os pares do mapeamento, com \textit{slicings} para posicionar os blocos em suas novas posições.

Vale ressaltar que a implementação não foi feita \textit{inplace}: primeiro é criada uma imagem vazia, a qual é alterada durante a iteração sobre o mapeamento, com base na imagem original. Caso contrário, 
"""