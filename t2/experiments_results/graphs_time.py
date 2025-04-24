import matplotlib
matplotlib.use('Agg')  # Evita erro de backend GUI

import matplotlib.pyplot as plt

# Flag para usar escala logarítmica
usar_escala_log = False

# Dados fictícios (substitua pelos reais)
tamanhos_imagem = [64, 128, 256, 512, 1024, 2048]
tempos_gray_slice = [
0.026413440704345703 ,
0.10518383979797363 ,
0.42850518226623535 ,
1.7985835075378418 ,
7.0238800048828125 ,
28.375667333602905 ,
]

# Tempos para RGB
tempos_rgb_slice = [
0.032613277435302734 ,
0.13045048713684082 ,
0.5261874198913574 ,
2.1269326210021973 ,
8.486279964447021 ,
34.273688316345215 ,
]

tempos_gray_for = [
0.026068687438964844 ,
0.10533428192138672 ,
0.42127490043640137 ,
1.7588698863983154 ,
7.165060758590698 ,
28.617188692092896 ,
]


tempos_rgb_for = [
0.05777239799499512 ,
0.2517843246459961 ,
0.9556105136871338 ,
3.8323535919189453 ,
15.343618392944336 ,
63.07882809638977 ,
]

# RGB
plt.figure(figsize=(8, 4))
plt.plot(tamanhos_imagem, tempos_rgb_for, marker='o', label='for loop')
plt.plot(tamanhos_imagem, tempos_rgb_slice, marker='s', label='slicing')
if usar_escala_log:
    plt.yscale('log')

# Define os ticks manualmente no eixo X
plt.xticks(tamanhos_imagem)  # Define os ticks no eixo X apenas nos pontos desejados

plt.xlabel('Dimensões da imagem (px)', fontsize=12)
plt.ylabel('Tempo de execução (s)', fontsize=12)
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig('tempo_rgb.png')

# Grayscale
plt.figure(figsize=(8, 4))
plt.plot(tamanhos_imagem, tempos_gray_for, marker='o', label='for loop')
plt.plot(tamanhos_imagem, tempos_gray_slice, linestyle='--', label='slicing')
if usar_escala_log:
    plt.yscale('log')

# Define os ticks manualmente no eixo X
plt.xticks(tamanhos_imagem)  # Define os ticks no eixo X apenas nos pontos desejados

plt.xlabel('Dimensões da imagem (px)', fontsize=12)
plt.ylabel('Tempo de execução (s)', fontsize=12)
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig('tempo_grayscale.png')

# Grayscale
plt.figure(figsize=(8, 4))
plt.plot(tamanhos_imagem, tempos_rgb_slice, color='red', marker='o', label='RGB')
plt.plot(tamanhos_imagem, tempos_gray_slice, color='gray', marker = 'o', label='Escala de cinza')
if usar_escala_log:
    plt.yscale('log')

# Define os ticks manualmente no eixo X
plt.xticks(tamanhos_imagem)  # Define os ticks no eixo X apenas nos pontos desejados

plt.xlabel('Dimensões da imagem (px)', fontsize=12)
plt.ylabel('Tempo de execução (s)', fontsize=12)
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig('tempos_rgb_vs_grayscale.png')
