import cv2
import numpy as np

def calcular_metricas(imagem_ref, imagem_testada, multicanal=True):
    """
    Calcula MSE, RMSE, PSNR e SSIM entre duas imagens usando apenas OpenCV e NumPy.
    
    Parâmetros:
        imagem_ref (ndarray): imagem original
        imagem_testada (ndarray): imagem gerada/testada
        multicanal (bool): True para RGB, False para grayscale
    
    Retorna:
        dict com as métricas.
    """
    if imagem_ref.shape != imagem_testada.shape:
        raise ValueError("As imagens devem ter o mesmo shape.")

    imagem_ref = imagem_ref.astype(np.float32)
    imagem_testada = imagem_testada.astype(np.float32)

    mse = np.mean((imagem_ref - imagem_testada) ** 2)
    rmse = np.sqrt(mse)
    psnr = 20 * np.log10(255.0) - 10 * np.log10(mse) if mse != 0 else float('inf')
    
    # SSIM baseado na fórmula de Wang et al., implementado manualmente
    def ssim(x, y):
        C1 = (0.01 * 255) ** 2
        C2 = (0.03 * 255) ** 2

        kernel = cv2.getGaussianKernel(11, 1.5)
        window = kernel @ kernel.T

        mu1 = cv2.filter2D(x, -1, window)
        mu2 = cv2.filter2D(y, -1, window)

        mu1_sq = mu1 * mu1
        mu2_sq = mu2 * mu2
        mu1_mu2 = mu1 * mu2

        sigma1_sq = cv2.filter2D(x * x, -1, window) - mu1_sq
        sigma2_sq = cv2.filter2D(y * y, -1, window) - mu2_sq
        sigma12 = cv2.filter2D(x * y, -1, window) - mu1_mu2

        ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / \
                   ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
        return ssim_map.mean()

    if multicanal:
        ssim_val = np.mean([ssim(imagem_ref[..., i], imagem_testada[..., i]) for i in range(3)])
    else:
        ssim_val = ssim(imagem_ref, imagem_testada)

    return {
        'MSE': mse,
        'RMSE': rmse,
        'PSNR': psnr,
        'SSIM': ssim_val,
        'correlacao': np.corrcoef(imagem_ref.flatten(), imagem_testada.flatten())[0, 1],
    }

imgname = "seagull"
img = cv2.imread(f"images/{imgname}.png", cv2.IMREAD_GRAYSCALE)
res_troca = cv2.imread(f"experiments_results/{imgname}_meio_tom_floyd_steinberg_troca.png", cv2.IMREAD_GRAYSCALE)
res_sem_troca = cv2.imread(f"experiments_results/{imgname}_meio_tom_floyd_steinberg_sem.png", cv2.IMREAD_GRAYSCALE)

print("SEM TROCA")
print(calcular_metricas(
    img,
    res_sem_troca,
    multicanal=False
))
print("COM TROCA")
print(calcular_metricas(
    img,
    res_troca,
    multicanal=False
))

print(np.sum(np.where(res_troca == res_sem_troca, 1, 0)) / (res_troca.shape[0] * res_troca.shape[1]))
