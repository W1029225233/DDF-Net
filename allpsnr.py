# compare_feat2feat_lowpass.py
import torch
import numpy as np

device = 'cuda' if torch.cuda.is_available() else 'cpu'

# ========= 用户输入（四条特征图路径） =========
residual_blur_path = r'/home/wjh/Deep Learning/去模糊/ECBSR/checkpoints/experiment_name/features1/residual2_780.pt'   # 空域模糊
residual_clear_path= r'/home/wjh/Deep Learning/去模糊/ECBSR/checkpoints/experiment_name/features/residual2_780.pt'     # 空域清晰
fft_blur_path      = r'/home/wjh/Deep Learning/去模糊/ECBSR/checkpoints/experiment_name/features1/fft2_780.pt'       # 频域模糊
fft_clear_path     = r'/home/wjh/Deep Learning/去模糊/ECBSR/checkpoints/experiment_name/features/fft2_780.pt'          # 频域清晰


def psnr(mse):
    return 20 * np.log10(1.0 / np.sqrt(mse)) if mse != 0 else float('inf')

# ========= 加载特征图（同尺寸） =========
res_blur = torch.load(residual_blur_path).to(device)
res_clear= torch.load(residual_clear_path).to(device)
fft_blur = torch.load(fft_blur_path).to(device)
fft_clear= torch.load(fft_clear_path).to(device)

# ========= 共享 1×1 解码器（C→C） =========
C = res_blur.shape[1]
decoder = torch.nn.Conv2d(C, C, 1).to(device)
dec_res, dec_fft = [torch.nn.Conv2d(C, C, 1).to(device) for _ in range(2)]
dec_res.load_state_dict(decoder.state_dict())
dec_fft.load_state_dict(decoder.state_dict())

# ========= 直接重建特征图本身 =========
with torch.no_grad():
    pred_res_blur = dec_res(res_blur)
    pred_fft_blur = dec_fft(fft_blur)
    # 以“清晰特征图”为真值
    mse_res = torch.mean((pred_res_blur - res_clear) ** 2).item()
    mse_fft = torch.mean((pred_fft_blur - fft_clear) ** 2).item()

# ========= 结果 =========
print('=== 特征图直接重建 PSNR（清晰特征 = 真值）===')
print(f'空域模糊特征 → 空域清晰特征  PSNR: {psnr(mse_res):.2f} dB')
print(f'频域模糊特征 → 频域清晰特征  PSNR: {psnr(mse_fft):.2f} dB')
winner = '频域' if mse_fft < mse_res else '空域'
print(f'→ {winner} 分支特征更接近自身清晰特征，全局-局部保真度更高。')