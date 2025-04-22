import torch
import numpy as np
from torch.utils.data import DataLoader
from skimage.metrics import peak_signal_noise_ratio
from skimage.metrics import structural_similarity
import numpy as np
from tqdm import tqdm  


from unet.dataset import FMRI3DDataset
from unet.model import UNet3DfMRI

def evaluate_model(model: torch.nn.Module, test_loader: DataLoader, device: torch.device):
    model.eval()
    model.to(device)

    psnr_total = 0.0
    ssim_total = 0.0
    count = 0

    with torch.no_grad():
        progress_bar = tqdm(test_loader, desc="Evaluating", leave=False)
        for i, (x, y) in enumerate(progress_bar):
            x = x.to(device)
            y = y.to(device)

            pred = model(x)

            # If padded during training, remove extra depth slice
            if pred.shape[4] > y.shape[4]:
                pred = pred[..., :y.shape[4]]

            # Convert to numpy for metric computation
            pred_np = pred.cpu().numpy()
            y_np = y.cpu().numpy()

            # Batch-wise metric calculation
            for j in range(pred_np.shape[0]):
                pred_vol = np.squeeze(pred_np[j])
                y_vol = np.squeeze(y_np[j])

                psnr = peak_signal_noise_ratio(y_vol, pred_vol, data_range=1.0)
                ssim = structural_similarity(y_vol, pred_vol, data_range=1.0)

                psnr_total += psnr
                ssim_total += ssim
                count += 1

            # Optional: update tqdm postfix with running averages
            avg_psnr = psnr_total / count
            avg_ssim = ssim_total / count
            progress_bar.set_postfix(PSNR=f"{avg_psnr:.2f}", SSIM=f"{avg_ssim:.3f}")

            if i > 100:
                break

    avg_psnr = psnr_total / count
    avg_ssim = ssim_total / count

    print(f"Evaluation Results — PSNR: {avg_psnr:.4f}, SSIM: {avg_ssim:.4f}")
    return avg_psnr, avg_ssim



if __name__ == "__main__":

    root_dir = "./validation"

    device = torch.device("cpu")

    dataset = FMRI3DDataset(root_dir)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True)

    model = UNet3DfMRI()
    model.load_state_dict(torch.load("model_weights_.pth"))
    model.to(device)
    model.eval()  

    print("Evaluating and visualizing...")
    evaluate_model(model, dataloader, device)