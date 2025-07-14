import os
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader, random_split
from torch.utils.tensorboard import SummaryWriter
from torchvision.utils import make_grid
from tqdm import tqdm  
from skimage.metrics import peak_signal_noise_ratio, structural_similarity


from unet.dataset import FMRI3DDataset
from unet.model import UNet3DfMRI
from unet.evaluate import evaluate_model


# -----------------------------
# Training
# -----------------------------

def train_model(model: nn.Module, dataloader: DataLoader, optimizer: torch.optim.Optimizer, device, epochs: int = 6, writer: SummaryWriter = None, resume_from: int = 0):
    model.train()
    model.to(device)

    loss_fn = nn.MSELoss()

    global_step = 0

    for epoch in range(resume_from, epochs):
        total_loss = 0
        progress_bar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{epochs}", leave=False)

        for batch_idx, (x, y) in enumerate(progress_bar):
            x, y = x.to(device), y.to(device)
            pred = model(x)

            # Delete extra padded dimension before computing loss
            if pred.shape[4] > y.shape[4]:
                pred = pred[..., :y.shape[4]]

            # Create masks
            threshold = 0.3  
            bright_mask = (y > threshold).float()
            dark_mask = 1.0 - bright_mask 

            per_voxel_loss = loss_fn(pred, y)

            # Apply masks
            bright_loss = (per_voxel_loss * bright_mask).sum() / bright_mask.sum().clamp(min=1.0)
            dark_loss = (per_voxel_loss * dark_mask).sum() / dark_mask.sum().clamp(min=1.0)

            # weighted total loss
            loss = 0.8 * bright_loss + 0.2 * dark_loss
            loss *= 1e2

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            progress_bar.set_postfix(loss=loss.item())

            # logging every 10 iterations
            if global_step % 10 == 0:
                writer.add_scalar("Loss/train", loss.item(), global_step)
                pred_np = pred.detach().cpu().numpy()
                y_np = y.detach().cpu().numpy()

                psnr_total = 0.0
                ssim_total = 0.0
                
                # Calculate over entire batch
                for i in range(pred_np.shape[0]):  
                    pred_vol = np.squeeze(pred_np[i])
                    y_vol = np.squeeze(y_np[i])

                    psnr = peak_signal_noise_ratio(y_vol, pred_vol, data_range=1.0)
                    ssim = structural_similarity(y_vol, pred_vol, data_range=1.0)

                    psnr_total += psnr
                    ssim_total += ssim

                avg_psnr = psnr_total / pred_np.shape[0]
                avg_ssim = ssim_total / pred_np.shape[0]

                writer.add_scalar("Train/Loss", loss.item(), global_step)
                writer.add_scalar("Train/PSNR", avg_psnr, global_step)
                writer.add_scalar("Train/SSIM", avg_ssim, global_step)

            # log every 1/20 epoch
            if batch_idx % (len(dataloader) // 20 + 1) == 0:
                with torch.no_grad():
                    x_slice = x[0, 0, :, :, x.shape[4] // 2].cpu().unsqueeze(0)
                    y_slice = y[0, 0, :, :, y.shape[4] // 2].cpu().unsqueeze(0)
                    pred_slice = pred[0, 0, :, :, pred.shape[4] // 2].cpu().unsqueeze(0)

                    grid = make_grid(torch.stack([x_slice, y_slice, pred_slice]), nrow=3, normalize=False)
                    writer.add_image(f"Epoch_{epoch+1}/Input_GT_Pred", grid, global_step)

            global_step += 1

        avg_loss = total_loss / len(dataloader)
        print(f"Epoch {epoch + 1}/{epochs}, Avg Loss: {avg_loss:.4f}")

        torch.save(model.state_dict(), f"./runs/checkpoints/test/model_weights_epoch{epoch + 1}.pth")


if __name__ == "__main__":
    root_dir = "./data"
    checkpoint_path = ''
    resume= False
    batch_size = 4
    num_epochs = 6
    learning_rate = 1e-4
    
    writer = SummaryWriter()
    

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    full_dataset = FMRI3DDataset(root_dir)

    total_size = len(full_dataset)
    train_size = int(0.8 * total_size)
    test_size = total_size - train_size

    train_dataset, test_dataset = random_split(full_dataset, [train_size, test_size])

    train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=4, shuffle=False)

    model = UNet3DfMRI()
    print("Starting training...")
    if resume:
        model.load_state_dict(torch.load(checkpoint_path))
    

    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
    train_model(model, train_loader, device, epochs=num_epochs, optimizer=optimizer, writer=writer, resume_from=0)

    print("Evaluating and visualizing...")
    evaluate_model(model, test_loader, device)
    writer.close()


