
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from torch.utils.tensorboard import SummaryWriter
from torchvision.utils import make_grid
from tqdm import tqdm  

from unet.dataset import FMRI3DDataset
from unet.model import UNet3DfMRI
from unet.evaluate import evaluate_model


# -----------------------------
# Training & Evaluation
# -----------------------------

def train_model(model: nn.Module, dataloader: DataLoader, device, epochs: int = 5, lr: float = 1e-4, writer: SummaryWriter = None):
    model.train()
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()

    global_step = 0

    for epoch in range(epochs):
        total_loss = 0
        progress_bar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{epochs}", leave=False)

        for batch_idx, (x, y) in enumerate(progress_bar):
            x, y = x.to(device), y.to(device)
            pred = model(x)

            y_cropped = UNet3DfMRI.crop_to_match(y, pred)

            loss = loss_fn(pred, y_cropped) * 1e2
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

            # 🔁 Scalar logging every 10 iterations
            if writer and (global_step % 10 == 0):
                writer.add_scalar("Loss/train", loss.item(), global_step)

            # 🔁 Visual log every 1/10 epoch
            if writer and (batch_idx % (len(dataloader) // 10 + 1) == 0):
                with torch.no_grad():
                    x_slice = x[0, 0, :, :, x.shape[4] // 2].cpu().unsqueeze(0)
                    y_slice = y_cropped[0, 0, :, :, y.shape[4] // 2].cpu().unsqueeze(0)
                    pred_slice = pred[0, 0, :, :, pred.shape[4] // 2].cpu().unsqueeze(0)

                    grid = make_grid(torch.stack([x_slice, y_slice, pred_slice]), nrow=3, normalize=True)
                    writer.add_image(f"Epoch_{epoch+1}/Input_GT_Pred", grid, global_step)

            global_step += 1

        avg_loss = total_loss / len(dataloader)
        print(f"Epoch {epoch + 1}/{epochs}, Avg Loss: {avg_loss:.4f}")

        if writer:
            writer.add_scalar("Loss/epoch_avg", avg_loss, epoch + 1)

        torch.save(model.state_dict(), f"model_weights_epoch{epoch + 1}.pth")



if __name__ == "__main__":
    root_dir = "./data"
    batch_size = 2
    num_epochs = 4
    learning_rate = 1e-4
    writer = SummaryWriter()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    full_dataset = FMRI3DDataset(root_dir)

    # Example: assuming `full_dataset` is your full fMRI dataset
    total_size = len(full_dataset)
    train_size = int(0.8 * total_size)
    test_size = total_size - train_size

    train_dataset, test_dataset = random_split(full_dataset, [train_size, test_size])

    train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=4, shuffle=False)

    model = UNet3DfMRI()
    print("Starting training...")
    train_model(model, train_loader, device, epochs=num_epochs, lr=learning_rate, writer=writer)

    print("Evaluating and visualizing...")
    evaluate_model(model, test_loader, device)
    writer.close()


