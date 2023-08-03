import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import Dataset
from torchvision import transforms
from torch.utils.data import DataLoader
import torch.optim as optim
from torchvision.utils import save_image
from pytorch_msssim import SSIM
from monai.losses import LocalNormalizedCrossCorrelationLoss as LNCC


import random
import os 
import copy
import sys
import numpy as np
from PIL import Image
from tqdm import tqdm

from GnD import Generator, Discriminator
from test_model import testing



DEVICE = torch.device("cuda:4" if torch.cuda.is_available() else "cpu")
DEVICE_IDS = [4,6,3,1]
TRAIN_DIR = "data"
VAL_DIR = "data"
BATCH_SIZE = 36
LR_GEN = 3e-4
LR_DISC = 3e-3
LR_GAMMA = 0.5
LR_STEP = 100
LAMBDA_ADV = 1.0
LAMBDA_IDENTITY = 5.0
LAMBDA_CYCLE = 5.0
LAMBDA_SSIM = 12.0
LAMBDA_LNCC = 1.0
NUM_WORKERS = 12
NUM_EPOCHS = 400
LOAD_MODEL = False
SAVE_MODEL = True
CHECKPOINT_genL2H = "genH.pth.tar"
CHECKPOINT_genH2L = "genL.pth.tar"
CHECKPOINT_discHigh = "discH.pth.tar"
CHECKPOINT_discLow = "discL.pth.tar"

transform = transforms.Compose([transforms.ToTensor()])

def save_checkpoint(model, optimizer, filename="my_checkpoint.pth.tar"):
    print("=> Saving checkpoint")
    checkpoint = {
        "state_dict": model.state_dict(),
        "optimizer": optimizer.state_dict(),
    }
    torch.save(checkpoint, filename)


def load_checkpoint(checkpoint_file, model, optimizer, lr):
    print("=> Loading checkpoint")
    checkpoint = torch.load(checkpoint_file, map_location=DEVICE)
    model.load_state_dict(checkpoint["state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer"])

    # If we don't do this then it will just have learning rate of old checkpoint
    # and it will lead to many hours of debugging \:
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr


def seed_everything(seed=42):
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

class ImageEnhancementDataset(Dataset):
    def __init__(self, low_folder, high_folder, transform):
        self.low_folder = low_folder
        self.high_folder = high_folder
        self.transform = transform
        self.low_filenames = sorted(os.listdir(self.low_folder))
        self.high_filenames = sorted(os.listdir(self.high_folder))

    def __len__(self):
        return len(self.low_filenames)

    def __getitem__(self, idx):
        low_path = os.path.join(self.low_folder, self.low_filenames[idx])
        high_path = os.path.join(self.high_folder, self.high_filenames[idx])

        low_image = Image.open(low_path).convert('L')
        high_image = Image.open(high_path).convert('L')
        low_image = self.transform(low_image)*2 - 1
        high_image = self.transform(high_image)*2 - 1
        return low_image, high_image

def train_fn(discHigh, discLow, genH2L, genL2H, loader, opt_disc, opt_gen, l1, mse, ssimLoss, lnccLoss, d_scaler, g_scaler):
    H_reals = 0
    H_fakes = 0
    DiscLoss = 0
    GenLoss = 0
    loop = tqdm(loader, leave=True)

    for idx, (low, high) in enumerate(loop):
        low = low.to(DEVICE)
        high = high.to(DEVICE)

        # Train Discriminators H and L
        with torch.cuda.amp.autocast():
            fake_high = genL2H(low)
            D_H_real = discHigh(high)
            D_H_fake = discHigh(fake_high.detach())
            H_reals += D_H_real.mean().item()
            H_fakes += D_H_fake.mean().item()
            factor = 1.0 if (D_H_real.mean() < 0.9 and D_H_fake.mean() < 0.9) else 0.9
            D_H_loss = mse(D_H_real, torch.ones_like(D_H_real)*factor) + mse(D_H_fake, torch.zeros_like(D_H_fake))

            fake_low = genH2L(high)
            D_L_real = discLow(low)
            D_L_fake = discLow(fake_low.detach())
            factor = 1.0 if (D_L_real.mean() < 0.9 and D_L_fake.mean() < 0.9) else 0.9
            D_L_loss = mse(D_L_real, torch.ones_like(D_L_real)*0.9) + mse(D_L_fake, torch.zeros_like(D_L_fake))

            # put it together
            D_loss = D_H_loss + D_L_loss

        opt_disc.zero_grad()
        d_scaler.scale(D_loss).backward()
        d_scaler.step(opt_disc)
        d_scaler.update()
        DiscLoss += D_loss.item()

        # Train Generators H and L
        with torch.cuda.amp.autocast():
            # adversarial loss for both generators
            D_H_fake = discHigh(fake_high)
            D_L_fake = discLow(fake_low)
            loss_G_H = mse(D_H_fake, torch.ones_like(D_H_fake))
            loss_G_L = mse(D_L_fake, torch.ones_like(D_L_fake))

            # cycle loss
            cycle_low = genH2L(fake_high)
            cycle_high = genL2H(fake_low)
            cycle_low_loss = l1(low, cycle_low)
            cycle_high_loss = l1(high, cycle_high)

            # identity loss (remove these for efficiency if you set lambda_identity=0)
            identity_low = genH2L(low)
            identity_high = genL2H(high)
            identity_low_loss = l1(low, identity_low)
            identity_high_loss = l1(high, identity_high)

            # SSIM loss
            ssimLow = 1 - ssimLoss((low+1)/2.0, (fake_low+1)/2.0)
            ssimHigh = 1 - ssimLoss((high+1)/2.0, (fake_high+1)/2.0)
            
            #LNCC loss
            lnccLow = lnccLoss((low+1)/2.0, (fake_low+1)/2.0)
            lnccHigh = lnccLoss((high+1)/2.0, (fake_high+1)/2.0)

            # add all together
            G_loss = ((loss_G_L + loss_G_H) * LAMBDA_ADV + 
                (cycle_low_loss + cycle_high_loss) * LAMBDA_CYCLE + 
                (identity_high_loss + identity_low_loss) * LAMBDA_IDENTITY + 
                (ssimLow + ssimHigh) * LAMBDA_SSIM + 
                (lnccLow + lnccHigh) * LAMBDA_LNCC)

        opt_gen.zero_grad()
        g_scaler.scale(G_loss).backward()
        g_scaler.step(opt_gen)
        g_scaler.update()
        GenLoss += G_loss.item()

        if idx % 100 == 0:
            # fake_high = (fake_high - fake_high.min())/ (fake_high.max() - fake_high.min())
            # fake_low = (fake_low - fake_low.min())/ (fake_low.max() - fake_low.min())
            fake_high = (fake_high + 1) / 2.0
            fake_low = (fake_low + 1) / 2.0
            save_image(fake_high, f"saved_images/high.png")
            save_image(fake_low, f"saved_images/low.png")

        loop.set_postfix(H_real=H_reals / (idx + 1), H_fake=H_fakes / (idx + 1), D_loss=DiscLoss / (idx + 1), G_loss=GenLoss / (idx + 1))

def train_model():
    discHigh= nn.DataParallel(Discriminator(),device_ids = DEVICE_IDS).to(DEVICE)
    discLow = nn.DataParallel(Discriminator(),device_ids = DEVICE_IDS).to(DEVICE)
    genH2L = nn.DataParallel(Generator(),device_ids = DEVICE_IDS).to(DEVICE)
    genL2H = nn.DataParallel(Generator(),device_ids = DEVICE_IDS).to(DEVICE)

    opt_disc = optim.Adam(list(discHigh.parameters()) + list(discLow.parameters()), lr=LR_DISC, betas=(0.9, 0.9))

    opt_gen = optim.Adam(list(genH2L.parameters()) + list(genL2H.parameters()), lr=LR_GEN, betas=(0.9, 0.9))

    # decay learning rate by a factor of LR_GAMMA every LR_STEP epochs
    lr_scheduler_disc = optim.lr_scheduler.StepLR(opt_disc, step_size=LR_STEP, gamma=LR_GAMMA)

    lr_scheduler_gen = optim.lr_scheduler.StepLR(opt_gen, step_size=LR_STEP, gamma=LR_GAMMA)

    L1 = nn.L1Loss()
    mse = nn.MSELoss()
    ssimLoss = SSIM(data_range=1.0, size_average=True, channel=1, nonnegative_ssim=True)
    lnccLoss = LNCC(spatial_dims = 2, kernel_size = 3, kernel_type = "rectangular", reduction = "mean", smooth_nr = 0.0, smooth_dr = 1.0)

    if LOAD_MODEL:
        load_checkpoint(CHECKPOINT_genL2H,genL2H,opt_gen,LR_GEN,LR_DISC)
        load_checkpoint(CHECKPOINT_genH2L,genH2L,opt_gen,LR_GEN,LR_DISC)
        load_checkpoint(CHECKPOINT_discHigh,discHigh,opt_disc,LR_GEN,LR_DISC)
        load_checkpoint(CHECKPOINT_discLow,discLow,opt_disc,LR_GEN,LR_DISC)

    dataset = ImageEnhancementDataset(low_folder=TRAIN_DIR + "/Low", high_folder=TRAIN_DIR + "/High", transform=transform)

    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS, pin_memory=True)

    val_dataset = ImageEnhancementDataset(low_folder=VAL_DIR + "/Low", high_folder=VAL_DIR + "/High", transform=transform)

    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, pin_memory=True)
    
    g_scaler = torch.cuda.amp.GradScaler()
    d_scaler = torch.cuda.amp.GradScaler()

    for epoch in range(NUM_EPOCHS):
        print(f"Epoch [{epoch}/{NUM_EPOCHS}]")
        train_fn(discHigh, discLow, genH2L, genL2H, loader, opt_disc, opt_gen, L1, mse, ssimLoss, lnccLoss, d_scaler, g_scaler)
        lr_scheduler_disc.step()
        lr_scheduler_gen.step()

        if SAVE_MODEL:
            save_checkpoint(genL2H, opt_gen, filename=CHECKPOINT_genL2H)
            save_checkpoint(genH2L, opt_gen, filename=CHECKPOINT_genH2L)
            save_checkpoint(discHigh, opt_disc, filename=CHECKPOINT_discHigh)
            save_checkpoint(discLow, opt_disc, filename=CHECKPOINT_discLow)

    print("Done training!")

if __name__ == "__main__":
    train_model()
    testing(DEVICE)
