import torch
from torch import nn
import torch.nn.functional as F
from torch.nn.modules.loss import _Loss
from torch.utils.data import Dataset
from torchvision import transforms
from torch.utils.data import DataLoader, random_split
import torch.optim as optim
from torchvision.utils import save_image
from pytorch_msssim import ssim, SSIM
from monai.losses import LocalNormalizedCrossCorrelationLoss as LNCC
import lpips
import torchvision.transforms.functional as TF
import random
from typing import Sequence

import random
import os 
import copy
import sys
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from tqdm import tqdm

from GnD import Generator, Discriminator
from test_model2 import testing

DEVICE = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
DEVICE_IDS = [1,3,4]
TRAIN_DIR = "data"
VAL_DIR = "data"
BATCH_SIZE = 24
LR_GEN = 3e-4
LR_DISC = 3e-3
LR_GAMMA = 0.5
LR_STEP = 100
LAMBDA_ADV = 1.0
LAMBDA_IDENTITY = 1.0
LAMBDA_CYCLE = 12.0
LAMBDA_SSIM = 12.0
LAMBDA_PER = 1.0
# LAMBDA_L1 = 5.0
# LAMBDA_PSNR = 1.0
# LAMBDA_LNCC = 1.0
NUM_WORKERS = 12
NUM_EPOCHS = 300
LOAD_MODEL = False
SAVE_MODEL = True
SEED = 127
CHECKPOINT_genL2H = "genH.pth.tar"
CHECKPOINT_genH2L = "genL.pth.tar"
CHECKPOINT_discHigh = "discH.pth.tar"
CHECKPOINT_discLow = "discL.pth.tar"
BEST_gen = "bestGen.pth.tar"
transform = transforms.Compose([transforms.ToTensor()])

class MyRotateTransform:
    def __init__(self, angles: Sequence[int]):
        self.angles = angles

    def __call__(self, x, y):
        angle = random.choice(self.angles)
        return TF.rotate(x, angle), TF.rotate(y, angle)
    
class CharbonnierLoss(nn.Module):
    """Charbonnier Loss (L1)"""
    def __init__(self, eps=1e-6):
        super(CharbonnierLoss, self).__init__()
        self.eps = eps

    def forward(self, x, y):
        b, c, h, w = y.size()
        loss = torch.sum(torch.sqrt((x - y).pow(2) + self.eps**2))
        return loss/(c*b*h*w)

def save_checkpoint(model, optimizer, filename="my_checkpoint.pth.tar", verbose=False):
    checkpoint = {"state_dict": model.state_dict(),"optimizer": optimizer.state_dict()}
    torch.save(checkpoint, filename)
    if verbose:
        print("=> Saving checkpoint")


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

def compute_LNCC(img1, img2, kernel_size=9):
    """
    Compute the Local Normalized Cross-Correlation between two images.
    Args:
    - img1: the first image, of shape [B, C, H, W] (for 2D) or [B, C, D, H, W] (for 3D)
    - img2: the second image, must have the same shape as img1
    - kernel_size: the size of the window to compute local NCC, should be odd.
    Returns:
    - ncc: the Local Normalized Cross-Correlation map
    """

    if kernel_size % 2 == 0:
        raise ValueError("Kernel size should be odd.")
    
    padding = kernel_size // 2
    img1 = F.pad(img1, [padding]*4, 'reflect')  # assuming 2D images; for 3D you'd pad with [padding]*6
    img2 = F.pad(img2, [padding]*4, 'reflect')

    # Compute local sums for img1, img2, img1^2, img2^2, and img1*img2
    sum_img1 = F.avg_pool2d(img1, kernel_size, stride=1)
    sum_img2 = F.avg_pool2d(img2, kernel_size, stride=1)
    sum_img1_sq = F.avg_pool2d(img1 * img1, kernel_size, stride=1)
    sum_img2_sq = F.avg_pool2d(img2 * img2, kernel_size, stride=1)
    sum_img1_img2 = F.avg_pool2d(img1 * img2, kernel_size, stride=1)
    
    # Compute mean and variance in local window for img1 and img2
    mean_img1 = sum_img1 / (kernel_size ** 2)
    mean_img2 = sum_img2 / (kernel_size ** 2)
    var_img1 = sum_img1_sq - mean_img1 ** 2
    var_img2 = sum_img2_sq - mean_img2 ** 2

    # Compute the local normalized cross-correlation
    ncc = (sum_img1_img2 - mean_img1 * mean_img2) / (torch.sqrt(var_img1 * var_img2) + 1e-5)
    
    return torch.mean(ncc)


class LNCCLoss(_Loss):
    def __init__(self, kernel_size=9):
        super(LNCCLoss, self).__init__()
        self.kernel_size = kernel_size
    
    def forward(self, input1, input2):
        assert input1.shape == input2.shape
        assert self.kernel_size%2 != 0

        padding = self.kernel_size // 2
        input1 = F.pad(input1, [padding]*4, 'reflect')  # assuming 2D images; for 3D you'd pad with [padding]*6
        input2 = F.pad(input2, [padding]*4, 'reflect')

        # Compute local sums for img1, img2, img1^2, img2^2, and img1*img2
        sum_img1 = F.avg_pool2d(input1, self.kernel_size, stride=1)
        sum_img2 = F.avg_pool2d(input2, self.kernel_size, stride=1)
        sum_img1_sq = F.avg_pool2d(input1 * input1, self.kernel_size, stride=1)
        sum_img2_sq = F.avg_pool2d(input2 * input2, self.kernel_size, stride=1)
        sum_img1_img2 = F.avg_pool2d(input1 * input2, self.kernel_size, stride=1)
        
        # Compute mean and variance in local window for img1 and img2
        mean_img1 = sum_img1 / (self.kernel_size ** 2)
        mean_img2 = sum_img2 / (self.kernel_size ** 2)
        var_img1 = sum_img1_sq - mean_img1 ** 2
        var_img2 = sum_img2_sq - mean_img2 ** 2

        # Compute the local normalized cross-correlation
        ncc = (sum_img1_img2 - mean_img1 * mean_img2) / (torch.sqrt(var_img1 * var_img2) + 1e-5)

        return (1 - torch.mean(ncc))


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
        # low_image, high_image = MyRotateTransform([0, 90, 180, 270])(low_image, high_image)
        # if random.random() > 0.5:
        #     low_image = TF.hflip(low_image)
        #     high_image = TF.hflip(high_image)
        return low_image, high_image

def train_fn(discHigh, discLow, genH2L, genL2H, loader, opt_disc, opt_gen, l1, mse, ssimLoss, perLoss, d_scaler, g_scaler):
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
            fake_high = genL2H(low)
            fake_low = genH2L(high)
            D_H_fake = discHigh(fake_high)
            D_L_fake = discLow(fake_low)
            loss_G_H = mse(D_H_fake, torch.ones_like(D_H_fake))
            loss_G_L = mse(D_L_fake, torch.ones_like(D_L_fake))

            # cycle loss
            cycle_low = genH2L(fake_high)
            cycle_high = genL2H(fake_low)
            cycleLow = l1(low, cycle_low)
            cycleHigh = l1(high, cycle_high)

            # identity loss (remove these for efficiency if you set lambda_identity=0)
            identity_low = genH2L(low)
            identity_high = genL2H(high)
            identityLow = l1(low, identity_low)
            identityHigh = l1(high, identity_high)

            # SSIM loss
            # ssimLow = 1 - ssimLoss((low+1)/2.0, (fake_low+1)/2.0)
            # ssimHigh = 1 - ssimLoss((high+1)/2.0, (fake_high+1)/2.0)

            # Perceptual loss
            perLow = torch.mean(perLoss.forward(low,fake_low))
            perHigh = torch.mean(perLoss.forward(high,fake_high))
            # L1 loss
            # l1Low = l1(low, fake_low)
            # l1High = l1(high, fake_high)
            
            # LNCC loss
            # lnccHigh = lnccLoss((fake_high+1)/2.0, (high+1)/2.0)
            # print(lnccHigh)

            # psnr loss
            # psnrHigh = 100 - (10 * torch.log10(1/mse((fake_high+1)/2,(high+1)/2)))
            # psnrLow = 100 - (10 * torch.log10(1/mse((fake_low+1)/2,(low+1)/2))) 

            # add all together
            G_loss = ((loss_G_L + loss_G_H) * LAMBDA_ADV + 
                      (cycleLow + cycleHigh) * LAMBDA_CYCLE + 
                      (identityLow + identityHigh) * LAMBDA_IDENTITY + 
                      (perLow + perHigh) * LAMBDA_PER)
                    # (ssimLow + ssimHigh) * LAMBDA_SSIM + 
                    # (l1Low + l1High) * LAMBDA_L1 + 
                    # (psnrLow + psnrHigh) * LAMBDA_PSNR)

        opt_gen.zero_grad()
        g_scaler.scale(G_loss).backward()
        g_scaler.step(opt_gen)
        g_scaler.update()
        GenLoss += G_loss.item()

        if idx % 100 == 0:
            fake_high = (fake_high + 1) / 2.0
            diffHigh = torch.abs(fake_high - ((high + 1)/2.0))
            diffLow = torch.abs(fake_high - ((low + 1)/2.0))
            save_image(fake_high, f"saved_images/trainHigh.png")
            save_image(diffHigh, f"saved_images/trDiffHigh.png")
            save_image(diffLow, f"saved_images/trDiffLow.png")

        loop.set_postfix(H_real=H_reals / (idx + 1), H_fake=H_fakes / (idx + 1), D_loss=DiscLoss / (idx + 1), G_loss=GenLoss / (idx + 1))


def val_fn(Gen, val_loader, mse):
    Gen.eval()  # set the generator in evaluation mode
    psnrScore = 0.0
    lnccScore = 0.0
    ssimScore = 0.0
    highL1 = 0.0
    lowL1 = 0.0

    with torch.no_grad():  # disable gradients for validation
        for idx, (low, high) in enumerate(val_loader):
            low = low.to(DEVICE)
            high = high.to(DEVICE)
            fake_high = Gen(low)

            ssimScore += ssim((fake_high+1)/2.0, (high+1)/2.0, data_range=1.0, size_average=True, nonnegative_ssim=True).item()

            lnccScore += compute_LNCC((fake_high+1)/2, (high+1)/2, kernel_size=9).item()
            
            mseVal = mse((fake_high+1)/2,(high+1)/2)
            psnrScore += (10 * torch.log10(1/mseVal)).item() if (mseVal > 0) else 100

            highL1 += torch.mean(torch.abs(high-fake_high))

            lowL1 += torch.mean(torch.abs(low-fake_high))

            if idx % 100 == 0:
                fake_high = (fake_high + 1) / 2.0
                diffHigh = torch.abs(fake_high - ((high + 1)/2.0))
                diffLow = torch.abs(fake_high - ((low + 1)/2.0))
                save_image(fake_high, f"saved_images/high.png")
                save_image(diffHigh, f"saved_images/diffHigh.png")
                save_image(diffLow, f"saved_images/diffLow.png")

    Gen.train()  # set the generator back to training mode
    print(f"SSIM: {ssimScore/len(val_loader):.6f}, LNCC: {lnccScore/len(val_loader):.6f}, PSNR: {psnrScore/len(val_loader):.6f}, HighL1: {highL1/len(val_loader):.6f}, LowL1: {lowL1/len(val_loader):.6f}")

    return (ssimScore/len(val_loader), lnccScore/len(val_loader), psnrScore/len(val_loader), highL1/len(val_loader), lowL1/len(val_loader))


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
    perLoss = lpips.LPIPS(net='vgg')
    perLoss.cuda(1)
    # lnccLoss = LNCC(spatial_dims = 2, kernel_size = 5, kernel_type = "rectangular", reduction = "mean", smooth_nr = 0.01, smooth_dr = 0.01)
    # lnccLoss = LNCCLoss(kernel_size=9)

    if LOAD_MODEL:
        load_checkpoint(CHECKPOINT_genL2H,genL2H,opt_gen,LR_GEN,LR_DISC)
        load_checkpoint(CHECKPOINT_genH2L,genH2L,opt_gen,LR_GEN,LR_DISC)
        load_checkpoint(CHECKPOINT_discHigh,discHigh,opt_disc,LR_GEN,LR_DISC)
        load_checkpoint(CHECKPOINT_discLow,discLow,opt_disc,LR_GEN,LR_DISC)

    dataset = ImageEnhancementDataset(low_folder=TRAIN_DIR + "/Low", high_folder=TRAIN_DIR + "/High", transform=transform)

    train_size = int(0.9 * len(dataset))
    val_size = len(dataset) - train_size

    rand_gen = torch.Generator().manual_seed(SEED)
    train_dataset, val_dataset = random_split(dataset, [train_size,val_size], rand_gen)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS, pin_memory=True)

    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, pin_memory=True)
    
    g_scaler = torch.cuda.amp.GradScaler()
    d_scaler = torch.cuda.amp.GradScaler()

    ssimVals = []
    lnccVals = []
    psnrVals = []
    highL1Vals = []
    lowL1Vals = []
    bestSSIM, bestPSNR, bestLNCC = 0.0, 0.0, 0.0

    for epoch in range(NUM_EPOCHS):
        print(f"Epoch [{epoch}/{NUM_EPOCHS}]")
        train_fn(discHigh, discLow, genH2L, genL2H, train_loader, opt_disc, opt_gen, L1, mse, ssimLoss, perLoss, d_scaler, g_scaler)
        (ssimScore, lnccScore, psnrScore, highL1, lowL1) = val_fn(genL2H, val_loader, mse)
        
        lr_scheduler_disc.step()
        lr_scheduler_gen.step()

        ssimVals.append(ssimScore)
        lnccVals.append(lnccScore)
        psnrVals.append(psnrScore)
        highL1Vals.append(highL1.cpu().numpy())
        lowL1Vals.append(lowL1.cpu().numpy())

        if SAVE_MODEL:
            save_checkpoint(genL2H, opt_gen, filename=CHECKPOINT_genL2H)
            save_checkpoint(genH2L, opt_gen, filename=CHECKPOINT_genH2L)
            save_checkpoint(discHigh, opt_disc, filename=CHECKPOINT_discHigh)
            save_checkpoint(discLow, opt_disc, filename=CHECKPOINT_discLow)
        
        current_scores = {'SSIM': ssimScore,'PSNR': psnrScore,'LNCC': lnccScore}
        best_scores = {'SSIM': bestSSIM,'PSNR': bestPSNR,'LNCC': bestLNCC}
        impr = {k:current_scores[k]>best_scores[k] for k in best_scores}

        if (sum(impr.values()) >= 2):
            print(f"Best epoch:{epoch}", end=" ")
            save_checkpoint(genL2H, opt_gen, filename=BEST_gen, verbose = True)
            bestSSIM, bestPSNR, bestLNCC, bestEPOCH = ssimScore, psnrScore, lnccScore, epoch

    epochs = np.arange(0,NUM_EPOCHS)
    plt.figure()
    plt.plot(epochs,ssimVals)
    plt.title("SSIM plot")
    plt.savefig("saved_images/ssim.png", dpi=300)
    plt.figure()
    plt.plot(epochs,lnccVals)
    plt.title("LNCC plot")
    plt.savefig("saved_images/lncc.png", dpi=300)
    plt.figure()
    plt.plot(epochs,psnrVals)
    plt.title("PSNR plot")
    plt.savefig("saved_images/psnr.png", dpi=300)
    plt.figure()
    plt.plot(epochs,highL1Vals)
    plt.title("High L1 plot")
    plt.savefig("saved_images/highL1.png", dpi=300)
    plt.figure()
    plt.plot(epochs,lowL1Vals)
    plt.title("Low L1 plot")
    plt.savefig("saved_images/lowL1.png", dpi=300)
    
    print("Done training!")
    print(f"Best SSIM: {bestSSIM:.6f}, Best LNCC: {bestLNCC:.6f}, Best PSNR: {bestPSNR:.6f}, Best Epoch: {bestEPOCH}")

if __name__ == "__main__":
    seed_everything(SEED)
    train_model()
    testing(DEVICE)
