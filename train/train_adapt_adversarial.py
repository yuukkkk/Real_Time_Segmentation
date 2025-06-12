import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import os
import time
from tqdm import tqdm 
import numpy as np 
import argparse 
from collections import OrderedDict 

# Import project components
from datasets.gta5 import GTA5
from datasets.cityscapes import CityScapes 
from models.bisenet.build_bisenet import BiSeNet # Generator
from models.discriminator import Discriminator      # Discriminator
from utils.lr_scheduler import lr_poly 

# --- Default Configurations for ADAM experiment ---
DEFAULT_GTA5_ROOT = "/kaggle/input/datasetcityscapes/GTA5/GTA5/" 
DEFAULT_CITYSCAPES_ROOT = "/kaggle/input/datasetcityscapes/Cityscapes/Cityscapes/Cityspaces/" 
DEFAULT_CHECKPOINT_DIR = "/kaggle/working/checkpoints_bisenet_adversarial_ext" # New dir for extension experiments
DEFAULT_RUN_NAME = "adv_adam_generator" 
DEFAULT_NUM_CLASSES = 19
DEFAULT_IGNORE_INDEX = 255
DEFAULT_GTA5_TRAIN_SIZE = (720, 1280) 
DEFAULT_CITYSCAPES_TRAIN_SIZE = (512, 1024) 
DEFAULT_NUM_EPOCHS = 50 
DEFAULT_BATCH_SIZE = 4 

# --- IMPORTANT: Changed Learning Rates for Adam ---
DEFAULT_LR_G = 1e-4  # Adam requires a much smaller learning rate than SGD
DEFAULT_LR_D = 1e-4  # Also using Adam for the Discriminator for consistency here
DEFAULT_LAMBDA_ADV = 0.001 
# Adam does not use momentum. Betas are the standard defaults.
DEFAULT_WEIGHT_DECAY = 0 # Adam's weight decay (AdamW) is often handled differently, starting with 0 is safe.
AUX_LOSS_WEIGHT = 1.0

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- Argument Parser ---
def parse_args():
    parser = argparse.ArgumentParser(description="Adversarial Adaptation with Adam Optimizer")
    parser.add_argument('--lr_g', type=float, default=DEFAULT_LR_G, help='Learning rate for Generator (Adam)')
    # ... (Other arguments like resume, paths, epochs, batch_size, aug flags remain the same) ...
    parser.add_argument('--resume', type=str, default=None, help='Path to checkpoint to resume training')
    parser.add_argument('--batch_size', type=int, default=DEFAULT_BATCH_SIZE, help='Batch size for both loaders')
    parser.add_argument('--lr_d', type=float, default=DEFAULT_LR_D, help='Learning rate for Discriminator (Adam)')
    parser.add_argument('--lambda_adv', type=float, default=DEFAULT_LAMBDA_ADV, help='Weight for adversarial loss')
    parser.add_argument('--epochs', type=int, default=DEFAULT_NUM_EPOCHS, help='Total epochs')
    parser.add_argument('--gta5_root', type=str, default=DEFAULT_GTA5_ROOT, help='Path to GTA5 root')
    parser.add_argument('--cityscapes_root', type=str, default=DEFAULT_CITYSCAPES_ROOT, help='Path to Cityscapes root')
    parser.add_argument('--run_name', type=str, default=DEFAULT_RUN_NAME, help='Run name for checkpoints')
    parser.add_argument('--checkpoint_dir', type=str, default=DEFAULT_CHECKPOINT_DIR, help='Directory to save checkpoints')
    parser.add_argument('--aug_hflip', action='store_true', help='Enable Random Horizontal Flip')
    parser.add_argument('--aug_colorjitter', action='store_true', help='Enable Color Jitter')
    parser.add_argument('--aug_gblur', action='store_true', help='Enable Gaussian Blur')

    args = parser.parse_args()
    return args

# --- Helper Function: save_checkpoint (same as before) ---
def save_checkpoint(generator, discriminator, optimizer_G, optimizer_D, epoch, run_ckpt_dir, filename="checkpoint.pth.tar"):
    # (This function can remain the same as in Turn 95)
    state = {
        'epoch': epoch + 1, 
        'generator_state_dict': generator.module.state_dict() if isinstance(generator, nn.DataParallel) else generator.state_dict(),
        'discriminator_state_dict': discriminator.module.state_dict() if isinstance(discriminator, nn.DataParallel) else discriminator.state_dict(),
        'optimizer_G_state_dict': optimizer_G.state_dict(),
        'optimizer_D_state_dict': optimizer_D.state_dict(),
    }
    filepath = os.path.join(run_ckpt_dir, filename)
    os.makedirs(run_ckpt_dir, exist_ok=True)
    torch.save(state, filepath)


# --- Main Training Function ---
def main(args): 
    # ... (Initial setup: print device, config, run_checkpoint_dir - same as before) ...
    print(f"Using device: {DEVICE}"); print(f"Configuration: {args}")
    run_checkpoint_dir = os.path.join(args.checkpoint_dir, args.run_name); os.makedirs(run_checkpoint_dir, exist_ok=True)
    print(f"Checkpoints will be saved in: {run_checkpoint_dir}")
    start_epoch = 0

    # --- Models ---
    generator = BiSeNet(num_classes=DEFAULT_NUM_CLASSES, context_path='resnet18')
    discriminator = Discriminator(num_classes=DEFAULT_NUM_CLASSES)
    if torch.cuda.device_count() > 1:
        generator = nn.DataParallel(generator); discriminator = nn.DataParallel(discriminator)
    generator.to(DEVICE); discriminator.to(DEVICE)

    # --- DataLoaders (same as before) ---
    # (Code for creating source_loader and target_loader remains the same as in Turn 95)
    gta_train_dataset = GTA5(root_dir=args.gta5_root, target_size=DEFAULT_GTA5_TRAIN_SIZE,
                             use_horizontal_flip=args.aug_hflip, use_color_jitter=args.aug_colorjitter,
                             use_gaussian_blur=args.aug_gblur)
    source_loader = DataLoader(gta_train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=True, drop_last=True)
    cityscapes_target_dataset = CityScapes(root_dir=args.cityscapes_root, split='train', target_size=DEFAULT_CITYSCAPES_TRAIN_SIZE)
    target_loader = DataLoader(cityscapes_target_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=True, drop_last=True)

    # --- Optimizers ---
    # --- THIS IS THE KEY CHANGE ---
    print("Using SGD optimizer for both Generator and Discriminator.")
    optimizer_G = optim.SGD(generator.parameters(), lr=args.lr_g, momentum=0.9, weight_decay=5e-4)
    optimizer_D = optim.SGD(discriminator.parameters(), lr=args.lr_d, momentum=0.9, weight_decay=5e-4)
    # --- END OF KEY CHANGE ---

    # --- Losses (same as before) ---
    criterion_seg = nn.CrossEntropyLoss(ignore_index=DEFAULT_IGNORE_INDEX).to(DEVICE)
    criterion_adv = nn.BCEWithLogitsLoss().to(DEVICE)
    source_label_val = 0
    target_label_val = 1
    
    # --- Training Loop ---
    # (The main training loop logic is the same as the corrected version in Turn 95)
    print(f"Starting Adversarial Training for {args.epochs} epochs...")
    max_iterations = args.epochs * len(source_loader)
    current_iteration = start_epoch * len(source_loader)
    
    for epoch in range(start_epoch, args.epochs):
        generator.train(); discriminator.train()
        target_iter = iter(target_loader)
        progress_bar = tqdm(enumerate(source_loader), total=len(source_loader), desc=f"Epoch {epoch+1}/{args.epochs}")
        
        for i, (source_images, source_seg_labels) in progress_bar:
            # D-Step
            optimizer_D.zero_grad()
            try: target_images, _ = next(target_iter)
            except StopIteration: target_iter = iter(target_loader); target_images, _ = next(target_iter)
            target_images = target_images.to(DEVICE)
            with torch.no_grad(): target_preds, _, _ = generator(target_images)
            D_out_target = discriminator(target_preds.detach())
            loss_D_target = criterion_adv(D_out_target, torch.full_like(D_out_target, target_label_val))
            loss_D_target.backward()
            source_images = source_images.to(DEVICE)
            with torch.no_grad(): source_preds, _, _ = generator(source_images)
            D_out_source = discriminator(source_preds.detach())
            loss_D_source = criterion_adv(D_out_source, torch.full_like(D_out_source, source_label_val))
            loss_D_source.backward()
            optimizer_D.step()

            # G-Step
            optimizer_G.zero_grad()
            final_pred_G, aux1_pred_G, aux2_pred_G = generator(source_images)
            loss_G_seg_main = criterion_seg(final_pred_G, source_seg_labels.to(DEVICE))
            loss_G_seg_aux1 = criterion_seg(aux1_pred_G, source_seg_labels.to(DEVICE))
            loss_G_seg_aux2 = criterion_seg(aux2_pred_G, source_seg_labels.to(DEVICE))
            loss_G_seg = loss_G_seg_main + AUX_LOSS_WEIGHT * (loss_G_seg_aux1 + loss_G_seg_aux2)
            D_out_adv = discriminator(final_pred_G)
            loss_G_adv = criterion_adv(D_out_adv, torch.full_like(D_out_adv, target_label_val))
            loss_G = loss_G_seg + args.lambda_adv * loss_G_adv
            loss_G.backward()
            optimizer_G.step()
            
            # Update LR and Progress Bar
            # Note: For simplicity, we use the same poly decay for Adam's LR here.
            optimizer_G.param_groups[0]['lr'] = lr_poly(args.lr_g, current_iteration, max_iterations)
            optimizer_D.param_groups[0]['lr'] = lr_poly(args.lr_d, current_iteration, max_iterations)
            current_iteration += 1
            progress_bar.set_postfix(D_Loss=f"{loss_D_source.item() + loss_D_target.item():.4f}", G_Loss_Seg=f"{loss_G_seg.item():.4f}", G_Loss_Adv=f"{loss_G_adv.item():.4f}")

        # Checkpoint Saving
        save_checkpoint(generator, discriminator, optimizer_G, optimizer_D, epoch, run_checkpoint_dir, filename=f"{args.run_name}_epoch_{epoch+1}.pth.tar")
        save_checkpoint(generator, discriminator, optimizer_G, optimizer_D, epoch, run_checkpoint_dir, filename=f"{args.run_name}_latest.pth.tar")
        print(f"\nEpoch {epoch+1} training completed. Checkpoint saved.")

    print(f"\nAdversarial training with Adam finished.")

# --- Script Entry Point ---
if __name__ == "__main__":
    cmd_args = parse_args() 
    main(cmd_args)
