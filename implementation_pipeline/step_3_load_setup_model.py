import os
import sys

# Add both the parent directory and project root to the Python path
project_root = os.path.dirname(os.path.dirname(__file__))
sys.path.insert(0, project_root)
sys.path.insert(0, os.path.join(project_root, 'tl'))
#add runs directory to path
sys.path.insert(0, os.path.join(project_root, 'runs'))

import sys

import torch
import torch.nn as nn
import numpy as np
import argparse
import copy
from torch.utils.data import DataLoader, TensorDataset
import torch.linalg as LA
from utils.alg_utils import EA_online
from utils.network import backbone_net
from utils.loss import Entropy

from tl.ttime import TTIME

def _reset_batchnorm(m):
    """Reset BatchNorm running statistics"""
    if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
        m.reset_running_stats()

def load_pretrained_model(args, idt_str, extra=""):
    """
    Load pre-trained source model checkpoint
    
    Args:
        args: Arguments namespace with model configuration
        idt_str: Subject identifier string 
        extra: Extra suffix for checkpoint name
    
    Returns:
        model: Loaded PyTorch model ready for inference
    """
    # Initialize network architecture
    netF, netC = backbone_net(args, return_type='xy')
    model = nn.Sequential(netF, netC).to(args.device)
    
    # Load checkpoint - CustomEpoch uses all subject IDs in filename
    if args.data_name == 'CustomEpoch':
        # For CustomEpoch, the checkpoint includes all subjects (0_1_2_3_4_5_6_7_8_9_10_11)
        all_subjects = '_'.join(map(str, range(12)))  # Assuming 12 subjects total
        ckpt_path = f'./runs/{args.data_name}/{args.backbone}_S{all_subjects}_seed{args.SEED}{extra}_best.ckpt'
    else:
        # Standard naming for other datasets
        ckpt_path = f'./runs/{args.data_name}/{args.backbone}_S{idt_str}_seed{args.SEED}{extra}_best.ckpt'
    
    if not os.path.isfile(ckpt_path):
        raise FileNotFoundError(f"Pretrained checkpoint not found: {ckpt_path}")
    
    model.load_state_dict(torch.load(ckpt_path, map_location=args.device))
    model.eval()
    
    print(f"Loaded pretrained model from: {ckpt_path}")
    return model

def setup_pre_tta_model(args, idt_str, extra=""):
    """
    Setup model for Pre-TTA inference (EA alignment only)
    
    Args:
        args: Arguments namespace
        idt_str: Subject identifier string
        extra: Extra suffix for checkpoint name
    
    Returns:
        model: Model ready for Pre-TTA inference
        R: Initial covariance matrix for EA alignment (if args.align=True)
    """
    # Load base model
    model = load_pretrained_model(args, idt_str, extra)
    model.eval()
    
    # Initialize EA alignment matrix if needed
    R = None
    if args.align:
        # Initialize R from source data covariance (placeholder - would need source data)
        R = np.eye(args.chn) * 1e-6  # Simple initialization
    
    print("Pre-TTA model setup complete (EA initialization only)")
    return model, R

def setup_tta_model(args, idt_str, extra=""):
    """
    Setup model for T-TIME Test-Time Adaptation
    
    Args:
        args: Arguments namespace
        idt_str: Subject identifier string 
        extra: Extra suffix for checkpoint name
    
    Returns:
        model: Model ready for TTA
        optimizer: Optimizer for model adaptation
        R: Initial covariance matrix for EA alignment (if args.align=True)
        data_cum: Cumulative data buffer for sliding window
    """
    # Load and setup base model (same as Pre-TTA)
    model, R = setup_pre_tta_model(args, idt_str, extra)
    
    # Setup optimizer for adaptation
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    
    # Initialize cumulative data buffer
    data_cum = None
    
    print("TTA model setup complete (with optimizer and data buffer)")
    return model, optimizer, R, data_cum

def infer_pre_tta(model, trial, args, R=None, trial_idx=0):
    """
    Perform Pre-TTA inference (with optional EA alignment)
    
    Args:
        model: Pre-TTA model 
        trial: Preprocessed trial data (chn, time)
        args: Arguments namespace
        R: EA covariance matrix (updated in-place if not None)
        trial_idx: Trial index for EA update
    
    Returns:
        prediction: Model prediction probabilities
        R: Updated EA matrix (if alignment enabled)
    """
    model.eval()
    
    # Convert to tensor and add batch dimension
    if not isinstance(trial, torch.Tensor):
        trial = torch.from_numpy(trial)
    
    trial = trial.to(args.device, dtype=torch.float32)
    
    # Apply EA alignment if enabled
    if args.align and R is not None:
        # Update R incrementally
        trial_np = trial.cpu().numpy()
        R = EA_online(trial_np, R, trial_idx)
        
        # Compute alignment transform
        R_reg = R + np.eye(R.shape[0]) * 1e-6
        R_tensor = torch.from_numpy(R_reg).to(device=args.device, dtype=trial.dtype)
        eigvals, eigvecs = LA.eigh(R_tensor)
        sqrtRefEA = eigvecs @ torch.diag(eigvals.pow(-0.5)) @ eigvecs.T
        
        # Apply alignment
        trial = sqrtRefEA @ trial
    
    # Reshape for model input (batch, channel, height, width)
    trial_input = trial.reshape(1, 1, args.chn, args.time_sample_num)
    
    # Inference
    with torch.no_grad():
        _, outputs = model(trial_input)
        probabilities = torch.softmax(outputs, dim=1)
    
    return probabilities.cpu().numpy(), R

def infer_tta(model, optimizer, trial, args, R=None, data_cum=None, trial_idx=0):
    """
    Perform T-TIME Test-Time Adaptation inference
    
    Args:
        model: TTA model
        optimizer: Model optimizer
        trial: Preprocessed trial data (chn, time)
        args: Arguments namespace
        R: EA covariance matrix (updated in-place if not None)  
        data_cum: Cumulative data buffer (updated in-place)
        trial_idx: Trial index
    
    Returns:
        prediction: Model prediction probabilities
        R: Updated EA matrix
        data_cum: Updated data buffer
    """
    # Phase 1: Initial prediction (same as Pre-TTA)
    trial_tensor = torch.from_numpy(trial) if not isinstance(trial, torch.Tensor) else trial
    trial_tensor = trial_tensor.to(args.device, dtype=torch.float32)
    
    # Update cumulative data buffer
    trial_input = trial_tensor.reshape(1, 1, args.chn, args.time_sample_num)
    if data_cum is None:
        data_cum = trial_input.clone()
    else:
        data_cum = torch.cat((data_cum, trial_input), 0)
        # Keep only last max_tta samples
        if data_cum.size(0) > args.max_tta:
            data_cum = data_cum[-args.max_tta:]
    
    # Apply EA alignment if enabled
    if args.align and R is not None:
        trial_np = trial_tensor.cpu().numpy()
        R = EA_online(trial_np, R, trial_idx)
        
        R_reg = R + np.eye(R.shape[0]) * 1e-6
        R_tensor = torch.from_numpy(R_reg).to(device=args.device, dtype=trial_tensor.dtype)
        eigvals, eigvecs = LA.eigh(R_tensor)
        sqrtRefEA = eigvecs @ torch.diag(eigvals.pow(-0.5)) @ eigvecs.T
        
        trial_aligned = sqrtRefEA @ trial_tensor
        sample_test = trial_aligned.reshape(1, 1, args.chn, args.time_sample_num)
    else:
        sample_test = trial_input
    
    # Phase 1: Initial inference
    model.eval()
    with torch.no_grad():
        _, outputs = model(sample_test)
        softmax_out = torch.softmax(outputs, dim=1)
    
    # Phase 2: Model adaptation (if confidence is high enough)
    conf, _ = softmax_out.max(dim=1)
    conf_thresh = getattr(args, 'conf_thresh', 0.1)
    
    if conf.item() >= conf_thresh and (trial_idx + 1) >= args.max_tta and (trial_idx + 1) % args.stride == 0:
        model.train()
        win = args.max_tta

        if args.align and R is not None:
            # reuse the running R to align the entire window batch
            raw_batch = data_cum[-win:].squeeze(1)  # (win, chn, time)
            # regularize and move to tensor
            R_reg     = R + np.eye(R.shape[0]) * 1e-6
            R_tensor  = torch.from_numpy(R_reg).to(device=args.device, dtype=raw_batch.dtype)
            eigvals, eigvecs = LA.eigh(R_tensor)
            sqrtRefEA      = eigvecs @ torch.diag(eigvals.pow(-0.5)) @ eigvecs.T
            aligned        = torch.einsum('ij,bjt->bit', sqrtRefEA, raw_batch)
            batch_test     = aligned.unsqueeze(1)    # (win,1,chn,time)
        else:
            batch_test = data_cum[-win:]            # no change here

        # Adaptation steps
        for step in range(args.steps):
            optimizer.zero_grad()
            _, outputs = model(batch_test)
            
            # Entropy minimization loss
            softmax_out_batch = torch.softmax(outputs / args.t, dim=1)
            CEM_loss = torch.mean(Entropy(softmax_out_batch))
            
            # Marginal Distribution Regularization
            msoftmax = softmax_out_batch.mean(dim=0)
            epsilon = getattr(args, 'epsilon', 1e-5)
            MDR_loss = torch.sum(msoftmax * torch.log(msoftmax + epsilon))
            
            loss = CEM_loss + MDR_loss
            loss.backward()
            optimizer.step()
        
        # Phase 3: Post-adaptation inference
        model.eval() 
        with torch.no_grad():
            _, outputs_post = model(sample_test)
            softmax_out = torch.softmax(outputs_post, dim=1)
    
    return softmax_out.cpu().numpy(), R, data_cum

# Example usage function
def setup_inference_pipeline(data_name, subject_id, seed=2, mode='tta'):
    """
    Setup complete inference pipeline for either Pre-TTA or TTA
    
    Args:
        data_name: Dataset name (e.g., 'CustomEpoch')
        subject_id: Target subject ID
        seed: Random seed
        mode: 'pre_tta' or 'tta'
    
    Returns:
        Configured model and inference function
    """
    # Create basic args (you may need to adjust these based on your data)
    args = argparse.Namespace()
    args.data_name = data_name
    args.SEED = seed
    args.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    args.backbone = 'EEGNet'
    args.chn = 27  # Adjust based on your data
    args.time_sample_num = 1515  # Adjust based on your data
    args.class_num = 2  # Binary classification (add this missing attribute)
    args.feature_deep_dim = 1504  # Add this required attribute
    args.align = True
    args.lr = 0.0001
    args.max_tta = 8
    args.stride = 1
    args.steps = 3
    args.t = 1.7
    args.conf_thresh = 0.1
    args.epsilon = 1e-5
    args.sample_rate = 100
    idt_str = str(subject_id)
    
    if mode == 'pre_tta':
        model, R = setup_pre_tta_model(args, idt_str)
        return model, R, args, 'pre_tta'
    elif mode == 'tta':
        model, optimizer, R, data_cum = setup_tta_model(args, idt_str)
        return (model, optimizer, R, data_cum), args, 'tta'
    else:
        raise ValueError("mode must be 'pre_tta' or 'tta'")

if __name__ == "__main__":
    print("Model loading and setup functions ready.")
    
    # Example: Setup Pre-TTA pipeline
    try:
        model, R, args, mode = setup_inference_pipeline('CustomEpoch', subject_id=0, mode='pre_tta')
        print(f"Successfully setup {mode} pipeline")
    except Exception as e:
        print(f"Setup failed: {e}")
