"""
Training script for DA3 based on the original paper.
https://arxiv.org/abs/2511.10647

Key features:
1. Camera Input: Inject camera parameters via camera encoding tokens
2. Confidence-weighted depth loss with gradient regularization
3. Ray prediction loss
4. Teacher-student distillation (using DA3 as teacher)
"""
import os
import sys
import argparse
import yaml
from datetime import datetime
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from src.model import DepthAnything3Net
from src.model.utils.transform import extri_intri_to_pose_encoding
from src.model.utils.geometry import affine_inverse
from src.dataset import WaymoDataset
from src.dataset.waymo import collate_fn
from src.losses import DA3Loss


def parse_args():
    parser = argparse.ArgumentParser(description='Train DA3 based on original paper')
    parser.add_argument('--config', type=str, default='config/train_waymo.yaml',
                        help='Path to config file')
    parser.add_argument('--resume', type=str, default=None,
                        help='Path to checkpoint to resume from')
    parser.add_argument('--local_rank', type=int, default=0,
                        help='Local rank for distributed training')
    parser.add_argument('--wandb', action='store_true',
                        help='Use wandb for logging')
    parser.add_argument('--debug', action='store_true',
                        help='Enable debug mode with debugpy')
    return parser.parse_args()


def load_config(config_path):
    """Load configuration from yaml file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def setup_distributed():
    """Setup distributed training."""
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ['WORLD_SIZE'])
        local_rank = int(os.environ['LOCAL_RANK'])
        dist.init_process_group(backend='nccl')
        torch.cuda.set_device(local_rank)
        return rank, world_size, local_rank, True
    else:
        return 0, 1, 0, False


def cleanup_distributed():
    """Cleanup distributed training."""
    if dist.is_initialized():
        dist.destroy_process_group()


class Trainer:
    def __init__(self, config, rank=0, world_size=1, local_rank=0, distributed=False):
        self.config = config
        self.rank = rank
        self.world_size = world_size
        self.local_rank = local_rank
        self.distributed = distributed
        self.device = torch.device(f'cuda:{local_rank}')

        # Create output directory
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        self.output_dir = Path(config.get('output_dir', 'outputs')) / timestamp
        if rank == 0:
            self.output_dir.mkdir(parents=True, exist_ok=True)

        # Setup model
        self.setup_model()

        # Setup teacher model (using DA3 as teacher per user request)
        self.setup_teacher_model()

        # Setup data
        self.setup_data()

        # Setup loss
        self.setup_loss()

        # Setup optimizer and scheduler
        self.setup_optimizer()

        # Setup logging
        if rank == 0:
            self.writer = SummaryWriter(log_dir=str(self.output_dir / 'logs'))
        else:
            self.writer = None

        self.global_step = 0
        self.epoch = 0

        # Training configuration
        train_config = self.config.get('training', {})
        self.use_camera_input = train_config.get('use_camera_input', True)
        self.camera_dropout_prob = train_config.get('camera_dropout_prob', 0.2)

    def setup_model(self):
        """Setup model."""
        model_config = self.config.get('model', {})

        self.model = DepthAnything3Net(
            encoder_name=model_config.get('encoder_name', 'vitl'),
            out_layers=model_config.get('out_layers', [11, 15, 19, 23]),
            features=model_config.get('features', 256),
            out_channels=model_config.get('out_channels', [256, 512, 1024, 1024]),
            alt_start=model_config.get('alt_start', 8),
            qknorm_start=model_config.get('qknorm_start', 8),
            rope_start=model_config.get('rope_start', 8),
            predict_camera=model_config.get('predict_camera', True),
            # Enable camera encoder for camera input
            use_camera_enc=model_config.get('use_camera_enc', True),
        )

        # Load pretrained weights
        pretrained_path = model_config.get('pretrained', None)
        if pretrained_path and os.path.exists(pretrained_path):
            self.model.load_pretrained(pretrained_path)
            if self.rank == 0:
                print(f"Loaded pretrained weights from {pretrained_path}")

        self.model = self.model.to(self.device)

        if self.distributed:
            self.model = DDP(self.model, device_ids=[self.local_rank],
                            find_unused_parameters=True)

    def setup_teacher_model(self):
        """
        Setup teacher model for knowledge distillation.

        Per user request: Use DA3 model as teacher (since DA3-Teacher is not open-sourced).
        The teacher model is frozen and used to generate pseudo-labels.
        """
        teacher_config = self.config.get('teacher', {})
        use_teacher = teacher_config.get('enabled', False)

        if not use_teacher:
            self.teacher_model = None
            if self.rank == 0:
                print("Teacher model disabled")
            return

        model_config = self.config.get('model', {})

        # Create teacher model with same architecture
        self.teacher_model = DepthAnything3Net(
            encoder_name=model_config.get('encoder_name', 'vitl'),
            out_layers=model_config.get('out_layers', [11, 15, 19, 23]),
            features=model_config.get('features', 256),
            out_channels=model_config.get('out_channels', [256, 512, 1024, 1024]),
            alt_start=model_config.get('alt_start', 8),
            qknorm_start=model_config.get('qknorm_start', 8),
            rope_start=model_config.get('rope_start', 8),
            predict_camera=False,  # Teacher doesn't need camera prediction
            use_camera_enc=False,  # Teacher doesn't use camera encoding
        )

        # Load teacher weights
        teacher_path = teacher_config.get('checkpoint', model_config.get('pretrained', None))
        if teacher_path and os.path.exists(teacher_path):
            self.teacher_model.load_pretrained(teacher_path)
            if self.rank == 0:
                print(f"Loaded teacher weights from {teacher_path}")
        else:
            if self.rank == 0:
                print("Warning: No teacher checkpoint specified, using initialized weights")

        self.teacher_model = self.teacher_model.to(self.device)
        self.teacher_model.eval()

        # Freeze teacher model
        for param in self.teacher_model.parameters():
            param.requires_grad = False

        if self.rank == 0:
            print("Teacher model setup complete (frozen)")

    def setup_data(self):
        """Setup data loaders."""
        data_config = self.config.get('data', {})

        # Training dataset
        train_dataset = WaymoDataset(
            root=data_config.get('train_root'),
            valid_camera_id_list=data_config.get('camera_ids', ["1", "2", "3"]),
            intervals=data_config.get('intervals', [1, 2, 3]),
            num_views=data_config.get('num_views', 4),
            resolution=data_config.get('resolution', 518),
            split='train',
        )

        if self.distributed:
            train_sampler = DistributedSampler(train_dataset, shuffle=True)
        else:
            train_sampler = None

        self.train_loader = DataLoader(
            train_dataset,
            batch_size=data_config.get('batch_size', 2),
            shuffle=(train_sampler is None),
            sampler=train_sampler,
            num_workers=data_config.get('num_workers', 4),
            pin_memory=True,
            collate_fn=collate_fn,
            drop_last=True,
        )
        self.train_sampler = train_sampler

        # Validation dataset (optional)
        val_root = data_config.get('val_root')
        if val_root and os.path.exists(val_root):
            val_dataset = WaymoDataset(
                root=val_root,
                valid_camera_id_list=data_config.get('camera_ids', ["1", "2", "3"]),
                intervals=[1],
                num_views=data_config.get('num_views', 4),
                resolution=data_config.get('resolution', 518),
                split='val',
            )
            self.val_loader = DataLoader(
                val_dataset,
                batch_size=1,
                shuffle=False,
                num_workers=2,
                pin_memory=True,
                collate_fn=collate_fn,
            )
        else:
            self.val_loader = None

    def setup_loss(self):
        """Setup loss function based on DA3 paper."""
        loss_config = self.config.get('loss', {})

        self.criterion = DA3Loss(
            # Depth loss (confidence-weighted L1 with log regularization)
            use_depth=loss_config.get('use_depth', True),
            depth_weight=loss_config.get('depth_weight', 1.0),
            depth_gamma=loss_config.get('depth_gamma', 1.0),
            depth_alpha=loss_config.get('depth_alpha', 0.2),
            depth_valid_range=loss_config.get('depth_valid_range', 0.95),
            disable_depth_conf=loss_config.get('disable_depth_conf', False),
            # Gradient loss (edge-aware)
            use_gradient=loss_config.get('use_gradient', True),
            gradient_weight=loss_config.get('gradient_weight', 1.0),
            gradient_loss_type=loss_config.get('gradient_loss_type', 'grad'),
            # Ray loss
            use_ray=loss_config.get('use_ray', True),
            ray_weight=loss_config.get('ray_weight', 1.0),
            # Point loss (L_P in paper)
            use_point=loss_config.get('use_point', True),
            point_weight=loss_config.get('point_weight', 1.0),
            point_gamma=loss_config.get('point_gamma', 1.0),
            point_alpha=loss_config.get('point_alpha', 0.2),
            point_valid_range=loss_config.get('point_valid_range', 0.95),
            disable_point_conf=loss_config.get('disable_point_conf', False),
            # Camera loss
            use_camera=loss_config.get('use_camera', True),
            camera_weight=loss_config.get('camera_weight', 1.0),
            camera_loss_type=loss_config.get('camera_loss_type', 'l1'),
            camera_weight_T=loss_config.get('camera_weight_T', 1.0),
            camera_weight_R=loss_config.get('camera_weight_R', 1.0),
            camera_weight_fl=loss_config.get('camera_weight_fl', 0.5),
            # Teacher settings (per paper: switch at 120k out of 200k steps)
            use_teacher=loss_config.get('use_teacher', False),
            switch_to_teacher_step=loss_config.get('switch_to_teacher_step', 120000),
        )

        if self.rank == 0:
            print(f"Loss configuration:")
            print(f"  - Depth loss: {loss_config.get('use_depth', True)}")
            print(f"  - Gradient loss: {loss_config.get('use_gradient', True)}")
            print(f"  - Ray loss: {loss_config.get('use_ray', True)}")
            print(f"  - Point loss: {loss_config.get('use_point', True)}")
            print(f"  - Camera loss: {loss_config.get('use_camera', True)}")
            print(f"  - Teacher: {loss_config.get('use_teacher', False)} (switch at step {loss_config.get('switch_to_teacher_step', 120000)})")

    def setup_optimizer(self):
        """Setup optimizer and scheduler."""
        optim_config = self.config.get('optimizer', {})

        # Get model parameters
        model = self.model.module if self.distributed else self.model

        # Separate encoder and decoder parameters
        encoder_params = []
        decoder_params = []
        for name, param in model.named_parameters():
            if 'backbone' in name or 'encoder' in name:
                encoder_params.append(param)
            else:
                decoder_params.append(param)

        # Different learning rates for encoder and decoder
        param_groups = [
            {'params': encoder_params, 'lr': optim_config.get('encoder_lr', 1e-6)},
            {'params': decoder_params, 'lr': optim_config.get('decoder_lr', 1e-4)},
        ]

        self.optimizer = optim.AdamW(
            param_groups,
            weight_decay=optim_config.get('weight_decay', 0.01),
        )

        # Learning rate scheduler
        scheduler_config = self.config.get('scheduler', {})
        if scheduler_config.get('type') == 'cosine':
            self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=scheduler_config.get('T_max', 100),
                eta_min=scheduler_config.get('eta_min', 1e-7),
            )
        elif scheduler_config.get('type') == 'step':
            self.scheduler = optim.lr_scheduler.StepLR(
                self.optimizer,
                step_size=scheduler_config.get('step_size', 30),
                gamma=scheduler_config.get('gamma', 0.1),
            )
        else:
            self.scheduler = None

    def get_teacher_predictions(self, images):
        """
        Get teacher model predictions.

        Args:
            images: [B, S, C, H, W] input images

        Returns:
            teacher_depth: [B, S, H, W] teacher depth predictions
        """
        if self.teacher_model is None:
            return None

        with torch.no_grad():
            teacher_outputs = self.teacher_model(images)
            return teacher_outputs['depth']

    def train_epoch(self):
        """Train for one epoch."""
        self.model.train()

        if self.train_sampler is not None:
            self.train_sampler.set_epoch(self.epoch)

        loss_config = self.config.get('loss', {})

        for batch_idx, batch in enumerate(self.train_loader):
            # Move batch to device
            batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v
                    for k, v in batch.items()}

            # Forward pass
            self.optimizer.zero_grad()

            model = self.model.module if self.distributed else self.model

            # Camera input: pass extrinsics and intrinsics to model
            # Based on DA3 paper Section 3.2: Input-adaptive camera conditioning
            # With probability camera_dropout_prob, don't provide camera input 
            use_camera_cond = self.use_camera_input and torch.rand(1).item() > self.camera_dropout_prob

            if use_camera_cond and 'extrinsics' in batch and 'intrinsics' in batch:
                outputs = model(
                    batch['images'],
                    extrinsics=batch['extrinsics'],
                    intrinsics=batch['intrinsics'],
                )
            else:
                outputs = model(batch['images'])

            # Get teacher predictions if needed
            teacher_depth = None
            if loss_config.get('use_teacher', False) and self.teacher_model is not None:
                teacher_depth = self.get_teacher_predictions(batch['images'])

            # Compute losses using DA3Loss
            # Pass current_step so loss can decide when to switch to teacher labels
            loss_dict = self.criterion(
                outputs, batch,
                teacher_depth=teacher_depth,
                current_step=self.global_step
            )
            total_loss = loss_dict['total_loss']

            # Backward pass
            total_loss.backward()

            # Gradient clipping
            max_grad_norm = self.config.get('optimizer', {}).get('max_grad_norm', 1.0)
            if max_grad_norm > 0:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_grad_norm)

            self.optimizer.step()

            # Logging
            if self.rank == 0 and batch_idx % self.config.get('log_interval', 10) == 0:
                log_str = f"Epoch {self.epoch} [{batch_idx}/{len(self.train_loader)}]"
                log_str += f" total_loss: {total_loss.item():.4f}"
                for k, v in loss_dict.items():
                    if k == 'total_loss':
                        continue
                    if isinstance(v, torch.Tensor):
                        log_str += f" {k}: {v.item():.4f}"
                        if self.writer:
                            self.writer.add_scalar(f'train/{k}', v.item(), self.global_step)
                    elif isinstance(v, (int, float)):
                        log_str += f" {k}: {v:.4f}"
                        if self.writer:
                            self.writer.add_scalar(f'train/{k}', v, self.global_step)
                print(log_str)

            self.global_step += 1

    @torch.no_grad()
    def validate(self):
        """Run validation."""
        if self.val_loader is None:
            return {}

        self.model.eval()
        val_losses = {}

        for batch in self.val_loader:
            batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v
                    for k, v in batch.items()}

            model = self.model.module if self.distributed else self.model

            # Use camera input during validation
            if 'extrinsics' in batch and 'intrinsics' in batch:
                outputs = model(
                    batch['images'],
                    extrinsics=batch['extrinsics'],
                    intrinsics=batch['intrinsics'],
                )
            else:
                outputs = model(batch['images'])

            # Compute losses (use current global_step for teacher switch logic)
            loss_dict = self.criterion(outputs, batch, current_step=self.global_step)
            for k, v in loss_dict.items():
                if k not in val_losses:
                    val_losses[k] = []
                if isinstance(v, torch.Tensor):
                    val_losses[k].append(v.item())
                elif isinstance(v, (int, float)):
                    val_losses[k].append(v)

        # Average losses
        avg_losses = {k: sum(v) / len(v) for k, v in val_losses.items() if v}

        if self.rank == 0:
            for k, v in avg_losses.items():
                if self.writer:
                    self.writer.add_scalar(f'val/{k}', v, self.epoch)
            print(f"Validation - " + " ".join([f"{k}: {v:.4f}" for k, v in avg_losses.items()]))

        return avg_losses

    def save_checkpoint(self, filename='checkpoint.pth'):
        """Save checkpoint."""
        if self.rank != 0:
            return

        model = self.model.module if self.distributed else self.model

        checkpoint = {
            'epoch': self.epoch,
            'global_step': self.global_step,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'config': self.config,
        }

        if self.scheduler is not None:
            checkpoint['scheduler_state_dict'] = self.scheduler.state_dict()

        save_path = self.output_dir / filename
        torch.save(checkpoint, save_path)
        print(f"Saved checkpoint to {save_path}")

    def load_checkpoint(self, checkpoint_path):
        """Load checkpoint."""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)

        model = self.model.module if self.distributed else self.model
        model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

        if self.scheduler is not None and 'scheduler_state_dict' in checkpoint:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

        self.epoch = checkpoint['epoch']
        self.global_step = checkpoint['global_step']

        print(f"Loaded checkpoint from {checkpoint_path} (epoch {self.epoch})")

    def train(self):
        """Main training loop."""
        train_config = self.config.get('training', {})
        num_epochs = train_config.get('num_epochs', 100)
        save_interval = train_config.get('save_interval', 5)
        val_interval = train_config.get('val_interval', 5)

        for epoch in range(self.epoch, num_epochs):
            self.epoch = epoch

            if self.rank == 0:
                print(f"\n{'='*50}")
                print(f"Epoch {epoch + 1}/{num_epochs}")
                print(f"{'='*50}")

            self.train_epoch()

            # Update learning rate
            if self.scheduler is not None:
                self.scheduler.step()

            # Validation
            if (epoch + 1) % val_interval == 0:
                self.validate()

            # Save checkpoint
            if self.rank == 0 and (epoch + 1) % save_interval == 0:
                self.save_checkpoint(f'checkpoint_epoch{epoch+1}.pth')

        # Final save
        if self.rank == 0:
            self.save_checkpoint('checkpoint_final.pth')


def main():
    args = parse_args()

    # debug
    if args.debug:
        import debugpy
        debugpy.listen(5678)
        print("Waiting for debugger attach...")
        debugpy.wait_for_client()
        print("Debugger attached.")

    # Setup distributed training
    rank, world_size, local_rank, distributed = setup_distributed()

    # Load config
    config = load_config(args.config)

    # Create trainer
    trainer = Trainer(
        config,
        rank=rank,
        world_size=world_size,
        local_rank=local_rank,
        distributed=distributed,
    )

    # Resume from checkpoint
    if args.resume:
        trainer.load_checkpoint(args.resume)

    # Train
    try:
        trainer.train()
    finally:
        cleanup_distributed()


if __name__ == '__main__':
    main()
