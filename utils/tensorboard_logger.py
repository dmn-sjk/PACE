import os
from typing import Dict

from torch.utils.tensorboard import SummaryWriter


class TensorBoardLogger:
    """
    Custom TensorBoard logger for continual adaptation experiments.
    Logs training metrics, validation metrics, and model information.
    """
    
    def __init__(self, 
                 log_dir: str = "logs/tensorboard"):
        """
        Initialize TensorBoard logger.
        
        Args:
            log_dir: Base directory for TensorBoard logs
            experiment_name: Name of the experiment
            seed: Random seed for experiment organization
        """
        self.log_dir = log_dir
        os.makedirs(self.log_dir, exist_ok=True)
        
        self.writer = SummaryWriter(log_dir=self.log_dir)
        
        # Track training step
        self.step = 0
        self.epoch = 0
        
    def log_scalar(self, tag: str, value: float):
        """Log a scalar value."""
        self.writer.add_scalar(tag, value, self.step)
        
    def log_scalars(self, main_tag: str, tag_scalar_dict: Dict[str, float]):
        """Log multiple scalar values with a common prefix."""
        # do not use add_scalars from tb, since it creates new log files for some reason
        for tag, value in tag_scalar_dict.items():
            if value is None:
                continue
            full_tag = f"{main_tag}/{tag}" if main_tag else tag
            self.writer.add_scalar(full_tag, value, self.step)
        
    def set_step(self, step: int):
        self.step = step
        
    def close(self):
        """Close the TensorBoard writer."""
        self.writer.close()
        
    def __enter__(self):
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close() 