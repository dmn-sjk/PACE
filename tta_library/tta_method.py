from typing import Optional

import logging
import torch.nn as nn

from utils.tensorboard_logger import TensorBoardLogger


class TTAMethod(nn.Module):
    def __init__(self, logger: logging.Logger, tb_logger: TensorBoardLogger, args=None, univ: Optional[float] = None):
        super().__init__()
        self.logger = logger
        self.tb_logger = tb_logger
        self.args = args
        self.univ = univ
