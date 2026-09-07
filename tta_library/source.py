
import torch

from tta_library.tta_method import TTAMethod


class Source(TTAMethod):
    def __init__(self, model, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.model = model
        self.model.eval()

    def forward(self, x, y=None):
        with torch.no_grad():
            outputs = self.model(x)
            if self.imagenet_mask is not None:
                outputs = outputs[:, self.imagenet_mask]

        return outputs
