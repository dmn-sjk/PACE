import os
from collections import defaultdict
import torch


class GradientSaver:
    def __init__(self, save_dir, model, optimizer, prefix="gradients"):
        self.prefix = prefix
        self._param_names = {id(p): n for n, p in model.named_parameters()}
        self._unnamed_param_counter = 0
        self.optimizer = optimizer
        self._gradient_records = defaultdict(list)
        self.save_dir = save_dir

        os.makedirs(self.save_dir, exist_ok=True)
        self._counter = 0

    def save(self):
        if self._gradient_records:
            gradient_tensor = {
                name: torch.stack(grads, dim=0)
                for name, grads in self._gradient_records.items()
                if len(grads) > 0
            }
            save_path = os.path.join(
                self.save_dir,
                f"{self.prefix}_{self._counter}.pt"
            )
            torch.save(gradient_tensor, save_path)
            
            self._counter += 1

    def reset(self):
        self._gradient_records = defaultdict(list)
    
    def _get_param_name(self, param: torch.nn.Parameter) -> str:
        name = self._param_names.get(id(param))
        if name is None:
            name = f"unnamed_param_{self._unnamed_param_counter}"
            self._unnamed_param_counter += 1
            self._param_names[id(param)] = name
        return name
    
    def store_gradients(self):
        for group in self.optimizer.param_groups:
            for param in group['params']:
                if param.grad is not None:
                    name = self._get_param_name(param)
                    self._gradient_records[name].append(param.grad.detach().flatten().cpu())
    