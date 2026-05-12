"""
zo_optimizer.py — Zero-order optimizer skeleton (student-implemented).

Students: Implement your gradient-free optimization logic inside
``ZeroOrderOptimizer``. The skeleton uses a 2-point central-difference
estimator as a starting point — you are expected to replace or extend it.

Key design points
-----------------
* **Layer selection** is entirely your responsibility. Set ``self.layer_names``
  to the list of parameter names you want to optimize. You can change this list
  at any time — even between ``.step()`` calls — to implement curriculum or
  progressive-layer strategies.
* **Compute budget** is enforced by ``validate.py``: ``.step()`` is called
  exactly ``n_batches`` times. Each call may invoke the model as many times as
  your estimator requires, but be mindful that more evaluations per step leave
  fewer steps in the total budget.
* **No gradients** are computed anywhere in this file. All updates must be
  derived from scalar loss values obtained by calling ``loss_fn()``.
"""

from __future__ import annotations

import os, random
import numpy as np
import json
import sys

import torch
from torch import optim, nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import torchvision.datasets as datasets
from typing import Union, Callable, List, Dict, Any
from dataclasses import dataclass

from augmentation import get_transforms
from model import get_model, get_model_imagenet_head
from train_data import get_train_dataset_loader


# --------------------------------------- DeepZero optimizer --------------------------------------- #

@dataclass
class DZConfig:                       # nicely suited config for DeepZero
    a: float = 1e-2
    mu: float = 1e-3
    n_coords: int = 1024
    clip_value: float = 1.0
    momentum: float = 0.9
    eps: float = 1e-8
    use_symmetric: bool = True
    layer_sampling: bool = True
    max_per_layer: int = 256


class DeepZeroOptimizer:
    def __init__(self, model: nn.Module, layer_names: List[str], config: DZConfig):
        self.model = model
        self.layer_names = layer_names
        self.cfg = config
        self.sum_squared_grads = {n: torch.zeros_like(p) for n, p in model.named_parameters()}
        self.velocity = {n: torch.zeros_like(p) for n, p in model.named_parameters()}

    def _named_params(self) -> Dict[str, nn.Parameter]:
        named = dict(self.model.named_parameters())
        missing = [n for n in self.layer_names if n not in named]
        if missing:
            raise KeyError(f"Missed params: {missing}")
        return {n: named[n] for n in self.layer_names}

    def _eval_loss(self, loss_fn: Callable[[], torch.Tensor | float]) -> float:
        out = loss_fn()
        if isinstance(out, torch.Tensor):
            return float(out.detach().item())
        return float(out)

    @torch.no_grad()
    def _estimate_grad(self, loss_fn: Callable[[], torch.Tensor | float]) -> Dict[str, torch.Tensor]:
        params = self._named_params()
        backup = {n: p.detach().clone() for n, p in params.items()}
        grads = {n: torch.zeros_like(p) for n, p in params.items()}

        base_loss = self._eval_loss(loss_fn)
        layer_items = list(params.items())

        if self.cfg.layer_sampling:
            layer_probs = torch.tensor([p.numel() for _, p in layer_items], device=next(iter(params.values())).device, dtype=torch.float32)
            layer_probs = layer_probs / layer_probs.sum()
            n_layers = min(len(layer_items), max(1, self.cfg.n_coords // max(1, self.cfg.max_per_layer)))
            chosen_layers = torch.multinomial(layer_probs, n_layers, replacement=True).tolist()
        else:
            chosen_layers = list(range(len(layer_items)))

        total_budget = self.cfg.n_coords

        for li in chosen_layers:
            name, param = layer_items[li]
            flat = param.view(-1)
            gflat = grads[name].view(-1)
            budget = min(self.cfg.max_per_layer, total_budget, flat.numel())
            if budget <= 0:
                continue

            idx = torch.randperm(flat.numel(), device=flat.device)[:budget]

            for i in idx:
                orig = flat[i].item()

                if self.cfg.use_symmetric:
                    flat[i] = orig + self.cfg.mu
                    f_plus = self._eval_loss(loss_fn)
                    flat[i] = orig - self.cfg.mu
                    f_minus = self._eval_loss(loss_fn)
                    flat[i] = orig
                    gflat[i] = (f_plus - f_minus) / (2.0 * self.cfg.mu)
                else:
                    flat[i] = orig + self.cfg.mu
                    f_plus = self._eval_loss(loss_fn)
                    flat[i] = orig
                    gflat[i] = (f_plus - base_loss) / self.cfg.mu

            total_budget -= budget
            if total_budget <= 0:
                break

        for n, p in params.items():
            p.copy_(backup[n])

        return grads

    @torch.no_grad()
    def step(self, loss_fn: Callable[[], torch.Tensor | float]) -> float:
        loss_before = self._eval_loss(loss_fn)
        grads = self._estimate_grad(loss_fn)

        for n in grads:
            grads[n].clamp_(-self.cfg.clip_value, self.cfg.clip_value)

        for name, param in self.model.named_parameters():
            if name not in grads:
                continue
            g = grads[name]

            self.sum_squared_grads[name].add_(g * g)
            lr = self.cfg.a / (self.sum_squared_grads[name].sqrt() + self.cfg.eps)

            if self.cfg.momentum > 0:
                self.velocity[name].mul_(self.cfg.momentum).add_(lr * g)
                param.add_(-self.velocity[name])
            else:
                param.add_(-lr * g)

        return loss_before
    

# --------------------------------------- SPSA optimizer --------------------------------------- #

@dataclass
class SPSAConfig:                 # nicely suited config for SPSA 
    a: float = 1e-2
    c: float = 1e-3
    alpha: float = 0.602
    gamma: float = 0.101
    A: float = 10.0
    clip_value: float = 1.0
    momentum: float = 0.0
    eps: float = 1e-8
    n_avg: int = 1


class SPSAOptimizer:
    def __init__(
        self,
        model: nn.Module,
        layer_names: List[str],
        cfg: SPSAConfig,
    ):
        self.model = model
        self.layer_names = layer_names
        self.cfg = cfg

        self.sum_squared_grads = {
            n: torch.zeros_like(p) for n, p in model.named_parameters()
        }
        self.velocity = {
            n: torch.zeros_like(p) for n, p in model.named_parameters()
        }

    def _active_params(self) -> Dict[str, nn.Parameter]:
        named = dict(self.model.named_parameters())
        missing = [n for n in self.layer_names if n not in named]
        if missing:
            raise KeyError(
                f"Missed params: {missing}."
            )
        return {n: named[n] for n in self.layer_names}

    def _eval_loss(self, loss_fn: Callable[[], torch.Tensor | float]) -> float:
        out = loss_fn()
        return float(out.detach().item()) if isinstance(out, torch.Tensor) else float(out)

    @torch.no_grad()
    def _estimate_grad(self, loss_fn: Callable[[], torch.Tensor | float]) -> Dict[str, torch.Tensor]:
        params = self._active_params()
        backup = {name: p.detach().clone() for name, p in params.items()}
        grads = {name: torch.zeros_like(p) for name, p in params.items()}

        n_avg = max(1, self.cfg.n_avg)
        for _ in range(n_avg):
            delta = {}
            for name, p in params.items():
                sign = torch.randint(0, 2, p.shape, device=p.device)
                d = torch.where(sign == 0, -torch.ones_like(p), torch.ones_like(p))
                delta[name] = d

            for name, p in params.items():
                p.add_(self.cfg.c * delta[name])
            f_plus = self._eval_loss(loss_fn)

            for name, p in params.items():
                p.add_(-2.0 * self.cfg.c * delta[name])
            f_minus = self._eval_loss(loss_fn)

            for name, p in params.items():
                p.add_(self.cfg.c * delta[name])

            for name in params:
                grads[name].add_((f_plus - f_minus) / (2.0 * self.cfg.c) * delta[name])

        for name in grads:
            grads[name].div_(float(n_avg))

        for name, p in params.items():
            p.copy_(backup[name])

        return grads

    @torch.no_grad()
    def step(self, loss_fn: Callable[[], torch.Tensor | float]) -> float:
        loss_before = self._eval_loss(loss_fn)
        grads = self._estimate_grad(loss_fn)

        for name in grads:
            grads[name].clamp_(-self.cfg.clip_value, self.cfg.clip_value)

        lr = self.cfg.a / ((1.0 + self.cfg.A) ** self.cfg.alpha)

        for name, param in self.model.named_parameters():
            if name not in grads:
                continue

            g = grads[name]
            self.sum_squared_grads[name].add_(g.pow(2))
            adaptive = lr / (self.sum_squared_grads[name].sqrt() + self.cfg.eps)

            if self.cfg.momentum > 0:
                self.velocity[name].mul_(self.cfg.momentum).add_(adaptive * g)
                param.add_(-self.velocity[name])
            else:
                param.add_(-adaptive * g)

        return loss_before


# ------------------------------------------- Mutual ------------------------------------------- #
# -> Build final ZO optimizer with alternatinf DeepZero and SPSA steps 
# -> Basic setup: 1 SPSA step (more about fast movement) -> 5 DeepZero steps (more about local refinements)

class ZeroOrderOptimizer:
    def __init__(self, model, 
                 layer_names=["fc.weight", "fc.bias"],
                 dz_cfg=DZConfig(),
                 spsa_cfg=SPSAConfig(), 
                 n_spsa_steps=1, 
                 n_dz_steps=5):
        
        self.spsa_opt = SPSAOptimizer(model, layer_names, spsa_cfg)
        self.deepzero_opt = DeepZeroOptimizer(model, layer_names, dz_cfg)
        self.n_spsa_steps = n_spsa_steps
        self.n_dz_steps = n_dz_steps
        self.step_cnt = 0

    def _schedule_opt(self):
        step_within_cycle = self.step_cnt % (self.n_dz_steps + self.n_spsa_steps)
        if step_within_cycle <= self.n_spsa_steps:
            return self.spsa_opt
        else:
            return self.deepzero_opt

    def step(self, *args, **kwargs):
        opt_output = self._schedule_opt().step(*args, **kwargs)
        self.step_cnt += 1

        return opt_output
