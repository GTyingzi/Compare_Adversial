#!/usr/bin/env python
# -*- coding: utf-8 -*-
# author： yingzi
# datetime： 2022/10/28 20:19 
# ide： PyCharm
import torch

# FGSM
class FGSM:
    def __init__(self, model, eps=1):
        self.model = model
        self.emb_name = 'embeeding'
        self.eps = eps
        self.backup = {}

    def attack(self):
        # emb_name这个参数要换成你模型中embedding的参数名
        for name, param in self.model.named_parameters():

            if param.requires_grad and self.emb_name in name:
                self.backup[name] = param.data.clone()
                r_at = self.eps * param.grad.sign()
                param.data.add_(r_at)

    def restore(self):
        # emb_name这个参数要换成你模型中embedding的参数名
        for name, para in self.model.named_parameters():
            if para.requires_grad and self.emb_name in name:
                assert name in self.backup
                para.data = self.backup[name]
        self.backup = {}


# FGM
class FGM:
    def __init__(self, model, eps=1):
        self.model = model
        self.emb_name = 'embedding'
        self.eps = eps
        self.backup = {}

    def attack(self):
        # emb_name这个参数要换成你模型中embedding的参数名
        for name, param in self.model.named_parameters():
            if param.requires_grad and self.emb_name in name:
                self.backup[name] = param.data.clone()
                norm = torch.norm(param.grad)
                if norm and not torch.isnan(norm):
                    r_at = self.eps * param.grad / norm
                    param.data.add_(r_at)

    def restore(self):
        # emb_name这个参数要换成你模型中embedding的参数名
        for name, para in self.model.named_parameters():
            if para.requires_grad and self.emb_name in name:
                assert name in self.backup
                para.data = self.backup[name]
        self.backup = {}


# PGD
class PGD:
    def __init__(self, model, eps=1, alpha=0.3):
        self.model = model
        self.emb_name = 'embedding'
        self.eps = eps
        self.alpha = alpha
        self.emb_backup = {}
        self.grad_backup = {}

    def attack(self, is_first_attack=False):
        for name, param in self.model.named_parameters():
            if param.requires_grad and self.emb_name in name:
                if is_first_attack:
                    self.emb_backup[name] = param.data.clone()
                norm = torch.norm(param.grad)
                if norm != 0 and not torch.isnan(norm):
                    r_at = self.alpha * param.grad / norm
                    param.data.add_(r_at)
                    param.data = self.project(name, param.data)

    def restore(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad and self.emb_name in name:
                assert name in self.emb_backup
                param.data = self.emb_backup[name]
        self.emb_backup = {}

    def project(self, param_name, param_data):
        r = param_data - self.emb_backup[param_name]
        if torch.norm(r) > self.eps:
            r = self.eps * r / torch.norm(r)
        return self.emb_backup[param_name] + r

    def backup_grad(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad and param.grad is not None:
                self.grad_backup[name] = param.grad.clone()

    def restore_grad(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad and param.grad is not None:
                param.grad = self.grad_backup[name]

# FreeAT
class FreeAT:
    def __init__(self, model, eps=0.1):
        self.model = model
        self.emb_name = 'embedding'
        self.eps = eps
        self.emb_backup = {}
        self.grad_backup = {}
        self.last_r_at = 0

    def attack(self, is_first_attack=False):
        for name, param in self.model.named_parameters():
            if param.requires_grad and self.emb_name in name:
                if is_first_attack:
                    self.emb_backup[name] = param.data.clone()
                param.data.add_(self.last_r_at)
                param.data = self.project(name, param.data)
                self.last_r_at = self.last_r_at + self.eps * param.grad.sign()

    def restore(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad and self.emb_name in name:
                assert name in self.emb_backup
                param.data = self.emb_backup[name]
        self.emb_backup = {}

    def project(self, param_name, param_data):
        r = param_data - self.emb_backup[param_name]
        if torch.norm(r) > self.eps:
            r = self.eps * r / torch.norm(r)
        return self.emb_backup[param_name] + r

    def backup_grad(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad and param.grad is not None:
                self.grad_backup[name] = param.grad.clone()

    def restore_grad(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad and param.grad is not None:
                param.grad = self.grad_backup[name]