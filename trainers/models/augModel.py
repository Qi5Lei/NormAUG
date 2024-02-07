import copy
import torch.nn as nn
import torch
import numpy as np
from torch.nn import InstanceNorm2d

from trainers.models.dson import OptimizedNorm2d


class _DSBN(nn.Module):
    def __init__(self, batchnorm, num_domains, idx, is_fix, op):
        super(_DSBN, self).__init__()
        self._check_bn_type(batchnorm)
        # batchnorm = InstanceNorm2d(batchnorm.num_features)
        batchnorm1 = OptimizedNorm2d(batchnorm.num_features)
        batchnorm1.load_state_dict(batchnorm.state_dict(), strict=False)
        assert num_domains > 0, "The number of domains should be > 0!"
        self.num_domains = num_domains
        self.idx = idx
        self.is_fix = is_fix
        self.op = op
        self.bns = nn.ModuleList(
            [copy.deepcopy(batchnorm) for _ in range(1 << self.num_domains)])
        self.bns.append(batchnorm1)
        # self.scl = SupConLoss(temperature=0.07)
        # self.loss = 0
        # self.scls = nn.ModuleList(
        #     [copy.deepcopy(scl) for _ in range(self.num_domains)]
        # )


    def reset_running_stats(self):
        for bn in self.bns:
            bn.reset_running_stats()

    def reset_parameters(self):
        for bn in self.bns:
            bn.reset_parameters()

    def _check_input_dim(self, inputs):
        raise NotImplementedError

    def _check_bn_type(self, batchnorm):
        raise NotImplementedError

    def forward(self, x):
        if self.op[0] == ((1 << self.num_domains) - 1):
            return self._forward_train(x)
        else:
            return self._forward_train_op(x)

    def _forward_train(self, x):
        bs = x.size(0)
        # assert bs % self.num_domains == 0, "the batch size should be times of BN groups"
        num_block = bs // 48
        num_block = 2 if num_block == 1 else num_block
        # num_block = 2
        out = []
        if num_block == 2: # Train
            x = x.chunk(num_block)
            global_x = []
            for _ in range(1, num_block):
                global_x.append(x[_])
            global_x = torch.cat(global_x, 0)
            split = x[0].chunk(self.num_domains)
            for idx, subx in enumerate(split):
                out.append(self.bns[1 << idx](subx.contiguous()))
            out.append(self.bns[-1](global_x))
            # self.loss += self.scl(torch.cat(out, 0))
        else: # multi domain # Test
            x = x.chunk(self.num_domains + 1)
            for idx, subx in enumerate(x):
                if idx != self.num_domains:
                    out.append(self.bns[1 << idx](subx.contiguous()))
                else:
                    out.append(self.bns[-1](subx.contiguous()))
        return torch.cat(out, 0)

    def _forward_train_op(self, x):
        bs = x.size(0)
        # assert bs % self.num_domains == 0, "the batch size should be times of BN groups"
        num_block = bs // 48
        num_block = 2 if num_block == 1 else num_block
        # num_block = 2
        out = []
        x = x.chunk(num_block)
        global_x = []
        for _ in range(1, num_block):
            global_x.append(x[_])
        global_x = torch.cat(global_x, 0)
        split = x[0].chunk(self.num_domains)
        ds_mix = []
        num = 0
        for idx, subx in enumerate(split):
            if (1 << idx) & self.op[0]:
                # out.append(self.bns[self.op[0]](subx.contiguous()))
                ds_mix.append(subx)
                num += 1
        ds_mix = self.bns[self.op[0]](torch.cat(ds_mix)).chunk(num)
        # ds_mix_1 = self.bns[-1](torch.cat(ds_mix_1))
        ds_mix_pos = 0
        for idx, subx in enumerate(split):
            if (1 << idx) & self.op[0]:
                # out.append(self.bns[self.op[0]](subx.contiguous()))
                # ds_mix.append(subx)
                out.append(ds_mix[ds_mix_pos])
                ds_mix_pos += 1
            else:
                out.append(self.bns[1 << idx](subx.contiguous()))
                # ds_mix_1.append(subx)
        out.append(self.bns[-1](global_x))
        return torch.cat(out, 0)


class DSBN2d(_DSBN):
    def _check_bn_type(self, batchnorm):
        if not isinstance(batchnorm, (nn.BatchNorm2d)):
            raise TypeError('expected BatchNorm2d (got norm type {})'
                            .format(type(batchnorm)))

    def _check_input_dim(self, inputs):
        if inputs.dim() != 4:
            raise ValueError('expected 4D input (got {}D input)'
                             .format(inputs.dim()))

class AugModel(nn.Module):
    def __init__(self, model, num_domains):
        super(AugModel, self).__init__()
        self.model = model
        self.num_domains = num_domains
        self.domain_idx = -1
        self.is_fix = [False]
        self.op = [(1 << num_domains) - 1]
        self._init_dsbn2d(self.model)
        self.fdim = model.fdim


    def _init_dsbn2d(self, model):
        if isinstance(model, DSBN2d):
            return
        for name, module in model.named_children():
            if isinstance(module, (nn.BatchNorm2d)):
                model.__setattr__(name, DSBN2d(module, self.num_domains, self.domain_idx, self.is_fix, self.op))
            else:
                self._init_dsbn2d(module)

    def forward(self, inputs, is_fix=False, op=7):
        # in-place
        self.is_fix.clear()
        self.is_fix.append(is_fix)
        self.op.clear()
        self.op.append(op)
        return self.model(inputs)
