import contextlib
import copy
import random
import os
import time
import datetime

import cv2
import numpy as np

import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.nn.functional import softmax

from trainers.models.augModel import DSBN2d, AugModel
from dassl.data import DataManager
from dassl.engine import TRAINER_REGISTRY, TrainerXU, SimpleNet, TrainerX
from dassl.optim import build_lr_scheduler, build_optimizer
from dassl.data.transforms import build_transform
from dassl.utils import count_num_param
from dassl.utils import (
    MetricMeter, AverageMeter, tolist_if_not, count_num_param, load_checkpoint,
    save_checkpoint, mkdir_if_missing, resume_from_checkpoint,
    load_pretrained_weights
)


from tqdm import tqdm



@contextlib.contextmanager
def freeze_models_params(models):
    try:
        for model in models:
            for param in model.parameters():
                param.requires_grad_(False)
        yield
    finally:
        for model in models:
            for param in model.parameters():
                param.requires_grad_(True)


def freeze_models(model):
    for name, param in model.named_parameters():
        param.requires_grad_(False)


def unfreeze_models(model):
    for name, param in model.named_parameters():
        param.requires_grad_(True)


class NormalClassifier(nn.Module):
    def __init__(self, num_features, num_classes):
        super().__init__()
        self.linear = nn.Linear(num_features, num_classes)
    def forward(self, x):
        return self.linear(x)


@TRAINER_REGISTRY.register()
class NormAUG(TrainerX):

    def __init__(self, cfg):
        super().__init__(cfg)
        self.start_step = False

        norm_mean = None
        norm_std = None

        if "normalize" in cfg.INPUT.TRANSFORMS:
            norm_mean = cfg.INPUT.PIXEL_MEAN
            norm_std = cfg.INPUT.PIXEL_STD


    def check_cfg(self, cfg):
        assert len(cfg.TRAINER.NORMAUG.STRONG_TRANSFORMS) > 0
        assert cfg.DATALOADER.TRAIN_X.SAMPLER == "SeqDomainSampler"
        # Sequential domain sampler, which randomly samples K images from each domain to form a minibatch.
        assert cfg.DATALOADER.TRAIN_U.SAME_AS_X

    def build_data_loader(self):
        cfg = self.cfg
        tfm_train = build_transform(cfg, is_train=True)
        custom_tfm_train = [tfm_train]
        choices = cfg.TRAINER.NORMAUG.STRONG_TRANSFORMS
        tfm_train_strong = build_transform(cfg, is_train=True, choices=choices)
        custom_tfm_train += [tfm_train_strong]
        custom_tfm_train += [tfm_train_strong] # global aug
        dm = DataManager(self.cfg, custom_tfm_train=custom_tfm_train)
        self.train_loader_x = dm.train_loader_x
        self.val_loader = dm.val_loader
        self.test_loader = dm.test_loader
        self.num_classes = dm.num_classes
        self.num_source_domains = dm.num_source_domains
        self.lab2cname = dm.lab2cname

    def build_model(self):
        cfg = self.cfg

        print("Building G")
        self.G = AugModel(SimpleNet(cfg, cfg.MODEL, 0), self.num_source_domains)  # n_class=0: only produce features # only output features # basic backbones
        self.G.to(self.device)
        print("# params: {:,}".format(count_num_param(self.G)))
        self.optim_G = build_optimizer(self.G, cfg.OPTIM)
        self.sched_G = build_lr_scheduler(self.optim_G, cfg.OPTIM)
        self.register_model("G", self.G, self.optim_G, self.sched_G)

        print("Building C")
        self.C = DS_Classifier(self.G.fdim, self.num_classes, self.num_source_domains)
        self.C.to(self.device)
        print("# params: {:,}".format(count_num_param(self.C)))
        self.optim_C = build_optimizer(self.C, cfg.TRAINER.NORMAUG.C_OPTIM)
        self.sched_C = build_lr_scheduler(self.optim_C, cfg.TRAINER.NORMAUG.C_OPTIM)
        self.register_model("C", self.C, self.optim_C, self.sched_C)

        print("Building G_C")
        self.G_C = NormalClassifier(self.G.fdim, self.num_classes)
        self.G_C.to(self.device)
        print("# params: {:,}".format(count_num_param(self.G_C)))
        self.optim_G_C = build_optimizer(self.G_C, cfg.TRAINER.NORMAUG.C_OPTIM)
        self.sched_G_C = build_lr_scheduler(self.optim_G_C, cfg.TRAINER.NORMAUG.C_OPTIM)
        self.register_model("G_C", self.G_C, self.optim_G_C, self.sched_G_C)

    def run_epoch(self):
        self.set_model_mode("train")
        losses = MetricMeter()
        batch_time = AverageMeter()
        data_time = AverageMeter()
        self.num_batches = len(self.train_loader_x)

        end = time.time()
        for self.batch_idx, batch in enumerate(self.train_loader_x):
            self.set_model_mode("train")

            # Generate random combination operations
            myop = random.randint(0, self.num_source_domains)

            if myop == self.num_source_domains:
                # Go to the N independent paths
                myop = (1 << self.num_source_domains) - 1
            else:
                # combination path
                myop = (1 << myop) ^ ((1 << self.num_source_domains) - 1)
            loss_summary = self.forward_backward_G(batch, myop)
            data_time.update(time.time() - end)
            batch_time.update(time.time() - end)
            losses.update(loss_summary)

            if (
                self.batch_idx + 1
            ) % self.cfg.TRAIN.PRINT_FREQ == 0 or self.num_batches < self.cfg.TRAIN.PRINT_FREQ:
                nb_remain = 0
                nb_remain += self.num_batches - self.batch_idx - 1
                nb_remain += (
                    self.max_epoch - self.epoch - 1
                ) * self.num_batches
                eta_seconds = batch_time.avg * nb_remain
                eta = str(datetime.timedelta(seconds=int(eta_seconds)))
                print(
                    "epoch [{0}/{1}][{2}/{3}]\t"
                    "time {batch_time.val:.3f} ({batch_time.avg:.3f})\t"
                    "data {data_time.val:.3f} ({data_time.avg:.3f})\t"
                    "eta {eta}\t"
                    "{losses}\t"
                    "lr {lr:.6e}".format(
                        self.epoch + 1,
                        self.max_epoch,
                        self.batch_idx + 1,
                        self.num_batches,
                        batch_time=batch_time,
                        data_time=data_time,
                        eta=eta,
                        losses=losses,
                        lr=self.get_current_lr(),
                    )
                )

            n_iter = self.epoch * self.num_batches + self.batch_idx

            for name, meter in losses.meters.items():
                self.write_scalar("train/" + name, meter.avg, n_iter)
            self.write_scalar("train/lr", self.get_current_lr(), n_iter)

            end = time.time()


    def forward_backward_G(self, batch_x, op):
        parsed_batch = self.parse_batch_train(batch_x)
        y_x_true = parsed_batch["y_x_true"]
        batch_size = y_x_true.size(0)
        x = parsed_batch["x"].chunk(batch_size)
        x_aug = parsed_batch["x_aug"].chunk(batch_size)
        x_g_aug = parsed_batch["g_aug"].chunk(batch_size)
        x_all = []
        for _ in range(batch_size):
            if random.random() < 0.5:
                x_all.append(x[_])
            else:
                x_all.append(x_aug[_])
        # random get strong aug_imgs
        for _ in range(batch_size):
            if random.random() < 0.5:
                x_all.append(x[_])
            else:
                x_all.append(x_g_aug[_])

        y_x_true_ds = y_x_true
        y_x_true_g = y_x_true #torch.cat((y_x_true, y_x_true), 0)
        x_all = torch.cat(x_all, 0)
        bz = x_all.size(0)
        loss_x = 0
        f_x = self.G(x_all, False, op)
        z_x = self.C(f_x[: bz // 2], op)
        z_g = self.G_C(f_x[bz // 2:])
        loss_x += F.cross_entropy(z_x, y_x_true_ds)
        loss_x += F.cross_entropy(z_g, y_x_true_g)

        loss_summary = {}

        loss_all = 0
        loss_all += loss_x
        loss_summary["loss_x"] = loss_x.item()
        self.model_backward_and_update(loss_all, names=['G', 'G_C', 'C'])
        if (self.batch_idx + 1) == self.num_batches:
            self.update_lr()

        return loss_summary


    def parse_batch_train(self, batch_x):
        x0 = batch_x["img0"]  # no augmentation
        x = batch_x["img"]   # weak augmentation
        x_aug = batch_x["img2"]  # strong augmentation
        x_aug_g = batch_x["img3"] # global augmentation
        y_x_true = batch_x["label"]

        x0 = x0.to(self.device)
        x = x.to(self.device)
        x_aug = x_aug.to(self.device)
        x_aug_g = x_aug_g.to(self.device)
        y_x_true = y_x_true.to(self.device)

        batch = {
            "x0": x0,
            "x": x,
            "x_aug": x_aug,
            "g_aug": x_aug_g,
            "y_x_true": y_x_true
        }
        return batch


    def model_inference(self, input):
        input_all = []
        for _ in range(self.num_source_domains + 1):
            input_all.append(input)
        input_all = torch.cat(input_all, 0)
        features = self.G(input_all, True, (1 << self.num_source_domains) - 1)
        features = features.chunk(self.num_source_domains + 1)
        pred = torch.stack(self.C(features[:-1], (1 << self.num_source_domains) - 1).chunk(self.num_source_domains))
        pred1 = self.G_C(features[-1])
        # mean & mean
        return torch.stack((pred.mean(0), pred1)).mean(0)

    def load_model(self, directory, epoch=None):
        if not directory:
            print(
                "Note that load_model() is skipped as no pretrained "
                "model is given (ignore this if it's done on purpose)"
            )
            return

        names = self.get_model_names()

        # By default, the best model is loaded
        model_file = "model-best.pth.tar"

        if epoch is not None:
            model_file = "model.pth.tar-" + str(epoch)
        import os.path as osp
        for name in names:
            model_path = osp.join(directory, name, model_file)

            if not osp.exists(model_path):
                # raise FileNotFoundError(
                #     'Model not found at "{}"'.format(model_path)
                # )
                continue

            checkpoint = load_checkpoint(model_path)
            state_dict = checkpoint["state_dict"]
            epoch = checkpoint["epoch"]

            print(
                "Loading weights to {} "
                'from "{}" (epoch = {})'.format(name, model_path, epoch)
            )
            self._models[name].load_state_dict(state_dict)

    def after_train(self):
        print("Finished training")
        # Do testing
        if not self.cfg.TEST.NO_TEST:
            self.test()
        # Save model
        self.save_model(self.epoch, self.output_dir)

        # Show elapsed time
        elapsed = round(time.time() - self.time_start)
        elapsed = str(datetime.timedelta(seconds=elapsed))
        print("Elapsed: {}".format(elapsed))

        # Close writer
        self.close_writer()


class DS_Classifier(nn.Module):

    def __init__(self, num_features, num_classes, num_domains):
        self.num_domains = num_domains
        self.idx = -1
        super().__init__()
        self.linear = nn.ModuleList(
            [nn.Linear(num_features, num_classes)
             for _ in range(1 << num_domains)])

    def forward(self, x, op):
        if op == (1 << self.num_domains) - 1:
            return self._forward_test(x)
        else:
            return self._forward_op(x, op)

    def _forward_test(self, x):
        out = []
        if not isinstance(x, tuple):
            x = x.chunk(self.num_domains)
        for idx, subx in enumerate(x):
            out.append(self.linear[1 << idx](subx.contiguous()))
        return torch.cat(out, 0)

    def _forward_op(self, x, op):
        out = []
        if not isinstance(x, tuple):
            x = x.chunk(self.num_domains)
        for idx, subx in enumerate(x):
            if (1 << idx) & op:
                out.append(self.linear[op](subx.contiguous()))
            else:
                out.append(self.linear[1 << idx](subx.contiguous()))
        return torch.cat(out, 0)



