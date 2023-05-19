import os
import numpy as np
from tensorboardX import SummaryWriter
import torch
from utils.timer import Timer, AverageMeter
from utils.loss import ContrastiveLoss, CircleLoss, DetLoss, cdist
from utils.metrics import corr_dist, _hash
import open3d as o3d

eps = np.finfo(float).eps

def generate_rand_negative_pairs(positive_pairs, hash_seed, N0, N1, N_neg=0):
    """
    Generate random negative pairs
    """
    if not isinstance(positive_pairs, np.ndarray):
        positive_pairs = np.array(positive_pairs, dtype=np.int64)
    if N_neg < 1:
        N_neg = positive_pairs.shape[0] * 2
    pos_keys = _hash(positive_pairs, hash_seed)

    neg_pairs = np.floor(np.random.rand(int(N_neg), 2) * np.array([[N0, N1]])).astype(
        np.int64)
    neg_keys = _hash(neg_pairs, hash_seed)
    mask = np.isin(neg_keys, pos_keys, assume_unique=False)
    return neg_pairs[np.logical_not(mask)]


class Trainer(object):
    def __init__(self, config):
        # parameters
        # train&val parameters
        self.config = config
        self.start_epoch = 0
        self.max_epoch = config.max_epoch
        self.training_max_iter = config.training_max_iter
        self.val_max_iter = config.val_max_iter
        self.save_dir = config.save_dir
        self.verbose = config.verbose
        self.best_acc = 0
        self.best_loss = 10000000
        self.num_node = 64

        # model parameters
        self.device = config.device
        self.model = config.model.to(self.device)
        self.optimizer = config.optimizer
        self.scheduler = config.scheduler
        self.scheduler_interval = config.scheduler_interval
        self.snapshot_interval = config.snapshot_interval
        self.evaluation_metric = config.evaluation_metric
        self.metric_weight = config.metric_weight
        self.writer = SummaryWriter(log_dir=config.tboard_dir)

        if config.pretrain_time != '':
            self._load_pretrain(f'./results/3dmatch/snapshot/V2PNet-{config.pretrain_time}/models/model_best_acc.pth')

        self.train_loader = config.train_loader
        self.val_loader = config.val_loader

        self.iter_size = config.iter_size
        self.batch_size = config.batch_size

        self.test_valid = True if self.val_loader is not None else False
        self.log_step = int(np.sqrt(self.config.batch_size))

    def train(self):
        """
         Major interface
         Full training logic: train, valid, and save
         """
        self.model.train()

        # Train and valid
        for epoch in range(self.start_epoch, self.max_epoch):
            self.train_epoch(epoch + 1)

            if (epoch + 1) % 1 == 0:
                res = self.valid_epoch(epoch + 1)
                if res['desc_loss'] < self.best_loss:
                    self.best_loss = res['desc_loss']
                    self.snapshot(epoch + 1, 'best_loss')
                if res['accuracy'] > self.best_acc:
                    self.best_acc = res['accuracy']
                    self.snapshot(epoch + 1, 'best_acc')

            for k, v in res.items():
                self.writer.add_scalar(f'val/{k}', v, epoch + 1)

            if (epoch + 1) % self.scheduler_interval == 0:
                self.scheduler.step()

            if (epoch + 1) % self.snapshot_interval == 0:
                self.snapshot(epoch + 1)

        # finish all epoch
        print("Now, save training results.")
        print("Congratulation! Training finish! ")

    def train_epoch(self, epoch):
        # Timers for profiling
        data_timer = Timer()
        model_timer = Timer()

        # define average meter
        desc_loss_meter, det_loss_meter = AverageMeter(), AverageMeter()
        acc_meter = AverageMeter()
        d_pos_meter, d_neg_meter = AverageMeter(), AverageMeter()

        num_iter = int(len(self.train_loader.dataset) // self.train_loader.batch_size)
        num_iter = min(self.training_max_iter, num_iter)
        train_loader_iter = self.train_loader.__iter__()

        # Main training
        for iter in range(num_iter):
            data_timer.tic()
            inputs = train_loader_iter.next()
            for k, v in inputs.items():  # load inputs to device.
                if type(v) == list:
                    inputs[k] = [item.to(self.device) for item in v]
                elif type(v) == torch.Tensor:
                    inputs[k] = v.to(self.device)
                elif type(v) == tuple:
                    inputs[k] = [item.to(self.device) for item in v]
                else:
                    inputs[k] = v

            data_timer.toc()

            model_timer.tic()

            # forward
            self.optimizer.zero_grad()

            features, scores = self.model(inputs)
            anc_features = features[inputs["corr"][:, 0].long()]
            pos_features = features[inputs["corr"][:, 1].long() + inputs['stack_lengths'][0][0]]
            anc_scores = scores[inputs["corr"][:, 0].long()]
            pos_scores = scores[inputs["corr"][:, 1].long() + inputs['stack_lengths'][0][0]]

            descriptor_loss, acc, d_pos, d_neg, dist = self.evaluation_metric["desc_loss"](anc_features, pos_features,
                                                                                        inputs['dist_keypts'])
            detector_loss = self.evaluation_metric['det_loss'](dist, anc_scores, pos_scores)
            loss = descriptor_loss * self.metric_weight['desc_loss'] + detector_loss * self.metric_weight['det_loss']
            d_pos = np.mean(d_pos)
            d_neg = np.mean(d_neg)

            # backward
            loss.backward()

            do_step = True
            for param in self.model.parameters():
                if param.grad is not None:
                    if (1 - torch.isfinite(param.grad).long()).sum() > 0:
                        do_step = False
                        break
            if do_step is True:
                self.optimizer.step()
            model_timer.toc()

            desc_loss_meter.update(float(descriptor_loss))
            det_loss_meter.update(float(detector_loss))
            d_pos_meter.update(float(d_pos))
            d_neg_meter.update(float(d_neg))
            acc_meter.update(float(acc))

            if (iter + 1) % 100 == 0 and self.verbose:
                curr_iter = num_iter * (epoch - 1) + iter
                self.writer.add_scalar('train/Desc_Loss', float(desc_loss_meter.avg), curr_iter)
                self.writer.add_scalar('train/Det_Loss', float(det_loss_meter.avg), curr_iter)
                self.writer.add_scalar('train/D_pos', float(d_pos_meter.avg), curr_iter)
                self.writer.add_scalar('train/D_neg', float(d_neg_meter.avg), curr_iter)
                self.writer.add_scalar('train/Accuracy', float(acc_meter.avg), curr_iter)
                print(f"Train Epoch: {epoch} [{iter + 1:4d}/{num_iter}] "
                             f"desc loss: {desc_loss_meter.avg:.2f} "
                             f"det loss: {det_loss_meter.avg:.2f} "
                             f"acc:  {acc_meter.avg:.2f} "
                             f"d_pos: {d_pos_meter.avg:.2f} "
                             f"d_neg: {d_neg_meter.avg:.2f} "
                             f"data time: {data_timer.avg:.2f}s "
                             f"model time: {model_timer.avg:.2f}s")

        # finish one epoch
        epoch_time = model_timer.total_time + data_timer.total_time
        lr = self.scheduler.get_last_lr()
        print(
            f'Train Epoch {epoch}: Descriptor Loss: {desc_loss_meter.avg:.2f}, Detection Loss : {det_loss_meter.avg:.2f}, Accuracy: {acc_meter.avg:.2f}, D_pos: {d_pos_meter.avg:.2f}, D_neg: {d_neg_meter.avg:.2f}, time {epoch_time:.2f}s, LR: {lr}')

    def valid_epoch(self, epoch):
        self.model.eval()

        data_timer, model_timer, matching_time = Timer(), Timer(), Timer()
        desc_loss_meter, det_loss_meter, acc_meter, d_pos_meter, d_neg_meter = AverageMeter(), AverageMeter(), AverageMeter(), AverageMeter(), AverageMeter()

        num_iter = int(len(self.val_loader.dataset) // self.val_loader.batch_size)
        num_iter = min(self.val_max_iter, num_iter)
        val_loader_iter = self.val_loader.__iter__()

        for iter in range(num_iter):
            data_timer.tic()
            inputs = val_loader_iter.next()
            for k, v in inputs.items():  # load inputs to device.
                if type(v) == list:
                    inputs[k] = [item.to(self.device) for item in v]
                elif type(v) == torch.Tensor:
                    inputs[k] = v.to(self.device)
                elif type(v) == tuple:
                    inputs[k] = [item.to(self.device) for item in v]
                else:
                    inputs[k] = v
            data_timer.toc()

            model_timer.tic()

            features, scores = self.model(inputs)

            anc_features = features[inputs["corr"][:, 0].long()]
            pos_features = features[inputs["corr"][:, 1].long() + inputs['stack_lengths'][0][0]]
            anc_scores = scores[inputs["corr"][:, 0].long()]
            pos_scores = scores[inputs["corr"][:, 1].long() + inputs['stack_lengths'][0][0]]

            descriptor_loss, acc, d_pos, d_neg, dist = self.evaluation_metric['desc_loss'](anc_features, pos_features,
                                                                                        inputs['dist_keypts'])
            detector_loss = self.evaluation_metric['det_loss'](dist, anc_scores, pos_scores)
            loss = descriptor_loss * self.metric_weight['desc_loss'] + detector_loss * self.metric_weight['det_loss']
            d_pos = np.mean(d_pos)
            d_neg = np.mean(d_neg)

            model_timer.toc()

            desc_loss_meter.update(float(descriptor_loss))
            det_loss_meter.update(float(detector_loss))
            d_pos_meter.update(float(d_pos))
            d_neg_meter.update(float(d_neg))
            acc_meter.update(float(acc))

            torch.cuda.empty_cache()
            if (iter + 1) % 100 == 0 and self.verbose:
                print(f"Valid Epoch: {epoch} [{iter + 1:4d}/{num_iter}] "
                      f"descriptor loss: {desc_loss_meter.avg:.2f} "
                      f"detector loss: {det_loss_meter.avg:.2f} "
                      f"acc:  {acc_meter.avg:.2f} "
                      f"d_pos: {d_pos_meter.avg:.2f} "
                      f"d_neg: {d_neg_meter.avg:.2f} "
                      f"data time: {data_timer.avg:.2f}s "
                      f"model time: {model_timer.avg:.2f}s ")

        self.model.train()
        res = {
            'desc_loss': desc_loss_meter.avg,
            'det_loss': det_loss_meter.avg,
            'accuracy': acc_meter.avg,
            'd_pos': d_pos_meter.avg,
            'd_neg': d_neg_meter.avg,
            # 'FMR': feat_match_ratio.avg,
        }
        print(f'Valid Epoch {epoch}: Descriptor Loss {res["desc_loss"]}, Detector Loss {res["det_loss"]}, Accuracy {res["accuracy"]}')
        return res

    def snapshot(self, epoch, name=None):
        state = {
            'epoch': epoch,
            'state_dict': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'scheduler': self.scheduler.state_dict(),
            'best_loss': self.best_loss,
        }
        if name is None:
            filename = os.path.join(self.save_dir, f'model_{epoch}.pth')
        else:
            filename = os.path.join(self.save_dir, f'model_{name}.pth')
        print(f"Save model to {filename}")
        torch.save(state, filename)

    def _load_pretrain(self, resume):
        if os.path.isfile(resume):
            print("==================================Load Pretrain model==================================")
            print(f"Checkpoint File Path: {resume}")
            state = torch.load(resume)
            self.start_epoch = state['epoch']
            self.model.load_state_dict(state['state_dict'])
            self.scheduler.load_state_dict(state['scheduler'])
            self.optimizer.load_state_dict(state['optimizer'])
            self.best_loss = state['best_loss']

            # import pdb
            # pdb.set_trace()
            # self.best_acc = state['best_acc']
        else:
            raise ValueError(f"Cannot Find Checkpoint File Path at '{resume}'")

