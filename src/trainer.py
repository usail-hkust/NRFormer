import torch.optim as optim
import math
# from net import *
import torch
from src import utils


class Trainer():
    def __init__(self, model, config, scaler, device):
        self.model = model
        self.model.to(device)
        self.scaler = scaler
        self.config = config

        self.iter = 0
        self.task_level = 1
        self.seq_out_len = config['out_length']
        self.max_value = scaler.max_value

        self.loss = utils.masked_mae

        self.weight_optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=float(config['weight_lr']),
            weight_decay=float(config['weight_decay'])
        )

        self.weight_scheduler = torch.optim.lr_scheduler.MultiStepLR(
            optimizer=self.weight_optimizer,
            milestones=list(config['weight_lr_decay_milestones']),
            gamma=float(config['weight_lr_decay_ratio'])
        )

    def train_weight(self, inputs, loc_feature, real_val):

        self.weight_optimizer.zero_grad()
        self.model.train()

        # input (64, 2, 207, 12)
        output = self.model(inputs, loc_feature)
        # output (64, 1, 207, 12)

        real = torch.unsqueeze(real_val, dim=1)
        predict = self.scaler.inverse_transform(output)

        loss = self.loss(predict, real, 0.0)
        loss.backward(retain_graph=False)

        mae = utils.masked_mae(predict, real, 0.0).item()
        mape = utils.masked_mape(predict, real, 0.0).item()
        rmse = utils.masked_rmse(predict, real, 0.0).item()

        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config['weight_clip_gradient'])
        self.weight_optimizer.step()
        self.weight_optimizer.zero_grad()

        return loss.item(), mae, mape, rmse

    def eval(self, inputs, loc_feature, real_val):
        self.model.eval()
        with torch.no_grad():
            output = self.model(inputs, loc_feature)
        # output = output.transpose(1, 3)
        real = torch.unsqueeze(real_val, dim=1)
        predict = self.scaler.inverse_transform(output)

        predict = torch.clamp(predict, min=0., max=self.max_value)

        loss = self.loss(predict, real, 0.0)
        mae = utils.masked_mae(predict, real, 0.0).item()
        mape = utils.masked_mape(predict, real, 0.0).item()
        rmse = utils.masked_rmse(predict, real, 0.0).item()
        return loss.item(), mae, mape, rmse

    def load(self, model_path):
        states = torch.load(model_path)

        # load net
        self.model.load_state_dict(states['net'])

        # load optimizer
        self.weight_optimizer.load_state_dict(states['weight_optimizer'])
        self.weight_scheduler.load_state_dict(states['weight_scheduler'])

        # load historical records
        return states['best_epoch']
