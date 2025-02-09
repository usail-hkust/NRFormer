from collections import namedtuple
import datetime
import ruamel.yaml as yaml
import torch
import os, sys

def dict_to_namedtuple(dic: dict):
    return namedtuple('tuple', dic.keys())(**dic)


class Settings:
    def __init__(self):
        pass

    def load_settings(self, dataset):
        settings_file = './model_settings/'+dataset+'.yaml'
        with open(settings_file, 'r') as f:
            setting = yaml.load(f, Loader=yaml.RoundTripLoader)
        self.data = dict_to_namedtuple(setting['data'])
        self.model = dict_to_namedtuple(setting['model'])
        self.trainer = dict_to_namedtuple(setting['trainer'])

        self.data_dict = setting['data']
        self.model_dict = setting['model']
        self.trainer_dict = setting['trainer']


def load_server_config():

    settings_file = '../server_config.yaml'
    with open(settings_file, 'r') as f:
        setting = yaml.load(f, Loader=yaml.RoundTripLoader)
    server_config = dict_to_namedtuple(setting['config'])

    return server_config

class SetWANDB():
    def __init__(self, args):
        self.args = args

        # load model parameters
        settings = Settings()
        settings.load_settings(args.settings)
        self.settings = settings

    def print_config(self):
        print(self.args)
        print("settings.data:")
        for key, value in self.settings.data_dict.items():
            print("\t{}: {}".format(key, value))

        print("settings.trainer:")
        for key, value in self.settings.trainer_dict.items():
            print("\t{}: {}".format(key, value))

        print("settings.model:")
        for key, value in self.settings.model_dict.items():
            if str(key) not in 'candidate_op_profiles':
                print("\t{}: {}".format(key, value))
            else:
                print("\t{}:".format(key))
                for i in value:
                    print("\t\t{}: {}".format(i[0], i[1]))
        print('\n')

    def set_wandb(self):
        # load server config
        server_config = load_server_config()
        if server_config.server_name in ['ust29']:
            ROOT_PATH = '../../../..'
            DATA_PATH = ROOT_PATH + '/nas_data/lyutengfei/MyDatasets/RadiationForecasting'
        elif server_config.server_name in ['ust27', 'ust35', 'ust36', 'T12', 'T243', 'MacBook Pro', 'HPC']:
            DATA_PATH = '../MyDatasets/RadiationForecasting'
        else:
            DATA_PATH = None
            print('DATA_PATH error...')
            sys.exit()

        noaa_list = []
        if self.args.Is_wind_angle:
            noaa_list.append('wind_angle')
        if self.args.Is_wind_speed:
            noaa_list.append('wind_speed')
        if self.args.Is_air_temperature:
            noaa_list.append('air_temperature')
        if self.args.Is_dew_point:
            noaa_list.append('dew_point')

        # check of time embedding for different datasets
        if 'D' in self.args.dataset:
            self.args.IsTimeEmbedding = False
        elif 'month' in self.args.dataset:
            self.args.IsTimeEmbedding = False
            self.args.IsDayEmbedding = False
        in_channels = 1
        in_channels = in_channels + len(noaa_list)

        # set gpu ids
        str_ids = self.args.gpu_ids.split(',')
        self.args.gpu_ids = []
        for str_id in str_ids:
            id = int(str_id)
            if id >= 0:
                self.args.gpu_ids.append(id)
        if len(self.args.gpu_ids) > 0:
            torch.cuda.manual_seed_all(self.args.seed)
            torch.cuda.set_device(self.args.gpu_ids[0])
        self.device = torch.device('cuda:{}'.format(self.args.gpu_ids[0])) if self.args.gpu_ids else torch.device('cpu')

        self.config = {
            'model_name': self.args.model_name,
            'dataset': self.args.dataset,
            'settings': self.args.settings,
            'epochs': self.args.epochs,
            'model_des': self.args.model_des,

            'use_NSnorm':self.args.use_NSnorm,

            'data_path': DATA_PATH,
            'device': self.device,

            'run_time': datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'),

            'server_name': server_config.server_name,
            'GPU_type': server_config.GPU_type,

            'weight_lr': self.settings.trainer.weight_lr,
            'weight_lr_decay_ratio': self.settings.trainer.weight_lr_decay_ratio,
            'weight_decay': self.settings.trainer.weight_decay,
            'weight_clip_gradient': self.settings.trainer.weight_clip_gradient,
            'weight_lr_decay_milestones': self.settings.trainer.weight_lr_decay_milestones,

            'temporal_type': self.args.temporal_type,
            'IsLocationInfo': self.args.IsLocationInfo,
            'noaa_list': noaa_list,

            'distance': self.settings.data.distance,

            'batch_size': self.settings.data.batch_size,
            'early_stop': self.settings.trainer.early_stop,
            'early_stop_steps': self.settings.trainer.early_stop_steps,

            'train_prop': float(self.settings.data.train_prop),
            'valid_prop': float(self.settings.data.valid_prop),

            'day_size': float(self.settings.data.day_size),
            'month_size': float(self.settings.data.month_size),
            'year_size': float(self.settings.data.year_size),

            'num_sensors': self.settings.data.num_sensors,
            'in_channels': in_channels,
            'out_channels': self.args.out_channels,
            'in_length': self.args.in_length,
            'out_length': self.args.out_length,

            'adj_type': self.settings.model.adj_type,
            'end_channels': self.settings.model.end_channels,
            'hidden_channels': self.settings.model.hidden_channels,

            'num_mlp_layer': self.args.num_mlp_layer,
            'num_mlp_dim': self.args.num_mlp_dim,
            'num_temporal_att_layer': self.args.num_temporal_att_layer,
            'num_spatial_att_layer': self.args.num_spatial_att_layer,
            'num_noaa_mlp_layer': self.args.num_noaa_mlp_layer,
            'num_noaa_mlp_dim': self.args.num_noaa_mlp_dim,

            'num_loc_mlp_dim': self.args.num_loc_mlp_dim
        }

    def print_config(self):
        """Print and save options
        It will print both current options and default values(if different).
        It will save options into a text file / [checkpoints_dir] / opt.txt
        """
        message = ''
        message += '--------------------------- config ---------------------------\n'
        for k, v in self.config.items():
            message += '{:>25}: {:<30}\n'.format(str(k), str(v))
        message += '--------------------------- End ---------------------------\n'
        print(message)