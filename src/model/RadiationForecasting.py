import torch
import torch.nn as nn
import torch.nn.functional as F
import scipy.sparse as sp
import numpy as np
from src.model.transformer import SpatialFormerLayer, SpatialFormer, LearnedPositionalEncoding, SelfAttentionLayer
from src.model.Norm import NSnorm

class SFormer(nn.Module):
    def __init__(self, config):
        super(SFormer, self).__init__()

        self.heads = 8
        self.layers = config.num_spatial_att_layer
        self.hid_dim = config.hidden_channels

        self.attention_layer = SpatialFormerLayer(self.hid_dim, self.heads, self.hid_dim * 4)
        self.attention_norm = nn.LayerNorm(self.hid_dim)
        self.attention = SpatialFormer(self.attention_layer, self.layers, self.attention_norm)
        self.lpos = LearnedPositionalEncoding(self.hid_dim, max_len=config.num_sensors)

    def forward(self, input, input_v, mask):
        # print('hid_dim: ', self.hid_dim)
        x = input.permute(1, 0, 2)
        x_v = input_v.permute(1, 0, 2)
        x = self.lpos(x)
        x_v = self.lpos(x_v)
        output = self.attention(x, x_v, mask)
        output = output.permute(1, 0, 2)
        return output


class MultiLayerPerceptron(nn.Module):
    """Multi-Layer Perceptron with residual links."""

    def __init__(self, input_dim, hidden_dim) -> None:
        super().__init__()
        self.fc1 = nn.Conv2d(
            in_channels=input_dim, out_channels=hidden_dim, kernel_size=(1, 1), bias=True)
        self.fc2 = nn.Conv2d(
            in_channels=hidden_dim, out_channels=hidden_dim, kernel_size=(1, 1), bias=True)
        self.act = nn.ReLU()
        self.drop = nn.Dropout(p=0.15)

    def forward(self, input_data: torch.Tensor) -> torch.Tensor:
        """Feed forward of MLP.

        Args:
            input_data (torch.Tensor): input data with shape [B, D, N]

        Returns:
            torch.Tensor: latent repr
        """

        hidden = self.fc2(self.drop(self.act(self.fc1(input_data))))  # MLP
        hidden = hidden + input_data  # residual
        return hidden

class loc_MLP(nn.Module):
    def __init__(self, num_dim):
        super(loc_MLP, self).__init__()
        self.fc1 = nn.Linear(2, num_dim)
        self.fc2 = nn.Linear(num_dim, num_dim)
        self.fc3 = nn.Linear(num_dim, 32)

    def forward(self, x):
        x = nn.functional.relu(self.fc1(x))
        x = nn.functional.relu(self.fc2(x))
        x = self.fc3(x)
        return x


class SelfAttention(nn.Module):
    def __init__(self, config):
        super(SelfAttention, self).__init__()
        self.hidden_channels = config['hidden_channels']
        self.num_temporal_att_layer = config['num_temporal_att_layer']

        self.time_series_learning = nn.ModuleList(
            [
                SelfAttentionLayer(self.hidden_channels, self.hidden_channels, num_heads=4, dropout=0.3)
                for _ in range(self.num_temporal_att_layer)
            ]
        )

    def forward(self, x):
        x = x.transpose(1, 3)
        for attn in self.time_series_learning:
            x = attn(x, dim=1)
        x = x.transpose(1, 3)
        return x


class NRFormer(nn.Module):
    def __init__(self, config, mask_support_adj):
        super(NRFormer, self).__init__()

        self.config = config

        self.in_length = config['in_length']
        self.out_length = config['out_length']
        self.num_sensors = config['num_sensors']

        self.in_channels = config['in_channels']
        self.out_channels = config['out_channels']
        self.hidden_channels = config['hidden_channels']
        self.end_channels = config['end_channels']

        self.num_temporal_att_layer = config['num_temporal_att_layer']
        self.num_spatial_att_layer = config['num_spatial_att_layer']

        # mlp
        self.num_mlp_layer = config['num_mlp_layer']
        self.num_mlp_dim = config['num_mlp_dim']
        self.num_noaa_mlp_layer = config['num_noaa_mlp_layer']
        self.num_noaa_mlp_dim = config['num_noaa_mlp_dim']
        self.num_loc_mlp_dim = config['num_loc_mlp_dim']

        self.day_size = int(config['day_size'])
        self.month_size = int(config['month_size'])
        self.year_size = int(config['year_size'])

        self.use_NSnorm = config['use_NSnorm']
        if config['use_NSnorm']:
            self.revin = NSnorm(config['num_sensors'])

        # 1. temporal time series learning
        self.start_time_series = nn.Conv2d(in_channels=1, out_channels=self.hidden_channels, kernel_size=(1, 1), bias=True)
        if config['temporal_type'] == 'Attention':
            self.time_series_learning = SelfAttention(self.config)

        self.end_time_series = nn.Conv2d(in_channels=self.hidden_channels * self.in_length,
                                      out_channels=self.hidden_channels, kernel_size=(1, 1), bias=True)

        # 2. location embedding learning
        if self.config['IsLocationInfo']:
            self.loc_mlp = loc_MLP(self.num_loc_mlp_dim)

        # 3. node and time embedding learning
        # time embeddings
        time_embed_num = 0
        # node embeddings
        # self.node_emb = nn.Parameter(torch.empty(self.num_sensors, self.mlp_node_dim))
        # nn.init.xavier_uniform_(self.node_emb)
        # time series, node, time embedding layer
        self.time_series_emb_layer = nn.Conv2d(
            in_channels=(time_embed_num+1) * self.in_length,
            out_channels=self.hidden_channels, kernel_size=(1, 1), bias=True)
        # temporal mlp
        self.hidden_dim = self.hidden_channels
        self.temporal_mlp = nn.Sequential(
            *[MultiLayerPerceptron(self.hidden_dim, self.hidden_dim) for _ in range(self.num_mlp_layer)])

        # 4. meteorological mlp
        if len(self.config['noaa_list'])>0:
            self.meteo_start = nn.Conv2d(
                in_channels=len(self.config['noaa_list']) * self.in_length,
                out_channels=self.num_noaa_mlp_dim, kernel_size=(1, 1), bias=True)
            self.meteo_mlp = nn.Sequential(
                *[MultiLayerPerceptron(self.num_noaa_mlp_dim, self.num_noaa_mlp_dim) for _ in range(self.num_noaa_mlp_layer)])
            self.meteo_end = nn.Conv2d(
                in_channels=self.num_noaa_mlp_dim,
                out_channels=self.hidden_channels, kernel_size=(1, 1), bias=True)

        # 5. temporal fusion
        fusion_dim = 0
        if len(self.config['noaa_list'])>0: fusion_dim += 32
        if self.config['IsLocationInfo']: fusion_dim += 32
        self.temporal_fusion = nn.Conv2d(in_channels=self.hidden_channels*2+fusion_dim, out_channels=self.hidden_channels, kernel_size=(1, 1), bias=True)

        # 6. spatial learning
        mask0 = mask_support_adj[0].detach()
        mask1 = mask_support_adj[1].detach()
        mask = mask0 + mask1
        self.mask = mask == 0
        self.LightTransfer = SFormer(config)

        # 7. end fusion
        end_dim = self.hidden_channels*2
        # if len(self.config['noaa_list']) > 0: end_dim += 32
        self.end_conv1 = nn.Linear(end_dim, self.end_channels)
        self.end_conv2 = nn.Linear(self.end_channels, self.out_length * self.out_channels)

    def forward(self, inputs, loc_feature):
        # inputs [64, 3, 307, 12]
        batch_size, num_features, num_nodes, his_steps = inputs.shape
        all_t_embedding = []

        if self.use_NSnorm:
            x_enc = inputs[:, 0:1, :, :].squeeze(dim=1).transpose(1, 2)
            x_enc = self.revin(x_enc, 'norm')
            x_enc = x_enc.transpose(1, 2).unsqueeze(dim=1)
            if num_features>1:
                inputs = torch.cat((x_enc, inputs[:, 1:, :, :]), dim=1)
            else:
                inputs = x_enc

        # 1. temporal time series learning
        time_series = inputs[:, 0:1, :, :]
        temporal_start = self.start_time_series(time_series)
        temporal_conv = self.time_series_learning(temporal_start)
        temporal_conv = temporal_conv.reshape(batch_size, -1, num_nodes, 1)
        time_series_embedding = self.end_time_series(temporal_conv).squeeze(dim=-1).transpose(1, 2)
        all_t_embedding.append(time_series_embedding)

        # 2. location embedding learning
        if self.config['IsLocationInfo']:
            loc_fts = torch.from_numpy(loc_feature).to(device=self.config['device'], dtype=torch.float)
            loc_embedding = self.loc_mlp(loc_fts)
            loc_embedding = loc_embedding.repeat(batch_size, 1, 1)
            all_t_embedding.append(loc_embedding)

        # 3. node and time embedding learning
        history_data = inputs.transpose(1, 3)
        history_data = history_data[:, :, :, 0:num_features-len(self.config['noaa_list'])]
        # time embeddings
        tem_emb = []
        time_num = 1
        # node embedding
        # node_emb = []
        # node_emb.append(self.node_emb.unsqueeze(0).expand(batch_size, -1, -1).transpose(1, 2).unsqueeze(-1))
        # time series and time embedding layer
        input_data = history_data.transpose(1, 2).contiguous()
        input_data = input_data.view(batch_size, num_nodes, -1).transpose(1, 2).unsqueeze(-1)
        time_series_emb = self.time_series_emb_layer(input_data)
        # concat all embeddings
        hidden = torch.cat([time_series_emb] + tem_emb, dim=1)
        # temporal mlp
        temporal_mlp = self.temporal_mlp(hidden).squeeze(dim=-1).transpose(1, 2).contiguous()
        all_t_embedding.append(temporal_mlp)

        # 4. meteorological mlp
        if len(self.config['noaa_list'])>0:
            meteorological_data = inputs[:, time_num:, :, :]
            meteorological_data = meteorological_data.transpose(1, 2)
            meteorological_data = meteorological_data.reshape(batch_size, num_nodes, -1).transpose(1, 2).unsqueeze(-1)
            meteorological_mlp = self.meteo_start(meteorological_data)
            meteorological_mlp = self.meteo_mlp(meteorological_mlp)
            meteorological_embedding = self.meteo_end(meteorological_mlp).squeeze(dim=-1).transpose(1, 2).contiguous()
            all_t_embedding.append(meteorological_embedding)

        # 5. temporal fusion
        x_temporal = torch.cat(all_t_embedding, dim=-1).unsqueeze(dim=-1).transpose(1,2)
        x_temporal = self.temporal_fusion(x_temporal).squeeze(dim=-1).transpose(1,2)

        # 6. spatial learning
        x_spatial = self.LightTransfer(x_temporal, temporal_mlp, self.mask)

        # 7. end fusion
        x = torch.cat([x_temporal]+[x_spatial], dim=-1)
        x = self.end_conv1(x)
        x = F.relu(x)
        x = self.end_conv2(x)
        x = x.unsqueeze(dim=1)

        if self.use_NSnorm:
            x = x.squeeze(dim=1).transpose(1, 2).contiguous()
            x = self.revin(x, 'denorm')
            x = x.transpose(1, 2).unsqueeze(dim=1).contiguous()

        return x