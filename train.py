import torch
import numpy as np
import argparse
import time
from src.settings import SetWANDB
from src import utils
from src.trainer import Trainer
from src.model.RadiationForecasting import NRFormer
from src.DataProcessing import RadiationDataProcessing
import os, sys, random

import wandb, datetime
os.environ['WANDB_MODE'] = 'offline'
wandb.login()

import warnings
warnings.filterwarnings('ignore')

parser = argparse.ArgumentParser()
parser.add_argument('--gpu_ids', type=str, default='1', help='gpu ids: e.g. 0  0,1,2, 0,2. use -1 for CPU')
parser.add_argument('--dataset', type=str, default='1D-data-v2-3841', help='model dataset')
parser.add_argument('--settings', type=str, default='1D-data-v2-3841', help='model settings')
parser.add_argument('--model_name', type=str, default='NRFormer-v2', help='model settings')

parser.add_argument('--out_channels', type=int, default=1, help='')
parser.add_argument('--in_channels', type=int, default=1, help='')
parser.add_argument('--in_length', type=int, default=24, help='')
parser.add_argument('--out_length', type=int, default=24, help='')

parser.add_argument('--use_NSnorm', type=bool, default=True, help='')

parser.add_argument('--temporal_type', type=str, default='Attention', help='Attention')
parser.add_argument('--IsLocationInfo', type=bool, default=True, help='location information, e.g. longitude, latitude')
parser.add_argument('--Is_wind_angle', type=bool, default=True, help='NOAA information, e.g wind angle, wind speed')
parser.add_argument('--Is_wind_speed', type=bool, default=True, help='NOAA information, e.g wind angle, wind speed')
parser.add_argument('--Is_air_temperature', type=bool, default=True, help='NOAA information, e.g wind angle, wind speed')
parser.add_argument('--Is_dew_point', type=bool, default=True, help='NOAA information, e.g wind angle, wind speed')

parser.add_argument('--num_mlp_layer', type=int, default=3, help='number of MLP')
parser.add_argument('--num_temporal_att_layer', type=int, default=3, help='number of attention layers')
parser.add_argument('--num_spatial_att_layer', type=int, default=2, help='number of attention layers')
parser.add_argument('--num_noaa_mlp_layer', type=int, default=2, help='number of attention layers')
parser.add_argument('--num_mlp_dim', type=int, default=32, help='number of MLP')
parser.add_argument('--num_noaa_mlp_dim', type=int, default=64, help='')
parser.add_argument('--num_loc_mlp_dim', type=int, default=32, help='')

parser.add_argument('--epochs', type=int, default=300, help='number of epochs to search')
parser.add_argument('--run_times', type=int, default=1, help='number of run')
parser.add_argument('--model_des', type=str, default='anything', help='save model param')
parser.add_argument('--seed', type=int, default=2024, help='save model param')
args = parser.parse_args()

random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)

my_wandb = SetWANDB(args)
my_wandb.set_wandb()
device = my_wandb.device
config = my_wandb.config
my_wandb.print_config()

# data processing
TFdata = RadiationDataProcessing(config)
scaler = TFdata.scaler
dataloader = TFdata.dataloader
adj_mx_gwn = [torch.tensor(i).to(device) for i in TFdata.adj_mx_gwn]
adj_mx = [torch.tensor(TFdata.adj_mx_dcrnn).to(device), adj_mx_gwn, torch.tensor(TFdata.adj_mx_01).to(device)]
mask_support_adj = [torch.tensor(i).to(device) for i in TFdata.adj_mx_01]

def main(run_id):
    seed = run_id
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True

    model = NRFormer(config, mask_support_adj)

    num_Params = sum([p.nelement() for p in model.parameters()])
    print('Number of model parameters is', num_Params)
    wandb.log({'num_params': num_Params})

    save_folder = './model_param/'+args.dataset
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)
    train_model_path = os.path.join(save_folder, args.dataset+'_best_train_model_'+str(args.model_des)+'.pt')

    engine = Trainer(model, config, scaler, device)

    his_valid_time = []
    his_train_time = []
    his_valid_loss = []
    min_valid_loss = 1000
    best_epoch = 0
    all_start_time = time.time()

    print("start training...\n", flush=True)
    for epoch in range(best_epoch+1, best_epoch+args.epochs+1):
        epoch_start_time = time.time()

        train_loss = []
        train_mae = []
        train_mape = []
        train_rmse = []
        train_start_time = time.time()
        dataloader['train_loader'].shuffle()
        for iter, (x, y) in enumerate(dataloader['train_loader'].get_iterator()):
            train_x = torch.Tensor(x).to(device)
            train_x = train_x.transpose(1, 3)
            train_y = torch.Tensor(y).to(device)
            train_y = train_y.transpose(1, 3)
            train_metrics = engine.train_weight(train_x, dataloader['loc_feature'], train_y[:, 0, :, :])
            train_loss.append(train_metrics[0])
            train_mae.append(train_metrics[1])
            train_mape.append(train_metrics[2])
            train_rmse.append(train_metrics[3])
        engine.weight_scheduler.step()

        train_end_time = time.time()
        t_time = train_end_time - train_start_time
        his_train_time.append(t_time)

        valid_loss = []
        valid_mae = []
        valid_mape = []
        valid_rmse = []
        valid_start_time = time.time()
        for iter, (x, y) in enumerate(dataloader['valid_loader'].get_iterator()):
            val_x = torch.Tensor(x).to(device)
            val_x = val_x.transpose(1, 3)
            val_y = torch.Tensor(y).to(device)
            val_y = val_y.transpose(1, 3)
            val_metrics = engine.eval(val_x, dataloader['loc_feature'], val_y[:, 0, :, :])
            valid_loss.append(val_metrics[0])
            valid_mae.append(val_metrics[1])
            valid_mape.append(val_metrics[2])
            valid_rmse.append(val_metrics[3])

        valid_end_time = time.time()
        v_time = valid_end_time - valid_start_time
        his_valid_time.append(v_time)

        epoch_time = time.time() - epoch_start_time

        mean_train_loss = np.mean(train_loss)
        mean_train_mae = np.mean(train_mae)
        mean_train_mape = np.mean(train_mape)
        mean_train_rmse = np.mean(train_rmse)

        mean_valid_loss = np.mean(valid_loss)
        mean_valid_mae = np.mean(valid_mae)
        mean_valid_mape = np.mean(valid_mape)
        mean_valid_rmse = np.mean(valid_rmse)

        his_valid_loss.append(mean_valid_loss)

        now_time = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        print(now_time)

        print('{}, Epoch: {:03d}, Epoch Training Time: {:.4f}'.format(args.dataset, epoch, epoch_time))
        wandb.log({'Epoch Time': epoch_time})

        log_loss = 'Train Loss: {:.4f}, Train MAE: {:.4f}, Train MAPE: {:.4f}, Train RMSE: {:.4f}, Train Time: {:.4f}\n' \
                   'Valid Loss: {:.4f}, Valid MAE: {:.4f}, Valid MAPE: {:.4f}, Valid RMSE: {:.4f}, Valid Time: {:.4f}\n'
        print(log_loss.format(mean_train_loss, mean_train_mae, mean_train_mape, mean_train_rmse, t_time,
                              mean_valid_loss, mean_valid_mae, mean_valid_mape, mean_valid_rmse, v_time),flush=True)

        wandb.log({'Epoch': epoch,
                   'Epoch Search/Train Loss': mean_train_loss, 'Epoch Search/Train MAE': mean_train_mae,
                   'Epoch Search/Train MAPE': mean_train_mape, 'Epoch Search/Train RMSE': mean_train_rmse,
                   'Epoch Search/Train Time': t_time,

                   'Epoch Valid Loss': mean_valid_loss, 'Epoch Valid MAE': mean_valid_mae,
                   'Epoch Valid MAPE': mean_valid_mape, 'Epoch Valid RMSE': mean_valid_rmse,
                   'Epoch Valid Time': v_time})

        if mean_valid_loss < min_valid_loss:
            best_epoch = epoch
            states = {
                'net': engine.model.state_dict(),
                'weight_optimizer': engine.weight_optimizer.state_dict(),
                'weight_scheduler': engine.weight_scheduler.state_dict(),
                'best_epoch': best_epoch
            }
            torch.save(obj=states, f=train_model_path)
            print('[eval]\tepoch {}\tsave parameters to {}\n'.format(best_epoch, train_model_path))
            min_valid_loss = mean_valid_loss

        elif config['early_stop'] and epoch - best_epoch > config['early_stop_steps']:
            print('-' * 40)
            print('Early Stopped, best train epoch:', best_epoch)
            print('-' * 40)
            break

    all_end_time = time.time()

    print('\n')
    print(datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'))

    best_id = np.argmin(his_valid_loss)
    print("Training finished.")
    best_epoch = engine.load(train_model_path)
    print("The valid loss on best trained model is {}, epoch:{}\n"
          .format(str(round(his_valid_loss[best_id], 4)), best_epoch))

    print("All Training Time: {:.4f} secs".format(all_end_time-all_start_time))
    print("Average Training Time: {:.4f} secs/epoch".format(np.mean(his_train_time)))
    print("Average Inference Time: {:.4f} secs/epoch".format(np.mean(his_valid_time)))
    wandb.log({'All Running Time': all_end_time-all_start_time,
               'Average Epoch Search/Train Time': np.mean(his_train_time),
               'Average Epoch Inference Time': np.mean(his_valid_time)})

    print('\n')
    print("Best Train Model Loaded")

    outputs = []
    true_valid_y = []
    for iter, (x, y) in enumerate(dataloader['valid_loader'].get_iterator()):
        valid_x = torch.Tensor(x).to(device)
        valid_x = valid_x.transpose(1, 3)
        valid_y = torch.Tensor(y).to(device)
        valid_y = valid_y.transpose(1, 3)[:, 0, :, :]

        with torch.no_grad():
            preds = engine.model(valid_x, dataloader['loc_feature'],)

        outputs.append(preds.squeeze(dim=1))
        true_valid_y.append(valid_y)
    valid_yhat = torch.cat(outputs, dim=0)
    true_valid_y = torch.cat(true_valid_y, dim=0)
    # valid_yhat = valid_yhat[:valid_y.size(0), ...]
    valid_pred = scaler.inverse_transform(valid_yhat)
    valid_pred = torch.clamp(valid_pred, min=scaler.min_value, max=scaler.max_value)
    valid_mae, valid_mape, valid_rmse = utils.metric(valid_pred, true_valid_y)

    log = '{} Average Performance on Valid Data - Valid MAE: {:.4f}, Valid MAPE: {:.4f}, Valid RMSE: {:.4f}'
    print(log.format(config['out_length'], valid_mae, valid_mape, valid_rmse))
    wandb.log({"valid MAE": valid_mae, "valid MAPE": valid_mape, "valid RMSE": valid_rmse})

    outputs = []
    true_test_y = []
    for iter, (x, y) in enumerate(dataloader['test_loader'].get_iterator()):
        test_x = torch.Tensor(x).to(device)
        test_x = test_x.transpose(1, 3)
        test_y = torch.Tensor(y).to(device)
        test_y = test_y.transpose(1, 3)[:, 0, :, :]

        with torch.no_grad():
            preds = engine.model(test_x, dataloader['loc_feature'],)

        outputs.append(preds.squeeze(dim=1))
        true_test_y.append(test_y)
    test_yhat = torch.cat(outputs, dim=0)
    true_test_y = torch.cat(true_test_y, dim=0)
    # test_yhat = test_yhat[:test_y.size(0), ...]
    test_pred = scaler.inverse_transform(test_yhat)
    test_pred = torch.clamp(test_pred, min=scaler.min_value, max=scaler.max_value)
    test_mae, test_mape, test_rmse = utils.metric(test_pred, true_test_y)

    log = '{} Average Performance on Test Data - Test MAE: {:.4f}, Test MAPE: {:.4f}, Test RMSE: {:.4f} \n'
    print(log.format(config['out_length'], test_mae, test_mape, test_rmse))
    wandb.log({"test MAE": test_mae, "test MAPE": test_mape, "test RMSE": test_rmse})

    print('Single steps test:')
    mae = []
    mape = []
    rmse = []
    for i in step_list:
        i=i-1
        pred_singlestep = test_pred[:, :, i]
        real = true_test_y[:, :, i]
        metrics_single = utils.metric(pred_singlestep, real)
        log = 'horizon {:d}, Test MAE: {:.4f}, Test MAPE: {:.4f}, Test RMSE: {:.4f}'
        print(log.format(i + 1, metrics_single[0], metrics_single[1], metrics_single[2]))
        wandb.log({str(i + 1) + '_MAE_single': metrics_single[0],
                   str(i + 1) + '_MAPE_single': metrics_single[1],
                   str(i + 1) + '_RMSE_single': metrics_single[2]})
        mae.append(metrics_single[0])
        mape.append(metrics_single[1])
        rmse.append(metrics_single[2])

    print('\nAverage steps test:')
    mae_avg = []
    mape_avg = []
    rmse_avg = []
    for i in step_list:
        pred_avg_step = test_pred[:, :, :i]
        real = true_test_y[:, :, :i]
        metrics_avg = utils.metric(pred_avg_step, real)
        log = 'average {:d}, Test MAE: {:.4f}, Test MAPE: {:.4f}, Test RMSE: {:.4f}'
        print(log.format(i, metrics_avg[0], metrics_avg[1], metrics_avg[2]))
        wandb.log({str(i) + '_MAE_avg': metrics_avg[0],
                   str(i) + '_MAPE_avg': metrics_avg[1],
                   str(i) + '_RMSE_avg': metrics_avg[2]})
        mae_avg.append(metrics_avg[0])
        mape_avg.append(metrics_avg[1])
        rmse_avg.append(metrics_avg[2])

    wandb.finish()
    return valid_mae, valid_mape, valid_rmse, test_mae, test_mape, test_rmse, mae, mape, rmse, mae_avg, mape_avg, rmse_avg


if __name__ == "__main__":

    valid_MAE = []
    valid_MAPE = []
    valid_RMSE = []

    test_MAE = []
    teat_MAPE = []
    test_RMSE = []

    MAE = []
    MAPE = []
    RMSE = []

    MAE_avg = []
    MAPE_avg = []
    RMSE_avg = []

    if args.out_length == 12:
        step_list = [3,6,9,12]
    elif args.out_length == 24:
        step_list = [6,9,12,24]

    for run_num in range(args.run_times):

        run_name = 'NRFormer_train_{}'.format(config['model_des'])
        wandb.init(
            # set the wandb project where this run will be logged
            project='NRFormer-AblationStudy-v2',
            name=run_name,
            config=config
        )

        # track hyperparameters and run metadata
        config = wandb.config
        vm1, vm2, vm3, tm1, tm2, tm3, m1, m2, m3, ma1, ma2, ma3, = main(run_num)

        valid_MAE.append(vm1)
        valid_MAPE.append(vm2)
        valid_RMSE.append(vm3)

        test_MAE.append(tm1)
        teat_MAPE.append(tm2)
        test_RMSE.append(tm3)

        MAE.append(m1)
        MAPE.append(m2)
        RMSE.append(m3)

        MAE_avg.append(ma1)
        MAPE_avg.append(ma2)
        RMSE_avg.append(ma3)

    mae_single = np.mean(np.array(MAE), 0)
    mape_single = np.mean(np.array(MAPE), 0)
    rmse_single = np.mean(np.array(RMSE), 0)

    mae_single_std = np.std(np.array(MAE), 0)
    mape_single_std = np.std(np.array(MAPE), 0)
    rmse_single_std = np.std(np.array(RMSE), 0)

    mae_avg = np.mean(np.array(MAE_avg), 0)
    mape_avg = np.mean(np.array(MAPE_avg), 0)
    rmse_avg = np.mean(np.array(RMSE_avg), 0)

    mae_avg_std = np.std(np.array(MAE_avg), 0)
    mape_avg_std = np.std(np.array(MAPE_avg), 0)
    rmse_avg_std = np.std(np.array(RMSE_avg), 0)

    print('\n')
    print(args.dataset)
    print('\n')

    print('valid\t MAE\t RMSE\t MAPE')
    log = 'mean:\t {:.4f}\t {:.4f}\t {:.4f}'
    print(log.format(np.mean(valid_MAE), np.mean(valid_RMSE), np.mean(valid_MAPE)))
    log = 'std:\t{:.4f}\t{:.4f}\t{:.4f}'
    print(log.format(np.std(valid_MAE), np.std(valid_RMSE), np.std(valid_MAPE)))
    print('\n')

    print('test\t MAE\t RMSE\t MAPE')
    log = 'mean:\t {:.4f}\t {:.4f}\t {:.4f}'
    print(log.format(np.mean(test_MAE), np.mean(test_RMSE), np.mean(teat_MAPE)))
    log = 'std:\t {:.4f}\t {:.4f}\t {:.4f}'
    print(log.format(np.std(test_MAE), np.std(test_RMSE), np.std(teat_MAPE)))
    print('\n')

    print('single test:')
    print('horizon\t MAE-mean\t RMSE-mean\t MAPE-mean\t MAE-std\t RMSE-std\t MAPE-std')
    for i in range(4):
        log = '{:d}\t {:.4f}\t {:.4f}\t {:.4f}\t {:.4f}\t {:.4f}\t {:.4f}'
        print(log.format(step_list[i], mae_single[i], rmse_single[i], mape_single[i], mae_single_std[i],
                         rmse_single_std[i], mape_single_std[i]))

    print('\n')
    print('avg test:')
    print('average\t MAE-mean\t RMSE-mean\t MAPE-mean\t MAE-std\t RMSE-std\t MAPE-std')
    for i in range(4):
        log = '{:d}\t {:.4f}\t {:.4f}\t {:.4f}\t {:.4f}\t {:.4f}\t {:.4f}'
        print(log.format(step_list[i], mae_avg[i], rmse_avg[i], mape_avg[i], mae_avg_std[i],
                         rmse_avg_std[i], mape_avg_std[i]))

    print('Train Finish!\n')
