import time
import swanlab
import yaml
import argparse
import logging
import numpy as np
from tqdm import tqdm
import os
import random
import torch
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import  DataLoader
from torch.nn.parallel import DataParallel

from easytorch.utils.dist import master_only
from basicts.stgcn_arch import STGCN
from basicts.utils import load_adj, load_pkl
# from basicts.data import TimeSeriesForecastingDataset
from data import PretrainingDataset
from data import ForecastingDataset

from basicts.data.transform import maskTransforms
from basicts.data import SCALER_REGISTRY
from basicts.losses import sce_loss
from basicts.mask.model import pretrain_model, finetune_model
from basicts.metrics import masked_mae,masked_mape,masked_rmse
from basicts.utils.utils import metric_forward, select_input_features, select_target_features
import matplotlib.pyplot as plt
metrics = {"MAE": masked_mae, "RMSE": masked_rmse, "MAPE": masked_mape, "SCE": sce_loss}

def plot_seq(preTrain_train_dataset):

    print("history_data")
    # print(history_data.shape)
    # print(future_data.shape)
    print(preTrain_train_dataset[0][0].shape)
    x = [1, 2, 3, 4, 5]
    y = [2, 4, 6, 8, 10]

    # 绘制折线图
    plt.figure(figsize=(8, 6))
    plt.plot(x, y, marker='o', linestyle='-', linewidth=2)

    # 添加标题和标签
    plt.title('Line Chart Example', fontsize=16)
    plt.xlabel('X-axis', fontsize=12)
    plt.ylabel('Y-axis', fontsize=12)

    # 显示网格
    plt.grid(True, linestyle='--', alpha=0.7)

    # 显示图形
    plt.show()
def val(val_data_loader, model, config, scaler, epoch):
    model.eval()
    
    prediction = []
    real_value = []
    with torch.no_grad():
        for data in tqdm(val_data_loader, ncols=100):
            future_data, history_data, long_history_data = data
            batch_size = future_data.shape[0]
            long_history_data = select_input_features(long_history_data, config['froward_features'])
            history_data = select_input_features(history_data, config['target_features'])
            future_data = select_input_features(future_data, config['target_features'])
            
            labels = future_data.to(config['device'])
            history_data = history_data.to(config['device'])
            long_history_data = long_history_data.to(config['device'])
            
            preds = model(history_data, long_history_data, future_data, batch_size, epoch)

            prediction.append(preds.detach().cpu())        # preds = forward_return[0]
            real_value.append(labels.detach().cpu())        # testy = forward_return[1]

        prediction = torch.cat(prediction, dim=0)
        real_value = torch.cat(real_value, dim=0)
        # re-scale data
        prediction_rescaled = SCALER_REGISTRY.get(scaler["func"])(prediction, **scaler["args"])
        real_value_rescaled = SCALER_REGISTRY.get(scaler["func"])(real_value, **scaler["args"])
        # print(real_value_rescaled)
        # dd
        metric_results = {}
        for metric_name, metric_func in metrics.items():
            metric_item = metric_forward(metric_func, [prediction_rescaled, real_value_rescaled])
            metric_results[metric_name] = metric_item.item()
        print("Evaluate val data" + \
                    "val MAE: {:.4f}, val RMSE: {:.4f}, val MAPE: {:.4f}".format(metric_results["MAE"], metric_results["RMSE"], metric_results["MAPE"]))
        logging.info("Evaluate val data" + \
                    "val MAE: {:.4f}, val RMSE: {:.4f}, val MAPE: {:.4f}".format(metric_results["MAE"], metric_results["RMSE"], metric_results["MAPE"]))
def test(test_data_loader, model, config, scaler, epoch):
    """Evaluate the model.

    Args:
        train_epoch (int, optional): current epoch if in training process.
    """
    model.eval()
    # test loop
    

    prediction = []
    real_value = []
    with torch.no_grad():
        for data in tqdm(test_data_loader, ncols=100):
            future_data, history_data, long_history_data = data
            batch_size = future_data.shape[0]
            long_history_data = select_input_features(long_history_data, config['froward_features'])
            history_data = select_input_features(history_data, config['target_features'])
            future_data = select_input_features(future_data, config['target_features'])
            
            labels = future_data.to(config['device'])
            history_data = history_data.to(config['device'])
            long_history_data = long_history_data.to(config['device'])
            
            preds = model(history_data, long_history_data, future_data, batch_size, epoch)
            
            prediction.append(preds.detach().cpu())        # preds = forward_return[0]
            real_value.append(labels.detach().cpu())        # testy = forward_return[1]s

        prediction = torch.cat(prediction, dim=0)
        real_value = torch.cat(real_value, dim=0)
        # re-scale data
        prediction = SCALER_REGISTRY.get(scaler["func"])(prediction, **scaler["args"])
        real_value = SCALER_REGISTRY.get(scaler["func"])(real_value, **scaler["args"])
        np.savez(f"./save_test/data_{epoch}.npz", array1 = prediction.numpy(), array2 = real_value.numpy())
        metric_results = {}
        for metric_name, metric_func in metrics.items():
            metric_item = metric_forward(metric_func, [prediction, real_value])
            metric_results[metric_name] = metric_item.item()
        MAE = metric_results["MAE"]
        RMSE = metric_results["RMSE"]
        MAPE = metric_results["MAPE"]
        print("Evaluate val data" + \
                    "val MAE: {:.4f}, val RMSE: {:.4f}, val MAPE: {:.4f}".format(MAE, RMSE, MAPE))
        # logging.info("Evaluate val data" + \
        #             "val MAE: {:.4f}, val RMSE: {:.4f}, val MAPE: {:.4f}".format(MAE, RMSE, MAPE))
        swanlab.log({"Evaluate_val_MAE": MAE, "Evaluate_val_RMSE": RMSE, "Evaluate_val_MAPE": MAPE, "epoch": epoch})
        # swanlab.log({"Evaluate val data MAE": loss_all, "epoch": epoch})
        

def finetune(config, args):
    print('### start finetune ... ###')
    adj_mx, _ = load_adj(config['adj_dir'], "doubletransition")
    # print(config)
    # adj_mx = torch.Tensor(adj_mx[0])

    # config['num_node'] = 207
    config['backend_args']['supports'] = [torch.tensor(i) for i in adj_mx]
    # print(config['backend_args']['supports'])
    # print(config['backend_args'])
    # dd
    # adj_mx, _ = load_adj("datasets/" + CFG.DATASET_NAME + "/adj_mx.pkl", "doubletransition")
    scaler = load_pkl(config['scaler_dir'])
    # config['gso'] = adj_mx.to(config['device'])
    
    train_dataset = ForecastingDataset(config['dataset_dir'],config['dataset_index_dir'],'train',config['seq_len'])
    val_dataset = ForecastingDataset(config['dataset_dir'],config['dataset_index_dir'],'valid',config['seq_len'])
    test_dataset = ForecastingDataset(config['dataset_dir'],config['dataset_index_dir'],'test',config['seq_len'])
    
    train_data_loader = DataLoader(train_dataset, batch_size=config['batch_size'], num_workers =8, shuffle=True)
    val_data_loader = DataLoader(val_dataset, batch_size=config['batch_size'], num_workers = 8, shuffle=False)
    test_data_loader = DataLoader(test_dataset, batch_size=config['batch_size'], num_workers = 8, shuffle=False)
    model = finetune_model(config['pre_trained_path'], config['mask_args'], config['backend_args'])
    model = model.to(config['device'])
    optimizer = optim.Adam(model.parameters(), config['lr'], weight_decay=1.0e-5,eps=1.0e-8)
    
    for epoch in range(config['finetune_epochs']):
        print('============ epoch {:d} ============'.format(epoch))
        for idx, data in enumerate(tqdm(train_data_loader, ncols=100)):
            # if idx > 0:
            #     break
            future_data, history_data, long_history_data = data
            batch_size = future_data.shape[0]
            # labels = future_data[:,:,:,config['target_features']]
            # history_data = history_data[:,:,:,config['froward_features']]
            

            long_history_data = select_input_features(long_history_data, config['froward_features'])
            history_data = select_input_features(history_data, config['target_features'])
            future_data = select_input_features(future_data, config['target_features'])
            
            labels = future_data.to(config['device'])
            history_data = history_data.to(config['device'])
            long_history_data = long_history_data.to(config['device'])
            
            preds = model(history_data, long_history_data, future_data, batch_size, epoch)

            prediction_rescaled = SCALER_REGISTRY.get(scaler["func"])(preds, **scaler["args"])
            real_value_rescaled = SCALER_REGISTRY.get(scaler["func"])(labels, **scaler["args"])
            
            
            loss = metric_forward(masked_mae, [prediction_rescaled, real_value_rescaled])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        print('============ val and test ============')
        # val(val_data_loader, model, config, scaler, epoch)
        test(test_data_loader, model, config, scaler, epoch)
        
@torch.no_grad()
@master_only
def preTrain_test(data_loader, model, scaler, mode = 'val'):
    """Evaluate the model.

    Args:
        train_epoch (int, optional): current epoch if in training process.
    """
    model.eval()
    
    prediction = []
    real_value = []
    MAE, RMSE, MAPE = 0.0, 0.0, 0.0
    with torch.no_grad():
        for idx, data in enumerate(tqdm(data_loader, ncols=100)):
            future_data, history_data = data
            # if idx > 0:
            #     break
            # labels = future_data.to(config['device'])
            
            history_data = select_input_features(history_data, config['froward_features'])
            history_data = history_data.to(config['device'])
            reconstruction_masked_tokens, label_masked_tokens = model(history_data, 0)

            # re-scale data

            prediction = reconstruction_masked_tokens.detach().cpu()        # preds = forward_return[0]
            real_value = label_masked_tokens.detach().cpu()        # testy = forward_return[1]s
            prediction_rescaled = SCALER_REGISTRY.get(scaler["func"])(prediction, **scaler["args"])
            real_value_rescaled = SCALER_REGISTRY.get(scaler["func"])(real_value, **scaler["args"])
            # metrics
            metric_results = {}
            for metric_name, metric_func in metrics.items():
                metric_item = metric_forward(metric_func, [prediction_rescaled, real_value_rescaled])
                metric_results[metric_name] = metric_item.item()
            metric_1, metric_2, metric_3 = metric_results["MAE"], metric_results["RMSE"], metric_results["MAPE"]
            MAE += metric_1
            RMSE += metric_2
            MAPE += metric_3
        MAE = MAE / (idx + 1)
        RMSE = RMSE / (idx + 1)
        MAPE = MAPE / (idx + 1)
        MAE = metric_results["MAE"]
        RMSE = metric_results["RMSE"]
        MAPE = metric_results["MAPE"]
        print("Evaluate {} data MAE: {:.4f},  RMSE: {:.4f},  MAPE: {:.4f}".format(mode, MAE, RMSE, MAPE))
        swanlab.log({"Evaluate_MAE": MAE, "Evaluate_RMSE": RMSE, "Evaluate_MAPE": MAPE})


@torch.no_grad()
@master_only
def preTrain_test_save(preTrain_train_dataset, model, scaler, epoch):
    model.eval()
    
    prediction = []
    real_value = []
    MAE, RMSE, MAPE = 0.0, 0.0, 0.0
    with torch.no_grad():
        history_data = preTrain_train_dataset[0][1].unsqueeze(0)
        # print(history_data.shape)
        
        history_data = select_input_features(history_data, config['froward_features'])
        history_data = history_data.to(config['device'])

        reconstruction_masked_tokens, label_masked_tokens = model(history_data, 0)
        
        # Save the reconstructed and original masked tokens
        reconstruction_rescaled = SCALER_REGISTRY.get(scaler["func"])(reconstruction_masked_tokens.detach().cpu(), **scaler["args"])
        original_rescaled = SCALER_REGISTRY.get(scaler["func"])(label_masked_tokens.detach().cpu(), **scaler["args"])
        
        # Convert to numpy for easier handling
        reconstruction_rescaled = reconstruction_rescaled.numpy()
        original_rescaled = original_rescaled.numpy()
        # print(f"./save_mask/data_{epoch}.npz")
        np.savez(f"./save_mask/data_{str(epoch)}.npz", array1=original_rescaled, array2=reconstruction_rescaled)
        # Save to CSV or any other format for later analysis
        # pd.DataFrame(reconstruction_rescaled).to_csv(f'reconstructed.csv')
        # pd.DataFrame(original_rescaled).to_csv(f'original.csv')
        
def pretrain(config, args):
    print('### start pre-training ... ###')
    adj_mx, _ = load_adj(config['adj_dir'], "doubletransition")

    # adj_mx_ = torch.Tensor(adj_mx[0])

    # config['num_node'] = 207

    scaler = load_pkl(config['preTrain_scaler_dir'])

    # transform =  maskTransforms(config['mask_ratio'])
    preTrain_train_dataset = PretrainingDataset(config['preTrain_dataset_dir'], config['preTrain_dataset_index_dir'],'train', config['device'])
    preTrain_val_dataset = PretrainingDataset(config['preTrain_dataset_dir'], config['preTrain_dataset_index_dir'],'valid', config['device'])
    preTrain_test_dataset = PretrainingDataset(config['preTrain_dataset_dir'], config['preTrain_dataset_index_dir'],'test', config['device'])
    # print(preTrain_train_dataset[0].shape)
    # dd
    # plot_seq(preTrain_train_dataset)
    # dd
    train_data_loader = DataLoader(preTrain_train_dataset, batch_size=config['preTrain_batch_size'],num_workers = 16, shuffle=True, )
    val_data_loader = DataLoader(preTrain_val_dataset, batch_size=config['preTrain_batch_size'], num_workers = 16, shuffle=False)
    test_data_loader = DataLoader(preTrain_test_dataset, batch_size=config['preTrain_batch_size'], num_workers = 16, shuffle=False)

    
    model = pretrain_model(config['num_nodes'], config['dim'], config['topK'], config['adaptive'], config['pretrain_epochs'], config['patch_size'], config['in_channel'], config['embed_dim'], config['num_heads'], config['mlp_ratio'], config['dropout'], config['mask_ratio'], config['encoder_depth'], config['decoder_depth'])
    # if True:
    #     checkpoint_dict = torch.load('./checkpoints/PEMS03/1/PEMS03/checkpoint_12.pt',map_location=torch.device('cpu'))
    #     model.load_state_dict(checkpoint_dict)
    # dd
    # print(args.device)
    device_ids = list(range(torch.cuda.device_count()))
    if device_ids:
        model = DataParallel(model, device_ids=device_ids).to(device_ids[0]) 
    else:
        model = model.to(config['device'])
    
    # model = model.to(config['device'])
    optimizer = optim.Adam(model.parameters(), config['lr'], weight_decay=1.0e-5,eps=1.0e-8)
    if args.lossType == 'mae':
        lossType = masked_mae
    elif args.lossType == 'sce':
        lossType = sce_loss
    for epoch in range(config['pretrain_epochs']):
        print('============ epoch {:d} ============'.format(epoch))
        loss_all = 0.0
        mask_ratio = 0
        for idx, data in enumerate(tqdm(train_data_loader, ncols=100)):
            # start_data_loader = time.time()
            
            # if idx > 0:
            #     break
            future_data, history_data = data
            print("history_data", history_data.shape)

            history_data = select_input_features(history_data, config['froward_features'])
            history_data = history_data.to(config['device'])
            # model_time = time.time()
            print("history_data", history_data.shape)
            
            reconstruction_masked_tokens, label_masked_tokens= model(history_data, epoch)
            # print("Model forward time:", time.time() - model_time)
            # print("reconstruction_masked_tokens shape:", reconstruction_masked_tokens.shape)
            # print("label_masked_tokens shape:", label_masked_tokens.shape)
            # start = time.time()
            loss = metric_forward(lossType, [reconstruction_masked_tokens, label_masked_tokens])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            loss_all += loss.item()
            # print("metric_forward time:", time.time() - start)
            # print("DataLoader time:", time.time() - start_data_loader)
        # model.module.get_mask_ratio()
        loss_all = loss_all / (idx + 1)
        print("preTrain loss: ", loss_all)
        swanlab.log({"preTrain_loss": loss_all, "epoch": epoch})
        
        if args.preTrainVal == 'true':
            preTrain_test_save(preTrain_train_dataset, model, scaler, epoch)
            preTrain_test(val_data_loader, model, scaler, 'val' )
            preTrain_test(test_data_loader, model, scaler, 'test' )
        if config["save_model"]:
            print("Saveing Model ...")
            pre_trained_path = config["model_save_path"] + "checkpoint_" + str(epoch) +".pt"
            torch.save(model.module.state_dict(), pre_trained_path)
        config['pre_trained_path'] = pre_trained_path
        # finetune(config, args)

    # return best_model
    return model

def main(config, args):

    run = swanlab.init(
        project="GPT-GNN",  # 项目名称
        config=config
    )
    if args.preTrain == 'true':
        model = pretrain(config, args)
        model = model.cpu()
        # finetune(config, args)
    else:
        finetune(config, args)
    # if load_mod
    # el:
    #     logging.info("Loading Model ... ")
    #     model.load_state_dict(torch.load("checkpoint/checkpoint.pt"))
    # if save_model:
    #     logging.info("Saveing Model ...")
    #     

def update_config(config, args):

    device = torch.device("cuda:" + str(args.device)) if torch.cuda.is_available() else torch.device("cpu")
    # if args.device is not None: 
    config['device'], args.device = device, device
        
    config['preTrain_batch_size'] = args.preTrain_batch_size
    config['batch_size'] = args.batch_size
    
    
    config['pretrain_epochs'] = args.pretrain_epochs
    config['finetune_epochs'] = args.finetune_epochs
    config['mask_ratio'] = args.mask_ratio
    
    
    return config
def seed_torch(seed=0):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed) # 为了禁止hash随机化，使得实验可复现
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed) # if you are using multi-GPU.
    # torch.backends.cudnn.benchmark = False
    # torch.backends.cudnn.deterministic = True


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='PyTorch Training')
    parser.add_argument('--config', default = './parameters/PEMS03.yaml', type = str, help = 'Path to the YAML config file')
    parser.add_argument('--device', default = 0, type = int, help = 'device')
    parser.add_argument('--preTrain', default = 'true', type = str, help = 'pre-training or not')
    parser.add_argument('--lossType', default = 'mae', type = str, help = 'pre-training loss type and default is mae. {mae, sce}')
    parser.add_argument('--preTrain_batch_size', default = 32, type = int, help = 'pre-training batch size')
    parser.add_argument('--batch_size', default = 32, type = int, help = 'fine-tuning batch size')
    
    parser.add_argument('--pretrain_epochs', default = 100, type = int, help = 'pre-training epochs')
    parser.add_argument('--finetune_epochs', default = 100, type = int, help = 'fine-tuning epochs')
    
    parser.add_argument('--preTrainVal', default = "false", type = str, help = 'pre-training validate or not')
    parser.add_argument('--mask_ratio', default = 0.25, type = float, help = 'mask ratio')
    parser.add_argument('--device_ids', default = 0, type = int, help = 'Number of GPUs available ')
    
    
    

    args = parser.parse_args()
    # 读取配置文件
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    # 更新配置文件中的参数


    config = update_config(config, args)
    # print(args)
    # # 输出更新后的配置
    # torch.manual_seed(0)
    # np.random.seed(0)
    # if torch.cuda.is_available():
    #     torch.cuda.manual_seed_all(0)
    
    seed_torch(seed=0)
    main(config, args)