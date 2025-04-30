import torch
import torch.nn as nn
import numpy as np
import scipy.io as scio
from SQNet import SQNet 
import time
from argparse import ArgumentParser
import os
from collections import OrderedDict

parser = ArgumentParser(description='iQSM')

parser.add_argument('-I', '--InputFile', type=str, default='./',
                    help='Input file saved using SaveInput.m')
parser.add_argument('-O', '--OutputDirectory', type=str, default='./',
                    help='output folder for iQSM and iQFM reconstruction')
parser.add_argument('-C', '--CheckpointsDirectory', type=str, default='./',
                    help='checkpoints folder for iQSM and iQFM pretrained networks')

args = parser.parse_args()

InputPath = args.InputFile
OutputPath = args.OutputDirectory
CheckpointsPath = args.CheckpointsDirectory

# 加载均值与标准差
def load_all_mean_std(file_path):
    data = scio.loadmat(file_path)
    all_mean_std = data['all_mean_std']

    mean_std_dict = {}
    for key in ['qsm', 'lfs', 'r2_prime', 'chi_neg', 'chi_pos']:
        if key not in all_mean_std.dtype.names:
            raise KeyError(f"Key '{key}' not found in 'all_mean_std.mat'.")
        mean = all_mean_std[key][0, 0]['mean'][0, 0]
        std = all_mean_std[key][0, 0]['std'][0, 0]
        mean_std_dict[key] = {'mean': float(mean), 'std': float(std)}
    return mean_std_dict

# 标准化和逆标准化函数
def standardize(tensor, mean, std):
    return (tensor - mean) / std

def inverse_standardize(tensor, mean, std):
    return tensor * std + mean

if __name__ == '__main__':
    print('SQ-Net-v1')
    with torch.no_grad():
        print('Network Loading')

        # 加载均值和标准差
        all_mean_std_path = '/Users/uqygao10/deepMRI/iQSM_Plus/PythonCodes/Evaluation/SQNet_series/SQNet_v1/norm/all_mean_std.mat'
        mean_std_dict = load_all_mean_std(all_mean_std_path)

        qsm_mean, qsm_std = mean_std_dict['qsm']['mean'], mean_std_dict['qsm']['std']
        lfs_mean, lfs_std = mean_std_dict['lfs']['mean'], mean_std_dict['lfs']['std']
        r2p_mean, r2p_std = mean_std_dict['r2_prime']['mean'], mean_std_dict['r2_prime']['std']
        chi_neg_mean, chi_neg_std = mean_std_dict['chi_neg']['mean'], mean_std_dict['chi_neg']['std']
        chi_pos_mean, chi_pos_std = mean_std_dict['chi_pos']['mean'], mean_std_dict['chi_pos']['std']


        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        model = SQNet().to(device)

        checkpoint_path = os.path.expanduser(
            CheckpointsPath) + '/SQNet.pth'
        model.load_state_dict(torch.load(
            checkpoint_path, map_location=device))
        model.eval()

        # state_dict_old = torch.load(
        #     checkpoint_path, map_location=device)

        # state_dict_new = OrderedDict()

        # for k, v in state_dict_old.items():
        #     if 'module.EncodeEmbedding' in k:
        #         k = k.replace('module.EncodeEmbedding', 'EncodeEmbedding') 
        #     if 'module.DecodeEmbedding' in k:
        #         k = k.replace('module.DecodeEmbedding', 'DecodeEmbedding') 
        #     state_dict_new[k] = v

        matImage = scio.loadmat(os.path.abspath(os.path.expanduser(InputPath)))
        lfs = matImage['lfs']
        lfs = np.array(lfs)

        qsm = matImage['qsm_init']
        qsm = np.array(qsm)

        r2_prime = matImage['r2_prime']
        r2_prime  = np.array(r2_prime )


        qsm = torch.from_numpy(qsm)
        lfs = torch.from_numpy(lfs)
        r2_prime = torch.from_numpy(r2_prime)


        r2_prime = r2_prime.float().unsqueeze(0)
        lfs = lfs.float().unsqueeze(0)
        qsm = qsm.float().unsqueeze(0)
        
        # normalization
        qsm = standardize(qsm, qsm_mean, qsm_std)
        lfs = standardize(lfs, lfs_mean, lfs_std)
        r2_prime = standardize(r2_prime, r2p_mean, r2p_std)
        
        # inputs = torch.cat([qsm, r2_prime, lfs], dim=0).unsqueeze(0).to(device)
        # print(f"Input size before entering model: {inputs.size()}")

        # 模型预测  
        model.eval()
        time_start = time.time()
        with torch.no_grad():
            chi_pos, chi_neg , _, _, _, _ = model(qsm.unsqueeze(0).to(device), r2_prime.unsqueeze(0).to(device), lfs.unsqueeze(0).to(device))

        time_end = time.time()
        print(time_end - time_start)

        # 输出尺寸
        chi_pos = chi_pos[0]
        chi_neg = chi_neg[0]
        print(f"Output size for chi_pos: {chi_pos.size()}")
        print(f"Output size for chi_neg: {chi_neg.size()}")

        chi_pos = inverse_standardize(chi_pos, chi_pos_mean, chi_pos_std).cpu().numpy()[0] # (256, 256, 256)
        chi_neg = inverse_standardize(chi_neg, chi_neg_mean, chi_neg_std).cpu().numpy()[0] # (256, 256, 256)

        print('Saving results')

        path = os.path.expanduser(OutputPath) + '/SQNet.mat'
        scio.savemat(path, {'chi_neg': chi_neg, 'chi_pos': chi_pos})

        print('end')
