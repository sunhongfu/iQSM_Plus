clc;
clear;

% 输入路径
qsm_folder = '/mnt/c932dbb8-268e-4455-a67b-4748c3fd3509/Chi-sepration/Data/Train/Patch_QSM';
phase_folder = '/mnt/c932dbb8-268e-4455-a67b-4748c3fd3509/Chi-sepration/Data/Train/Patch_Lfs';
r2_prime_folder = '/mnt/c932dbb8-268e-4455-a67b-4748c3fd3509/Chi-sepration/Data/Train/Patch_R2_prime';
r2star_folder = '/mnt/c932dbb8-268e-4455-a67b-4748c3fd3509/Chi-sepration/Data/Train/Patch_a_map';
chi_neg_folder = '/mnt/c932dbb8-268e-4455-a67b-4748c3fd3509/Chi-sepration/Data/Train/Patch_Chi_neg';
chi_pos_folder = '/mnt/c932dbb8-268e-4455-a67b-4748c3fd3509/Chi-sepration/Data/Train/Patch_Chi_pos';

% 计算每个图像类型的均值和标准差
disp('开始计算 QSM 图像的均值和标准差...');
[qsm_mean, qsm_std] = compute_mean_and_std(qsm_folder);
disp('QSM 图像均值和标准差计算完成。');

disp('开始计算 Phase 图像的均值和标准差...');
[phase_mean, phase_std] = compute_mean_and_std(phase_folder);
disp('Phase 图像均值和标准差计算完成。');

disp('开始计算 R2_prime 图像的均值和标准差...');
[r2_prime_mean, r2_prime_std] = compute_mean_and_std(r2_prime_folder);
disp('R2_prime 图像均值和标准差计算完成。');

disp('开始计算 R2star 图像的均值和标准差...');
[r2star_mean, r2star_std] = compute_mean_and_std(r2star_folder);
disp('R2star 图像均值和标准差计算完成。');

disp('开始计算 Chi_neg 图像的均值和标准差...');
[chi_neg_mean, chi_neg_std] = compute_mean_and_std(chi_neg_folder);
disp('Chi_neg 图像均值和标准差计算完成。');

disp('开始计算 Chi_pos 图像的均值和标准差...');
[chi_pos_mean, chi_pos_std] = compute_mean_and_std(chi_pos_folder);
disp('Chi_pos 图像均值和标准差计算完成。');

% 创建一个结构体保存所有均值和标准差
disp('正在整理均值和标准差数据...');
all_mean_std = struct(...
    'qsm', struct('mean', qsm_mean, 'std', qsm_std), ...
    'lfs', struct('mean', phase_mean, 'std', phase_std), ...
    'r2_prime', struct('mean', r2_prime_mean, 'std', r2_prime_std), ...
    'a_map', struct('mean', r2star_mean, 'std', r2star_std), ...
    'chi_neg', struct('mean', chi_neg_mean, 'std', chi_neg_std), ...
    'chi_pos', struct('mean', chi_pos_mean, 'std', chi_pos_std));

% 保存所有均值和标准差到一个 .mat 文件
output_matfile = '/mnt/c932dbb8-268e-4455-a67b-4748c3fd3509/Chi-sepration/Data/norm/all_mean_std.mat';
save(output_matfile, 'all_mean_std');
disp('所有均值和标准差已保存为一个 .mat 文件。');

% 加载图像并计算均值与标准差
function [mean_val, std_val] = compute_mean_and_std(image_folder)
    disp(['正在加载文件夹: ', image_folder]);
    image_list = dir(fullfile(image_folder, '*.nii'));
    image_data = [];
    
    % 遍历文件夹中的所有图像文件
    for i = 1:length(image_list)
        image_path = fullfile(image_folder, image_list(i).name);
        disp(['加载图像文件: ', image_list(i).name]);
        img_data = load_nii(image_path); % 使用NIfTI加载函数
        image_data = cat(4, image_data, img_data.img); % 将图像数据堆叠成一个4D数组
    end
    
    % 计算全局均值和标准差
    disp('正在计算全局均值和标准差...');
    mean_val = mean(image_data(:)); % 将所有数据展平后计算均值
    std_val = std(image_data(:));   % 将所有数据展平后计算标准差
    disp('全局均值和标准差计算完成。');
end
