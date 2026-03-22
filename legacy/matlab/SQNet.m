function [chi_pos, chi_neg, chi_tot] = SQNet(lfs, qsm_init, r2_prime, output_dir)
%UNTITLED7 Summary of this function goes here
%   Detailed explanation goes here


if ~exist('output_dir','var') || isempty(output_dir)

    output_dir = pwd; 
end

if ~exist('lfs','var') || isempty(lfs)

    cprintf('*red', 'local field data input is missing! \n');
    cprintf('-[0, 128, 19]',['The phase input data should be a 3-D (single-echo data, e.g., a 256x256x128 volume) \n ' ...
        'or 4-D (multi-echo data: 3D image * echo_num, e.g., 256x256x128x8) volume \n'])
    cprintf('-[0, 128, 19]','The users are suggest to use our iQFM.m function to obtain local field data. \n')
    cprintf('*red', 'Key Parameter Missing, Ends with Error! \n');
    error(' ');
end

if ~exist('qsm_init','var') || isempty(qsm_init)

    cprintf('*red', 'initial QSM data is missing! \n');
    cprintf('-[0, 128, 19]','The QSM data should be in the same size of local field data. \n')
    cprintf('-[0, 128, 19]','The users are suggest to use our iQSM_plus.m function to obtain initial QSM data. \n')
    cprintf('*red', 'Key Parameter Missing, Ends with Error! \n');
    error(' ');
end


if ~exist('r2_prime','var') || isempty(r2_prime)

    cprintf('*red', 'R2 prime is missing! \n');
    cprintf('-[0, 128, 19]','The R2 prime data should be in the same size of local field data. \n')
    cprintf('-[0, 128, 19]','advanced functions . \n')
    cprintf('*red', 'Key Parameter Missing, Ends with Error! \n');
    error(' ');
end



% try to automatically locate where the 'SQNet' folder is downloaded and assign to 'SQNet_dir'
[SQNet_dir, ~, ~] = fileparts(which('SQNet.m'));
% try to automatically locate where the 'deepMRI' repository is downloaded and assign to 'deepMRI_dir'
deepMRI_dir = fileparts(SQNet_dir);

% add MATLAB paths of deepMRI repository
% add necessary utility function for saving data and echo-fitting;
% add NIFTI saving and loading functions;
addpath(genpath(deepMRI_dir));

%% Set checkpoint versions and location
CheckPoints_folder = [SQNet_dir, '/PythonCodes/Evaluation/checkpoints'];
PyFolder = [SQNet_dir, '/PythonCodes/Evaluation/SQNet_series'];
KeyWord = 'SQNet_v1';

checkpoints  = fullfile(CheckPoints_folder, KeyWord);
InferencePath = fullfile(PyFolder, KeyWord, 'Inference_SQNet.py');



cprintf('*[0, 0, 0]', 'Saving all data as NetworkInput.mat for Pytorch Recon! \n');


[lfs, pos] = ZeroPadding(lfs, 16);
[qsm_init, pos] = ZeroPadding(qsm_init, 16);
[r2_prime, pos] = ZeroPadding(r2_prime, 16);


Save_Input_SQNet(lfs, qsm_init, r2_prime, output_dir);

cprintf('*[0, 0, 0]', 'Network Input File generated successfully! \n');

cprintf('*[0, 0, 0]', 'Pytorch Reconstruction Starts! \n');

% Call Python script to conduct the reconstruction; use python API to run iQSM on the demo data
PythonRecon(InferencePath, [output_dir,filesep,'Network_Input.mat'], output_dir, checkpoints);

cprintf('*[0, 0, 0]', 'Reconstruction Ends! \n');

%% load reconstruction data and save as NIFTI
load([output_dir,'/SQNet.mat']);
% pred_chi = pred_lfs;

chi_pos = squeeze(chi_pos);
chi_neg = squeeze(chi_neg);

chi_neg = ZeroRemoving(chi_neg, pos);
chi_pos = ZeroRemoving(chi_pos, pos);

chi_tot = chi_pos - chi_neg; 

nii = make_nii(chi_pos, [1 1 1]);
save_nii(nii, [output_dir,'/chi_pos.nii']);


nii = make_nii(chi_neg, [1 1 1]);
save_nii(nii, [output_dir,'/chi_neg.nii']);

nii = make_nii(chi_pos - chi_neg, [1 1 1]);
save_nii(nii, [output_dir,'/chi_tot.nii']);

delete([output_dir,'/Network_Input.mat']);
delete([output_dir,'/SQNet.mat']);


end








