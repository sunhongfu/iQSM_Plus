function Save_Input_SQNet(lfs, qsm_init, r2_prime, Folder)

% phs_tissue: raw warpped phase images (single echo);
% mask: brain mask from FSL-BET tool; 
% TE: echo time in seconds; for example 20e-3 for 20 milliseconds; 
% B0: main field strength (T)
% erode_voxels: number of voxels for brain edge erosion;  0 for no erosion;
% Folder: saving folder; 

% example usasge: 
% SaveInput(phase, mask, TE, B0, 3, './PhaseInputs/', 1);


if ~ exist(Folder, 'dir')
    mkdir(Folder);
end

save(sprintf('%s/Network_Input.mat', Folder), 'lfs', 'qsm_init', 'r2_prime'); 

end

