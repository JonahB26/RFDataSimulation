% function [IQ1,IQ2,ENV1,ENV2,Idb1,Idb2,info, sim] = ...
%     ultra_generate_two_frames_with_fea(param, xs, zs, RC, Xg, Zg, Ux, Uz, varargin)
% 
% arr = ultra.LinearArray(param);
% grid= ultra.ImagingGrid(varargin{:});
% arr = arr.trimNtToDepth(grid.zax(end));
% prn = ultra.Pruner();
% 
% sim = ultra.Simulator(arr, grid, prn);
% % opt = struct('WaitBar', false);
% opt = struct('WaitBar', false, 'ProgressHz',2);
% 
% [IQ1,IQ2,ENV1,ENV2,Idb1,Idb2,info] = ultra.Elastography.twoFrames(sim, xs, zs, RC, Xg, Zg, Ux, Uz, opt);
% end
function [IQ1,IQ2,ENV1,ENV2,Idb1,Idb2,info, sim] = ...
    ultra_generate_two_frames_with_fea(param, xs, zs, RC, Xg, Zg, Ux, Uz, varargin)

    arr  = ultra.LinearArray(param);
    grid = ultra.ImagingGrid(varargin{:});
    arr  = arr.trimNtToDepth(grid.zax(end));
    prn  = ultra.Pruner();

    sim  = ultra.Simulator(arr, grid, prn);
    opt  = struct('WaitBar', false); %#ok<NASGU>  % (keep if you want)

    % Use the robust twoFrames. Units 'mm' matches your U arrays.
    [IQ1,IQ2,ENV1,ENV2,Idb1,Idb2,info] = ultra.Elastography.twoFrames( ...
         sim, xs, zs, RC, Xg, Zg, Ux, Uz, struct('Units','mm','Verbose',true));
end
