function [RF, IQIMG, ENV, Idb, info, sim] = ultra_generate_frame(param, xs, zs, RC, varargin)
% Create a simulator and generate one focused frame (your exact math).
arr = ultra.LinearArray(param);
grid= ultra.ImagingGrid(varargin{:});
arr = arr.trimNtToDepth(grid.zax(end));   % trim Nt to depth
prn = ultra.Pruner();

sim = ultra.Simulator(arr, grid, prn);
opt = struct('WaitBar', false, 'ProgressHz',2);

[RF, IQIMG, ENV, Idb, info] = sim.generateFrame(xs, zs, RC, opt);
end
