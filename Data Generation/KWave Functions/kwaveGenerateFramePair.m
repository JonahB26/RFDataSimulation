function [Frame1,Frame2] = kwaveGenerateFramePair(medium_pre,medium_post, source, sensor, array,varargin)
%KWAVEGENERATEFRAMEPAIR Summary of this function goes here
%   Detailed explanation goes here
p = inputParser;
p.addParameter('Parallel', false, @islogical);
p.addParameter('Visualize', false, @islogical);
p.addParameter('PoolType', 'threads', @islogical);
p.parse(varargin{:});
opt = p.Results;
%% Generate base pulse
% ---------- build the base pulse (exactly like legacy) ----------
t  = array.t(:).';               % 1×Nt
f0 = array.f0;                   % 5e6
ncyc = 5;
gauss = exp(-((t - (ncyc/f0)*0.6)/(0.35/f0)).^2);
base = single(sin(2*pi*f0*t) .* gauss);

%% First frame
notifyMe('Starting Simulation')
Frame1 = kwaveGenerateFrame(medium_pre, source, sensor, base,array,'Parallel',opt.Parallel,'PoolType', opt.PoolType,'TxAperture', array.tx_aperture, ...
    'TxFocusZ',   array.tx_focus_z);
notifyMe('Done Frame1')
% Frame2 = kwaveGenerateFrame(medium_post, source, sensor, base, array,'Parallel',opt.Parallel,'PoolType', opt.PoolType);
% notifyMe('Done Frame2')
Frame2 = [];
end

