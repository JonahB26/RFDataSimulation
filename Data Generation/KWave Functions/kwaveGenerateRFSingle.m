function rf = kwaveGenerateRFSingle(medium, source, sensor, varargin)
%KWAVEGENERATERFSINGLE Run k-Wave once and return ONE beamformed RF line.
% New Name-Value (used by caller):
%   'BP_SOS' : numeric SOS matrix for BPF (from butter/zp2sos), default []
%   'BP_G'   : numeric gain scalar for BPF, default 1

% ---- unwrap your wrappers ----
% medium = medium.getmediumStruct();
source = source.getSourceStruct();
sensor = sensor.getSensorStruct();

% ---- parse opts ----
p = inputParser;
p.addParameter('Array', [], @(x) isempty(x) || isa(x,'KWaveLinearArray'));
p.addParameter('KGrid', [], @(x) isempty(x) || isa(x,'kWaveGrid'));
p.addParameter('F0', [], @(x) isempty(x) || isnumeric(x));
p.addParameter('UseBPF', true, @islogical);
p.addParameter('BP_SOS', [], @(x) isempty(x) || isnumeric(x));
p.addParameter('BP_G',   1,  @(x) isnumeric(x) && isscalar(x));
p.addParameter('UseTGC', false, @islogical);
p.addParameter('FnumRx', 1.5, @isnumeric);
p.addParameter('ApMin_m', 0.5e-3, @isnumeric);
p.addParameter('ZStart_m', 4e-3, @isnumeric);
p.addParameter('Apod', @(N) tukeywin(N,0.5)', @(f) isa(f,'function_handle'));
p.addParameter('YLine_m', 0, @isnumeric);
p.addParameter('SolverArgs', {}, @(c) iscell(c));
p.addParameter('BFc0', [], @(x) isempty(x)||isnumeric(x));
p.addParameter('RxMode', 'das', @(s) any(strcmpi(s,{'das','single'})));

p.parse(varargin{:});
opt = p.Results;

% ---- context ----
arr = opt.Array; kgrid = opt.KGrid;
if isempty(kgrid) && ~isempty(arr) && isprop(arr,'kgrid'), kgrid = arr.kgrid; end

% Nt = size(source.p,2);
if ~isempty(arr), Nt = arr.Nt; else, Nt = kgrid.Nt; end
assert(size(source.p,2) == Nt, 'source.p length (%d) ≠ Nt (%d)', size(source.p,2), Nt);

if ~isempty(arr)
  dt = arr.dt; fs = arr.fs; t = arr.t(:).'; depth_m = arr.depth_m(:).'; c0 = arr.c0; data_type = arr.data_type;
  % c0_delay = opt.BFc0; if isempty(c0_delay), c0_delay = arr.c0; end
    if isempty(opt.BFc0)
        disp('USING KWAVE C0 DELAY')
        c0_delay = median(medium.sound_speed(:),'omitnan');  % use what k-Wave used
    else
        disp('NOT USING KWAVE DELAY')
        c0_delay = opt.BFc0;
    end
elseif ~isempty(kgrid)
  dt = kgrid.dt; fs = 1/dt;
  if isprop(kgrid,'t_array') && ~isempty(kgrid.t_array), t = kgrid.t_array(:).';
  else, t = (0:Nt-1)*dt; end
  c0 = median(medium.sound_speed(:),'omitnan');
  depth_m = c0 * t / 2;
  data_type = class(source.p);
  % c0_delay = opt.BFc0; if isempty(c0_delay), c0_delay = c0; end
    if isempty(opt.BFc0)
        disp('USING KWAVE C0 DELAY')
        c0_delay = median(medium.sound_speed(:),'omitnan');  % use what k-Wave used
    else
        disp('NOT USING KWAVE DELAY')
        c0_delay = opt.BFc0;
    end
else
  error('Provide ''Array'' or ''KGrid'' (or have arr/kgrid in caller workspace).');
end

% ---- f0 ----
if isempty(opt.F0), if ~isempty(arr) && isprop(arr,'f0'), f0 = arr.f0; else, f0 = 5e6; end
else, f0 = opt.F0; end

% ---- geometry ----
if ~isempty(arr)
  elem_y_m  = arr.elem_centers_m;    num_elems = arr.num_elems;
else
  if ismatrix(sensor.mask) && size(sensor.mask,1)==2
    elem_y_m  = sensor.mask(2,:);    num_elems = numel(elem_y_m);
  elseif ~isempty(kgrid) && islogical(sensor.mask)
    [xr,yc]=find(sensor.mask); x_rcv=mode(xr); cols=sort(yc(xr==x_rcv)).';
    elem_y_m=kgrid.y_vec(cols); num_elems=numel(elem_y_m);
  else
    error('Cannot infer element positions from sensor.mask without kgrid.');
  end
end

% ---- TGC ----
% alpha_bg   = median(medium.alpha_coeff(:), 'omitnan');
% alpha_np_m = (alpha_bg * (f0/1e6)^(medium.alpha_power)) * 100 * log(10)/20; % nepers/m
% tgc = min(exp(2*alpha_np_m*depth_m), 1.10);   % 1×Nt
% ---- TGC (two-way attenuation; optional geometric term) ----
alpha_bg   = median(medium.alpha_coeff(:), 'omitnan');    % dB/(MHz^ap * cm)
ap         = medium.alpha_power;                          % power-law exponent
f0_mhz     = f0/1e6;

% attenuation in dB per meter at f0
alpha_db_per_m = alpha_bg * (f0_mhz^ap) * 100;            % (dB/m)

g_db = 2 * alpha_db_per_m * arr.depth_m;                      % two-way dB vs depth

% optional geometric compensation (use small gamma if desired)
gamma = 0;                                              % try 0–1
g_db = g_db + 20*gamma*log10((arr.depth_m + 1e-3) / 1e-3);
MAX_TGC_DB = 30;
% cap to avoid overflow, but MUCH higher than 1.10
g_db = min(g_db, MAX_TGC_DB);                                   % cap if you like
tgc  = 10.^(g_db/20);
tgc  = reshape(tgc, 1, Nt);   


% ---- k-Wave run ----
solver_args_default = {'PMLInside',false,'PlotSim',false,'PlotLayout',false,'PlotPML',false};
if ~isempty(arr)
  solver_args_default = [solver_args_default, {'PMLSize',arr.PML_size,'DataCast','gpuArray-single'}];
else
  solver_args_default = [solver_args_default, {'PMLSize',12,'DataCast','gpuArray-single'}];
end
solver_args = [solver_args_default, opt.SolverArgs];

srcS_dbg = source;
fprintf('[DBG] nnz(source.p)=%d\n', nnz(srcS_dbg.p));

sensor_data = kspaceFirstOrder2D(kgrid, medium, source, sensor, solver_args{:});

% ---- shape channels to [nChan × Nt] ----
chan = single(sensor_data.p);
if size(chan,2) == Nt
  % ok
elseif size(chan,1) == Nt && size(chan,2) ~= Nt
  chan = chan.';
elseif numel(chan) == Nt
  chan = reshape(chan, 1, Nt);
else
  error('Unexpected sensor_data.p size: %s', mat2str(size(chan)));
end

% ---- OPTIONAL: single-channel sanity path (no Rx beamforming) ----
if strcmpi(opt.RxMode,'single')
    i_center = round(size(chan,1)/2);
    rf = single(chan(i_center,:).');

    % shallow mute
    mute_mm = 1.0; ramp_mm = 1.0;
    g0 = max(0, round((2*mute_mm/1e3)/c0 * fs));
    Nr = max(1, round((2*ramp_mm/1e3)/c0 * fs));
    w = ones(Nt,1,'single'); if g0>0, w(1:min(g0,Nt)) = 0; end
    if g0+Nr <= Nt, r = (1:Nr)'; w(g0+r) = 0.5 - 0.5*cos(pi*r/Nr); end
    rf = rf .* w;

    % BPF (if requested)
    if opt.UseBPF && ~isempty(opt.BP_SOS)
        rf = single(filtfilt(opt.BP_SOS, opt.BP_G, double(rf)));
    end

    % TGC (if requested)
    if opt.UseTGC
        if ~exist('tgc','var')
            depth_m = c0 * (0:Nt-1)/fs / 2;
            alpha_bg   = median(medium.alpha_coeff(:), 'omitnan');
            ap         = medium.alpha_power;
            f0_mhz     = f0/1e6;
            alpha_db_per_m = alpha_bg * (f0_mhz^ap) * 100;
            g_db = 2 * alpha_db_per_m * depth_m;
            g_db = min(g_db, 30);
            tgc  = 10.^(g_db/20);
        end
        rf = rf .* single(tgc(:));

        % --- DEBUG: received RF spectrum peak ---
        rf_dbg = double(rf(:)).';
        rf_dbg = rf_dbg - mean(rf_dbg);
        N2 = 2^nextpow2(numel(rf_dbg));
        RFF = abs(fft(rf_dbg, N2)).^2;
        [~,ipk] = max(RFF(1:floor(N2/2)));
        f_pk_rx = (ipk-1)*(fs/N2);
        fprintf('[RX] peak ≈ %.2f MHz (expect %.2f MHz)\n', f_pk_rx/1e6, f0/1e6);


    end
    return
end


% ---- shallow mute ----
mute_mm = 1.0; ramp_mm = 1.0;
g0 = max(0, round((2*mute_mm/1e3)/c0 * fs));
Nr = max(1, round((2*ramp_mm/1e3)/c0 * fs));
w = ones(1,Nt,'single'); if g0>0, w(1:min(g0,Nt)) = 0; end
if g0+Nr <= Nt, r = 1:Nr; w(g0+r) = 0.5 - 0.5*cos(pi*r/Nr); end
chan = chan .* w;

% ---- bandpass with numeric SOS+G (thread-safe) ----
if opt.UseBPF && ~isempty(opt.BP_SOS)
  chan = filtfilt(opt.BP_SOS, opt.BP_G, chan.').';   % filter columns
end

% % ---- dynamic receive beamforming ----
% y_line = opt.YLine_m; rx_inds = 1:num_elems; apod = opt.Apod(numel(rx_inds));
% z_eff = max(depth_m, opt.ZStart_m); z_eff = z_eff(:).';
% half_width_t = sqrt( (z_eff./opt.FnumRx).^2 + opt.ApMin_m.^2 );
% 
% bsum  = zeros(1,Nt,'double');
% w_sum = zeros(1,Nt,'double');
% for ii = 1:num_elems
%   yi = elem_y_m(ii);
%   extra_time = (sqrt(z_eff.^2 + (yi - y_line).^2) - z_eff) / c0;
%   t_idx = (1:Nt) - extra_time/dt;
%   ch = double(chan(ii,:));
%   ch_shifted = interp1(1:Nt, ch, t_idx, 'pchip', 0);
%   dlat  = abs(yi - y_line);
%   fmask = double(dlat <= half_width_t);
%   wgt   = apod(ii) .* fmask;
%   bsum  = bsum + wgt .* ch_shifted;
%   w_sum = w_sum + wgt;
% end
% bsum = bsum ./ (w_sum + eps);
% 
% if opt.UseTGC, bsum = bsum .* double(tgc); end
% rf = single(bsum(:));

% ---- dynamic receive beamforming (per-depth apodization) ----
y_line = opt.YLine_m;

% effective depth and half-width of the dynamic aperture at each time sample
z_eff = max(depth_m, opt.ZStart_m);           % 1×Nt
z_eff = z_eff(:).';
half_width_t = sqrt( (z_eff./opt.FnumRx).^2 + opt.ApMin_m.^2 );   % 1×Nt

% active(e,t): element e is inside the aperture at time/depth t
active = abs(elem_y_m(:) - y_line) <= half_width_t;               % [num_elems×Nt]

% per-depth apodization weights over the active elements
w_apod = zeros(num_elems, numel(z_eff), 'double');
for tIdx = 1:numel(z_eff)
  idx = find(active(:, tIdx));                 % active element indices at this depth
  if ~isempty(idx)
    taper = opt.Apod(numel(idx));              % e.g., tukeywin(N,0.5)' from your Apod handle
    w_apod(idx, tIdx) = taper(:).';            % assign only to the active subset
  end
end
w_sum = sum(w_apod, 1);                         % 1×Nt (for normalization)

% delay-and-sum with higher-order fractional delay
bsum  = zeros(1, Nt, 'double');
xgrid = 1:Nt;

% % for ii = 1:num_elems
% %   yi = elem_y_m(ii);
% %   % extra_time = (sqrt(z_eff.^2 + (yi - y_line).^2) - z_eff) / c0;  % 1×Nt
% %   extra_time = (sqrt(z_eff.^2 + (yi - y_line).^2) - z_eff) / c0_delay;
% %   t_idx = xgrid - extra_time/dt;                                  % fractional sample times
% % 
% %   ch = double(chan(ii,:));
% %   ch_shifted = interp1(xgrid, ch, t_idx, 'linear', 'extrap');             % fractional delay
% % 
% % 
% %   bsum = bsum + (w_apod(ii, :) .* ch_shifted);                    % depth-dependent weight
% % end
% 
% 
% for ii = 1:num_elems
%     yi = elem_y_m(ii);
%     extra_time = (sqrt(z_eff.^2 + (yi - y_line).^2) - z_eff) / c0_delay; % 1×Nt
%     s_vec      = extra_time ./ dt;                                       % samples
% 
%     ch         = double(chan(ii,:));                                     % 1×Nt
%     ch_shifted = fracdelay_lagrange3_vec(ch, s_vec);                     % 1×Nt
% 
%     bsum = bsum + (w_apod(ii, :) .* ch_shifted);
% end
% 
% bsum = bsum ./ (w_sum + eps);                                      % normalize aperture gain

% Geometric receive delays (use BFc0 if provided)
% extra_time_all = zeros(num_elems, numel(z_eff));
% for ii = 1:num_elems
%     yi = elem_y_m(ii);
%     extra_time_all(ii,:) = (sqrt(z_eff.^2 + (yi - y_line).^2) - z_eff) / c0_delay; % 1×Nt
% end
% 
% % *** NEW: compute and subtract the common (aperture-avg) delay per depth ***
% common_delay = sum(w_apod .* extra_time_all, 1) ./ (w_sum + eps);   % 1×Nt
% 
% bsum  = zeros(1, Nt, 'double');
% xgrid = 1:Nt;
% 
% for ii = 1:num_elems
%     % ch = double(chan(ii,:));
%     % % per-depth fractional delay for this element AFTER removing the common delay
%     % s_vec = (extra_time_all(ii,:) - common_delay) / dt;  % in samples, 1×Nt
%     % 
%     % % robust fractional delay (linear is fine; pchip/makima is even nicer)
%     % t_idx = xgrid - s_vec;
%     % ch_shifted = interp1(xgrid, ch, t_idx, 'pchip', 0);
% 
%     extra_time = (sqrt(z_eff.^2 + (yi - y_line).^2) - z_eff) / c0_delay; % 1×Nt
%     s_vec      = extra_time ./ dt;                                       % samples
% 
%     ch         = double(chan(ii,:));                                     % 1×Nt
%     ch_shifted = fracdelay_lagrange3_vec(ch, s_vec);            
% 
%     bsum = bsum + (w_apod(ii, :) .* ch_shifted);
% end
% 
% bsum = bsum ./ (w_sum + eps);

extra_time_all = zeros(num_elems, numel(z_eff));
for ii = 1:num_elems
    yi = elem_y_m(ii);
    extra_time_all(ii,:) = (sqrt(z_eff.^2 + (yi - y_line).^2) - z_eff) / c0_delay;
end

% Optionally remove the common delay so absolute time is the same per depth
common_delay = (w_apod .* extra_time_all);
common_delay = sum(common_delay,1) ./ (w_sum + eps);   % 1×Nt

bsum = zeros(1,Nt,'double');
xgrid = 1:Nt;
for ii = 1:num_elems
    ch = double(chan(ii,:));
    s_vec = (extra_time_all(ii,:) - common_delay) / dt;   % samples
    t_idx = xgrid - s_vec;
    ch_shifted = fracdelay_lagrange3_vec(ch,s_vec);
    bsum = bsum + (w_apod(ii,:) .* ch_shifted);
end
bsum = bsum ./ (w_sum + eps);

if opt.UseTGC, bsum = bsum .* double(tgc); end
rf = single(bsum.');


function y = fracdelay_lagrange3_vec(x, s)
% Lagrange(3) fractional delay (time-varying).
% x : 1×Nt or Nt×1 real (RF)
% s : scalar or 1×Nt fractional delays in *samples* (positive s delays x)
%
% Implements y(k) ≈ x(k - s(k)).

  % -- normalize shapes --
  x  = x(:).';                    % row
  Nt = numel(x);
  if isscalar(s), s = repmat(s, 1, Nt); else, s = s(:).'; end

  % -- integer / fractional parts --
  n  = floor(s);                  % integer shift
  mu = s - n;                     % 0..1 for "nice" cases, but can be outside

  % -- 2-sample symmetric padding (end-point hold) --
  xpad = [x(1) x(1) x x(end) x(end)];           % length Nt+4

  % -- base index in padded coords for each k (see comment above) --
  % We approximate x(k - s) with the 4-point Lagrange kernel centered at
  % i = k - n + 2 (padded space). We must CLIP i so [i-1..i+2] is valid.
  k = 1:Nt;
  i = k - n + 2;                               % 1×Nt, integer
  i = round(i);                                % guard against accidental float
  i = max(2, min(Nt+2, i));                    % clip so i-1>=1 and i+2<=Nt+4

  % -- Lagrange basis coefficients (cubic, order-3) --
  m  = mu;                                     % 1×Nt
  c0 = (-1/6).*m.*(m-1).*(m-2);
  c1 =  (1/2).*(m+1).*(m-1).*(m-2);
  c2 = (-1/2).*(m+1).*m.*(m-2);
  c3 =  (1/6).*(m+1).*m.*(m-1);

  % -- gather and combine (vectorized) --
  y = c0.*xpad(i-1) + c1.*xpad(i) + c2.*xpad(i+1) + c3.*xpad(i+2);
  y = reshape(y, 1, Nt);                        % row out
end



end
