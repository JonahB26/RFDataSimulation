% This script uses the developed kwave functions to batch generate RF data
% using the kwave simulator. Created by Jonah Boutin on 09/01/2025.
% Initialize kwave and
% clear current variables and command window
kwaveInit();
% unload(clibConfiguration('FEM_Interface'))
%% Initialize array, source, sensor
% Create the arrray. Go into class def to check defaults
array = KWaveLinearArray('n_lines',135,'USE_TGC',true,'USE_BPF',true);

% Create the source.
source = KWaveSource(array);


% Create the sensor
sensor = KWaveSensor(array);
%% Generate medium based on inclusion
% test circle
% Create empty mask
Ms = zeros(256, 256);

% Define circle parameters
center = [130, 130];     % center of the circle (row, col)
radius = 45;             % radius in pixels

% Generate circular mask
[xx, yy] = meshgrid(1:256, 1:256);
circle_mask = (xx - center(2)).^2 + (yy - center(1)).^2 <= radius^2;

% Assign to main mask
Ms(circle_mask) = 1;
Ms = logical(Ms);

% medium = KWaveMedium(array,Ms);

%% FEA
% ligament_thickness = 1.75;
% ligament_stiffness = 300000;
% ligament = Ligament(ligament_thickness,ligament_stiffness);
% 
% label = 'malignant';
% Tumor_YM_middle = 120000;
% [simresult, cooper_mask, ~, ~, ~] = FiniteElementAnalysisBatch( ...
% Ms, label, Tumor_YM_middle, ligament.thickness, ligament.stiffness, true);
load("simresult.mat")
% load("cooper_mask.mat")
cooper_mask = zeros([256 256]);

medium = KWaveMedium( ...
    array, Ms, cooper_mask, simresult, ...
    'z_mask_top_mm',  6, ...        % depth from top of grid to TOP of Ms
    'y_target_mm',    0, ...        % lateral center (0 = probe center)
    'rngSeed',        123, ...      % reproducible speckle
    'corr_in_lambda', 1.0, ...      % ~1 λ correlation
    'Dc_mean',       0.06, ...     % -4% c (hypoechoic)
    'Drho_mean',     0.02, ...     % -2% ρ
    'k_var_c',        0.35, ...
    'k_var_rho',      0.4, ...
    'alpha_inside_scale', 1.3, ...
    'smooth_edge_px', 4,...
    'corr_in_lambda',1);
% 
% save("simresult.mat","simresult")
% save("cooper_mask.mat","cooper_mask")


% Ms = logical(Ms + cooper_mask);

%%
[med_pre, med_post] = medium.getMediumStructs();
figure;subplot(221);imshow(med_pre.density,[]);title('Dens')
subplot(222);imshow(med_pre.sound_speed,[]);title('ss')
subplot(223);imshow(med_pre.alpha_power,[]);title('power')
subplot(224);imshow(med_pre.alpha_coeff,[]);title('alpha')

figure;subplot(221);imshow(med_post.density,[]);title('Dens')
subplot(222);imshow(med_post.sound_speed,[]);title('ss')
subplot(223);imshow(med_post.alpha_power,[]);title('power')
subplot(224);imshow(med_post.alpha_coeff,[]);title('alpha')
% Frame1 = kwaveGenerateFrame(med_pre, source, sensor, array,'Parallel',false,'PoolType', 'threads');

% — Use desired imaging depth, not the current timeline —
zmax  = array.z_max;                       % meters (design range), not depth_m(end)
cmin  = min(med_pre.sound_speed(:));       % slowest region controls travel time
yspan = array.elem_centers_m(end) - array.elem_centers_m(1);

t_rt  = 2*zmax / cmin;                                         % round-trip to zmax
t_lat = (sqrt(zmax.^2 + (yspan/2).^2) - zmax) / cmin;          % max lateral slant
tpulse = 2/array.f0;                                           % ~2 cycles margin
t_need = t_rt + t_lat + tpulse + 2/array.fs;                   % + a couple samples

t_have = array.t(end);                                         % current timeline
fprintf('Need %.2fus, have %.2fus\n', t_need*1e6, t_have*1e6);

%% Generate RF frame pair
start = cputime;
[Frame1,~] = kwaveGenerateFramePair(med_pre, med_post, source, sensor, array,'Parallel',false,'Visualize',true);
clc;
elapsedTime = cputime - start;
fprintf('Elapsed time: %.2f seconds\n', elapsedTime);
% Frame1 = imresize(Frame1,[2500 256],"cubic");
% Frame2 = imresize(Frame2,[2500 256],"cubic");
% Save the generated RF frames for later use
save('Frame1K.mat', 'Frame1');
% save('Frame2K.mat', 'Frame2');


function B = bmode_from_rf(RF, dynRange_dB)
    if nargin<2, dynRange_dB = 50; end
    env = abs(hilbert(RF));                 % envelope (analytic signal)
    % env = env ./ max(env(:) + eps);
    scale = prctile(env(:), 99.5);
    B = 20*log10( env / (scale + eps) + eps );
    % B   = 20*log10(env + eps);              % log compression
    B   = max(B, -dynRange_dB);             % limit
end

B1 = imresize(bmode_from_rf(Frame1, 70),[256 256]);
% B2 = imresize(bmode_from_rf(Frame2, 70),[ 256 256]);

figure; subplot(1,2,1); imagesc(B1); colormap gray; axis image ij; title('B-mode pre');
colormap gray; axis image ij;
caxis([-70 0]);   % lock grayscale
hold on; contour(imresize(Ms,[256 256]), [0.5 0.5], 'r', 'LineWidth', 1);
%%
% ------- QUICK QC ON RAW RF (NO RESIZE) -------
% --------- QC v2: robust for 1 line or many lines ---------
RF = Frame1;                 % Nt × n_lines  (n_lines may be 1)
Nt = size(RF,1);
nL = size(RF,2);

fs = array.fs;
dt = array.dt;
depth_mm = array.depth_m(:)*1e3;

% lateral coordinates for each column we actually have
if isprop(array,'line_y_m') && numel(array.line_y_m)==nL
    y_cols_mm = array.line_y_m(:)'*1e3;     % 1×nL
else
    % fall back to element span
    y_cols_mm = linspace(array.elem_centers_m(1), array.elem_centers_m(end), nL)*1e3;
end

% envelope & quick B-mode (no resize)
ENV = abs(hilbert(double(RF)));                         % Nt×nL
scale = prctile(ENV(:), 99.5);
B = 20*log10(ENV/(scale+eps) + eps);

% ---------- spectrum check on the middle column ----------
mid = ceil(nL/2);
x = double(RF(:,mid));
x = x - mean(x);
N2 = 2^nextpow2(numel(x));
P  = abs(fft(x, N2)).^2;
[~,pk] = max(P(1:floor(N2/2)));
fpk = (pk-1)*(fs/N2);
fprintf('RF spectrum peak = %.2f MHz (expect %.2f)\n', fpk/1e6, array.f0/1e6);

% ---------- depth decay (10–23 mm) on middle line ----------
ix1 = find(depth_mm>=10,1,'first');  ix2 = find(depth_mm<=23,1,'last');
env_mid = ENV(:,mid) / (prctile(ENV(ix1:ix2,mid),99.5)+eps);
dB  = 20*log10(env_mid+eps);
fprintf('Depth decay (10–23 mm): %.1f dB\n', dB(ix2)-dB(ix1));

% ---------- CNR on native grid ----------
% expected ROI (edit to your inclusion)
z0_mm = 13.5;                 % center depth
y0_mm = 0.0;                  % center lateral
R_mm  = 4.5;                  % radius

% Build Z,Y grids matching B's size
[Zmm, Ymm] = ndgrid(depth_mm, y_cols_mm);  % Nt×nL

% circular ROI and a ring
M  = ((Zmm - z0_mm).^2 + (Ymm - y0_mm).^2) <= R_mm^2;
R1 = ((Zmm - z0_mm).^2 + (Ymm - y0_mm).^2) <= (1.6*R_mm)^2 & ~M;

% If we only have 1 line, degrade the masks to axial windows
if nL == 1
    M  = abs(Zmm - z0_mm) <= R_mm;                       % slab around z0
    R1 = abs(Zmm - z0_mm) >  R_mm & abs(Zmm - z0_mm) <= 1.6*R_mm;
end

% Guard: ensure masks are same size as B
assert(isequal(size(M),size(B)) && isequal(size(R1),size(B)), 'Mask size mismatch.');

% Compute CNR
mu_in  = mean(B(M),'omitnan');
mu_out = mean(B(R1),'omitnan');
v_in   = var(B(M), 0,'omitnan');
v_out  = var(B(R1),0,'omitnan');
cnr = (mu_in - mu_out) / sqrt(0.5*(v_in + v_out) + eps);

fprintf('CNR (native grid): %.2f (%.2f dB)\n', cnr, 20*log10(max(cnr,eps)));

%% ---------- QUICK QC: confirm inclusion on native grid (no resize) ----------
% Required: Frame1  -> [Nt x n_lines] beamformed RF (single frame)
%           array   -> your KWaveLinearArray (depth_m, line_y_m, f0, etc.)

% ---------- user knobs (mm) ----------
roi_center_mm = [0,  13.5];   % [lateral_mm, depth_mm]  (center of inclusion)
roi_radius_mm = 4.5;          % radius of inclusion
bg_margin_mm  = 1.0;          % gap between ROI and BG ring
bg_width_mm   = 3.0;          % thickness of the BG ring
mute_shallow_mm = 2.0;        % ignore super shallow region
dynRange_dB = 60;             % for plots only

% ---------- axes (native) ----------
z = array.depth_m(:);                  % Nt x 1 (m)
y = array.line_y_m(:);                 % n_lines x 1 (m)
Nt = numel(z); Ny = numel(y);
[Z,Y] = ndgrid(z*1e3, y*1e3);          % mm grids

% ---------- envelope (no resizing) ----------
env = abs(hilbert(Frame1));            % Nt x Ny
% robust scale for display, but not used for metrics
scale = prctile(env(:), 99.5);
B = 20*log10(env/(scale+eps)+eps); B = max(B, -dynRange_dB);

% ---------- build ROI & BG masks ----------
cy = roi_center_mm(1); cz = roi_center_mm(2);
R0 = roi_radius_mm;
R1 = R0 + bg_margin_mm;                % inner ring radius
R2 = R1 + bg_width_mm;                 % outer ring radius

dist_mm = hypot(Y - cy, Z - cz);
ROI = (dist_mm <= R0);
RING = (dist_mm >= R1) & (dist_mm <= R2);

% don't use shallow region or outside valid depth
ROI(Z < mute_shallow_mm)   = false;
RING(Z < mute_shallow_mm)  = false;

% safety: need enough samples
assert(nnz(ROI)>50 && nnz(RING)>50, 'QC masks too small—check center/radius.');

% ---------- CNR (dB) on envelope (not log space) ----------
mu_in  = mean(env(ROI), 'omitnan');
mu_bg  = mean(env(RING),'omitnan');
var_in = var(env(ROI),  0,'omitnan');
var_bg = var(env(RING), 0,'omitnan');
CNR    = abs(mu_in - mu_bg) / sqrt(0.5*(var_in + var_bg));
CNR_dB = 20*log10(max(CNR, eps));

% ---------- axial "bump" check (mean A-line over ROI laterals) ----------
% collapse laterally only over ROI columns near the center band
cols_roi = any(ROI,1);
aline_roi = mean(env(:, cols_roi), 2);
% reference: same collapse over ring columns
cols_bg  = any(RING,1);
aline_bg = mean(env(:, cols_bg), 2);

% depth window around ROI center for peak search
z_win = (Z(:,1) >= (cz - R0)) & (Z(:,1) <= (cz + R0));
peak_in   = max(aline_roi(z_win));
peak_adj  = max(aline_bg(z_win));
bump_dB   = 20*log10((peak_in+eps)/(peak_adj+eps));

% ---------- spectrum sanity at ROI depth ----------
fs = array.fs; f0 = array.f0;
win_samples = max(64, round(1.5*fs/f0));                  % ~1.5 cycles window
[~,k0] = min(abs(z*1e3 - cz)); k0 = max(k0, ceil(win_samples/2));
seg = aline_roi( (k0-floor(win_samples/2)) : min(Nt, k0+floor(win_samples/2)) );
seg = seg - mean(seg);
Nfft = 8192; F = (0:Nfft-1)*(fs/Nfft)/1e6;
S = abs(fft(seg, Nfft)); S = S./max(S+eps);
[~,pkId] = max(S(1:round(Nfft/2)));
f_peak_MHz = F(pkId);  f_err_pct = 100*(f_peak_MHz*1e6 - f0)/f0;

% ---------- depth decay slope (10→23 mm) ----------
zA = 10; zB = min(23, z(end)*1e3-1);
iA = find(z*1e3 >= zA, 1, 'first');  iB = find(z*1e3 <= zB, 1, 'last');
env_db = 20*log10(aline_roi/(max(aline_roi)+eps)+eps);
decay_dB = env_db(iB) - env_db(iA);

% ---------- compact report ----------
fprintf('\n=== QC (native grid) ===\n');
fprintf('ROI center = [y=%.1f mm, z=%.1f mm], R=%.1f mm\n', cy, cz, R0);
fprintf('CNR = %.2f  (%.2f dB)\n', CNR, CNR_dB);
fprintf('Axial bump inside ROI vs ring = %.2f dB\n', bump_dB);
fprintf('Spectrum peak ≈ %.2f MHz (error %.2f %%)\n', f_peak_MHz, f_err_pct);
fprintf('Depth decay (%.0f→%.0f mm) ≈ %.1f dB\n', zA, zB, decay_dB);

% ---------- quick plots (optional) ----------
figure('Name','QC – B-mode (native)'); 
imagesc(y*1e3, z*1e3, B); axis image ij; colormap gray; colorbar
hold on; theta = linspace(0,2*pi,256);
plot(cy + R0*cos(theta), cz + R0*sin(theta),'r','LineWidth',1.5);      % ROI
plot(cy + R1*cos(theta), cz + R1*sin(theta),'c:');                     % ring
plot(cy + R2*cos(theta), cz + R2*sin(theta),'c:');
xlabel('Lateral (mm)'); ylabel('Depth (mm)'); title('B-mode (no resize)');

figure('Name','QC – A-line profiles');
plot(z*1e3, 20*log10(aline_roi/max(aline_roi)+eps),'b'); hold on
plot(z*1e3, 20*log10(aline_bg /max(aline_bg) +eps),'r');
grid on; xlabel('Depth (mm)'); ylabel('dB');
legend('ROI laterals','Ring laterals'); title('Axial profiles (env, norm)');

figure('Name','QC – Spectrum @ ROI depth');
plot(F, 20*log10(S+eps)); xlim([0 3*f0/1e6]); grid on
xlabel('MHz'); ylabel('dB (norm)'); title('RF spectrum near ROI depth');



% %
% ---------- inputs ----------
% RF = Frame1;                      % Nt×n_lines or Nt×1
% fs = array.fs;  dt = array.dt;
% c0 = median(array.c0);            % or median(med0.sound_speed(:),'omitnan')
% t  = (0:size(RF,1)-1)*dt;
% z  = c0*t/2;                      % depth (m)
% 
% pick a line: center if multiple
% if size(RF,2) > 1
%     li = round(size(RF,2)/2);
% else
%     li = 1;
% end
% rf_line = RF(:,li);
% 
% ---------- 1) envelope A-line (depth) ----------
% env = abs(hilbert(double(rf_line)));
% env = env./(prctile(env,99.5)+eps);      % robust scale
% LdB = 20*log10(env+eps);
% 
% figure; plot(z*1e3, LdB); grid on
% xlabel('Depth (mm)'); ylabel('dB'); title('Single A-line (env)');
% xlim([0, max(z)*1e3]);
% 
% ---------- 2) quick spectrum around f0 ----------
% x = rf_line(round(0.15*end):end);                 % avoid near-field mute
% N = 2^nextpow2(numel(x));
% X = abs(fft(double(x).*hann(numel(x)), N));
% f = (0:N-1)/N*fs;
% figure; plot(f/1e6, 20*log10(X/max(X))); grid on
% xlim([0 fs/2]/1e6); xlabel('MHz'); ylabel('dB'); title('RF spectrum');
% 
% ---------- 3) numbers: mean/std/NaN & depth decay ----------
% fprintf('RF mean=%g  std=%g  min=%g  max=%g  hasNaN=%d\n', ...
%     mean(rf_line), std(rf_line), min(rf_line), max(rf_line), any(isnan(rf_line)));
% 
% decay between ~10 mm and ~25 mm (adjust if your max depth < 25 mm)
% d1 = 10e-3; d2 = min(25e-3, max(z)*0.9);
% i1 = max(1, round(d1/(c0/2)*fs));
% i2 = min(numel(LdB), round(d2/(c0/2)*fs));
% decay_db = mean(LdB(i2-20:i2)) - mean(LdB(i1-20:i1));
% fprintf('Depth decay ~%.1f dB from %.0f to %.0f mm\n', decay_db, d1*1e3, d2*1e3);
% 
% ---------- 4) quick B-mode view (if you have multiple lines) ----------
% if size(RF,2) > 1
%     B = imresize(bmode_from_rf(RF,60), [256 256]);   % your helper
%     figure; imagesc(B); colormap gray; axis image ij; title('B-mode quick check');
% end

% %%
% % Frame2_saved = Frame2;
% % for li = 1:5
% %     [xc,lags] = xcorr(Frame1(:,li), Frame2(:,li), 200, 'coeff');
% %     [~,i] = max(xc);
% %     bulk_lag(li) = lags(i);    % integer drift (samples)
% % 
% %     rf_post_aligned(:,li) = circshift(Frame2(:,li), -bulk_lag(li));
% %     % optional: sub-sample shift with interp1
% % end
% % Frame2 = rf_post_aligned;
% %% NCC Check
% function [lag_samp, rmax] = depth_lag(rf1, rf2, fs, maxlag, wlen, hop)
% % rf1, rf2: [Nt x 1] (one beam line each)
% % returns per-depth integer lag (samples) and the peak corr value
% Nt    = size(rf1,1);
% idxs  = 1:hop:(Nt-wlen+1);
% lag_samp = zeros(numel(idxs),1);
% rmax     = zeros(numel(idxs),1);
% for k = 1:numel(idxs)
%     i = idxs(k);
%     x = rf1(i:i+wlen-1);
%     y = rf2(i:i+wlen-1);
%     [xc,lags] = xcorr(x, y, maxlag, 'coeff');
%     [rmax(k),ii] = max(xc);
%     lag_samp(k) = lags(ii);
% end
% end
% 
% maxlag = 64;      % samples
% wlen   = 256;     % samples (≈ 4.3 µs @ 60 MHz)
% hop    = 64;      % stride
% [lag1, r1] = depth_lag(Frame1(:,1), Frame2(:,1), array.fs, maxlag, wlen, hop);
% [lag2, r2] = depth_lag(Frame1(:,2), Frame2(:,2), array.fs, maxlag, wlen, hop);
% 
% figure; subplot(2,1,1); plot(lag1,'-'); hold on; plot(lag2,'-'); 
% ylabel('lag (samples)'); title('Depth-varying lag per line'); grid on
% subplot(2,1,2); plot(r1); hold on; plot(r2); ylabel('peak NCC'); grid on
% %% NCC on envelopes
% env_pre  = abs(hilbert(Frame1));
% env_post = abs(hilbert(Frame2));
% 
% roi = round(0.5e-3/array.c0*array.fs) : round(25e-3/array.c0*array.fs);  % e.g., 0.5–25 mm
% E1 = env_pre(roi,:); E2 = env_post(roi,:);
% 
% % per-depth NCC heatmap with sliding window:
% win  = 256; hop = 64; maxlag = 0;  % set maxlag=0 to measure pure similarity
% nRows = floor( (numel(roi)-win)/hop ) + 1;
% nCols = size(E1,2);
% NCC = zeros(nRows, nCols);
% for li = 1:nCols
%     r = 1;
%     for i = 1:hop:(numel(roi)-win+1)
%         x = E1(i:i+win-1, li);
%         y = E2(i:i+win-1, li);
%         x = (x - mean(x)); y = (y - mean(y));
%         NCC(r,li) = (x'*y) / (sqrt(x'*x)*sqrt(y'*y) + eps);
%         r = r+1;
%     end
% end
% figure;imagesc(NCC,[0 1]); axis image; colormap parula; colorbar
% %% Generate elastography
% 
% % Frame1 = double(load("Frame1K.mat").Frame1);
% % Frame2 = double(load("Frame2K.mat").Frame2);
% % 
% % % 
% % Frame1 = Frame1./max(Frame1(:));
% % Frame2 = Frame2./max(Frame2(:));
% % if size(Frame1,1) >= size(Frame2,1)
% %     Frame1 = Frame1(1:size(Frame2,1),:);
% % else
% %     Frame2 = Frame2(1:size(Frame1,1),:);
% % end
% % Frame1 = imresize(Frame1, [2500, 256]);
% % Frame2 = imresize(Frame2, [2500, 256]);
% % % 
% % reconstruction = GenerateElastographyImage(Frame1,Frame2,true,Ms,1);
% 
% %%
% % ==== INPUT ====
% % Frame1, Frame2 : [Nt x nLines] (pre/post)
% 
% F0 = Frame1; 
% F1 = Frame2;
% 
% % --- If RF (real), make envelopes; if already complex/envelope, just |.| ---
% if isreal(F0), E0 = abs(hilbert(F0)); else, E0 = abs(F0); end
% if isreal(F1), E1 = abs(hilbert(F1)); else, E1 = abs(F1); end
% 
% % Optional gentle normalization to reduce scale bias (safe for correlation)
% E0 = E0 ./ (median(E0(:))+eps);
% E1 = E1 ./ (median(E1(:))+eps);
% 
% [Nt, nL] = size(E0);
% 
% % ---- Correlation vs depth (across lines at each depth) ----
% corr_vs_z = nan(Nt,1);
% for z = 1:Nt
%   a = double(E0(z,:)); b = double(E1(z,:));
%   if any(a) || any(b)
%     r = corrcoef(a(:), b(:));  corr_vs_z(z) = r(1,2);
%   else
%     corr_vs_z(z) = 1;
%   end
% end
% 
% % ---- Correlation vs lateral (across depth in each line) ----
% corr_vs_y = nan(1,nL);
% for y = 1:nL
%   a = double(E0(:,y)); b = double(E1(:,y));
%   if any(a) || any(b)
%     r = corrcoef(a(:), b(:));  corr_vs_y(y) = r(1,2);
%   else
%     corr_vs_y(y) = 1;
%   end
% end
% 
% % ---- Plots ----
% figure; plot(corr_vs_z,'LineWidth',1.5); grid on; ylim([0 1]);
% xlabel('Depth (samples)'); ylabel('corr(E0,E1)'); title('Correlation vs Depth');
% 
% figure; plot(corr_vs_y,'LineWidth',1.5); grid on; ylim([0 1]);
% xlabel('Lateral line');    ylabel('corr(E0,E1)'); title('Correlation vs Lateral');
% 
% % ---- Quick readouts ----
% [czmin, iz] = min(corr_vs_z);   % depth index of min corr
% [cymin, iy] = min(corr_vs_y);   % line index of min corr
% fprintf('Min corr vs depth = %.3f at depth sample %d\n', czmin, iz);
% fprintf('Min corr vs lateral = %.3f at line  %d\n', cymin, iy);
% 
% % If you have a depth vector (meters) named depth_m matching Nt, label in mm:
% if exist('depth_m','var') && numel(depth_m)==Nt
%   figure(gcf-1); cla; plot(depth_m*1e3, corr_vs_z,'LineWidth',1.5);
%   grid on; ylim([0 1]); xlabel('Depth (mm)'); ylabel('corr'); title('Correlation vs Depth');
%   fprintf('Depth of min corr ≈ %.2f mm\n', depth_m(iz)*1e3);
% end
% 
% %%
% arr = array;
% % ---- inputs you need ----
% elem_y_m = arr.elem_centers_m(:);     % E×1
% c0  = arr.c0;
% dt  = arr.dt;
% z   = arr.depth_m(:).';                   % 1×Nt   (matches your frames)
% Nt  = numel(z);
% 
% % lateral positions used to form the frame:
% if exist('y_lines','var') && numel(y_lines)==arr.n_lines
%   yL_all = y_lines(:).';
% else
%   yL_all = linspace(elem_y_m(1), elem_y_m(end), arr.n_lines);   % fallback
% end
% nL = numel(yL_all);
% 
% % ---- OOB fraction per depth & line (all elements) ----
% oob = zeros(nL, Nt);
% xgrid = 1:Nt;
% 
% for li = 1:nL
%   yL = yL_all(li);
%   for t = 1:Nt
%     zt = z(t);
%     extra_time = (sqrt(zt.^2 + (elem_y_m - yL).^2) - zt) / c0;   % E×1
%     t_idx = t - extra_time/dt;                                  % E×1
%     oob(li, t) = mean(t_idx < 1 | t_idx > Nt);
%   end
% end
% 
% % ---- visualize ----
% figure; 
% imagesc(1:nL, 1:Nt, oob.'); axis xy tight;
% xlabel('Lateral line'); ylabel('Depth (samples)');
% title('RX fractional-delay out-of-range (OOB)'); colorbar; caxis([0 1]);
% 
% % mean & max across laterals vs depth
% mean_oob = mean(oob,1);
% max_oob  = max(oob,[],1);
% figure; 
% plot(mean_oob,'LineWidth',1.5); hold on; plot(max_oob,'--','LineWidth',1.0);
% grid on; ylim([0 1]); xlim([1 Nt]);
% xlabel('Depth (samples)'); ylabel('OOB fraction');
% legend('mean across lines','max across lines');
% title('OOB vs depth (summary)');
% 
% 
% %%
% % ---- inputs you need ----
% arr        = array;                      % your array object
% elem_y_m   = arr.elem_centers_m(:);         % [E×1]
% pitch_m    = arr.elem_pitch_m;                   % element pitch (meters)
% depth_m    = array.depth_m(:).';                  % [1×Nt] depth for each sample
% y_line     = mean(elem_y_m);                % use center scanline (or your y_line)
% 
% % use the same knobs as in your beamformer
% FnumRx   = 1.5;
% ApMin_m  = 0.8e-3;
% 
% % ---- dynamic aperture half-width at each depth (meters) ----
% half_width_t = sqrt( (depth_m./FnumRx).^2 + ApMin_m.^2 );    % [1×Nt]
% 
% % ---- which elements are active at each depth? ----
% % active(e,t) = |elem_y - y_line| <= half_width_t(t)
% active = bsxfun(@le, abs(elem_y_m - y_line), half_width_t);  % [E×Nt]
% 
% % number of active elements at each depth
% nActive = sum(active, 1);                                    % [1×Nt]
% 
% % physical aperture diameter (approx) = nActive * pitch
% aperture_diam_m = max(nActive,1) * pitch_m;                  % avoid division by 0
% 
% % effective F# = depth / aperture_diameter
% Fnum_eff = depth_m ./ aperture_diam_m;                       % [1×Nt]
% 
% % ---- plot ----
% figure; 
% plot(depth_m*1e3, nActive, 'LineWidth', 1.5); grid on;
% xlabel('Depth (mm)'); ylabel('# active RX elements');
% title('Dynamic aperture growth with depth');
% 
% figure; 
% plot(depth_m*1e3, Fnum_eff, 'LineWidth', 1.5); grid on;
% xlabel('Depth (mm)'); ylabel('Effective F-number');
% title('Effective F# vs depth');
% 
% 
% 
% %%
% 
% 
% % Inputs:
% E0 = Frame1;                % [Nt x nL]
% E1 = Frame2;
% 
% [Nt,nL] = size(E0);
% mute  = round(0.05*Nt);
% roi   = (mute+1):round(0.80*Nt);
% maxLag = round(0.01*Nt);
% 
% E1_reg = E1;
% bulkLag = zeros(1,nL);
% corr_before = zeros(1,nL);
% corr_after  = zeros(1,nL);
% 
% t = (1:Nt).';
% for li = 1:nL
%     a = double(E0(roi,li));
%     b = double(E1(roi,li));
%     [r,lags] = xcorr(b, a, maxLag, 'coeff');   % if you lack xcorr, tell me; I'll give a no-toolbox version
%     [~,iMax] = max(r);
%     bulkLag(li) = lags(iMax);
% 
%     t_idx = t - bulkLag(li);
%     E1_reg(:,li) = interp1(t, double(E1(:,li)), t_idx, 'pchip', 0);
% 
%     corr_before(li) = corr1D(a,b);
%     corr_after(li)  = corr1D(double(E0(roi,li)), double(E1_reg(roi,li)));
% end
% 
% fprintf('Median corr before %.3f | after bulk-axial %.3f\n', ...
%         median(corr_before), median(corr_after));
% 
% figure; 
% subplot(2,1,1); plot(corr_before,'-'); hold on; plot(corr_after,'-');
% ylabel('corr(ROI)'); legend('before','after'); title('Per-line corr change');
% 
% subplot(2,1,2); plot(bulkLag,'-'); xlabel('lateral line'); ylabel('best axial lag (samples)');
% title('Estimated bulk axial shift per line');
% 
% 
% 
% E1_reg2 = E1;
% W   = max(64, round(0.03*Nt));    % window
% Hop = max(16, round(W/2));        % hop
% lag_map = zeros(Nt, nL, 'single');
% 
% for li = 1:nL
%     z0 = mute+1;
%     while z0+W-1 <= Nt
%         z1 = z0+W-1;
%         a  = double(E0(z0:z1, li));
%         b  = double(E1(z0:z1, li));
%         [r,lags] = xcorr(b, a, round(W/6), 'coeff');
%         [~,iMax] = max(r);
%         lag_map(z0:z1, li) = lags(iMax);
%         z0 = z0 + Hop;
%     end
% 
%     d = lag_map(:,li);
%     first = find(d~=0,1,'first'); last = find(d~=0,1,'last');
%     if ~isempty(first)
%         d(1:first-1)  = d(first);
%         d(last+1:end) = d(last);
%     end
% 
%     t = (1:Nt).';
%     t_idx = t - d;
%     E1_reg2(:,li) = interp1(t, double(E1(:,li)), t_idx, 'pchip', 0);
% end
% 
% % Corr vs depth (windowed) before/after
% win = 128; step = 32;
% zz = 1:step:(Nt-win+1);
% c_before = zeros(numel(zz),1);
% c_after  = zeros(numel(zz),1);
% for k = 1:numel(zz)
%     z = zz(k):(zz(k)+win-1);
%     a  = E0(z,:); b0 = E1(z,:); b1 = E1_reg2(z,:);
%     c_before(k) = corr1D(a(:), b0(:));
%     c_after(k)  = corr1D(a(:), b1(:));
% end
% 
% figure; plot(zz, c_before,'k-'); hold on; plot(zz, c_after,'r-');
% xlabel('depth (sample)'); ylabel('corr'); legend('before','after');
% title('Depth-averaged corr before/after axial-only sliding compensation');
% 
% 
% 
% % ---- helper (no toolboxes) ----
% function r = corr1D(a,b)
% a = a(:); b = b(:);
% a = a - mean(a); b = b - mean(b);
% den = sqrt(sum(a.^2) * sum(b.^2));
% if den==0, r = 0; else, r = sum(a.*b)/den; end
% end
% 
% 
% %% BMODE
% function B = bmode_from_rf(RF, dynRange_dB)
%     if nargin<2, dynRange_dB = 50; end
%     env = abs(hilbert(RF));                 % envelope (analytic signal)
%     % env = env ./ max(env(:) + eps);
%     scale = prctile(env(:), 99.5);
%     B = 20*log10( env / (scale + eps) + eps );
%     % B   = 20*log10(env + eps);              % log compression
%     B   = max(B, -dynRange_dB);             % limit
% end
% 
% B1 = imresize(bmode_from_rf(Frame1, 70),[256 256]);
% % B2 = imresize(bmode_from_rf(Frame2, 70),[ 256 256]);
% 
% figure; subplot(1,2,1); imagesc(B1); colormap gray; axis image ij; title('B-mode pre');
% colormap gray; axis image ij;
% caxis([-70 0]);   % lock grayscale
% hold on; contour(imresize(Ms,[256 256]), [0.5 0.5], 'r', 'LineWidth', 1);
% 
% % subplot(1,2,2); imagesc(B2); colormap gray; axis image ij; title('B-mode post');
% 
% 
% % % ===== 0) Basics from your objects =====
% % Nx = array.Nx; Ny = array.Ny;
% % dx = array.dx; dy = array.dy;
% % x_src = array.x_src;                   % k-Wave source row index (top reference)
% % z_depth = array.depth_m(:);            % Nt×1 depth for B-mode samples (m)
% % 
% % % scanlines actually used for the frame:
% % if exist('y_lines','var') && numel(y_lines)==array.n_lines
% %     yL = y_lines(:).';                 % meters
% % else
% %     yL = linspace(array.elem_centers_m(1), ...
% %                   array.elem_centers_m(end), array.n_lines);
% % end
% % 
% % % ===== 1) Where did the inclusion end up on the k-Wave grid? =====
% % M  = medium.inc_mask;                  % [Nx×Ny] logical
% % [r,c] = find(M);
% % assert(~isempty(r),'inc_mask is empty');
% % 
% % % bounding box & centroid on GRID (indices) and in METERS
% % r0 = min(r); r1 = max(r); c0 = min(c); c1 = max(c);
% % rc = mean(r); cc = mean(c);
% % 
% % z_top_m    = (r0 - x_src) * dx;
% % z_bot_m    = (r1 - x_src) * dx;
% % z_cent_m   = (rc - x_src) * dx;
% % 
% % y_vec_m    = array.kgrid.y_vec(:).';
% % y_left_m   = y_vec_m(c0);
% % y_right_m  = y_vec_m(c1);
% % y_cent_m   = y_vec_m(round(cc));
% % 
% % fprintf('INC on k-Wave grid:\n');
% % fprintf('  depth top..bot:  %.1f .. %.1f mm   center: %.1f mm\n', ...
% %         z_top_m*1e3, z_bot_m*1e3, z_cent_m*1e3);
% % fprintf('  lateral left..right: %.1f .. %.1f mm   center: %.1f mm\n', ...
% %         y_left_m*1e3, y_right_m*1e3, y_cent_m*1e3);
% % 
% % % quick visual on the GRID (not B-mode)
% % figure; imagesc(y_vec_m*1e3, ((1:Nx)-x_src)*dx*1e3, M); axis image ij
% % xlabel('lateral (mm)'); ylabel('depth (mm)');
% % title('inc\_mask on k-Wave grid');
% % 
% % % ===== 2) Where should that appear on the B-mode grid (Nt×nLines)? =====
% % % map centroid to nearest B-mode depth sample and scanline
% % [~, iz_cent] = min(abs(z_depth - z_cent_m));
% % [~, iy_cent] = min(abs(yL       - y_cent_m));
% % 
% % fprintf('Expected centroid on B-mode: depth idx=%d  line idx=%d (of %d)\n', ...
% %         iz_cent, iy_cent, numel(yL));
% % 
% % % build a full overlay by resampling the MASK to B-mode coordinates
% % % Fmask = griddedInterpolant({ ((1:Nx)-x_src)*dx, y_vec_m }, double(M), 'nearest','nearest');
% % Fmask = griddedInterpolant({ (0:Nx-1)*dx, y_vec }, double(M));
% % [ZQ,YQ] = ndgrid(z_depth, yL);
% % OV = Fmask(ZQ,YQ) > 0.5;          % Nt×nLines logical
% 
% % % ===== 3) Plot overlay on native B-mode (no resizing!) =====
% % B1_native = 20*log10( abs(hilbert(Frame1))./(max(abs(hilbert(Frame1(:))))+eps) + eps );
% % B1_native = max(B1_native, -55);
% % 
% % figure; imagesc(B1_native); colormap gray; axis image ij
% % title('B-mode pre with inclusion overlay (native)');
% % hold on; contour(OV,[0.5 0.5],'r','LineWidth',1); hold off
% % 
% % % sanity: how many distinct lines does the mask span?
% % nLinesHit = nnz(any(OV,1));
% % fprintf('Overlay spans %d of %d scanlines.\n', nLinesHit, size(OV,2));
% 
% %% Power Ma-
% 
% function P = local_power(RF, win_ax, win_lat)
%     if nargin<2, win_ax=round(3*size(RF,1)/256); end  % ~3% of depth
%     if nargin<3, win_lat=7; end
%     kax = ones(win_ax,1)/win_ax; klat = ones(1,win_lat)/win_lat;
%     P = sqrt(conv2(RF.^2, kax*klat, 'same'));         % moving RMS
%     P = 20*log10(P / max(P(:) + eps));
% end
% 
% P1 = local_power(Frame1, 41, 11);
% figure; imagesc(P1); colormap gray; axis image ij; title('Local RF power (pre)');
% 
% %% Sliding Window NCC map
% function C = ncc_map(A,B,win_ax,win_lat)
%     if nargin<3, win_ax=64; end
%     if nargin<4, win_lat=9;  end
%     A = double(A); B = double(B);
%     k = ones(win_ax,win_lat);
%     muA = conv2(A, k, 'same') ./ numel(k);
%     muB = conv2(B, k, 'same') ./ numel(k);
%     sA2 = conv2(A.^2, k, 'same') ./ numel(k) - muA.^2;
%     sB2 = conv2(B.^2, k, 'same') ./ numel(k) - muB.^2;
%     cov = conv2(A.*B, k, 'same') ./ numel(k) - muA.*muB;
%     C = cov ./ sqrt(max(sA2,0).*max(sB2,0) + eps);
% end
% 
% Craw = ncc_map(Frame1, Frame2, 64, 9);
% figure; imagesc(Craw, [0 1]); colormap parula; axis image ij; colorbar;
% title('Pre/Post NCC (raw)');
% 
% 
% %%
% % --- coarse axial shift with parabolic peak around max correlation (small window) ---
% function [d_ax] = axial_shift_phase(A,B,ax_win)
%     if nargin<3, ax_win = 64; end
%     [Nt,Ny] = size(A);
%     d_ax = zeros(Nt,Ny);
%     for y = 1:Ny
%         a = A(:,y); b = B(:,y);
%         aA = hilbert(a); aB = hilbert(b);
%         % local phase difference (unwrap along depth)
%         phi = angle(aB .* conj(aA));
%         d_ax(:,y) = unwrap(phi);             % radians
%     end
%     % convert phase → time lag: Δt = Δφ/(2πf0)
%     % you know f0; use your array f0:
%     f0 = 5e6;  % <- set from your array
%     dt = 1/(4*1540/( (1540/f0)/3 )); %#ok<NASGU> % only if you need dt; phase gives subsample
%     % turn radians into samples directly:
%     samples_per_cycle = (1/f0) / (1/ (4*1540/((1540/f0)/3)));  % if you want units of samples
%     % but for strain you only need relative gradient:
% end
% 
% % Simpler: estimate displacement by maximizing 1D NCC within small axial search
% function d_samp = axial_shift_ncc(A,B,ax_win,ax_search)
%     if nargin<3, ax_win=64; end
%     if nargin<4, ax_search=6; end
%     [Nt,Ny] = size(A);
%     d_samp = zeros(Nt,Ny);
%     w = ones(ax_win,1); w = w/sum(w);
%     for y=1:Ny
%         a = A(:,y); b = B(:,y);
%         num = conv(a.*b, w, 'same'); den = sqrt(conv(a.^2,w,'same').*conv(b.^2,w,'same') + eps);
%         c0  = num./den; best = c0; lag = zeros(size(best));
%         for k = 1:ax_search
%             num = conv(a(1+ k:end).*b(1:end-k), w, 'same'); den = sqrt(conv(a(1+k:end).^2,w,'same').*conv(b(1:end-k).^2,w,'same')+eps);
%             cc = [zeros(k,1); num./den];
%             upd = abs(cc) > abs(best);
%             best(upd) = cc(upd); lag(upd) =  k;
%             num = conv(a(1:end-k).*b(1+k:end), w, 'same'); den = sqrt(conv(a(1:end-k).^2,w,'same').*conv(b(1+k:end).^2,w,'same')+eps);
%             cc = [num./den; zeros(k,1)];
%             upd = abs(cc) > abs(best);
%             best(upd) = cc(upd); lag(upd) = -k;
%         end
%         d_samp(:,y) = lag;
%     end
% end
% 
% % Displacement → axial strain (finite difference in depth)
% d_samp = axial_shift_ncc(Frame1, Frame2, 64, 6);
% strain = diff(d_samp,1,1);          % simple derivative; smooth afterwards
% strain = medfilt2(strain, [9 3]);   % robustify
% 
% figure;
% imagesc(strain, [-0.2 0.2]); axis image ij; colormap gray; colorbar;
% title('Axial strain (a.u.) — stiff inclusion = darker (lower strain)');
% 
% 
% %%
% function L = lateral_coherence(RF, ax_win, lat_half)
%     if nargin<2, ax_win=64; end
%     if nargin<3, lat_half=3; end
%     [Nt,Ny] = size(RF);
%     L = zeros(Nt,Ny);
%     k = ones(ax_win,1)/ax_win;
%     for y=1+lat_half:Ny-lat_half
%         R = RF(:, y-lat_half:y+lat_half);
%         ref = R(:,lat_half+1);
%         num = conv2(ref.*mean(R,2), k, 'same');
%         den = sqrt(conv(ref.^2, k, 'same') .* conv(mean(R.^2,2), k, 'same') + eps);
%         L(:,y) = num ./ den;
%     end
% end
% 
% Lc = lateral_coherence(Frame1, 64, 3);
% figure; imagesc(Lc, [0 1]); axis image ij; colormap parula; colorbar;
% title('Lateral coherence map (pre)');
% 
% 
% %%
% notifyMe('Done Batch Script')
% %% DEBUG
% %% === Two-line timing debug (no code edits needed) ===
% 
% % 0) Make sure you are NOT resampling RF anywhere (no imresize on RF).
% useBPF = false;  useTGC = false;   % keep off for timing debug only
% 
% % 1) Build/initialize everything exactly as you normally do for PRE and POST
% %    (whatever your normal entry point is — KWaveBatchGeneration or your own setup)
% 
% % --- create array / media / sources (your usual code) ---
% % Example placeholders; replace with your normal construction:
% % [array, med_pre, source_pre, med_post, source_post, sensor] = yourSetup();
% 
% % 2) Choose two lateral lines (center + small offset)
% yc   = mean(array.elem_centers_m);
% dy   = 3e-3;                                       % 3 mm offset
% y2   = [yc, yc + dy];
% 
% % 3) Generate PRE 2-line frame (no BPF/TGC; no parallel)
% RFpre2 = kwaveGenerateFrame(med_pre,  source,  sensor, array, ...
%            'YLines_m', y2, 'UseBPF', useBPF, 'UseTGC', useTGC, ...
%            'Parallel', false, 'ShowWaitbar', false, ...
%            'TxAperture', array.tx_aperture, 'TxFocusZ', array.tx_focus_z);
% 
% % 4) Generate POST 2-line frame (same options!)
% RFpost2 = kwaveGenerateFrame(med_post, source, sensor, array, ...
%            'YLines_m', y2, 'UseBPF', useBPF, 'UseTGC', useTGC, ...
%            'Parallel', false, 'ShowWaitbar', false, ...
%            'TxAperture', array.tx_aperture, 'TxFocusZ', array.tx_focus_z);
% 
% % 5) Verify timebase matches exactly
% fprintf('Nt=%d  dt=%.3f ns  t0=%.3f ns\n', array.Nt, array.dt*1e9, array.t(1)*1e9);
% 
% % 6) Quick sanity: are these RFs non-zero?
% fprintf('RMS pre lines:  [%.3g  %.3g]\n', rms(RFpre2(:,1)), rms(RFpre2(:,2)));
% fprintf('RMS post lines: [%.3g  %.3g]\n', rms(RFpost2(:,1)), rms(RFpost2(:,2)));
% 
% % 7) Measure integer bulk drift (±1% search window) for each line
% Nt = size(RFpre2,1); mL = round(0.01*Nt);
% perLineBulk = zeros(1,2);
% for li = 1:2
%     [r,L] = xcorr(RFpost2(:,li), RFpre2(:,li), mL, 'coeff');
%     [~,i] = max(r); perLineBulk(li) = L(i);
% end
% fprintf('Integer bulk drift per line (samples): [%d  %d]\n', perLineBulk(1), perLineBulk(2));
% 
% % 8) Optional: refine to sub-sample using phase slope (per line)
% perLineFrac = zeros(1,2);
% for li = 1:2
%     A = fft(hilbert(RFpre2(:,li))); B = fft(hilbert(RFpost2(:,li)));
%     H = B.*conj(A); ang = unwrap(angle(H));
%     p = polyfit((0:Nt-1).', ang, 1);
%     perLineFrac(li) = -p(1)/(2*pi/Nt);            % fractional samples
% end
% fprintf('Fractional drift per line (samples):   [%.2f  %.2f]\n', perLineFrac(1), perLineFrac(2));
% 
% % 9) One-shot bulk alignment (apply to POST) to zero the drift
% RFpost2_fix = align_bulk_phase(RFpre2, RFpost2);
% 
% % 10) Re-measure (should be ~[0 0])
% perLineBulk2 = zeros(1,2);
% for li = 1:2
%     [r,L] = xcorr(RFpost2_fix(:,li), RFpre2(:,li), mL, 'coeff');
%     [~,i] = max(r); perLineBulk2(li) = L(i);
% end
% fprintf('After align: integer drift per line:   [%d  %d]\n', perLineBulk2(1), perLineBulk2(2));
% 
% % 11) (Optional) Visual check
% figure; 
% subplot(1,2,1); plot(RFpre2(:,1)); hold on; plot(RFpost2(:,1)); title('Line 1: pre vs post'); legend pre post
% subplot(1,2,2); plot(RFpre2(:,1)); hold on; plot(RFpost2_fix(:,1)); title('Line 1: pre vs aligned post'); legend pre aligned
% 
% function E1fix = align_bulk_phase(E0,E1)
% [Nt,nL] = size(E0); W = 2*pi*(0:Nt-1)'/Nt;
% E1fix = zeros(size(E1),'like',E1);
% mL = round(0.01*Nt);
% for li=1:nL
%     a = E0(:,li); b = E1(:,li);
%     [r,lags] = xcorr(b,a,mL,'coeff'); [~,ix]=max(r); d0 = lags(ix);  % integer
%     A = fft(hilbert(a)); B = fft(hilbert(b));
%     H = B.*conj(A); ang = unwrap(angle(H));
%     p = polyfit((0:Nt-1).', ang, 1); dfrac = -p(1)/(2*pi/Nt);        % fractional
%     d = d0 + dfrac;  
%     E1fix(:,li) = real(ifft(fft(b).*exp(-1j*W*d)));
% end
% end
% 
% %%
% function [d_int, d_frac, b_aligned] = drift_and_align_roi(a, b, fs, f0, idx_roi)
% % a,b: Nt×1 RF (pre,post)
% % idx_roi: [i0 i1] sample indices for depth ROI (after first arrival)
% % Returns integer lag, fractional lag (samples), and aligned post trace.
% 
% Nt = numel(a);
% i0 = max(1, idx_roi(1));  i1 = min(Nt, idx_roi(2));
% ae = double(a(i0:i1));  be = double(b(i0:i1));
% N  = numel(ae);
% 
% % Light bandpass *only for estimation* (keeps estimator stable)
% bp = designfilt('bandpassiir','FilterOrder',4, ...
%       'HalfPowerFrequency1',0.6*f0,'HalfPowerFrequency2',1.6*f0, ...
%       'SampleRate',fs);
% ae = filtfilt(bp, ae);  be = filtfilt(bp, be);
% 
% % GCC-PHAT (linear xcorr on windowed signals)
% M = round(0.02*N);              % ±2% window is plenty
% A = fft(ae, 2^nextpow2(2*N-1));
% B = fft(be, 2^nextpow2(2*N-1));
% G = A .* conj(B);  G = G ./ (abs(G) + 1e-12);
% r = real(ifft(G));
% r = fftshift(r);                           % center lag=0
% L = -(numel(r)-1)/2 : (numel(r)-1)/2;     % symmetric lags
% mask = (L>=-M & L<=M);
% r = r(mask);  L = L(mask);
% 
% [~,ix] = max(r);
% d_int = L(ix);
% 
% % Fractional via parabolic fit around peak
% if ix>1 && ix<numel(r)
%     y1 = r(ix-1); y2 = r(ix); y3 = r(ix+1);
%     d_frac = 0.5*(y1 - y3) / (y1 - 2*y2 + y3);
% else
%     d_frac = 0;
% end
% 
% % Apply integer+fractional lag to full-length POST using a phase ramp
% W  = 2*pi*(0:Nt-1)'/Nt;
% Bf = fft(double(b));
% d  = d_int + d_frac;
% b_aligned = real(ifft(Bf .* exp(+1j*W*d)));
% end
% %%
% function idx_roi = pick_depth_roi(a, fs, f0)
% % a: Nt×1 RF (pre)
% a   = double(a(:));
% Nt  = numel(a);
% 
% % --- envelope ---
% env = abs(hilbert(a));
% 
% % --- smoothing window: ~3 cycles at f0, but never <5 samples and odd ---
% win = max(5, round(3 * fs / max(f0, eps)));   % ~3 periods @ f0
% if mod(win,2)==0, win = win + 1; end
% 
% % smooth (fallback if movmean is unavailable)
% try
%     env_s = movmean(env, win);
% catch
%     k = ones(win,1)/win;
%     env_s = conv(env, k, 'same');
% end
% 
% % --- find a strong point not too shallow (skip top 5%) ---
% i_lo = max(1, floor(0.05*Nt));
% i_hi = min(Nt, floor(0.5*Nt));          % look in top half
% [~,i_pk_rel] = max(env_s(i_lo:i_hi));
% i_pk = i_lo + i_pk_rel - 1;
% 
% % --- ROI: start a bit before the peak, span ~12 µs (tunable) ---
% lead  = round(0.5e-6 * fs);             % 0.5 µs
% span  = round(12e-6 * fs);              % 12 µs
% i0    = max(1, i_pk - lead);
% i1    = min(Nt, i0 + span);
% idx_roi = [i0 i1];
% end
% 
% %%
% fs = array.fs; f0 = array.f0;
% 
% for li = 1:2
%     roi = pick_depth_roi(RFpre2(:,li), fs, f0);
%     [di, df, RFpost2_fix(:,li)] = drift_and_align_roi(RFpre2(:,li), RFpost2(:,li), fs, f0, roi);
%     fprintf('Line %d: integer=%d  frac=%.2f  total=%.2f samples\n', li, di, df, di+df);
% end
% 
% % sanity check with plain xcorr after alignment
% M = round(0.01*size(RFpre2,1));
% for li=1:2
%   [r,L] = xcorr(RFpost2_fix(:,li), RFpre2(:,li), M, 'coeff');
%   [~,i] = max(r); perLineAfter(li) = L(i);
% end
% fprintf('After align (integer drift): [%d %d]\n', perLineAfter(1), perLineAfter(2));
