% function rf_frame = kwaveGenerateFrame(medium, source, sensor, arrayObj, varargin)
% %KWAVEGENERATEFRAME Build an RF frame (Nt × n_lines) by sweeping laterally.
% %
% % Name-Value (all optional)
% %   'YLines_m'    : 1×n_lines vector of lateral positions (m)
% %   'TxBuilder'   : @(src, arr, y_line_m) -> src_out
% %   'UseBPF'      : logical (default = arrayObj.USE_BPF or true)
% %   'UseTGC'      : logical (default = arrayObj.USE_TGC or false)
% %   'Parallel'    : logical (default = true)
% %   'PoolType'    : 'threads' (default) or 'processes'
% %   'ShowWaitbar' : logical (default = true)
% 
% % ---------- parse options ----------
% p = inputParser;
% p.addParameter('YLines_m', [], @(v)isnumeric(v) && isvector(v));
% p.addParameter('TxBuilder', [], @(f) isempty(f) || isa(f,'function_handle'));
% use_bpf_default = true;  if isprop(arrayObj,'USE_BPF'), use_bpf_default = arrayObj.USE_BPF; end
% use_tgc_default = false; if isprop(arrayObj,'USE_TGC'), use_tgc_default = arrayObj.USE_TGC; end
% p.addParameter('UseBPF',      use_bpf_default, @islogical);
% p.addParameter('UseTGC',      use_tgc_default, @islogical);
% p.addParameter('Parallel',    true,            @islogical);
% p.addParameter('PoolType',   'threads',        @(s)ischar(s)||isstring(s)); % 'threads'|'processes'
% p.addParameter('ShowWaitbar', true,            @islogical);
% p.parse(varargin{:});
% opt = p.Results;
% 
% % ---------- basics ----------
% if ~isprop(arrayObj,'n_lines')
%   error('arrayObj.n_lines is required.');
% end
% n_lines = arrayObj.n_lines;
% Nt      = arrayObj.Nt;
% rf_frame = zeros(Nt, n_lines, 'single');
% 
% % ---------- lateral positions ----------
% if ~isempty(opt.YLines_m)
%   y_lines = opt.YLines_m(:).';
%   if numel(y_lines) ~= n_lines
%     error('YLines_m must have n_lines = %d entries.', n_lines);
%   end
% elseif isprop(arrayObj,'line_y_m') && ~isempty(arrayObj.line_y_m)
%   y_lines = arrayObj.line_y_m(:).';
%   if numel(y_lines) ~= n_lines
%     error('arrayObj.line_y_m must have n_lines = %d entries.', n_lines);
%   end
% else
%   y0 = arrayObj.elem_centers_m(1);
%   y1 = arrayObj.elem_centers_m(end);
%   y_lines = linspace(y0, y1, n_lines);
% end
% 
% % ---------- reuse baseline source; per-iter builder optional ----------
% src0 = source;
% txb  = opt.TxBuilder;
% 
% % ---------- waitbar via DataQueue ----------
% dq = []; h = []; count = 0;
% havePCT = license('test','Distrib_Computing_Toolbox') && ~isempty(ver('parallel'));
% if opt.ShowWaitbar
%   h = waitbar(0,'Starting RF frame...','Name','k-Wave RF Generation');
%   cwb = onCleanup(@() safeDelete(h)); %#ok<NASGU>
%   if havePCT && exist('parallel.pool.DataQueue','class') == 8
%     dq = parallel.pool.DataQueue;
%     afterEach(dq, @updateWaitbar);
%   end
% end
% 
% % ---------- PRE-DESIGN BPF AS SOS+G (only change) ----------
% bp_sos = []; bp_g = 1;
% if opt.UseBPF
%   f0 = arrayObj.f0; fs = arrayObj.fs;
%   f1 = 0.8*f0; f2 = min(1.4*f0, 0.45*fs);      % guard for realizability
%   if f2 > f1 && f2 < fs/2
%     Wn = [f1 f2] / (fs/2);
%     [z,p,k] = butter(4, Wn, 'bandpass');       % same spec as before
%     [bp_sos, bp_g] = zp2sos(z,p,k);
%   else
%     opt.UseBPF = false;
%   end
% end
% 
% % ---------- pool setup ----------
% usePar = opt.Parallel && havePCT;
% useThreads = strcmpi(opt.PoolType,'threads');
% 
% if usePar
%   % avoid oversubscription by inner libs
%   setenv('OMP_NUM_THREADS','1');
%   setenv('MKL_NUM_THREADS','1');
% 
%   pool = gcp('nocreate');
%   needsNewPool = isempty(pool) ...
%       || (useThreads  && ~isa(pool,'parallel.ThreadPool')) ...
%       || (~useThreads && ~isa(pool,'parallel.ProcessPool'));
%   if needsNewPool
%     if ~isempty(pool), delete(pool); end
%     if useThreads
%       parpool('threads');              % thread-based pool
%     else
%       parpool('local');                % process-based pool (profile 'local')
%     end
%   end
% end
% 
% % broadcast heavy objects once per process worker (not needed for threads)
% useProcess = usePar && ~useThreads;
% if useProcess
%   medC  = parallel.pool.Constant(@() medium);
%   sensC = parallel.pool.Constant(@() sensor);
%   arrC  = parallel.pool.Constant(@() arrayObj);
% else
%   medC = []; sensC = []; arrC = [];
% end
% 
% % ---------- main loop ----------
% if usePar
%   parfor li = 1:n_lines
%     y_line = y_lines(li);
% 
%     % per-line TX update
%     if ~isempty(txb)
%       source_line = txb(src0, arrayObj, y_line);
%     else
%       source_line = src0;
%     end
% 
%     % select objects per pool type
%     if useProcess
%       medObj = medC.Value; sensObj = sensC.Value; arrObj = arrC.Value;
%     else
%       medObj = medium;      sensObj = sensor;      arrObj = arrayObj;
%     end
% 
%     % single-line synthesis (only filter args changed)
%     rf = kwaveGenerateRFSingle( ...
%            medObj, source_line, sensObj, ...
%            'Array',   arrObj, ...
%            'YLine_m', y_line, ...
%            'F0',      arrObj.f0, ...
%            'UseBPF',  opt.UseBPF, ...
%            'BP_SOS',  bp_sos, ...
%            'BP_G',    bp_g, ...
%            'UseTGC',  opt.UseTGC, ...
%            'SolverArgs', {'DataCast','single','PlotSim',false,'PlotLayout',false,'PlotPML',false} ...
%          );
% 
%     rf_frame(:, li) = rf;
%     if ~isempty(dq), send(dq,1); end
%   end
% else
%   for li = 1:n_lines
%     y_line = y_lines(li);
% 
%     if ~isempty(txb)
%       source_line = txb(src0, arrayObj, y_line);
%     else
%       source_line = src0;
%     end
% 
%     rf = kwaveGenerateRFSingle( ...
%            medium, source_line, sensor, ...
%            'Array',   arrayObj, ...
%            'YLine_m', y_line, ...
%            'F0',      arrayObj.f0, ...
%            'UseBPF',  opt.UseBPF, ...
%            'BP_SOS',  bp_sos, ...
%            'BP_G',    bp_g, ...
%            'UseTGC',  opt.UseTGC, ...
%            'SolverArgs', {'DataCast','single','PlotSim',false,'PlotLayout',false,'PlotPML',false} ...
%          );
% 
%     rf_frame(:, li) = rf;
% 
%     if ~isempty(dq)
%       send(dq,1);
%     elseif ~isempty(h) && isvalid(h)
%       count = count + 1;
%       waitbar(count/n_lines, h, sprintf('Line %d / %d', count, n_lines));
%     end
%   end
% end
% 
% % ---------- finalize waitbar ----------
% if ~isempty(h) && isvalid(h)
%   waitbar(1,h,'Done.'); pause(0.05); delete(h);
% end
% 
% % ---------- helpers ----------
%   function updateWaitbar(~)
%     if isempty(h) || ~isvalid(h), return; end
%     count = count + 1;
%     waitbar(count/n_lines, h, sprintf('Line %d / %d', count, n_lines));
%   end
% 
%   function safeDelete(hb)
%     if ~isempty(hb) && isvalid(hb), delete(hb); end
%   end
% end
function rf_frame = kwaveGenerateFrame(medium, source, sensor, base_pulse,arrayObj, varargin)
%KWAVEGENERATEFRAME Build an RF frame (Nt × n_lines) by sweeping laterally.
%
% Name-Value (all optional)
%   'YLines_m'    : 1×n_lines vector of lateral positions (m)
%   'TxBuilder'   : @(src, arr, y_line_m) -> src_out  (not used if TxAperture/TxFocusZ provided)
%   'TxAperture'  : integer (# elements)  [default arrayObj.tx_aperture]
%   'TxFocusZ'    : meters                [default arrayObj.tx_focus_z]
%   'UseBPF'      : logical (default = arrayObj.USE_BPF or true)
%   'UseTGC'      : logical (default = arrayObj.USE_TGC or false)
%   'Parallel'    : logical (default = true)
%   'PoolType'    : 'threads' (default) or 'processes'
%   'ShowWaitbar' : logical (default = true)

% ---------- parse options ----------
p = inputParser;
p.addParameter('YLines_m',     [], @(v)isnumeric(v) && isvector(v));
p.addParameter('TxBuilder',    [], @(f) isempty(f) || isa(f,'function_handle'));
use_bpf_default = true;  if isprop(arrayObj,'USE_BPF'), use_bpf_default = arrayObj.USE_BPF; end
use_tgc_default = false; if isprop(arrayObj,'USE_TGC'), use_tgc_default = arrayObj.USE_TGC; end
p.addParameter('UseBPF',       use_bpf_default, @islogical);
p.addParameter('UseTGC',       use_tgc_default, @islogical);
p.addParameter('Parallel',     true,            @islogical);
p.addParameter('PoolType',    'threads',        @(s)ischar(s)||isstring(s));
p.addParameter('ShowWaitbar',  true,            @islogical);
% NEW: per-line TX parameters (fallback to arrayObj.* if present)
tx_ap_default = []; if isprop(arrayObj,'tx_aperture'), tx_ap_default = arrayObj.tx_aperture; end
tx_fz_default = []; if isprop(arrayObj,'tx_focus_z'),  tx_fz_default = arrayObj.tx_focus_z;  end
p.addParameter('TxAperture',   tx_ap_default,   @(x) isempty(x) || (isscalar(x)&&x>0));
p.addParameter('TxFocusZ',     tx_fz_default,   @(x) isempty(x) || isscalar(x));
p.parse(varargin{:});
opt = p.Results;

% ---------- basics ----------
if ~isprop(arrayObj,'n_lines'), error('arrayObj.n_lines is required.'); end
n_lines = arrayObj.n_lines;
Nt      = arrayObj.Nt;
rf_frame = zeros(Nt, n_lines, 'single');

% ---------- lateral positions ----------
if ~isempty(opt.YLines_m)
  y_lines = opt.YLines_m(:).';
  if numel(y_lines) ~= n_lines, error('YLines_m must have n_lines = %d entries.', n_lines); end
elseif isprop(arrayObj,'line_y_m') && ~isempty(arrayObj.line_y_m)
  y_lines = arrayObj.line_y_m(:).';
  if numel(y_lines) ~= n_lines, error('arrayObj.line_y_m must have n_lines = %d entries.', n_lines); end
else
  y0 = arrayObj.elem_centers_m(1);
  y1 = arrayObj.elem_centers_m(end);
  y_lines = linspace(y0, y1, n_lines);
end

% ---------- reuse baseline source; per-iter builder optional ----------
src0 = source;
txb  = opt.TxBuilder; % will be ignored if TxAperture/TxFocusZ provided

% ---------- waitbar via DataQueue ----------
dq = []; h = []; count = 0;
havePCT = license('test','Distrib_Computing_Toolbox') && ~isempty(ver('parallel'));
if opt.ShowWaitbar
  h = waitbar(0,'Starting RF frame...','Name','k-Wave RF Generation');
  cwb = onCleanup(@() safeDelete(h)); %#ok<NASGU>
  if havePCT && exist('parallel.pool.DataQueue','class') == 8
    dq = parallel.pool.DataQueue;
    afterEach(dq, @updateWaitbar);
  end
end

% ---------- PRE-DESIGN BPF AS SOS+G ----------
bp_sos = []; bp_g = 1;
if opt.UseBPF
  f0 = arrayObj.f0; fs = arrayObj.fs;
  f1 = 0.85*f0; f2 = min(1.3*f0, 0.45*fs);
  if f2 > f1 && f2 < fs/2
    Wn = [f1 f2] / (fs/2);

    if f2 <= f1 || f2 >= fs/2
      opt.UseBPF = false;
      fprintf('[BPF] disabled (f1=%.2f, f2=%.2f, fs/2=%.2f MHz)\n', ...
              f1/1e6, f2/1e6, fs/2/1e6);
    else
      fprintf('[BPF] %.2f–%.2f MHz\n', f1/1e6, f2/1e6);
    end



    [z,p,k] = butter(4, Wn, ...
        'bandpass');    % matches legacy order/band
    [bp_sos, bp_g] = zp2sos(z,p,k);
  else
    opt.UseBPF = false;
  end
end

% % ---------- build the base pulse (exactly like legacy) ----------
% t  = arrayObj.t(:).';
% f0 = arrayObj.f0;
% gauss = exp(-((t - 4/f0)/(1.5/f0)).^2);
% base  = single( sin(2*pi*f0*t) .* gauss );   % 1×Nt
% base = base_pulse;
base = single(base_pulse(:).');              % row
if numel(base) > Nt
    base = base(1:Nt);
elseif numel(base) < Nt
    base = [base, zeros(1, Nt-numel(base), 'like', base)];
end

% ---------- rows_per_elem mapping from source.p_mask (legacy behavior) ----------
srcS  = source.getSourceStruct();
pmask = logical(srcS.p_mask);
[Nx,Ny] = size(pmask);
tx_rows = find(any(pmask,2));
if isempty(tx_rows), error('source.p_mask has no active TX rows.'); end
x_src = tx_rows(1);                         % assume single TX row (as before)
cols  = find(pmask(x_src,:));
if isempty(cols), error('TX row has no active columns.'); end

% group contiguous TX columns → elements
breaks = [1, find(diff(cols) > 1) + 1, numel(cols)+1];
elem_groups = cell(1, numel(breaks)-1);
for i = 1:numel(elem_groups)
    elem_groups{i} = cols(breaks(i):breaks(i+1)-1);
end

num_elems = arrayObj.num_elems;
% if mask-derived group count doesn't match, fall back to nearest-center grouping
if numel(elem_groups) ~= num_elems
    elem_groups = cell(1,num_elems);
    % map each active TX column to nearest array element center (in pixels)
    y_vec = arrayObj.kgrid.y_vec;
    centers_col = round( (arrayObj.elem_centers_m - y_vec(1)) / arrayObj.kgrid.dy ) + 1;
    for e = 1:num_elems
        elem_groups{e} = [];
    end
    for c = cols(:).'
        [~, eIdx] = min(abs(centers_col - c));
        elem_groups{eIdx}(end+1) = c; %#ok<AGROW>
    end
end

src_lin_idx   = find(pmask);
rows_per_elem = cell(1,num_elems);
for e = 1:num_elems
    if isempty(elem_groups{e}), rows_per_elem{e} = []; continue; end
    lidx = sub2ind([Nx,Ny], repmat(x_src,1,numel(elem_groups{e})), elem_groups{e});
    rows_per_elem{e} = find(ismember(src_lin_idx, lidx));
end

% ---------- pool setup ----------
usePar     = opt.Parallel && havePCT;
useThreads = strcmpi(opt.PoolType,'threads');

if usePar
  setenv('OMP_NUM_THREADS','1');  % avoid oversubscription
  setenv('MKL_NUM_THREADS','1');

  pool = gcp('nocreate');
  needsNewPool = isempty(pool) ...
      || (useThreads  && ~isa(pool,'parallel.ThreadPool')) ...
      || (~useThreads && ~isa(pool,'parallel.ProcessPool'));
  if needsNewPool
    if ~isempty(pool), delete(pool); end
    if useThreads
      parpool('threads');
    else
      parpool('local');            % process-based
    end
  end
end

% constants used in TX synthesis
elem_y_m   = arrayObj.elem_centers_m;
c0         = arrayObj.c0;
dt         = arrayObj.dt;
xgrid      = 1:Nt;
% aperture and focus
tx_aperture = opt.TxAperture;
tx_focus_z  = opt.TxFocusZ;
if isempty(tx_aperture), error('Provide TxAperture (or set arrayObj.tx_aperture).'); end
if isempty(tx_focus_z),  error('Provide TxFocusZ (or set arrayObj.tx_focus_z).'); end

% ---------- main loop ----------
if usePar
  parfor li = 1:n_lines
    y_line = y_lines(li);

    % --- per-line TX synthesis (legacy) ---
    % nearest element to line
    [~, cidx] = min(abs(elem_y_m - y_line));
    s = max(1, cidx - floor(tx_aperture/2));
    e = min(num_elems, cidx + floor(tx_aperture/2) - 1);
    tx_inds = s:e;

    % geometric delays to focus
    r_tx = sqrt(tx_focus_z.^2 + (elem_y_m(tx_inds) - y_line).^2);
    tau  = (r_tx - min(r_tx))/c0;
    tx_win = hann(numel(tx_inds)).'.^2;

    % build per-element waveforms by fractional delay of 'base'
    src_elems = zeros(numel(tx_inds), Nt, 'like', base);
    for k = 1:numel(tx_inds)
      t_idx = xgrid - tau(k)/dt;
      src_elems(k,:) = tx_win(k) * interp1(xgrid, base, t_idx, 'linear', 0);
    end

    % write into a per-line source.p via rows_per_elem
    srcS_line = src0.getSourceStruct();
    srcS_line.p = zeros(numel(src_lin_idx), Nt, class(base));
    for eElem = 1:num_elems
      if any(eElem == tx_inds)
        rows = rows_per_elem{eElem};
        if ~isempty(rows)
          w = src_elems(find(tx_inds==eElem,1), :);
          srcS_line.p(rows, :) = repmat(w, numel(rows), 1);
        end
      end
    end
    source_line = src0;        % wrapper object
    source_line.p = srcS_line.p;


    % --- run one line ---
    % rf = kwaveGenerateRFSingle( ...
    %        medium, source_line, sensor, ...
    %        'Array',   arrayObj, ...
    %        'YLine_m', y_line, ...
    %        'F0',      arrayObj.f0, ...
    %        'UseBPF',  opt.UseBPF, ...
    %        'BP_SOS',  bp_sos, ...
    %        'BP_G',    bp_g, ...
    %        'UseTGC',  opt.UseTGC, ...
    %        'BFc0', 1540, ...
    %        'SolverArgs', {'DataCast','single','PlotSim',false,'PlotLayout',false,'PlotPML',false} ...
    %      );
    % kwaveGenerateRFSingle(med0, src, sens, ...
    % 'Array', arr, 'YLine_m', arr.line_y_m(round(end/2)), ...
    % 'F0', arr.f0, 'UseBPF', true, 'UseTGC', false, ...
    % 'RxMode','single');    % <-- single-channel

    rf_frame(:, li) = rf;
    if ~isempty(dq), send(dq,1); end
  end
else

    disp('NOT USING PARALLEL')
    % --- build a stable column→element map from geometry ---
srcS  = source.getSourceStruct();
pmask = logical(srcS.p_mask);
[Nx,Ny] = size(pmask);

% (1) find the single TX row used by p_mask
tx_rows = find(any(pmask,2));  assert(~isempty(tx_rows),'p_mask has no active row');
x_tx = tx_rows(1);
cols_tx = find(pmask(x_tx,:)); assert(~isempty(cols_tx),'TX row has no active columns');

% (2) convert element centers (meters) → grid columns (1-based)
y_vec = arrayObj.kgrid.y_vec;  dy = arrayObj.kgrid.dy;  Ny_check = numel(y_vec);
assert(Ny_check==Ny,'p_mask width != Ny');
centers_col = round( (arrayObj.elem_centers_m - y_vec(1)) / dy ) + 1;
centers_col = max(1, min(Ny, centers_col));   % clamp

% (3) for each active TX column, pick nearest element
elem_of_col = zeros(1,Ny,'uint16');
for c = 1:Ny
    [~, elem_of_col(c)] = min(abs(centers_col - c));
end

% (4) rows_per_elem: which p_mask rows correspond to each element
src_lin_idx   = find(pmask);              % linear indices of p_mask==1
rows_per_elem = cell(1, arrayObj.num_elems);
for c = cols_tx(:).'
    e = elem_of_col(c);
    lidx = sub2ind([Nx,Ny], x_tx, c);
    ridx = find(src_lin_idx==lidx, 1);
    if ~isempty(ridx), rows_per_elem{e}(end+1) = ridx; end
end
assert(sum(cellfun(@numel, rows_per_elem))>0, 'rows_per_elem empty (mapping failed)');


  for li = 1:n_lines
    y_line = y_lines(li);

    % --- per-line TX synthesis (legacy) ---
    [~, cidx] = min(abs(elem_y_m - y_line));
    % s = max(1, cidx - floor(tx_aperture/2));
    % e = min(num_elems, cidx + floor(tx_aperture/2) - 1);
    % tx_inds = s:e;
    
    L = tx_aperture;                      % requested #elements
    s = min(max(1, cidx - floor(L/2)), num_elems - L + 1);
    e = s + L - 1;
    tx_inds = s:e;


    r_tx = sqrt(tx_focus_z.^2 + (elem_y_m(tx_inds) - y_line).^2);
    tau  = (r_tx - min(r_tx))/c0;
    tx_win = hann(numel(tx_inds)).'.^2;

    src_elems = zeros(numel(tx_inds), Nt, 'like', base);
    for k = 1:numel(tx_inds)
      t_idx = xgrid - tau(k)/dt;
      src_elems(k,:) = tx_win(k) * interp1(xgrid, base, t_idx,'linear', 0);
    end

    % --- DEBUG: report transmit spectrum peak of the first active TX waveform ---
dbg_tx = double(src_elems(1,:));
dbg_tx = dbg_tx - mean(dbg_tx);
N2 = 2^nextpow2(numel(dbg_tx));
TXF = abs(fft(dbg_tx, N2)).^2;
[~,ipk] = max(TXF(1:floor(N2/2)));
f_pk_tx = (ipk-1)*(arrayObj.fs/N2);
fprintf('[TX] peak ≈ %.2f MHz (expect %.2f MHz)\n', f_pk_tx/1e6, arrayObj.f0/1e6);


% srcS_line = src0.getSourceStruct();
% srcS_line.p = zeros(numel(src_lin_idx), Nt, 'like', base);
%     % for eElem = 1:num_elems
%     %   if any(eElem == tx_inds)
%     %     rows = rows_per_elem{eElem};
%     %     if ~isempty(rows)
%     %       w = src_elems(find(tx_inds==eElem,1), :);
%     %       srcS_line.p(rows, :) = repmat(w, numel(rows), 1);
%     %     end
%     %   end
%     % end
%     for eElem = tx_inds
%        rows = rows_per_elem{eElem};
%         if ~isempty(rows)
%             k = find(tx_inds==eElem,1,'first');   % which waveform in src_elems
%             srcS_line.p(rows, :) = repmat(src_elems(k,:), numel(rows), 1);
%         end
%     end
%     assert(nnz(srcS_line.p)>0, 'TX mapping wrote zero rows for this line');
% source_line = src0; source_line.p = srcS_line.p;
    srcS_line = src0.getSourceStruct();
srcS_line.p = zeros(numel(src_lin_idx), Nt, 'like', base);

for eElem = tx_inds
    rows = rows_per_elem{eElem};
    if ~isempty(rows)
        k = find(tx_inds==eElem,1,'first');      % waveform index for this element
        srcS_line.p(rows, :) = repmat(src_elems(k,:), numel(rows), 1);
    end
end
assert(nnz(srcS_line.p)>0, 'TX mapping wrote zero rows for this line');

source_line = src0;
source_line.p = srcS_line.p;


% if nnz(srcS_line.p) == 0
%     disp('Empty source line')
%     % Fallback: drive every active source point so the line isn't blank
%     pmask = logical(srcS_line.p_mask);
%     npts  = nnz(pmask);
%     srcS_line.p = repmat(base, npts, 1);
% end


    source_line = src0;
    source_line.p = srcS_line.p;

    fprintf('[TX MAP FINAL] y=%.2f mm  tx_inds=[%d..%d]  nnz(source_line.p)=%d  rows=%d Nt=%d\n', ...
        y_line*1e3, tx_inds(1), tx_inds(end), nnz(source_line.p), size(source_line.p,1), size(source_line.p,2));

    % rf = kwaveGenerateRFSingle( ...
    %        medium, source_line, sensor, ...
    %        'Array',   arrayObj, ...
    %        'YLine_m', y_line, ...
    %        'F0',      arrayObj.f0, ...
    %        'UseBPF',  opt.UseBPF, ...
    %        'BP_SOS',  bp_sos, ...
    %        'BP_G',    bp_g, ...
    %        'UseTGC',  opt.UseTGC, ...
    %        'BFc0', medium.sound_speed(round(end/2), round(end/2)), ...
    %        'SolverArgs', {'DataCast','single','PlotSim',false,'PlotLayout',false,'PlotPML',false} ...
    %      );
    rf = kwaveGenerateRFSingle(medium, source_line, sensor, ...
    'Array', arrayObj, 'YLine_m', arrayObj.line_y_m(round(end/2)), ...
    'F0', arrayObj.f0, 'UseBPF', true, 'UseTGC', false, ...
    'RxMode','single');    % <-- single-channel

    rf_frame(:, li) = rf;

    if ~isempty(dq)
      send(dq,1);
    elseif ~isempty(h) && isvalid(h)
      count = count + 1;
      waitbar(count/n_lines, h, sprintf('Line %d / %d', count, n_lines));
    end
  end
end

% ---------- finalize waitbar ----------
if ~isempty(h) && isvalid(h)
  waitbar(1,h,'Done.'); pause(0.05); delete(h);
end

% ---------- helpers ----------
  function updateWaitbar(~)
    if isempty(h) || ~isvalid(h), return; end
    count = count + 1;
    waitbar(count/n_lines, h, sprintf('Line %d / %d', count, n_lines));
  end

  function safeDelete(hb)
    if ~isempty(hb) && isvalid(hb), delete(hb); end
  end
end
