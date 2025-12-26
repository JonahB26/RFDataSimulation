classdef KWaveLinearArray
  %KWaveLinearArray  Convenience wrapper for k-Wave 2D linear array setup.
  %
  % Usage:
  %   arr = KWaveLinearArray( ...
  %       'f0',5e6,'c0',1540,'ppw',3, ...
  %       'Nx',256,'Ny',256, ...
  %       'z_max',22e-3,'Nt',2500, ...
  %       'PML_size',12,'data_type','single', ...
  %       'USE_BPF',true,'USE_TGC',false, ...
  %       'num_elems',62,'elem_pitch_m',0.3e-3,'elem_w_m',0.25e-3, ...
  %       'x_src',1,'x_rcv',12,'first_center',2);
  %
  % Key getters (examples):
  %   arr.kgrid                 % kWaveGrid
  %   arr.fs, arr.dt, arr.t     % sampling props
  %   arr.depth_m               % depth axis in meters (c0 * t / 2)
  %   arr.elemCentersPx()       % [1xN] lateral centers (pixel indices)
  %   arr.elemCentersM()        % [1xN] lateral centers (meters)
  %   arr.elementBandPx(i)      % [ystart yend] lateral index span for element i
  %   arr.elementMask(i,'src')  % Nx-by-Ny logical mask for element i (source row)
  %   arr.fullApertureMask('rcv')% mask of entire receive aperture
  %
  % Notes:
  % - Coordinates follow k-Wave 2D convention: size [Nx (x=depth), Ny (y=lateral)].
  % - Elements lie along x = x_src / x = x_rcv rows, spanning elem_w_px laterally.
  % - All inputs have sensible defaults that reproduce your snippet.

  properties (SetAccess = immutable)           % core acoustics & grid
    f0 (1,1) double      = 5e6
    c0 (1,1) double      = 1540
    ppw (1,1) double     = 3
    dx (1,1) double
    dy (1,1) double
    Nx (1,1) double      = 256
    Ny (1,1) double      = 256
    kgrid                % kWaveGrid
    fs (1,1) double
    dt (1,1) double
    z_max (1,1) double   = 22e-3
    t_end (1,1) double
    Nt (1,1) double    
    t   (:,1) double
    depth_m (:,1) double
  end

  properties (SetAccess = immutable, Hidden = true)
  f_cycles       (1,1) double = 5
  time_margin_us (1,1) double = 10
  safety_factor  (1,1) double = 1.30
  c_plan         (1,1) double = NaN    % if NaN, fall back to c0
end

  properties (SetAccess = immutable)           % numerics / paddings
    PML_size (1,1) double = 12
    data_type (1,:) char  = 'single'
    USE_BPF (1,1) logical = true
    USE_TGC (1,1) logical = false
  end

  properties (SetAccess = immutable)           % array geometry
    num_elems (1,1) double      = 62
    elem_pitch_m (1,1) double   = 0.3e-3
    elem_w_m (1,1) double       = 0.25e-3
    pitch_px (1,1) double
    elem_w_px (1,1) double
    first_center (1,1) double   = 2
    elem_centers_px (1,:) double
    elem_centers_m  (1,:) double
    x_src (1,1) double          = 1
    x_rcv (1,1) double          = 2
  end

  properties (SetAccess = immutable)           % scan definition
    tx_aperture (1,1) double    = 32
    tx_focus_z (1,1) double     = 10e-3
    n_lines = 64
    line_centers
    line_y_m

  end

  methods
    function obj = KWaveLinearArray(varargin)
      % Parse name/value inputs
      if mod(nargin,2) ~= 0
        error('Use name/value pairs. See class header for example.');
      end
      S = struct(varargin{:});

      % Overwrite defaults if provided
      flds = fieldnames(S);
      for k = 1:numel(flds)
        if isprop(obj, flds{k})
          obj.(flds{k}) = S.(flds{k});
        else
          error('Unknown parameter: %s', flds{k});
        end
      end

      % Derived spacings
      obj.dx = (obj.c0/obj.f0)/obj.ppw;
      obj.dy = obj.dx;

      % Grid
      obj.kgrid = kWaveGrid(obj.Nx, obj.dx, obj.Ny, obj.dy);

      % Time sampling (your scheme)
      obj.fs    = 4*obj.c0/obj.dx;
      obj.dt    = 1/obj.fs;
      obj.t_end = 2*obj.z_max/obj.c0;
      % Either use provided Nt or recompute from t_end*fs (+ safety margin)
      % if isfield(S,'Nt')
      %   obj.Nt = S.Nt;
      % else
      %   obj.Nt = ceil(obj.t_end*obj.fs) + 300; % matches your commented suggestion
      % end
      % obj.kgrid.setTime(obj.Nt, obj.dt);
      % Nt_new = ceil( (0.5e-4) / obj.dt ); % for ~75 µs
      obj.Nt = ceil(obj.t_end*obj.fs) + 300; % matches your commented suggestion
      fprintf('Nt: %d\n',obj.Nt);
        % obj.Nt = Nt_new;
        obj.kgrid.setTime(obj.Nt, obj.dt);
      obj.elem_centers_px = obj.first_center + (0:obj.num_elems-1) * obj.pitch_px;
    % if isnan(obj.c_plan)
    %     c_ref = obj.c0;
    % else
    %     c_ref = obj.c_plan;
    % end
    % t_rt  = 2*obj.z_max / c_ref;
    % yspan = obj.kgrid.y_vec(obj.elem_centers_px(end)) - obj.kgrid.y_vec(obj.elem_centers_px(1));
    % t_lat = (sqrt(obj.z_max.^2 + (yspan/2).^2) - obj.z_max) / obj.c0;
    % t_need = t_rt + t_lat + 2/obj.f0;            % small pulse margin
    % disp(ceil(t_need/obj.dt) + 4)
    % obj.Nt = ceil(t_need/obj.dt) + 4;            % a couple extra samples
    % obj.kgrid.setTime(obj.Nt, obj.dt);

      obj.t = obj.kgrid.t_array(:);
      obj.depth_m = (0:obj.Nt-1).' * (obj.c0/(2*obj.fs));   % meters, 2-way ToF

        % obj.t  = (0:Nt_new-1)*obj.dt;



      % Discretized array geometry
      obj.pitch_px  = max(1, round(obj.elem_pitch_m / obj.dy));
      obj.elem_w_px = max(1, round(obj.elem_w_m   / obj.dy));

      % Centers (lateral indices start at 1)
       obj.elem_centers_px = obj.first_center + (0:obj.num_elems-1) * obj.pitch_px;

      % Bounds check
      if any(obj.elem_centers_px < 1 | obj.elem_centers_px > obj.Ny)
        error('Element centers fall outside lateral grid (1..Ny). Adjust first_center/pitch or Ny.');
      end

      % Centers in meters (y-axis)
      obj.elem_centers_m = obj.kgrid.y_vec(obj.elem_centers_px);

      % Basic sanity: source/receiver rows inside grid
      if obj.x_src < 1 || obj.x_src > obj.Nx || obj.x_rcv < 1 || obj.x_rcv > obj.Nx
        error('x_src/x_rcv must be between 1 and Nx.');
      end

      % Scan definition
      obj.line_centers = round(linspace(obj.elem_centers_px(6), obj.elem_centers_px(end-5), obj.n_lines));
      obj.line_y_m = obj.kgrid.y_vec(obj.line_centers);
    end
    function kgridStruct = getKGridStruct(obj)
        kgridStruct = obj.kgrid;
    end
  end
end