classdef ImagingGrid
    % Lateral/depth grid for beamforming.
    properties
        Nax   (1,1) {mustBePositive,mustBeInteger} = 2500
        Nlat  (1,1) {mustBePositive,mustBeInteger} = 96
        xL    double   % 1xNlat lateral [m]
        zax   double   % Naxx1 depth   [m]
        zf    double = 20e-3           % Tx focus depth [m]
        Fnum  double = 1.7             % Rx F-number
    end

    methods
        function obj = ImagingGrid(varargin)
            p = inputParser;
            addParameter(p,'Nax',2500);
            addParameter(p,'Nlat',96);
            addParameter(p,'xLFOVmm',[-20 20]); % mm
            addParameter(p,'zmaxmm',30);        % mm
            addParameter(p,'zf',20);            % mm
            addParameter(p,'Fnum',1.7);
            parse(p,varargin{:});
            S = p.Results;

            obj.Nax  = S.Nax;  obj.Nlat = S.Nlat;
            obj.xL   = linspace(S.xLFOVmm(1), S.xLFOVmm(2), obj.Nlat) * 1e-3;
            obj.zax  = linspace(eps, S.zmaxmm*1e-3, obj.Nax).';
            obj.zf   = S.zf * 1e-3;
            obj.Fnum = S.Fnum;
        end
    end
end
