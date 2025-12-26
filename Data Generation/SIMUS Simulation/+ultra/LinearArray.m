classdef LinearArray
    properties
        % keep originals
        baseParam struct = struct()     % << stores the original param (all fields)

        % key fields we sometimes override
        c0 double = 1540
        fs double = 40e6
        fc double = 7.5e6
        Nelements (1,1) double {mustBeInteger,mustBePositive} = 1
        pitch double = 0.0003
        xe double = 0
        Nt double = 1024
    end

    methods
        function obj = LinearArray(param)
            arguments; param struct; end
            obj.baseParam = param;                               % << keep everything

            assert(isfield(param,'Nelements') && param.Nelements>0, 'param.Nelements must be > 0');
            obj.c0       = param.c0;
            obj.fs       = param.fs;
            obj.fc       = param.fc;
            obj.Nelements= double(param.Nelements);
            obj.pitch    = param.pitch;
            obj.xe       = ((0:obj.Nelements-1) - (obj.Nelements-1)/2) * obj.pitch;

            if isfield(param,'Nt') && ~isempty(param.Nt)
                obj.Nt = param.Nt;
            end
        end

        function obj = trimNtToDepth(obj, zmax)
            tmax = 2*zmax/obj.c0;
            obj.Nt = max(ceil(tmax*obj.fs)+16, 512);
        end

        function s = toParamStruct(obj)
            % Start from the original struct so we keep width/kerf/whatever else
            s = obj.baseParam;

            % Overwrite critical fields that we manage
            s.c0        = obj.c0;
            s.fs        = obj.fs;
            s.fc        = obj.fc;
            s.Nelements = obj.Nelements;
            s.pitch     = obj.pitch;
            s.Nt        = obj.Nt;

            % Ensure element geometry is present for SIMUS
            hasWidth = isfield(s,'width') && ~isempty(s.width);
            hasKerf  = isfield(s,'kerf')  && ~isempty(s.kerf);
            if ~(hasWidth || hasKerf)
                % fallback: no gaps → kerf=0, width=pitch
                s.kerf  = 0;
                s.width = s.pitch;
            elseif hasWidth && ~hasKerf
                s.kerf  = max(s.pitch - s.width, 0);
            elseif hasKerf && ~hasWidth
                s.width = max(s.pitch - s.kerf, 0);
            end
        end
    end
end
