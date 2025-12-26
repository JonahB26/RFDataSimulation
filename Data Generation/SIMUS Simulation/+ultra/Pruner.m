classdef Pruner
    % Field-II/k-Wave style pruning around current scan line.
    properties
        K_pitch double = 3        % half-width ≈ K*pitch
        fallbackCoeff double = 0.08  % if lat span unknown
    end

    methods
        function oc = effectiveCoeff(obj, pitch, lat_span)
            % Convert K*pitch to coeff of total lateral span if span is valid
            if isfinite(lat_span) && lat_span > 0
                oc = max((obj.K_pitch * pitch) / lat_span, obj.fallbackCoeff);
            else
                oc = obj.fallbackCoeff;
            end
        end

        function [xsL, zsL, RCL] = apply(~, xs, zs, RC, xl, oc, zmin, zmax, lat_span)
            if ~(isfinite(lat_span) && lat_span>0)
                lat_span = max(xs) - min(xs);
            end
            start_range = xl - oc * lat_span;
            end_range   = xl + oc * lat_span;
            keep = (xs > start_range) & (xs < end_range) & (zs >= zmin) & (zs <= zmax);
            if ~any(keep), keep = true(size(xs)); end
            xsL = xs(keep); zsL = zs(keep); RCL = RC(keep);
        end
    end
end
