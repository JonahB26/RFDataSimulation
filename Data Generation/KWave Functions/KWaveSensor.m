classdef KWaveSensor
    %KWaveSensor  Wrapper for k-Wave sensor definition (coordinate format)
    %
    %   sensor = KWaveSensor(arrayObj)
    %
    % Creates:
    %   sensor.mask   -> 2 x num_elems coordinates [x; y] in meters
    %   sensor.record -> {'p'} by default

    properties
        mask    % 2 x num_elems, coordinates in meters
        record  % cell array of fields to record
    end
    
    methods
        function obj = KWaveSensor(arrayObj)
            % Build coordinate mask in meters
            x_coords = repmat(arrayObj.kgrid.x_vec(arrayObj.x_rcv), 1, arrayObj.num_elems);
            y_coords = arrayObj.kgrid.y_vec(arrayObj.elem_centers_px).';
            obj.mask = [x_coords ; y_coords];
            
            % Default: record acoustic pressure
            obj.record = {'p'};
        end
        function sensorStruct = getSensorStruct(obj)
            sensorStruct.mask = obj.mask;
            sensorStruct.record = obj.record;
        end
    end
end
