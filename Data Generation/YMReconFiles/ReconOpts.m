classdef ReconOpts
    %ReconOpts Summary of this class goes here
    % tolerance is a numeric value that determines the convergence property
    %   for FEM.
    % type can be three values: 'combined', 'axial', or 'lateral'. This
    %   determines whether the YM represents the axial YM, lateral YM, or a
    %   weighted average of the two.
    % converged is a logical value. Set convergence to false to initiate
    %   loop.
    % debugFlag is a logical value. When set to true, it displays iteration
    %   number, mean absolute difference, current mean YM (YM), and previous
    %   mean YM (YM_Image).
    % Rw is the ratio of a/b that determines the ratio of axial to lateral
    %   YM when calculating total YM.
    % filter_size is a numeric value that defines the size of the image
    %   filter.
    % disp_type determines whether the displacements are calculated using
    % either AM2D, GLUE, AM2D+Streal, or GLUE+Streal. Values can be either
    % 'am2d', 'glue', 'am2d_s', or 'glue_s'.
    
    properties
        tolerance {mustBeNumeric}
        converged {mustBeNumericOrLogical}
        debugFlag {mustBeNumericOrLogical}
        type 
        Rw {mustBeNumeric}
        filter_size {mustBeNumeric}
        visualize {mustBeNumericOrLogical}
        disp_type 

    end
    
    methods
        function obj = ReconOpts(tolerance,converged,debugFlag,type,Rw,filter_size,visualize,disp_type)
            %UNTITLED Construct an instance of this class
            %   Detailed explanation goes here
            obj.tolerance = tolerance;
            obj.converged = converged;
            obj.debugFlag = debugFlag;
            obj.type = type;
            obj.Rw = Rw;
            obj.filter_size = filter_size;
            obj.visualize = visualize;
            obj.disp_type = disp_type;
        end
    end
end

