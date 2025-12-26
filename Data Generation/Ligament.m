classdef Ligament
    %LIGAMENT Summary of this class goes here
    %   Detailed explanation goes here
    
    properties
        thickness double
        stiffness double
    end
    
    methods
        function obj = Ligament(thickness,stiffness)
            %LIGAMENT Construct an instance of this class
            %   Detailed explanation goes here
            obj.thickness = thickness;
            obj.stiffness = stiffness;
        end
        

    end
end

