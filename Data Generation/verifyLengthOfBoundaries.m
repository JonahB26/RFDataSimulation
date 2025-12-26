function boundary_conditions = verifyLengthOfBoundaries(boundary_conditions,lateralresolution,axialresolution)

    % This function takes in the previously defined boundary conditions,
    % and checks if they are the correct length to work as boundary
    % conditions with the YM Image defined in the image, which the function
    % checks the size of by using lateral and axial resolution.

    if lateralresolution ~= 200 && axialresolution ~= 200
        
        %First interpolate for the left/right sides of the BCs
        x = 1:200; %Original length we chose for each BC to have uniformity in the PCA
        newx = linspace(max(x),min(x),axialresolution);
    
        boundary_conditions.right_axial = interp1(x,boundary_conditions.right_axial,newx,"spline");
        boundary_conditions.left_axial = interp1(x,boundary_conditions.left_axial,newx,"spline");        
        
        boundary_conditions.right_lateral = interp1(x,boundary_conditions.right_lateral,newx,"spline");
        boundary_conditions.left_lateral = interp1(x,boundary_conditions.left_lateral,newx,"spline");

        %Next interpolate for the top/bottom of the BCs
        newx = linspace(max(x),min(x),lateralresolution);
    
        boundary_conditions.top_axial = interp1(x,boundary_conditions.top_axial,newx,"spline");
        boundary_conditions.bottom_axial = interp1(x,boundary_conditions.bottom_axial,newx,"spline");
        
        
        boundary_conditions.top_lateral = interp1(x,boundary_conditions.top_lateral,newx,"spline");
        boundary_conditions.bottom_lateral = interp1(x,boundary_conditions.bottom_lateral,newx,"spline");
    
    elseif lateralresolution ~=200 && axialresolution == 200
      
        %Only interpolate for the top/bottom of the BCs
        x = 1:200; %Original length we chose for each BC to have uniformity in the PCA
        newx = linspace(max(x),min(x),lateralresolution);
    
        boundary_conditions.top_axial = interp1(x,boundary_conditions.top_axial,newx,"spline");
        boundary_conditions.bottom_axial = interp1(x,boundary_conditions.bottom_axial,newx,"spline");
    
        boundary_conditions.top_lateral = interp1(x,boundary_conditions.top_lateral,newx,"spline");
        boundary_conditions.bottom_lateral = interp1(x,boundary_conditions.bottom_lateral,newx,"spline");
    
    elseif lateralresolution == 200 && axialresolution ~= 200
      
        %Only interpolate for the left/right of the BCs
        x = 1:200; %Original length we chose for each BC to have uniformity in the PCA
        newx = linspace(max(x),min(x),axialresolution);
    
        boundary_conditions.right_axial = interp1(x,boundary_conditions.right_axial,newx,"spline");
        boundary_conditions.left_axial = interp1(x,boundary_conditions.left_axial,newx,"spline");
    
        boundary_conditions.right_lateral = interp1(x,boundary_conditions.right_lateral,newx,"spline");
        boundary_conditions.left_lateral = interp1(x,boundary_conditions.left_lateral,newx,"spline");
    
    end
end

    
