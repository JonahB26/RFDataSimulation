function [score, grad] = GetSxx(input, kernel, dx, dy, dz)

    % set kernel
    lateral_kernel = kernel;
    axial_kernel = kernel';
        
    eyy = input(:,:,:,2);
    ezz = input(:,:,:,3);
    eyz = input(:,:,:,5);

    % equation info
    eyydz2 = ((eyy(:,:,3) - eyy(:,:,2)) - (eyy(:,:,2) - eyy(:,:,1))) * (1/(dz^2));
    ezzdy2 = conv2(conv2(ezz(:,:,1), lateral_kernel, 'valid'), lateral_kernel, 'valid') * (1/(dy^2));
    eyzdyz = conv2((eyz(:,:,2) - eyz(:,:,1)), lateral_kernel, 'valid') * (1/(dy*dz));
    Sxx = eyydz2(3:end-2, 3:end-2) + ezzdy2 - 2*eyzdyz(2:end-1,2:end-1);
    
    % calculate gradient
    dL_Sxx = 2*Sxx;
    dL_eyy = (1/(dz^2)) * 2*Sxx;

    % function output
    score = sum(Sxx.^2,"all");
    grad = padarray(dL_eyy, [2 2], 0, "both");

end