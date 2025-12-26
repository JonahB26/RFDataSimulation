function [score, grad] = GetSzz(input, kernel, dx, dy, dz)
    % set kernel    
    lateral_kernel = kernel;
    axial_kernel = kernel';
        
    exx = input(:,:,:,1);
    eyy = input(:,:,:,2);
    exy = input(:,:,:,4);
  
    % equation info
    exxdy2 = conv2(conv2(exx(:,:,1), lateral_kernel, 'valid'), lateral_kernel, 'valid') * (1/(dy^2));
    eyydx2 = conv2(conv2(eyy(:,:,1), axial_kernel, 'valid'), axial_kernel, 'valid') * (1/(dx^2));
    exydxy = conv2(conv2(exy(:,:,1), axial_kernel, 'valid'), lateral_kernel, 'valid') * (1/(dx*dy));
    Szz = exxdy2 + eyydx2 - 2*exydxy;

    % calculate gradient
    dL_dSzz = 2*Szz;
    dS_deyy = (1/(dx^2)) * (convgrad(convgrad(dL_dSzz, axial_kernel), axial_kernel));

    % function output
    score = sum(Szz.^2,'all');
    grad = 0 + dS_deyy;
end