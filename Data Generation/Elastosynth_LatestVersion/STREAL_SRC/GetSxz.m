function [score, grad] = GetSxz(input, kernel, dx, dy, dz)

    % set kernel    
    lateral_kernel = kernel;
    axial_kernel = kernel';

    eyy = input(:,:,:,2);
    exy = input(:,:,:,4);
    eyz = input(:,:,:,5);
    exz = input(:,:,:,6);

    % equation info
    eyydxz = conv2((eyy(:,:,2) - eyy(:,:,1)), axial_kernel, 'same') * (1/(dx*dz));
    eyzdxy = conv2(conv2(eyz(:,:,1), axial_kernel, 'same'), lateral_kernel, 'same') * (1/(dx*dy));
    exzdy2 = conv2(conv2(exz(:,:,1), lateral_kernel, 'same'), lateral_kernel, 'same') * (1/(dy^2));
    exydyz = conv2((exy(:,:,2) - exy(:,:,1)), lateral_kernel, 'same') * (1/(dy*dz));
    Sxz = -eyydxz + eyzdxy - exzdy2 + exydyz;
 
    % calculate gradient
    dL_Sxz = 2*Sxz;
    dL_eyy = (1/(dx*dz)) * convgrad(dL_Sxz, axial_kernel);

    % function output
    score = sum(Sxz.^2,"all");
    grad = dL_eyy(2:end-1, 2:end-1);


end