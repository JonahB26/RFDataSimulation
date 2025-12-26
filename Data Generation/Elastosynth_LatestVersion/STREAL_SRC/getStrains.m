function strains = getStrains(disps, dx, dy, dz, lateral_kernel)

    axial_kernel = lateral_kernel'; % derivative in axial dimension
    disp_dims = size(disps);
    strain_dims = disp_dims - [2 2 1 -3];
    strains = zeros(strain_dims);
        
    exx = conv2(disps(:,:,1,1),axial_kernel,'valid') * (1/dx);
    strains(:,:,1,1) = exx;

    eyy = conv2(disps(:,:,1,2),lateral_kernel,'valid') * (1/dy);
    strains(:,:,1,2) = eyy;

    ezz = (disps(:,:,2,3) - disps(:,:,1,3))/dz;
    strains(:,:,1,3) = ezz(2:disp_dims(1)-1, 2:disp_dims(2)-1, 1);