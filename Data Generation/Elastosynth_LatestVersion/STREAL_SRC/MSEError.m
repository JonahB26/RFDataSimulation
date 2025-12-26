function [score, ygrad, xgrad] = MSEError(disps, kernel, dx, dy, k_mat, init_disps, alpha)

    xErr = disps(:,:,1,1) - init_disps(:,:,1,1);
    yErr = disps(:,:,1,2) - init_disps(:,:,1,2);

    score = alpha.*sum((xErr + yErr).^2,'all');

    ygrad = alpha.*2*yErr;
    xgrad = alpha.*2*xErr;

end