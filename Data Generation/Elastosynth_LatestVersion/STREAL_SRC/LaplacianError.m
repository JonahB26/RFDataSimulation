function [score, ygrad, xgrad] = LaplacianError(disps, kernel, dx, dy, k_mat, init_disps, alpha)

    laplace_kernel = [-1,-1,-1; -1,8,-1; -1,-1,-1];

    xLaplacian = conv2(disps(:,:,1,1), laplace_kernel, 'valid');
    yLaplacian = conv2(disps(:,:,1,2), laplace_kernel, 'valid');

    L = xLaplacian + yLaplacian;

    score = alpha.*sum(L.^2,'all');
    
    ygrad = alpha.*convgrad(2*yLaplacian, laplace_kernel);
    xgrad = alpha.*convgrad(2*xLaplacian, laplace_kernel);


end