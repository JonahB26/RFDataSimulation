function [score, ygrad, xgrad] = IncmpError(disps, kernel, dx, dy, k_mat, init_disps, alpha)

    exx = conv2(disps(:,:,1,1),kernel','valid') * (1/dx);
    eyy = conv2(disps(:,:,1,2),kernel,'valid') * (1/dy);
    
    I = (1 + k_mat).*exx + eyy;
    score = alpha*sum(I.^2, 'all');

    ygrad = alpha.*(1/dy)*convgrad(2*(I), kernel);
    xgrad = alpha.*0.1*(1/dx)*convgrad(2*(I).*(1+k_mat), kernel');

end