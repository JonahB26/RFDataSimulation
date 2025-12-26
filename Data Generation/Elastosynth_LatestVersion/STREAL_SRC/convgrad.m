
function grad = convgrad(dL_dO, kernel)

    grad = conv2(rot90(rot90(kernel)), dL_dO, 'full');

end