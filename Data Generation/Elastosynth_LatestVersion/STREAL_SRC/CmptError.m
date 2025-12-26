function [score, grad] = CmptError(inputs, weights, kernel, dx, dy, dz, dims)

    NO_ERRORS = 3;

    errs = cell(1, NO_ERRORS);
    scores = zeros(1, NO_ERRORS);
    grads = zeros(dims(1), dims(2), NO_ERRORS);

    errs{1} = @GetSxx;
    errs{2} = @GetSzz;
    errs{3} = @GetSxz;
    
    for i = 1:NO_ERRORS
        [scores(i), grads(:,:,i)] = errs{i}(inputs,kernel,dx,dy,dz);
        scores(i) = scores(i)*weights(i);
        grads(:,:,i) = grads(:,:,i)*weights(i);
    end

    score = sum(scores);
    grad = sum(grads,3);


end