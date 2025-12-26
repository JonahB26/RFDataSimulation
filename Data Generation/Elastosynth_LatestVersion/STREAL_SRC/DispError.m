function [score, ygrad, xgrad] = DispError(disps, init_disps, kernel, dx, dy, k_mat, dims, alpha)

    NO_ERRORS = 3;

    errs = cell(1, NO_ERRORS);
    scores = zeros(1, NO_ERRORS);
    ygrads = zeros(dims(1), dims(2), NO_ERRORS);
    xgrads = zeros(dims(1), dims(2), NO_ERRORS);
    
    errs{1} = @IncmpError;
    errs{2} = @LaplacianError;
    errs{3} = @MSEError;

    for i = 2:NO_ERRORS
        [scores(i), ygrads(:,:,i), xgrads(:,:,i)] = errs{i}(disps, kernel, dx, dy, k_mat, init_disps, alpha(i));
    end

    score = sum(scores);
    ygrad = sum(ygrads,3);
    xgrad = sum(xgrads,3);

end