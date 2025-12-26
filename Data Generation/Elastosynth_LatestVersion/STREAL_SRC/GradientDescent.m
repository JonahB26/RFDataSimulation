function [elasped_time, last_iter, curr_strains] = GradientDescent(strains, kernel, weights, dx, dy, dz, DECAY, TOLERANCE_BEFORE_EXIT, MIN_IMPROVEMENT, THRESHOLD_OF_IMPROVEMENT, EPSILON, lr)

tic

iter = 1000;
count = 0;
last_iter = 0;

strain_dim = size(strains);
curr_strains = strains;

[init_error, grad] = CmptError(curr_strains, weights, kernel, dx, dy, dz, [strain_dim(1) strain_dim(2)]);

for i = 1:iter
    [score, grad] = CmptError(curr_strains, weights, kernel, dx, dy, dz, [strain_dim(1) strain_dim(2)]);

    if score > prev_error
        count = count + 1;
        lr = lr * DECAY;
        if count > TOLERANCE_BEFORE_EXIT
            "Min Error Exit"
            score
            break
        end
    elseif score* MIN_IMPROVEMENT > prev_error
        count = count + 1;
        lr = lr * DECAY;
        if count > TOLERANCE_BEFORE_EXIT
            "Min Improvement Exit"
            score
            break
        end
    elseif (score/init_error) < (1 - THRESHOLD_OF_IMPROVEMENT)
        "Good Enough Exit"
        score
        break
    else
        count = 0;
    end

    curr_strains(:,:,1,2) = curr_strains(:,:,1,2) - lr.*grad;

    prev_error = score;
    
    if score < EPSILON
        "Low Error Exit"
        score
        break
    end
    
    last_iter = i;
end

end