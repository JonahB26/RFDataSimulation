function [refined_disps, last_iter] = Optimize_Incompressibility(disps, kernel, dx, dy, k_mat, dims, TOLERANCE_BEFORE_EXIT, THRESHOLD_OF_IMPROVEMENT, MIN_IMPROVEMENT, mask)
    
    params = [0    1    100   0.0113    0.0106];

    alpha = params(1:3);
    xlr = params(4) * dx/(60/200);
    ylr = params (5) * dy/(50/180);

    EPSILON = 0.0001;
    iter = 2000;
    curr_disps = disps;
    init_disps = disps;
    
    error = zeros(iter,1);
    grad_norm = zeros(iter,1);
    prev_error = 1e6;
    count = 0;

    [init_error, ygrad, xgrad] = DispError(curr_disps, init_disps, kernel, dx, dy, k_mat, dims, alpha);

    for i = 1:iter
        [score, ygrad, xgrad] = DispError(curr_disps, init_disps, kernel, dx, dy, k_mat, dims, alpha);
    
        if score > prev_error
            count = count + 1;
            if count > TOLERANCE_BEFORE_EXIT
                "Min Error Exit"
                score
                break
            end
        elseif score* MIN_IMPROVEMENT > prev_error
            count = count + 1;
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
    
        size(mask*ylr.*ygrad)
        curr_disps(:,:,1,2) = curr_disps(:,:,1,2) - mask*ylr.*ygrad;
        curr_disps(:,:,1,1) = curr_disps(:,:,1,1) - mask*xlr.*xgrad;
    
        prev_error = score;
    
        error(i) = score;
        grad_norm(i) = norm(ygrad+xgrad);
        
        if score < EPSILON
            "Low Error Exit"
            score
            break
        end
        
        last_iter = i;
    end
    refined_disps = curr_disps;
end