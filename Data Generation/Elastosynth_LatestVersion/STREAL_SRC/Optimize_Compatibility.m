function [improved_strains, last_iter] = Optimize_Compatibility(strains, weights, kernel, dx, dy, dims, TOLERANCE_BEFORE_EXIT, THRESHOLD_OF_IMPROVEMENT, MIN_IMPROVEMENT, lr)
    
    EPSILON = 0.00001;
    iter = 1000;
    error = zeros(iter,1);
    grad_norm = zeros(iter,1);
    curr_strains = strains;
    count = 0;
    last_iter = 0;
    
    [init_error, grad] = CmptError(curr_strains, weights, kernel, dx, dy, 0.001, dims - [2 2]);
    prev_error = init_error;
    
    for i = 1:iter
    
        [score, grad] = CmptError(curr_strains, weights, kernel, dx, dy, 0.001, dims - [2 2]);
    
        if score > prev_error
            count = count + 1;
            if count > TOLERANCE_BEFORE_EXIT
                disp("Min Error Exit")
                % score
                break
            end
        elseif score* MIN_IMPROVEMENT > prev_error
            count = count + 1;
            if count > TOLERANCE_BEFORE_EXIT
                disp("Min Improvement Exit")
                % score
                break
            end
        elseif (score/init_error) < (1 - THRESHOLD_OF_IMPROVEMENT)
            disp("Good Enough Exit")
            % score
            break
        else
            count = 0;
        end
    
        curr_strains(:,:,1,2) = curr_strains(:,:,1,2) - lr.*grad;
    
        prev_error = score;
        
        if score < EPSILON
            disp("Low Error Exit")
            % score
            break
        end
        
        last_iter = i;
    end
    
    improved_strains = curr_strains;

end