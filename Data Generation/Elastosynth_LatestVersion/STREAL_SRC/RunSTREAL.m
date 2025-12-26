function [improved_ex, improved_ey, improved_ux, improved_uy] = RunSTREAL(ux, uy, kernel, target_dims, kmat, L, D, DECAY, TOLERANCE_BEFORE_EXIT, MIN_IMPROVEMENT, THRESHOLD_OF_IMPROVEMENT)

    % Calculate dx, dy
    dx = D/target_dims(1);
    dy = L/target_dims(2);

    % Resize the images and merge into one matrix
    ux = imresize(ux,target_dims,'method', "bilinear");
    uy = imresize(uy,target_dims,'method', "bilinear");
    disps = cat(4,ux,uy);
    size(disps);

%     figure
%     subplot(2,3,1)
%     imshow(conv2(ux, kernel', "valid"), [])
%     title("Resize Only ex")
%     colorbar
%     subplot(2,3,4)
%     imshow(conv2(uy, kernel, "valid"),[])
%     title("Resize Only ey")
%     colorbar

    % Slice kMat
    k_mat_layer1 = kmat(:,:,1,3);

    mask = 1;

    % Step 1
    [curr_disps, last_iter] = Optimize_Incompressibility(disps, kernel, dx, dy, k_mat_layer1, target_dims, ...
        TOLERANCE_BEFORE_EXIT, THRESHOLD_OF_IMPROVEMENT, MIN_IMPROVEMENT, mask);

    % last_iter
    
    improved_ux = curr_disps(:,:,1,1);
    improved_uy = curr_disps(:,:,1,2);

    % improved_ux = max(improved_ux(:)) + min(improved_ux(:)) - improved_ux;
    % improved_uy = max(improved_uy(:)) + min(improved_uy(:)) - improved_uy;

%     subplot(2,3,2)
%     imshow(conv2(improved_ux, kernel', "valid"), [])
%     title("Incmp Only ex")
%     colorbar
%     subplot(2,3,5)
%     imshow(conv2(improved_uy, kernel, "valid"),[])
%     title("Incmp Only ey")
%     colorbar

    ex = conv2(improved_ux,kernel',"valid");
    ey = conv2(improved_uy,kernel,"valid");

    strains = zeros(size(ex, 1), size(ex,2), 3, 6);

    for i = 1:6
        for j = 1:3
            strains(:,:,j,i) = ex.*kmat(:,:,j,i);
        end
    end
    
    strains(:,:,1,2) = ey;

    iter = 200;
    lr = 1e-12 * dx/(40/180);
    weights = [1 0 0];
    
    [improved_strains, last_iter] = Optimize_Compatibility(strains, weights, kernel, dx, dy, target_dims, TOLERANCE_BEFORE_EXIT, THRESHOLD_OF_IMPROVEMENT, MIN_IMPROVEMENT, lr);

    improved_ex = improved_strains(:,:,1,1);
    improved_ey = improved_strains(:,:,1,2);

%     subplot(2,3,3)
%     imshow(improved_ex, [])
%     title("Full ex")
%     colorbar
%     subplot(2,3,6)
%     imshow(improved_ey,[])
%     title("Full ey")
%     colorbar
end