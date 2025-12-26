function [Disp_ax,Disp_lat,strainA,strainL,strainS]...
              = prepdispsforreconstruction(axial_disp, lateral_disp)

    Disp_ax = imresize(axial_disp, [222 201],"bicubic");
    Disp_lat = imresize(lateral_disp, [222 201],"bicubic");
    
    strainA = conv2(Disp_ax,[-1;1],'valid');
    strainL = conv2(Disp_lat,[-1 1],'valid');
    strainL = strainL(1:end-2,:);
    strainA = strainA(1:end-1,1:end-1);
    Disp_ax = Disp_ax(1:end-1,:);
    Disp_lat = Disp_lat(1:end-1,:);
    
    t1 = conv2(Disp_ax,[-1 1],'valid');
    t2 = conv2(Disp_lat,[-1; 1],'valid');
    t1 = t1(1:end-1,:);
    t2 = t2(:,1:end-1);
    strainS = 0.5*(t1+t2);

end