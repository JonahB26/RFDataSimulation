<<<<<<< HEAD:obtainYM_Image.m
function [YM_Image,minTumorYM,maxTumorYM,cooperImageData,TumorMask] = obtainYM_Image(ImagePath,TumorMaskPath,varargin)
=======
function [YM_Image,minTumorYM,maxTumorYM,cooperImageData,TumorMask] = obtainYM_Image(ImagePath,TumorMaskPath,Scale,varargin)
>>>>>>> bc846e5e2df480e9c9ebfdf1415ec87892d556c4:Data Generation/obtainYM_Image.m
%UNTITLED3 Summary of this function goes here
%   Detailed explanation goes here
   
    
    TumorMask = imread(TumorMaskPath);
    
    if size(TumorMask,3) == 3
        TumorMask = rgb2gray(TumorMask);
    end
    TumorMask = logical(TumorMask);


    %Check and see if there were additional input arguments, and adjust
    %tumor mask as needed.
    if ~isempty(varargin)

        for i = 1:length(varargin)
            
            %Load image and convert to gray.
            TumorMaskPath = varargin{1};
            TumorMaskadded = imread(TumorMaskPath);
            if size(TumorMaskadded,3) == 3
                TumorMaskadded = rgb2gray(TumorMaskadded);
            end
            TumorMaskadded = logical(TumorMaskadded);

            TumorMask = TumorMask + TumorMaskadded;
        end
    end
    

    
    axialresolution = size(TumorMask,1);
    lateralresolution = size(TumorMask,2);
    
    % fig = openfig(ImagePath);
    % 
    % pos = get(gcf,'Position');
    % fig_width_pixel = pos(3);
    % fig_height_pixel = pos(4);
    % screen_dpi = get(0,'ScreenPixelsPerInch');
    % fig_width_inches = fig_width_pixel/screen_dpi;
    % fig_height_inches = fig_height_pixel/screen_dpi;
    % dpi_width = lateralresolution / fig_width_inches;
    % dpi_height = axialresolution / fig_height_inches;
    % dpi = max(dpi_width,dpi_height);
    % dpi = round(dpi);
    % 
    % set(gcf,'Color','none');
    % set(gca,'Color','none','Position',[0 0 1 1],'Units','normalized');
    % exportgraphics(gca,'Updated.png','BackgroundColor','none','Resolution',dpi);
    % 
    % CoopersData = findobj(gca,'Type','line');
    % CoopersX = get(CoopersData,'XData');
    % CoopersY = get(CoopersData,'YData');
    % 
    % 
    % hold on
    % for i = 1:size(CoopersX)
    %     plot(cell2mat(CoopersX(i)),cell2mat(CoopersY(i)),'Color',[1 1 1],'Linewidth',2)
    %     set(gca,'XAxisLocation','top','YAxisLocation','left','ydir','reverse');
    %     axis([0 axialresolution 0 lateralresolution])
    % end
    % hold off
    % 
    % CoopermaskImage = getframe(gca);
    % Coopermaskdata = rgb2gray(CoopermaskImage.cdata);
    
    load(ImagePath);
    %cooperImageData.TumorMask = TumorMask;

    %Image.loaded = imresize(Image.loaded,[axialresolution lateralresolution]);

    % Coopermask = Image.loaded > 254;
    % Image.Coopermask = Coopermask;
    % Cooperimg = im2gray(uint8(Image.cooperImageData.cooperImage));
    % Originalimg = im2gray(uint8(Image.cooperImageData.originalImage));
    % Coopermask = Cooperimg - Originalimg;
    % Coopermask = logical(Coopermask);
    % % if size(Coopermask,3) == 3
    % %     Coopermask = logical(double(rgb2gray(uint8(Coopermask))));
    % % end
    %figure,imshow(Coopermask)
    Coopermask = cooperImageData.cooperMask;
    %figure,imshow(Coopermask)

    %Coopermask = Coopermaskdata > 0;
    
    YM_Image = zeros(axialresolution,lateralresolution);

    assert(isequal(size(YM_Image),size(TumorMask)),'YM_Image and TumorMask must be same size.')
    assert(isequal(size(Coopermask),size(TumorMask)),'Coopermask and TumorMask must be same size.')

    randTumorYM = 5000 + (15000 - 5000)*rand(1);%Generate random YM between 5-15kPa
    sprintf('RandomTumorYM is %d',randTumorYM)
    minTumorYM = randTumorYM - 1500; %Set range for tumor YM throughout image
    sprintf('MinTumorYM is %d',minTumorYM)
    maxTumorYM = randTumorYM + 1500;
    sprintf('MaxTumorYM is %d',maxTumorYM)
    TumorMask = logical(TumorMask);
    tumorSize = nnz(TumorMask);
    disp(tumorSize)
    tumorAdditions = minTumorYM + (maxTumorYM - minTumorYM)*rand(tumorSize,1);
    disp(length(tumorAdditions))
    assert(isequal(length(tumorAdditions),sum(TumorMask(:))),'Added points must equal size of tumors.')
    YM_Image(TumorMask) = tumorAdditions; %Generate random YM values across image based on randomly calculated 3kPa range
    
    cooperAdditions = 2500000 + (3500000 - 2500000)*rand(nnz(Coopermask),1);
    disp(size(Coopermask));
    disp(size(YM_Image));
<<<<<<< HEAD:obtainYM_Image.m
    YM_Image(Coopermask) = cooperAdditions; %2.5-2.5MPa range, SCALED 0.75 FOR TESTING
=======

    % Adjusted so that user can input the magnitude of ligaments they want.
    % highest is 3 MPa
    YM_Image(Coopermask) = Scale*cooperAdditions; %2.5-2.5MPa range, SCALED 0.75 FOR TESTING
>>>>>>> bc846e5e2df480e9c9ebfdf1415ec87892d556c4:Data Generation/obtainYM_Image.m
    
    backgroundAdditions = 2500 + (3500 - 2500)*rand(sum(YM_Image==0,"all"),1);
    YM_Image(YM_Image==0) = backgroundAdditions; %Generate random YM values across image between 2.5-3.5kPa
end