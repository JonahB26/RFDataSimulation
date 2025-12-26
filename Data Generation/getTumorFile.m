function [TumorMaskPath,tumorFileLocation,numTumorFile,numImageFile] = getTumorFile(CurrentImageFile,TumorFiles,TumorMasksFolder)
% This function takes an image filename as the input, and locates the corresponding tumor mask file. 
   [startindA,endindA] = regexp(CurrentImageFile,'\d*');
   numImageFile = CurrentImageFile(startindA:endindA);
    %Find matching tumor mask file and obtain its path.

    for j = 1:length(TumorFiles)

        TumorFileName = TumorFiles(j).name;
        
        [startindB,endindB] = regexp(TumorFileName,'\d*');

        numTumorFile = TumorFileName(startindB:endindB);

        % %Need to check and remove any leading zeros of the string
        % for k = 1:length(numTumorFile)
        %     if numTumorFile(1) ~= '0'
        %         break
        %     else
        %         numTumorFile(1) = [];
        %     end
        % end
                
        if strcmp(numImageFile,numTumorFile)

            %This statement checks to see if it contains any of the -l, -r,
            %-s endings as shown in the BUSBRA filenames.
            if contains(TumorFileName,'-l') && contains(CurrentImageFile,'-l')
                disp(['Matching file for ',CurrentImageFile,' is ',TumorFileName]);
                tumorFileLocation = j;
                TumorMaskPath = fullfile(TumorMasksFolder,TumorFileName);
                break
            elseif contains(TumorFileName,'-r') && contains(CurrentImageFile,'-r')
                disp(['Matching file for ',CurrentImageFile,' is ',TumorFileName]);
                tumorFileLocation = j;
                TumorMaskPath = fullfile(TumorMasksFolder,TumorFileName);
                break
            elseif contains(TumorFileName,'-s') && contains(CurrentImageFile,'-s')
                disp(['Matching file for ',CurrentImageFile,' is ',TumorFileName]);
                tumorFileLocation = j;
                TumorMaskPath = fullfile(TumorMasksFolder,TumorFileName);
                break
            elseif contains(TumorFileName,'-s') || contains(TumorFileName,'-r') || contains(TumorFileName,'-l')
                continue
            end

            disp(['Matching file for ',CurrentImageFile,' is ',TumorFileName]);
            tumorFileLocation = j;
            TumorMaskPath = fullfile(TumorMasksFolder,TumorFileName);
            break
        end
    end
end