function [Frame1,Frame2,transducer_num] = FieldIIBatch(simresult,cooper_mask,tumor_mask,label)
transducer_list = genTransducers();
transducer_num = 3;
transducer = transducer_list{transducer_num};
sim_resolution = [256 256];

% Calculate Image Options
imageopts = ImageOpts((transducer.N_elements-transducer.N_active)/2, (transducer.element_width+transducer.kerf)*(transducer.N_elements-transducer.N_active),...
40/1000, 40/1000,10/1000, 256*256,100);
imageopts.decimation_factor = 2;
imageopts.axial_FOV = 60/1000;
imageopts.lateral_FOV = 1.2*(transducer.element_width+transducer.kerf)*(transducer.N_elements-transducer.N_active-1) + transducer.kerf;
imageopts.slice_thickness = 10/1000;

field_init(0);

D = imageopts.axial_FOV;
L = imageopts.lateral_FOV;
Z = imageopts.slice_thickness;

[X,Y] = meshgrid(linspace(-L/2,L/2,sim_resolution(2)+1),linspace(0,D,sim_resolution(1)+1)+0.03);
I = zeros(256,256); %background
I(tumor_mask) = 1; %tumor
I(cooper_mask) = 2; %coopers

[phantom_positions, phantom_amplitudes] = ImageToScatterers(I, D,L, Z, imageopts.n_scatterers,label);

phantom = Phantom(phantom_positions, phantom_amplitudes);

dispx = interp2(X,Y,simresult.axial_disp,phantom_positions(:,1),phantom_positions(:,3));
dispy = interp2(X,Y,simresult.lateral_disp,phantom_positions(:,1),phantom_positions(:,3));

displacements = zeros(imageopts.n_scatterers, 3);
displacements(:,3) = dispx/1000;
displacements(:,1) = dispy/1000;
displacements(:,2) = 0;

clc
disp('Running FIELD II...')
[Frame1, Frame2] = GenerateFramePairLinear(phantom, displacements, transducer, imageopts, imageopts.speed_factor);

% Process FIELD II result
Frame1 = Frame1./max(Frame1(:));
Frame2 = Frame2./max(Frame2(:));

if size(Frame1,1) >= size(Frame2,1)
    Frame1 = Frame1(1:size(Frame2,1),:);
else
    Frame2 = Frame2(1:size(Frame1,1),:);
end

Frame1 = imresize(Frame1, [2500, 256]);
Frame2 = imresize(Frame2, [2500, 256]);

disp(['Min RF data 1 is: ',num2str(min(Frame1(:)))]);
disp(['Max RF data 1 is: ',num2str(max(Frame1(:)))]);

disp(['Min RF data 2 is: ',num2str(min(Frame2(:)))]);
disp(['Max RF data 2 is: ',num2str(max(Frame2(:)))]);