axialres = 800;
latres = 800;
x = 1:latres;
%convpoint = [200 100];
%lateralshifts = [1*convpoint(1) 1.1*convpoint(1) 1.4*convpoint(1)];
%d = 0.1*800;
% lig1a = convpoint(1) + 50;
% lig2a = convpoint(1) - 50;
% lig3a = convpoint(1) + 200;
% lig4a = convpoint(1) - 200;
% lig5a = convpoint(1) + 400;
%lig6a = convpoint(1) -400;
% axialshifts = [lig1a lig2a lig3a lig4a lig5a];% lig6a];
% withinboundaries = (axialshifts>=0) & (axialshifts<=800);
% axialshifts = axialshifts(withinboundaries);
axialshifts = [0.2*axialres 0.4*axialres 0.6*axialres 0.8*axialres];
for i = 1:length(axialshifts)
    y = -0.0005*(x-400).^2 +axialshifts(i);%300;
    plot(x,y)
    hold on
end
% lig1 = convpoint(2) + 100;
% lig2 = convpoint(2) - 100;
% lig3 = convpoint(2) + 200;
% lig4 = convpoint(2) - 200;
% lig5 = convpoint(2) + 400;
%lig6 = convpoint(2) - 400;
% lig5 = convpoint(2) + 350;
% lig6 = convpoint(2) - 350;
% latshifts = [lig1 lig2 lig3 lig4 lig5];% lig6];
% withinboundaries = (latshifts>=0) & (latshifts<=800);
% latshifts = latshifts(withinboundaries);
latshifts = [0.2*latres 0.4*latres 0.6*latres 0.8*latres];
for i = 1:length(latshifts)
    y = 0.0005*(x-400).^2 +axialshifts(i);%300;
    plot(y,x)
    hold on
end

% for i = 1:length(latshifts)
% 
% %t = -0.0005*(x-300).^2 +500;
set(gca,'XAxisLocation','top','YAxisLocation','left','ydir','reverse');
axis([0 800 0 800])



% t = 1:800;
% x = t;
% y = -0.0006.*(t-600).^2+350;
% alpha1 = 0.6;
% 
% % xtran1 = x + alpha1.*(300-x);
% % ytran1 = y + alpha1*(500-y);
% % plot(xtran1,ytran1)
% % hold on
% 
% % y2 = -0.0008*(x-700).^2+600;
% % alpha2 = 0.6;
% % y2tran = y + alpha2*(500-y);
% % plot(xtran1,y2tran)
% 
% 
% 
% 
% 
% 
% 
% 
% 
% % plot(x,y1)
% % y2 = -0.0008*(x-700).^2+600;
% % plot(x,y2)
grid on
set(gca,'XAxisLocation','top','YAxisLocation','left','ydir','reverse');
axis([0 800 0 800])