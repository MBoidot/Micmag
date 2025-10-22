fig1=figure('OuterPosition',[0,40,600,500],'Color','w'); caxis([0 1]);
axis off
NN=boolean(N);
image(N(plagex,plagey),'CDataMapping','scaled');
set(gca,'Visible','off');
set(gcf,'Colormap',[0,0,0;1,1,1]);
print('-dtiff','-r120',strcat(chePH,'-binaire.TIF'));

fig3=figure('OuterPosition',[0,540,600,500],'Color','w'); caxis([0 1]); 
axis off
image(DM(plagex,plagey),'CDataMapping','scaled');
set(gca,'Visible','off');
print('-dtiff','-r120',strcat(chePH,'-traité.TIF'));

fig4=figure('OuterPosition',[600,40,600,500],'Color','w'); caxis([0 1]); 
plot(Classes, distri,'LineWidth',1.5);
title(photo);
print('-dtiff','-r120',strcat(chePH,'-data.TIF'));

DAT=dataset(Classes,distri); % a cet stade la valeur i représente l'occurence entre les classes i et i+1.
export(DAT,'XLSfile',cheRES); 