close all

fig1=figure('OuterPosition',[0,40,600,600],'Color','w','PaperPositionMode','auto'); 
caxis([0 1]); 
hold on
distrimax=0;
for i=2:nbtypes
    plot(districorr(:,1), districorr(:,i),'Color',[1 0.5*i/nbtypes+0.4 0.2*i/nbtypes+0.1],'LineWidth',1.5);
    distrimax=max(distrimax,max(districorr(:,i)));
end
plot(districorr(:,1),districorr(:,nbtypes+1),'Color',[0.2 0.2 1],'LineWidth',3);
title(strcat(manip,'-',echt),'FontWeight','bold','FontSize',14);
set(gca,'FontSize',12);
set(gca,'ytick',[])
axis([0,dmaxfigures,0,1.1*distrimax]);
xlabel('NdFeB lamella width [µm]','FontWeight','bold','FontSize',12);
ylabel('Occurence','FontWeight','bold','FontSize',12);
plot([D10(nbtypes),D10(nbtypes)],[0,HD10(nbtypes)],'LineStyle','--','Color',[0.5 0.5 0.5],'LineWidth',1)
plot([D50(nbtypes),D50(nbtypes)],[0,HD50(nbtypes)],'LineStyle','--','Color',[0.5 0.5 0.5],'LineWidth',1)
plot([D90(nbtypes),D90(nbtypes)],[0,HD90(nbtypes)],'LineStyle','--','Color',[0.5 0.5 0.5],'LineWidth',1)
t = text(D10(nbtypes)+0.1,HD10(nbtypes),strcat(' d10=',num2str(round(D10(nbtypes)*10)/10),'µm'),  'FontSize',10, 'FontWeight','bold','Color',[0.2 0.2 0.2]);
u = text(D50(nbtypes)+0.1,HD50(nbtypes),strcat(' d50=',num2str(round(10*D50(nbtypes))/10),'µm'),  'FontSize',10, 'FontWeight','bold','Color',[0.2 0.2 0.2]);
v = text(D90(nbtypes)+0.1,HD90(nbtypes),strcat(' d90=',num2str(round(10*D90(nbtypes))/10),'µm'),  'FontSize',10, 'FontWeight','bold','Color',[0.2 0.2 0.2]);
t5 = text(dmaxfigures*0.65,HD10(nbtypes),strcat('Mean spacing: ',num2str(round(spacingcorr(nbtypes,1)*100)/100),'µm'),  'FontSize',12, 'FontWeight','bold','Color',[0.5 0.2 1]);
t6 = text(dmaxfigures*0.65,HD10(nbtypes)*1.3,strcat('Thickness: ',num2str(epaisseurs(ech)),'µm'),  'FontSize',12, 'FontWeight','bold','Color',[0.5 0.2 1]);
legend('Wheel','1/3 from wheel','1/3 from rear','Rear','All');
legend('boxoff');
hold off

fig2=figure('OuterPosition',[600,40,600,600],'Color','w','PaperPositionMode','auto'); 
caxis([0 1]); 
axis([0,4.5,0,90]);
hold on
plot(1:4,anglesmoy(1:4),'Color','blue','LineStyle','none','Marker','Diamond','MarkerSize',10,'LineWidth',2);
title(strcat(manip,'-',echt),'FontWeight','bold','FontSize',14);
set(gca,'FontSize',12);
set(gca,'xtick',[])
ylabel('Mean angle','FontWeight','bold','FontSize',12);

t1 = text(0.8,-2,strcat('Wheel'),  'FontSize',10, 'FontWeight','bold','Color',[0.2 0.2 0.2]);
t2 = text(1.8,-2,strcat('1/3 deep'),  'FontSize',10, 'FontWeight','bold','Color',[0.2 0.2 0.2]);
t3 = text(2.8,-2,strcat('2/3 deep'),  'FontSize',10, 'FontWeight','bold','Color',[0.2 0.2 0.2]);
t4 = text(3.8,-2,strcat('Rear'),  'FontSize',10, 'FontWeight','bold','Color',[0.2 0.2 0.2]);

for i=1:(nbtypes-1)
plot([i-0.1,i+0.1,i,i,i-0.1,i+0.1],[anglesmoy(i)+anglesigma(i),anglesmoy(i)+anglesigma(i),anglesmoy(i)+anglesigma(i),anglesmoy(i)-anglesigma(i),anglesmoy(i)-anglesigma(i),anglesmoy(i)-anglesigma(i)],'LineStyle','-','Color',[0.4 0.4 0.4],'LineWidth',1)
end
hold off

cheRES=strcat(cheM,'Résultats\');
print(fig1,'-dpng',strcat(cheRES,manip,'-',echt,'-distributions.png'));
print(fig2,'-dpng',strcat(cheRES,manip,'-',echt,'-angles.png'));