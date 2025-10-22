% Angles : Enlève les zéros inutiles dnas les colonnes d'angles et calcule les angles moyens + écart-type par catégorie (type RRR etc)

echt=ECHTS(ech);
v=[];
for i=1:nbtypes-1
    ii=(nbtypes-1)*(ech-1)+i;
    t=angles(:,ii);
    j=length(t);
    while j>0 && t(j)==0
        j=j-1;
    end
    u=t(1:j);
    v=[v;u];
    anglesmoy(i,1)=mean(u);
    anglesigma(i,1)=std(u);

    % Valeurs moyennes
    spacingbrut(i,1)=sum((C+pasdistri/2).*distri(:,ii));
    spacingcorr(i,1)=spacingbrut(i,1)*cosd(anglesmoy(i));

    % Distributions interpolées selon des points réguliers et où les valeurs y correspondent à une population centrée sur x
    districorr(:,1)=dx;
    distribrut(:,1)=dx;
    districorrcum(:,1)=dx;
    distribrutcum(:,1)=dx;
    db=interpole(C+pasdistri/2,distri(:,ii),dx);
    for j=1:LX
        dbc(j)=sum(db(1:j));
    end
    distribrut(:,i+1)=db/sum(db);
    distribrutcum(:,i+1)=dbc/sum(db);
    dc=interpole((C+pasdistri/2)*cosd(anglesmoy(i)),distri(:,ii),dx);
    for j=1:LX
        dcc(j)=sum(dc(1:j));
    end
    districorr(:,i+1)=dc/sum(dc);
    districorrcum(:,i+1)=dcc/sum(dc);
end

% Angles en moyenne sur le vecteur angle assemblé

anglesmoy(nbtypes*1)=mean(v);
anglesigma(nbtypes,1)=std(v);

% Distribution sur tout l'échantillon : rajoute une dernière colonne dans les districorr
districorr(:,nbtypes+1)=mean(districorr(:,2:nbtypes),2);
districorrcum(:,nbtypes+1)=mean(districorrcum(:,2:nbtypes),2);
spacingbrut(nbtypes,1)=0; % pas de sens de moyenner en brut
spacingcorr(nbtypes,1)=mean(spacingcorr(1:(nbtypes-1)));

for i=1:(nbtypes)
    % Calcul des dx par zone et en moyenne, en utilisant les distributions corrigées cumulées
    id10=1;id50=1;id90=1;
    for k=1:LX
        if districorrcum(k,i+1)<0.1 id10=k; end
        if districorrcum(k,i+1)<0.5 id50=k; end
        if districorrcum(k,i+1)<0.9 id90=k; end
    end
    % Interpolation pour trouver les valeurs
    D10(i,1)=((0.1-districorrcum(id10,i+1))*dx(id10+1)+(districorrcum(id10+1,i+1)-0.1)*dx(id10))/(districorrcum(id10+1,i+1)-districorrcum(id10,i+1));
    HD10(i,1)=((D10(i,1)-dx(id10))*districorr(id10+1,i+1)+(-D10(i,1)+dx(id10+1))*districorr(id10,i+1))/pasdistrifinal;
    D50(i,1)=((0.5-districorrcum(id50,i+1))*dx(id50+1)+(districorrcum(id50+1,i+1)-0.5)*dx(id50))/(districorrcum(id50+1,i+1)-districorrcum(id50,i+1));
    HD50(i,1)=((D50(i,1)-dx(id50))*districorr(id50+1,i+1)+(-D50(i,1)+dx(id50+1))*districorr(id50,i+1))/pasdistrifinal;
    D90(i,1)=((0.9-districorrcum(id90,i+1))*dx(id90+1)+(districorrcum(id90+1,i+1)-0.9)*dx(id90))/(districorrcum(id90+1,i+1)-districorrcum(id90,i+1));
    HD90(i,1)=((D90(i,1)-dx(id90))*districorr(id90+1,i+1)+(-D90(i,1)+dx(id90+1))*districorr(id90,i+1))/pasdistrifinal;
end

% EXPORT ET DESSIN

DS1=dataset(anglesmoy,anglesigma,spacingbrut,spacingcorr,D10,D50,D90);
DS2=dataset(districorr);
cheRES=strcat(cheM,'Résultats\');
export(DS1,'XLSfile',strcat(cheRES,manip,'-',echt,'-synthèse')); 
export(DS2,'XLSfile',strcat(cheRES,manip,'-',echt,'-Distribution-Spacing-angle-corrigé')); 


