% Analyse les images et retourne un ficher .xls pour chaque image.

for ech=1:length(ECHTS)             % Pour chaque échantillon
    echt=ECHTS(ech);        
    nbtypes=floor(length(TYPES)/3);
    for typ=0:(nbtypes-1)           % Pour chaque type correspondant à une profondeur
        type=TYPES(typ*3+1:typ*3+3);
        if typ==(nbtypes-1)
            type=type(1:2);
        end
        nbrephotos=NBPHOTOS(5*(ech-1)+typ+1);
        for fich=1:nbrephotos       % Pour chaque photos de l'échantillon et du type choisi
            close all;
            photo=strcat(nomanip,echt,'-',type,num2str(fich));
            chePH=strcat(cheM,photo);    % vers la photo
            cheRES=strcat(cheM,'Résultats\',photo,'.xls');  % le résultat de l'analyse
            M=imread(strcat(chePH,'.TIF'));
            M=double(M)/255;
            [hauteur,largeur]=size(M);
            plagex=10:hauteur;  % mettre 10 pour accélérer pour test
            plagey=10:largeur;  % mettre 10 pour accélérer pour test

            if not(sum(type=='T'))  % vues de face (calcul spacing)
                binarisation;
                distance;
                distribution;
            else   % tranches (calcul angle, pour correction du spacing)
                binarisation;
                calculangle;
            end
            dessinexport;
         end
    end
end







