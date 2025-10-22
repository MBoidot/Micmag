% Pour importer les données des images consigénes dans les fichiers excel
% sous forme de matrices utilisables pour editer des données synthétiques

%E : liste des profondeurs en pixels pour les photos TR
%A : liste des angles moyens en fonction de E pour les TR
%C : liste des bords des classes granulométriques
%D : liste des distributions granulométriques selon C 

n=1; %numéro de la photo FACE   en prenant les types dans l'ordre et les photos dans l'ordre
m=1; %numéro de la photo PROFIL en prenant les types dans l'ordre et les photos dans l'ordre




for ech=1:length(ECHTS)
    echt=ECHTS(ech);
    for typ=0:(nbtypes-1) 
        type=TYPES(typ*3+1:typ*3+3);
        if typ==(nbtypes-1) % cas des photos TR
            type=type(1:2);
            nbrephotos=NBPHOTOS(nbtypes*(ech-1)+typ+1);
            for fich=1:nbrephotos
                photo=strcat(nomanip,echt,'-',type,num2str(fich));
                cheRES=strcat(cheM,'Résultats\',photo,'.xls');
                % traitement tranche
                Z=xlsread(cheRES);
                E=Z(:,1);
                A(:,m)=Z(:,2);
                m=m+1;
            end
        else  % cas des photos de face
            nbrephotos=NBPHOTOS(nbtypes*(ech-1)+typ+1);
            for fich=1:nbrephotos
                photo=strcat(nomanip,echt,'-',type,num2str(fich));
                cheRES=strcat(cheM,'Résultats\',photo,'.xls');  
                Z=xlsread(cheRES);
                % traitement vue de face
                C=Z(:,1);
                D(:,n)=Z(:,2);
                n=n+1;
             end
        end
    end
end

% Correction de l'angle apparent vers l'angle réel (correction statistique)
