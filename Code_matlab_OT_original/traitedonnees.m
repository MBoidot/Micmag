clear('angles','anglesmoy','anglesigma','spacingbrut','spacingcorr','distri');
tournangle=[0 0.23783	0.47566	0.71349	0.95132	1.29036	1.65543	2.02314	2.43516	2.84718	3.28375	3.7348	4.20001	4.68543	5.18187	5.69862	6.22752	6.77344	7.33547	7.90894	8.50452	9.10867	9.73369	10.37271	11.02294	11.69604	12.38178	13.08045	13.79911	14.53362	15.28227	16.04573	16.82911	17.62874	18.44391	19.27505	20.12256	20.98705	21.87074	22.77113	23.68847	24.62296	25.57479	26.5441	27.53101	28.53561	29.55795	30.59807	31.65599	32.73169	33.82515	34.93631	36.066	37.21428	38.38011	39.56333	40.76379	41.98131	43.21815	44.47189	45.74207	47.02874	48.33399	49.65485	50.99103	52.34514	53.71393	55.0977	56.49756	57.9108	59.33932	60.78092	62.23597	63.70385	65.18369	66.67555	68.17812	69.6915	71.21446	72.74675	74.28751	75.83595	77.39169	78.95341	80.52117	82.09336	83.66984	85.24942	86.83146	88.41515	90];

%E : liste des profondeurs en pixels pour les photos TR
%A : liste des angles moyens en fonction de E pour les TR
%C : liste des bords des classes granulométriques
%D : liste des distributions granulométriques selon C 

hauteur=max(E);
Eech=(hauteur-E)/echelleTR;
hph=hauteur/echelleTR;
distri=zeros(length(C),(nbtypes-1)*length(ECHTS));

n=1; %numéro de la photo FACE   en prenant les types dans l'ordre et les photos dans l'ordre
m=1; %numéro de la photo PROFIL en prenant les types dans l'ordre et les photos dans l'ordre

for ech=1:length(ECHTS)
    echt=ECHTS(ech);
    ep=epaisseurs(ech);
    EechR=Eech;             % les profondeurs en microns en partant de la roue
    EechA=ep-hph+Eech;      % les profondeurs en microns en partant de la roue, mais sur les photos arrière
    compteur=ones(1,nbtypes-1);  % pour mettre bout à bout toutes les valeurs
    
    for typ=0:(nbtypes-1) 
        if typ==(nbtypes-1) % cas des photos TR : traitement de l'angle
            type=type(1:2);
            nbrephotos=NBPHOTOS(5*(ech-1)+typ+1);
            for fich=1:nbrephotos
                for i=1:length(bpf)  % les types de photos de face
                    a=bpf(i,1)*ep/100;
                    b=bpf(i,2)*ep/100;
                    for k=1:hauteur     % on balaye tout et on regarde si on est à la bonne profondeur
                        if (mod(fich,2)==1 && EechR(k)<=b && EechR(k)>=a)
                            angles(compteur(i),i+(ech-1)*(nbtypes-1))=tournangle(round(A(k,m)));
                            compteur(i)=compteur(i)+1;
                        elseif (mod(fich,2)==0 && EechA(k)<=b && EechA(k)>=a)
                            angles(compteur(i),i+(ech-1)*(nbtypes-1))=tournangle(round(A(k,m)));
                            compteur(i)=compteur(i)+1;
                        end
                    end
                end
                m=m+1;
            end
            
            % angles contient tous les angles corrigés en autant de colonnes que de types de photos de face

        else  % cas des photos de face : traitement du Ndrich spacing : moyenne par type
            nbrephotos=NBPHOTOS(nbtypes*(ech-1)+typ+1);
            for fich=1:nbrephotos
                distri(:,(nbtypes-1)*(ech-1)+typ+1)=distri(:,(nbtypes-1)*(ech-1)+typ+1)+D(:,n);
                n=n+1;
            end
            distri(:,(nbtypes-1)*(ech-1)+typ+1)=distri(:,(nbtypes-1)*(ech-1)+typ+1)/sum(distri(:,(nbtypes-1)*(ech-1)+typ+1));
         end
    end
end

% Synthèse, mise en forme, export données, figures

dx=[0:pasdistrifinal:(2*dmax)]';  % pour avoir toutes les distributions avec la même abscisse (interpolation)
LX=length(dx);
