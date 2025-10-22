% On parcourt les pixels allumés de M pour déterminer la direction dominante en chaque point, sans prendre en compte les trop isolés.
% Ensuite on calcule la distance en prenant en compte les points isolés restés dans la matrice MM.

A=-120*ones(hauteur,largeur); 

angleverti=[-90:pas:-45,45:pas:90]; 
nombredanglesverti=length(angleverti); 
vverti=zeros(size(angleverti));
anglehoriz=[-45:pas:45];
nombredangleshoriz=length(anglehoriz); 
vhoriz=zeros(size(anglehoriz));

% Précalculs
an=1:nbangles;
Angles=-90+an*180/nbangles;
llex=2*fl*lex+1;


% Détermination de l'angle

for i=plagex
    for j=plagey
        if N(i,j)  % calcul pour les pixels allumés : trouver l'angle -> A
            for k=1:nombredanglesverti % boucle pour les angles plutôt verticaux
                angle=angleverti(k);
                total=0;
                allumes=0;
                for m=i-envergure:i+envergure
                    if m>0&&m<=hauteur
                        jj=round(j+sign(angle)*(m-i)*cos(angle*pi/180)); 
                        if jj>1 && jj<=largeur-1
                            total=total+3;
                            allumes=allumes+sum(N(m,jj-1:jj+1));
                        end
                    end
                end
                vverti(k)=allumes/total;
            end
            for k=1:nombredangleshoriz
                angle=anglehoriz(k);
                total=0;
                allumes=0;
                for m=j-envergure:j+envergure
                    if m>0 && m<=largeur
                        ii=round(i+(m-j)*sin(angle*pi/180));
                        if ii>1 && ii<=hauteur-1
                            total=total+3;
                            allumes=allumes+sum(N(ii-1:ii+1,m));
                        end
                    end
                end
                vhoriz(k)=allumes/total;
            end
            [valeurverti,indicemaxverti]=max(vverti);  
            [valeurhoriz,indicemaxhoriz]=max(vhoriz);
            if valeurverti>valeurhoriz  % choisit si l'angle effectif est plutôt vertical ou horizontal
                A(i,j)=angleverti(indicemaxverti);
            else
                A(i,j)=anglehoriz(indicemaxhoriz);
            end
        end
        % à enlever
        %[i,j]
    end
end

DM=A;
Classes=[1:hauteur]';
distri=zeros(hauteur,1);
releve=0;
for i=1:hauteur
    t=1;
    for j=1:largeur
        if N(i,j)
            releve(t)=A(i,j);
            t=t+1;
        end
        distri(i)=mean(90-abs(releve));
    end
end

% Classes correspond à la profondeur en pixels en partant du bas
% distri correspond à la moyenne de la valeur absolue de l'angle par
% rapport à la verticale

                


            
