MM=-(dmax/2)*ones(hauteur,largeur);   % distance provisoire  
D=-(dmax/2)*ones(hauteur,largeur);   % distance définitive  
N=ones(hauteur,largeur);    % image binarisée      
DM=-(dmax/2)*ones(hauteur,largeur);  % pour le moyennage 

% Précalculs
an=1:nbangles;
Angles=-90+an*180/nbangles;
llex=2*fl*lex+1;

% Balayage de l'image : premier passage pour binariser

for x=plagex
    for y=plagey
        angles=(Angles+90)*pi/180+pi*rand;
        Mloc=M(max(1,(x-lex)):min(hauteur,(x+lex)),max(1,(y-lex)):min(largeur,(y+lex)));
        [a,b]=size(Mloc);
        T=a*b;
        mi=min(min(Mloc));
        ma=max(max(Mloc));
        w=zeros(1,dec);
        W=zeros(1,dec+1);
        es=(ma-mi)/dec;
        i2=1;
        for i=1:dec
            w(i)=sum(sum((Mloc<mi+i*es).*(Mloc>=mi+(i-1)*es)));        % distribution d'intensités
            W(i+1)=sum(w(1:i))/T;                      % distribution cumulée
            if W(i)<q2
                i2=i;
            end
        end
        m2=((q2-W(i2))*(mi+i2*es)+(W(i2+1)-q2)*(mi+i2*es-es))/(W(i2+1)-W(i2));        % valeur au-delà duquel on est en Nd-rich dans cette région de la photo      
        % d=dmax*ones(1,nbangles);
        if M(x,y)<m2 || (x>2 && x<hauteur-1 && y>2 && y<largeur-1 && sum(sum((M(x-2:x+2,y-2:y+2)>m2)))<5) 
        % càd si je suis dans la phase magnétique, binarisation locale OU un point isolé, le critère étant : moins 4 pixels allumés dans un voisinage de 5x5=25, en utilisant la binarisation locale   %&& M(x,y)>m1-bruit/2 && 
            N(x,y)=0;  % si je suis dans la phase magnétique je continue et je mets le pixel de N à 0. Donc jusque là on a binarisé l'image.
        end
    end
end
