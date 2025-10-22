% Balayage de l'image : deuxième passage pour trouver le plus grand disque centré en x,y

for x=plagex
    for y=plagey
        angles=(Angles+90)*pi/180+pi*rand;
        if N(x,y)==0
            stop=0;
            for ang=1:nbangles
                angle=angles(ang);
                v=-ones(1,llex);
                for i=1:llex
                    ec=i-fl*lex-1;
                    ecx=round(cos(angle)*ec/fl);
                    ecy=round(sin(angle)*ec/fl);
                    if x+ecx>0 && x+ecx<=hauteur && y+ecy>0 && y+ecy<=largeur
                        v(i)=N(x+ecx,y+ecy);
                    end
                end
                vtest=(v<1);            % vtest=1 si je suis dans la phase magnétique ou hors de l'image, zéro si je suis dans la Nd rich
                vtest2=2*(v>=0)-1;       % pour les bords : v est négatif hors de l'image, positif dans l'image
                i1=1;
                rd=1; % cherche le bord à droite
                while (i1<=fl*lex && rd==1)
                    rd=min(rd,min(vtest(fl*lex+1+i1),vtest2(fl*lex+1+i1)));
                    i1=i1+1;
                end
                i2=1;
                rg=1;  % cherche le bord à gauche
                while (i2<=fl*lex && rg==1)
                    rg=min(rg,min(vtest(fl*lex+1-i2),vtest2(fl*lex+1-i2)));
                    i2=i2+1;
                end
                if rd>=0 && rg>=0
                    %d=min(d,(i2+i1-1)/echelle);
                    d(ang)=min(i1-1,i2-2)/echelle/fl;
                else
                    stop=stop+1;  % sert à ne pas compter les bords
                end
            end
            if stop<bords*nbangles % si stop est supérieur, alors on laisse la valeur négative de l'initialisation
                ds=sort(d);
                MM(x,y)=mean(ds(1:nbamoy));
            end
        end
    end
end

% On dispose maintenant de trois matrices : l'originale M, la binarisée N et la matrice des disques maximaux MM

% Balayage de l'image : deuxième passage pour trouver le plus grand disque centré qui contient x,y ; on utilise la matrice MM
for x=plagex
    for y=plagey
        angles=(Angles+90)*pi/180+pi*rand;
        if N(x,y)==0
            d=-dmax/2;
            for ang=1:nbangles
                angle=angles(ang);
                pasbloqued=1; % vaut 1 tant qu'on ne bute pas sur du Ndrich ; gauche et droite
                pasbloqueg=1; % vaut 1 tant qu'on ne bute pas sur du Ndrich ; gauche et droite
                for ec=1:lex
                    ecx=round(cos(angle)*ec);
                    ecy=round(sin(angle)*ec);
                    if x+ecx>0 && x+ecx<=hauteur && y+ecy>0 && y+ecy<=largeur
                        if pasbloqued
                            if N(x+ecx,y+ecy)==1
                                pasbloqued=0;
                            elseif ec/echelle < MM(x+ecx,y+ecy)
                                d=max(d,2*MM(x+ecx,y+ecy));
                            end
                        end
                    end
                    if x-ecx>0 && x-ecx<=hauteur && y-ecy>0 && y-ecy<=largeur
                        if pasbloqueg
                            if N(x-ecx,y-ecy)==1
                                pasbloqueg=0;
                            elseif MM(x-ecx,y-ecy)>0 && ec/echelle < MM(x-ecx,y-ecy)
                                d=max(d,2*MM(x-ecx,y-ecy));
                            end
                        end
                    end
                end
                D(x,y)=d;
             end
        end
    end
end

lrg=largeur_moyennage;

for x=plagex
    for y=plagey
        if D(x,y)>0
        Dloc=D(max(1,(x-lrg)):min(hauteur,(x+lrg)),max(1,(y-lrg)):min(largeur,(y+lrg)));
        [a,b]=size(Dloc);
        v=0;
        i=1;
        for k=1:a
            for l=1:b
                if Dloc(k,l)>0
                    v(i)=Dloc(k,l);
                    i=i+1;
                end
            end
        end
        DM(x,y)=mean(mean(v));
        end
    end
end





