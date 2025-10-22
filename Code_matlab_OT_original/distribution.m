classes=0:pasdistri:(2*dmax+(pasdistri*0.9)); 
Classes=classes';
L=length(classes);
districum=zeros(1,L);
distri=zeros(1,L-1);

for i=1:hauteur
    for j=1:largeur
        districum=districum+(classes>DM(i,j));
    end
end

for i=1:L-1
    distri(i)=districum(i+1)-districum(i);
end

distri=([distri,0])'/sum(distri);








