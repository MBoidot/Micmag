function [res] = interpole(vxi,vyi,vxf)
%res est vxf. vxi et vyi ont la même longueur
% prend un vecteur x, un vecteur y, et un nouveau vecteur x sur lequel
% interpoler le y

L=length(vxf);
res=zeros(L,1);
Li=length(vxi);
for i=1:L
    x=vxf(i);
    ind=0;
    for j=1:Li
        if vxi(j)<x
            ind=j;
        end
    end
    if ind==0
        res(i)=vyi(1);
    elseif ind==Li
        res(i)=vyi(Li);
    else
        res(i)=((x-vxi(ind))*vyi(ind+1)+(vxi(ind+1)-x)*vyi(ind))/(vxi(ind+1)-vxi(ind));
    end
end

