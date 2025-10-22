% Reprend la boucle en rassemblant les données exportées en XLS par échantillon, type et photo
% Compile les angles le long de l'épaisseur 
% Prend les photos aux différentes épaisseurs et corrige avec l'angle
% Produit un gros tableau de données, une courbe d'angle et de spacing


recolte;            % les distributions sont sous une forme où la valeur i représente l'occurence entre les classes i et i+1.
traitedonnees;      % Les ditributions sont regroupées, corrigées, et recentrées pour que la valeur i représente l'occurence sur un intervalle centré sur i. 
for ech=1:length(ECHTS)
    synthese;       % Les données sont synthétisées et exportées en tableaux excel dans le sous-dossier Résultats de la manip
    dessine;        % Les figures sont créées et exportées dans le sous-dossier Résultats de la manip
end




