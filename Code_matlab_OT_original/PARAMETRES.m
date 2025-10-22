clear all; close all;

% INSTRUCTION
% ETAPE 1
% Ouvrir ce fichier, le fichier ANALYSE_IMAGES et le fichier RASSEMBLE. 
% Activer la fenêtre de PARAMETRES, mettre les bons noms et numéros. Rentrer les épaisseurs, les nombres de photos.Faire F5. C'est instantané.
% Activer la fenêtre de ANALYSE_IMAGES, faire F5. C'est cette partie qui est longue, lancer plutôt le soir et sur un PC valable. Cela crée des fichiers de données pour chaque photo

% ETAPE 2
% Vérifier dans le sous-dossier résultats que les données ont été exportées en XLS pour chaque image
% Activer la fenêtre PARAMETRES et faire F5 pour relancer MATLAB et recharger les paramètres qui se sont peut-être perdus depuis le temps
% Activer la fenêtre RASSEMBLE, faire F5. 
% Environ 1 minute plus tard les courbes apparaissent dans le dossier Résultats

% COULEE A ANALYSER

nomanip='18';                    % Pour les ruban SC, mettre le numéro de la coulée, '18' pour SC18
manip=strcat('SC',nomanip);     % 'SC' ou 'VAC' ou...
ECHTS='AB';                     % Les désignations des différents rubans échantillonnés, mis à la suite. Exple : 'ABC' si on a trois rubans A, B et C.

TYPES='RRRRRARAAAAATR_';        % Les types font 3 lettres, les juxtaposer à la suite. 
NB_PHOTOS=[3,3,3,3,6];           % par type, donc 5 valeurs par échantillon. L'ordre est celui des types à la ligne au-dessus
nb_identiques=1;                % Si les nombres de photos sont identiques pour tous les échantillons --> 1. Sinon, 0, et mettre tous les nombres dans le vecteur au-dessus
epaisseurs=[155,185];           % Une valeur par échantillon, séparer avec des virgules
rmax=15;                         % Rayon d'exploration à partir du point courant en µm ; standard : 5. Le spacing max vaut 2*rmax.
dmaxfigures=25;                 % Standard : 2*dmax. On peut mettre moins pour la partie RASSEMBLE, mais alors cela veut dire que le dmax avait été mal choisi !
bpf=[10,20;30,40;60,70;80,90];   % bornes pour les profondeurs correspondant aux types en % depuis la roue ; nbtypes-1 lignes et 2 colonnes (début/fin)

% Génération des données à partir des images

che='C:\Users\MB232649\Documents\M.Boidot Local\Informatique\Projets-info\2025-Nd_Rich_Spacing\Code_matlab_OT_original\';   % vers le dossier parent AJUSTER SI NECESSAIRE
cheM=strcat(che,manip,'\');                                                                         % vers la manip IL FAUT UN DOSSIER AVEC CE NOM ET UN SOUS-DOSSIER RESULTATS

% PARAMETRAGE du traitement d'images

dmax=rmax;
echelle=5.2;                % Rapport d'échelle pour les photos de face : 1 µm pour echelle pixels ; 5.2 pour grossissement 1000x
echelleTR=2.6;              % Rapport d'échelle pour les photos de face : 1 µm pour echelle pixels ; 2.6 pour grossissement 500x
lex=round(dmax*5.2);        % longueur d'exploration à partir du point courant en pixels
fl=2;         % 3           % facteur multiplicateur dans l'exploration pour meilleure granularité
nbangles=40;   %70          % nombre d'angles entre 0 et 180° pour trouver le segment le plus court
nbamoy=4;    %4             % nombre d'angles moyennés pour calculer d (on prend les plus petits valeurs, mais pas QUE la plus petite pour éviter les traits moches)
dec=30;       %20           % nombre de classes calculées dans les vecteurs v pour trouver les quantiles
q2=0.88;        %0.85       % quantile supérieur pour l'appartenance à Nd-rich (évaluée localement)
largeur_moyennage=3; %3     % largeur de la zone moyennée à la fin du calcul pour le rendu graphique
pas=3;              %3      % Pas angulaire lors de l'analyse d'angle sur les tranches, en degrés
envergure=15;       %20     % Nombre de pixels à droite et à gauche du point pris en compte pour le calcul de l'angle
pasdistri=2/echelle;        % pas de la distribution granulométrique rendue dans les fichiers de chaque image. Correspond aux pixels.
pasdistrifinal=0.2;   %0.2  % pas des distribution granulométrique synthétique rendues à la fin. 
bords=0.6;          %0.6    % pour l'élimination des bords. Si la proportion d'intersections avec un bord excède ce nombre, on laisse la zone éteinte.

nbtypes=floor(length(TYPES)/3);

% Gestion du nombre de photos
NBPHOTOS=[]
if nb_identiques
    for i=1:(length(ECHTS))
        NBPHOTOS=[NBPHOTOS,NB_PHOTOS];
    end
else 
    NBPHOTOS=NB_PHOTOS;
end



