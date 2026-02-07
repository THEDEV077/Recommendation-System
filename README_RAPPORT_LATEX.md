# Rapport LaTeX du Système de Recommandation de Films

## 📄 Description

Ce document est un rapport technique complet en français décrivant le projet de système de recommandation de films. Il a été rédigé en LaTeX et est prêt à être compilé sur Overleaf.

## 📋 Contenu du rapport

Le rapport (`rapport.tex`) contient :

1. **Page de titre et résumé** - Vue d'ensemble du projet
2. **Introduction** - Contexte, objectifs et organisation
3. **Architecture technique** - Structure du code et stack technologique
4. **Modèles implémentés** - Description détaillée de 6 modèles :
   - Modèles classiques : Baseline, KNN, SVD, NMF
   - Modèles deep learning : NCF, AutoRec
5. **Résultats du benchmark** - Comparaison des performances (RMSE, MAE)
6. **Fonctionnalités de l'application** - Interface Streamlit
7. **Installation et déploiement** - Guide complet
8. **Conclusion et perspectives** - Limites et améliorations futures
9. **Annexes** - Code source, statistiques, guide Overleaf

## 🚀 Utilisation sur Overleaf

### Méthode 1 : Upload direct

1. Aller sur [Overleaf](https://www.overleaf.com)
2. Se connecter ou créer un compte
3. Cliquer sur "New Project" → "Upload Project"
4. Uploader le fichier `rapport.tex`
5. Cliquer sur "Recompile" pour générer le PDF

### Méthode 2 : Créer un nouveau projet

1. Créer un nouveau projet sur Overleaf
2. Copier le contenu de `rapport.tex`
3. Coller dans le fichier principal
4. Compiler avec pdfLaTeX

### Configuration recommandée

- **Compilateur** : pdfLaTeX
- **Version TeX** : TeX Live 2023 ou supérieur
- **Encodage** : UTF-8

## 📦 Packages utilisés

Tous les packages suivants sont standards et disponibles par défaut sur Overleaf :

- `inputenc`, `fontenc`, `babel` - Encodage et langue française
- `geometry` - Marges de page
- `graphicx` - Images
- `amsmath`, `amssymb` - Formules mathématiques
- `hyperref` - Liens hypertexte
- `listings` - Code source
- `xcolor` - Couleurs
- `booktabs` - Tableaux professionnels
- `float` - Positionnement des figures
- `fancyhdr` - En-têtes personnalisés
- `titlesec` - Sections personnalisées

## 📊 Contenu clé

### Tableaux de résultats

Le rapport inclut un tableau comparatif complet des performances :

| Modèle | RMSE | MAE |
|--------|------|-----|
| NCF | 0.9339 | 0.7353 |
| AutoRec | 0.9658 | 0.7663 |
| KNN | 0.9692 | 0.7583 |
| Baseline | 1.1239 | 0.9420 |

### Formules mathématiques

Toutes les formules sont correctement formatées en LaTeX :
- Similarité cosinus
- Décomposition SVD
- Architecture NCF et AutoRec
- Métriques RMSE et MAE

### Code source

Extraits de code Python avec coloration syntaxique :
- Architecture des modèles PyTorch
- Pipeline d'entraînement
- Interface Streamlit

## 🔧 Compilation locale (optionnel)

Si vous souhaitez compiler localement au lieu d'utiliser Overleaf :

```bash
# Installer LaTeX (Ubuntu/Debian)
sudo apt-get install texlive-full

# Compiler le document
pdflatex rapport.tex
pdflatex rapport.tex  # Deux fois pour les références

# Générer le PDF final
# Le fichier rapport.pdf sera créé
```

## 📝 Personnalisation

### Modifier les informations

Pour personnaliser le rapport, éditer les sections suivantes dans `rapport.tex` :

- **Titre** : Ligne `\title{...}`
- **Auteur** : Ligne `\author{...}`
- **Date** : Ligne `\date{...}` (ou garder `\today` pour la date automatique)

### Ajouter du contenu

Pour ajouter une nouvelle section :

```latex
\section{Titre de la section}
Votre contenu ici...

\subsection{Sous-section}
Plus de détails...
```

### Ajouter des images

```latex
\begin{figure}[H]
    \centering
    \includegraphics[width=0.8\textwidth]{image.png}
    \caption{Description de l'image}
    \label{fig:mon_image}
\end{figure}
```

## 📖 Structure du document

```
rapport.tex
├── Préambule (packages, configuration)
├── Page de titre
├── Table des matières
├── Résumé (abstract)
├── 7 sections principales
├── Bibliographie
└── Annexes
```

## 🎯 Points forts du rapport

- ✅ **Complet** : Couvre tous les aspects du projet (50+ pages)
- ✅ **Professionnel** : Formatage LaTeX de haute qualité
- ✅ **Bilingue-ready** : Entièrement en français comme demandé
- ✅ **Technique** : Formules mathématiques détaillées
- ✅ **Illustré** : Tableaux, code source, équations
- ✅ **Académique** : Bibliographie et citations
- ✅ **Pratique** : Guide d'installation et déploiement

## 📚 Références

Le rapport inclut une bibliographie complète avec :
- Article original NCF (He et al., 2017)
- Article original AutoRec (Sedhain et al., 2015)
- Dataset MovieLens (Harper & Konstan, 2015)
- Documentation PyTorch, Scikit-learn, Streamlit

## 🆘 Aide et support

### Problèmes de compilation

Si vous rencontrez des erreurs :

1. Vérifier que le compilateur est bien pdfLaTeX
2. S'assurer que tous les packages sont installés
3. Compiler deux fois (pour les références)
4. Sur Overleaf, utiliser le bouton "Clear cached files" si nécessaire

### Personnalisation avancée

Pour modifier les couleurs, marges, polices, etc., consulter :
- [Overleaf Documentation](https://www.overleaf.com/learn)
- [LaTeX Wikibook](https://en.wikibooks.org/wiki/LaTeX)

## 📧 Contact

Pour toute question sur le rapport ou le projet, consulter le README principal du projet.

---

**Note** : Ce rapport est conçu pour Overleaf mais peut être compilé avec n'importe quelle distribution LaTeX complète (TeX Live, MiKTeX, etc.)
