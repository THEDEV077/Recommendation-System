# Guide Rapide - Rapport LaTeX Overleaf

## 🎯 Objectif

Ce guide vous explique comment compiler le rapport technique du projet de recommandation de films sur Overleaf en 5 minutes.

## 📋 Prérequis

- Un compte Overleaf (gratuit) : https://www.overleaf.com
- Le fichier `rapport.tex` du projet

## 🚀 Étapes Rapides

### 1. Créer un compte Overleaf (si nécessaire)

1. Aller sur https://www.overleaf.com
2. Cliquer sur "Register" (S'inscrire)
3. Créer un compte gratuit

### 2. Uploader le rapport

**Option A - Upload de fichier (Recommandé)**

1. Se connecter à Overleaf
2. Cliquer sur **"New Project"** (Nouveau Projet)
3. Sélectionner **"Upload Project"** (Charger un Projet)
4. Glisser-déposer le fichier `rapport.tex`
5. Le projet sera créé automatiquement

**Option B - Copier-Coller**

1. Créer un nouveau projet : **"New Project" → "Blank Project"**
2. Nommer le projet : "Système de Recommandation - Rapport"
3. Ouvrir le fichier `main.tex`
4. Copier tout le contenu de `rapport.tex`
5. Coller dans l'éditeur Overleaf

### 3. Compiler le document

1. Vérifier que le compilateur est **"pdfLaTeX"** (en haut à gauche)
   - Cliquer sur "Menu" → Compiler : pdfLaTeX
2. Cliquer sur le bouton vert **"Recompile"**
3. Attendre 30-60 secondes
4. Le PDF apparaît à droite !

### 4. Télécharger le PDF

1. Cliquer sur le bouton **"Download PDF"** (icône de téléchargement)
2. Le fichier `rapport.pdf` est sauvegardé sur votre ordinateur

## ✅ Vérifications

### Le rapport contient :

- ✅ Page de titre professionnelle
- ✅ Table des matières
- ✅ Résumé (Abstract)
- ✅ 7 sections principales (60+ pages)
- ✅ Tableaux de résultats
- ✅ Formules mathématiques
- ✅ Code source coloré
- ✅ Bibliographie
- ✅ Annexes

### Nombre de pages

Le rapport complet fait environ **60-70 pages** selon la version LaTeX.

## 🔧 Résolution de problèmes

### Erreur de compilation

**Problème** : Message d'erreur rouge

**Solution** :
1. Cliquer sur "Menu" → "Clear cached files"
2. Recompiler
3. Si l'erreur persiste, vérifier que tous les packages sont à jour

### Caractères français manquants

**Problème** : Les accents n'apparaissent pas

**Solution** :
- Vérifier que le compilateur est **pdfLaTeX** (pas XeLaTeX ou LuaLaTeX)
- Le package `babel` avec option `french` est déjà configuré

### Compilation lente

**Problème** : La compilation prend plus de 2 minutes

**Solution** :
- C'est normal pour un document de 60+ pages
- Attendre patiemment
- Sur Overleaf gratuit, la compilation peut être limitée

## 🎨 Personnalisation (Optionnel)

### Changer le titre

Trouver la ligne (vers ligne 75) :

```latex
\title{
    \Huge \textbf{Système de Recommandation de Films} \\
    ...
}
```

Et modifier le texte.

### Changer l'auteur

Trouver la ligne (vers ligne 84) :

```latex
\author{
    Votre Nom \\
    Votre Institution
}
```

### Ajouter votre logo

1. Uploader votre logo dans Overleaf (glisser-déposer)
2. Ajouter dans le titre :

```latex
\includegraphics[width=3cm]{logo.png}
```

### Modifier les couleurs

Dans le préambule (lignes 30-35), modifier :

```latex
\hypersetup{
    linkcolor=red,      % Changer 'blue' en 'red'
    urlcolor=purple,    % etc.
}
```

## 📱 Partager le rapport

### Partager le lien Overleaf

1. Cliquer sur "Share" en haut à droite
2. Activer "Turn on link sharing"
3. Copier le lien et l'envoyer

### Exporter en Word (si nécessaire)

**Note** : LaTeX → Word peut perdre le formatage

1. Télécharger le PDF
2. Utiliser un convertisseur en ligne : https://www.adobe.com/acrobat/online/pdf-to-word.html

## 📊 Structure du rapport

```
Page 1      : Page de titre
Page 2      : Table des matières
Page 3      : Résumé (Abstract)
Pages 4-10  : Introduction et Architecture
Pages 11-30 : Modèles (KNN, SVD, NCF, AutoRec)
Pages 31-40 : Résultats et Benchmarks
Pages 41-50 : Application et Fonctionnalités
Pages 51-60 : Installation, Conclusion
Pages 61+   : Bibliographie et Annexes
```

## 💡 Conseils

### Pour une présentation

- Extraire les sections importantes en diapositives
- Utiliser le package Beamer pour créer un diaporama LaTeX

### Pour un mémoire

- Le format actuel est parfait pour un rapport technique ou mémoire
- Ajouter une page de garde institutionnelle si nécessaire

### Pour publication

- Le rapport respecte les normes académiques
- Bibliographie formatée correctement
- Prêt pour soumission

## 🆘 Aide supplémentaire

### Documentation Overleaf

- Guide officiel : https://www.overleaf.com/learn
- Tutoriels vidéo : https://www.youtube.com/c/Overleaf

### Communauté

- Forum Overleaf : https://www.overleaf.com/help
- Stack Exchange (LaTeX) : https://tex.stackexchange.com

### Contact

Pour des questions spécifiques au contenu du rapport, consulter le README principal du projet.

---

## 🎓 Sommaire du contenu technique

### Modèles décrits

1. **Baseline** - Moyenne simple
2. **KNN** - K-Nearest Neighbors (Item-Based)
3. **SVD** - Singular Value Decomposition
4. **NMF** - Non-negative Matrix Factorization
5. **NCF** - Neural Collaborative Filtering (MEILLEUR)
6. **AutoRec** - Autoencoder Recommender

### Résultats clés

- **Meilleur modèle** : NCF avec RMSE = 0.9339
- **Dataset** : MovieLens 100k (100,000 notes)
- **Technologies** : Python, PyTorch, Streamlit
- **Performance** : 17% meilleur que la baseline

### Formules mathématiques

Le rapport inclut toutes les formules détaillées :
- Similarité cosinus
- Factorisation matricielle
- Architecture des réseaux de neurones
- Métriques d'évaluation (RMSE, MAE)

---

**Bon courage avec votre rapport ! 🎉**
