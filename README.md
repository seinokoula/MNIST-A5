# Classification de Chiffres Manuscrits - MNIST

Une application web interactive pour la reconnaissance de chiffres manuscrits utilisant un réseau de neurones convolutionnel (CNN) entraîné sur le dataset MNIST.

## 🎯 Fonctionnalités

- **Interface de dessin interactive** : Dessinez des chiffres directement sur le canvas avec la souris ou le tactile
- **Prédiction en temps réel** : Classification instantanée des chiffres dessinés (0-9)
- **Visualisation des probabilités** : Affichage graphique de la confiance pour chaque chiffre
- **Modèle optimisé** : Réseau de neurones convolutionnel avec normalisation par batch et dropout

## 🚀 Utilisation

1. Ouvrez `index.html` dans votre navigateur web
2. Dessinez un chiffre sur le canvas noir
3. Cliquez sur "Prédire" pour obtenir la classification
4. Utilisez "Effacer" pour recommencer

## 🏗️ Architecture du Modèle

Le modèle utilise une architecture CNN avec :
- 2 blocs convolutionnels (32 et 64 filtres)
- Normalisation par batch et MaxPooling
- Couches fully connected avec dropout (0.2)
- Sortie : 10 classes (chiffres 0-9)

## 📁 Structure du Projet

```
MNIST-a5/
├── index.html          # Interface utilisateur
├── script.js           # Logique de l'application
├── project.ipynb       # Entraînement du modèle
├── mnist_model.onnx    # Modèle exporté pour le web
├── mnist_model.pth     # Modèle PyTorch
└── data/MNIST/         # Dataset MNIST
```

## 🛠️ Technologies

- **Frontend** : HTML5 Canvas, JavaScript ES6
- **Machine Learning** : PyTorch, ONNX Runtime Web
- **Dataset** : MNIST (60k images d'entraînement, 10k de test)

## 📊 Performance

Le modèle atteint une précision de **~98%** sur le dataset de test MNIST après 5 époques d'entraînement.

## 🔧 Développement

Pour réentraîner le modèle :
1. Installez les dépendances : `torch`, `torchvision`, `onnx`
2. Exécutez `project.ipynb` pour l'entraînement
3. Le modèle sera exporté automatiquement au format ONNX

## 📝 Notes Techniques

- Le canvas utilise une résolution de 280x280px, redimensionnée à 28x28px pour l'inférence
- Normalisation des données : moyenne=0.1307, écart-type=0.3081 (standards MNIST)
- Support tactile pour les appareils mobiles