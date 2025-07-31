# Classification MNIST - Web

Ce projet implémente une IA de classification de chiffres manuscrits MNIST avec une interface web interactive.

## Structure du projet

- `train_mnist.py` - Script d'entraînement du modèle PyTorch et export ONNX
- `index.html` - Page web interactive
- `script.js` - Logique JavaScript pour le canvas et l'inférence ONNX
- `requirements.txt` - Dépendances Python

## Installation et utilisation

### 1. Entraînement du modèle

```bash
# Installer les dépendances
pip install -r requirements.txt

# Entraîner le modèle et l'exporter en ONNX
python train_mnist.py
```

Cette étape génère:
- `mnist_model.pth` - Modèle PyTorch sauvegardé
- `mnist_model.onnx` - Modèle exporté en format ONNX

### 2. Interface web

Servir les fichiers avec un serveur web local:

```bash
# Avec Python
python -m http.server 8000

# Avec Node.js (si disponible)
npx serve .
```

Ouvrir http://localhost:8000 dans votre navigateur.

## Fonctionnalités

- ✅ Canvas interactif pour dessiner des chiffres
- ✅ Inférence en temps réel côté client avec ONNX Runtime Web
- ✅ Affichage des probabilités pour chaque chiffre (0-9)
- ✅ Interface responsive et intuitive
- ✅ Support tactile pour appareils mobiles

## Architecture

1. **Entraînement**: PyTorch avec un réseau CNN simple
2. **Export**: Conversion au format ONNX pour l'interopérabilité
3. **Inférence web**: ONNX Runtime Web pour l'exécution côté client
4. **Interface**: Canvas HTML5 pour la saisie utilisateur