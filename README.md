# Classification de symboles avec PyTorch

## 🚀 Description
Ce projet implémente un **réseau de neurones simple** en utilisant [PyTorch](https://pytorch.org/) pour effectuer une **classification binaire** entre les symboles `*` et `/`.

- Encodage **one-hot** des données :  
  - `"*"` → `[1.0, 0.0]`  
  - `"/"` → `[0.0, 1.0]`  
- Division des données en **80% entraînement / 20% test**.  
- Entraînement avec la **fonction de perte CrossEntropy** et l’optimiseur **Adam**.  
- Évaluation avec la **précision** sur l’ensemble de test.  

---

## 📂 Structure du projet
├── symbol_classifier.py # Script principal avec le modèle et l'entraînement
├── README.md # Documentation du projet
└── requirements.txt # Dépendances (PyTorch, scikit-learn, numpy)

---

## ⚙️ Installation

Cloner le dépôt :
```bash
git clone https://github.com/<ton-username>/symbol-classifier.git
cd symbol-classifier
```
Créer un environnement virtuel et installer les dépendances :
python -m venv venv
source venv/bin/activate   # sous Linux/Mac
venv\Scripts\activate      # sous Windows

pip install -r requirements.txt
🛠️ Outils et technologies

  - Python

  - PyTorch

  - NumPy

scikit-learn
📌 Améliorations possibles

   - Ajouter un dataset plus complexe avec d’autres symboles.

   - Tester différentes architectures (plus de couches, dropout).

   - Visualiser la courbe d’apprentissage (loss/accuracy par epoch).
