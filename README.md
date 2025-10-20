# 🧬 Predictive Ranking of Protein Multi-Mutants  
### A Hybrid Computational and Machine Learning Approach  
*University of Florida · CAP 5510: Bioinformatics Project*

---

## 📘 Overview
Protein engineering faces a **combinatorial explosion** in the number of possible amino acid variants. Even double mutants create \(19^2 = 361\) combinations, while triple mutants yield \(19^3 = 6,859\). Testing all variants experimentally is infeasible.  

This project proposes a **hybrid computational pipeline** that combines **machine learning** and **search algorithms** to efficiently predict and rank protein multi-mutants based on their predicted stability (ΔΔG).  

We first train an ML model on experimental single-mutation ΔΔG data from **ProTherm/ThermoMutDB**, using hybrid **sequence, structure, and physics-based features**.  
The trained model then serves as a **fast surrogate fitness function** for search algorithms (Genetic Algorithm and Monte Carlo baseline) exploring double (n=2) and triple (n=3) mutants.  
Top-ranked variants are further validated using **physics-based tools** (FoldX / Rosetta).

---

## 🧩 Project Objectives
- Train an ML regression model to predict protein stability changes (ΔΔG) from hybrid bioinformatics features.  
- Integrate the ML model as a surrogate scoring function inside **Genetic Algorithm (GA)** and **Monte Carlo (MC)** search frameworks.  
- Evaluate and rank multi-mutants (n=2, n=3) for stabilizing potential.  
- Validate top candidates using **FoldX** or **Rosetta** energy minimization.  
- Compare performance (runtime, convergence, accuracy) between GA and MC approaches.  

---

## 🧠 Methodology Pipeline
Experimental ΔΔG Data (ProTherm/ThermoMutDB)
│
▼
Feature Engineering (Sequence + Structure + Physics)
│
▼
Regression Model (RF / GBoost → Predict ΔΔG)
│
▼
Search Algorithms (GA vs Monte Carlo)
│
▼
Validation with FoldX / Rosetta
│
▼
Final Ranked List of Stabilizing Variants

---

## ⚙️ Tools & Libraries
| Category | Tools / Libraries |
|-----------|------------------|
| Databases | ProTherm, ThermoMutDB, Protein Data Bank (PDB) |
| Structure | DSSP, NACCESS, FoldX, Rosetta |
| Machine Learning | Scikit-learn, XGBoost, Pandas, NumPy |
| Search Algorithms | DEAP (Genetic Algorithms), Random baseline |
| Visualization | Matplotlib, Seaborn |
| Environment | Python 3.10+, Conda, Git |

---

## 📂 Repository Structure
```
protein-multimutant-ranking/
│
├── data/
│   ├── raw/              # ProTherm / PDB / DSSP / NACCESS outputs
│   └── processed/        # Feature tables
│
├── notebooks/
│   ├── 01_EDA.ipynb
│   ├── 02_ML_Model.ipynb
│   └── 03_GA_vs_MC.ipynb
│
├── src/
│   ├── feature_engineering.py
│   ├── train_model.py
│   ├── genetic_algorithm.py
│   ├── montecarlo_baseline.py
│   └── validation_foldx.py
│
├── results/
│   ├── plots/
│   ├── tables/
│   └── validation/
│
├── docs/
│   ├── proposal.pdf
│   ├── report.tex
│   └── poster.pptx
│
├── requirements.txt
├── LICENSE
└── README.md
```
---

## 🚀 Getting Started

### 1️⃣ Clone Repository
```bash
git clone https://github.com/your-username/protein-multimutant-ranking.git
cd protein-multimutant-ranking
```

### 2️⃣ Create Environment
```
conda create -n proteinML python=3.10
conda activate proteinML
pip install -r requirements.txt
```

### 3️⃣ Folder Initialization
```
mkdir -p data/raw data/processed results/plots results/validation
```

### 5️⃣ Launch Genetic Algorithm
```
python src/genetic_algorithm.py
```

## 🧮 Expected Outputs
	•	Trained ML model (model.pkl)
	•	GA vs MC comparison plots (fitness convergence, runtime)
	•	Top-ranked stabilizing variants (validated_mutants.csv)
	•	Final report with ΔΔG correlations between ML and FoldX/Rosetta.
	
📚 Key References
	•	Guerois et al., Predicting Changes in Protein Stability Upon Mutation, Nature Structural Biology (2002).
	•	Kellogg et al., Role of Conformational Sampling in Computing Mutation-Induced Changes in Protein Stability, JMB (2011).
	•	Alley et al., Unified Rational Protein Engineering with Sequence-Based Deep Representation Learning, Nature Methods (2019).

👥 Authors
	•	[Deepika Sarala Pratapa] — M.S. Applied Data Science, University of Florida - [dpratapa@ufl.edu](mailto:deepikapratapa27@gmail.com)  
	•	[Rohit Bogulla] — M.S. Computer Science, University of Florida - [rbogulla@ufl.edu](mailto:deepikapratapa27@gmail.com)  

	
🧾 License
This repository is released under the MIT License.
Please cite appropriately if you use the dataset or code for academic purposes.















