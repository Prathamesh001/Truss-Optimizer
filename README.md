# Differential Evolution Truss Optimizer 🏗️

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://truss-optimizer.streamlit.app/)

A web-based structural engineering tool designed to automate the optimization of steel roof trusses. This application minimizes the total structural weight while strictly adhering to safety and design constraints.

Developed as a course project under the guidance of Prof. Vasan Arunachalam at BITS Pilani.

## 🚀 Live Application
**Try the interactive optimizer here:** [Insert your Streamlit App URL]

## 🧠 How It Works
The tool leverages a metaheuristic optimization algorithm to simultaneously solve for the optimal geometric height and discrete cross-sectional member sizes.

* **Finite Element Engine:** Uses **OpenSeesPy** to compute self-weight, extract axial forces, and determine maximum nodal deflections.
* **Optimization Algorithm:** Uses **SciPy's Differential Evolution** to navigate the highly non-linear design space of continuous coordinates and discrete steel sections.
* **Design Constraints:** * Tensile yielding capacity ($P_{act} \le A \cdot F_y$)
  * Euler buckling resistance for compression members
  * Maximum span deflection limit ($Span / 250$)

## ✨ Features
* **Live Visualizer:** Watch the evolutionary algorithm adapt the truss topology and member sizing in real-time as it converges to an optimal design.
* **Dynamic Loading:** Automatically lumps member self-weight and combines it with user-defined nodal loads.
* **Pre-built Topologies:** Includes benchmark presets like a Warren Truss and a 6-Bay Pratt Bridge to instantly test the algorithm's capabilities.
* **Data Export:** Download optimized member properties, axial forces, and nodal deflections as CSV files for reporting.

## 🛠️ Tech Stack
* **Python 3.8**
* **Streamlit** (Frontend GUI & Cloud Deployment)
* **OpenSeesPy** (Structural Analysis)
* **SciPy** (Optimization)
* **Matplotlib & Pandas** (Data Processing & Visualization)

## 👨‍💻 Author
**Prathamesh Varma** M.E. Structural Engineering | BITS Pilani – Hyderabad Campus  
🔗 [LinkedIn Profile](https://www.linkedin.com/in/prathameshvarma/)
