import streamlit as st

st.set_page_config(
    page_title="Structural Optimization Suite",
    page_icon="🌉",
    layout="centered"
)

st.title("🌉 Intelligent Structural Optimization Suite")
st.markdown("---")

st.markdown("""
Welcome to the Structural Optimization Suite. This tool uses advanced evolutionary algorithms to optimize the topology and member sizing of 2D steel trusses.

Please select an optimization module from the sidebar to the left:

### 1️⃣ Single-Objective Optimization
* **Algorithm:** Differential Evolution (SciPy)
* **Objective:** Minimize total structural weight.
* **Constraints:** Hard limits on maximum vertical deflection and member capacities (Yield & Euler Buckling).

### 2️⃣ Multi-Objective Optimization
* **Algorithm:** NSGA-II (Pymoo)
* **Objectives:** Simultaneously minimize weight AND maximize stiffness (minimize deflection).
* **Output:** Generates a **Pareto Front**, allowing the engineer to explore the trade-offs between a lightweight/flexible structure and a heavy/stiff structure.

---
""")

# --- DEVELOPER INFO ---
st.info("""
**Developed by:** Prathamesh Varma  
[Connect on LinkedIn](https://www.linkedin.com/in/prathameshvarma)  
**Guidance:** Prof. Vasan Arunachalam  
[Web Link](https://www.bits-pilani.ac.in/hyderabad/a-vasan/)  

""")
