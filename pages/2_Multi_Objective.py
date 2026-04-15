import streamlit as st
import pandas as pd
import numpy as np
import openseespy.opensees as ops
import math
import matplotlib.pyplot as plt

from pymoo.core.problem import ElementwiseProblem
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.optimize import minimize
from pymoo.core.callback import Callback
from pymoo.termination import get_termination

st.set_page_config(page_title="MOO Truss Optimizer", layout="wide")
st.title("Multi-Objective Truss Optimizer (NSGA-II)")
st.markdown("Optimizing for **Minimum Weight** and **Maximum Stiffness** simultaneously.")

# --- 0. SESSION STATE & PRESETS ---
def clear_editor_keys():
    for key in ['sections_editor', 'nodes_editor', 'elements_editor', 'loads_editor', 'pareto_X', 'pareto_F']:
        if key in st.session_state:
            del st.session_state[key]


def load_preset_10_bar():
    clear_editor_keys()
    # Paper parameters: Areas from 0.1 to 40 in^2. 
    # Since your app uses discrete catalogs, we generate 40 evenly spaced sections.
    areas_in2 = np.linspace(0.1, 40.0, 40)
    areas_m2 = areas_in2 * 0.00064516 # Convert in^2 to m^2
    inertias_m4 = (areas_m2**2) / 12 # Approximate I for penalty calculation
    
    st.session_state.sections_data = pd.DataFrame({
        "Section_ID": list(range(40)),
        "Area_m2": areas_m2,
        "Inertia_m4": inertias_m4
    })
    
    # Paper Geometry: 360 inch (9.144 m) square bays
    st.session_state.nodes_data = pd.DataFrame({
        "Node_ID": [1, 2, 3, 4, 5, 6], 
        "X_m": [18.288, 18.288, 9.144, 9.144, 0.0, 0.0], 
        "Y_m": [9.144, 0.0, 9.144, 0.0, 9.144, 0.0], 
        "Support_X": [0, 0, 0, 0, 1, 1], # Nodes 5 and 6 are pinned to the wall
        "Support_Y": [0, 0, 0, 0, 1, 1]
    })
    
    st.session_state.elements_data = pd.DataFrame({
        "Element_ID": list(range(1, 11)), 
        "Start_Node": [5, 3, 6, 4, 3, 1, 5, 6, 3, 4], 
        "End_Node":   [3, 1, 4, 2, 4, 2, 4, 3, 2, 1]
    })
    
    # Paper Load: 100 kips downward at nodes 2 and 4. (100 kips = 444,822 Newtons)
    st.session_state.loads_data = pd.DataFrame({
        "Node_ID": [2, 4], 
        "Load_X_N": [0.0, 0.0], 
        "Load_Y_N": [-444822.0, -444822.0]
    })
def load_preset_king_post():
    clear_editor_keys()
    st.session_state.sections_data = pd.DataFrame({"Section_ID": [0, 1, 2, 3, 4], "Area_m2": [0.001, 0.002, 0.004, 0.006, 0.010], "Inertia_m4": [0.01, 0.02, 0.03, 0.04, 0.05]})
    st.session_state.nodes_data = pd.DataFrame({"Node_ID": [1, 2, 3, 4], "X_m": [0.0, 5.0, 10.0, 5.0], "Y_m": [0.0, 2.0, 0.0, 0.0], "Support_X": [1, 0, 0, 0], "Support_Y": [1, 0, 1, 0]})
    st.session_state.elements_data = pd.DataFrame({"Element_ID": [1, 2, 3, 4, 5], "Start_Node": [1, 4, 1, 2, 4], "End_Node": [4, 3, 2, 3, 2]})
    st.session_state.loads_data = pd.DataFrame({"Node_ID": [2, 4], "Load_X_N": [0.0, 0.0], "Load_Y_N": [-60000.0, -20000.0]})

def load_preset_howe():
    clear_editor_keys()
    st.session_state.sections_data = pd.DataFrame({"Section_ID": [0, 1, 2, 3], "Area_m2": [0.001, 0.003, 0.005, 0.008], "Inertia_m4": [0.01, 0.02, 0.03, 0.04]})
    st.session_state.nodes_data = pd.DataFrame({
        "Node_ID": [1, 2, 3, 4, 5, 6, 7, 8], 
        "X_m": [0.0, 4.0, 8.0, 12.0, 16.0, 4.0, 8.0, 12.0], 
        "Y_m": [0.0, 0.0, 0.0, 0.0, 0.0, 3.0, 3.0, 3.0], 
        "Support_X": [1, 0, 0, 0, 0, 0, 0, 0], 
        "Support_Y": [1, 0, 0, 0, 1, 0, 0, 0]
    })
    st.session_state.elements_data = pd.DataFrame({
        "Element_ID": list(range(1, 16)), 
        "Start_Node": [1, 2, 3, 4, 1, 6, 7, 8, 2, 3, 4, 2, 6, 4, 8], 
        "End_Node": [2, 3, 4, 5, 6, 7, 8, 5, 6, 7, 8, 7, 3, 7, 3]
    })
    st.session_state.loads_data = pd.DataFrame({"Node_ID": [2, 3, 4], "Load_X_N": [0.0, 0.0, 0.0], "Load_Y_N": [-25000.0, -40000.0, -25000.0]})

def load_preset_fink():
    clear_editor_keys()
    st.session_state.sections_data = pd.DataFrame({"Section_ID": [0, 1, 2], "Area_m2": [0.001, 0.0025, 0.005], "Inertia_m4": [0.01, 0.02, 0.03]})
    st.session_state.nodes_data = pd.DataFrame({
        "Node_ID": [1, 2, 3, 4, 5, 6, 7], 
        "X_m": [0.0, 3.0, 6.0, 9.0, 12.0, 3.0, 9.0], 
        "Y_m": [0.0, 0.0, 0.0, 0.0, 0.0, 1.5, 1.5], 
        "Support_X": [1, 0, 0, 0, 0, 0, 0], 
        "Support_Y": [1, 0, 0, 0, 1, 0, 0]
    })
    st.session_state.elements_data = pd.DataFrame({
        "Element_ID": list(range(1, 12)), 
        "Start_Node": [1, 2, 3, 4, 1, 6, 3, 7, 5, 2, 4], 
        "End_Node": [2, 3, 4, 5, 6, 3, 7, 5, 7, 6, 7]
    })
    st.session_state.loads_data = pd.DataFrame({"Node_ID": [6, 3, 7], "Load_X_N": [0.0, 0.0, 0.0], "Load_Y_N": [-10000.0, -15000.0, -10000.0]})

if 'nodes_data' not in st.session_state:
    load_preset_king_post()

# --- 1. UI INPUTS ---
st.sidebar.header("Library Presets")
if st.sidebar.button("10-Bar Benchmark"):
    load_preset_10_bar()
    st.rerun()
if st.sidebar.button("King Post"):
    load_preset_king_post()
    st.rerun()
if st.sidebar.button("Howe Truss"):
    load_preset_howe()
    st.rerun()
if st.sidebar.button("Fink Roof Truss"):
    load_preset_fink()
    st.rerun()

st.sidebar.markdown("---")
st.sidebar.header("Optimization Bounds & Material")
max_height = st.sidebar.number_input("Maximum Allowable Height (m)", value=4.0, step=0.5)
min_height = st.sidebar.number_input("Minimum Allowable Height (m)", value=1.0, step=0.5)

st.sidebar.markdown("---")
E = st.sidebar.number_input("Young's Modulus, E (Pa)", value=2e11, format="%e")
fy = st.sidebar.number_input("Yield Strength, Fy (Pa)", value=250e6, format="%e")
density = st.sidebar.number_input("Density (kg/m3)", value=7850.0)
gravity = 9.81 

# --- DEVELOPER INFO ---
st.sidebar.markdown("---")
st.sidebar.subheader("About the Developer")
st.sidebar.markdown("**Varma**\nM.E. Structural Engineering\nID: 2025H1430015H")
st.sidebar.markdown("🔗 [LinkedIn](https://www.linkedin.com/in/prathameshvarma)")
st.sidebar.markdown("Under the Guidance of\nProf. [Vasan Arunachalam](https://www.bits-pilani.ac.in/hyderabad/a-vasan/)")

col1, col2 = st.columns(2)

with col1:
    st.subheader("1. Available Sections")
    sections_df = st.data_editor(st.session_state.sections_data, num_rows="dynamic", key="sections_editor")
    st.subheader("2. Node Coordinates")
    nodes_df = st.data_editor(st.session_state.nodes_data, num_rows="dynamic", key="nodes_editor")

with col2:
    st.subheader("3. Element Connectivity")
    elements_df = st.data_editor(st.session_state.elements_data, num_rows="dynamic", key="elements_editor")
    st.subheader("4. Nodal Loading (External)")
    loads_df = st.data_editor(st.session_state.loads_data, num_rows="dynamic", key="loads_editor")

actual_span = nodes_df["X_m"].max() - nodes_df["X_m"].min()
initial_peak_y = nodes_df["Y_m"].max() if nodes_df["Y_m"].max() != 0 else 1e-9 
deflection_limit = actual_span / 250.0

# --- HELPER FUNCTIONS ---
def get_total_nodal_loads(current_H, sec_indices=None):
    y_scale = current_H / initial_peak_y
    nodal_loads = {int(n): [0.0, 0.0] for n in nodes_df["Node_ID"]}
    for _, row in loads_df.iterrows():
        n_id = int(row["Node_ID"])
        if n_id in nodal_loads:
            nodal_loads[n_id][0] += float(row["Load_X_N"])
            nodal_loads[n_id][1] += float(row["Load_Y_N"])

    for i, (_, row) in enumerate(elements_df.iterrows()):
        n1, n2 = int(row["Start_Node"]), int(row["End_Node"])
        sec_idx = sec_indices[i] if sec_indices is not None else len(sections_df) - 1
        A = sections_df.iloc[sec_idx]["Area_m2"]
        x1, y1 = nodes_df.loc[nodes_df['Node_ID']==n1, 'X_m'].values[0], nodes_df.loc[nodes_df['Node_ID']==n1, 'Y_m'].values[0] * y_scale
        x2, y2 = nodes_df.loc[nodes_df['Node_ID']==n2, 'X_m'].values[0], nodes_df.loc[nodes_df['Node_ID']==n2, 'Y_m'].values[0] * y_scale
        weight_N = A * math.hypot(x2 - x1, y2 - y1) * density * gravity
        if n1 in nodal_loads: nodal_loads[n1][1] -= (weight_N / 2.0)
        if n2 in nodal_loads: nodal_loads[n2][1] -= (weight_N / 2.0)
    return nodal_loads

def plot_truss(nodes, elements, current_H, sec_indices=None, title="Truss"):
    fig, ax = plt.subplots(figsize=(10, 5))
    y_scale = current_H / initial_peak_y
    node_coords = {}
    
    for _, row in nodes.iterrows():
        n_id = int(row["Node_ID"])
        x = row["X_m"]
        y_actual = row["Y_m"] * y_scale 
        node_coords[n_id] = (x, y_actual)
        ax.plot(x, y_actual, 'ko', markersize=6, zorder=3)
        label = f"N{n_id}\n({x:.2f}, {y_actual:.2f})"
        ax.text(x, y_actual + 0.2, label, fontsize=8, ha='center', color='darkred', weight='bold')

    combined_loads = get_total_nodal_loads(current_H, sec_indices)
    for n_id, forces in combined_loads.items():
        Fx_kN = forces[0] / 1000.0
        Fy_kN = forces[1] / 1000.0
        
        if abs(Fx_kN) > 0.01 or abs(Fy_kN) > 0.01:
            x, y_actual = node_coords[n_id]
            if abs(Fx_kN) < 0.01: 
                load_str = f"↓ {abs(Fy_kN):.1f} kN" if Fy_kN < 0 else f"↑ {abs(Fy_kN):.1f} kN"
                offset_y = -0.6 if Fy_kN < 0 else 0.6
                ax.text(x, y_actual + offset_y, load_str, fontsize=8, ha='center', color='purple', weight='bold')
            elif abs(Fy_kN) < 0.01:
                load_str = f"← {abs(Fx_kN):.1f} kN" if Fx_kN < 0 else f"→ {abs(Fx_kN):.1f} kN"
                ax.text(x + ( -0.8 if Fx_kN < 0 else 0.8), y_actual, load_str, fontsize=8, ha='center', color='purple', weight='bold')
            else:
                load_str = f"({Fx_kN:.1f}, {Fy_kN:.1f}) kN"
                ax.text(x, y_actual - 0.6, load_str, fontsize=8, ha='center', color='purple', weight='bold')

    for i, (_, row) in enumerate(elements.iterrows()):
        n1, n2 = int(row["Start_Node"]), int(row["End_Node"])
        lw, color = 2, 'gray'
        if sec_indices is not None:
            A = sections_df.iloc[sec_indices[i]]["Area_m2"]
            lw = 1.5 + 4.0 * (A / sections_df["Area_m2"].max())
            color = 'navy'
        ax.plot([node_coords[n1][0], node_coords[n2][0]], [node_coords[n1][1], node_coords[n2][1]], color=color, linewidth=lw, zorder=2)

    ax.set_title(title)
    ax.set_xlabel("Span (m)")
    ax.set_ylabel("Height (m)")
    ax.set_ylim(bottom=-1.5, top=current_H + 1.5)
    ax.grid(True, linestyle='--', alpha=0.6)
    ax.axis('equal') 
    return fig

# --- SHOW INITIAL GRAPH ---
st.markdown("---")
st.subheader("Initial Truss Geometry & Loading")
st.markdown("*Note: Purple loads show combined external forces + calculated self-weight at nodes.*")
fig_init = plot_truss(nodes_df, elements_df, initial_peak_y, title=f"Input Topology (Span: {actual_span:.2f} m)")
st.pyplot(fig_init)

def convert_df(df):
    return df.to_csv(index=False).encode('utf-8')

# --- 2. OPENSEES EVALUATION CORE ---
def evaluate_truss_core(vars, return_results=False):
    H = vars[0]
    section_indices = np.round(vars[1:]).astype(int)
    y_scale = H / initial_peak_y 
    
    ops.wipe()
    ops.model('basic', '-ndm', 2, '-ndf', 2)
    
    for _, row in nodes_df.iterrows():
        ops.node(int(row["Node_ID"]), float(row["X_m"]), float(row["Y_m"] * y_scale))
        ops.fix(int(row["Node_ID"]), int(row["Support_X"]), int(row["Support_Y"]))
        
    ops.uniaxialMaterial('Elastic', 1, E)
    total_volume = 0.0
    element_lengths, element_areas, element_inertias = {}, {}, {}
    
    for i, (_, row) in enumerate(elements_df.iterrows()):
        e_id, n1, n2 = int(row["Element_ID"]), int(row["Start_Node"]), int(row["End_Node"])
        A, I = sections_df.iloc[section_indices[i]]["Area_m2"], sections_df.iloc[section_indices[i]]["Inertia_m4"]
        ops.element('Truss', e_id, n1, n2, A, 1)
        
        c1, c2 = ops.nodeCoord(n1), ops.nodeCoord(n2)
        L_elem = math.hypot(c2[0] - c1[0], c2[1] - c1[1])
        total_volume += A * L_elem
        element_lengths[e_id], element_areas[e_id], element_inertias[e_id] = L_elem, A, I

    total_weight_kg = total_volume * density
    combined_loads = get_total_nodal_loads(H, section_indices)
    
    ops.timeSeries('Linear', 1)
    ops.pattern('Plain', 1, 1)
    for n_id, forces in combined_loads.items():
        if forces[0] != 0 or forces[1] != 0: ops.load(n_id, float(forces[0]), float(forces[1]))
        
    ops.system('BandSPD')
    ops.numberer('RCM')
    ops.constraints('Plain')
    ops.integrator('LoadControl', 1.0)
    ops.algorithm('Linear')
    ops.analysis('Static')
    
    if ops.analyze(1) != 0: return 1e9, 1e9, 1e9, [], {} 
        
    penalty = 0.0
    max_vertical_deflection = 0.0
    node_deflections = {} 
    
    for _, row in nodes_df.iterrows():
        disp_y = ops.nodeDisp(int(row["Node_ID"]), 2) 
        node_deflections[int(row["Node_ID"])] = disp_y
        max_vertical_deflection = max(max_vertical_deflection, abs(disp_y))
        
    for i, (_, row) in enumerate(elements_df.iterrows()):
        e_id = int(row["Element_ID"])
        force = ops.basicForce(e_id)[0] 
        if force > 0: 
            if force > element_areas[e_id] * fy: penalty += (force - element_areas[e_id] * fy)
        elif force < 0: 
            P_cr = (math.pi**2 * E * element_inertias[e_id]) / (element_lengths[e_id]**2)
            if abs(force) > P_cr: penalty += (abs(force) - P_cr)
                
    if return_results:
        final_forces_kN = [ops.basicForce(int(row["Element_ID"]))[0] / 1000.0 for _, row in elements_df.iterrows()]
        return total_weight_kg, max_vertical_deflection, penalty, final_forces_kN, node_deflections
    
    return total_weight_kg, max_vertical_deflection, penalty

# --- 3. PYMOO PROBLEM DEFINITION ---
class MultiObjectiveTruss(ElementwiseProblem):
    def __init__(self, num_elements, min_h, max_h, max_sec_idx):
        xl = [min_h] + [0] * num_elements
        xu = [max_h] + [max_sec_idx] * num_elements
        super().__init__(n_var=1 + num_elements, n_obj=2, n_ieq_constr=1, xl=xl, xu=xu)

    def _evaluate(self, x, out, *args, **kwargs):
        weight, deflection, penalty = evaluate_truss_core(x)
        out["F"] = [weight, deflection * 1000]
        out["G"] = [penalty] 

class StreamlitParetoCallback(Callback):
    def __init__(self, placeholder):
        super().__init__()
        self.placeholder = placeholder

    def notify(self, algorithm):
        F = algorithm.opt.get("F")
        if F is not None and len(F) > 0:
            fig, ax = plt.subplots(figsize=(8, 4))
            ax.scatter(F[:, 0], F[:, 1], color='teal', edgecolor='k')
            ax.set_title(f"Live Pareto Front (Generation {algorithm.n_gen})")
            ax.set_xlabel("Objective 1: Weight (kg)")
            ax.set_ylabel("Objective 2: Max Deflection (mm)")
            ax.grid(True, linestyle='--', alpha=0.6)
            self.placeholder.pyplot(fig)
            plt.close(fig)

# --- 4. EXECUTION ---
st.markdown("---")
if st.button("Run Multi-Objective Optimization", type="primary"):
    num_elements = len(elements_df)
    max_sec_idx = len(sections_df) - 1
    
    plot_placeholder = st.empty()
    problem = MultiObjectiveTruss(num_elements, min_height, max_height, max_sec_idx)
    algorithm = NSGA2(pop_size=40, eliminate_duplicates=True)
    
    with st.spinner('Running NSGA-II Evolutionary Algorithm...'):
        res = minimize(problem, algorithm, get_termination("n_gen", 50), seed=1, callback=StreamlitParetoCallback(plot_placeholder), save_history=False, verbose=False)
        
        if res.F is not None:
            sort_idx = np.argsort(res.F[:, 0])
            st.session_state.pareto_F = res.F[sort_idx]
            st.session_state.pareto_X = res.X[sort_idx]
            st.success("✅ Multi-Objective Optimization Complete! Pareto Front Generated.")
        else:
            st.error("Optimization failed to find feasible solutions within constraints.")

# --- 5. POST-OPTIMIZATION EXPLORER ---
if 'pareto_F' in st.session_state:
    st.markdown("### Explore the Pareto Front")
    st.markdown("Slide through the optimal trade-offs. The tables will update dynamically based on the design you select.")
    
    F = st.session_state.pareto_F
    X = st.session_state.pareto_X
    num_solutions = len(F)
    
    selected_idx = st.slider(f"Select a Design Strategy (0 = Lightest/Flexible, {num_solutions-1} = Heaviest/Stiffest)", 
                             min_value=0, max_value=num_solutions-1, value=num_solutions//2)
    
    opt_X = X[selected_idx]
    opt_weight, opt_disp = F[selected_idx][0], F[selected_idx][1]
    best_H = opt_X[0]
    best_sections = np.round(opt_X[1:]).astype(int)
    
    final_weight, max_disp, final_penalty, final_forces, opt_defs = evaluate_truss_core(opt_X, return_results=True)
    
    col_res1, col_res2 = st.columns([1, 1])
    
    with col_res1:
        st.info(f"**Selected Design Metrics:**\n* Peak Height: **{best_H:.2f} m**\n* Weight: **{opt_weight:.2f} kg**\n* Max Deflection: **{opt_disp:.2f} mm**")
        fig_final = plot_truss(nodes_df, elements_df, best_H, sec_indices=best_sections, title="Topology & Final Loading for Selected Strategy")
        st.pyplot(fig_final)

    with col_res2:
        fig_p, ax_p = plt.subplots(figsize=(6, 4))
        ax_p.plot(F[:, 0], F[:, 1], 'k-', zorder=1, alpha=0.3) 
        ax_p.scatter(F[:, 0], F[:, 1], color='gray', zorder=2, label="Pareto Optimal Solutions")
        ax_p.scatter(opt_weight, opt_disp, color='red', s=100, zorder=3, edgecolors='black', label="Selected Design")
        ax_p.set_title("Pareto Front")
        ax_p.set_xlabel("Weight (kg)")
        ax_p.set_ylabel("Max Deflection (mm)")
        ax_p.legend()
        ax_p.grid(True, linestyle='--', alpha=0.6)
        st.pyplot(fig_p)

    # --- RESULTS TABLES FOR SELECTED DESIGN ---
    st.markdown("---")
    st.write("### Nodal Deflections")
    # Get initial deflections for comparison by passing a baseline array
    initial_vars = [initial_peak_y] + [len(sections_df)-1] * len(elements_df)
    _, _, _, _, initial_defs = evaluate_truss_core(initial_vars, return_results=True)
    
    deflection_data = {
        "Node_ID": nodes_df["Node_ID"].tolist(),
        "Initial_Deflection_Y_mm": [round(initial_defs[n_id] * 1000, 2) for n_id in nodes_df["Node_ID"]],
        "Optimized_Deflection_Y_mm": [round(opt_defs[n_id] * 1000, 2) for n_id in nodes_df["Node_ID"]]
    }
    deflection_df = pd.DataFrame(deflection_data)
    st.dataframe(deflection_df, use_container_width=True)
    st.download_button("Download Deflections CSV", data=convert_df(deflection_df), file_name="moo_deflections.csv", mime="text/csv")
    
    st.write("### Final Member Sizing & Forces")
    st.markdown("*(Positive = Tension, Negative = Compression)*")
    
    optimized_areas = [round(sections_df.iloc[idx]["Area_m2"], 4) for idx in best_sections]
    final_forces_rounded = [round(f, 2) for f in final_forces]
    
    final_data = {
        "Element_ID": elements_df["Element_ID"].tolist(),
        "Start_Node": elements_df["Start_Node"].tolist(),
        "End_Node": elements_df["End_Node"].tolist(),
        "Optimized_Area_m2": optimized_areas,
        "Axial_Force_kN": final_forces_rounded
    }
    final_design_df = pd.DataFrame(final_data)
    st.dataframe(final_design_df, use_container_width=True)
    st.download_button("Download Forces CSV", data=convert_df(final_design_df), file_name="moo_forces.csv", mime="text/csv")
