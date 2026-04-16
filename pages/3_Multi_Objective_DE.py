import streamlit as st
import pandas as pd
import numpy as np
import openseespy.opensees as ops
from scipy.optimize import differential_evolution
import math
import matplotlib.pyplot as plt

st.set_page_config(page_title="MOO Truss Optimizer (DE)", layout="wide")
st.title("Multi-Objective Truss Optimizer (SciPy DE)")
st.markdown("Optimizing for **Minimum Weight** and **Maximum Stiffness** using **Differential Evolution** via the Weighted Sum method.")

# --- 0. SESSION STATE & PRESETS (Same as before) ---
def clear_editor_keys():
    for key in ['sections_editor', 'nodes_editor', 'elements_editor', 'loads_editor', 'pareto_X', 'pareto_F']:
        if key in st.session_state:
            del st.session_state[key]

def load_preset_king_post():
    clear_editor_keys()
    st.session_state.sections_data = pd.DataFrame({"Section_ID": [0, 1, 2, 3, 4], "Area_m2": [0.001, 0.002, 0.004, 0.006, 0.010], "Inertia_m4": [0.01, 0.02, 0.03, 0.04, 0.05]})
    st.session_state.nodes_data = pd.DataFrame({"Node_ID": [1, 2, 3, 4], "X_m": [0.0, 5.0, 10.0, 5.0], "Y_m": [0.0, 2.0, 0.0, 0.0], "Support_X": [1, 0, 0, 0], "Support_Y": [1, 0, 1, 0]})
    st.session_state.elements_data = pd.DataFrame({"Element_ID": [1, 2, 3, 4, 5], "Start_Node": [1, 4, 1, 2, 4], "End_Node": [4, 3, 2, 3, 2]})
    st.session_state.loads_data = pd.DataFrame({"Node_ID": [2, 4], "Load_X_N": [0.0, 0.0], "Load_Y_N": [-60000.0, -20000.0]})

if 'nodes_data' not in st.session_state:
    load_preset_king_post()

# --- 1. UI INPUTS ---
st.sidebar.header("Optimization Bounds & Material")
max_height = st.sidebar.number_input("Maximum Allowable Height (m)", value=4.0, step=0.5)
min_height = st.sidebar.number_input("Minimum Allowable Height (m)", value=1.0, step=0.5)
st.sidebar.markdown("---")
E = st.sidebar.number_input("Young's Modulus, E (Pa)", value=2e11, format="%e")
fy = st.sidebar.number_input("Yield Strength, Fy (Pa)", value=250e6, format="%e")
density = st.sidebar.number_input("Density (kg/m3)", value=7850.0)
gravity = 9.81 

st.sidebar.markdown("---")
st.sidebar.subheader("About the Developer")
st.sidebar.markdown("**Prathamesh Varma**\nM.E. Structural Engineering\nID: 2025H1430015H")
st.sidebar.markdown("🔗 [LinkedIn](https://www.linkedin.com/in/prathameshvarma)")

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
            if force > element_areas[e_id] * fy: penalty += (force - element_areas[e_id] * fy) * 1e3
        elif force < 0: 
            P_cr = (math.pi**2 * E * element_inertias[e_id]) / (element_lengths[e_id]**2)
            if abs(force) > P_cr: penalty += (abs(force) - P_cr) * 1e3
                
    if return_results:
        final_forces_kN = [ops.basicForce(int(row["Element_ID"]))[0] / 1000.0 for _, row in elements_df.iterrows()]
        return total_weight_kg, max_vertical_deflection, penalty, final_forces_kN, node_deflections
    
    return total_weight_kg, max_vertical_deflection, penalty

# --- 3. WEIGHTED SCALARIZATION FOR DIFFERENTIAL EVOLUTION ---
def evaluate_weighted_truss(vars, weight_factor, norm_w=5000.0, norm_d=0.05):
    """
    Transforms the multi-objective problem into a single objective 
    by weighting normalized Weight and Deflection.
    """
    weight, deflection, penalty = evaluate_truss_core(vars)
    
    # Normalize so both objectives scale between ~0.0 and 1.0
    normalized_weight = weight / norm_w
    normalized_deflection = deflection / norm_d
    
    # Weighted Sum Equation
    # If weight_factor = 1.0, it only optimizes weight. 
    # If weight_factor = 0.0, it only optimizes deflection.
    objective = (weight_factor * normalized_weight) + ((1.0 - weight_factor) * normalized_deflection) + penalty
    return objective

# --- 4. EXECUTION VIA LOOPED SCIPY DE ---
st.markdown("---")
st.write("### Weighted Sum Pareto Generation")
num_points = st.slider("Number of Trade-off Points (Pareto Resolution)", min_value=3, max_value=20, value=8)

if st.button("Run Multi-Objective DE", type="primary"):
    num_elements = len(elements_df)
    max_sec_idx = len(sections_df) - 1
    bounds = [(min_height, max_height)] + [(0, max_sec_idx) for _ in range(num_elements)]
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    plot_placeholder = st.empty()
    
    pareto_F = []
    pareto_X = []
    
    # Generate weights from 0.0 to 1.0
    weights = np.linspace(0, 1, num_points)
    
    with st.spinner('Running Differential Evolution iteratively to build Pareto Front...'):
        for i, w in enumerate(weights):
            status_text.markdown(f"**Optimizing Point {i+1}/{num_points}** (Weight Factor $w$ = {w:.2f})...")
            
            # Run SciPy's single-objective DE for the current weight
            res = differential_evolution(
                evaluate_weighted_truss, 
                bounds, 
                args=(w, 5000.0, 0.05), # Passing normalization constants
                maxiter=1000, 
                popsize=15, 
                mutation=(0.5, 1.0), 
                recombination=0.7, 
                tol=1e-3
            )
            
            if res.success:
                pareto_X.append(res.x)
                # Recover the actual (non-normalized) weight and deflection for plotting
                actual_weight, actual_deflection, _ = evaluate_truss_core(res.x)
                pareto_F.append([actual_weight, actual_deflection * 1000]) # Deflection in mm
                
                # Live update of the Pareto Front
                current_F = np.array(pareto_F)
                fig, ax = plt.subplots(figsize=(8, 4))
                ax.scatter(current_F[:, 0], current_F[:, 1], color='teal', edgecolor='k', s=80)
                ax.set_title(f"Live Pareto Front Generation (Point {i+1}/{num_points})")
                ax.set_xlabel("Objective 1: Weight (kg)")
                ax.set_ylabel("Objective 2: Max Deflection (mm)")
                ax.grid(True, linestyle='--', alpha=0.6)
                plot_placeholder.pyplot(fig)
                plt.close(fig)
                
            progress_bar.progress((i + 1) / num_points)
            
        if pareto_F:
            # Sort the points so they form a clean line in the Post-Explorer
            pareto_F_arr = np.array(pareto_F)
            pareto_X_arr = np.array(pareto_X)
            sort_idx = np.argsort(pareto_F_arr[:, 0])
            
            st.session_state.pareto_F = pareto_F_arr[sort_idx]
            st.session_state.pareto_X = pareto_X_arr[sort_idx]
            status_text.success("✅ Multi-Objective Optimization via Differential Evolution Complete!")
        else:
            st.error("Optimization failed to find feasible solutions.")

# --- 5. POST-OPTIMIZATION EXPLORER ---
if 'pareto_F' in st.session_state:
    st.markdown("### Explore the Pareto Front")
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
        fig_final = plot_truss(nodes_df, elements_df, best_H, sec_indices=best_sections, title="Topology for Selected Strategy")
        st.pyplot(fig_final)

    with col_res2:
        fig_p, ax_p = plt.subplots(figsize=(6, 4))
        ax_p.plot(F[:, 0], F[:, 1], 'k-', zorder=1, alpha=0.3) 
        ax_p.scatter(F[:, 0], F[:, 1], color='gray', zorder=2, label="Pareto Optimal Solutions")
        ax_p.scatter(opt_weight, opt_disp, color='red', s=100, zorder=3, edgecolors='black', label="Selected Design")
        ax_p.set_title("Pareto Front (Generated by DE)")
        ax_p.set_xlabel("Weight (kg)")
        ax_p.set_ylabel("Max Deflection (mm)")
        ax_p.legend()
        ax_p.grid(True, linestyle='--', alpha=0.6)
        st.pyplot(fig_p)

    st.markdown("---")
    st.write("### Final Member Sizing & Forces")
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
    st.download_button("Download Forces CSV", data=convert_df(final_design_df), file_name="moo_de_forces.csv", mime="text/csv")
