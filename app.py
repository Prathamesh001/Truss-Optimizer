import streamlit as st
import pandas as pd
import numpy as np
import openseespy.opensees as ops
from scipy.optimize import differential_evolution
import math
import matplotlib.pyplot as plt

st.set_page_config(page_title="Truss Optimizer", layout="wide")
st.title("Differential Evolution Truss Optimizer")

# --- 0. SESSION STATE & PRESETS ---
def clear_editor_keys():
    for key in ['sections_editor', 'nodes_editor', 'elements_editor', 'loads_editor']:
        if key in st.session_state:
            del st.session_state[key]

def load_preset_king_post():
    clear_editor_keys()
    st.session_state.sections_data = pd.DataFrame({"Section_ID": [0, 1, 2, 3, 4], "Area_m2": [0.001, 0.002, 0.004, 0.006, 0.010], "Inertia_m4": [1e-6, 3e-6, 8e-6, 1.5e-5, 3.5e-5]})
    st.session_state.nodes_data = pd.DataFrame({"Node_ID": [1, 2, 3, 4], "X_m": [0.0, 5.0, 10.0, 5.0], "Y_m": [0.0, 2.0, 0.0, 0.0], "Support_X": [1, 0, 0, 0], "Support_Y": [1, 0, 1, 0]})
    st.session_state.elements_data = pd.DataFrame({"Element_ID": [1, 2, 3, 4, 5], "Start_Node": [1, 4, 1, 2, 4], "End_Node": [4, 3, 2, 3, 2]})
    st.session_state.loads_data = pd.DataFrame({"Node_ID": [2, 4], "Load_X_N": [0.0, 0.0], "Load_Y_N": [-60000.0, -20000.0]})

def load_preset_warren():
    clear_editor_keys()
    st.session_state.sections_data = pd.DataFrame({"Section_ID": [0, 1, 2, 3], "Area_m2": [0.001, 0.002, 0.003, 0.005], "Inertia_m4": [1e-6, 3e-6, 6e-6, 1.5e-5]})
    st.session_state.nodes_data = pd.DataFrame({"Node_ID": [1, 2, 3, 4, 5], "X_m": [0.0, 5.0, 10.0, 2.5, 7.5], "Y_m": [0.0, 0.0, 0.0, 2.0, 2.0], "Support_X": [1, 0, 0, 0, 0], "Support_Y": [1, 0, 1, 0, 0]})
    st.session_state.elements_data = pd.DataFrame({"Element_ID": [1, 2, 3, 4, 5, 6, 7], "Start_Node": [1, 2, 4, 1, 4, 2, 5], "End_Node": [2, 3, 5, 4, 2, 5, 3]})
    st.session_state.loads_data = pd.DataFrame({"Node_ID": [2, 4, 5], "Load_X_N": [0.0, 0.0, 0.0], "Load_Y_N": [-40000.0, -15000.0, -15000.0]})

def load_preset_long_pratt():
    clear_editor_keys()
    # 5 sections available
    st.session_state.sections_data = pd.DataFrame({"Section_ID": [0, 1, 2, 3, 4], "Area_m2": [0.001, 0.002, 0.004, 0.006, 0.010], "Inertia_m4": [1e-6, 3e-6, 8e-6, 1.5e-5, 3.5e-5]})
    
    # 12 Nodes: 7 on bottom chord, 5 on top chord. 30m total span.
    st.session_state.nodes_data = pd.DataFrame({
        "Node_ID": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12], 
        "X_m": [0.0, 5.0, 10.0, 15.0, 20.0, 25.0, 30.0, 5.0, 10.0, 15.0, 20.0, 25.0], 
        "Y_m": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 4.0, 4.0, 4.0, 4.0, 4.0], 
        "Support_X": [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], 
        "Support_Y": [1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0]
    })
    
    # 21 Elements: Chords, Verticals, and Pratt Diagonals
    st.session_state.elements_data = pd.DataFrame({
        "Element_ID": list(range(1, 22)), 
        "Start_Node": [1, 2, 3, 4, 5, 6, 8, 9, 10, 11, 2, 3, 4, 5, 6, 1, 8, 9, 12, 11, 10], 
        "End_Node":   [2, 3, 4, 5, 6, 7, 9, 10, 11, 12, 8, 9, 10, 11, 12, 8, 3, 4, 7, 6, 5]
    })
    
    # 5 point loads across the bottom chord mimicking a heavy deck
    st.session_state.loads_data = pd.DataFrame({
        "Node_ID": [2, 3, 4, 5, 6], 
        "Load_X_N": [0.0, 0.0, 0.0, 0.0, 0.0], 
        "Load_Y_N": [-15000.0, -15000.0, -25000.0, -15000.0, -15000.0]
    })

if 'nodes_data' not in st.session_state:
    load_preset_king_post()

# --- 1. UI INPUTS ---
st.sidebar.header("Library Presets")
if st.sidebar.button("King Post"):
    load_preset_king_post()
    st.rerun()
if st.sidebar.button("Warren Truss"):
    load_preset_warren()
    st.rerun()
if st.sidebar.button("Long Pratt Bridge"):
    load_preset_long_pratt()
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
st.sidebar.markdown("**Prathamesh Varma**\nM.E. Structural Engineering")

# Hyperlinks use standard Markdown: [Text to Display](URL)
st.sidebar.markdown("🔗 [LinkedIn](https://www.linkedin.com/in/prathameshvarma)")
st.sidebar.markdown("Under the Guidance of\nProf. [Vasan Arunachalam](https://www.bits-pilani.ac.in/hyderabad/a-vasan/)")

col1, col2 = st.columns(2)

with col1:
    st.subheader("1. Available Sections Catalog")
    sections_df = st.data_editor(
        st.session_state.sections_data, 
        num_rows="dynamic", 
        key="sections_editor",
        column_config={
            "Area_m2": st.column_config.NumberColumn(format="%.4f"),
            "Inertia_m4": st.column_config.NumberColumn(format="%.2e") 
        }
    )

    st.subheader("2. Node Coordinates")
    nodes_df = st.data_editor(st.session_state.nodes_data, num_rows="dynamic", key="nodes_editor")

with col2:
    st.subheader("3. Element Connectivity")
    elements_df = st.data_editor(st.session_state.elements_data, num_rows="dynamic", key="elements_editor")

    st.subheader("4. Nodal Loading (External Loads)")
    st.markdown("*Hit **Enter** after typing a load to update the graph!*")
    loads_df = st.data_editor(st.session_state.loads_data, num_rows="dynamic", key="loads_editor")

# --- DYNAMIC CALCULATIONS ---
actual_span = nodes_df["X_m"].max() - nodes_df["X_m"].min()
initial_peak_y = nodes_df["Y_m"].max()
if initial_peak_y == 0: 
    initial_peak_y = 1e-9 
deflection_limit = actual_span / 250.0

# --- VALIDATION CHECK ---
existing_nodes = {int(node_id) for node_id in nodes_df["Node_ID"]}
for _, row in elements_df.iterrows():
    n1, n2 = int(row["Start_Node"]), int(row["End_Node"])
    if n1 not in existing_nodes or n2 not in existing_nodes:
        st.error(f"🚨 **Topology Error:** Element {int(row['Element_ID'])} connects to missing nodes. Please fix the tables.")
        st.stop() 

# --- HELPER: CALCULATE AGGREGATED NODAL LOADS ---
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
        
        x1 = nodes_df.loc[nodes_df['Node_ID']==n1, 'X_m'].values[0]
        y1 = nodes_df.loc[nodes_df['Node_ID']==n1, 'Y_m'].values[0] * y_scale
        x2 = nodes_df.loc[nodes_df['Node_ID']==n2, 'X_m'].values[0]
        y2 = nodes_df.loc[nodes_df['Node_ID']==n2, 'Y_m'].values[0] * y_scale
        L = math.hypot(x2 - x1, y2 - y1)
        
        weight_N = A * L * density * gravity
        if n1 in nodal_loads: nodal_loads[n1][1] -= (weight_N / 2.0)
        if n2 in nodal_loads: nodal_loads[n2][1] -= (weight_N / 2.0)

    return nodal_loads

# --- 2. PLOTTING FUNCTION ---
def plot_truss(nodes, elements, current_H, sec_indices=None, title="Truss"):
    fig, ax = plt.subplots(figsize=(8, 4))
    y_scale = current_H / initial_peak_y
    node_coords = {}
    
    for _, row in nodes.iterrows():
        n_id = int(row["Node_ID"])
        x = row["X_m"]
        y_actual = row["Y_m"] * y_scale 
        node_coords[n_id] = (x, y_actual)
        ax.plot(x, y_actual, 'ko', markersize=6, zorder=3)
        label = f"N{n_id}\n({x:.2f}, {y_actual:.2f})"
        ax.text(x, y_actual + 0.15, label, fontsize=9, ha='center', color='darkred', weight='bold')

    combined_loads = get_total_nodal_loads(current_H, sec_indices)
    for n_id, forces in combined_loads.items():
        Fx_kN = forces[0] / 1000.0
        Fy_kN = forces[1] / 1000.0
        
        if abs(Fx_kN) > 0.01 or abs(Fy_kN) > 0.01:
            x, y_actual = node_coords[n_id]
            if abs(Fx_kN) < 0.01: 
                load_str = f"↓ {abs(Fy_kN):.2f} kN" if Fy_kN < 0 else f"↑ {abs(Fy_kN):.2f} kN"
                offset_y = -0.4 if Fy_kN < 0 else 0.4
                ax.text(x, y_actual + offset_y, load_str, fontsize=8, ha='center', color='purple', weight='bold')
            elif abs(Fy_kN) < 0.01:
                load_str = f"← {abs(Fx_kN):.2f} kN" if Fx_kN < 0 else f"→ {abs(Fx_kN):.2f} kN"
                ax.text(x + ( -0.5 if Fx_kN < 0 else 0.5), y_actual, load_str, fontsize=8, ha='center', color='purple', weight='bold')
            else:
                load_str = f"({Fx_kN:.2f}, {Fy_kN:.2f}) kN"
                ax.text(x, y_actual - 0.5, load_str, fontsize=8, ha='center', color='purple', weight='bold')

    for i, (_, row) in enumerate(elements.iterrows()):
        n1, n2 = int(row["Start_Node"]), int(row["End_Node"])
        x_coords = [node_coords[n1][0], node_coords[n2][0]]
        y_coords = [node_coords[n1][1], node_coords[n2][1]]
        
        lw, color = 2, 'gray'
        if sec_indices is not None:
            sec_idx = sec_indices[i]
            A = sections_df.iloc[sec_idx]["Area_m2"]
            max_A = sections_df["Area_m2"].max()
            lw = 1.5 + 4.0 * (A / max_A) 
            color = 'navy'
            
        ax.plot(x_coords, y_coords, color=color, linewidth=lw, zorder=2)

    ax.set_title(title, fontsize=14)
    ax.set_xlabel("Span (m)")
    ax.set_ylabel("Height (m)")
    ax.set_ylim(bottom=-1.0, top=current_H + 1.0)
    ax.grid(True, linestyle='--', alpha=0.6)
    ax.axis('equal') 
    return fig

# Show Initial Geometry
st.markdown("---")
st.subheader(f"Initial Truss Geometry (Calculated Span: {actual_span:.2f} m)")
fig_init = plot_truss(nodes_df, elements_df, initial_peak_y, title="User Defined Topology + Initial Loads")
st.pyplot(fig_init)

# --- CSV EXPORT HELPER ---
# Removed @st.cache_data to prevent Pandas type-inference hashing bugs in older Python versions
def convert_df(df):
    return df.to_csv(index=False).encode('utf-8')

# --- 3. OPENSEES EVALUATION FUNCTION ---
def evaluate_truss(vars, return_results=False):
    H = vars[0]
    section_indices = np.round(vars[1:]).astype(int)
    y_scale = H / initial_peak_y 
    
    ops.wipe()
    ops.model('basic', '-ndm', 2, '-ndf', 2)
    
    for _, row in nodes_df.iterrows():
        n_id = int(row["Node_ID"])
        ops.node(n_id, float(row["X_m"]), float(row["Y_m"] * y_scale))
        ops.fix(n_id, int(row["Support_X"]), int(row["Support_Y"]))
        
    mat_tag = 1
    ops.uniaxialMaterial('Elastic', mat_tag, E)
    
    total_volume = 0.0
    element_lengths, element_areas, element_inertias = {}, {}, {}
    
    for i, (_, row) in enumerate(elements_df.iterrows()):
        e_id = int(row["Element_ID"])
        n1, n2 = int(row["Start_Node"]), int(row["End_Node"])
        sec_idx = section_indices[i]
        
        A = sections_df.iloc[sec_idx]["Area_m2"]
        I = sections_df.iloc[sec_idx]["Inertia_m4"]
        
        ops.element('Truss', e_id, n1, n2, A, mat_tag)
        
        coords1, coords2 = ops.nodeCoord(n1), ops.nodeCoord(n2)
        L_elem = math.hypot(coords2[0] - coords1[0], coords2[1] - coords1[1])
        
        total_volume += A * L_elem
        element_lengths[e_id], element_areas[e_id], element_inertias[e_id] = L_elem, A, I

    total_weight_kg = total_volume * density
    combined_loads = get_total_nodal_loads(H, section_indices)
    
    ops.timeSeries('Linear', 1)
    ops.pattern('Plain', 1, 1)
    for n_id, forces in combined_loads.items():
        if forces[0] != 0 or forces[1] != 0:
            ops.load(n_id, float(forces[0]), float(forces[1]))
        
    ops.system('BandSPD')
    ops.numberer('RCM')
    ops.constraints('Plain')
    ops.integrator('LoadControl', 1.0)
    ops.algorithm('Linear')
    ops.analysis('Static')
    
    if ops.analyze(1) != 0:
        return 1e9 
        
    penalty = 0.0
    max_vertical_deflection = 0.0
    node_deflections = {} 
    
    for _, row in nodes_df.iterrows():
        n_id = int(row["Node_ID"])
        disp_y = ops.nodeDisp(n_id, 2) 
        node_deflections[n_id] = disp_y
        max_vertical_deflection = max(max_vertical_deflection, abs(disp_y))
        
    if max_vertical_deflection > deflection_limit:
        penalty += (max_vertical_deflection - deflection_limit) * 1e8
        
    for i, (_, row) in enumerate(elements_df.iterrows()):
        e_id = int(row["Element_ID"])
        A, I, L_elem = element_areas[e_id], element_inertias[e_id], element_lengths[e_id]
        force = ops.basicForce(e_id)[0] 
        
        if force > 0: 
            capacity = A * fy
            if force > capacity: penalty += (force - capacity) * 1e3
        elif force < 0: 
            P_comp = abs(force)
            P_cr = (math.pi**2 * E * I) / (L_elem**2)
            if P_comp > P_cr: penalty += (P_comp - P_cr) * 1e3
                
    objective = total_weight_kg + penalty
    
    if return_results:
        final_forces_kN = []
        for i, (_, row) in enumerate(elements_df.iterrows()):
            e_id = int(row["Element_ID"])
            force_N = ops.basicForce(e_id)[0] 
            final_forces_kN.append(force_N / 1000.0) 
            
        return total_weight_kg, penalty, max_vertical_deflection, final_forces_kN, node_deflections
    return objective

# --- 4. OPTIMIZATION EXECUTION ---
st.markdown("---")
if st.button("Run Optimization", type="primary"):
    
    st.write("### Live Optimization Progress")
    live_plot_placeholder = st.empty()
    
    # --- NEW: Stagnation Tracker Setup ---
    STAGNATION_LIMIT = 5          # Number of generations to look back
    STAGNATION_TOLERANCE = 0.001  # 0.1% improvement required to keep going
    st.session_state.best_history = []
    
    def live_update_callback(xk, convergence):
        # 1. Update the live plot (Your existing code)
        current_H = xk[0]
        current_sections = np.round(xk[1:]).astype(int)
        fig = plot_truss(nodes_df, elements_df, current_H, sec_indices=current_sections, title=f"Evolving... (Convergence: {convergence:.3f})")
        live_plot_placeholder.pyplot(fig)
        plt.close(fig)
        
        # 2. Evaluate the current best solution to track its weight
        current_weight = evaluate_truss(xk)
        st.session_state.best_history.append(current_weight)
        
        # Keep only the last 5 generations in memory
        if len(st.session_state.best_history) > STAGNATION_LIMIT:
            st.session_state.best_history.pop(0)
            
        # 3. Check for stagnation
        if len(st.session_state.best_history) == STAGNATION_LIMIT:
            oldest_weight = st.session_state.best_history[0]
            newest_weight = st.session_state.best_history[-1]
            
            # Calculate the percentage improvement
            improvement = abs(oldest_weight - newest_weight) / oldest_weight
            
            if improvement <= STAGNATION_TOLERANCE:
                # Returning True forces SciPy to immediately halt the optimization
                return True 

    with st.spinner('Running Differential Evolution...'):
        num_elements = len(elements_df)
        num_sections = len(sections_df) - 1
        
        initial_vars = [initial_peak_y] + [num_sections for _ in range(num_elements)]
        _, _, _, _, initial_defs = evaluate_truss(initial_vars, return_results=True)
        
        bounds = [(min_height, max_height)] + [(0, num_sections) for _ in range(num_elements)]
        
        # Note: tol is set very low so the built-in convergence doesn't stop it before our custom callback does
        result = differential_evolution(
            evaluate_truss, 
            bounds, 
            maxiter=10000, 
            popsize=15, 
            mutation=(0.5, 1.0), 
            recombination=0.7, 
            tol=1e-6, 
            callback=live_update_callback
        )
        
        best_H = result.x[0]
        best_sections = np.round(result.x[1:]).astype(int)
        
        final_weight, final_penalty, max_disp, final_forces, opt_defs = evaluate_truss(result.x, return_results=True)
        
        final_fig = plot_truss(nodes_df, elements_df, best_H, sec_indices=best_sections, title="Final Optimized Topology")
        live_plot_placeholder.pyplot(final_fig)
        plt.close(final_fig)
        
        # Updated success messaging to account for the custom callback
        if result.success:
            st.success(f"✅ **Optimization Converged!** The whole population reached the absolute minimum. \n\n**Minimum Weight:** {final_weight:.2f} kg")
        elif "halted" in result.message.lower():
            st.success(f"🛑 **Smart Stop Triggered!** The best design didn't improve by more than 0.1% over {STAGNATION_LIMIT} generations, saving computational time. \n\n**Minimum Weight:** {final_weight:.2f} kg")
        else:
            st.warning(f"⚠️ **Max Iterations Reached!** ({result.message}) \n\n**Best Weight Found:** {final_weight:.2f} kg")
            
        col_res1, col_res2 = st.columns(2)
        with col_res1:
            st.write(f"**Optimal Truss Peak Height:** {best_H:.2f} m")
            st.write(f"**Max System Deflection:** {max_disp*1000:.2f} mm (Limit: {deflection_limit*1000:.2f} mm)")
            if final_penalty > 0:
                st.warning(f"Constraint Violations detected. Penalty score: {final_penalty:.2f}")
                
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
            
            st.write("### Final Member Sizing & Forces")
            st.markdown("*(Positive = Tension, Negative = Compression)*")
            st.dataframe(final_design_df)
            st.download_button("Download Forces CSV", data=convert_df(final_design_df), file_name="truss_forces.csv", mime="text/csv")
            
        with col_res2:
            deflection_data = {
                "Node_ID": nodes_df["Node_ID"].tolist(),
                "Initial_Deflection_Y_mm": [round(initial_defs[n_id] * 1000, 2) for n_id in nodes_df["Node_ID"]],
                "Optimized_Deflection_Y_mm": [round(opt_defs[n_id] * 1000, 2) for n_id in nodes_df["Node_ID"]]
            }
            deflection_df = pd.DataFrame(deflection_data)
            
            st.write("### Nodal Deflections")
            st.dataframe(deflection_df)
            st.download_button("Download Deflections CSV", data=convert_df(deflection_df), file_name="truss_deflections.csv", mime="text/csv")
