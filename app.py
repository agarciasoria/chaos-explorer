# app.py
# ============================================
# Chaos & Nonlinear Dynamics Explorer (Lorenz tab demo)
# ============================================

import io
import numpy as np
import streamlit as st
from scipy.integrate import solve_ivp
import plotly.graph_objects as go
import time

# ---------- Utilities ----------
def lorenz_system(t, state, sigma, rho, beta):
    x, y, z = state
    dxdt = sigma * (y - x)
    dydt = x * (rho - z) - y
    dzdt = x * y - beta * z
    return [dxdt, dydt, dzdt]

def integrate_lorenz(x0, y0, z0, sigma, rho, beta, t_max, dt):
    t_eval = np.arange(0, t_max, dt)
    sol = solve_ivp(
        lorenz_system,
        [0, t_max],
        [x0, y0, z0],
        t_eval=t_eval,
        args=(sigma, rho, beta),
        rtol=1e-8,
        atol=1e-10
    )
    return sol.t, sol.y[0], sol.y[1], sol.y[2]

# ---------- Page setup ----------
st.set_page_config(page_title="Chaos & Nonlinear Dynamics Explorer", layout="wide")

st.sidebar.title("Chaos & Nonlinear Dynamics Explorer")
st.sidebar.info("Author: Alejandro García Soria")

tabs = st.tabs([
    "🏋️ Double Pendulum",
    "🌀 Lorenz Attractor",
    "🌿 Bifurcation Diagrams",
    "📈 Lyapunov Exponents",
    "✨ Hopf Explorer"
])

# ============================================
# TAB 1: DOUBLE PENDULUM (placeholder)
# ============================================
with tabs[0]:
    st.header("Double Pendulum")
    st.write("Coming soon...")

# ============================================
# TAB 2: LORENZ ATTRACTOR
# ============================================
with tabs[1]:
    st.header("Lorenz Attractor")
    st.write("Explore chaos through the 3D Lorenz system and its 2D projections")
    st.header("Lorenz Attractor")
    st.write("Explore chaos through the 3D Lorenz system and its 2D projections")
    #------- Controls ------------------
    col1, col2, col3 = st.columns([1,1,1])
    with col1:
        sigma = st.slider("σ (Prandtl number)", 0.0, 20.0, 10.0, 0.1)
        x0 = st.number_input("x₀", 0.0, step=0.1)
    with col2:
        rho = st.slider("ρ (Rayleigh number)", 0.0, 60.0, 28.0, 0.5)
        y0 = st.number_input("y₀", 1.0, step=0.1)
    with col3:
        beta = st.slider("β", 0.0, 10.0, float(8/3), 0.1)
        z0 = st.number_input("z₀", 1.05, step=0.1)

    colA, colB = st.columns(2)
    with colA:
        t_max = st.slider("Simulation Time (t_max)", 5.0, 120.0, 40.0, 1.0)
    with colB:
        dt = st.slider("Time Step (Δt)", 0.001, 0.05, 0.01, 0.001)

    # ------------------ Options ------------------
    colO1, colO2 = st.columns(2)
    with colO1:
        show_equations = st.checkbox("Show equations", True)
    with colO2:
        enable_downloads = st.checkbox("Enable downloads", True)

    # Second trajectory
    show_second = st.checkbox("Compare two nearby trajectories", False)
    if show_second:
        perturb = st.slider("Initial condition perturbation δ", 1e-5, 1.0, 0.01, 1e-5)

    # ------------------ Equations ------------------
    if show_equations:
        st.markdown("### 📐 Mathematical Description")
        c1, c2 = st.columns(2)
        with c1:
            st.latex(r"\dot{x} = \sigma (y - x)")
            st.latex(r"\dot{y} = x (\rho - z) - y")
            st.latex(r"\dot{z} = xy - \beta z")
        with c2:
            st.markdown("""
            - **σ**: Prandtl number  
            - **ρ**: Rayleigh number  
            - **β**: geometric factor
            - **Chaos** arises for certain parameters: nearby trajectories diverge exponentially on a strange attractor
            """)
    # Initialize session state
    if 'lorenz_animation_step' not in st.session_state:
        st.session_state.lorenz_animation_step = 0
    if 'lorenz_traj_data' not in st.session_state:
        st.session_state.lorenz_traj_data = None
    if 'lorenz_current_state' not in st.session_state:
        st.session_state.lorenz_current_state = None
    if 'lorenz_num_points' not in st.session_state:
        st.session_state.lorenz_num_points = 0

    # ----------- Real-time Animation ------------------
    view = st.radio("Select View", ["3D Attractor", "2D Projection (x–y)", "2D Projection (x–z)", "2D Projection (y–z)"], horizontal=True)
    
    col1, col2, col3 = st.columns([1, 1, 4])
    with col1:
        play_button = st.button("▶ Play Animation")
    with col2:
        stop_button = st.button("⏹ Stop")
    with col3:
        reset_button = st.button("🔄 Reset")

    # Animation settings
    steps_per_frame = st.slider("Steps per frame", min_value=1, max_value=10, value=3)

    if reset_button or stop_button:
        st.session_state.lorenz_animation_step = 0
        if reset_button:
            st.session_state.lorenz_traj_data = None
            st.session_state.lorenz_current_state = None

    if play_button:
        # Initialize animation
        st.session_state.lorenz_animation_step = 1
        st.session_state.lorenz_num_points = int(t_max/dt)
        
        # Initialize trajectories
        trajs = [(x0, y0, z0)]
        if show_second:
            trajs.append((x0+perturb, y0+perturb, z0+perturb))
        
        st.session_state.lorenz_traj_data = [[[xi], [yi], [zi]] for xi, yi, zi in trajs]
        st.session_state.lorenz_current_state = {
            'x': [traj[0] for traj in trajs],
            'y': [traj[1] for traj in trajs],
            'z': [traj[2] for traj in trajs]
        }
        st.rerun()

    # Continue animation if in progress
    if st.session_state.lorenz_animation_step > 0 and st.session_state.lorenz_animation_step <= st.session_state.lorenz_num_points:
        traj_data = st.session_state.lorenz_traj_data
        current_state = st.session_state.lorenz_current_state
        
        # Compute multiple steps per frame for smoother animation
        for _ in range(steps_per_frame):
            if st.session_state.lorenz_animation_step > st.session_state.lorenz_num_points:
                break
                
            for j in range(len(traj_data)):
                x = current_state['x'][j]
                y = current_state['y'][j]
                z = current_state['z'][j]
                
                # RK4 integration step
                dx = sigma * (y - x)
                dy = x * (rho - z) - y
                dz = x * y - beta * z
                
                x_new = x + dx*dt
                y_new = y + dy*dt
                z_new = z + dz*dt
                
                current_state['x'][j] = x_new
                current_state['y'][j] = y_new
                current_state['z'][j] = z_new
                
                traj_data[j][0].append(x_new)
                traj_data[j][1].append(y_new)
                traj_data[j][2].append(z_new)
            
            st.session_state.lorenz_animation_step += 1

        # Create and display plot
        fig = go.Figure()
        colors = ["blue", "red"]
        
        for j, data in enumerate(traj_data):
            if view == "3D Attractor":
                fig.add_trace(go.Scatter3d(
                    x=data[0], y=data[1], z=data[2], 
                    mode="lines", 
                    line=dict(color=colors[j % len(colors)], width=2),
                    name=f"Trajectory {j+1}"
                ))
                fig.update_layout(
                    scene=dict(
                        xaxis_title="X", yaxis_title="Y", zaxis_title="Z",
                        camera=dict(eye=dict(x=1.5, y=1.5, z=1.5))
                    )
                )
            elif view == "2D Projection (x–y)":
                fig.add_trace(go.Scatter(
                    x=data[0], y=data[1], 
                    mode="lines", 
                    line=dict(color=colors[j % len(colors)], width=2),
                    name=f"Trajectory {j+1}"
                ))
                fig.update_layout(xaxis_title="X", yaxis_title="Y")
            elif view == "2D Projection (x–z)":
                fig.add_trace(go.Scatter(
                    x=data[0], y=data[2], 
                    mode="lines", 
                    line=dict(color=colors[j % len(colors)], width=2),
                    name=f"Trajectory {j+1}"
                ))
                fig.update_layout(xaxis_title="X", yaxis_title="Z")
            else:  # y-z projection
                fig.add_trace(go.Scatter(
                    x=data[1], y=data[2], 
                    mode="lines", 
                    line=dict(color=colors[j % len(colors)], width=2),
                    name=f"Trajectory {j+1}"
                ))
                fig.update_layout(xaxis_title="Y", yaxis_title="Z")

        progress = st.session_state.lorenz_animation_step / st.session_state.lorenz_num_points
        fig.update_layout(
            title=f"Lorenz Attractor ({progress*100:.1f}% complete)", 
            height=560, 
            margin=dict(l=0, r=0, b=0, t=30),
            showlegend=True
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Progress bar
        st.progress(progress)
        
        # Continue animation
        if st.session_state.lorenz_animation_step <= st.session_state.lorenz_num_points:
            time.sleep(0.01)  # Small delay to control speed
            st.rerun()
    
    # Show final plot if animation is complete
    elif st.session_state.lorenz_traj_data is not None:
        traj_data = st.session_state.lorenz_traj_data
        fig = go.Figure()
        colors = ["blue", "red"]
        
        for j, data in enumerate(traj_data):
            if view == "3D Attractor":
                fig.add_trace(go.Scatter3d(
                    x=data[0], y=data[1], z=data[2], 
                    mode="lines", 
                    line=dict(color=colors[j % len(colors)], width=2),
                    name=f"Trajectory {j+1}"
                ))
                fig.update_layout(
                    scene=dict(
                        xaxis_title="X", yaxis_title="Y", zaxis_title="Z",
                        camera=dict(eye=dict(x=1.5, y=1.5, z=1.5))
                    )
                )
            elif view == "2D Projection (x–y)":
                fig.add_trace(go.Scatter(
                    x=data[0], y=data[1], 
                    mode="lines", 
                    line=dict(color=colors[j % len(colors)], width=2),
                    name=f"Trajectory {j+1}"
                ))
                fig.update_layout(xaxis_title="X", yaxis_title="Y")
            elif view == "2D Projection (x–z)":
                fig.add_trace(go.Scatter(
                    x=data[0], y=data[2], 
                    mode="lines", 
                    line=dict(color=colors[j % len(colors)], width=2),
                    name=f"Trajectory {j+1}"
                ))
                fig.update_layout(xaxis_title="X", yaxis_title="Z")
            else:  # y-z projection
                fig.add_trace(go.Scatter(
                    x=data[1], y=data[2], 
                    mode="lines", 
                    line=dict(color=colors[j % len(colors)], width=2),
                    name=f"Trajectory {j+1}"
                ))
                fig.update_layout(xaxis_title="Y", yaxis_title="Z")

        fig.update_layout(
            title="Lorenz Attractor (Complete)", 
            height=560, 
            margin=dict(l=0, r=0, b=0, t=30),
            showlegend=True
        )
        
        st.plotly_chart(fig, use_container_width=True)

    # Metrics
    m1, m2, m3 = st.columns(3)
    with m1:
        st.metric("σ (Prandtl)", f"{sigma:.2f}")
    with m2:
        st.metric("ρ (Rayleigh)", f"{rho:.2f}")
    with m3:
        st.metric("β", f"{beta:.3f}")

    # Downloads - only show if trajectory data exists
    if st.session_state.lorenz_traj_data is not None and enable_downloads:
        st.write("### Download Options")
        
        traj_data = st.session_state.lorenz_traj_data
        
        # CSV downloads
        col1, col2 = st.columns(2)
        for j in range(len(traj_data)):
            csv_lines = ["t,x,y,z"]
            for i in range(len(traj_data[j][0])):
                t_val = i * dt
                csv_lines.append(f"{t_val:.6f},{traj_data[j][0][i]:.6f},{traj_data[j][1][i]:.6f},{traj_data[j][2][i]:.6f}")
            csv_data = "\n".join(csv_lines)
            
            with col1 if j == 0 else col2:
                st.download_button(
                    label=f"📥 Download Trajectory {j+1} (CSV)",
                    data=csv_data,
                    file_name=f"lorenz_trajectory_{j+1}.csv",
                    mime="text/csv",
                    key=f"lorenz_csv_download_{j}"
                )
        # Create final figure for downloads
        fig_final = go.Figure()
        colors = ["blue", "red"]
        for j, data in enumerate(traj_data):
            if view == "3D Attractor":
                fig_final.add_trace(go.Scatter3d(x=data[0], y=data[1], z=data[2], mode="lines", 
                                         line=dict(color=colors[j % len(colors)], width=2),
                                         name=f"Trajectory {j+1}"))
                fig_final.update_layout(scene=dict(xaxis_title="X", yaxis_title="Y", zaxis_title="Z"))
            elif view == "2D Projection (x–y)":
                fig_final.add_trace(go.Scatter(x=data[0], y=data[1], mode="lines", 
                                       line=dict(color=colors[j % len(colors)], width=2),
                                       name=f"Trajectory {j+1}"))
                fig_final.update_layout(xaxis_title="X", yaxis_title="Y")
            elif view == "2D Projection (x–z)":
                fig_final.add_trace(go.Scatter(x=data[0], y=data[2], mode="lines", 
                                       line=dict(color=colors[j % len(colors)], width=2),
                                       name=f"Trajectory {j+1}"))
                fig_final.update_layout(xaxis_title="X", yaxis_title="Z")
            else:
                fig_final.add_trace(go.Scatter(x=data[1], y=data[2], mode="lines", 
                                       line=dict(color=colors[j % len(colors)], width=2),
                                       name=f"Trajectory {j+1}"))
                fig_final.update_layout(xaxis_title="Y", yaxis_title="Z")

        fig_final.update_layout(title="Lorenz Attractor", height=560, showlegend=True)

        # Interactive HTML
        html_buf = io.StringIO()
        fig_final.write_html(html_buf, include_plotlyjs="cdn")
        st.download_button(
            label="📥 Download interactive plot (HTML)",
            data=html_buf.getvalue().encode(),
            file_name="lorenz_plot.html",
            mime="text/html",
            key="html_download"
        )

        # Static PNG (requires 'kaleido' in requirements)
        try:
            png_bytes = fig_final.to_image(format="png", scale=2)
            st.download_button(
                label="🖼️ Download plot (PNG)",
                data=png_bytes,
                file_name="lorenz_plot.png",
                mime="image/png",
                key="png_download"
            )
        except Exception:
            st.info("To enable PNG export, add **kaleido** to requirements.txt.")

    # Rich, optional theory (hidden by default)
    with st.expander("📚 Learn More — Detailed Dynamics & Chaos"):
        st.markdown(
            r"""
            ### 🌪 Origins and Model
            Edward N. Lorenz (1963) derived this 3-ODE model as a truncation of the Boussinesq equations for thermal convection.
            Despite its simplicity, it exhibits **deterministic chaos**.

            ### 🧮 Equilibria (fixed points)
            Solve $\dot{x}=\dot{y}=\dot{z}=0$:
            - $E_0 = (0,0,0)$ for all parameters.
            - For $\rho > 1$, two additional equilibria appear via a pitchfork:
              $$E_{\pm} = \left(\pm\sqrt{\beta(\rho-1)},\ \pm\sqrt{\beta(\rho-1)},\ \rho-1\right).$$

            ### 🔒 Stability (classical results)
            - $E_0$ is stable for $0 < \rho < 1$ and loses stability at $\rho=1$.
            - For the standard parameters $\sigma=10,\ \beta=\tfrac{8}{3}$,
              the nontrivial equilibria $E_\pm$ lose stability via a Hopf bifurcation at
              $$\rho_H \approx \frac{\sigma(\sigma+\beta+3)}{\sigma-\beta-1} \approx 24.74,$$
              beyond which chaotic dynamics (the strange attractor) arise for typical initial conditions.

            ### 🧭 Symmetry & Dissipation
            - **Symmetry:** $(x,y,z)\mapsto(-x,-y,z)$ leaves the system invariant (two symmetric wings).
            - **Phase-space volume contraction:** $\nabla\cdot f = -(\sigma + 1 + \beta) < 0$ — trajectories are attracted to a lower-dimensional set (the strange attractor).

            ### 📈 Sensitive Dependence
            Nearby trajectories $\delta \mathbf{x}(t)$ grow as $\|\delta \mathbf{x}(t)\|\sim e^{\lambda t}$ with positive largest Lyapunov exponent $\lambda$ (quantified in your Lyapunov tab).

            ### 🌍 Applications & Uses
            - Conceptual model for **weather/climate predictability limits**.
            - Canonical example in **nonlinear dynamics/chaos** courses.
            - Inspiration for **secure comms** and **control** strategies.
            
            ### 🧪 Tips for Exploration
            - Try $(\sigma,\beta)=(10,8/3)$ and sweep $\rho$ around 20–30.
            - Toggle between 3D and 2D projections to study structure.
            - Reduce Δt or increase $t_{\max}$ to see finer features (heavier compute).
            """
        )
# ============================================
# TAB 3..5 (placeholders)
# ============================================
with tabs[2]:
    st.header("Bifurcation Diagrams")
    st.write("Coming soon...")

with tabs[3]:
    st.header("Lyapunov Exponents")
    st.write("Coming soon...")

with tabs[4]:
    st.header("Hopf Explorer")
    st.write("Coming soon...")
