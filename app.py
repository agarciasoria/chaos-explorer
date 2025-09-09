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
# TAB 2: LORENZ ATTRACTOR WITH ANIMATIONS
# ============================================
with tabs[1]:
    st.header("Lorenz Attractor")
    st.write("Explore chaos through the 3D Lorenz system and its 2D projections")
    
    #------- Controls ------------------
    col1, col2, col3 = st.columns([1,1,1])
    with col1:
        sigma = st.slider("σ (Prandtl number)", 0.0, 20.0, 10.0, 0.1)
        x0 = st.number_input("x₀", value=0.0, step=0.1)
    with col2:
        rho = st.slider("ρ (Rayleigh number)", 0.0, 60.0, 28.0, 0.5)
        y0 = st.number_input("y₀", value=1.0, step=0.1)
    with col3:
        beta = st.slider("β", 0.0, 10.0, float(8/3), 0.1)
        z0 = st.number_input("z₀", value=1.05, step=0.1)

    colA, colB = st.columns(2)
    with colA:
        t_max = st.slider("Simulation Time (t_max)", 5.0, 120.0, 40.0, 1.0)
    with colB:
        dt = st.slider("Time Step (Δt)", 0.001, 0.05, 0.01, 0.001)

    # ------------------ Options ------------------
    colO1, colO2, colO3 = st.columns(3)
    with colO1:
        show_equations = st.checkbox("Show equations", True)
    with colO2:
        enable_downloads = st.checkbox("Enable downloads", True)
    with colO3:
        animation_frames = st.slider("Animation frames", 50, 200, 100)

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
            - **Chaos** arises for certain parameters
            """)

    # View selection
    view = st.radio("Select View", ["3D Attractor", "2D Projection (x–y)", "2D Projection (x–z)", "2D Projection (y–z)"], horizontal=True)
    
    # Animation controls
    col1, col2 = st.columns(2)
    with col1:
        generate_animation = st.button("🎬 Generate Animation", type="primary")
    with col2:
        generate_static = st.button("📊 Generate Static Plot")

    if generate_animation or generate_static:
        with st.spinner("Computing Lorenz attractor..."):
            # Initialize trajectories
            trajs = [(x0, y0, z0)]
            if show_second:
                trajs.append((x0+perturb, y0+perturb, z0+perturb))
            
            num_points = int(t_max/dt)
            traj_data = [[[xi], [yi], [zi]] for xi, yi, zi in trajs]
            
            # Integrate
            x_curr = [traj[0] for traj in trajs]
            y_curr = [traj[1] for traj in trajs]
            z_curr = [traj[2] for traj in trajs]
            
            progress_bar = st.progress(0)
            
            for i in range(num_points):
                for j, (x, y, z) in enumerate(zip(x_curr, y_curr, z_curr)):
                    # RK4 step
                    dx = sigma * (y - x)
                    dy = x * (rho - z) - y
                    dz = x * y - beta * z
                    
                    x_new = x + dx*dt
                    y_new = y + dy*dt
                    z_new = z + dz*dt
                    
                    x_curr[j], y_curr[j], z_curr[j] = x_new, y_new, z_new
                    traj_data[j][0].append(x_new)
                    traj_data[j][1].append(y_new)
                    traj_data[j][2].append(z_new)
                
                if i % (num_points // 20) == 0:
                    progress_bar.progress(min(i / num_points, 1.0))
            
            progress_bar.empty()
            st.session_state.lorenz_traj_data_final = traj_data

        # Create animation or static plot
        if generate_animation:
            st.info("🎬 Animation ready! Use the controls below to play/pause.")
            
            # Create frames for animation
            frames = []
            colors = ["blue", "red"]
            
            # Calculate step size for frames
            step_size = max(1, len(traj_data[0][0]) // animation_frames)
            
            for k in range(0, len(traj_data[0][0]), step_size):
                frame_data = []
                for j, data in enumerate(traj_data):
                    if view == "3D Attractor":
                        # Add trajectory line
                        frame_data.append(go.Scatter3d(
                            x=data[0][:k+1], 
                            y=data[1][:k+1], 
                            z=data[2][:k+1],
                            mode='lines',
                            line=dict(color=colors[j % len(colors)], width=2),
                            name=f'Trajectory {j+1}',
                            showlegend=(k == 0)
                        ))
                        # Add current position marker
                        if k > 0:
                            frame_data.append(go.Scatter3d(
                                x=[data[0][k]], 
                                y=[data[1][k]], 
                                z=[data[2][k]],
                                mode='markers',
                                marker=dict(color=colors[j % len(colors)], size=8),
                                name=f'Position {j+1}',
                                showlegend=False
                            ))
                    elif view == "2D Projection (x–y)":
                        frame_data.append(go.Scatter(
                            x=data[0][:k+1], 
                            y=data[1][:k+1],
                            mode='lines',
                            line=dict(color=colors[j % len(colors)], width=2),
                            name=f'Trajectory {j+1}',
                            showlegend=(k == 0)
                        ))
                        if k > 0:
                            frame_data.append(go.Scatter(
                                x=[data[0][k]], 
                                y=[data[1][k]],
                                mode='markers',
                                marker=dict(color=colors[j % len(colors)], size=10),
                                showlegend=False
                            ))
                    elif view == "2D Projection (x–z)":
                        frame_data.append(go.Scatter(
                            x=data[0][:k+1], 
                            y=data[2][:k+1],
                            mode='lines',
                            line=dict(color=colors[j % len(colors)], width=2),
                            name=f'Trajectory {j+1}',
                            showlegend=(k == 0)
                        ))
                        if k > 0:
                            frame_data.append(go.Scatter(
                                x=[data[0][k]], 
                                y=[data[2][k]],
                                mode='markers',
                                marker=dict(color=colors[j % len(colors)], size=10),
                                showlegend=False
                            ))
                    else:  # y-z projection
                        frame_data.append(go.Scatter(
                            x=data[1][:k+1], 
                            y=data[2][:k+1],
                            mode='lines',
                            line=dict(color=colors[j % len(colors)], width=2),
                            name=f'Trajectory {j+1}',
                            showlegend=(k == 0)
                        ))
                        if k > 0:
                            frame_data.append(go.Scatter(
                                x=[data[1][k]], 
                                y=[data[2][k]],
                                mode='markers',
                                marker=dict(color=colors[j % len(colors)], size=10),
                                showlegend=False
                            ))
                
                frames.append(go.Frame(data=frame_data, name=str(k)))

            # Create figure with animation
            fig = go.Figure(
                data=frames[0].data if frames else [],
                frames=frames
            )

            # Update layout based on view
            if view == "3D Attractor":
                fig.update_layout(
                    scene=dict(
                        xaxis_title="X", yaxis_title="Y", zaxis_title="Z",
                        camera=dict(eye=dict(x=1.5, y=1.5, z=1.5))
                    ),
                    height=600
                )
            else:
                axis_titles = {
                    "2D Projection (x–y)": ("X", "Y"),
                    "2D Projection (x–z)": ("X", "Z"),
                    "2D Projection (y–z)": ("Y", "Z")
                }
                fig.update_layout(
                    xaxis_title=axis_titles[view][0],
                    yaxis_title=axis_titles[view][1],
                    height=600
                )

            # Add animation controls
            fig.update_layout(
                title="Lorenz Attractor Animation",
                updatemenus=[{
                    'type': 'buttons',
                    'showactive': False,
                    'y': 0,
                    'x': 0.1,
                    'xanchor': 'right',
                    'yanchor': 'bottom',
                    'buttons': [
                        {
                            'label': '▶️ Play',
                            'method': 'animate',
                            'args': [None, {
                                'frame': {'duration': 50, 'redraw': True},
                                'fromcurrent': True,
                                'transition': {'duration': 0}
                            }]
                        },
                        {
                            'label': '⏸️ Pause',
                            'method': 'animate',
                            'args': [[None], {
                                'frame': {'duration': 0, 'redraw': False},
                                'mode': 'immediate',
                                'transition': {'duration': 0}
                            }]
                        }
                    ]
                }],
                sliders=[{
                    'active': 0,
                    'steps': [{
                        'label': f'{i}',
                        'method': 'animate',
                        'args': [[f'{i*step_size}'], {
                            'frame': {'duration': 0, 'redraw': True},
                            'mode': 'immediate',
                            'transition': {'duration': 0}
                        }]
                    } for i in range(len(frames))],
                    'y': 0,
                    'len': 0.9,
                    'x': 0.1,
                    'xanchor': 'left',
                    'y': -0.1,
                    'yanchor': 'top',
                    'transition': {'duration': 0},
                    'currentvalue': {
                        'font': {'size': 12},
                        'prefix': 'Frame: ',
                        'visible': True,
                        'xanchor': 'right'
                    }
                }]
            )

            st.plotly_chart(fig, use_container_width=True)

        else:  # Static plot
            # Create static figure
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
                        ),
                        height=600
                    )
                elif view == "2D Projection (x–y)":
                    fig.add_trace(go.Scatter(
                        x=data[0], y=data[1], 
                        mode="lines", 
                        line=dict(color=colors[j % len(colors)], width=2),
                        name=f"Trajectory {j+1}"
                    ))
                    fig.update_layout(xaxis_title="X", yaxis_title="Y", height=600)
                elif view == "2D Projection (x–z)":
                    fig.add_trace(go.Scatter(
                        x=data[0], y=data[2], 
                        mode="lines", 
                        line=dict(color=colors[j % len(colors)], width=2),
                        name=f"Trajectory {j+1}"
                    ))
                    fig.update_layout(xaxis_title="X", yaxis_title="Z", height=600)
                else:  # y-z projection
                    fig.add_trace(go.Scatter(
                        x=data[1], y=data[2], 
                        mode="lines", 
                        line=dict(color=colors[j % len(colors)], width=2),
                        name=f"Trajectory {j+1}"
                    ))
                    fig.update_layout(xaxis_title="Y", yaxis_title="Z", height=600)
            
            fig.update_layout(
                title="Lorenz Attractor", 
                showlegend=True,
                margin=dict(l=0, r=0, b=0, t=30)
            )
            
            st.plotly_chart(fig, use_container_width=True)

    # Show existing plot if available
    elif 'lorenz_traj_data_final' in st.session_state and st.session_state.lorenz_traj_data_final is not None:
        traj_data = st.session_state.lorenz_traj_data_final
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
                    ),
                    height=600
                )
            elif view == "2D Projection (x–y)":
                fig.add_trace(go.Scatter(
                    x=data[0], y=data[1], 
                    mode="lines", 
                    line=dict(color=colors[j % len(colors)], width=2),
                    name=f"Trajectory {j+1}"
                ))
                fig.update_layout(xaxis_title="X", yaxis_title="Y", height=600)
            elif view == "2D Projection (x–z)":
                fig.add_trace(go.Scatter(
                    x=data[0], y=data[2], 
                    mode="lines", 
                    line=dict(color=colors[j % len(colors)], width=2),
                    name=f"Trajectory {j+1}"
                ))
                fig.update_layout(xaxis_title="X", yaxis_title="Z", height=600)
            else:  # y-z projection
                fig.add_trace(go.Scatter(
                    x=data[1], y=data[2], 
                    mode="lines", 
                    line=dict(color=colors[j % len(colors)], width=2),
                    name=f"Trajectory {j+1}"
                ))
                fig.update_layout(xaxis_title="Y", yaxis_title="Z", height=600)
        
        fig.update_layout(
            title="Lorenz Attractor", 
            showlegend=True,
            margin=dict(l=0, r=0, b=0, t=30)
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    else:
        st.info("👆 Click 'Generate Animation' or 'Generate Static Plot' to visualize the Lorenz attractor")

    # Metrics
    st.write("---")
    m1, m2, m3 = st.columns(3)
    with m1:
        st.metric("σ (Prandtl)", f"{sigma:.2f}")
    with m2:
        st.metric("ρ (Rayleigh)", f"{rho:.2f}")
    with m3:
        st.metric("β", f"{beta:.3f}")

    # Downloads - only show if trajectory data exists
    if 'lorenz_traj_data_final' in st.session_state and st.session_state.lorenz_traj_data_final is not None and enable_downloads:
        st.write("### 📥 Download Options")
        
        traj_data = st.session_state.lorenz_traj_data_final
        
        # CSV downloads
        col1, col2, col3 = st.columns(3)
        
        for j in range(len(traj_data)):
            csv_lines = ["t,x,y,z"]
            for i in range(len(traj_data[j][0])):
                t_val = i * dt
                csv_lines.append(f"{t_val:.6f},{traj_data[j][0][i]:.6f},{traj_data[j][1][i]:.6f},{traj_data[j][2][i]:.6f}")
            csv_data = "\n".join(csv_lines)
            
            with col1 if j == 0 else col2:
                st.download_button(
                    label=f"📊 Trajectory {j+1} (CSV)",
                    data=csv_data,
                    file_name=f"lorenz_trajectory_{j+1}.csv",
                    mime="text/csv",
                    key=f"lorenz_csv_download_{j}"
                )
        
        # Create final figure for downloads
        with col3:
            # Interactive HTML
            fig_download = go.Figure()
            colors = ["blue", "red"]
            for j, data in enumerate(traj_data):
                if view == "3D Attractor":
                    fig_download.add_trace(go.Scatter3d(
                        x=data[0], y=data[1], z=data[2], 
                        mode="lines", 
                        line=dict(color=colors[j % len(colors)], width=2),
                        name=f"Trajectory {j+1}"
                    ))
                    fig_download.update_layout(scene=dict(xaxis_title="X", yaxis_title="Y", zaxis_title="Z"))
                elif view == "2D Projection (x–y)":
                    fig_download.add_trace(go.Scatter(
                        x=data[0], y=data[1], 
                        mode="lines", 
                        line=dict(color=colors[j % len(colors)], width=2),
                        name=f"Trajectory {j+1}"
                    ))
                    fig_download.update_layout(xaxis_title="X", yaxis_title="Y")
                elif view == "2D Projection (x–z)":
                    fig_download.add_trace(go.Scatter(
                        x=data[0], y=data[2], 
                        mode="lines", 
                        line=dict(color=colors[j % len(colors)], width=2),
                        name=f"Trajectory {j+1}"
                    ))
                    fig_download.update_layout(xaxis_title="X", yaxis_title="Z")
                else:
                    fig_download.add_trace(go.Scatter(
                        x=data[1], y=data[2], 
                        mode="lines", 
                        line=dict(color=colors[j % len(colors)], width=2),
                        name=f"Trajectory {j+1}"
                    ))
                    fig_download.update_layout(xaxis_title="Y", yaxis_title="Z")

            fig_download.update_layout(title="Lorenz Attractor", height=560, showlegend=True)

            html_buf = io.StringIO()
            fig_download.write_html(html_buf, include_plotlyjs="cdn")
            st.download_button(
                label="🌐 Interactive HTML",
                data=html_buf.getvalue().encode(),
                file_name="lorenz_plot.html",
                mime="text/html",
                key="html_download_lorenz"
            )

    # Rich theory section
    with st.expander("📚 Learn More — Detailed Dynamics & Chaos"):
        st.markdown(
            r"""
            ### 🌪 Origins and Model
            Edward N. Lorenz (1963) derived this 3-ODE model as a truncation of the Boussinesq equations for thermal convection.
            Despite its simplicity, it exhibits **deterministic chaos**.

            ### 🧮 The Lorenz System
            The equations describe the motion of a fluid layer heated from below:
            
            $$\frac{dx}{dt} = \sigma (y - x)$$
            $$\frac{dy}{dt} = x (\rho - z) - y$$
            $$\frac{dz}{dt} = xy - \beta z$$

            ### 🎯 Fixed Points
            - **Origin**: $(0,0,0)$ - always exists
            - **Wings**: $(\pm\sqrt{\beta(\rho-1)}, \pm\sqrt{\beta(\rho-1)}, \rho-1)$ when $\rho > 1$

            ### 🌊 Bifurcations
            1. **$\rho = 1$**: Pitchfork bifurcation (origin loses stability)
            2. **$\rho \approx 24.74$**: Hopf bifurcation (for $\sigma=10, \beta=8/3$)
            3. **$\rho > 24.74$**: Chaotic behavior emerges

            ### 🦋 The Strange Attractor
            - **Butterfly shape**: Two lobes corresponding to rotation around each fixed point
            - **Fractal dimension**: Approximately 2.05
            - **Sensitive dependence**: Nearby trajectories diverge exponentially

            ### 📊 Lyapunov Exponents
            For standard parameters $(\sigma=10, \rho=28, \beta=8/3)$:
            - $\lambda_1 \approx 0.9056$ (positive - chaos indicator)
            - $\lambda_2 \approx 0$ (neutral direction)
            - $\lambda_3 \approx -14.572$ (strong contraction)

            ### 🔬 Physical Interpretation
            - **x**: Convective intensity
            - **y**: Temperature difference between ascending and descending currents
            - **z**: Deviation from linear temperature profile

            ### 💡 Applications
            1. **Weather prediction**: Demonstrates fundamental limits
            2. **Chaos theory**: Canonical example of deterministic chaos
            3. **Secure communications**: Chaos-based encryption
            4. **Laser dynamics**: Similar equations describe some lasers
            5. **Chemical reactions**: Belousov-Zhabotinsky reaction analogs

            ### 🎮 Interactive Tips
            - **Explore transitions**: Vary $\rho$ from 20 to 30 to see order→chaos
            - **Initial conditions**: Try $(1, 0, 0)$ vs $(0, 1, 0)$ for different approaches
            - **Time scales**: Increase $t_{max}$ to see long-term behavior
            - **Projections**: 2D views reveal hidden structure
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
