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
# TAB 1: DOUBLE PENDULUM
# ============================================
with tabs[0]:
    st.header("Double Pendulum")

    # Parameters
    st.sidebar.subheader("Double Pendulum Parameters")
    m1 = st.sidebar.slider("Mass m1", 0.1, 5.0, 1.0, 0.1)
    m2 = st.sidebar.slider("Mass m2", 0.1, 5.0, 1.0, 0.1)
    L1 = st.sidebar.slider("Length L1", 0.1, 5.0, 1.0, 0.1)
    L2 = st.sidebar.slider("Length L2", 0.1, 5.0, 1.0, 0.1)
    g = st.sidebar.slider("Gravity g", 1.0, 20.0, 9.81, 0.1)

    # Initial conditions
    st.sidebar.subheader("Initial Conditions")
    theta1_0 = st.sidebar.slider("θ1 (radians)", -3.14, 3.14, 1.0, 0.01)
    theta2_0 = st.sidebar.slider("θ2 (radians)", -3.14, 3.14, -1.0, 0.01)
    omega1_0 = st.sidebar.slider("ω1", -10.0, 10.0, 0.0, 0.1)
    omega2_0 = st.sidebar.slider("ω2", -10.0, 10.0, 0.0, 0.1)

    # Simulation settings
    st.sidebar.subheader("Simulation Settings")
    T = st.sidebar.slider("Total Time", 1.0, 50.0, 20.0, 1.0)
    dt = st.sidebar.slider("Time Step", 0.001, 0.1, 0.02, 0.001)

    show_anim = st.sidebar.checkbox("Animate", value=True)
    show_eq = st.sidebar.checkbox("Show Equations", value=False)

    # Equations of motion
    def derivatives(state, t):
        theta1, omega1, theta2, omega2 = state
        delta = theta2 - theta1

        den1 = (m1 + m2) * L1 - m2 * L1 * np.cos(delta)**2
        den2 = (L2 / L1) * den1

        domega1 = ((m2 * L1 * omega1**2 * np.sin(delta) * np.cos(delta) +
                    m2 * g * np.sin(theta2) * np.cos(delta) +
                    m2 * L2 * omega2**2 * np.sin(delta) -
                    (m1 + m2) * g * np.sin(theta1)) / den1)

        domega2 = ((-m2 * L2 * omega2**2 * np.sin(delta) * np.cos(delta) +
                    (m1 + m2) * g * np.sin(theta1) * np.cos(delta) -
                    (m1 + m2) * L1 * omega1**2 * np.sin(delta) -
                    (m1 + m2) * g * np.sin(theta2)) / den2)

        return np.array([omega1, domega1, omega2, domega2])

    # Time array
    t = np.arange(0, T, dt)

    # RK4 integrator
    def rk4_step(f, y, t, dt):
        k1 = f(y, t)
        k2 = f(y + 0.5 * dt * k1, t + 0.5 * dt)
        k3 = f(y + 0.5 * dt * k2, t + 0.5 * dt)
        k4 = f(y + dt * k3, t + dt)
        return y + (dt/6) * (k1 + 2*k2 + 2*k3 + k4)

    # Integrate trajectory
    state = np.array([theta1_0, omega1_0, theta2_0, omega2_0])
    trajectory = []
    for ti in t:
        trajectory.append(state)
        state = rk4_step(derivatives, state, ti, dt)
    trajectory = np.array(trajectory)

    # Convert to cartesian coords
    theta1, omega1, theta2, omega2 = trajectory.T
    x1 = L1 * np.sin(theta1)
    y1 = -L1 * np.cos(theta1)
    x2 = x1 + L2 * np.sin(theta2)
    y2 = y1 - L2 * np.cos(theta2)

    # Plot
    fig, ax = plt.subplots(figsize=(6, 6))
    if show_anim:
        def update(frame):
            ax.clear()
            ax.plot([0, x1[frame], x2[frame]], [0, y1[frame], y2[frame]], "o-", lw=2)
            ax.set_xlim(-L1 - L2 - 0.5, L1 + L2 + 0.5)
            ax.set_ylim(-L1 - L2 - 0.5, L1 + L2 + 0.5)
            ax.set_title(f"t = {t[frame]:.2f}")
        ani = FuncAnimation(fig, update, frames=len(t), interval=30, blit=False)
        st.pyplot(fig)
    else:
        ax.plot(x2, y2, label="Mass 2 trajectory")
        ax.set_aspect("equal")
        ax.legend()
        st.pyplot(fig)

    # Download data
    csv_data = "t,x1,y1,x2,y2\n" + "\n".join(
        f"{ti},{x1i},{y1i},{x2i},{y2i}" for ti, x1i, y1i, x2i, y2i in zip(t, x1, y1, x2, y2)
    )
    st.download_button("Download Trajectory (CSV)", data=csv_data, file_name="double_pendulum.csv")

    # Optional equations
    if show_eq:
        st.latex(r"""
        \begin{aligned}
        \dot{\theta}_1 &= \omega_1 \\
        \dot{\theta}_2 &= \omega_2 \\
        \dot{\omega}_1 &= \frac{m_2 L_1 \omega_1^2 \sin\delta \cos\delta + m_2 g \sin\theta_2 \cos\delta + m_2 L_2 \omega_2^2 \sin\delta - (m_1+m_2) g \sin\theta_1}{(m_1+m_2)L_1 - m_2 L_1 \cos^2\delta} \\
        \dot{\omega}_2 &= \frac{-m_2 L_2 \omega_2^2 \sin\delta \cos\delta + (m_1+m_2)(g \sin\theta_1 \cos\delta - L_1 \omega_1^2 \sin\delta - g \sin\theta_2)}{(L_2/L_1)[(m_1+m_2)L_1 - m_2 L_1 \cos^2\delta]}
        \end{aligned}
        """)

    # Theory section
    with st.expander("Theory"):
        st.markdown(r"""
        The **double pendulum** is a classical example of a chaotic system, 
        consisting of two pendulums attached end to end.  

        - The equations of motion are derived using the **Lagrangian formalism**.  
        - For small angles, the system behaves regularly, but for larger ones, it exhibits **chaotic dynamics**.  
        - It is sensitive to initial conditions, making it a perfect playground for studying chaos in mechanics.  

        In this simulation, you can vary the masses, lengths, and initial angles 
        to explore different dynamical regimes.
        """)


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
    
    # Add speed control slider
    animation_speed = st.slider("Animation Speed (ms per frame)", 
                               min_value=10, 
                               max_value=200, 
                               value=50, 
                               step=10,
                               help="Lower values = faster animation")

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
            
            # Create frames
            for k in range(0, len(traj_data[0][0]), step_size):
                frame_data = []
                
                if view == "3D Attractor":
                    # Add trajectory data
                    for j, data in enumerate(traj_data):
                        frame_data.append(go.Scatter3d(
                            x=data[0][:k+1], 
                            y=data[1][:k+1], 
                            z=data[2][:k+1],
                            mode='lines',
                            line=dict(color=colors[j % len(colors)], width=2),
                            name=f'Trajectory {j+1}'
                        ))
                    # Add marker data
                    for j, data in enumerate(traj_data):
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
                    for j, data in enumerate(traj_data):
                        frame_data.append(go.Scatter(
                            x=data[0][:k+1], 
                            y=data[1][:k+1],
                            mode='lines',
                            line=dict(color=colors[j % len(colors)], width=2),
                            name=f'Trajectory {j+1}'
                        ))
                    for j, data in enumerate(traj_data):
                        if k > 0:
                            frame_data.append(go.Scatter(
                                x=[data[0][k]], 
                                y=[data[1][k]],
                                mode='markers',
                                marker=dict(color=colors[j % len(colors)], size=10),
                                showlegend=False
                            ))
                            
                elif view == "2D Projection (x–z)":
                    for j, data in enumerate(traj_data):
                        frame_data.append(go.Scatter(
                            x=data[0][:k+1], 
                            y=data[2][:k+1],
                            mode='lines',
                            line=dict(color=colors[j % len(colors)], width=2),
                            name=f'Trajectory {j+1}'
                        ))
                    for j, data in enumerate(traj_data):
                        if k > 0:
                            frame_data.append(go.Scatter(
                                x=[data[0][k]], 
                                y=[data[2][k]],
                                mode='markers',
                                marker=dict(color=colors[j % len(colors)], size=10),
                                showlegend=False
                            ))
                            
                else:  # y-z projection
                    for j, data in enumerate(traj_data):
                        frame_data.append(go.Scatter(
                            x=data[1][:k+1], 
                            y=data[2][:k+1],
                            mode='lines',
                            line=dict(color=colors[j % len(colors)], width=2),
                            name=f'Trajectory {j+1}'
                        ))
                    for j, data in enumerate(traj_data):
                        if k > 0:
                            frame_data.append(go.Scatter(
                                x=[data[1][k]], 
                                y=[data[2][k]],
                                mode='markers',
                                marker=dict(color=colors[j % len(colors)], size=10),
                                showlegend=False
                            ))
                
                frames.append(go.Frame(data=frame_data, name=str(k)))

            # Create figure with first frame as initial data
            if frames:
                fig = go.Figure(
                    data=frames[0].data,
                    frames=frames
                )
            else:
                fig = go.Figure()

            # Update layout based on view
            if view == "3D Attractor":
                fig.update_layout(
                    scene=dict(
                        xaxis_title="X", yaxis_title="Y", zaxis_title="Z",
                        camera=dict(eye=dict(x=1.5, y=1.5, z=1.5)),
                        aspectmode='auto'
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
                    height=600,
                    xaxis=dict(autorange=True),
                    yaxis=dict(autorange=True)
                )

            # Add animation controls
            fig.update_layout(
                title="Lorenz Attractor Animation",
                showlegend=True,
                updatemenus=[{
                    'type': 'buttons',
                    'showactive': False,
                    'y': 1.15,
                    'x': 0.0,
                    'xanchor': 'left',
                    'yanchor': 'top',
                    'buttons': [
                        {
                            'label': '▶️ Play',
                            'method': 'animate',
                            'args': [None, {
                                'frame': {'duration': animation_speed, 'redraw': True},  # Use the slider value
                                'fromcurrent': True,
                                'mode': 'immediate',
                                'transition': {'duration': 0}
                            }]
                        },
                        {
                            'label': '⏸️ Pause',
                            'method': 'animate',
                            'args': [[None], {
                                'frame': {'duration': 0, 'redraw': False},
                                'mode': 'immediate'
                            }]
                        },
                        {
                            'label': '⏮️ Reset',
                            'method': 'animate',
                            'args': [[str(0)], {
                                'frame': {'duration': 0, 'redraw': True},
                                'mode': 'immediate'
                            }]
                        }
                    ]
                }],
                sliders=[{
                    'active': 0,
                    'steps': [{
                        'label': f't={i*step_size*dt:.1f}',
                        'method': 'animate',
                        'args': [[str(i*step_size)], {
                            'frame': {'duration': 0, 'redraw': True},
                            'mode': 'immediate'
                        }]
                    } for i in range(len(frames))],
                    'len': 0.9,
                    'x': 0.05,
                    'xanchor': 'left',
                    'y': 0,
                    'yanchor': 'top',
                    'currentvalue': {
                        'font': {'size': 12},
                        'prefix': 'Time: ',
                        'visible': True,
                        'xanchor': 'left'
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
    with st.expander("📚 Learn More — The Mathematics and Physics of Chaos"):
        st.markdown(
            r"""
            ### 🌪 The Birth of Chaos Theory
            
            In 1963, meteorologist Edward N. Lorenz was running weather simulations on an early computer when he made a startling discovery. 
            By entering initial conditions with slightly less precision (0.506 instead of 0.506127), he found that the weather patterns 
            diverged completely after just a few days of simulated time. This serendipitous observation led to one of the most profound 
            insights in modern science: **deterministic chaos**.
            
            ### 🧮 Understanding the Lorenz Equations
            
            The Lorenz system arose from a dramatic simplification of the equations governing atmospheric convection—the rising of warm 
            air and sinking of cool air that drives weather patterns. Starting from the Navier-Stokes equations for fluid flow coupled 
            with heat transport, Lorenz applied a technique called Galerkin truncation to reduce the infinite-dimensional system to just 
            three ordinary differential equations:
            
            $$\frac{dx}{dt} = \sigma (y - x)$$
            $$\frac{dy}{dt} = x (\rho - z) - y$$
            $$\frac{dz}{dt} = xy - \beta z$$
            
            **What do these variables represent?**
            - **x(t)**: The intensity of convective motion (how fast the fluid is circulating)
            - **y(t)**: The temperature difference between rising and falling fluid columns
            - **z(t)**: The distortion of the vertical temperature profile from linearity
            
            **The parameters have physical meaning:**
            - **σ (sigma)**: The Prandtl number, representing the ratio of viscous diffusion to thermal diffusion
            - **ρ (rho)**: The Rayleigh number, measuring the temperature difference driving convection
            - **β (beta)**: A geometric factor related to the physical dimensions of the convection cells
            
            ### 🔄 The Dance of Trajectories
            
            When you enable "Compare two nearby trajectories" and set a perturbation δ, you're exploring one of chaos theory's most 
            fundamental properties: **sensitive dependence on initial conditions**. The second trajectory starts at position 
            $(x_0 + \delta, y_0 + \delta, z_0 + \delta)$, representing a tiny uncertainty in our knowledge of the initial state.
            
            In a predictable system, this small difference would remain small. But in the Lorenz system, the trajectories diverge 
            exponentially—at least initially. The rate of this divergence is quantified by the positive Lyapunov exponent 
            $\lambda \approx 0.906$, meaning nearby trajectories separate as $\sim e^{0.906t}$.
            
            ### 🎭 Order Hidden in Chaos
            
            Despite the apparent randomness, the Lorenz system exhibits profound order:
            
            **Fixed Points and Their Stability**
            
            The system always has an equilibrium at the origin (0,0,0), representing no convection. When $\rho > 1$, two additional 
            fixed points emerge at $C^{\pm} = (\pm\sqrt{\beta(\rho-1)}, \pm\sqrt{\beta(\rho-1)}, \rho-1)$, representing steady 
            convection rotating clockwise or counterclockwise.
            
            As you increase ρ from 1 to 30, the system undergoes a remarkable sequence of transitions:
            - **ρ < 1**: All trajectories decay to the origin (no convection)
            - **1 < ρ < 24.74**: Trajectories spiral into one of the fixed points $C^{\pm}$ (steady convection)
            - **ρ > 24.74**: The strange attractor emerges—trajectories never settle down but jump irregularly between the two wings
            
            ### 🦋 The Strange Attractor
            
            The butterfly-shaped object you see is not just a trajectory—it's an **attractor**, a set toward which all nearby 
            trajectories converge. What makes it "strange" is its fractal structure: it has zero volume but infinite surface area, 
            with a fractal dimension of approximately 2.06.
            
            This means that while trajectories are attracted to this set, once on it, they diverge from each other. It's like a 
            cosmic dance where all dancers must stay on the same intricate stage, but their individual movements remain forever 
            unpredictable.
            
            ### 🌍 Why This Matters
            
            The Lorenz system fundamentally changed our understanding of prediction and determinism. Before Lorenz, scientists 
            believed that deterministic equations always led to predictable behavior—if you knew the equations and initial conditions 
            precisely, you could predict the future indefinitely. The Lorenz system shattered this illusion, showing that even simple 
            deterministic systems can generate behavior so complex it appears random.
            
            This insight explains why weather prediction has fundamental limits (about 2 weeks), why ecosystems can suddenly collapse, 
            and why financial markets exhibit wild swings. It's also inspired applications in secure communications (using chaos to 
            encrypt messages), understanding cardiac arrhythmias, and even creating better random number generators.
            
            ### 🎮 Exploration Guide
            
            To truly appreciate the richness of this system, try these experiments:
            
            1. **Witness the birth of chaos**: Set ρ = 20 and slowly increase it to 30. Around ρ ≈ 24.74, you'll see the transition 
               from predictable spiraling to chaotic wandering.
            
            2. **Test sensitivity**: Enable two trajectories with δ = 0.00001. Watch how they stay together initially but eventually 
               end up on opposite wings of the attractor.
            
            3. **Find periodic windows**: Not all is chaos! Try ρ = 99.65 or ρ = 160—you'll find periodic behavior hidden within 
               the chaotic regime.
            
            4. **Explore projections**: The 2D views reveal hidden structure. The x-z projection shows how trajectories spiral outward 
               before jumping between wings.
            
            Remember: you're not just watching equations evolve—you're witnessing the fundamental unpredictability woven into the 
            fabric of nature itself. Every flutter of the Lorenz butterfly reminds us that the universe is far more mysterious and 
            beautiful than our intuitions suggest.
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
