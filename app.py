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
# TAB 1: DOUBLE PENDULUM WITH ANIMATIONS
# ============================================
with tabs[1]:
    st.header("Double Pendulum")
    st.write("Experience chaos in mechanical systems through the mesmerizing double pendulum")
    
    #------- Controls ------------------
    col1, col2, col3 = st.columns([1,1,1])
    with col1:
        m1 = st.slider("m₁ (mass 1)", 0.1, 5.0, 1.0, 0.1)
        L1 = st.slider("L₁ (length 1)", 0.5, 2.0, 1.0, 0.1)
    with col2:
        m2 = st.slider("m₂ (mass 2)", 0.1, 5.0, 1.0, 0.1)
        L2 = st.slider("L₂ (length 2)", 0.5, 2.0, 1.0, 0.1)
    with col3:
        g = st.slider("g (gravity)", 1.0, 20.0, 9.81, 0.1)
        damping = st.slider("Damping", 0.0, 0.5, 0.0, 0.01)

    # Initial conditions
    st.write("### Initial Conditions")
    colA, colB, colC, colD = st.columns(4)
    with colA:
        theta1_0 = st.slider("θ₁ (deg)", -180.0, 180.0, 90.0, 1.0)
        theta1_0_rad = np.deg2rad(theta1_0)
    with colB:
        theta2_0 = st.slider("θ₂ (deg)", -180.0, 180.0, 45.0, 1.0)
        theta2_0_rad = np.deg2rad(theta2_0)
    with colC:
        omega1_0 = st.slider("ω₁ (rad/s)", -10.0, 10.0, 0.0, 0.1)
    with colD:
        omega2_0 = st.slider("ω₂ (rad/s)", -10.0, 10.0, 0.0, 0.1)

    colT1, colT2 = st.columns(2)
    with colT1:
        t_max_dp = st.slider("Simulation Time", 5.0, 100.0, 30.0, 1.0)
    with colT2:
        dt_dp = st.slider("Time Step (Δt)", 0.001, 0.05, 0.01, 0.001, key="dp_dt")

    # ------------------ Options ------------------
    colO1, colO2, colO3 = st.columns(3)
    with colO1:
        show_equations_dp = st.checkbox("Show equations", True, key="dp_eq")
    with colO2:
        enable_downloads_dp = st.checkbox("Enable downloads", True, key="dp_dl")
    with colO3:
        animation_frames_dp = st.slider("Animation frames", 50, 200, 100, key="dp_frames")

    # Second pendulum for comparison
    show_second_dp = st.checkbox("Compare two nearby initial conditions", False)
    if show_second_dp:
        perturb_dp = st.slider("Initial angle perturbation δ (deg)", 0.01, 10.0, 1.0, 0.01)
        perturb_dp_rad = np.deg2rad(perturb_dp)

    # ------------------ Equations ------------------
    if show_equations_dp:
        st.markdown("### 📐 Mathematical Description")
        st.markdown("""
        The double pendulum follows Lagrangian mechanics. The equations of motion are:
        """)
        st.latex(r"\ddot{\theta}_1 = \frac{-g(2m_1+m_2)\sin\theta_1 - m_2g\sin(\theta_1-2\theta_2) - 2\sin(\theta_1-\theta_2)m_2(\dot{\theta}_2^2L_2 + \dot{\theta}_1^2L_1\cos(\theta_1-\theta_2))}{L_1(2m_1+m_2-m_2\cos(2\theta_1-2\theta_2))}")
        st.latex(r"\ddot{\theta}_2 = \frac{2\sin(\theta_1-\theta_2)(\dot{\theta}_1^2L_1(m_1+m_2) + g(m_1+m_2)\cos\theta_1 + \dot{\theta}_2^2L_2m_2\cos(\theta_1-\theta_2))}{L_2(2m_1+m_2-m_2\cos(2\theta_1-2\theta_2))}")
        st.markdown("""
        - **θ₁, θ₂**: Angles from vertical
        - **ω₁, ω₂**: Angular velocities
        - **m₁, m₂**: Masses of pendulum bobs
        - **L₁, L₂**: Lengths of pendulum arms
        """)

    # View selection
    view_dp = st.radio("Select View", 
                       ["Full Motion", "Phase Space (θ₁-ω₁)", "Phase Space (θ₂-ω₂)", 
                        "Configuration Space (θ₁-θ₂)", "Energy Evolution"], 
                       horizontal=True)
    
    # Animation controls
    col1_dp, col2_dp = st.columns(2)
    with col1_dp:
        generate_animation_dp = st.button("🎬 Generate Animation", type="primary", key="dp_anim")
    with col2_dp:
        generate_static_dp = st.button("📊 Generate Static Plot", key="dp_static")
    
    # Speed control
    animation_speed_dp = st.slider("Animation Speed (ms per frame)", 
                                  min_value=10, 
                                  max_value=200, 
                                  value=50, 
                                  step=10,
                                  help="Lower values = faster animation",
                                  key="dp_speed")

    # Define double pendulum dynamics
    def double_pendulum_derivatives(state, t, m1, m2, L1, L2, g, damping):
        theta1, omega1, theta2, omega2 = state
        
        # Precompute trig values
        sin1 = np.sin(theta1)
        sin2 = np.sin(theta2)
        sin12 = np.sin(theta1 - theta2)
        cos12 = np.cos(theta1 - theta2)
        
        # Denominator
        den = 2*m1 + m2 - m2*np.cos(2*theta1 - 2*theta2)
        
        # Angular accelerations
        num1 = (-g*(2*m1 + m2)*sin1 - m2*g*np.sin(theta1 - 2*theta2) - 
                2*sin12*m2*(omega2**2*L2 + omega1**2*L1*cos12))
        alpha1 = num1 / (L1 * den) - damping*omega1
        
        num2 = (2*sin12*(omega1**2*L1*(m1 + m2) + g*(m1 + m2)*np.cos(theta1) + 
                omega2**2*L2*m2*cos12))
        alpha2 = num2 / (L2 * den) - damping*omega2
        
        return np.array([omega1, alpha1, omega2, alpha2])

    if generate_animation_dp or generate_static_dp:
        with st.spinner("Computing double pendulum dynamics..."):
            # Initialize pendulums
            initial_states = [[theta1_0_rad, omega1_0, theta2_0_rad, omega2_0]]
            if show_second_dp:
                initial_states.append([theta1_0_rad + perturb_dp_rad, omega1_0, 
                                     theta2_0_rad + perturb_dp_rad, omega2_0])
            
            num_points = int(t_max_dp/dt_dp)
            time_points = np.linspace(0, t_max_dp, num_points)
            
            # Storage for trajectories
            trajectories = []
            for initial_state in initial_states:
                trajectory = [initial_state]
                state = np.array(initial_state)
                
                # RK4 integration
                for i in range(1, num_points):
                    k1 = double_pendulum_derivatives(state, time_points[i], m1, m2, L1, L2, g, damping)
                    k2 = double_pendulum_derivatives(state + 0.5*dt_dp*k1, time_points[i], m1, m2, L1, L2, g, damping)
                    k3 = double_pendulum_derivatives(state + 0.5*dt_dp*k2, time_points[i], m1, m2, L1, L2, g, damping)
                    k4 = double_pendulum_derivatives(state + dt_dp*k3, time_points[i], m1, m2, L1, L2, g, damping)
                    
                    state = state + (dt_dp/6) * (k1 + 2*k2 + 2*k3 + k4)
                    trajectory.append(state.copy())
                
                trajectories.append(np.array(trajectory))
            
            # Convert to Cartesian coordinates for visualization
            cartesian_trajs = []
            for traj in trajectories:
                x1 = L1 * np.sin(traj[:, 0])
                y1 = -L1 * np.cos(traj[:, 0])
                x2 = x1 + L2 * np.sin(traj[:, 2])
                y2 = y1 - L2 * np.cos(traj[:, 2])
                cartesian_trajs.append({'x1': x1, 'y1': y1, 'x2': x2, 'y2': y2})
            
            # Calculate energies
            energies = []
            for traj in trajectories:
                KE1 = 0.5 * m1 * (L1 * traj[:, 1])**2
                KE2 = 0.5 * m2 * ((L1 * traj[:, 1])**2 + (L2 * traj[:, 3])**2 + 
                                  2 * L1 * L2 * traj[:, 1] * traj[:, 3] * np.cos(traj[:, 0] - traj[:, 2]))
                PE1 = m1 * g * L1 * (1 - np.cos(traj[:, 0]))
                PE2 = m2 * g * (L1 * (1 - np.cos(traj[:, 0])) + L2 * (1 - np.cos(traj[:, 2])))
                total_energy = KE1 + KE2 + PE1 + PE2
                energies.append(total_energy)
            
            st.session_state.dp_trajectories = trajectories
            st.session_state.dp_cartesian = cartesian_trajs
            st.session_state.dp_energies = energies
            st.session_state.dp_time = time_points

        # Create animation or static plot
        if generate_animation_dp:
            st.info("🎬 Animation ready! Use the controls below to play/pause.")
            
            # Create frames
            frames = []
            colors = ["blue", "red"]
            
            # Calculate step size for frames
            step_size = max(1, num_points // animation_frames_dp)
            
            # Create frames based on view
            for k in range(0, num_points, step_size):
                frame_data = []
                
                if view_dp == "Full Motion":
                    for j, (traj, cart) in enumerate(zip(trajectories, cartesian_trajs)):
                        # Pendulum arms
                        frame_data.append(go.Scatter(
                            x=[0, cart['x1'][k], cart['x2'][k]], 
                            y=[0, cart['y1'][k], cart['y2'][k]],
                            mode='lines+markers',
                            line=dict(color=colors[j % len(colors)], width=3),
                            marker=dict(size=[8, 10, 10]),
                                                        name=f'Pendulum {j+1}'
                        ))
                        # Trail
                        if k > 10:
                            trail_start = max(0, k - 50)
                            frame_data.append(go.Scatter(
                                x=cart['x2'][trail_start:k+1], 
                                y=cart['y2'][trail_start:k+1],
                                mode='lines',
                                line=dict(color=colors[j % len(colors)], width=1, opacity=0.3),
                                showlegend=False,
                                name=f'Trail {j+1}'
                            ))
                
                elif view_dp == "Phase Space (θ₁-ω₁)":
                    for j, traj in enumerate(trajectories):
                        frame_data.append(go.Scatter(
                            x=np.rad2deg(traj[:k+1, 0]), 
                            y=traj[:k+1, 1],
                            mode='lines',
                            line=dict(color=colors[j % len(colors)], width=2),
                            name=f'Pendulum {j+1}'
                        ))
                        if k > 0:
                            frame_data.append(go.Scatter(
                                x=[np.rad2deg(traj[k, 0])], 
                                y=[traj[k, 1]],
                                mode='markers',
                                marker=dict(color=colors[j % len(colors)], size=10),
                                showlegend=False
                            ))
                
                elif view_dp == "Phase Space (θ₂-ω₂)":
                    for j, traj in enumerate(trajectories):
                        frame_data.append(go.Scatter(
                            x=np.rad2deg(traj[:k+1, 2]), 
                            y=traj[:k+1, 3],
                            mode='lines',
                            line=dict(color=colors[j % len(colors)], width=2),
                            name=f'Pendulum {j+1}'
                        ))
                        if k > 0:
                            frame_data.append(go.Scatter(
                                x=[np.rad2deg(traj[k, 2])], 
                                y=[traj[k, 3]],
                                mode='markers',
                                marker=dict(color=colors[j % len(colors)], size=10),
                                showlegend=False
                            ))
                
                elif view_dp == "Configuration Space (θ₁-θ₂)":
                    for j, traj in enumerate(trajectories):
                        frame_data.append(go.Scatter(
                            x=np.rad2deg(traj[:k+1, 0]), 
                            y=np.rad2deg(traj[:k+1, 2]),
                            mode='lines',
                            line=dict(color=colors[j % len(colors)], width=2),
                            name=f'Pendulum {j+1}'
                        ))
                        if k > 0:
                            frame_data.append(go.Scatter(
                                x=[np.rad2deg(traj[k, 0])], 
                                y=[np.rad2deg(traj[k, 2])],
                                mode='markers',
                                marker=dict(color=colors[j % len(colors)], size=10),
                                showlegend=False
                            ))
                
                else:  # Energy Evolution
                    for j, energy in enumerate(energies):
                        frame_data.append(go.Scatter(
                            x=time_points[:k+1], 
                            y=energy[:k+1],
                            mode='lines',
                            line=dict(color=colors[j % len(colors)], width=2),
                            name=f'Pendulum {j+1} Energy'
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
            if view_dp == "Full Motion":
                max_range = 1.1 * (L1 + L2)
                fig.update_layout(
                    xaxis=dict(range=[-max_range, max_range], title="x"),
                    yaxis=dict(range=[-max_range, 0.5], title="y"),
                    aspectratio=dict(x=1, y=1),
                    height=600
                )
            elif view_dp == "Phase Space (θ₁-ω₁)":
                fig.update_layout(
                    xaxis_title="θ₁ (degrees)",
                    yaxis_title="ω₁ (rad/s)",
                    height=600
                )
            elif view_dp == "Phase Space (θ₂-ω₂)":
                fig.update_layout(
                    xaxis_title="θ₂ (degrees)",
                    yaxis_title="ω₂ (rad/s)",
                    height=600
                )
            elif view_dp == "Configuration Space (θ₁-θ₂)":
                fig.update_layout(
                    xaxis_title="θ₁ (degrees)",
                    yaxis_title="θ₂ (degrees)",
                    height=600
                )
            else:  # Energy Evolution
                fig.update_layout(
                    xaxis_title="Time (s)",
                    yaxis_title="Total Energy (J)",
                    height=600
                )

            # Add animation controls
            fig.update_layout(
                title="Double Pendulum Animation",
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
                                'frame': {'duration': animation_speed_dp, 'redraw': True},
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
                        'label': f't={i*step_size*dt_dp:.2f}s',
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
            
            if view_dp == "Full Motion":
                # Show final position and full trail
                for j, (traj, cart) in enumerate(zip(trajectories, cartesian_trajs)):
                    # Trail
                    fig.add_trace(go.Scatter(
                        x=cart['x2'], 
                        y=cart['y2'],
                        mode='lines',
                        line=dict(color=colors[j % len(colors)], width=1, opacity=0.3),
                        name=f'Trail {j+1}'
                    ))
                    # Final pendulum position
                    fig.add_trace(go.Scatter(
                        x=[0, cart['x1'][-1], cart['x2'][-1]], 
                        y=[0, cart['y1'][-1], cart['y2'][-1]],
                        mode='lines+markers',
                        line=dict(color=colors[j % len(colors)], width=3),
                        marker=dict(size=[8, 10, 10]),
                        name=f'Pendulum {j+1}'
                    ))
                max_range = 1.1 * (L1 + L2)
                fig.update_layout(
                    xaxis=dict(range=[-max_range, max_range], title="x"),
                    yaxis=dict(range=[-max_range, 0.5], title="y"),
                    aspectratio=dict(x=1, y=1),
                    height=600
                )
            
            elif view_dp == "Phase Space (θ₁-ω₁)":
                for j, traj in enumerate(trajectories):
                    fig.add_trace(go.Scatter(
                        x=np.rad2deg(traj[:, 0]), 
                        y=traj[:, 1],
                        mode='lines',
                        line=dict(color=colors[j % len(colors)], width=2),
                        name=f'Pendulum {j+1}'
                    ))
                fig.update_layout(xaxis_title="θ₁ (degrees)", yaxis_title="ω₁ (rad/s)", height=600)
            
            elif view_dp == "Phase Space (θ₂-ω₂)":
                for j, traj in enumerate(trajectories):
                    fig.add_trace(go.Scatter(
                        x=np.rad2deg(traj[:, 2]), 
                        y=traj[:, 3],
                        mode='lines',
                        line=dict(color=colors[j % len(colors)], width=2),
                        name=f'Pendulum {j+1}'
                    ))
                fig.update_layout(xaxis_title="θ₂ (degrees)", yaxis_title="ω₂ (rad/s)", height=600)
            
            elif view_dp == "Configuration Space (θ₁-θ₂)":
                for j, traj in enumerate(trajectories):
                    fig.add_trace(go.Scatter(
                        x=np.rad2deg(traj[:, 0]), 
                        y=np.rad2deg(traj[:, 2]),
                        mode='lines',
                        line=dict(color=colors[j % len(colors)], width=2),
                        name=f'Pendulum {j+1}'
                    ))
                fig.update_layout(xaxis_title="θ₁ (degrees)", yaxis_title="θ₂ (degrees)", height=600)
            
            else:  # Energy Evolution
                for j, energy in enumerate(energies):
                    fig.add_trace(go.Scatter(
                        x=time_points, 
                        y=energy,
                        mode='lines',
                        line=dict(color=colors[j % len(colors)], width=2),
                        name=f'Pendulum {j+1} Energy'
                    ))
                fig.update_layout(xaxis_title="Time (s)", yaxis_title="Total Energy (J)", height=600)
            
            fig.update_layout(
                title="Double Pendulum", 
                showlegend=True,
                margin=dict(l=0, r=0, b=0, t=30)
            )
            
            st.plotly_chart(fig, use_container_width=True)

    # Show existing plot if available
    elif 'dp_trajectories' in st.session_state and st.session_state.dp_trajectories is not None:
        trajectories = st.session_state.dp_trajectories
        cartesian_trajs = st.session_state.dp_cartesian
        energies = st.session_state.dp_energies
        time_points = st.session_state.dp_time
        
        fig = go.Figure()
        colors = ["blue", "red"]
        
        # [Same static plot code as above]
        if view_dp == "Full Motion":
            for j, (traj, cart) in enumerate(zip(trajectories, cartesian_trajs)):
                fig.add_trace(go.Scatter(
                    x=cart['x2'], y=cart['y2'],
                    mode='lines',
                    line=dict(color=colors[j % len(colors)], width=1, opacity=0.3),
                    name=f'Trail {j+1}'
                ))
                fig.add_trace(go.Scatter(
                    x=[0, cart['x1'][-1], cart['x2'][-1]], 
                    y=[0, cart['y1'][-1], cart['y2'][-1]],
                    mode='lines+markers',
                    line=dict(color=colors[j % len(colors)], width=3),
                    marker=dict(size=[8, 10, 10]),
                    name=f'Pendulum {j+1}'
                ))
            max_range = 1.1 * (L1 + L2)
            fig.update_layout(
                xaxis=dict(range=[-max_range, max_range], title="x"),
                yaxis=dict(range=[-max_range, 0.5], title="y"),
                aspectratio=dict(x=1, y=1),
                height=600
            )
        elif view_dp == "Phase Space (θ₁-ω₁)":
            for j, traj in enumerate(trajectories):
                fig.add_trace(go.Scatter(
                    x=np.rad2deg(traj[:, 0]), y=traj[:, 1],
                    mode='lines',
                    line=dict(color=colors[j % len(colors)], width=2),
                    name=f'Pendulum {j+1}'
                ))
            fig.update_layout(xaxis_title="θ₁ (degrees)", yaxis_title="ω₁ (rad/s)", height=600)
        elif view_dp == "Phase Space (θ₂-ω₂)":
            for j, traj in enumerate(trajectories):
                fig.add_trace(go.Scatter(
                    x=np.rad2deg(traj[:, 2]), y=traj[:, 3],
                    mode='lines',
                    line=dict(color=colors[j % len(colors)], width=2),
                    name=f'Pendulum {j+1}'
                ))
            fig.update_layout(xaxis_title="θ₂ (degrees)", yaxis_title="ω₂ (rad/s)", height=600)
        elif view_dp == "Configuration Space (θ₁-θ₂)":
            for j, traj in enumerate(trajectories):
                fig.add_trace(go.Scatter(
                    x=np.rad2deg(traj[:, 0]), y=np.rad2deg(traj[:, 2]),
                    mode='lines',
                    line=dict(color=colors[j % len(colors)], width=2),
                    name=f'Pendulum {j+1}'
                ))
            fig.update_layout(xaxis_title="θ₁ (degrees)", yaxis_title="θ₂ (degrees)", height=600)
        else:  # Energy Evolution
            for j, energy in enumerate(energies):
                fig.add_trace(go.Scatter(
                    x=time_points, y=energy,
                    mode='lines',
                    line=dict(color=colors[j % len(colors)], width=2),
                    name=f'Pendulum {j+1} Energy'
                ))
            fig.update_layout(xaxis_title="Time (s)", yaxis_title="Total Energy (J)", height=600)
        
        fig.update_layout(title="Double Pendulum", showlegend=True, margin=dict(l=0, r=0, b=0, t=30))
        st.plotly_chart(fig, use_container_width=True)
    
    else:
        st.info("👆 Click 'Generate Animation' or 'Generate Static Plot' to visualize the double pendulum")

    #


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
        dt = st.slider("Time Step (Δt)", 0.001, 0.05, 0.01, 0.001, key="lorenz_dt")

    # ------------------ Options ------------------
    colO1, colO2, colO3 = st.columns(3)
    with colO1:
        show_equations = st.checkbox("Show equations", True, key="lorenz_eq")
    with colO2:
        enable_downloads = st.checkbox("Enable downloads", True, key="lorenz_dl")
    with colO3:
        animation_frames = st.slider("Animation frames", 50, 200, 100, key="lorenz_frames")

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
                           help="Lower values = faster animation",
                           key="lorenz_speed")

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
