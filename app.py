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
from plotly.subplots import make_subplots

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

def analyze_duffing_bifurcations(gamma_vals, x_vals, delta, alpha, beta, omega):
    """Automatically detect bifurcation points from the data"""
    bifurcation_points = {}
    
    # Group data by parameter value
    from collections import defaultdict
    param_groups = defaultdict(list)
    for g, x in zip(gamma_vals, x_vals):
        param_groups[g].append(x)
    
    # Analyze each parameter value
    gamma_sorted = sorted(param_groups.keys())
    periods = []
    
    for g in gamma_sorted:
        values = param_groups[g]
        if len(values) > 2:
            # Detect unique values (approximate due to numerical errors)
            unique_vals = []
            for v in values:
                is_new = True
                for u in unique_vals:
                    if abs(v - u) < 0.01:  # tolerance
                        is_new = False
                        break
                if is_new:
                    unique_vals.append(v)
            periods.append((g, len(unique_vals)))
    
    # Detect transitions
    if periods:
        last_period = periods[0][1]
        for g, p in periods:
            if p != last_period and p > last_period:
                if last_period == 1 and p == 2:
                    bifurcation_points['first_bifurcation'] = g
                elif p > 3 and 'chaos_onset' not in bifurcation_points:
                    bifurcation_points['chaos_onset'] = g
            last_period = p
    
    # Analytical approximations for Duffing oscillator
    if alpha < 0:  # Double-well potential
        # Linear resonance approximation
        omega_0 = np.sqrt(-alpha)
        if abs(omega - omega_0) < 0.3:  # Near resonance
            bifurcation_points['resonance'] = 2 * delta * omega_0
        
        # Approximate chaos threshold (empirical formula)
        if delta > 0:
            bifurcation_points['chaos_threshold'] = 0.3 * np.sqrt(-alpha) / delta
    
    return bifurcation_points



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
with tabs[0]:
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
            
            # Determine the number of pendulums
            num_pendulums = len(trajectories)
            
            # Create frames based on view
            for k in range(0, num_points, step_size):
                frame_data = []
                
                if view_dp == "Full Motion":
                    # First add all pendulum arm traces
                    for j, (traj, cart) in enumerate(zip(trajectories, cartesian_trajs)):
                        frame_data.append(go.Scatter(
                            x=[0, cart['x1'][k], cart['x2'][k]], 
                            y=[0, cart['y1'][k], cart['y2'][k]],
                            mode='lines+markers',
                            line=dict(color=colors[j % len(colors)], width=3),
                            marker=dict(size=[8, 10, 10]),
                            name=f'Pendulum {j+1}'
                        ))
                    
                    # Then add all trail traces (even if empty for early frames)
                    for j, (traj, cart) in enumerate(zip(trajectories, cartesian_trajs)):
                        if k > 10:
                            trail_start = max(0, k - 50)
                            frame_data.append(go.Scatter(
                                x=cart['x2'][trail_start:k+1], 
                                y=cart['y2'][trail_start:k+1],
                                mode='lines',
                                line=dict(color=colors[j % len(colors)], width=1),
                                opacity=0.3,
                                showlegend=False,
                                name=f'Trail {j+1}'
                            ))
                        else:
                            # Add empty trace to maintain consistent trace count
                            frame_data.append(go.Scatter(
                                x=[], 
                                y=[],
                                mode='lines',
                                line=dict(color=colors[j % len(colors)], width=1),
                                opacity=0.3,
                                showlegend=False,
                                name=f'Trail {j+1}'
                            ))
                
                elif view_dp == "Phase Space (θ₁-ω₁)":
                    # First add all trajectory traces
                    for j, traj in enumerate(trajectories):
                        frame_data.append(go.Scatter(
                            x=np.rad2deg(traj[:k+1, 0]), 
                            y=traj[:k+1, 1],
                            mode='lines',
                            line=dict(color=colors[j % len(colors)], width=2),
                            name=f'Pendulum {j+1}'
                        ))
                    # Then add all marker traces
                    for j, traj in enumerate(trajectories):
                        if k > 0:
                            frame_data.append(go.Scatter(
                                x=[np.rad2deg(traj[k, 0])], 
                                y=[traj[k, 1]],
                                mode='markers',
                                marker=dict(color=colors[j % len(colors)], size=10),
                                showlegend=False
                            ))
                        else:
                            # Add empty marker for first frame
                            frame_data.append(go.Scatter(
                                x=[], 
                                y=[],
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
                    for j, traj in enumerate(trajectories):
                        if k > 0:
                            frame_data.append(go.Scatter(
                                x=[np.rad2deg(traj[k, 2])], 
                                y=[traj[k, 3]],
                                mode='markers',
                                marker=dict(color=colors[j % len(colors)], size=10),
                                showlegend=False
                            ))
                        else:
                            frame_data.append(go.Scatter(
                                x=[], 
                                y=[],
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
                    for j, traj in enumerate(trajectories):
                        if k > 0:
                            frame_data.append(go.Scatter(
                                x=[np.rad2deg(traj[k, 0])], 
                                y=[np.rad2deg(traj[k, 2])],
                                mode='markers',
                                marker=dict(color=colors[j % len(colors)], size=10),
                                showlegend=False
                            ))
                        else:
                            frame_data.append(go.Scatter(
                                x=[], 
                                y=[],
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
                if view_dp == "Full Motion":
                    max_range = 1.1 * (L1 + L2)
                    fig.update_layout(
                        xaxis=dict(range=[-max_range, max_range], title="x", scaleanchor="y", scaleratio=1),
                        yaxis=dict(range=[-max_range, 0.5], title="y"),
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
                        line=dict(color=colors[j % len(colors)], width=1),
                        opacity=0.3,
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
                    xaxis=dict(range=[-max_range, max_range], title="x", scaleanchor="y", scaleratio=1),
                    yaxis=dict(range=[-max_range, 0.5], title="y"),
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
                    line=dict(color=colors[j % len(colors)], width=1),
                    opacity=0.3,
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
                xaxis=dict(range=[-max_range, max_range], title="x", scaleanchor="y", scaleratio=1),
                yaxis=dict(range=[-max_range, 0.5], title="y"),
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
    # Metrics
    st.write("---")
    m1_col, m2_col, m3_col, m4_col = st.columns(4)
    with m1_col:
        st.metric("Total Mass", f"{m1 + m2:.1f} kg")
    with m2_col:
        st.metric("Total Length", f"{L1 + L2:.1f} m")
    with m3_col:
        if 'dp_energies' in st.session_state and st.session_state.dp_energies:
            st.metric("Initial Energy", f"{st.session_state.dp_energies[0][0]:.2f} J")
        else:
            st.metric("Initial Energy", "—")
    with m4_col:
        st.metric("g", f"{g:.2f} m/s²")

    # Downloads - only show if trajectory data exists
    if 'dp_trajectories' in st.session_state and st.session_state.dp_trajectories is not None and enable_downloads_dp:
        st.write("### 📥 Download Options")
        
        trajectories = st.session_state.dp_trajectories
        time_points = st.session_state.dp_time
        
        # CSV downloads
        col1, col2, col3 = st.columns(3)
        
        for j in range(len(trajectories)):
            csv_lines = ["t,theta1,omega1,theta2,omega2,x1,y1,x2,y2"]
            cart = st.session_state.dp_cartesian[j]
            for i in range(len(trajectories[j])):
                t_val = time_points[i]
                theta1, omega1, theta2, omega2 = trajectories[j][i]
                csv_lines.append(f"{t_val:.6f},{theta1:.6f},{omega1:.6f},{theta2:.6f},{omega2:.6f},"
                               f"{cart['x1'][i]:.6f},{cart['y1'][i]:.6f},{cart['x2'][i]:.6f},{cart['y2'][i]:.6f}")
            csv_data = "\n".join(csv_lines)
            
            with col1 if j == 0 else col2:
                st.download_button(
                    label=f"📊 Pendulum {j+1} (CSV)",
                    data=csv_data,
                    file_name=f"double_pendulum_{j+1}.csv",
                    mime="text/csv",
                    key=f"dp_csv_download_{j}"
                )
        
        # Interactive HTML download
        with col3:
            # Create figure for download
            fig_download = go.Figure()
            colors = ["blue", "red"]
            
            if view_dp == "Full Motion":
                for j, cart in enumerate(st.session_state.dp_cartesian):
                    fig_download.add_trace(go.Scatter(
                        x=cart['x2'], y=cart['y2'],
                        mode='lines',
                        line=dict(color=colors[j % len(colors)], width=1),
                        opacity=0.3,
                        name=f'Pendulum {j+1} Trail'
                    ))
            
            html_buf = io.StringIO()
            fig_download.write_html(html_buf, include_plotlyjs="cdn")
            st.download_button(
                label="🌐 Interactive HTML",
                data=html_buf.getvalue().encode(),
                file_name="double_pendulum.html",
                mime="text/html",
                key="dp_html_download"
            )

    # Rich theory section
    with st.expander("📚 Learn More — The Mathematics and Physics of the Double Pendulum"):
        st.markdown(
            r"""
            ### 🎭 A Dance of Complexity
            
            The double pendulum is perhaps the simplest mechanical system that exhibits chaotic behavior. Unlike the single pendulum, 
            which can be solved exactly and shows predictable motion, the double pendulum demonstrates how adding just one more 
            degree of freedom can lead to extraordinarily complex dynamics. This deceptively simple system—just two rigid rods 
            connected by frictionless pivots—has captivated physicists, mathematicians, and artists alike with its hypnotic, 
            unpredictable motion.
            
            ### 🧮 The Lagrangian Formulation
            
            To understand the double pendulum, we must venture into the elegant world of Lagrangian mechanics. Instead of tracking 
            forces directly, we use the principle of least action, working with energies:
            
            **The Lagrangian**: $\mathcal{L} = T - V$
            
            Where:
            - **T** is the total kinetic energy of both masses
            - **V** is the total potential energy
            
            The kinetic energy involves both translational motion of the masses and is given by:
            $$T = \frac{1}{2}m_1(L_1\dot{\theta}_1)^2 + \frac{1}{2}m_2[(L_1\dot{\theta}_1)^2 + (L_2\dot{\theta}_2)^2 + 2L_1L_2\dot{\theta}_1\dot{\theta}_2\cos(\theta_1-\theta_2)]$$
            
            The potential energy (taking the pivot as reference):
            $$V = -m_1gL_1\cos\theta_1 - m_2g(L_1\cos\theta_1 + L_2\cos\theta_2)$$
            
            Applying the Euler-Lagrange equations $\frac{d}{dt}\frac{\partial\mathcal{L}}{\partial\dot{\theta}_i} - \frac{\partial\mathcal{L}}{\partial\theta_i} = 0$ 
            yields the complex coupled differential equations shown above.
            
            ### 🌀 Understanding the Variables
            
            When you adjust the sliders, you're exploring a rich parameter space:
            
            - **θ₁, θ₂**: The angles from vertical. Small angles yield nearly linear behavior, while large angles unlock the full nonlinear dynamics
            - **ω₁, ω₂**: Angular velocities. Initial spin can dramatically alter the trajectory
            - **m₁, m₂**: The mass ratio affects the coupling strength between pendulums
            - **L₁, L₂**: Length ratio influences the time scales and resonances
            - **Damping**: Models air resistance and friction, eventually bringing chaos to rest
            
            ### 🔄 The Perturbation Experiment
            
            When you enable "Compare two nearby initial conditions" with perturbation δ, you're conducting a fundamental chaos experiment. 
            The second pendulum starts with angles $(θ_1 + δ, θ_2 + δ)$. This tiny difference—perhaps representing measurement 
            uncertainty or a slight breeze—leads to dramatically different trajectories.
            
            Unlike the exponential divergence in the Lorenz system, the double pendulum shows a more complex sensitivity:
            - **Short term** (~1-5 swings): Trajectories remain close, motion appears deterministic
            - **Medium term** (~5-20 swings): Differences amplify rapidly, trajectories diverge
            - **Long term**: Completely uncorrelated motion, despite identical parameters
            
            ### 🎯 Chaos Without Strange Attractors
            
            The double pendulum's chaos differs fundamentally from the Lorenz system:
            
            **Energy Conservation**: Without damping, total energy $E = T + V$ remains constant. The system explores a 
            3-dimensional energy surface in the 4-dimensional phase space $(θ_1, ω_1, θ_2, ω_2)$.
            
            **No Attractor**: Unlike Lorenz, trajectories don't converge to a fractal set. Instead, they wander chaotically 
            on the energy surface, filling it ergodically over time.
            
            **Poincaré Sections**: Slicing through phase space reveals intricate structures—islands of stability (KAM tori) 
            surrounded by chaotic seas, a hallmark of Hamiltonian chaos.
            
            ### 🌟 Special Behaviors to Explore
            
            The double pendulum exhibits fascinating regimes:
            
            1. **Low Energy**: Both pendulums swing like coupled harmonic oscillators
            2. **Medium Energy**: Complex periodic and quasi-periodic orbits emerge
            3. **High Energy**: Full rotations possible, leading to tumbling chaos
            4. **Resonances**: When $L_1/L_2$ forms simple ratios, special synchronized motions appear
            
            ### 🛠️ Real-World Applications
            
            Far from being just a mathematical curiosity, the double pendulum illuminates:
            
            - **Robotics**: Multi-link robot arms face similar control challenges
            - **Biomechanics**: Human limbs during sports (golf swing, gymnastics) behave like coupled pendulums
            - **Structural Engineering**: Building sway, crane dynamics, and suspension bridge oscillations
            - **Molecular Dynamics**: Simplified model for molecular conformational changes
            - **Space Mechanics**: Tethered satellite systems and space elevator dynamics
            
            ### 🎮 Exploration Guide
            
            To truly appreciate this system's richness, try these experiments:
            
            1. **Find the Separator**: Start with both angles at 180° (straight up). This unstable equilibrium separates 
               rotational from oscillatory motion. A tiny perturbation determines the system's fate.
            
            2. **Energy Transitions**: Begin with low energy (small angles, zero velocity) and gradually increase. Watch the 
               transition from regular to chaotic motion around 60-90° initial angles.
            
            3. **Mass Ratio Effects**: 
               - Set m₂ << m₁: The second pendulum acts like a probe, barely affecting the first
               - Set m₂ >> m₁: The first mass gets whipped around by the second
               - Equal masses: Maximum coupling and energy exchange
            
            4. **Period Doubling**: With specific initial conditions (try θ₁=45°, θ₂=0°, both velocities zero), you might 
               observe period-doubling cascades—a route to chaos discovered by Feigenbaum.
            
            5. **Phase Space Portraits**: The phase space views reveal hidden structure:
               - Closed loops indicate periodic motion
               - Wandering trajectories show chaos
               - Look for separatrices dividing different motion types
            
            6. **Damping Effects**: Add small damping (0.1) and watch how chaos eventually yields to equilibrium. The approach 
               to rest can be surprisingly complex, with temporary "trapping" in metastable states.
            
            ### 🔬 The Butterfly Effect Visualized
            
            The double pendulum provides one of the most visceral demonstrations of sensitive dependence on initial conditions. 
            Unlike weather prediction where we can't see alternate realities, here you can directly observe how microscopic 
            differences (δ = 0.01°) lead to macroscopic divergence. This isn't just mathematical abstraction—it's chaos you 
            can see, measure, and feel.
            
            ### 📐 Advanced Topics
            
            For the mathematically inclined, the double pendulum opens doors to:
            
            - **KAM Theory**: Kolmogorov-Arnold-Moser theory explains the coexistence of regular and chaotic motion
            - **Lyapunov Exponents**: Quantify the rate of trajectory separation
            - **Homoclinic Tangles**: The origin of chaos through intersecting stable and unstable manifolds
            - **Symplectic Integration**: Numerical methods that preserve the Hamiltonian structure
            
            ### 🌈 Aesthetic Mathematics
            
            Beyond science, the double pendulum has inspired artists and designers. Its trajectories create natural fractals, 
            painting abstract art through physics. LED-traced double pendulums produce mesmerizing light paintings, while 
            the motion itself has been choreographed into dance performances—chaos made tangible and beautiful.
            
            Remember: each trajectory you generate is unique in the universe's history. The exact path depends on initial 
            conditions specified to infinite precision, making every run a one-time cosmic event. You're not just observing 
            chaos—you're creating unique, never-to-be-repeated patterns in the mathematical universe.
            """
        )


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
# TAB 3: BIFURCATION DIAGRAMS
# ============================================
with tabs[2]:
    st.header("Bifurcation Diagrams")
    st.write("Explore how system behavior changes with parameters - the roadmap to chaos")
    
    # System selection
    system = st.selectbox(
        "Select System",
        ["Logistic Map", "Lorenz System (ρ variation)", "Duffing Oscillator", "Hénon Map"]
    )
    
    # System-specific parameters
    if system == "Logistic Map":
        st.latex(r"x_{n+1} = r \cdot x_n \cdot (1 - x_n)")
        
        col1, col2, col3 = st.columns(3)
        with col1:
            r_min = st.number_input("r min", value=2.5, step=0.1)
            r_max = st.number_input("r max", value=4.0, step=0.1)
        with col2:
            r_points = st.slider("Parameter points", 100, 2000, 1000)
            iterations = st.slider("Iterations per point", 100, 2000, 1000)
        with col3:
            last_points = st.slider("Points to plot", 10, 200, 100, help="Number of final points to show (after transient)")
            x0_logistic = st.slider("Initial condition x₀", 0.01, 0.99, 0.5, 0.01)
    
    elif system == "Lorenz System (ρ variation)":
        st.latex(r"\dot{x} = \sigma(y-x), \quad \dot{y} = x(\rho-z)-y, \quad \dot{z} = xy - \beta z")
        st.write("Plotting local maxima of x as ρ varies")
        
        col1, col2, col3 = st.columns(3)
        with col1:
            sigma_bif = st.slider("σ", 0.1, 20.0, 10.0, 0.1)
            beta_bif = st.slider("β", 0.1, 10.0, 8/3, 0.1)
        with col2:
            rho_min = st.number_input("ρ min", value=20.0, step=1.0)
            rho_max = st.number_input("ρ max", value=200.0, step=1.0)
        with col3:
            rho_points = st.slider("Parameter points", 100, 1000, 500)
            t_transient = st.slider("Transient time", 10.0, 200.0, 100.0)
            t_collect = st.slider("Collection time", 50.0, 500.0, 200.0)
    
    elif system == "Duffing Oscillator":
        st.latex(r"\ddot{x} + \delta\dot{x} + \alpha x + \beta x^3 = \gamma \cos(\omega t)")
        st.write("Poincaré section at phase ωt = 0")
        
        col1, col2, col3 = st.columns(3)
        with col1:
            delta_duff = st.slider("δ (damping)", 0.0, 1.0, 0.3, 0.01)
            alpha_duff = st.slider("α (linear stiffness)", -2.0, 2.0, -1.0, 0.1)
            beta_duff = st.slider("β (nonlinear stiffness)", 0.0, 2.0, 1.0, 0.1)
        with col2:
            gamma_min = st.number_input("γ min (forcing)", value=0.1, step=0.1)
            gamma_max = st.number_input("γ max (forcing)", value=0.5, step=0.1)
            omega_duff = st.slider("ω (frequency)", 0.5, 2.0, 1.2, 0.1)
        with col3:
            gamma_points = st.slider("Parameter points", 100, 1000, 500)
            periods_transient = st.slider("Transient periods", 50, 500, 200)
            periods_collect = st.slider("Collection periods", 50, 200, 100)
    
    else:  # Hénon Map
        st.latex(r"x_{n+1} = 1 - a x_n^2 + y_n, \quad y_{n+1} = b x_n")
        
        col1, col2, col3 = st.columns(3)
        with col1:
            a_min = st.number_input("a min", value=1.0, step=0.1)
            a_max = st.number_input("a max", value=1.4, step=0.1)
            b_henon = st.slider("b", 0.1, 0.4, 0.3, 0.01)
        with col2:
            a_points = st.slider("Parameter points", 100, 2000, 1000)
            iterations_henon = st.slider("Iterations per point", 100, 2000, 1000)
        with col3:
            last_points_henon = st.slider("Points to plot", 10, 200, 100)
            x0_henon = st.slider("Initial x₀", -2.0, 2.0, 0.0, 0.1)
            y0_henon = st.slider("Initial y₀", -2.0, 2.0, 0.0, 0.1)

    # Visualization options
    st.write("### Visualization Options")
    col1, col2, col3 = st.columns(3)
    with col1:
        color_scheme = st.selectbox("Color scheme", ["Viridis", "Blues", "Plasma", "Inferno", "Twilight"])
    with col2:
        point_size = st.slider("Point size", 0.1, 5.0, 1.0, 0.1)
    with col3:
        show_theory = st.checkbox("Show bifurcation theory", value=True)

    # Generate button
    generate_bifurcation = st.button("🎨 Generate Bifurcation Diagram", type="primary")
    
    if generate_bifurcation:
        with st.spinner("Computing bifurcation diagram..."):
            
            if system == "Logistic Map":
                # Logistic map bifurcation
                r_values = np.linspace(r_min, r_max, r_points)
                bifurcation_data = []
                
                progress_bar = st.progress(0)
                
                for i, r in enumerate(r_values):
                    x = x0_logistic
                    # Transient iterations
                    for _ in range(iterations - last_points):
                        x = r * x * (1 - x)
                    
                    # Collect data
                    r_data = []
                    for _ in range(last_points):
                        x = r * x * (1 - x)
                        r_data.append((r, x))
                    
                    bifurcation_data.extend(r_data)
                    
                    if i % (r_points // 20) == 0:
                        progress_bar.progress(i / r_points)
                
                progress_bar.empty()
                
                # Create plot
                fig = go.Figure()
                r_vals, x_vals = zip(*bifurcation_data)
                fig.add_trace(go.Scattergl(
                    x=r_vals,
                    y=x_vals,
                    mode='markers',
                    marker=dict(
                        size=point_size,
                        color=x_vals,
                        colorscale=color_scheme.lower(),
                        showscale=False,
                        opacity=0.6
                    ),
                    name='Bifurcation'
                ))
                
                fig.update_layout(
                    title="Logistic Map Bifurcation Diagram",
                    xaxis_title="r",
                    yaxis_title="x",
                    height=600,
                    template="plotly_white"
                )
                
                # Add annotations for key bifurcations
                if show_theory:
                    fig.add_vline(x=3.0, line_dash="dash", line_color="red", opacity=0.5)
                    fig.add_annotation(x=3.0, y=0.9, text="Period-2", showarrow=False, textangle=-90)
                    fig.add_vline(x=1+np.sqrt(6), line_dash="dash", line_color="red", opacity=0.5)
                    fig.add_annotation(x=1+np.sqrt(6), y=0.9, text="Period-4", showarrow=False, textangle=-90)
                    fig.add_vline(x=3.57, line_dash="dash", line_color="red", opacity=0.5)
                    fig.add_annotation(x=3.57, y=0.9, text="Chaos", showarrow=False, textangle=-90)
                
                st.plotly_chart(fig, use_container_width=True)
                st.session_state.bifurcation_data = bifurcation_data
                st.session_state.bifurcation_type = system
            
            elif system == "Lorenz System (ρ variation)":
                # Lorenz maxima bifurcation
                rho_values = np.linspace(rho_min, rho_max, rho_points)
                bifurcation_data = []
                
                progress_bar = st.progress(0)
                
                # Pre-allocate for efficiency
                dt_integrate = 0.01  # Fixed integration step
                n_transient = int(t_transient / dt_integrate)
                n_collect = int(t_collect / dt_integrate)
                t_total = (n_transient + n_collect) * dt_integrate
                
                for i, rho in enumerate(rho_values):
                    # Use a simpler integration method for speed
                    x, y, z = 1.0, 1.0, 1.0
                    
                    # Skip transient with simple Euler method (faster)
                    for _ in range(n_transient):
                        dx = sigma_bif * (y - x)
                        dy = x * (rho - z) - y
                        dz = x * y - beta_bif * z
                        x += dx * dt_integrate
                        y += dy * dt_integrate
                        z += dz * dt_integrate
                    
                    # Collect data with RK4 for accuracy
                    x_history = []
                    for _ in range(n_collect):
                        # RK4 step
                        k1x = sigma_bif * (y - x)
                        k1y = x * (rho - z) - y
                        k1z = x * y - beta_bif * z
                        
                        k2x = sigma_bif * ((y + 0.5*k1y*dt_integrate) - (x + 0.5*k1x*dt_integrate))
                        k2y = (x + 0.5*k1x*dt_integrate) * (rho - (z + 0.5*k1z*dt_integrate)) - (y + 0.5*k1y*dt_integrate)
                        k2z = (x + 0.5*k1x*dt_integrate) * (y + 0.5*k1y*dt_integrate) - beta_bif * (z + 0.5*k1z*dt_integrate)
                        
                        x += (k1x + k2x) * dt_integrate / 2
                        y += (k1y + k2y) * dt_integrate / 2
                        z += (k1z + k2z) * dt_integrate / 2
                        
                        x_history.append(x)
                    
                    # Find maxima more efficiently
                    x_array = np.array(x_history)
                    # Use diff instead of comparing neighbors
                    dx = np.diff(x_array)
                    sign_changes = np.diff(np.sign(dx))
                    maxima_indices = np.where(sign_changes < 0)[0] + 1
                    
                    if len(maxima_indices) > 0:
                        # Take last 20 maxima or fewer
                        last_maxima = x_array[maxima_indices[-min(20, len(maxima_indices)):]]
                        for max_val in last_maxima:
                            bifurcation_data.append((rho, max_val))
                    
                    if i % max(1, (rho_points // 20)) == 0:
                        progress_bar.progress((i + 1) / rho_points)
                
                progress_bar.empty()
                
                # Create plot
                fig = go.Figure()
                if bifurcation_data:
                    rho_vals, x_max_vals = zip(*bifurcation_data)
                    fig.add_trace(go.Scattergl(
                        x=rho_vals,
                        y=x_max_vals,
                        mode='markers',
                        marker=dict(
                            size=point_size,
                            color=x_max_vals,
                            colorscale=color_scheme.lower(),
                            showscale=False,
                            opacity=0.6
                        ),
                        name='Maxima'
                    ))
                
                fig.update_layout(
                    title="Lorenz System Bifurcation (Local Maxima of x)",
                    xaxis_title="ρ",
                    yaxis_title="x maxima",
                    height=600,
                    template="plotly_white"
                )
                # Add annotations for key transitions
                if show_theory:
                    # Pitchfork bifurcation
                    fig.add_vline(x=1.0, line_dash="dash", line_color="green", opacity=0.5)
                    fig.add_annotation(x=1.0, y=0.05, text="Pitchfork", showarrow=False, textangle=-90, yref="paper")
                    
                    # For standard parameters σ=10, β=8/3
                    if abs(sigma_bif - 10.0) < 0.1 and abs(beta_bif - 8/3) < 0.1:
                        # Onset of chaos
                        fig.add_vline(x=24.06, line_dash="dash", line_color="red", opacity=0.5)
                        fig.add_annotation(x=24.06, y=0.95, text="Chaos onset", showarrow=False, textangle=-90, yref="paper")
                        
                        # Periodic window
                        fig.add_vline(x=99.65, line_dash="dash", line_color="blue", opacity=0.5)
                        fig.add_annotation(x=99.65, y=0.95, text="Periodic window", showarrow=False, textangle=-90, yref="paper")
                        
                        # Another notable periodic window
                        fig.add_vline(x=160.0, line_dash="dash", line_color="blue", opacity=0.5)
                        fig.add_annotation(x=160.0, y=0.05, text="Period-3", showarrow=False, textangle=-90, yref="paper")
                    else:
                        # Generic annotation for non-standard parameters
                        fig.add_annotation(x=0.5, y=0.95, text=f"σ={sigma_bif:.1f}, β={beta_bif:.2f}", 
                                        xref="paper", yref="paper", showarrow=False)
                
                st.plotly_chart(fig, use_container_width=True)
                st.session_state.bifurcation_data = bifurcation_data
                st.session_state.bifurcation_type = system
            
            elif system == "Duffing Oscillator":
                # Duffing Poincaré section bifurcation
                gamma_values = np.linspace(gamma_min, gamma_max, gamma_points)
                bifurcation_data = []
                
                progress_bar = st.progress(0)
                
                # Pre-compute integration parameters
                period = 2 * np.pi / omega_duff
                dt_duff = period / 100  # 100 steps per period
                
                for i, gamma in enumerate(gamma_values):
                    x, v = 0.1, 0.0  # Initial conditions
                    t = 0.0
                    
                    # Skip transient using RK4 (faster than solve_ivp)
                    for _ in range(periods_transient):
                        # Integrate for one period
                        for _ in range(100):  # 100 steps per period
                            # RK4 integration
                            k1x = v
                            k1v = -delta_duff*v - alpha_duff*x - beta_duff*x**3 + gamma*np.cos(omega_duff*t)
                            
                            k2x = v + 0.5*dt_duff*k1v
                            k2v = -delta_duff*(v + 0.5*dt_duff*k1v) - alpha_duff*(x + 0.5*dt_duff*k1x) - \
                                beta_duff*(x + 0.5*dt_duff*k1x)**3 + gamma*np.cos(omega_duff*(t + 0.5*dt_duff))
                            
                            k3x = v + 0.5*dt_duff*k2v
                            k3v = -delta_duff*(v + 0.5*dt_duff*k2v) - alpha_duff*(x + 0.5*dt_duff*k2x) - \
                                beta_duff*(x + 0.5*dt_duff*k2x)**3 + gamma*np.cos(omega_duff*(t + 0.5*dt_duff))
                            
                            k4x = v + dt_duff*k3v
                            k4v = -delta_duff*(v + dt_duff*k3v) - alpha_duff*(x + dt_duff*k3x) - \
                                beta_duff*(x + dt_duff*k3x)**3 + gamma*np.cos(omega_duff*(t + dt_duff))
                            
                            x += dt_duff * (k1x + 2*k2x + 2*k3x + k4x) / 6
                            v += dt_duff * (k1v + 2*k2v + 2*k3v + k4v) / 6
                            t += dt_duff
                    
                    # Collect Poincaré points
                    poincare_points = []
                    for _ in range(periods_collect):
                        # Integrate for one period and record at t = 0 (mod period)
                        for step in range(100):
                            # Same RK4 as above
                            k1x = v
                            k1v = -delta_duff*v - alpha_duff*x - beta_duff*x**3 + gamma*np.cos(omega_duff*t)
                            
                            k2x = v + 0.5*dt_duff*k1v
                            k2v = -delta_duff*(v + 0.5*dt_duff*k1v) - alpha_duff*(x + 0.5*dt_duff*k1x) - \
                                beta_duff*(x + 0.5*dt_duff*k1x)**3 + gamma*np.cos(omega_duff*(t + 0.5*dt_duff))
                            
                            k3x = v + 0.5*dt_duff*k2v
                            k3v = -delta_duff*(v + 0.5*dt_duff*k2v) - alpha_duff*(x + 0.5*dt_duff*k2x) - \
                                beta_duff*(x + 0.5*dt_duff*k2x)**3 + gamma*np.cos(omega_duff*(t + 0.5*dt_duff))
                            
                            k4x = v + dt_duff*k3v
                            k4v = -delta_duff*(v + dt_duff*k3v) - alpha_duff*(x + dt_duff*k3x) - \
                                beta_duff*(x + dt_duff*k3x)**3 + gamma*np.cos(omega_duff*(t + dt_duff))
                            
                            x += dt_duff * (k1x + 2*k2x + 2*k3x + k4x) / 6
                            v += dt_duff * (k1v + 2*k2v + 2*k3v + k4v) / 6
                            t += dt_duff
                        
                        # Record Poincaré point at the end of each period
                        bifurcation_data.append((gamma, x))
                    
                    if i % max(1, (gamma_points // 20)) == 0:
                        progress_bar.progress((i + 1) / gamma_points)
                
                progress_bar.empty()
                # STORE PARAMETERS for later use
                st.session_state.duffing_params = {
                    'delta': delta_duff,
                    'alpha': alpha_duff,
                    'beta': beta_duff,
                    'omega': omega_duff
                }
    
                # Create plot
                fig = go.Figure()
                if bifurcation_data:
                    gamma_vals, x_vals = zip(*bifurcation_data)
                    fig.add_trace(go.Scattergl(
                        x=gamma_vals,
                        y=x_vals,
                        mode='markers',
                        marker=dict(
                            size=point_size,
                            color=x_vals,
                            colorscale=color_scheme.lower(),
                            showscale=False,
                            opacity=0.6
                        ),
                        name='Poincaré section'
                    ))
                
                fig.update_layout(
                    title="Duffing Oscillator Bifurcation Diagram",
                    xaxis_title="γ (forcing amplitude)",
                    yaxis_title="x (displacement)",
                    height=600,
                    template="plotly_white"
                )
                
                # Add annotations for key transitions 
                if show_theory and bifurcation_data:
                    # NEW: Use the analysis function
                    param_min = min(gamma_vals)
                    param_max = max(gamma_vals)
                    
                    bifurcations = analyze_duffing_bifurcations(
                        gamma_vals, x_vals, 
                        delta_duff, alpha_duff, beta_duff, omega_duff
                    )
                    
                    # Add detected bifurcations
                    for bif_type, bif_value in bifurcations.items():
                        if param_min <= bif_value <= param_max:
                            color = {
                                'first_bifurcation': 'orange',
                                'chaos_onset': 'red',
                                'resonance': 'green',
                                'chaos_threshold': 'purple'
                            }.get(bif_type, 'gray')
                            
                            label = {
                                'first_bifurcation': 'First bifurcation',
                                'chaos_onset': 'Chaos onset',
                                'resonance': 'Linear resonance',
                                'chaos_threshold': 'Chaos threshold'
                            }.get(bif_type, bif_type)
                            
                            fig.add_vline(x=bif_value, line_dash="dash", line_color=color, opacity=0.5)
                            fig.add_annotation(
                                x=bif_value, y=0.95, 
                                text=label,
                                showarrow=False, textangle=-90, yref="paper",
                                bgcolor="rgba(255,255,255,0.8)"
                            )
                
                st.plotly_chart(fig, use_container_width=True)
                st.session_state.bifurcation_data = bifurcation_data
                st.session_state.bifurcation_type = system
            
            
            else:  # Hénon Map
                # Hénon map bifurcation
                a_values = np.linspace(a_min, a_max, a_points)
                bifurcation_data = []
                
                progress_bar = st.progress(0)
                
                for i, a in enumerate(a_values):
                    x, y = x0_henon, y0_henon
                    
                    # Transient iterations
                    for _ in range(iterations_henon - last_points_henon):
                        x_new = 1 - a * x**2 + y
                        y_new = b_henon * x
                        x, y = x_new, y_new
                    
                    # Collect data
                    for _ in range(last_points_henon):
                        x_new = 1 - a * x**2 + y
                        y_new = b_henon * x
                        x, y = x_new, y_new
                        bifurcation_data.append((a, x))
                    
                    if i % max(1, (a_points // 20)) == 0:
                        progress_bar.progress((i + 1) / a_points)
                
                progress_bar.empty()
                
                # Create plot
                fig = go.Figure()
                a_vals, x_vals = zip(*bifurcation_data)
                fig.add_trace(go.Scattergl(
                    x=a_vals,
                    y=x_vals,
                    mode='markers',
                    marker=dict(
                        size=point_size,
                        color=x_vals,
                        colorscale=color_scheme.lower(),
                        showscale=False,
                        opacity=0.6
                    ),
                    name='Bifurcation'
                ))
                
                fig.update_layout(
                    title="Hénon Map Bifurcation Diagram",
                    xaxis_title="a",
                    yaxis_title="x",
                    height=600,
                    template="plotly_white"
                )
                
                # Add annotations for Hénon map BEFORE displaying
                if show_theory:
                    # Add main title annotation about the map properties
                    fig.add_annotation(
                        x=0.5, y=0.98, xref="paper", yref="paper",
                        text="Hénon Map: Period-doubling cascade → Strange attractor",
                        showarrow=False, font=dict(size=12, color="black"),
                        bgcolor="rgba(255,255,255,0.8)"
                    )
                    
                    
                    # Period-2 bifurcation around a = 1.0
                    if a_min <= 1.0 <= a_max:
                        fig.add_vline(x=1.0, line_dash="dash", line_color="red", opacity=0.5)
                        fig.add_annotation(
                            x=1.0, y=0.95, 
                            text="Period-2",
                            showarrow=False, textangle=-90, yref="paper"
                        )
                    
                    # Period-4 bifurcation around a = 1.25
                    if a_min <= 1.25 <= a_max:
                        fig.add_vline(x=1.25, line_dash="dash", line_color="orange", opacity=0.5)
                        fig.add_annotation(
                            x=1.25, y=0.05, 
                            text="Period-4",
                            showarrow=False, textangle=-90, yref="paper"
                        )
                    
                    # Period-8 and beyond (cascade accelerates)
                    if a_min <= 1.315 <= a_max:
                        fig.add_vline(x=1.315, line_dash="dash", line_color="purple", opacity=0.3)
                        fig.add_annotation(
                            x=1.315, y=0.95, 
                            text="Period-8...",
                            showarrow=False, textangle=-90, yref="paper",
                            font=dict(size=9)
                        )
                    
                    # Chaos onset around a = 1.368
                    if a_min <= 1.368 <= a_max:
                        fig.add_vline(x=1.368, line_dash="dash", line_color="darkred", opacity=0.5)
                        fig.add_annotation(
                            x=1.368, y=0.05, 
                            text="Chaos onset",
                            showarrow=False, textangle=-90, yref="paper",
                            font=dict(color="darkred")
                        )
                    
                    # Strange attractor fully developed
                    if a_min <= 1.4 <= a_max and a_max >= 1.4:
                        fig.add_annotation(
                            x=1.4, y=0.7,
                            text="Strange attractor →",
                            showarrow=True, arrowhead=2,
                            ax=40, ay=0,
                            bgcolor="rgba(255,255,255,0.8)",
                            font=dict(size=11)
                        )
                    
                    # Highlight fractal structure
                    if a_max > 1.35:
                        # Add shaded region for chaotic regime
                        fig.add_vrect(
                            x0=max(1.368, a_min), x1=a_max,
                            fillcolor="red", opacity=0.05,
                            layer="below", line_width=0
                        )
                        
                        # Note about fractal nature
                        fig.add_annotation(
                            x=0.98, y=0.02, xref="paper", yref="paper",
                            text=f"b = {b_henon:.3f} | Fractal dimension ≈ 1.26",
                            showarrow=False, font=dict(size=9),
                            bgcolor="rgba(255,255,200,0.8)"
                        )
                    
                    # Show period-doubling accumulation point
                    accumulation = 1.368  # Approximate Feigenbaum point for Hénon
                    if a_min <= accumulation <= a_max:
                        fig.add_annotation(
                            x=accumulation, y=0.5,
                            text="δ ≈ 4.669...",  # Feigenbaum constant
                            showarrow=True, arrowhead=0,
                            ax=-40, ay=-30,
                            bgcolor="rgba(255,255,255,0.9)",
                            font=dict(size=9, color="purple")
                        )
                
                st.plotly_chart(fig, use_container_width=True)
                st.session_state.bifurcation_data = bifurcation_data
                st.session_state.bifurcation_type = system

    # Show existing bifurcation diagram if available
    elif 'bifurcation_data' in st.session_state and st.session_state.bifurcation_data is not None:
        bifurcation_data = st.session_state.bifurcation_data
        system = st.session_state.bifurcation_type
        
        fig = go.Figure()
        param_vals, state_vals = zip(*bifurcation_data)
        fig.add_trace(go.Scattergl(
            x=param_vals,
            y=state_vals,
            mode='markers',
            marker=dict(
                size=point_size,
                color=state_vals,
                colorscale=color_scheme.lower(),
                showscale=False,
                opacity=0.6
            ),
            name='Bifurcation'
        ))
        
        # Update layout based on system type
        if system == "Logistic Map":
            fig.update_layout(
                title="Logistic Map Bifurcation Diagram",
                xaxis_title="r",
                yaxis_title="x",
                height=600,
                template="plotly_white"
            )
        elif system == "Lorenz System (ρ variation)":
            fig.update_layout(
                title="Lorenz System Bifurcation (Local Maxima of x)",
                xaxis_title="ρ",
                yaxis_title="x maxima",
                height=600,
                template="plotly_white"
            )
        elif system == "Duffing Oscillator":
            fig.update_layout(
                title="Duffing Oscillator Bifurcation (Poincaré Section)",
                xaxis_title="γ (forcing amplitude)",
                yaxis_title="x (displacement)",
                height=600,
                template="plotly_white"
            )
        
        elif system == "Hénon Map":
            fig.update_layout(
                title="Hénon Map Bifurcation Diagram",
                xaxis_title="a",
                yaxis_title="x",
                height=600,
                template="plotly_white"
            )
        else:
            fig.update_layout(
                title=f"{system} Bifurcation Diagram",
                xaxis_title="Parameter",
                yaxis_title="State variable",
                height=600,
                template="plotly_white"
            )
        
        # Add system-specific annotations if show_theory is enabled
        if show_theory:
            if system == "Logistic Map":
                fig.add_vline(x=3.0, line_dash="dash", line_color="red", opacity=0.5)
                fig.add_annotation(x=3.0, y=0.9, text="Period-2", showarrow=False, textangle=-90)
                fig.add_vline(x=1+np.sqrt(6), line_dash="dash", line_color="red", opacity=0.5)
                fig.add_annotation(x=1+np.sqrt(6), y=0.9, text="Period-4", showarrow=False, textangle=-90)
                fig.add_vline(x=3.57, line_dash="dash", line_color="red", opacity=0.5)
                fig.add_annotation(x=3.57, y=0.9, text="Chaos", showarrow=False, textangle=-90)
            
            elif system == "Lorenz System (ρ variation)":
                # Pitchfork bifurcation
                fig.add_vline(x=1.0, line_dash="dash", line_color="green", opacity=0.5)
                fig.add_annotation(x=1.0, y=0.05, text="Pitchfork", showarrow=False, textangle=-90, yref="paper")
                
                # For standard parameters σ=10, β=8/3 (we can't check this from stored data, so show generic)
                fig.add_vline(x=24.06, line_dash="dash", line_color="red", opacity=0.5)
                fig.add_annotation(x=24.06, y=0.95, text="Chaos onset", showarrow=False, textangle=-90, yref="paper")
                
                # Periodic window
                fig.add_vline(x=99.65, line_dash="dash", line_color="blue", opacity=0.5)
                fig.add_annotation(x=99.65, y=0.95, text="Periodic window", showarrow=False, textangle=-90, yref="paper")
            
            elif system == "Duffing Oscillator":
                # Get stored parameters
                if 'duffing_params' in st.session_state:
                    params = st.session_state.duffing_params
                    
                    # Use the analysis function
                    bifurcations = analyze_duffing_bifurcations(
                        param_vals, state_vals,
                        params['delta'], params['alpha'], 
                        params['beta'], params['omega']
                    )
                    
                    # Add detected bifurcations
                    for bif_type, bif_value in bifurcations.items():
                        param_min = min(param_vals)
                        param_max = max(param_vals)
                        
                        if param_min <= bif_value <= param_max:
                            color = {
                                'first_bifurcation': 'orange',
                                'chaos_onset': 'red',
                                'resonance': 'green',
                                'chaos_threshold': 'purple'
                            }.get(bif_type, 'gray')
                            
                            label = {
                                'first_bifurcation': 'First bifurcation',
                                'chaos_onset': 'Chaos onset',
                                'resonance': 'Linear resonance',
                                'chaos_threshold': 'Chaos threshold'
                            }.get(bif_type, bif_type)
                            
                            fig.add_vline(x=bif_value, line_dash="dash", line_color=color, opacity=0.5)
                            fig.add_annotation(
                                x=bif_value, y=0.95, 
                                text=label,
                                showarrow=False, textangle=-90, yref="paper",
                                bgcolor="rgba(255,255,255,0.8)"
                            )
                else:
                    param_min = min(param_vals)
                    param_max = max(param_vals)
                    param_range = param_max - param_min
                    
                    # Adaptive annotations based on typical Duffing behavior
                    # These percentages are approximate for typical Duffing systems
                    
                    # Stable region (typically first 40-50% of range before chaos)
                    stable_point = param_min + 0.3 * param_range
                    fig.add_annotation(
                        x=stable_point, y=0.9,
                        text="Stable periodic",
                        showarrow=True, 
                        arrowhead=2,
                        ax=0, ay=-40,
                        bgcolor="rgba(255,255,255,0.8)"
                    )
                    
                    # Bifurcations (typically around 50-60% through range)
                    bifurc_point = param_min + 0.55 * param_range
                    fig.add_vline(x=bifurc_point, line_dash="dash", line_color="orange", opacity=0.5)
                    fig.add_annotation(
                        x=bifurc_point, y=0.5,
                        text="Bifurcations",
                        showarrow=False,
                        textangle=-90,
                        yref="paper",
                        bgcolor="rgba(255,255,255,0.8)"
                    )
                    
                    # Chaos (typically around 60-70% through range)
                    chaos_point = param_min + 0.65 * param_range
                    fig.add_annotation(
                        x=chaos_point, y=0.1,
                        text="Chaos",
                        showarrow=True,
                        arrowhead=2,
                        ax=20, ay=20,
                        bgcolor="rgba(255,255,255,0.8)"
                    )
                    
                    # Note about parameter dependence
                    fig.add_annotation(
                        x=0.98, y=0.02, xref="paper", yref="paper",
                        text=f"γ ∈ [{param_min:.2f}, {param_max:.2f}]",
                        showarrow=False, font=dict(size=9),
                        bgcolor="rgba(255,255,200,0.8)"
                    )
            
            
        st.plotly_chart(fig, use_container_width=True)

    # Metrics
    if 'bifurcation_data' in st.session_state and st.session_state.bifurcation_data:
        st.write("---")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total points", f"{len(st.session_state.bifurcation_data):,}")
        with col2:
            param_vals, state_vals = zip(*st.session_state.bifurcation_data)
            st.metric("Parameter range", f"[{min(param_vals):.3f}, {max(param_vals):.3f}]")
        with col3:
            st.metric("State range", f"[{min(state_vals):.3f}, {max(state_vals):.3f}]")

    # Downloads
    if 'bifurcation_data' in st.session_state and st.session_state.bifurcation_data:
        st.write("### 📥 Download Options")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # CSV download
            csv_lines = ["parameter,value"]
            for param, value in st.session_state.bifurcation_data:
                csv_lines.append(f"{param:.8f},{value:.8f}")
            csv_data = "\n".join(csv_lines)
            
            st.download_button(
                label="📊 Download Data (CSV)",
                data=csv_data,
                file_name=f"bifurcation_{st.session_state.bifurcation_type.lower().replace(' ', '_')}.csv",
                mime="text/csv",
                key="bif_csv_download"
            )
        
        with col2:
            # Create figure for HTML download
            fig_download = go.Figure()
            param_vals, state_vals = zip(*st.session_state.bifurcation_data)
            fig_download.add_trace(go.Scattergl(
                x=param_vals,
                y=state_vals,
                mode='markers',
                marker=dict(size=1, opacity=0.6),
                name='Bifurcation'
            ))
            
            html_buf = io.StringIO()
            fig_download.write_html(html_buf, include_plotlyjs="cdn")
            st.download_button(
                label="🌐 Interactive HTML",
                data=html_buf.getvalue().encode(),
                file_name=f"bifurcation_{st.session_state.bifurcation_type.lower().replace(' ', '_')}.html",
                mime="text/html",
                key="bif_html_download"
            )

        # Theory section
    with st.expander("📚 Learn More — Bifurcation Theory and Routes to Chaos"):
        st.markdown(
            r"""
            ### 🌉 The Architecture of Chaos
            
            Bifurcation diagrams are the roadmaps of nonlinear dynamics. They reveal how a system's behavior changes as we vary 
            a control parameter, exposing the hidden architecture of chaos. Like a genealogy of dynamical behaviors, these diagrams 
            show how simple periodic motion can give birth to complex chaos through a cascade of qualitative changes.
            
            ### 🔀 What is a Bifurcation?
            
            A bifurcation occurs when a small change in a system's parameter causes a qualitative change in its behavior. Imagine 
            slowly turning up the heat under a pot of water—at 100°C, the behavior suddenly changes from calm liquid to violent 
            boiling. That's a bifurcation: a tipping point where the system's fundamental character transforms.
            
            In dynamical systems, bifurcations mark transitions between:
            - Fixed points and periodic orbits
            - Stable and unstable behaviors
            - Order and chaos
            
            ### 🎭 The Main Characters: Types of Bifurcations
            
            **1. Saddle-Node (Fold) Bifurcation**
            - Two fixed points collide and annihilate
            - Creates or destroys equilibria
            - Example: The sudden collapse of ecosystems
            
            **2. Period-Doubling (Flip) Bifurcation**
            - A periodic orbit becomes unstable and splits into a orbit with twice the period
            - The hallmark of the Feigenbaum route to chaos
            - Visible in the logistic map as the cascade of period-2, 4, 8, 16...
            
            **3. Hopf Bifurcation**
            - A fixed point loses stability and births a limit cycle
            - Transforms equilibrium into oscillation
            - Seen in the Lorenz system at ρ ≈ 24.74
            
            **4. Pitchfork Bifurcation**
            - One fixed point becomes three (supercritical) or vice versa
            - Common in symmetric systems
            - The Lorenz system exhibits this at ρ = 1
            
            ### 🗺️ Understanding Each System
            
            **Logistic Map: The Period-Doubling Route**
            
            The logistic map $x_{n+1} = rx_n(1-x_n)$ is the fruit fly of chaos theory—simple yet profound. As you increase $r$:
            - $r < 1$: Population dies out (extinction)
            - $1 < r < 3$: Stable fixed point (sustainable population)
            - $3 < r < 1+\sqrt{6}$: Period-2 cycle (population oscillates between two values)
            - Further increases: Period-4, 8, 16... (period-doubling cascade)
            - $r \approx 3.57$: Chaos onset
            - $r > 3.57$: Chaotic behavior interspersed with periodic windows
            
            The period-doubling intervals follow Feigenbaum's constant: $\delta = 4.669...$, a universal number like π or e!
            
            **Lorenz System: Multiple Routes to Chaos**

            The Lorenz system exhibits a rich sequence of bifurcations as ρ increases:
            - **ρ = 1**: Pitchfork bifurcation - the origin loses stability and two new fixed points appear
            - **1 < ρ < 24.06**: The fixed points C± are stable (for σ=10, β=8/3)
            - **ρ ≈ 24.06**: Subcritical Hopf bifurcation - strange attractor suddenly appears
            - **ρ > 24.74**: Full chaos with strange attractor
            - **Periodic windows**: Islands of periodic behavior within chaos (e.g., ρ ≈ 99.65, 160)

            The exact values depend on σ and β. The famous value ρc ≈ 24.74 is specifically for σ=10, β=8/3.
            
            **Duffing Oscillator: Forced Transitions**
            
            This driven system shows how external forcing can induce chaos:
            - Small γ: Simple periodic response
            - Increasing γ: Period-doubling route to chaos
            - Large γ: Multiple coexisting attractors (multistability)
            
            The Poincaré section cuts through the continuous flow, revealing the discrete map hidden within.
            
            **Van der Pol Oscillator: Relaxation to Chaos**
            
            As μ increases:
            - μ ≈ 0: Nearly harmonic oscillation (circle in phase space)
            - μ > 0: Limit cycle emerges via Hopf bifurcation
            - Large μ: Relaxation oscillations (fast-slow dynamics)
            
            The amplitude approaches 2 for large μ, independent of initial conditions—a robust limit cycle.
            
            **Hénon Map: Strange Attractor Formation**
            
            This 2D map exhibits:
            - Period-doubling cascade similar to logistic map
            - Formation of a fractal strange attractor
            - Sensitive dependence on initial conditions
            
            ### 🔬 Universal Features
            
            Remarkably, different systems share universal properties near bifurcations:
            
            1. **Feigenbaum Constants**: Period-doubling cascades in different systems converge at the same rate
            2. **Normal Forms**: Near bifurcations, complex systems reduce to simple universal equations
            3. **Scaling Laws**: Power laws govern behavior near critical transitions
            
            ### 🌍 Real-World Implications
            
            Bifurcation theory illuminates critical transitions in:
            - **Climate**: Tipping points in Earth's climate system
            - **Ecology**: Sudden ecosystem collapses and regime shifts
            - **Economics**: Market crashes and bubble formations
            - **Neuroscience**: Epileptic seizures as bifurcations in brain dynamics
            - **Engineering**: Flutter in aircraft wings, buckling in structures
            
            ### 🎮 Exploration Guide
            
            1. **Find the Cascade**: In the logistic map, zoom into the region 2.8 < r < 3.6 to see the period-doubling cascade 
               in exquisite detail. Can you spot the self-similar structure?
            
            2. **Hunt for Windows**: In the chaotic region (r > 3.57), look for vertical white stripes—these are periodic 
               windows. The largest occurs at r ≈ 3.83 (period-3).
            
            3. **Parameter Sensitivity**: Make tiny changes to parameters near bifurcation points. Notice how behavior changes 
               dramatically—this sensitivity is why weather prediction is fundamentally limited.
            
            4. **Coexisting Attractors**: In the Duffing system, different initial conditions can lead to different attractors 
               at the same parameter value. This multistability is crucial in many applications.
            
            5. **Transient Chaos**: Some parameter values show chaotic behavior that eventually settles to periodic motion. 
               Increase the transient time to ensure you're seeing the true long-term behavior.
            
            ### 📊 Reading Bifurcation Diagrams
            
            - **Vertical slices**: All possible long-term behaviors at that parameter value
            - **Continuous curves**: Stable periodic behavior
            - **Fuzzy regions**: Chaotic behavior (many possible values)
            - **White space**: Unstable or transient behaviors not captured
            - **Sudden jumps**: Bifurcation points where behavior changes qualitatively
            
            Remember: these diagrams compress infinite-time behavior into a single image. Each point represents where the system 
            settles after transients die out—the mathematical equivalent of patience rewarded with insight.
            """
        )

with tabs[3]:  # Lyapunov Exponents tab
    st.header("Lyapunov Exponents Analysis")
    st.write("Explore how sensitive dynamical systems are to initial conditions through Lyapunov exponents.")
    
    # Create columns for controls
    col1, col2, col3 = st.columns([1, 1, 1])
    
    with col1:
        system = st.selectbox(
            "Select System",
            ["Logistic Map", "Lorenz System", "Duffing Oscillator", "Van der Pol Oscillator", "Hénon Map"],
            key="lyapunov_system"
        )
    
    with col2:
        viz_type = st.selectbox(
            "Visualization Type",
            ["LLE vs Parameter", "Convergence Plot", "Finite-Time Heatmap", "Spectrum Analysis"],
            key="lyapunov_viz"
        )
    
    with col3:
        color_scheme = st.selectbox(
            "Color Scheme",
            ["Viridis", "Plasma", "Turbo", "RdBu", "Coolwarm"],
            key="lyapunov_colors"
        )
    
    # Add explanation for selected visualization type
    st.markdown("---")
    
    if viz_type == "LLE vs Parameter":
        st.info("""
        ### 📊 **Largest Lyapunov Exponent (LLE) vs Parameter**
        
        This plot shows how the largest Lyapunov exponent changes as we vary a control parameter.
        
        **What it tells you:**
        - **LLE > 0** (above red line): System is chaotic - nearby trajectories diverge exponentially
        - **LLE = 0** (on red line): Marginally stable - often indicates periodic behavior
        - **LLE < 0** (below red line): System is stable - trajectories converge
        
        **How to read it:**
        - Sharp drops to negative values indicate periodic windows within chaos
        - Smooth transitions suggest gradual onset of chaos
        - Compare with bifurcation diagram to see chaos-order correspondence
        """)
    
    elif viz_type == "Convergence Plot":
        st.info("""
        ### 📈 **Lyapunov Exponent Convergence**
        
        Shows how the Lyapunov exponent estimate evolves over time and converges to its true value.
        
        **What it tells you:**
        - **Top panel**: Instantaneous stretching rates (noisy but revealing)
        - **Bottom panel**: Running average that should stabilize
        
        **How to read it:**
        - Fast convergence: System has strong, consistent behavior
        - Slow convergence: System may have multiple timescales or intermittency
        - Oscillating convergence: Possible periodic windows or transient chaos
        
        **Quality check:**
        - Good convergence: Flat line after sufficient time
        - Poor convergence: Still trending or oscillating - need longer integration
        """)
    
    elif viz_type == "Finite-Time Heatmap":
        st.info("""
        ### 🗺️ **Finite-Time Lyapunov Exponent (FTLE) Heatmap**
        
        Maps short-term predictability across parameter and initial condition space.
        
        **What it tells you:**
        - **Red regions**: Chaotic behavior even on short timescales
        - **Blue regions**: Stable or periodic behavior
        - **Sharp boundaries**: Separatrices between different dynamical regimes
        
        **How to read it:**
        - Horizontal stripes: Parameter controls dynamics regardless of initial condition
        - Vertical stripes: Initial condition matters more than parameter
        - Complex patterns: Multiple attractors or fractal basin boundaries
        
        **Applications:**
        - Identify safe operating regions
        - Find basin boundaries
        - Detect transient chaos
        """)
    
    elif viz_type == "Spectrum Analysis":
        st.info("""
        ### 📊 **Full Lyapunov Spectrum**
        
        Computes all Lyapunov exponents, revealing the complete stretching/contracting behavior.
        
        **What it tells you:**
        - **Number of positive λs**: Dimension of chaotic motion
        - **λ₁ > 0, λ₂ ≈ 0, λ₃ < 0**: Strange attractor (e.g., Lorenz)
        - **Sum of all λs**: Rate of phase space volume change
        
        **Kaplan-Yorke Dimension:**
        - Fractal dimension estimate: D = j + (λ₁+...+λⱼ)/|λⱼ₊₁|
        - Non-integer values indicate fractal structure
        
        **How to read it:**
        - Convergence plot: All exponents should stabilize
        - Bar chart: Ordered from largest to smallest
        - Sum rule: Σλᵢ should equal divergence of flow
        """)
    
    st.markdown("---")
    st.subheader("Parameters")
    
    # System-specific parameters (keeping all the existing parameter inputs)
    if system == "Logistic Map":
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            r_min = st.number_input("r min", value=2.5, min_value=0.0, max_value=4.0, step=0.1)
        with col2:
            r_max = st.number_input("r max", value=4.0, min_value=0.0, max_value=4.0, step=0.1)
        with col3:
            n_params = st.number_input("Parameter points", value=200, min_value=50, max_value=1000, step=50)
        with col4:
            iterations = st.number_input("Iterations", value=1000, min_value=100, max_value=10000, step=100)
    
    elif system == "Lorenz System":
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            sigma = st.number_input("σ", value=10.0, min_value=0.1, max_value=50.0, step=0.5)
        with col2:
            beta = st.number_input("β", value=8/3, min_value=0.1, max_value=10.0, step=0.1)
        with col3:
            rho_min = st.number_input("ρ min", value=0.0, min_value=0.0, max_value=200.0, step=1.0)
        with col4:
            rho_max = st.number_input("ρ max", value=50.0, min_value=0.0, max_value=200.0, step=1.0)
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            n_params = st.number_input("Parameter points", value=100, min_value=20, max_value=500, step=20)
        with col2:
            integration_time = st.number_input("Integration time", value=100.0, min_value=10.0, max_value=1000.0, step=10.0)
        with col3:
            dt = st.number_input("dt", value=0.01, min_value=0.001, max_value=0.1, step=0.001, format="%.3f")
        with col4:
            renorm_interval = st.number_input("Renorm interval", value=1.0, min_value=0.01, max_value=10.0, step=0.1)
    
    elif system == "Duffing Oscillator":
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            delta = st.number_input("δ (damping)", value=0.2, min_value=0.0, max_value=1.0, step=0.05)
        with col2:
            alpha = st.number_input("α", value=1.0, min_value=-2.0, max_value=2.0, step=0.1)
        with col3:
            beta_duff = st.number_input("β", value=1.0, min_value=-2.0, max_value=2.0, step=0.1)
        with col4:
            omega = st.number_input("ω", value=1.0, min_value=0.1, max_value=5.0, step=0.1)
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            gamma_min = st.number_input("γ min", value=0.0, min_value=0.0, max_value=2.0, step=0.05)
        with col2:
            gamma_max = st.number_input("γ max", value=0.5, min_value=0.0, max_value=2.0, step=0.05)
        with col3:
            n_params = st.number_input("Parameter points", value=100, min_value=20, max_value=500, step=20)
        with col4:
            integration_time = st.number_input("Integration time", value=200.0, min_value=50.0, max_value=1000.0, step=50.0)
    
    elif system == "Van der Pol Oscillator":
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            mu_min = st.number_input("μ min", value=0.0, min_value=0.0, max_value=10.0, step=0.1)
        with col2:
            mu_max = st.number_input("μ max", value=5.0, min_value=0.0, max_value=10.0, step=0.1)
        with col3:
            n_params = st.number_input("Parameter points", value=100, min_value=20, max_value=500, step=20)
        with col4:
            integration_time = st.number_input("Integration time", value=200.0, min_value=50.0, max_value=1000.0, step=50.0)
    
    else:  # Hénon Map
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            a_min = st.number_input("a min", value=1.0, min_value=0.0, max_value=2.0, step=0.05)
        with col2:
            a_max = st.number_input("a max", value=1.4, min_value=0.0, max_value=2.0, step=0.05)
        with col3:
            b_henon = st.number_input("b", value=0.3, min_value=0.0, max_value=1.0, step=0.05)
        with col4:
            n_params = st.number_input("Parameter points", value=200, min_value=50, max_value=1000, step=50)
        
        iterations = st.number_input("Iterations", value=5000, min_value=1000, max_value=50000, step=1000)
    
    # Advanced settings expander
    with st.expander("Advanced Settings"):
        col1, col2 = st.columns(2)
        with col1:
            discard_transient = st.slider("Discard transient (%)", 0, 50, 20)
            show_convergence = st.checkbox("Show convergence analysis", value=True)
        with col2:
            multiple_runs = st.number_input("Ensemble runs", value=1, min_value=1, max_value=10)
            show_theory = st.checkbox("Show theoretical predictions", value=True)
    
    # Add convergence-specific controls BEFORE the compute button
    if viz_type == "Convergence Plot":
        st.markdown("---")
        if system == "Logistic Map":
            r_convergence = st.slider("Select r value for convergence analysis", 
                                    r_min, r_max, 3.8, step=0.01,
                                    key="r_convergence_slider")
        elif system == "Lorenz System":
            rho_convergence = st.slider("Select ρ value for convergence analysis", 
                                    rho_min, rho_max, 28.0, step=0.1,
                                    key="rho_convergence_slider")
        elif system == "Hénon Map":
            a_convergence = st.slider("Select a value for convergence analysis", 
                                    a_min, a_max, 1.4, step=0.01,
                                    key="a_convergence_slider")
        elif system == "Duffing Oscillator":
            gamma_convergence = st.slider("Select γ value for convergence analysis", 
                                        gamma_min, gamma_max, (gamma_min + gamma_max)/2, step=0.01,
                                        key="gamma_convergence_slider")
        elif system == "Van der Pol Oscillator":
            mu_convergence = st.slider("Select μ value for convergence analysis", 
                                    mu_min, mu_max, (mu_min + mu_max)/2, step=0.1,
                                    key="mu_convergence_slider")
    
    # Add tips before compute button
    st.markdown("---")
    with st.expander("💡 Quick Tips", expanded=False):
        st.markdown(f"""
        **For {viz_type}:**
        
        - **Computation time**: {"Fast" if system in ["Logistic Map", "Hénon Map"] else "Moderate to slow"}
        - **Recommended settings**: 
          - Parameter points: {200 if viz_type == "LLE vs Parameter" else 100}
          - Integration time: {100 if system == "Lorenz System" else "N/A"}
          - Transient discard: {20}%
                - **What to look for**:
          - Smooth curves indicate good convergence
          - Jumpy results suggest need for more iterations/time
          - Compare with bifurcation diagram for validation
        """)
    
    # Compute button
    if st.button("Compute Lyapunov Exponents", type="primary", use_container_width=True):
        
        if viz_type == "LLE vs Parameter":
            # Compute Largest Lyapunov Exponent vs parameter
            if system == "Logistic Map":
                r_values = np.linspace(r_min, r_max, n_params)
                lyapunov_values = []
                
                progress_bar = st.progress(0)
                
                for i, r in enumerate(r_values):
                    # Analytical formula for logistic map
                    x = 0.5
                    lyap_sum = 0
                    
                    # Skip transient
                    for _ in range(int(iterations * discard_transient / 100)):
                        x = r * x * (1 - x)
                    
                    # Compute Lyapunov exponent
                    for _ in range(iterations):
                        x = r * x * (1 - x)
                        if 0 < x < 1:
                            lyap_sum += np.log(abs(r * (1 - 2*x)))
                    
                    lyapunov_values.append(lyap_sum / iterations)
                    progress_bar.progress((i + 1) / n_params)
                
                progress_bar.empty()
                
                # Create plot
                fig = go.Figure()
                
                # Add Lyapunov exponent trace
                fig.add_trace(go.Scatter(
                    x=r_values,
                    y=lyapunov_values,
                    mode='lines',
                    line=dict(color='blue', width=2),
                    name='Largest Lyapunov Exponent'
                ))
                
                # Add zero line
                fig.add_hline(y=0, line_dash="dash", line_color="red", opacity=0.5)
                
                # Color regions
                fig.add_hrect(y0=0, y1=max(lyapunov_values) + 0.5, 
                             fillcolor="red", opacity=0.1, 
                             annotation_text="Chaotic", annotation_position="top right")
                fig.add_hrect(y0=min(lyapunov_values) - 0.5, y1=0, 
                             fillcolor="green", opacity=0.1, 
                             annotation_text="Stable", annotation_position="bottom right")
                
                if show_theory:
                    # Add bifurcation points
                    fig.add_vline(x=3.0, line_dash="dash", line_color="orange", opacity=0.5)
                    fig.add_annotation(x=3.0, y=max(lyapunov_values)*0.8, text="Period-2", textangle=-90)
                    
                    fig.add_vline(x=1+np.sqrt(6), line_dash="dash", line_color="orange", opacity=0.5)
                    fig.add_annotation(x=1+np.sqrt(6), y=max(lyapunov_values)*0.8, text="Period-4", textangle=-90)
                    
                    fig.add_vline(x=3.57, line_dash="dash", line_color="purple", opacity=0.5)
                    fig.add_annotation(x=3.57, y=max(lyapunov_values)*0.8, text="Chaos onset", textangle=-90)
                
                fig.update_layout(
                    title="Largest Lyapunov Exponent vs Parameter (Logistic Map)",
                    xaxis_title="r",
                    yaxis_title="Largest Lyapunov Exponent",
                    height=600,
                    template="plotly_white"
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                # Show interpretation guide
                with st.expander("📖 How to interpret this plot", expanded=True):
                    st.markdown("""
                    **Key observations:**
                    - The LLE crosses zero around r ≈ 3.57, marking the onset of chaos
                    - Sharp dips to negative values indicate periodic windows within chaotic regions
                    - Maximum LLE occurs at r = 4, where the map is most chaotic
                    - The overall trend shows increasing chaos as r increases
                    
                    **Physical meaning:**
                    - Negative LLE: Perturbations decay exponentially → predictable dynamics
                    - Positive LLE: Perturbations grow exponentially → sensitive dependence on initial conditions
                    - The magnitude indicates how fast predictability is lost
                    """)
                
                # Show statistics
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Max LLE", f"{max(lyapunov_values):.4f}")
                with col2:
                    chaotic_fraction = sum(1 for l in lyapunov_values if l > 0) / len(lyapunov_values)
                    st.metric("Chaotic fraction", f"{chaotic_fraction:.1%}")
                with col3:
                    st.metric("Min LLE", f"{min(lyapunov_values):.4f}")
            
            elif system == "Lorenz System":
                rho_values = np.linspace(rho_min, rho_max, n_params)
                lyapunov_values = []
                
                progress_bar = st.progress(0)
                
                for i, rho in enumerate(rho_values):
                    # Compute Lyapunov exponent for Lorenz system
                    # Initial conditions
                    state = np.array([1.0, 1.0, 1.0])
                    perturbation = np.array([1e-8, 0, 0])
                    perturbation /= np.linalg.norm(perturbation)
                    
                    lyap_sum = 0
                    t = 0
                    n_renorm = 0
                    
                    # Skip transient
                    transient_steps = int(integration_time * discard_transient / 100 / dt)
                    for _ in range(transient_steps):
                        k1 = sigma * (state[1] - state[0])
                        k2 = state[0] * (rho - state[2]) - state[1]
                        k3 = state[0] * state[1] - beta * state[2]
                        state += dt * np.array([k1, k2, k3])
                    
                    # Compute Lyapunov
                    while t < integration_time:
                        # Evolve both trajectories
                        perturbed = state + perturbation
                        
                        # RK4 for main trajectory
                        k1 = np.array([sigma * (state[1] - state[0]),
                                      state[0] * (rho - state[2]) - state[1],
                                      state[0] * state[1] - beta * state[2]])
                        k2_state = state + 0.5 * dt * k1
                        k2 = np.array([sigma * (k2_state[1] - k2_state[0]),
                                      k2_state[0] * (rho - k2_state[2]) - k2_state[1],
                                      k2_state[0] * k2_state[1] - beta * k2_state[2]])
                        state += dt * k2
                        
                        # Same for perturbed
                        k1p = np.array([sigma * (perturbed[1] - perturbed[0]),
                                       perturbed[0] * (rho - perturbed[2]) - perturbed[1],
                                       perturbed[0] * perturbed[1] - beta * perturbed[2]])
                        k2_perturbed = perturbed + 0.5 * dt * k1p
                        k2p = np.array([sigma * (k2_perturbed[1] - k2_perturbed[0]),
                                       k2_perturbed[0] * (rho - k2_perturbed[2]) - k2_perturbed[1],
                                       k2_perturbed[0] * k2_perturbed[1] - beta * k2_perturbed[2]])
                        perturbed += dt * k2p
                        
                        # Update perturbation vector
                        perturbation = perturbed - state
                        
                        # Renormalization
                        if (n_renorm + 1) * renorm_interval < t:
                            d = np.linalg.norm(perturbation)
                            if d > 0:
                                lyap_sum += np.log(d)
                                perturbation /= d
                                n_renorm += 1
                        
                        t += dt
                    
                    if n_renorm > 0:
                        lyapunov_values.append(lyap_sum / (n_renorm * renorm_interval))
                    else:
                        lyapunov_values.append(0.0)
                    
                    progress_bar.progress((i + 1) / n_params)
                
                progress_bar.empty()
                
                # Create plot
                fig = go.Figure()
                
                fig.add_trace(go.Scatter(
                    x=rho_values,
                    y=lyapunov_values,
                    mode='lines',
                    line=dict(color='blue', width=2),
                    name='Largest Lyapunov Exponent'
                ))
                
                fig.add_hline(y=0, line_dash="dash", line_color="red", opacity=0.5)
                
                # Color regions
                if max(lyapunov_values) > 0:
                    fig.add_hrect(y0=0, y1=max(lyapunov_values) * 1.1, 
                                 fillcolor="red", opacity=0.1, 
                                 annotation_text="Chaotic", annotation_position="top right")
                if min(lyapunov_values) < 0:
                    fig.add_hrect(y0=min(lyapunov_values) * 1.1, y1=0, 
                                 fillcolor="green", opacity=0.1, 
                                 annotation_text="Stable", annotation_position="bottom right")
                
                if show_theory:
                    # Add critical points
                    fig.add_vline(x=1.0, line_dash="dash", line_color="green", opacity=0.5)
                    fig.add_annotation(x=1.0, y=0.1, text="Pitchfork", textangle=-90)
                    
                    if rho_max > 24.06:
                        fig.add_vline(x=24.06, line_dash="dash", line_color="purple", opacity=0.5)
                        fig.add_annotation(x=24.06, y=0.1, text="Chaos onset", textangle=-90)
                    
                    if rho_max > 24.74:
                        fig.add_vline(x=24.74, line_dash="dash", line_color="red", opacity=0.5)
                        fig.add_annotation(x=24.74, y=0.1, text="Strange attractor", textangle=-90)
                
                fig.update_layout(
                    title="Largest Lyapunov Exponent vs ρ (Lorenz System)",
                    xaxis_title="ρ",
                    yaxis_title="Largest Lyapunov Exponent",
                    height=600,
                    template="plotly_white"
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                # Show interpretation guide
                with st.expander("📖 How to interpret this plot", expanded=True):
                    st.markdown("""
                    **Key observations:**
                    - The system transitions from stable (negative LLE) to chaotic (positive LLE) as ρ increases
                    - The famous Lorenz attractor appears around ρ ≈ 24.74
                    - Notice the complex structure - not a simple monotonic transition
                    
                    **Physical meaning:**
                    - This models convection: ρ is the Rayleigh number (temperature difference)
                    - Low ρ: Heat conducts smoothly (stable)
                    - High ρ: Turbulent convection (chaotic)
                    - The positive LLE means weather prediction has fundamental limits!
                    """)
                
                # Show statistics
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Max LLE", f"{max(lyapunov_values):.4f}")
                with col2:
                    chaotic_fraction = sum(1 for l in lyapunov_values if l > 0) / len(lyapunov_values)
                    st.metric("Chaotic fraction", f"{chaotic_fraction:.1%}")
                with col3:
                    st.metric("Min LLE", f"{min(lyapunov_values):.4f}")
            
            elif system == "Hénon Map":
                a_values = np.linspace(a_min, a_max, n_params)
                lyapunov_values = []
                
                progress_bar = st.progress(0)
                
                for i, a in enumerate(a_values):
                    # Initial condition
                    x, y = 0.0, 0.0
                    
                    # Skip transient
                    for _ in range(int(iterations * discard_transient / 100)):
                        x_new = 1 - a * x**2 + y
                        y_new = b_henon * x
                        x, y = x_new, y_new
                    
                    # Compute Lyapunov using Jacobian method
                    lyap_sum = 0
                    valid_iterations = 0
                    
                    for _ in range(iterations):
                        # Check if point escaped to infinity
                        if abs(x) > 1e6 or abs(y) > 1e6:
                            break
                            
                        # Jacobian matrix
                        J = np.array([[-2*a*x, 1],
                                    [b_henon, 0]])
                        
                        # Eigenvalues
                        eigenvalues = np.linalg.eigvals(J)
                        max_eigenvalue = np.max(np.abs(eigenvalues))
                        
                        if max_eigenvalue > 0:
                            lyap_sum += np.log(max_eigenvalue)
                            valid_iterations += 1
                        
                        # Update map
                        x_new = 1 - a * x**2 + y
                        y_new = b_henon * x
                        x, y = x_new, y_new
                    
                    if valid_iterations > 0:
                        lyapunov_values.append(lyap_sum / valid_iterations)
                    else:
                        lyapunov_values.append(np.nan)
                    
                    progress_bar.progress((i + 1) / n_params)
                
                progress_bar.empty()
                
                # Create plot
                fig = go.Figure()
                
                # Add Lyapunov exponent trace
                fig.add_trace(go.Scatter(
                    x=a_values,
                    y=lyapunov_values,
                    mode='lines',
                    line=dict(color='blue', width=2),
                    name='Largest Lyapunov Exponent'
                ))
                
                # Add zero line
                fig.add_hline(y=0, line_dash="dash", line_color="red", opacity=0.5)
                
                # Color regions
                max_lyap = np.nanmax(lyapunov_values)
                min_lyap = np.nanmin(lyapunov_values)
                
                if max_lyap > 0:
                    fig.add_hrect(y0=0, y1=max_lyap * 1.1, 
                                fillcolor="red", opacity=0.1, 
                                annotation_text="Chaotic", annotation_position="top right")
                if min_lyap < 0:
                    fig.add_hrect(y0=min_lyap * 1.1, y1=0, 
                                fillcolor="green", opacity=0.1, 
                                annotation_text="Stable", annotation_position="bottom right")
                
                if show_theory:
                    # Add known bifurcation points
                    fig.add_vline(x=0.3675, line_dash="dash", line_color="orange", opacity=0.5)
                    fig.add_annotation(x=0.3675, y=0.1, text="Period-2", textangle=-90)
                    
                    fig.add_vline(x=1.4, line_dash="dash", line_color="purple", opacity=0.5)
                    fig.add_annotation(x=1.4, y=0.1, text="Standard Hénon", textangle=-90)
                
                fig.update_layout(
                    title=f"Largest Lyapunov Exponent vs a (Hénon Map, b = {b_henon})",
                    xaxis_title="a parameter",
                    yaxis_title="Largest Lyapunov Exponent",
                    height=600,
                    template="plotly_white"
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                # Show interpretation guide
                with st.expander("📖 How to interpret this plot", expanded=True):
                    st.markdown(f"""
                    **Key observations:**
                    - The Hénon map exhibits complex behavior as 'a' varies
                    - For b = {b_henon}, the map contracts area by factor {b_henon}
                    - Standard parameters (a=1.4, b=0.3) produce the famous Hénon attractor
                    
                    **Physical meaning:**
                    - This map models folding and stretching in phase space
                    - Positive LLE indicates chaotic dynamics on a strange attractor
                    - The sum of both Lyapunov exponents equals ln(b) = {np.log(b_henon):.3f}
                    """)
                
                # Show statistics
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Max LLE", f"{np.nanmax(lyapunov_values):.4f}")
                with col2:
                    chaotic_fraction = np.sum(np.array(lyapunov_values) > 0) / len([v for v in lyapunov_values if not np.isnan(v)])
                    st.metric("Chaotic fraction", f"{chaotic_fraction:.1%}")
                with col3:
                    st.metric("Min LLE", f"{np.nanmin(lyapunov_values):.4f}")
            
            elif system == "Duffing Oscillator":
                gamma_values = np.linspace(gamma_min, gamma_max, n_params)
                lyapunov_values = []
                
                progress_bar = st.progress(0)
                dt_duff = 0.01
                
                for i, gamma in enumerate(gamma_values):
                    # Initial conditions
                    state = np.array([0.1, 0.0])
                    phase = 0.0
                    perturbation = np.array([1e-8, 1e-8])
                    perturbation /= np.linalg.norm(perturbation)
                    
                    # Skip transient
                    transient_steps = int(integration_time * discard_transient / 100 / dt_duff)
                    for _ in range(transient_steps):
                        x, x_dot = state
                        x_ddot = -delta * x_dot - alpha * x - beta_duff * x**3 + gamma * np.cos(omega * phase)
                        state[0] += x_dot * dt_duff
                        state[1] += x_ddot * dt_duff
                        phase += omega * dt_duff
                    
                    # Compute Lyapunov
                    lyap_sum = 0
                    n_renorm = 0
                    t = 0
                    
                    while t < integration_time:
                        # Current and perturbed states
                        x, x_dot = state
                        perturbed = state + perturbation
                        x_p, x_dot_p = perturbed
                        
                        # Dynamics
                        x_ddot = -delta * x_dot - alpha * x - beta_duff * x**3 + gamma * np.cos(omega * phase)
                        x_ddot_p = -delta * x_dot_p - alpha * x_p - beta_duff * x_p**3 + gamma * np.cos(omega * phase)
                        
                        # Update states
                        state[0] += x_dot * dt_duff
                        state[1] += x_ddot * dt_duff
                        perturbed[0] += x_dot_p * dt_duff
                        perturbed[1] += x_ddot_p * dt_duff
                        
                        # Update perturbation
                        perturbation = perturbed - state
                        
                        # Renormalize
                        if (n_renorm + 1) * 1.0 <= t:  # Every 1 time unit
                            d = np.linalg.norm(perturbation)
                            if d > 0:
                                lyap_sum += np.log(d)
                                perturbation /= d
                                n_renorm += 1
                        
                        phase += omega * dt_duff
                        t += dt_duff
                    
                    if n_renorm > 0:
                        lyapunov_values.append(lyap_sum / n_renorm)
                    else:
                        lyapunov_values.append(0.0)
                    
                    progress_bar.progress((i + 1) / n_params)
                
                progress_bar.empty()
                
                # Create plot
                fig = go.Figure()
                
                fig.add_trace(go.Scatter(
                    x=gamma_values,
                    y=lyapunov_values,
                    mode='lines',
                    line=dict(color='blue', width=2),
                    name='Largest Lyapunov Exponent'
                ))
                
                fig.add_hline(y=0, line_dash="dash", line_color="red", opacity=0.5)
                
                # Color regions
                if max(lyapunov_values) > 0:
                    fig.add_hrect(y0=0, y1=max(lyapunov_values) * 1.1, 
                                fillcolor="red", opacity=0.1, 
                                annotation_text="Chaotic", annotation_position="top right")
                if min(lyapunov_values) < 0:
                    fig.add_hrect(y0=min(lyapunov_values) * 1.1, y1=0, 
                                fillcolor="green", opacity=0.1, 
                                annotation_text="Periodic", annotation_position="bottom right")
                
                fig.update_layout(
                    title=f"Largest Lyapunov Exponent vs γ (Duffing Oscillator)",
                    xaxis_title="Forcing amplitude γ",
                    yaxis_title="Largest Lyapunov Exponent",
                    height=600,
                    template="plotly_white"
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                # Show interpretation guide
                with st.expander("📖 How to interpret this plot", expanded=True):
                    st.markdown(f"""
                    **Key observations:**
                    - Chaos emerges as forcing amplitude γ increases
                    - Multiple transitions between periodic and chaotic behavior
                    - Complex structure due to coexisting attractors
                    
                    **System parameters:**
                    - Damping δ = {delta}
                    - Stiffness α = {alpha}, β = {beta_duff}
                    - Driving frequency ω = {omega}
                    
                    **Physical meaning:**
                    - Models nonlinear vibrations in mechanical systems
                    - Chaos indicates unpredictable oscillations
                    - Important for understanding structural dynamics
                    """)
                
                # Show statistics
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Max LLE", f"{max(lyapunov_values):.4f}")
                with col2:
                    chaotic_fraction = sum(1 for l in lyapunov_values if l > 0) / len(lyapunov_values)
                    st.metric("Chaotic fraction", f"{chaotic_fraction:.1%}")
                with col3:
                    st.metric("Min LLE", f"{min(lyapunov_values):.4f}")
            
            elif system == "Van der Pol Oscillator":
                mu_values = np.linspace(mu_min, mu_max, n_params)
                lyapunov_values = []
                
                progress_bar = st.progress(0)
                dt_vdp = 0.01
                
                for i, mu in enumerate(mu_values):
                    # Initial condition on approximate limit cycle
                    if mu < 0.1:
                        state = np.array([2.0, 0.0])
                    else:
                        A = 2.0 * (1 - 1/(8*mu))
                        state = np.array([A, 0.0])
                    
                    perturbation = np.array([1e-8, 1e-8])
                    perturbation /= np.linalg.norm(perturbation)
                    
                    # Skip transient
                    transient_steps = int(integration_time * discard_transient / 100 / dt_vdp)
                    for _ in range(transient_steps):
                        x, y = state
                        dx = y
                        dy = mu * (1 - x**2) * y - x
                        state[0] += dx * dt_vdp
                        state[1] += dy * dt_vdp
                    
                    # Compute Lyapunov
                    lyap_sum = 0
                    n_renorm = 0
                    t = 0
                    
                    while t < integration_time:
                        # Current and perturbed states
                        x, y = state
                        perturbed = state + perturbation
                        x_p, y_p = perturbed
                        
                        # Dynamics
                        dx = y
                        dy = mu * (1 - x**2) * y - x
                        dx_p = y_p
                        dy_p = mu * (1 - x_p**2) * y_p - x_p
                        
                        # Update
                        state[0] += dx * dt_vdp
                        state[1] += dy * dt_vdp
                        perturbed[0] += dx_p * dt_vdp
                        perturbed[1] += dy_p * dt_vdp
                        
                        perturbation = perturbed - state
                        
                        # Renormalize
                        if (n_renorm + 1) * 1.0 <= t:
                            d = np.linalg.norm(perturbation)
                            if d > 0:
                                lyap_sum += np.log(d)
                                perturbation /= d
                                n_renorm += 1
                        
                        t += dt_vdp
                    
                    if n_renorm > 0:
                        lyapunov_values.append(lyap_sum / n_renorm)
                    else:
                        lyapunov_values.append(0.0)
                    
                    progress_bar.progress((i + 1) / n_params)
                
                progress_bar.empty()
                
                # Create plot
                fig = go.Figure()
                
                fig.add_trace(go.Scatter(
                    x=mu_values,
                    y=lyapunov_values,
                    mode='lines',
                    line=dict(color='blue', width=2),
                    name='Largest Lyapunov Exponent'
                ))
                
                fig.add_hline(y=0, line_dash="dash", line_color="red", opacity=0.5)
                
                # The Van der Pol oscillator should have LLE ≤ 0
                fig.add_hrect(y0=min(lyapunov_values) * 1.1, y1=0.1, 
                            fillcolor="green", opacity=0.1, 
                            annotation_text="Limit Cycle", annotation_position="top right")
                
                fig.update_layout(
                    title="Largest Lyapunov Exponent vs μ (Van der Pol Oscillator)",
                    xaxis_title="μ parameter",
                    yaxis_title="Largest Lyapunov Exponent",
                    height=600,
                    template="plotly_white"
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                # Show interpretation guide
                with st.expander("📖 How to interpret this plot", expanded=True):
                    st.markdown("""
                    **Important:** The Van der Pol oscillator has a globally attracting limit cycle for μ > 0.
                    
                    **What the LLE tells you:**
                    - LLE ≈ 0: You're on the limit cycle (marginally stable)
                    - LLE < 0: Trajectory is converging to the limit cycle
                    - The Van der Pol oscillator cannot exhibit chaos (LLE always ≤ 0)
                    
                    **Physical meaning:**
                    - Models self-sustained oscillations (e.g., electronic circuits)
                    - μ controls the nonlinearity strength
                    - Large μ creates relaxation oscillations
                    """)
                
                # Show statistics
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Max LLE", f"{max(lyapunov_values):.4f}")
                with col2:
                    st.metric("Mean LLE", f"{np.mean(lyapunov_values):.4f}")
                with col3:
                    st.metric("Min LLE", f"{min(lyapunov_values):.4f}")
                
                 # Note about Van der Pol
                st.info("""
                **Note:** All Lyapunov exponents should be ≤ 0 for the Van der Pol oscillator 
                since it has a globally stable limit cycle. Positive values indicate numerical errors
                or insufficient integration time. The system CANNOT exhibit chaos.
                """)
            
        elif viz_type == "Convergence Plot":
            # Show convergence of Lyapunov exponent over time
            if system == "Logistic Map":
                
                
                x = 0.5
                lyap_values = []
                lyap_running_avg = []
                
                 # Skip initial transient
                for _ in range(100):
                    x = r_convergence * x * (1 - x)  # Use r_convergence here
                
                # Compute and track convergence
                lyap_sum = 0
                for i in range(iterations):
                    x = r_convergence * x * (1 - x)
                    if 0 < x < 1:
                        instant_lyap = np.log(abs(r_convergence * (1 - 2*x)))
                        lyap_sum += instant_lyap
                        lyap_values.append(instant_lyap)
                        lyap_running_avg.append(lyap_sum / (i + 1))
                
                                # Create convergence plot
                fig = make_subplots(rows=2, cols=1, 
                                   subplot_titles=("Instantaneous Lyapunov Exponent", 
                                                  "Running Average"),
                                   vertical_spacing=0.1)
                
                # Instantaneous values
                fig.add_trace(go.Scatter(
                    y=lyap_values[:1000],  # Show first 1000 points
                    mode='lines',
                    line=dict(color='lightblue', width=1),
                    name='Instantaneous'
                ), row=1, col=1)
                
                # Running average
                fig.add_trace(go.Scatter(
                    y=lyap_running_avg,
                    mode='lines',
                    line=dict(color='blue', width=2),
                    name='Running Average'
                ), row=2, col=1)
                
                # Add final value line
                final_value = lyap_running_avg[-1]
                fig.add_hline(y=final_value, line_dash="dash", line_color="red", 
                             row=2, col=1, opacity=0.5)
                
                # Add zero line for reference
                fig.add_hline(y=0, line_dash="dot", line_color="gray", 
                             row=2, col=1, opacity=0.3)
                
                fig.update_xaxes(title_text="Iteration", row=2, col=1)
                fig.update_yaxes(title_text="LLE", row=1, col=1)
                fig.update_yaxes(title_text="Average LLE", row=2, col=1)
                
                fig.update_layout(
                    title=f"Lyapunov Exponent Convergence (r = {r_convergence:.2f})",
                    height=700,
                    template="plotly_white",
                    showlegend=False
                )
                
                # Add annotation about convergence quality
                convergence_quality = "Good" if np.std(lyap_running_avg[-100:]) < 0.01 else "Poor"
                fig.add_annotation(
                    x=0.98, y=0.98, xref="paper", yref="paper",
                    text=f"Convergence: {convergence_quality}",
                    showarrow=False, 
                    bgcolor="lightgreen" if convergence_quality == "Good" else "lightcoral"
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                # Show interpretation guide
                with st.expander("📖 How to interpret this plot", expanded=True):
                    st.markdown(f"""
                    **What you're seeing:**
                    - **Top panel**: Individual stretching rates at each iteration - naturally noisy
                    - **Bottom panel**: Average converging to true Lyapunov exponent ≈ {final_value:.4f}
                    
                    **Convergence quality indicators:**
                    - **Good**: Flat line in bottom panel, small fluctuations
                    - **Poor**: Still trending or large oscillations
                    - **Current**: {convergence_quality} (based on last 100 iterations)
                    
                    **For r = {r:.2f}:**
                    - LLE = {final_value:.4f} → System is {"chaotic" if final_value > 0 else "stable"}
                    - Convergence after ~{len(lyap_running_avg)} iterations
                    - {"This is in the chaotic regime" if final_value > 0 else "This is in a stable/periodic window"}
                    """)
                
                # Convergence metrics
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Final LLE", f"{final_value:.4f}")
                with col2:
                    if final_value > 0:
                        st.metric("System State", "Chaotic", delta="Positive LLE")
                    else:
                        st.metric("System State", "Stable", delta="Negative LLE")
                with col3:
                    # Estimate convergence rate
                    last_100 = lyap_running_avg[-100:]
                    variation = np.std(last_100) if len(last_100) > 1 else 0
                    st.metric("Convergence σ", f"{variation:.6f}")

            elif system == "Lorenz System":
                # Initial conditions
                state = np.array([1.0, 1.0, 1.0])
                perturbation = np.array([1e-8, 0, 0])
                perturbation /= np.linalg.norm(perturbation)
                
                lyap_values = []
                lyap_running_avg = []
                
                # Skip initial transient
                t = 0
                transient_time = 50
                while t < transient_time:
                    # RK4 integration for state
                    k1 = np.array([sigma * (state[1] - state[0]),
                                state[0] * (rho_convergence - state[2]) - state[1],
                                state[0] * state[1] - beta * state[2]])
                    k2_state = state + 0.5 * dt * k1
                    k2 = np.array([sigma * (k2_state[1] - k2_state[0]),
                                k2_state[0] * (rho_convergence - k2_state[2]) - k2_state[1],
                                k2_state[0] * k2_state[1] - beta * k2_state[2]])
                    state += dt * k2
                    t += dt
                
                # Compute and track convergence
                t = 0
                lyap_sum = 0
                n_steps = 0
                renorm_count = 0
                
                while t < integration_time:
                    # Store old state for perturbation evolution
                    old_state = state.copy()
                    
                    # Evolve state
                    k1 = np.array([sigma * (state[1] - state[0]),
                                state[0] * (rho_convergence - state[2]) - state[1],
                                state[0] * state[1] - beta * state[2]])
                    k2_state = state + 0.5 * dt * k1
                    k2 = np.array([sigma * (k2_state[1] - k2_state[0]),
                                k2_state[0] * (rho_convergence - k2_state[2]) - k2_state[1],
                                k2_state[0] * k2_state[1] - beta * k2_state[2]])
                    state += dt * k2
                    
                    # Evolve perturbation
                    perturbed = old_state + perturbation
                    k1p = np.array([sigma * (perturbed[1] - perturbed[0]),
                                perturbed[0] * (rho_convergence - perturbed[2]) - perturbed[1],
                                perturbed[0] * perturbed[1] - beta * perturbed[2]])
                    k2_perturbed = perturbed + 0.5 * dt * k1p
                    k2p = np.array([sigma * (k2_perturbed[1] - k2_perturbed[0]),
                                k2_perturbed[0] * (rho_convergence - k2_perturbed[2]) - k2_perturbed[1],
                                k2_perturbed[0] * k2_perturbed[1] - beta * k2_perturbed[2]])
                    perturbed += dt * k2p
                    
                    # Update perturbation vector
                    perturbation = perturbed - state
                    
                    # Renormalization and Lyapunov calculation
                    if (renorm_count + 1) * renorm_interval <= t:
                        d = np.linalg.norm(perturbation)
                        if d > 0:
                            instant_lyap = np.log(d) / renorm_interval
                            lyap_sum += np.log(d)
                            renorm_count += 1
                            
                            lyap_values.append(instant_lyap)
                            lyap_running_avg.append(lyap_sum / (renorm_count * renorm_interval))
                            
                            # Renormalize
                            perturbation /= d
                    
                    t += dt
                    n_steps += 1
                
                # Create convergence plot
                fig = make_subplots(rows=2, cols=1, 
                                subplot_titles=("Instantaneous Lyapunov Exponent", 
                                                "Running Average"),
                                vertical_spacing=0.1)
                
                # Instantaneous values
                fig.add_trace(go.Scatter(
                    y=lyap_values,
                    mode='lines',
                    line=dict(color='lightblue', width=1),
                    name='Instantaneous'
                ), row=1, col=1)
                
                # Running average
                fig.add_trace(go.Scatter(
                    y=lyap_running_avg,
                    mode='lines',
                    line=dict(color='blue', width=2),
                    name='Running Average'
                ), row=2, col=1)
                
                # Add final value line
                if len(lyap_running_avg) > 0:
                    final_value = lyap_running_avg[-1]
                    fig.add_hline(y=final_value, line_dash="dash", line_color="red", 
                                row=2, col=1, opacity=0.5)
                    
                    # Add zero line for reference
                    fig.add_hline(y=0, line_dash="dot", line_color="gray", 
                                row=2, col=1, opacity=0.3)
                
                fig.update_xaxes(title_text="Renormalization Step", row=2, col=1)
                fig.update_yaxes(title_text="LLE", row=1, col=1)
                fig.update_yaxes(title_text="Average LLE", row=2, col=1)
                
                fig.update_layout(
                    title=f"Lyapunov Exponent Convergence (ρ = {rho_convergence:.2f})",
                    height=700,
                    template="plotly_white",
                    showlegend=False
                )
                
                # Add annotation about convergence quality
                if len(lyap_running_avg) > 100:
                    convergence_quality = "Good" if np.std(lyap_running_avg[-100:]) < 0.01 else "Moderate"
                else:
                    convergence_quality = "Insufficient data"
                    
                fig.add_annotation(
                    x=0.98, y=0.98, xref="paper", yref="paper",
                    text=f"Convergence: {convergence_quality}",
                    showarrow=False, 
                    bgcolor="lightgreen" if convergence_quality == "Good" else "lightyellow" if convergence_quality == "Moderate" else "lightcoral"
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                # Show interpretation guide
                with st.expander("📖 How to interpret this plot", expanded=True):
                    if len(lyap_running_avg) > 0:
                        st.markdown(f"""
                        **What you're seeing:**
                        - **Top panel**: Instantaneous stretching rates at each renormalization
                        - **Bottom panel**: Average converging to true Lyapunov exponent ≈ {final_value:.4f}
                        
                        **Convergence quality indicators:**
                        - **Good**: Flat line in bottom panel, small fluctuations
                        - **Moderate**: Some trending but stabilizing
                        - **Poor**: Still trending or large oscillations
                        - **Current**: {convergence_quality}
                        
                        **For ρ = {rho_convergence:.2f}:**
                        - LLE = {final_value:.4f} → System is {"chaotic" if final_value > 0 else "stable/periodic"}
                        - {"This is in the chaotic regime (strange attractor)" if final_value > 0 else "This is below the chaos threshold"}
                        - For reference: Classic Lorenz (ρ=28) has LLE ≈ 0.906
                        """)
                    else:
                        st.markdown("No data computed yet. Adjust parameters and try again.")
                
                # Convergence metrics
                if len(lyap_running_avg) > 0:
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Final LLE", f"{final_value:.4f}")
                    with col2:
                        if final_value > 0:
                            st.metric("System State", "Chaotic", delta="Positive LLE")
                        else:
                            st.metric("System State", "Stable", delta="Negative LLE")
                    with col3:
                        # Estimate convergence rate
                        if len(lyap_running_avg) > 100:
                            last_100 = lyap_running_avg[-100:]
                            variation = np.std(last_100)
                            st.metric("Convergence σ", f"{variation:.6f}")
                        else:
                            st.metric("Convergence σ", "N/A")
            
            elif system == "Hénon Map":
                # Hénon Map convergence plot
                x, y = 0.0, 0.0  # Initial condition
                lyap_values = []
                lyap_running_avg = []
                
                # Skip initial transient
                for _ in range(100):
                    x_new = 1 - a_convergence * x**2 + y
                    y_new = b_henon * x
                    x, y = x_new, y_new
                
                # Compute and track convergence
                lyap_sum = 0
                for i in range(iterations):
                    # Jacobian matrix eigenvalues give local stretching
                    jacobian = np.array([[-2*a_convergence*x, 1],
                                        [b_henon, 0]])
                    eigenvalues = np.linalg.eigvals(jacobian)
                    instant_lyap = np.log(np.max(np.abs(eigenvalues)))
                    
                    lyap_sum += instant_lyap
                    lyap_values.append(instant_lyap)
                    lyap_running_avg.append(lyap_sum / (i + 1))
                    
                    # Update map
                    x_new = 1 - a_convergence * x**2 + y
                    y_new = b_henon * x
                    x, y = x_new, y_new
                
                # Create convergence plot
                fig = make_subplots(rows=2, cols=1, 
                                subplot_titles=("Instantaneous Lyapunov Exponent", 
                                                "Running Average"),
                                vertical_spacing=0.1)
                
                # Show only first part of instantaneous values for clarity
                display_points = min(1000, len(lyap_values))
                fig.add_trace(go.Scatter(
                    y=lyap_values[:display_points],
                    mode='lines',
                    line=dict(color='lightblue', width=1),
                    name='Instantaneous'
                ), row=1, col=1)
                
                # Running average
                fig.add_trace(go.Scatter(
                    y=lyap_running_avg,
                    mode='lines',
                    line=dict(color='blue', width=2),
                    name='Running Average'
                ), row=2, col=1)
                
                # Add final value line
                final_value = lyap_running_avg[-1]
                fig.add_hline(y=final_value, line_dash="dash", line_color="red", 
                            row=2, col=1, opacity=0.5)
                
                # Add zero line
                fig.add_hline(y=0, line_dash="dot", line_color="gray", 
                            row=2, col=1, opacity=0.3)
                
                fig.update_xaxes(title_text="Iteration", row=2, col=1)
                fig.update_yaxes(title_text="LLE", row=1, col=1)
                fig.update_yaxes(title_text="Average LLE", row=2, col=1)
                
                fig.update_layout(
                    title=f"Lyapunov Exponent Convergence (a = {a_convergence:.2f})",
                    height=700,
                    template="plotly_white",
                    showlegend=False
                )
                
                # Convergence quality
                convergence_quality = "Good" if np.std(lyap_running_avg[-100:]) < 0.01 else "Moderate"
                fig.add_annotation(
                    x=0.98, y=0.98, xref="paper", yref="paper",
                    text=f"Convergence: {convergence_quality}",
                    showarrow=False, 
                    bgcolor="lightgreen" if convergence_quality == "Good" else "lightyellow"
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                # Show interpretation
                with st.expander("📖 How to interpret this plot", expanded=True):
                    st.markdown(f"""
                    **For a = {a_convergence:.2f}:**
                    - LLE = {final_value:.4f} → System is {"chaotic" if final_value > 0 else "stable/periodic"}
                    - Convergence after ~{len(lyap_running_avg)} iterations
                    - {"This is in the chaotic regime" if final_value > 0 else "This is in a stable/periodic window"}
                    
                    **Hénon Map specifics:**
                    - Standard chaotic parameters (a=1.4, b=0.3) give LLE ≈ 0.42
                    - The map contracts area by factor b = {b_henon} each iteration
                    - Two Lyapunov exponents sum to ln(b) = {np.log(b_henon):.3f}
                    """)
                
                # Convergence metrics
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Final LLE", f"{final_value:.4f}")
                with col2:
                    if final_value > 0:
                        st.metric("System State", "Chaotic", delta="Positive LLE")
                    else:
                        st.metric("System State", "Stable", delta="Negative LLE")
                with col3:
                    # Estimate convergence rate
                    last_100 = lyap_running_avg[-100:]
                    variation = np.std(last_100) if len(last_100) > 1 else 0
                    st.metric("Convergence σ", f"{variation:.6f}")
            
            elif system == "Duffing Oscillator":
                # Duffing oscillator convergence
                
                
                # Initial conditions
                state = np.array([0.1, 0.0])  # x, x_dot
                phase = 0.0
                
                # For Lyapunov calculation
                perturbation = np.array([1e-8, 1e-8])
                perturbation /= np.linalg.norm(perturbation)
                
                lyap_values = []
                lyap_running_avg = []
                
                dt_duffing = 0.01
                renorm_steps = int(1.0 / dt_duffing)  # Renormalize every 1 time unit
                
                # Skip transient
                transient_steps = int(50 / dt_duffing)
                for _ in range(transient_steps):
                    # Duffing equation: x'' + delta*x' + alpha*x + beta*x^3 = gamma*cos(omega*t)
                    x, x_dot = state
                    x_ddot = -delta * x_dot - alpha * x - beta_duff * x**3 + gamma_convergence * np.cos(omega * phase)
                    
                    # Update state
                    state[0] += x_dot * dt_duffing
                    state[1] += x_ddot * dt_duffing
                    phase += omega * dt_duffing
                
                # Compute Lyapunov
                lyap_sum = 0
                n_renorm = 0
                
                for i in range(int(integration_time / dt_duffing)):
                    # Current state
                    x, x_dot = state
                    
                    # Perturbed state
                    perturbed = state + perturbation
                    x_p, x_dot_p = perturbed
                    
                    # Accelerations
                    x_ddot = -delta * x_dot - alpha * x - beta_duff * x**3 + gamma_convergence * np.cos(omega * phase)
                    x_ddot_p = -delta * x_dot_p - alpha * x_p - beta_duff * x_p**3 + gamma_convergence * np.cos(omega * phase)
                    
                    # Update states
                    state[0] += x_dot * dt_duffing
                    state[1] += x_ddot * dt_duffing
                    
                    perturbed[0] += x_dot_p * dt_duffing
                    perturbed[1] += x_ddot_p * dt_duffing
                    
                    # Update perturbation
                    perturbation = perturbed - state
                    
                    # Renormalize periodically
                    if (i + 1) % renorm_steps == 0:
                        d = np.linalg.norm(perturbation)
                        if d > 0:
                            instant_lyap = np.log(d) / (renorm_steps * dt_duffing)
                            lyap_sum += np.log(d)
                            n_renorm += 1
                            
                            lyap_values.append(instant_lyap)
                            lyap_running_avg.append(lyap_sum / (n_renorm * renorm_steps * dt_duffing))
                            
                            perturbation /= d
                    
                    phase += omega * dt_duffing
                
                # Create plot
                fig = make_subplots(rows=2, cols=1, 
                                subplot_titles=("Instantaneous Lyapunov Exponent", 
                                                "Running Average"),
                                vertical_spacing=0.1)
                
                # Instantaneous values
                fig.add_trace(go.Scatter(
                    y=lyap_values,
                    mode='lines',
                    line=dict(color='lightblue', width=1),
                    name='Instantaneous'
                ), row=1, col=1)
                
                # Running average
                fig.add_trace(go.Scatter(
                    y=lyap_running_avg,
                    mode='lines',
                    line=dict(color='blue', width=2),
                    name='Running Average'
                ), row=2, col=1)
                
                if len(lyap_running_avg) > 0:
                    final_value = lyap_running_avg[-1]
                    fig.add_hline(y=final_value, line_dash="dash", line_color="red", 
                                row=2, col=1, opacity=0.5)
                
                fig.add_hline(y=0, line_dash="dot", line_color="gray", 
                            row=2, col=1, opacity=0.3)
                
                fig.update_xaxes(title_text="Renormalization Step", row=2, col=1)
                fig.update_yaxes(title_text="LLE", row=1, col=1)
                fig.update_yaxes(title_text="Average LLE", row=2, col=1)
                
                fig.update_layout(
                    title=f"Lyapunov Exponent Convergence (γ = {gamma_convergence:.2f})",
                    height=700,
                    template="plotly_white",
                    showlegend=False
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                # Interpretation
                if len(lyap_running_avg) > 0:
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Final LLE", f"{final_value:.4f}")
                    with col2:
                        st.metric("System State", "Chaotic" if final_value > 0 else "Regular")
                    with col3:
                        if len(lyap_running_avg) > 100:
                            st.metric("Convergence σ", f"{np.std(lyap_running_avg[-100:]):.6f}")
            
            elif system == "Van der Pol Oscillator":
                # Van der Pol convergence
                
                
                # Initial conditions on limit cycle
                if mu_convergence < 0.1:
                    state = np.array([2.0, 0.0])
                else:
                    # Approximate limit cycle amplitude
                    A = 2.0 * (1 - 1/(8*mu_convergence))
                    state = np.array([A, 0.0])
                
                perturbation = np.array([1e-8, 1e-8])
                perturbation /= np.linalg.norm(perturbation)
                
                lyap_values = []
                lyap_running_avg = []
                
                dt_vdp = 0.01
                renorm_steps = int(1.0 / dt_vdp)
                
                # Skip transient
                transient_steps = int(100 / dt_vdp)
                for _ in range(transient_steps):
                    x, y = state
                    # Van der Pol: x'' - μ(1-x²)x' + x = 0
                    # As system: x' = y, y' = μ(1-x²)y - x
                    dx = y
                    dy = mu_convergence * (1 - x**2) * y - x
                    
                    state[0] += dx * dt_vdp
                    state[1] += dy * dt_vdp
                
                # Compute Lyapunov
                lyap_sum = 0
                n_renorm = 0
                
                for i in range(int(integration_time / dt_vdp)):
                    # Current state
                    x, y = state
                    
                    # Perturbed state
                    perturbed = state + perturbation
                    x_p, y_p = perturbed
                    
                    # Derivatives
                    dx = y
                    dy = mu_convergence * (1 - x**2) * y - x
                    
                    dx_p = y_p
                    dy_p = mu_convergence * (1 - x_p**2) * y_p - x_p
                    
                    # Update states
                    state[0] += dx * dt_vdp
                    state[1] += dy * dt_vdp
                    
                    perturbed[0] += dx_p * dt_vdp
                    perturbed[1] += dy_p * dt_vdp
                    
                    # Update perturbation
                    perturbation = perturbed - state
                    
                    # Renormalize
                    if (i + 1) % renorm_steps == 0:
                        d = np.linalg.norm(perturbation)
                        if d > 0:
                            instant_lyap = np.log(d) / (renorm_steps * dt_vdp)
                            lyap_sum += np.log(d)
                            n_renorm += 1
                            
                            lyap_values.append(instant_lyap)
                            lyap_running_avg.append(lyap_sum / (n_renorm * renorm_steps * dt_vdp))
                            
                            perturbation /= d
                
                # Create plot
                fig = make_subplots(rows=2, cols=1, 
                                subplot_titles=("Instantaneous Lyapunov Exponent", 
                                                "Running Average"),
                                vertical_spacing=0.1)
                
                fig.add_trace(go.Scatter(
                    y=lyap_values,
                    mode='lines',
                    line=dict(color='lightblue', width=1),
                    name='Instantaneous'
                ), row=1, col=1)
                
                fig.add_trace(go.Scatter(
                    y=lyap_running_avg,
                    mode='lines',
                    line=dict(color='blue', width=2),
                    name='Running Average'
                ), row=2, col=1)
                
                if len(lyap_running_avg) > 0:
                    final_value = lyap_running_avg[-1]
                    fig.add_hline(y=final_value, line_dash="dash", line_color="red", 
                                row=2, col=1, opacity=0.5)
                
                fig.add_hline(y=0, line_dash="dot", line_color="gray", 
                            row=2, col=1, opacity=0.3)
                
                fig.update_xaxes(title_text="Renormalization Step", row=2, col=1)
                fig.update_yaxes(title_text="LLE", row=1, col=1)
                fig.update_yaxes(title_text="Average LLE", row=2, col=1)
                
                fig.update_layout(
                    title=f"Lyapunov Exponent Convergence (μ = {mu_convergence:.2f})",
                    height=700,
                    template="plotly_white",
                    showlegend=False
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                # Note about Van der Pol
                st.info("""
                **Note:** The Van der Pol oscillator has a stable limit cycle for μ > 0, 
                so the largest Lyapunov exponent should be ≤ 0. 
                A value near 0 indicates you're on the limit cycle, 
                while negative values indicate convergence to the cycle.
                """)
                
                if len(lyap_running_avg) > 0:
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Final LLE", f"{final_value:.4f}")
                    with col2:
                        st.metric("System State", "Limit Cycle" if abs(final_value) < 0.1 else "Transient")
                    with col3:
                        if len(lyap_running_avg) > 100:
                            st.metric("Convergence σ", f"{np.std(lyap_running_avg[-100:]):.6f}")
        
        elif viz_type == "Finite-Time Heatmap":
            # Create heatmap of finite-time Lyapunov exponents
            if system == "Logistic Map":
                # Grid of initial conditions and parameters
                r_grid = np.linspace(r_min, r_max, 50)
                x0_grid = np.linspace(0.1, 0.9, 50)
                
                ftle_map = np.zeros((len(x0_grid), len(r_grid)))
                
                progress_bar = st.progress(0)
                window_size = 100
                
                for i, x0 in enumerate(x0_grid):
                    for j, r in enumerate(r_grid):
                        x = x0
                        lyap_sum = 0
                        
                        # Compute finite-time Lyapunov
                        for _ in range(window_size):
                            x = r * x * (1 - x)
                            if 0 < x < 1:
                                lyap_sum += np.log(abs(r * (1 - 2*x)))
                        
                        ftle_map[i, j] = lyap_sum / window_size
                    
                    progress_bar.progress((i + 1) / len(x0_grid))
                
                progress_bar.empty()
                
                # Create heatmap
                fig = go.Figure(data=go.Heatmap(
                    z=ftle_map,
                    x=r_grid,
                    y=x0_grid,
                    colorscale=color_scheme,
                    colorbar=dict(title="FTLE"),
                    hoverongaps=False,
                    hovertemplate="r: %{x}<br>x₀: %{y}<br>FTLE: %{z:.3f}<extra></extra>"
                ))
                
                # Add contour at LLE = 0
                fig.add_contour(
                    z=ftle_map,
                    x=r_grid,
                    y=x0_grid,
                    contours=dict(
                        start=0,
                        end=0,
                        size=0.1,
                        coloring='lines',
                        showlabels=True
                    ),
                    line=dict(color='white', width=2),
                    name='LLE = 0'
                )
                
                fig.update_layout(
                    title=f"Finite-Time Lyapunov Exponent Heatmap (window = {window_size} iterations)",
                    xaxis_title="r parameter",
                    yaxis_title="Initial condition x₀",
                    height=600,
                    template="plotly_white"
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                # Show interpretation guide
                with st.expander("📖 How to interpret this heatmap", expanded=True):
                    st.markdown(f"""
                    **Color coding:**
                    - **Red/Yellow**: Positive FTLE → Chaotic behavior
                    - **Blue/Purple**: Negative FTLE → Stable/periodic behavior
                    - **White contour**: FTLE = 0 boundary
                    
                    **Key patterns:**
                    - **Vertical bands**: Parameter value dominates behavior
                    - **Horizontal variation**: Initial condition sensitivity
                    - **Sharp transitions**: Bifurcation points
                    
                    **Observations:**
                    - Notice how chaos (red) appears around r > 3.57
                    - The map becomes increasingly chaotic as r → 4
                    - Initial conditions matter less in strongly chaotic/stable regions
                    - Complex patterns indicate coexisting attractors
                    """)
                
                # Show regions analysis
                st.markdown("### Region Analysis")
                chaotic_fraction = np.sum(ftle_map > 0) / ftle_map.size
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Chaotic regions", f"{chaotic_fraction:.1%}")
                with col2:
                    st.metric("Max FTLE", f"{np.max(ftle_map):.3f}")
                with col3:
                    st.metric("Min FTLE", f"{np.min(ftle_map):.3f}")

            elif system == "Lorenz System":
                # Grid of initial conditions and parameters
                rho_grid = np.linspace(rho_min, rho_max, 50)
                # Initial conditions: vary x and y, keep z fixed
                x0_grid = np.linspace(-20, 20, 50)
                
                ftle_map = np.zeros((len(x0_grid), len(rho_grid)))
                
                progress_bar = st.progress(0)
                window_time = 10.0  # Finite time window
                
                for i, x0 in enumerate(x0_grid):
                    for j, rho in enumerate(rho_grid):
                        # Initial state
                        state = np.array([x0, x0, 20.0])  # Keep z fixed
                        perturbation = np.array([1e-8, 0, 0])
                        perturbation /= np.linalg.norm(perturbation)
                        
                        lyap_sum = 0
                        n_renorm = 0
                        t = 0
                        
                        # Skip short transient
                        transient_time = 5.0
                        while t < transient_time:
                            k1 = np.array([sigma * (state[1] - state[0]),
                                        state[0] * (rho - state[2]) - state[1],
                                        state[0] * state[1] - beta * state[2]])
                            k2_state = state + 0.5 * dt * k1
                            k2 = np.array([sigma * (k2_state[1] - k2_state[0]),
                                        k2_state[0] * (rho - k2_state[2]) - k2_state[1],
                                        k2_state[0] * k2_state[1] - beta * k2_state[2]])
                            state += dt * k2
                            t += dt
                        
                        # Compute finite-time Lyapunov
                        t = 0
                        while t < window_time:
                            # Evolve state
                            old_state = state.copy()
                            k1 = np.array([sigma * (state[1] - state[0]),
                                        state[0] * (rho - state[2]) - state[1],
                                        state[0] * state[1] - beta * state[2]])
                            k2_state = state + 0.5 * dt * k1
                            k2 = np.array([sigma * (k2_state[1] - k2_state[0]),
                                        k2_state[0] * (rho - k2_state[2]) - k2_state[1],
                                        k2_state[0] * k2_state[1] - beta * k2_state[2]])
                            state += dt * k2
                            
                            # Evolve perturbation
                            perturbed = old_state + perturbation
                            k1p = np.array([sigma * (perturbed[1] - perturbed[0]),
                                        perturbed[0] * (rho - perturbed[2]) - perturbed[1],
                                        perturbed[0] * perturbed[1] - beta * perturbed[2]])
                            k2_perturbed = perturbed + 0.5 * dt * k1p
                            k2p = np.array([sigma * (k2_perturbed[1] - k2_perturbed[0]),
                                        k2_perturbed[0] * (rho - k2_perturbed[2]) - k2_perturbed[1],
                                        k2_perturbed[0] * k2_perturbed[1] - beta * k2_perturbed[2]])
                            perturbed += dt * k2p
                            
                            perturbation = perturbed - state
                            
                            # Renormalize
                            if (n_renorm + 1) * renorm_interval <= t:
                                d = np.linalg.norm(perturbation)
                                if d > 0:
                                    lyap_sum += np.log(d)
                                    perturbation /= d
                                    n_renorm += 1
                            
                            t += dt
                        
                        if n_renorm > 0:
                            ftle_map[i, j] = lyap_sum / (n_renorm * renorm_interval)
                    
                    progress_bar.progress((i + 1) / len(x0_grid))
                
                progress_bar.empty()
                
                # Create heatmap
                fig = go.Figure(data=go.Heatmap(
                    z=ftle_map,
                    x=rho_grid,
                    y=x0_grid,
                    colorscale=color_scheme,
                    colorbar=dict(title="FTLE"),
                    hoverongaps=False,
                    hovertemplate="ρ: %{x}<br>x₀: %{y}<br>FTLE: %{z:.3f}<extra></extra>"
                ))
                
                # Add contour at FTLE = 0
                fig.add_contour(
                    z=ftle_map,
                    x=rho_grid,
                    y=x0_grid,
                    contours=dict(
                        start=0,
                        end=0,
                        size=0.1,
                        coloring='lines',
                        showlabels=True
                    ),
                    line=dict(color='white', width=2),
                    name='FTLE = 0'
                )
                
                fig.update_layout(
                    title=f"Finite-Time Lyapunov Exponent Heatmap (T = {window_time})",
                    xaxis_title="ρ parameter",
                    yaxis_title="Initial condition x₀",
                    height=600,
                    template="plotly_white"
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
            elif system == "Hénon Map":
                # Grid of initial conditions and parameters
                a_grid = np.linspace(a_min, a_max, 50)
                x0_grid = np.linspace(-2, 2, 50)
                
                ftle_map = np.zeros((len(x0_grid), len(a_grid)))
                
                progress_bar = st.progress(0)
                window_size = 100  # iterations
                
                for i, x0 in enumerate(x0_grid):
                    for j, a in enumerate(a_grid):
                        x, y = x0, 0.0
                        
                        # Skip transient - check for escape
                        escaped = False
                        for _ in range(50):
                            if abs(x) > 1e5 or abs(y) > 1e5:  # Check for escape
                                escaped = True
                                break
                            x_new = 1 - a * x**2 + y
                            y_new = b_henon * x
                            x, y = x_new, y_new
                        
                        if escaped:
                            ftle_map[i, j] = np.nan  # Mark as escaped
                            continue
                        
                        # Compute finite-time Lyapunov using Jacobian method
                        lyap_sum = 0
                        x_traj, y_traj = x, y
                        valid_iterations = 0
                        
                        for _ in range(window_size):
                            # Check for escape during computation
                            if abs(x_traj) > 1e5 or abs(y_traj) > 1e5 or np.isnan(x_traj) or np.isnan(y_traj):
                                break
                            
                            # Jacobian matrix
                            J = np.array([[-2*a*x_traj, 1],
                                        [b_henon, 0]])
                            
                            # Check if Jacobian contains valid values
                            if np.any(np.isnan(J)) or np.any(np.isinf(J)):
                                break
                            
                            # Largest eigenvalue magnitude
                            try:
                                eigenvalues = np.linalg.eigvals(J)
                                max_eigenvalue = np.max(np.abs(eigenvalues))
                                
                                if max_eigenvalue > 0 and not np.isnan(max_eigenvalue) and not np.isinf(max_eigenvalue):
                                    lyap_sum += np.log(max_eigenvalue)
                                    valid_iterations += 1
                            except:
                                break
                            
                            # Update
                            x_new = 1 - a * x_traj**2 + y_traj
                            y_new = b_henon * x_traj
                            x_traj, y_traj = x_new, y_new
                        
                        if valid_iterations > 0:
                            ftle_map[i, j] = lyap_sum / valid_iterations
                        else:
                            ftle_map[i, j] = np.nan
                    
                    progress_bar.progress((i + 1) / len(x0_grid))
                
                progress_bar.empty()
                
                # Create heatmap - handle NaN values
                fig = go.Figure(data=go.Heatmap(
                    z=ftle_map,
                    x=a_grid,
                    y=x0_grid,
                    colorscale=color_scheme,
                    colorbar=dict(title="FTLE"),
                    hoverongaps=True,  # Changed to True to handle gaps
                    hovertemplate="a: %{x}<br>x₀: %{y}<br>FTLE: %{z:.3f}<extra></extra>",
                    connectgaps=False  # Don't interpolate over NaN values
                ))
                
                # Add contour at FTLE = 0 - mask NaN values
                masked_ftle = np.ma.array(ftle_map, mask=np.isnan(ftle_map))
                
                fig.add_contour(
                    z=masked_ftle,
                    x=a_grid,
                    y=x0_grid,
                    contours=dict(
                        start=0,
                        end=0,
                        size=0.1,
                        coloring='lines',
                        showlabels=True
                    ),
                    line=dict(color='white', width=2),
                    name='FTLE = 0'
                )
                
                fig.update_layout(
                    title=f"Finite-Time Lyapunov Exponent Heatmap (window = {window_size} iterations)",
                    xaxis_title="a parameter",
                    yaxis_title="Initial condition x₀",
                    height=600,
                    template="plotly_white"
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
            elif system == "Duffing Oscillator":
                # Grid of forcing amplitude and initial position
                gamma_grid = np.linspace(gamma_min, gamma_max, 50)
                x0_grid = np.linspace(-2, 2, 50)
                
                ftle_map = np.zeros((len(x0_grid), len(gamma_grid)))
                
                progress_bar = st.progress(0)
                window_time = 20.0  # Time window
                dt_duff = 0.05
                
                for i, x0 in enumerate(x0_grid):
                    for j, gamma in enumerate(gamma_grid):
                        # Initial state
                        state = np.array([x0, 0.0])
                        phase = 0.0
                        perturbation = np.array([1e-8, 1e-8])
                        perturbation /= np.linalg.norm(perturbation)
                        
                        # Skip transient
                        for _ in range(int(20 / dt_duff)):
                            x, x_dot = state
                            x_ddot = -delta * x_dot - alpha * x - beta_duff * x**3 + gamma * np.cos(omega * phase)
                            state[0] += x_dot * dt_duff
                            state[1] += x_ddot * dt_duff
                            phase += omega * dt_duff
                        
                        # Compute FTLE
                        lyap_sum = 0
                        n_steps = 0
                        
                        for _ in range(int(window_time / dt_duff)):
                            # Current and perturbed states
                            x, x_dot = state
                            perturbed = state + perturbation
                            x_p, x_dot_p = perturbed
                            
                            # Dynamics
                            x_ddot = -delta * x_dot - alpha * x - beta_duff * x**3 + gamma * np.cos(omega * phase)
                            x_ddot_p = -delta * x_dot_p - alpha * x_p - beta_duff * x_p**3 + gamma * np.cos(omega * phase)
                            
                            # Update
                            state[0] += x_dot * dt_duff
                            state[1] += x_ddot * dt_duff
                            perturbed[0] += x_dot_p * dt_duff
                            perturbed[1] += x_ddot_p * dt_duff
                            
                            perturbation = perturbed - state
                            
                            # Renormalize every 1 time unit
                            if (n_steps + 1) % int(1.0 / dt_duff) == 0:
                                d = np.linalg.norm(perturbation)
                                if d > 0:
                                    lyap_sum += np.log(d)
                                    perturbation /= d
                            
                            phase += omega * dt_duff
                            n_steps += 1
                        
                        n_renorm = n_steps // int(1.0 / dt_duff)
                        if n_renorm > 0:
                            ftle_map[i, j] = lyap_sum / n_renorm
                    
                    progress_bar.progress((i + 1) / len(x0_grid))
                
                progress_bar.empty()
                
                # Create heatmap
                fig = go.Figure(data=go.Heatmap(
                    z=ftle_map,
                    x=gamma_grid,
                    y=x0_grid,
                    colorscale=color_scheme,
                    colorbar=dict(title="FTLE"),
                    hoverongaps=False,
                    hovertemplate="γ: %{x}<br>x₀: %{y}<br>FTLE: %{z:.3f}<extra></extra>"
                ))
                
                # Add contour at FTLE = 0
                fig.add_contour(
                    z=ftle_map,
                    x=gamma_grid,
                    y=x0_grid,
                    contours=dict(
                        start=0,
                        end=0,
                        size=0.1,
                        coloring='lines',
                        showlabels=True
                    ),
                    line=dict(color='white', width=2),
                    name='FTLE = 0'
                )
                
                fig.update_layout(
                    title=f"Finite-Time Lyapunov Exponent Heatmap (T = {window_time})",
                    xaxis_title="Forcing amplitude γ",
                    yaxis_title="Initial position x₀",
                    height=600,
                    template="plotly_white"
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
            elif system == "Van der Pol Oscillator":
                # Grid of mu values and initial conditions
                mu_grid = np.linspace(mu_min, mu_max, 50)
                x0_grid = np.linspace(-3, 3, 50)
                
                ftle_map = np.zeros((len(x0_grid), len(mu_grid)))
                
                progress_bar = st.progress(0)
                window_time = 20.0
                dt_vdp = 0.05
                
                for i, x0 in enumerate(x0_grid):
                    for j, mu in enumerate(mu_grid):
                        # Initial state
                        state = np.array([x0, 0.0])
                        perturbation = np.array([1e-8, 1e-8])
                        perturbation /= np.linalg.norm(perturbation)
                        
                        # Skip transient
                        for _ in range(int(20 / dt_vdp)):
                            x, y = state
                            dx = y
                            dy = mu * (1 - x**2) * y - x
                            state[0] += dx * dt_vdp
                            state[1] += dy * dt_vdp
                        
                        # Compute FTLE
                        lyap_sum = 0
                        n_steps = 0
                        
                        for _ in range(int(window_time / dt_vdp)):
                            x, y = state
                            perturbed = state + perturbation
                            x_p, y_p = perturbed
                            
                            # Dynamics
                            dx = y
                            dy = mu * (1 - x**2) * y - x
                            dx_p = y_p
                            dy_p = mu * (1 - x_p**2) * y_p - x_p
                            
                            # Update
                            state[0] += dx * dt_vdp
                            state[1] += dy * dt_vdp
                            perturbed[0] += dx_p * dt_vdp
                            perturbed[1] += dy_p * dt_vdp
                            
                            perturbation = perturbed - state
                            
                            # Renormalize every 1 time unit
                            if (n_steps + 1) % int(1.0 / dt_vdp) == 0:
                                d = np.linalg.norm(perturbation)
                                if d > 0:
                                    lyap_sum += np.log(d)
                                    perturbation /= d
                            
                            n_steps += 1
                        
                        n_renorm = n_steps // int(1.0 / dt_vdp)
                        if n_renorm > 0:
                            ftle_map[i, j] = lyap_sum / n_renorm
                    
                    progress_bar.progress((i + 1) / len(x0_grid))
                
                progress_bar.empty()
                
                # Create heatmap
                fig = go.Figure(data=go.Heatmap(
                    z=ftle_map,
                    x=mu_grid,
                    y=x0_grid,
                    colorscale=color_scheme,
                    colorbar=dict(title="FTLE"),
                    hoverongaps=False,
                    hovertemplate="μ: %{x}<br>x₀: %{y}<br>FTLE: %{z:.3f}<extra></extra>"
                ))
                
                # Add contour at FTLE = 0
                fig.add_contour(
                    z=ftle_map,
                    x=mu_grid,
                    y=x0_grid,
                    contours=dict(
                        start=0,
                        end=0,
                        size=0.1,
                        coloring='lines',
                        showlabels=True
                    ),
                    line=dict(color='white', width=2),
                    name='FTLE = 0'
                )
                
                fig.update_layout(
                    title=f"Finite-Time Lyapunov Exponent Heatmap (T = {window_time})",
                    xaxis_title="μ parameter",
                    yaxis_title="Initial condition x₀",
                    height=600,
                    template="plotly_white"
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                # Note about Van der Pol
                st.info("""
                **Note:** The Van der Pol oscillator always converges to a limit cycle for μ > 0.
                The FTLE values show transient behavior - how fast trajectories approach the cycle.
                Blue regions (negative FTLE) indicate faster convergence to the limit cycle.
                """)
            
            # Common interpretation guide for all systems
            with st.expander("📖 How to interpret this heatmap", expanded=True):
                st.markdown(f"""
                **Color coding:**
                - **Red/Yellow**: Positive FTLE → Chaotic/diverging behavior
                - **Blue/Purple**: Negative FTLE → Stable/converging behavior
                - **White contour**: FTLE = 0 boundary
                
                **Key patterns:**
                - **Vertical bands**: Parameter value dominates behavior
                - **Horizontal bands**: Initial condition insensitive
                - **Complex patterns**: Multiple attractors or basins
                - **Sharp transitions**: Bifurcations or crisis events
                
                **System-specific insights:**
                """)
                
                if system == "Logistic Map":
                    st.markdown("""
                    - Notice the period-doubling cascade as r increases
                    - Chaotic bands appear around r > 3.57
                    - White regions show periodic windows within chaos
                    """)
                elif system == "Lorenz System":
                    st.markdown("""
                    - The famous butterfly attractor appears for ρ > 24.74
                    - Initial conditions converge to the same attractor (horizontal bands)
                    - Complex structure near ρ ≈ 24 shows transition to chaos
                    """)
                elif system == "Hénon Map":
                    st.markdown("""
                    - The standard Hénon attractor exists for a ≈ 1.4
                    - Notice fractal basin boundaries (complex patterns)
                    - Some initial conditions escape to infinity (very positive FTLE)
                    """)
                elif system == "Duffing Oscillator":
                    st.markdown("""
                    - Multiple attractors create complex basin structure
                    - Chaos appears as forcing amplitude γ increases
                    - Initial conditions matter greatly due to coexisting attractors
                    - Fractal basin boundaries appear near chaotic transitions
                    """)
                elif system == "Van der Pol Oscillator":
                    st.markdown("""
                    - All trajectories converge to the same limit cycle
                    - FTLE shows transient behavior before reaching the cycle
                    - Larger μ creates relaxation oscillations with varying convergence rates
                    - No positive FTLE regions (no chaos possible)
                    """)
            
            # Show regions analysis
            st.markdown("### Region Analysis")
            chaotic_fraction = np.sum(ftle_map > 0) / ftle_map.size
            stable_fraction = np.sum(ftle_map < 0) / ftle_map.size
            
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Chaotic regions", f"{chaotic_fraction:.1%}")
            with col2:
                st.metric("Stable regions", f"{stable_fraction:.1%}")
            with col3:
                st.metric("Max FTLE", f"{np.max(ftle_map):.3f}")
            with col4:
                st.metric("Min FTLE", f"{np.min(ftle_map):.3f}")
            
            # Additional analysis box
            with st.expander("🔍 Detailed Analysis", expanded=False):
                st.markdown("""
                ### Understanding Finite-Time Lyapunov Exponents
                
                **What FTLE measures:**
                - Rate of separation between nearby trajectories over a finite time window
                - Unlike asymptotic Lyapunov exponents, FTLE captures transient behavior
                - Useful for identifying coherent structures and transport barriers
                
                **Applications:**
                - **Fluid dynamics**: Finding Lagrangian coherent structures
                - **Weather prediction**: Identifying stable/unstable manifolds
                - **Engineering**: Detecting regions of rapid mixing or transport
                
                **Key differences from asymptotic LLE:**
                1. **Time dependence**: FTLE varies with integration time
                2. **Initial condition sensitivity**: Shows local behavior
                3. **Transient chaos**: Can detect temporary chaotic behavior
                
                **Reading the heatmap:**
                - **Ridges** (high FTLE): Repelling material lines
                - **Valleys** (low FTLE): Attracting material lines
                - **Sharp gradients**: Separatrices between dynamical regimes
                """)
                
                # Parameter-specific insights
                if system == "Lorenz System":
                    st.markdown(f"""
                    **For your parameter range (ρ ∈ [{rho_min}, {rho_max}]):**
                    - Below ρ ≈ 1: All trajectories decay to origin
                    - 1 < ρ < 24.06: Stable fixed points
                    - 24.06 < ρ < 24.74: Metastable chaos
                    - ρ > 24.74: Strange attractor
                    
                    The heatmap shows how these transitions affect different initial conditions.
                    """)
                elif system == "Duffing Oscillator":
                    st.markdown(f"""
                    **For your parameter range (γ ∈ [{gamma_min}, {gamma_max}]):**
                    - Low γ: Simple periodic motion
                    - Intermediate γ: Coexisting periodic attractors
                    - High γ: Chaotic motion with fractal basins
                    
                    The complex patterns indicate sensitive dependence on initial conditions.
                    """)
        
        elif viz_type == "Spectrum Analysis":
            # Full Lyapunov spectrum for multi-dimensional systems
            if system == "Lorenz System":
                rho = st.slider("Select ρ value", rho_min, rho_max, 28.0)
                
                # Initial conditions
                state = np.array([1.0, 1.0, 1.0])
                
                # Initialize 3 orthonormal perturbation vectors
                Q = np.eye(3) * 1e-8
                
                lyap_sums = np.zeros(3)
                n_renorm = 0
                t = 0
                
                # For plotting convergence
                lyap_history = [[], [], []]
                
                progress_bar = st.progress(0)
                
                # Skip transient
                transient_time = 50
                while t < transient_time:
                    # RK4 integration
                    k1 = np.array([sigma * (state[1] - state[0]),
                                  state[0] * (rho - state[2]) - state[1],
                                  state[0] * state[1] - beta * state[2]])
                    k2_state = state + 0.5 * dt * k1
                    k2 = np.array([sigma * (k2_state[1] - k2_state[0]),
                                  k2_state[0] * (rho - k2_state[2]) - k2_state[1],
                                  k2_state[0] * k2_state[1] - beta * k2_state[2]])
                    state += dt * k2
                    t += dt
                
                # Compute spectrum
                t = 0
                while t < integration_time:
                    # Evolve perturbations
                    for j in range(3):
                        perturbed = state + Q[:, j]
                        
                        # RK4 for perturbed trajectory
                        k1p = np.array([sigma * (perturbed[1] - perturbed[0]),
                                       perturbed[0] * (rho - perturbed[2]) - perturbed[1],
                                       perturbed[0] * perturbed[1] - beta * perturbed[2]])
                        k2_perturbed = perturbed + 0.5 * dt * k1p
                        k2p = np.array([sigma * (k2_perturbed[1] - k2_perturbed[0]),
                                       k2_perturbed[0] * (rho - k2_perturbed[2]) - k2_perturbed[1],
                                       k2_perturbed[0] * k2_perturbed[1] - beta * k2_perturbed[2]])
                        perturbed += dt * k2p
                        
                        Q[:, j] = perturbed - state
                    
                    # Evolve main trajectory
                    k1 = np.array([sigma * (state[1] - state[0]),
                                  state[0] * (rho - state[2]) - state[1],
                                  state[0] * state[1] - beta * state[2]])
                    k2_state = state + 0.5 * dt * k1
                    k2 = np.array([sigma * (k2_state[1] - k2_state[0]),
                                  k2_state[0] * (rho - k2_state[2]) - k2_state[1],
                                  k2_state[0] * k2_state[1] - beta * k2_state[2]])
                    state += dt * k2
                    
                    # Gram-Schmidt orthogonalization
                    if (n_renorm + 1) * renorm_interval < t:
                        Q, R = np.linalg.qr(Q)
                        for j in range(3):
                            lyap_sums[j] += np.log(abs(R[j, j]))
                            if n_renorm > 0:
                                lyap_history[j].append(lyap_sums[j] / (n_renorm * renorm_interval))
                        n_renorm += 1
                    
                    t += dt
                    
                    if int(t / integration_time * 100) % 5 == 0:
                        progress_bar.progress(t / integration_time)
                
                progress_bar.empty()
                
                # Final spectrum
                spectrum = lyap_sums / (n_renorm * renorm_interval)
                spectrum_sorted = sorted(spectrum, reverse=True)
                
                                # Create plots
                fig = make_subplots(rows=2, cols=1,
                                   subplot_titles=("Lyapunov Spectrum Convergence", 
                                                  "Final Lyapunov Spectrum"),
                                   vertical_spacing=0.15)
                
                # Convergence plot
                colors = ['red', 'green', 'blue']
                for i, (history, color) in enumerate(zip(lyap_history, colors)):
                    fig.add_trace(go.Scatter(
                        y=history,
                        mode='lines',
                        line=dict(color=color, width=2),
                        name=f'λ{i+1}'
                    ), row=1, col=1)
                
                # Final spectrum bar plot
                fig.add_trace(go.Bar(
                    x=['λ₁', 'λ₂', 'λ₃'],
                    y=spectrum_sorted,
                    marker=dict(color=['red' if l > 0 else 'blue' if l < 0 else 'gray' for l in spectrum_sorted]),
                    text=[f'{l:.3f}' for l in spectrum_sorted],
                    textposition='outside',
                    name='Spectrum'
                ), row=2, col=1)
                
                # Add zero line
                fig.add_hline(y=0, line_dash="dash", line_color="gray", row=1, col=1)
                fig.add_hline(y=0, line_dash="dash", line_color="gray", row=2, col=1)
                
                fig.update_layout(
                    title=f"Lyapunov Spectrum Analysis (ρ = {rho:.1f})",
                    height=700,
                    template="plotly_white",
                    showlegend=True
                )
                
                fig.update_xaxes(title_text="Renormalization step", row=1, col=1)
                fig.update_yaxes(title_text="Lyapunov exponent", row=1, col=1)
                fig.update_yaxes(title_text="Value", row=2, col=1)
                
                st.plotly_chart(fig, use_container_width=True)
                
                # Show interpretation guide
                with st.expander("📖 How to interpret the spectrum", expanded=True):
                    st.markdown(f"""
                    **What the spectrum tells you:**
                    - **λ₁ = {spectrum_sorted[0]:.4f}**: {"Positive → Chaotic stretching" if spectrum_sorted[0] > 0 else "Non-positive → No chaos"}
                    - **λ₂ = {spectrum_sorted[1]:.4f}**: {"Near zero → Neutral direction" if abs(spectrum_sorted[1]) < 0.1 else "Negative → Contracting"}
                    - **λ₃ = {spectrum_sorted[2]:.4f}**: Strong contraction (most negative)
                    
                    **For the Lorenz attractor:**
                    - One positive exponent → chaotic dynamics
                    - One zero exponent → flow on attractor
                    - One negative exponent → attraction to attractor
                    - This (+, 0, -) signature is typical of strange attractors
                    
                    **Physical meaning:**
                    - The sum Σλᵢ = {sum(spectrum_sorted):.4f} represents volume contraction rate
                    - Positive λ₁ means weather prediction has fundamental limits
                    - The ratio |λ₁/λ₃| indicates how "strange" the attractor is
                    """)
                
                # Compute derived quantities
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("λ₁", f"{spectrum_sorted[0]:.4f}", 
                             delta="Chaotic" if spectrum_sorted[0] > 0 else "Stable")
                with col2:
                    st.metric("λ₂", f"{spectrum_sorted[1]:.4f}")
                with col3:
                    st.metric("λ₃", f"{spectrum_sorted[2]:.4f}")
                with col4:
                    # Kaplan-Yorke dimension
                    if spectrum_sorted[0] > 0 and spectrum_sorted[2] < 0:
                        if spectrum_sorted[1] >= 0:
                            D_KY = 2 + (spectrum_sorted[0] + spectrum_sorted[1]) / abs(spectrum_sorted[2])
                        else:
                            D_KY = 1 + spectrum_sorted[0] / abs(spectrum_sorted[1])
                        st.metric("Kaplan-Yorke Dim", f"{D_KY:.3f}")
                    else:
                        st.metric("Kaplan-Yorke Dim", "N/A")
                
                # Additional analysis
                st.markdown("### Additional Analysis")
                col1, col2 = st.columns(2)
                with col1:
                    # Lyapunov sum rule check
                    lyap_sum = sum(spectrum_sorted)
                    divergence = -sigma - 1 - beta  # ∇·f for Lorenz
                    st.info(f"""
                    **Sum rule check:** 
                    - Σλᵢ = {lyap_sum:.4f}
                    - ∇·f = {divergence:.4f}
                    - Difference: {abs(lyap_sum - divergence):.6f}
                    
                    These should be equal (validates computation)
                    """)
                
                with col2:
                    # Predictability analysis
                    if spectrum_sorted[0] > 0:
                        lyap_time = 1 / spectrum_sorted[0]
                        st.info(f"""
                        **Predictability Analysis:**
                        - Lyapunov time: τ = 1/λ₁ = {lyap_time:.2f} time units
                        - Error growth: e^(λ₁t) = e^({spectrum_sorted[0]:.3f}t)
                        - 10x error after: {np.log(10)/spectrum_sorted[0]:.2f} time units
                        
                        Beyond ~{2*lyap_time:.1f} time units, prediction becomes meaningless
                        """)
                    else:
                        st.info("**System is not chaotic** - Long-term prediction possible")
            
            
            elif system == "Duffing Oscillator":
                gamma_spectrum = st.slider("Select γ value", gamma_min, gamma_max, (gamma_min + gamma_max)/2)
                
                # Initial conditions
                state = np.array([0.1, 0.0])  # [x, x_dot]
                phase = 0.0
                
                # Initialize 2 orthonormal perturbation vectors
                Q = np.eye(2) * 1e-8
                
                lyap_sums = np.zeros(2)
                n_renorm = 0
                
                # For plotting convergence
                lyap_history = [[], []]
                
                progress_bar = st.progress(0)
                dt_duff = 0.01
                
                # Skip transient
                transient_steps = int(50 / dt_duff)
                for _ in range(transient_steps):
                    x, x_dot = state
                    x_ddot = -delta * x_dot - alpha * x - beta_duff * x**3 + gamma_spectrum * np.cos(omega * phase)
                    state[0] += x_dot * dt_duff
                    state[1] += x_ddot * dt_duff
                    phase += omega * dt_duff
                
                # Compute spectrum
                t = 0
                while t < integration_time:
                    # Evolve perturbations
                    Q_new = Q.copy()
                    for j in range(2):
                        perturbed = state + Q[:, j]
                        x_p, x_dot_p = perturbed
                        
                        # Dynamics for perturbed trajectory
                        x_ddot_p = -delta * x_dot_p - alpha * x_p - beta_duff * x_p**3 + gamma_spectrum * np.cos(omega * phase)
                        
                        # Update perturbed state
                        perturbed[0] += x_dot_p * dt_duff
                        perturbed[1] += x_ddot_p * dt_duff
                        
                        Q_new[:, j] = perturbed - state
                    
                    # Evolve main trajectory
                    x, x_dot = state
                    x_ddot = -delta * x_dot - alpha * x - beta_duff * x**3 + gamma_spectrum * np.cos(omega * phase)
                    state[0] += x_dot * dt_duff
                    state[1] += x_ddot * dt_duff
                    
                    Q = Q_new
                    
                    # Gram-Schmidt orthogonalization
                    if (n_renorm + 1) * 1.0 <= t:  # Every 1 time unit
                        Q, R = np.linalg.qr(Q)
                        for j in range(2):
                            lyap_sums[j] += np.log(abs(R[j, j]))
                            if n_renorm > 0:
                                lyap_history[j].append(lyap_sums[j] / n_renorm)
                        n_renorm += 1
                    
                    phase += omega * dt_duff
                    t += dt_duff
                    
                    if int(t / integration_time * 100) % 5 == 0:
                        progress_bar.progress(min(t / integration_time, 1.0))
                
                progress_bar.empty()
                
                # Final spectrum
                spectrum = lyap_sums / n_renorm
                spectrum_sorted = sorted(spectrum, reverse=True)
                
                # Create plots
                fig = make_subplots(rows=2, cols=1,
                                subplot_titles=("Lyapunov Spectrum Convergence", 
                                                "Final Lyapunov Spectrum"),
                                vertical_spacing=0.15)
                
                # Convergence plot
                colors = ['red', 'blue']
                for i, (history, color) in enumerate(zip(lyap_history, colors)):
                    if len(history) > 0:
                        fig.add_trace(go.Scatter(
                            y=history,
                            mode='lines',
                            line=dict(color=color, width=2),
                            name=f'λ{i+1}'
                        ), row=1, col=1)
                
                # Final spectrum bar plot
                fig.add_trace(go.Bar(
                    x=['λ₁', 'λ₂'],
                    y=spectrum_sorted,
                    marker=dict(color=['red' if l > 0 else 'blue' if l < 0 else 'gray' for l in spectrum_sorted]),
                    text=[f'{l:.3f}' for l in spectrum_sorted],
                    textposition='outside',
                    name='Spectrum'
                ), row=2, col=1)
                
                # Add zero line
                fig.add_hline(y=0, line_dash="dash", line_color="gray", row=1, col=1)
                fig.add_hline(y=0, line_dash="dash", line_color="gray", row=2, col=1)
                
                fig.update_layout(
                    title=f"Lyapunov Spectrum Analysis - Duffing (γ = {gamma_spectrum:.2f})",
                    height=700,
                    template="plotly_white",
                    showlegend=True
                )
                
                fig.update_xaxes(title_text="Renormalization step", row=1, col=1)
                fig.update_yaxes(title_text="Lyapunov exponent", row=1, col=1)
                fig.update_yaxes(title_text="Value", row=2, col=1)
                
                st.plotly_chart(fig, use_container_width=True)
                
                # Show interpretation guide
                with st.expander("📖 How to interpret the spectrum", expanded=True):
                    st.markdown(f"""
                    **What the spectrum tells you:**
                    - **λ₁ = {spectrum_sorted[0]:.4f}**: {"Positive → Chaotic" if spectrum_sorted[0] > 0 else "Non-positive → Periodic/stable"}
                    - **λ₂ = {spectrum_sorted[1]:.4f}**: Should be negative (dissipation due to damping)
                    
                    **For the Duffing oscillator:**
                    - Sum Σλᵢ = {sum(spectrum_sorted):.4f} (should be negative due to damping δ = {delta})
                    - λ₁ > 0 indicates chaotic motion
                    - |λ₂| > |λ₁| ensures bounded attractor
                    
                    **Physical meaning:**
                    - Models nonlinear vibrations in mechanical systems
                    - Positive λ₁ means unpredictable oscillations
                    - The sum represents energy dissipation rate
                    """)
                
                # Metrics
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("λ₁", f"{spectrum_sorted[0]:.4f}", 
                            delta="Chaotic" if spectrum_sorted[0] > 0 else "Regular")
                with col2:
                    st.metric("λ₂", f"{spectrum_sorted[1]:.4f}")
                with col3:
                    st.metric("Sum Σλᵢ", f"{sum(spectrum_sorted):.4f}")
                
                # Additional analysis
                if spectrum_sorted[0] > 0:
                    lyap_time = 1 / spectrum_sorted[0]
                    st.info(f"""
                    **Predictability Analysis:**
                    - Lyapunov time: τ = {lyap_time:.2f} time units
                    - Predictable for ~{2*lyap_time:.1f} time units
                    - Chaos parameter: γ/γc ≈ {gamma_spectrum/0.2:.2f} (γc ≈ 0.2 for onset)
                    """)

            elif system == "Van der Pol Oscillator":
                mu_spectrum = st.slider("Select μ value", mu_min, mu_max, (mu_min + mu_max)/2)
                
                # Initial conditions on limit cycle
                if mu_spectrum < 0.1:
                    state = np.array([2.0, 0.0])
                else:
                    A = 2.0 * (1 - 1/(8*mu_spectrum))
                    state = np.array([A, 0.0])
                
                # Initialize 2 orthonormal perturbation vectors
                Q = np.eye(2) * 1e-8
                
                lyap_sums = np.zeros(2)
                n_renorm = 0
                
                # For plotting convergence
                lyap_history = [[], []]
                
                progress_bar = st.progress(0)
                dt_vdp = 0.01
                
                # Skip transient
                transient_steps = int(100 / dt_vdp)
                for _ in range(transient_steps):
                    x, y = state
                    dx = y
                    dy = mu_spectrum * (1 - x**2) * y - x
                    state[0] += dx * dt_vdp
                    state[1] += dy * dt_vdp
                
                # Compute spectrum
                t = 0
                while t < integration_time:
                    # Evolve perturbations
                    Q_new = Q.copy()
                    for j in range(2):
                        perturbed = state + Q[:, j]
                        x_p, y_p = perturbed
                        
                        # Van der Pol dynamics
                        dx_p = y_p
                        dy_p = mu_spectrum * (1 - x_p**2) * y_p - x_p
                        
                        # Update perturbed state
                        perturbed[0] += dx_p * dt_vdp
                        perturbed[1] += dy_p * dt_vdp
                        
                        Q_new[:, j] = perturbed - state
                    
                    # Evolve main trajectory
                    x, y = state
                    dx = y
                    dy = mu_spectrum * (1 - x**2) * y - x
                    state[0] += dx * dt_vdp
                    state[1] += dy * dt_vdp
                    
                    Q = Q_new
                    
                    # Gram-Schmidt orthogonalization
                    if (n_renorm + 1) * 1.0 <= t:
                        Q, R = np.linalg.qr(Q)
                        for j in range(2):
                            lyap_sums[j] += np.log(abs(R[j, j]))
                            if n_renorm > 0:
                                lyap_history[j].append(lyap_sums[j] / n_renorm)
                        n_renorm += 1
                    
                    t += dt_vdp
                    
                    if int(t / integration_time * 100) % 5 == 0:
                        progress_bar.progress(min(t / integration_time, 1.0))
                
                progress_bar.empty()
                
                # Final spectrum
                spectrum = lyap_sums / n_renorm
                spectrum_sorted = sorted(spectrum, reverse=True)
                
                # Create plots
                fig = make_subplots(rows=2, cols=1,
                                subplot_titles=("Lyapunov Spectrum Convergence", 
                                                "Final Lyapunov Spectrum"),
                                vertical_spacing=0.15)
                
                # Convergence plot
                colors = ['red', 'blue']
                for i, (history, color) in enumerate(zip(lyap_history, colors)):
                    if len(history) > 0:
                        fig.add_trace(go.Scatter(
                            y=history,
                            mode='lines',
                            line=dict(color=color, width=2),
                            name=f'λ{i+1}'
                        ), row=1, col=1)
                
                # Final spectrum bar plot
                fig.add_trace(go.Bar(
                    x=['λ₁', 'λ₂'],
                    y=spectrum_sorted,
                    marker=dict(color=['green' if l <= 0 else 'red' for l in spectrum_sorted]),
                    text=[f'{l:.3f}' for l in spectrum_sorted],
                    textposition='outside',
                    name='Spectrum'
                ), row=2, col=1)
                
                # Add zero line
                fig.add_hline(y=0, line_dash="dash", line_color="gray", row=1, col=1)
                fig.add_hline(y=0, line_dash="dash", line_color="gray", row=2, col=1)
                
                fig.update_layout(
                    title=f"Lyapunov Spectrum Analysis - Van der Pol (μ = {mu_spectrum:.2f})",
                    height=700,
                    template="plotly_white",
                    showlegend=True
                )
                
                fig.update_xaxes(title_text="Renormalization step", row=1, col=1)
                fig.update_yaxes(title_text="Lyapunov exponent", row=1, col=1)
                fig.update_yaxes(title_text="Value", row=2, col=1)
                
                st.plotly_chart(fig, use_container_width=True)
                
                # Show interpretation guide
                with st.expander("📖 How to interpret the spectrum", expanded=True):
                    st.markdown(f"""
                    **What the spectrum tells you:**
                    - **λ₁ = {spectrum_sorted[0]:.4f}**: Should be ≈ 0 (on limit cycle) or < 0 (approaching cycle)
                    - **λ₂ = {spectrum_sorted[1]:.4f}**: Negative (transverse to limit cycle)
                    
                    **For the Van der Pol oscillator:**
                    - Always has a stable limit cycle for μ > 0
                    - NO CHAOS POSSIBLE - all exponents ≤ 0
                    - λ₁ ≈ 0 indicates trajectory on the limit cycle
                    - λ₁ < 0 indicates trajectory converging to cycle
                    
                    **Physical meaning:**
                    - Models self-sustained oscillations
                    - μ = {mu_spectrum} controls nonlinearity
                    - Sum Σλᵢ = {sum(spectrum_sorted):.4f} (energy balance on cycle)
                    """)
                
                # Metrics
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("λ₁", f"{spectrum_sorted[0]:.4f}", 
                            delta="On cycle" if abs(spectrum_sorted[0]) < 0.01 else "Transient")
                with col2:
                    st.metric("λ₂", f"{spectrum_sorted[1]:.4f}")
                with col3:
                    st.metric("Sum Σλᵢ", f"{sum(spectrum_sorted):.4f}")
                
                # Additional analysis
                st.info(f"""
                **Limit Cycle Analysis:**
                - Period ≈ {2*np.pi:.2f} for small μ
                - Period ≈ {1.614*mu_spectrum:.2f} for large μ (relaxation oscillations)
                - Amplitude ≈ {2.0 * (1 - 1/(8*mu_spectrum)) if mu_spectrum > 0.125 else 2.0:.2f}
                
                **Note:** Van der Pol oscillator CANNOT exhibit chaos.
                All trajectories converge to the same limit cycle.
                """)

            elif system == "Hénon Map":
                a_spectrum = st.slider("Select a value", a_min, a_max, 1.4)
                
                # For 2D map, we compute 2 Lyapunov exponents
                x, y = 0.0, 0.0
                
                # Skip transient
                for _ in range(1000):
                    x_new = 1 - a_spectrum * x**2 + y
                    y_new = b_henon * x
                    x, y = x_new, y_new
                
                # Initialize
                lyap_sums = np.zeros(2)
                lyap_history = [[], []]
                
                # Identity matrix for perturbations
                M = np.eye(2)
                
                progress_bar = st.progress(0)
                n_steps = 10000
                
                for i in range(n_steps):
                    # Jacobian matrix
                    J = np.array([[-2*a_spectrum*x, 1],
                                [b_henon, 0]])
                    
                    # Update M = J * M
                    M = J @ M
                    
                    # QR decomposition every 10 iterations
                    if (i + 1) % 10 == 0:
                        Q, R = np.linalg.qr(M)
                        
                        # Accumulate logs of diagonal elements
                        for j in range(2):
                            lyap_sums[j] += np.log(abs(R[j, j]))
                            lyap_history[j].append(lyap_sums[j] / (i + 1))
                        
                        # Reset M to Q for next iteration
                        M = Q
                    
                    # Update map
                    x_new = 1 - a_spectrum * x**2 + y
                    y_new = b_henon * x
                    x, y = x_new, y_new
                    
                    if i % 100 == 0:
                        progress_bar.progress(i / n_steps)
                
                progress_bar.empty()
                
                # Final spectrum
                spectrum = lyap_sums / n_steps
                spectrum_sorted = sorted(spectrum, reverse=True)
                
                # Create plots
                fig = make_subplots(rows=2, cols=1,
                                subplot_titles=("Lyapunov Spectrum Convergence", 
                                                "Final Lyapunov Spectrum"),
                                vertical_spacing=0.15)
                
                # Convergence plot - show every 10th point for clarity
                colors = ['red', 'blue']
                for i, (history, color) in enumerate(zip(lyap_history, colors)):
                    if len(history) > 0:
                        fig.add_trace(go.Scatter(
                            y=history[::10],  # Show every 10th point
                            mode='lines',
                            line=dict(color=color, width=2),
                            name=f'λ{i+1}'
                        ), row=1, col=1)
                
                # Final spectrum bar plot
                fig.add_trace(go.Bar(
                    x=['λ₁', 'λ₂'],
                    y=spectrum_sorted,
                    marker=dict(color=['red' if l > 0 else 'blue' if l < 0 else 'gray' for l in spectrum_sorted]),
                    text=[f'{l:.3f}' for l in spectrum_sorted],
                    textposition='outside',
                    name='Spectrum'
                ), row=2, col=1)
                
                # Add zero line
                fig.add_hline(y=0, line_dash="dash", line_color="gray", row=1, col=1)
                fig.add_hline(y=0, line_dash="dash", line_color="gray", row=2, col=1)
                
                # Add theoretical sum line
                theoretical_sum = np.log(b_henon)
                fig.add_hline(y=theoretical_sum, line_dash="dot", line_color="green", 
                            row=1, col=1, opacity=0.5,
                            annotation_text=f"Sum = ln(b) = {theoretical_sum:.3f}")
                
                fig.update_layout(
                    title=f"Lyapunov Spectrum Analysis - Hénon Map (a = {a_spectrum:.2f}, b = {b_henon})",
                    height=700,
                    template="plotly_white",
                    showlegend=True
                )
                
                fig.update_xaxes(title_text="Iteration (×10)", row=1, col=1)
                fig.update_yaxes(title_text="Lyapunov exponent", row=1, col=1)
                fig.update_yaxes(title_text="Value", row=2, col=1)
                
                st.plotly_chart(fig, use_container_width=True)
                
                # Show interpretation guide
                with st.expander("📖 How to interpret the spectrum", expanded=True):
                    st.markdown(f"""
                    **What the spectrum tells you:**
                    - **λ₁ = {spectrum_sorted[0]:.4f}**: {"Positive → Chaotic" if spectrum_sorted[0] > 0 else "Non-positive → Periodic/stable"}
                    - **λ₂ = {spectrum_sorted[1]:.4f}**: Negative (area contraction)
                    
                    **For the Hénon map:**
                    - Sum λ₁ + λ₂ = {sum(spectrum_sorted):.4f} ≈ ln(b) = {np.log(b_henon):.4f} ✓
                    - Standard parameters (a=1.4, b=0.3): λ₁ ≈ 0.42, λ₂ ≈ -1.62
                    - Positive λ₁ indicates strange attractor
                    
                    **Kaplan-Yorke dimension:**
                    """)
                    
                    if spectrum_sorted[0] > 0 and spectrum_sorted[1] < 0:
                        D_KY = 1 + spectrum_sorted[0] / abs(spectrum_sorted[1])
                        st.markdown(f"D_KY = 1 + λ₁/|λ₂| = {D_KY:.3f} (fractal dimension)")
                    else:
                        st.markdown("D_KY not defined (no chaos)")
                
                # Metrics
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("λ₁", f"{spectrum_sorted[0]:.4f}", 
                            delta="Chaotic" if spectrum_sorted[0] > 0 else "Regular")
                with col2:
                    st.metric("λ₂", f"{spectrum_sorted[1]:.4f}")
                with col3:
                    st.metric("Sum", f"{sum(spectrum_sorted):.4f}")
                with col4:
                    if spectrum_sorted[0] > 0 and spectrum_sorted[1] < 0:
                        D_KY = 1 + spectrum_sorted[0] / abs(spectrum_sorted[1])
                        st.metric("D_KY", f"{D_KY:.3f}")
                    else:
                        st.metric("D_KY", "N/A")
                
                # Sum rule check
                error = abs(sum(spectrum_sorted) - np.log(b_henon))
                st.success(f"✓ Sum rule check: |Σλᵢ - ln(b)| = {error:.6f} (should be ≈ 0)")

            else:  # Logistic Map
                st.warning("Spectrum analysis for 1D maps only has one Lyapunov exponent. Showing enhanced analysis instead.")
                
                r_spectrum = st.slider("Select r value", r_min, r_max, 3.8)
                
                # Compute single Lyapunov exponent with detailed convergence
                x = 0.5
                lyap_history = []
                x_history = []
                
                # Skip transient
                for _ in range(1000):
                    x = r_spectrum * x * (1 - x)
                
                # Compute with history
                lyap_sum = 0
                for i in range(10000):
                    x = r_spectrum * x * (1 - x)
                    if 0 < x < 1:
                        instant_lyap = np.log(abs(r_spectrum * (1 - 2*x)))
                        lyap_sum += instant_lyap
                        if i % 10 == 0:  # Record every 10th value
                            lyap_history.append(lyap_sum / (i + 1))
                            x_history.append(x)
                
                final_lyap = lyap_sum / 10000
                
                # Create enhanced analysis plot
                fig = make_subplots(rows=3, cols=1,
                                subplot_titles=("Lyapunov Exponent Convergence", 
                                                "Trajectory in Phase Space",
                                                "Local Stretching Rates"),
                                vertical_spacing=0.12,
                                row_heights=[0.4, 0.3, 0.3])
                
                # Convergence plot
                fig.add_trace(go.Scatter(
                    y=lyap_history,
                    mode='lines',
                    line=dict(color='blue', width=2),
                    name='LLE'
                ), row=1, col=1)
                
                # Add final value line
                fig.add_hline(y=final_lyap, line_dash="dash", line_color="red", 
                            row=1, col=1, opacity=0.5,
                            annotation_text=f"λ = {final_lyap:.4f}")
                
                # Phase space trajectory
                fig.add_trace(go.Scatter(
                    x=x_history[:-1],
                    y=x_history[1:],
                    mode='markers',
                    marker=dict(size=3, color=list(range(len(x_history)-1)), 
                            colorscale='Viridis', showscale=False),
                    name='Trajectory'
                ), row=2, col=1)
                
                # Add y=x line
                fig.add_trace(go.Scatter(
                    x=[0, 1], y=[0, 1],
                    mode='lines',
                    line=dict(color='gray', dash='dash'),
                    name='y=x'
                ), row=2, col=1)
                
                # Local stretching rates histogram
                stretching_rates = []
                x = 0.5
                for _ in range(1000):
                    x = r_spectrum * x * (1 - x)
                for _ in range(1000):
                    x = r_spectrum * x * (1 - x)
                    if 0 < x < 1:
                        stretching_rates.append(abs(r_spectrum * (1 - 2*x)))
                
                fig.add_trace(go.Histogram(
                    x=stretching_rates,
                    nbinsx=50,
                    marker=dict(color='lightblue'),
                    name='Stretching'
                ), row=3, col=1)
                
                # Add stretching = 1 line (chaos boundary)
                fig.add_vline(x=1, line_dash="dash", line_color="red", 
                            row=3, col=1, opacity=0.5,
                            annotation_text="Stretching = 1")
                
                # Zero line for convergence
                fig.add_hline(y=0, line_dash="dot", line_color="gray", row=1, col=1)
                
                fig.update_layout(
                    title=f"Enhanced Lyapunov Analysis - Logistic Map (r = {r_spectrum:.3f})",
                    height=900,
                    template="plotly_white",
                    showlegend=False
                )
                
                fig.update_xaxes(title_text="Iteration (×10)", row=1, col=1)
                fig.update_xaxes(title_text="x_n", row=2, col=1)
                fig.update_xaxes(title_text="Local Stretching Rate", row=3, col=1)
                
                fig.update_yaxes(title_text="LLE", row=1, col=1)
                fig.update_yaxes(title_text="x_{n+1}", row=2, col=1)
                fig.update_yaxes(title_text="Frequency", row=3, col=1)
                
                st.plotly_chart(fig, use_container_width=True)
                
                # Show interpretation guide
                with st.expander("📖 How to interpret this analysis", expanded=True):
                    st.markdown(f"""
                    **Single Lyapunov Exponent:**
                    - λ = {final_lyap:.4f} → System is {"chaotic" if final_lyap > 0 else "stable/periodic"}
                    
                    **Phase space plot (middle):**
                    - Points on diagonal: Fixed points
                    - Boxes: Period-2 cycle
                    - Complex patterns: Higher periods or chaos
                    - Color progression shows time evolution
                    
                    **Stretching rate distribution (bottom):**
                    - Values > 1: Local expansion
                    - Values < 1: Local contraction
                    - Wide distribution: Varying dynamics across attractor
                    - Peak location: Most common stretching factor
                    
                    **For r = {r_spectrum:.3f}:**
                    """)
                    
                    if final_lyap > 0:
                        st.markdown(f"""
                        - **Chaotic regime** with positive Lyapunov exponent
                        - Predictability time: ~{1/final_lyap:.1f} iterations
                        - Average stretching: {np.exp(final_lyap):.3f}× per iteration
                        """)
                    elif abs(final_lyap) < 0.001:
                        st.markdown("""
                        - **Marginally stable** (likely at bifurcation point)
                        - System transitioning between periodic and chaotic
                        - Very slow convergence/divergence
                        """)
                    else:
                        period = "fixed point" if r_spectrum < 3 else "periodic orbit"
                        st.markdown(f"""
                        - **Stable {period}** with negative Lyapunov exponent
                        - Perturbations decay by factor {np.exp(-abs(final_lyap)):.3f} per iteration
                        - Convergence time: ~{1/abs(final_lyap):.1f} iterations
                        """)
                
                # Metrics
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Lyapunov Exp.", f"{final_lyap:.4f}",
                            delta="Chaotic" if final_lyap > 0 else "Stable")
                with col2:
                    avg_stretch = np.mean(stretching_rates) if stretching_rates else 0
                    st.metric("Avg Stretching", f"{avg_stretch:.3f}")
                with col3:
                    if final_lyap != 0:
                        st.metric("Predictability", f"{abs(1/final_lyap):.1f} iter")
                    else:
                        st.metric("Predictability", "∞")
                
                # Additional info box
                st.info(f"""
                💡 **Note for 1D maps**: 
                - Only one Lyapunov exponent exists (system is 1-dimensional)
                - The exponent fully characterizes the dynamics
                - Compare with bifurcation diagram at r = {r_spectrum:.3f}
                - Chaos occurs when λ > 0 (average stretching > 1)
                """)
    
    # Theory expander
    with st.expander("📚 Theoretical Background", expanded=False):
        st.markdown("""
        ## Lyapunov Exponents: Measuring Chaos
        
        Lyapunov exponents quantify the average rate of separation of infinitesimally close trajectories:
        
        $$\\lambda = \\lim_{t \\to \\infty} \\frac{1}{t} \\ln\\left(\\frac{|\\delta\\mathbf{x}(t)|}{|\\delta\\mathbf{x}_0|}\\right)$$
        
        ### Interpretation:
        - **λ > 0**: Chaotic behavior (exponential divergence)
        - **λ = 0**: Marginally stable (e.g., limit cycles)
        - **λ < 0**: Stable/attracting behavior
        
        ### For Different Systems:
        
        #### 1. **Logistic Map**
        Analytical formula: $\\lambda = \\frac{1}{n}\\sum_{i=0}^{n-1} \\ln|r(1-2x_i)|$
        - Period-1: λ < 0
        - Chaos onset (r ≈ 3.57): λ crosses 0
        - Full chaos (r = 4): λ ≈ ln(2) ≈ 0.693
        
        #### 2. **Lorenz System**
        Three exponents (λ₁ ≥ λ₂ ≥ λ₃):
        - Strange attractor: λ₁ > 0, λ₂ ≈ 0, λ₃ < 0
        - Sum rule: Σλᵢ = ∇·f = -(σ + 1 + β)
        - Kaplan-Yorke dimension: $D_{KY} = j + \\frac{\\sum_{i=1}^j \\lambda_i}{|\\lambda_{j+1}|}$
        
        #### 3. **Hénon Map**
        Two exponents:
        - Chaotic regime: λ₁ > 0, λ₂ < 0
        - Area contraction: λ₁ + λ₂ = ln(b) < 0
        
        ### Practical Considerations:
        
        1. **Convergence**: Long integration times needed (typically 10³-10⁴ characteristic times)
        2. **Renormalization**: Essential to prevent overflow/underflow
        3. **Transients**: Discard initial behavior for accurate estimates
        4. **Finite-time LEs**: Useful for identifying coherent structures
        
        ### Numerical Methods:
        
        #### Standard Algorithm:
        1. Evolve reference trajectory
        2. Evolve perturbation vector(s)
        3. Renormalize periodically: δx → δx/|δx|
        4. Sum logarithms of stretching factors
        
        #### For Full Spectrum:
        - Use Gram-Schmidt orthogonalization
        - Maintain n orthonormal vectors for n-D system
        - QR decomposition at each renormalization
        
        ### Relationships:
        
        - **Positive LLE ↔ Chaos**: Necessary but not sufficient
        - **LLE ≈ Kolmogorov-Sinai entropy**: For chaotic systems
        - **Lyapunov time**: τ = 1/λ₁ (predictability horizon)
        """)
        
        # Add system-specific details
        if 'lyapunov_system' in st.session_state:
            selected_system = st.session_state.lyapunov_system
            
            st.markdown("---")
            st.markdown(f"### Specific Details for {selected_system}")
            
            if selected_system == "Logistic Map":
                st.markdown("""
                - **Periodic windows**: Sharp drops to negative λ
                - **Band-merging**: Local maxima in λ(r)
                - **Maximum chaos**: λ(4) = ln(2) ≈ 0.693
                - **Feigenbaum cascade**: λ → 0 at accumulation point
                """)
                
            elif selected_system == "Lorenz System":
                st.markdown("""
                - **Classic values** (σ=10, β=8/3, ρ=28): λ₁ ≈ 0.906, λ₂ ≈ 0, λ₃ ≈ -14.572
                - **Lorenz attractor dimension**: D_KY ≈ 2.06
                - **Predictability time**: ~1.1 time units
                - **Periodic windows**: e.g., around ρ ≈ 99.65
                """)
                
            elif selected_system == "Duffing Oscillator":
                st.markdown("""
                - **Driven system**: Largest LE typically increases with forcing γ
                - **Multiple attractors**: Different ICs → different LEs
                - **Resonance effects**: Sharp changes near ω ≈ ω₀
                - **Transient chaos**: Positive LE before settling to periodic
                """)
                
            elif selected_system == "Van der Pol Oscillator":
                st.markdown("""
                - **Always λ₁ ≤ 0**: No chaos (limit cycle attractor)
                - **λ₁ = 0**: On the limit cycle (marginal stability)
                - **λ₁ < 0**: Inside basin of attraction
                - **Relaxation oscillations**: Slower convergence for large μ
                """)
                
            elif selected_system == "Hénon Map":
                st.markdown("""
                - **Standard parameters** (a=1.4, b=0.3): λ₁ ≈ 0.42, λ₂ ≈ -1.62
                - **Hénon attractor dimension**: D_KY ≈ 1.26
                - **Area contraction**: λ₁ + λ₂ = ln(0.3) ≈ -1.20
                - **Crisis points**: Sudden changes in attractor structure
                """)
            
        st.markdown("""
        ---
        ### Tips for Your Analysis:
        
        1. **Compare with bifurcation diagram**: Overlay λ(parameter) with bifurcation plot
        2. **Check convergence**: Plot running average vs time
        3. **Parameter sweeps**: Find chaos boundaries precisely
        4. **Initial condition sensitivity**: Use FTLE maps
        5. **Validate numerics**: Check sum rules and known values
        
        ### Common Pitfalls:
        
        - **Too short integration**: Use at least 1000 × characteristic time
        - **Poor renormalization interval**: Causes numerical errors
        - **Ignoring transients**: Biases results toward initial behavior
        - **Single trajectory**: Multiple ICs reveal attractor structure
        
        ### Connection to Real-World Applications:
        
        - **Weather prediction**: Lorenz system shows ~2 week predictability limit
        - **Ecosystem dynamics**: Logistic map models population crashes
        - **Engineering**: Duffing oscillator represents structural vibrations
        - **Secure communications**: Chaotic systems used in encryption
        - **Heart rhythms**: Positive LE can indicate arrhythmia risk
        """)
                

with tabs[4]:
    st.header("Hopf Explorer")
    st.write("Coming soon...")
