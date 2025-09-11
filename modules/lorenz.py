# app.py
# ============================================
# Chaos & Nonlinear Dynamics Explorer (Lorenz tab demo)
# ============================================

import io
import numpy as np
import streamlit as st
from scipy.integrate import solve_ivp
import plotly.graph_objects as go

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

    # Controls in columns (style consistent with your Quantum Visualizer)
    col1, col2, col3 = st.columns([1, 1, 1])
    with col1:
        sigma = st.slider("σ (Prandtl number)", min_value=0.0, max_value=20.0, value=10.0, step=0.1)
        x0 = st.number_input("x₀", value=0.0, step=0.1)
    with col2:
        rho = st.slider("ρ (Rayleigh number)", min_value=0.0, max_value=60.0, value=28.0, step=0.5)
        y0 = st.number_input("y₀", value=1.0, step=0.1)
    with col3:
        beta = st.slider("β", min_value=0.0, max_value=10.0, value=float(8/3), step=0.1)
        z0 = st.number_input("z₀", value=1.05, step=0.1)

    colA, colB = st.columns(2)
    with colA:
        t_max = st.slider("Simulation Time (t_max)", min_value=5.0, max_value=120.0, value=40.0, step=1.0)
    with colB:
        dt = st.slider("Time Step (Δt)", min_value=0.001, max_value=0.05, value=0.01, step=0.001)

    # Options
    colO1, colO2 = st.columns(2)
    with colO1:
        show_equations = st.checkbox("Show equations", value=True)
    with colO2:
        enable_downloads = st.checkbox("Enable downloads", value=True)

    # Equations (same vibe as your particle-in-a-box tab)
    if show_equations:
        st.markdown("### 📐 Mathematical Description")
        c1, c2 = st.columns(2)
        with c1:
            st.latex(r"\dot{x} = \sigma (y - x)")
            st.latex(r"\dot{y} = x (\rho - z) - y")
            st.latex(r"\dot{z} = xy - \beta z")
        with c2:
            st.markdown(
                """
                - **σ** (*Prandtl*): ratio of momentum to thermal diffusivity  
                - **ρ** (*Rayleigh*): thermal forcing/instability parameter  
                - **β**: geometric factor

                **Deterministic chaos** arises for certain parameter ranges: nearby trajectories
                separate exponentially fast while remaining bounded on a **strange attractor**.
                """
            )

    # Integrate ODE
    t, x, y, z = integrate_lorenz(x0, y0, z0, sigma, rho, beta, t_max, dt)

    # View selector
    view = st.radio(
        "Select View",
        ["3D Attractor", "2D Projection (x–y)", "2D Projection (x–z)", "2D Projection (y–z)"],
        horizontal=True
    )

    # Plot
    if view == "3D Attractor":
        fig = go.Figure(
            data=[go.Scatter3d(
                x=x, y=y, z=z,
                mode="lines",
                line=dict(color=z, colorscale="Viridis", width=2)
            )]
        )
        fig.update_layout(
            scene=dict(xaxis_title="X", yaxis_title="Y", zaxis_title="Z"),
            margin=dict(l=0, r=0, b=0, t=30),
            title=f"Lorenz Attractor (σ={sigma}, ρ={rho}, β={beta:.3f})",
            height=560
        )
    elif view == "2D Projection (x–y)":
        fig = go.Figure(data=[go.Scatter(x=x, y=y, mode="lines", name="x–y")])
        fig.update_layout(xaxis_title="X", yaxis_title="Y", title="2D Projection: X vs Y", height=560)
    elif view == "2D Projection (x–z)":
        fig = go.Figure(data=[go.Scatter(x=x, y=z, mode="lines", name="x–z")])
        fig.update_layout(xaxis_title="X", yaxis_title="Z", title="2D Projection: X vs Z", height=560)
    else:
        fig = go.Figure(data=[go.Scatter(x=y, y=z, mode="lines", name="y–z")])
        fig.update_layout(xaxis_title="Y", yaxis_title="Z", title="2D Projection: Y vs Z", height=560)

    st.plotly_chart(fig, use_container_width=True)

    # Metrics (mirroring your style)
    m1, m2, m3 = st.columns(3)
    with m1:
        st.metric("σ (Prandtl)", f"{sigma:.2f}")
    with m2:
        st.metric("ρ (Rayleigh)", f"{rho:.2f}")
    with m3:
        st.metric("β", f"{beta:.3f}")

    # Downloads
    if enable_downloads:
        # CSV of trajectory
        csv_data = "t,x,y,z\n" + "\n".join(f"{ti},{xi},{yi},{zi}" for ti, xi, yi, zi in zip(t, x, y, z))
        st.download_button(
            label="📥 Download trajectory (CSV)",
            data=csv_data,
            file_name="lorenz_trajectory.csv",
            mime="text/csv"
        )

        # Interactive HTML
        html_buf = io.StringIO()
        fig.write_html(html_buf, include_plotlyjs="cdn")
        st.download_button(
            label="📥 Download interactive plot (HTML)",
            data=html_buf.getvalue().encode(),
            file_name="lorenz_plot.html",
            mime="text/html"
        )

        # Static PNG (requires 'kaleido' in requirements)
        try:
            png_bytes = fig.to_image(format="png", scale=2)
            st.download_button(
                label="🖼️ Download plot (PNG)",
                data=png_bytes,
                file_name="lorenz_plot.png",
                mime="image/png"
            )
        except Exception:
            st.info("To enable PNG export, add **kaleido** to requirements.txt.")

    # Rich, optional theory (hidden by default, like your detailed physics expander)
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
              \[
              E_{\pm} = \left(\pm\sqrt{\beta(\rho-1)},\ \pm\sqrt{\beta(\rho-1)},\ \rho-1\right).
              \]

            ### 🔒 Stability (classical results)
            - $E_0$ is stable for $0 < \rho < 1$ and loses stability at $\rho=1$.
            - For the standard parameters $\sigma=10,\ \beta=\tfrac{8}{3}$,
              the nontrivial equilibria $E_\pm$ lose stability via a Hopf bifurcation at
              \[
              \rho_H \approx \frac{\sigma(\sigma+\beta+3)}{\sigma-\beta-1} \approx 24.74,
              \]
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

