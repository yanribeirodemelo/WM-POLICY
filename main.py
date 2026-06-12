# -*- coding: utf-8 -*-
"""
Maintenance Policy Optimization and Simulation
"""
import streamlit as st
import numpy as np
from scipy.integrate import quad
from PIL import Image
import matplotlib.pyplot as plt

st.set_page_config(
    page_title="Maintenance Policy",
    page_icon="foto2.png",
    layout="wide"
)

# =============================================================================
# OPTIMIZATION FUNCTION (Vectorized and Cached)
# =============================================================================
@st.cache_data
def execute_optimization(n2, b2, cp, cd, cm, cf, q, s, W_max=50, M_max=50):
    # 1. Base functions
    def fx(x):
        return (b2/n2)*((x/n2)**(b2-1))*np.exp(-(x/n2)**b2)
        
    def x_fx(x):
        return x * fx(x)

    # 2. Pre-computation of integrals
    int_prob = np.zeros(M_max + 2)
    int_xfx = np.zeros(M_max + 2)
    int_prob_inf = np.zeros(M_max + 2)
    
    for k in range(1, M_max + 2):
        int_prob[k] = quad(fx, (k-1)*s, k*s)[0]
        int_xfx[k] = quad(x_fx, (k-1)*s, k*s)[0]
        int_prob_inf[k] = quad(fx, k*s, np.inf)[0]

    # 3. Output lists for 3D plotting
    list_W, list_M, list_rate, list_unavail, list_rel = [], [], [], [], []
    
    min_rate = float('inf')
    W_opt = M_opt = min_unavail = max_rel = 0
    
    # 4. Search for the optimal policy
    for W in range(1, W_max + 1):
        for M in range(W, M_max + 1):
            p11 = p12 = p2 = p31 = p32 = p4 = 0
            c11 = c12 = c2 = c31 = c32 = c4 = 0
            t11 = t12 = t2 = t31 = t32 = t4 = 0
            d11 = d12 = d2 = d31 = d32 = d4 = 0 
            
            # --- CASE 1 ---
            for i in range(1, W):
                for j in range(i, M):
                    prob_i = int_prob[i]
                    xfx_i = int_xfx[i]
                    factor = ((1-q)**(j-i)) * q
                    
                    p11 += prob_i * factor
                    c11 += ((cf + cd*j*s) * prob_i - cd * xfx_i) * factor
                    t11 += (j*s * prob_i) * factor
                    d11 += ((j*s * prob_i) - xfx_i) * factor

            for i in range(W, M):
                for j in range(i, M):
                    prob_i = int_prob[i]
                    xfx_i = int_xfx[i]
                    factor = ((1-q)**(j-W)) * q
                    
                    p12 += prob_i * factor
                    c12 += ((cf + cd*j*s) * prob_i - cd * xfx_i) * factor
                    t12 += (j*s * prob_i) * factor
                    d12 += ((j*s * prob_i) - xfx_i) * factor

            # --- CASE 2 ---
            for j in range(W, M):
                prob_inf_j = int_prob_inf[j]
                factor = ((1-q)**(j-W)) * q
                
                p2 += prob_inf_j * factor
                c2 += (cp * prob_inf_j) * factor
                t2 += (j*s * prob_inf_j) * factor

            # --- CASE 3 ---
            for i in range(1, W):
                prob_i = int_prob[i]
                xfx_i = int_xfx[i]
                factor = (1-q)**(M-i)
                
                p31 += prob_i * factor
                c31 += ((cf + cm + cd*M*s) * prob_i - cd * xfx_i) * factor
                t31 += (M*s * prob_i) * factor
                d31 += ((M*s * prob_i) - xfx_i) * factor

            for i in range(W, M + 1): 
                prob_i = int_prob[i]
                xfx_i = int_xfx[i]
                factor = (1-q)**(M-W)
                
                p32 += prob_i * factor
                c32 += ((cf + cm + cd*M*s) * prob_i - cd * xfx_i) * factor
                t32 += (M*s * prob_i) * factor
                d32 += ((M*s * prob_i) - xfx_i) * factor

            # --- CASE 4 ---
            prob_inf_M = int_prob_inf[M]
            factor = (1-q)**(M-W)
            
            p4 += prob_inf_M * factor
            c4 += ((cp + cm) * prob_inf_M) * factor
            t4 += (M*s * prob_inf_M) * factor

            # --- GLOBAL RESULTS FOR COMBINATION ---
            pe = p11 + p12 + p2 + p31 + p32 + p4
            ce = c11 + c12 + c2 + c31 + c32 + c4
            ve = t11 + t12 + t2 + t31 + t32 + t4
            de = d11 + d12 + d2 + d31 + d32 + d4
            
            tx = ce/ve if ve != 0 else 0
            dx = de/ve if ve != 0 else 0
            div_mu = (p11+p12+p31+p32)
            mu = ve/div_mu if div_mu != 0 else 0
            
            if tx < min_rate:
                min_rate = tx
                W_opt = W
                M_opt = M
                min_unavail = dx
                max_rel = mu
                
            list_W.append(W)
            list_M.append(M)
            list_rate.append(tx)
            list_unavail.append(dx)
            list_rel.append(mu)
            
    return W_opt, M_opt, min_rate, min_unavail, max_rel, list_W, list_M, list_rate, list_unavail, list_rel

# =============================================================================
# UI COMPONENTS AND MAIN APP
# =============================================================================
def display_input_parameters():
    with st.container(border=True):
        st.subheader("System and Cost Parameters")
        col1, col2 = st.columns(2)
        
        with col1:
            n2 = st.number_input(f"Characteristic lifetime - {chr(951)}", min_value=0.0, value=10.0, step=1.0) 
            b2 = st.number_input(f"Shape parameter - {chr(946)}", min_value=1.0, max_value=6.0, value=3.0, step=0.1) 
            q = st.number_input("Opportunity probability at a slot (q)", min_value=0.0, max_value=1.0, value=0.2, step=0.1) 
            s = st.number_input("Time interval between slots (s)", min_value=0.0, value=1.0, step=0.5) 
            
        with col2:
            cp = st.number_input("Cost of preventive replacement (cP)", min_value=0.0, value=1.0, step=0.5) 
            cf = st.number_input("Cost of corrective replacement (cF)", min_value=0.0, value=1.0, step=0.5) 
            cd = st.number_input("Rate of cost of downtime (cD)", min_value=0.0, value=0.5, step=0.1) 
            cm = st.number_input("Additional cost for guaranteed action (cM)", min_value=0.0, value=1.0, step=0.5) 

    return n2, b2, cp, cd, cm, cf, q, s

def create_3d_plot(x, y, z, z_label, w_min, w_max, m_min, m_max):
    fig = plt.figure(figsize=(8, 6), dpi=150)
    ax = fig.add_subplot(111, projection='3d')
    sc = ax.scatter(x, y, z, c=z, cmap='viridis', s=20, alpha=0.85)
    
    ax.set_xlim(w_min, w_max)
    ax.set_ylim(m_min, m_max)
    ax.set_xlabel("W", fontsize=10, labelpad=10)
    ax.set_ylabel("M", fontsize=10, labelpad=10)
    
    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False
    ax.grid(False)
    
    cbar = plt.colorbar(sc, shrink=0.6, aspect=10)
    cbar.set_label(z_label, fontsize=10)
    return fig


def main():
    col1, col2, col3 = st.columns([1, 2, 1])
    try:
        foto = Image.open('logo.png')
        col2.image(foto, use_column_width=True)
    except:
        pass 
    
    st.title("Quasi-Periodic Opportunistic Maintenance Policy")

    menu = ["Simulator", "Optimizer", "Information", "Research Group"]
    choice = st.sidebar.radio("Navigation", menu)
    
    # =============================================================================
    # MENU 0: SIMULATOR
    # =============================================================================
    if choice == menu[0]:
        st.header("Simulator")
        n2, b2, cp, cd, cm, cf, q, s = display_input_parameters()
        
        with st.container(border=True):
            st.subheader("Decision Variables")
            col_w, col_m = st.columns(2)
            with col_w:
                W = int(st.number_input("Lower limit for preventive maintenance (W)", min_value=0, max_value=50, step=1, value=6)) 
            with col_m:
                M = int(st.number_input("Age of guaranteed maintenance action (M)", min_value=0, max_value=50, step=1, value=14)) 
        
        if st.button("Execute Simulation"):
            def otm_single():
                def fx(x): return (b2/n2)*((x/n2)**(b2-1))*np.exp(-(x/n2)**b2)
                p11 = p12 = p2 = p31 = p32 = p4 = 0
                c11 = c12 = c2 = c31 = c32 = c4 = 0
                t11 = t12 = t2 = t31 = t32 = t4 = 0
                d11 = d12 = d2 = d31 = d32 = d4 = 0 
              
                def fprob1(x): return fx(x)
                def fcusto1(x, j): return (cf+cd*(j*s-x))*fx(x)
                def fvida1(x, j): return (j*s)*fx(x)
                def fdown1(x, j): return (j*s-x)*fx(x)
              
                for i in range(1, W):
                    for j in range(i, M):
                        p11 += quad(fprob1, (i-1)*s, i*s)[0]*((1-q)**(j-i))*(q)
                        c11 += quad(lambda x: fcusto1(x, j), (i-1)*s, i*s)[0]*((1-q)**(j-i))*(q)
                        t11 += quad(lambda x: fvida1(x, j), (i-1)*s, i*s)[0]*((1-q)**(j-i))*(q)
                        d11 += quad(lambda x: fdown1(x, j), (i-1)*s, i*s)[0]*((1-q)**(j-i))*(q)
                        
                for i in range(W, M):
                    for j in range(i, M):
                        p12 += quad(fprob1, (i-1)*s, i*s)[0]*((1-q)**(j-W))*(q)
                        c12 += quad(lambda x: fcusto1(x, j), (i-1)*s, i*s)[0]*((1-q)**(j-W))*(q)
                        t12 += quad(lambda x: fvida1(x, j), (i-1)*s, i*s)[0]*((1-q)**(j-W))*(q)
                        d12 += quad(lambda x: fdown1(x, j), (i-1)*s, i*s)[0]*((1-q)**(j-W))*(q)
                  
                def fcusto2(x): return cp*fx(x)
              
                for j in range(W, M):
                    p2 += quad(fprob1, j*s, np.inf)[0]*(((1-q)**(j-W))*q)
                    c2 += quad(fcusto2, j*s, np.inf)[0]*(((1-q)**(j-W))*q)
                    t2 += quad(lambda x: fvida1(x, j), j*s, np.inf)[0]*(((1-q)**(j-W))*q)
                  
                def fcusto3(x): return (cf+cd*(M*s-x)+cm)*fx(x)
                def fvida3(x): return (M*s)*fx(x)
                def fdown3(x): return (M*s-x)*fx(x)
              
                for i in range(1, W):
                    p31 += quad(fprob1, (i-1)*s, i*s)[0]*((1-q)**(M-i))
                    c31 += quad(fcusto3, (i-1)*s, i*s)[0]*((1-q)**(M-i))
                    t31 += quad(fvida3, (i-1)*s, i*s)[0]*((1-q)**(M-i))
                    d31 += quad(fdown3, (i-1)*s, i*s)[0]*((1-q)**(M-i))
              
                for i in range(W, M+1):
                    p32 += quad(fprob1, (i-1)*s, i*s)[0]*((1-q)**(M-W))
                    c32 += quad(fcusto3, (i-1)*s, i*s)[0]*((1-q)**(M-W))
                    t32 += quad(fvida3, (i-1)*s, i*s)[0]*((1-q)**(M-W))
                    d32 += quad(fdown3, (i-1)*s, i*s)[0]*((1-q)**(M-W))
              
                def fcusto4(x): return (cp+cm)*fx(x)
                
                p4 += quad(fprob1, M*s, np.inf)[0]*(((1-q)**(M-W)))
                c4 += quad(fcusto4, M*s, np.inf)[0]*(((1-q)**(M-W)))
                t4 += quad(fvida3, M*s, np.inf)[0]*(((1-q)**(M-W)))
  
                pe = p11+p12+p2+p31+p32+p4 
                ce = c11+c12+c2+c31+c32+c4 
                ve = t11+t12+t2+t31+t32+t4 
                de = d11+d12+d2+d31+d32+d4 
                
                tx = ce/ve if ve else 0
                dx = de/ve if ve else 0
                mu = ve/(p11+p12+p31+p32) if (p11+p12+p31+p32) else 0
              
                return tx, ce, ve, pe, dx, mu

            with st.spinner("Processing simulation..."):
                cost_rate_sim = otm_single()
            
            st.success("Simulation complete.")
            
            st.markdown(f"**Selected Maintenance Policy:** W = {W} | M = {M}")
            
            col_res1, col_res2, col_res3 = st.columns(3)
            col_res1.metric(label="Cost-rate", value=f"{round(cost_rate_sim[0], 3)}")
            col_res2.metric(label="Average Unavailability", value=f"{round(cost_rate_sim[4], 3)}")
            col_res3.metric(label="Operational Reliability (MTBOF)", value=f"{round(cost_rate_sim[5], 2)}")

    # =============================================================================
    # MENU 1: OPTIMIZER
    # =============================================================================
    if choice == menu[1]:
        st.header("Optimizer")
        n2, b2, cp, cd, cm, cf, q, s = display_input_parameters()

        if st.button("Execute Optimization"):
            with st.spinner("Processing optimal policy..."):
                (W_opt, M_opt, min_rate, min_unavail, max_rel, 
                 list_W, list_M, list_rate, list_unavail, list_rel) = execute_optimization(
                    n2, b2, cp, cd, cm, cf, q, s
                )
            
            st.success("Optimization complete.")
            
            st.markdown(f"**Optimal Maintenance Policy:** W = {W_opt} | M = {M_opt}")
    
            col_res1, col_res2, col_res3 = st.columns(3)
            col_res1.metric(label="Minimum Cost-rate", value=f"{round(min_rate, 3)}")
            col_res2.metric(label="Average Unavailability", value=f"{round(min_unavail, 3)}")
            col_res3.metric(label="Operational Reliability", value=f"{round(max_rel, 2)}")
    
            # Local zoom variables for charting
            delta = 5 
            w_min = max(W_opt - delta, 1)
            w_max = min(W_opt + delta, 50)
            m_min = max(M_opt - delta, 1)
            m_max = min(M_opt + delta, 50)
                
            indices_zoom = [i for i, (w, m) in enumerate(zip(list_W, list_M)) 
                            if abs(w - W_opt) <= delta and abs(m - M_opt) <= delta]
            
            list_W_zoom = [list_W[i] for i in indices_zoom]
            list_M_zoom = [list_M[i] for i in indices_zoom]
            list_rate_zoom = [list_rate[i] for i in indices_zoom]
            list_unavail_zoom = [list_unavail[i] for i in indices_zoom]
            list_rel_zoom = [list_rel[i] for i in indices_zoom]

            col_chart1, col_chart2, col_chart3 = st.columns(3)
            with col_chart1:
                fig_rate = create_3d_plot(list_W_zoom, list_M_zoom, list_rate_zoom, "Cost-rate", w_min, w_max, m_min, m_max)
                st.pyplot(fig_rate)
            with col_chart2:
                fig_unavail = create_3d_plot(list_W_zoom, list_M_zoom, list_unavail_zoom, "Unavailability", w_min, w_max, m_min, m_max)
                st.pyplot(fig_unavail)
            with col_chart3:
                fig_rel = create_3d_plot(list_W_zoom, list_M_zoom, list_rel_zoom, "Reliability", w_min, w_max, m_min, m_max)
                st.pyplot(fig_rel)
    
            st.info("Note: This prototype evaluates a restricted solution space where W, M ∈ {1,...,50}. If you wish to use a wider range of solutions or have questions regarding the study, please contact the authors. If this application is used for any purpose, all authors must be acknowledged.")
            st.caption("Contact: y.r.melo@random.org.br | c.a.v.cavalcante@random.org.br")

    # =============================================================================
    # MENU 2: INFORMATION
    # =============================================================================
    if choice == menu[2]:
        st.header("Information")
        
        st.markdown(
            """
            This technological product aims to provide decision support for the maintenance of critical systems where resources are limited or maintenance execution is not guaranteed. Motivating contexts include offshore wind farms, remote telecommunications installations, or power micro-grids.

            The proposed solution is based on an aged-based replacement policy structured in discrete periods. The instances for component replacement are restricted to instances of time (slots) that arise periodically. At each slot, an opportunity for replacement arises with a specific probability, characterizing the limitation of resources.

            **The {W, M}-Policy Decision Process:**
            The policy consists of two integer-valued decision variables (control limits), W and M. 
            * **First phase [0, Ws):** A corrective phase where the system is replaced at a slot only if it has failed and an opportunity arises.
            * **Second phase [Ws, Ms]:** An opportunistic phase where the system is preventively replaced at a slot, regardless of its state, if an opportunity arises. At the final slot Ms, replacement is guaranteed.

            A graphical representation of the proposed maintenance policy decision process is presented below:
            """
        )

        col1, col2, col3 = st.columns([1, 2, 1])
        try:
            foto3 = Image.open('foto3.png')
            col2.image(foto3, use_column_width=True, caption="Decision process at a slot for the quasi-periodic opportunistic policy.")
        except:
            pass
        
        st.info("This web-application allows maintainers and engineers to evaluate the cost-benefit of opportunistic replacements, calculating long-run cost per unit time and average availability in a renewal-reward framework.")
        
        st.subheader("Reference")
        st.markdown(
            """
            > Cavalcante, C. A. V., Scarf, P., Melo, Y. R., Rodrigues, A. J. S., & Alotaibi, N. (2024). 
            > **Planning maintenance when resources are limited: a study of periodic opportunistic replacement.** > *IMA Journal of Management Mathematics*, 35(4), 573-593.
            """
        )
        st.caption("Contact: y.r.melo@random.org.br | c.a.v.cavalcante@random.org.br")

    # =============================================================================
    # MENU 3: RESEARCH GROUP
    # =============================================================================
    if choice == menu[3]:
        st.header("Research Group")
        st.write("The Research Group on Risk and Decision Analysis in Operations and Maintenance (RANDOM) was created in 2012 with the objective of gathering researchers working in risk assessment, maintenance modeling, and operations research. Learn more about the group through our website.")
        st.markdown("[Click here to be redirected to the RANDOM website](https://sites.ufpe.br/random/#page-top)", unsafe_allow_html=True)

if __name__ == "__main__": 
    main()
