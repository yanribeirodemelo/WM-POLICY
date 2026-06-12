# -*- coding: utf-8 -*-
"""
Código de Análise de Sensibilidade

@author: yanri
"""
import streamlit as st
import numpy as np
from scipy.integrate import quad
from PIL import Image
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

st.set_page_config(
    page_title="Política de Manutenção",
    page_icon="foto2.png",
    layout="wide"
)

# =============================================================================
# FUNÇÃO DE OTIMIZAÇÃO (Agora isolada, vetorizada e com Cache)
# Isso transformará horas de cálculo em milissegundos!
# =============================================================================
@st.cache_data
def executar_otimizacao(n2, b2, cp, cd, cm, cf, q, s, W_max=50, M_max=50):
    # 1. Definir as funções base
    def fx(x):
        return (b2/n2)*((x/n2)**(b2-1))*np.exp(-(x/n2)**b2)
        
    def x_fx(x):
        return x * fx(x)

    # 2. Pré-computação das integrais (Executado apenas uma vez)
    int_prob = np.zeros(M_max + 2)
    int_xfx = np.zeros(M_max + 2)
    int_prob_inf = np.zeros(M_max + 2)
    
    for k in range(1, M_max + 2):
        int_prob[k] = quad(fx, (k-1)*s, k*s)[0]
        int_xfx[k] = quad(x_fx, (k-1)*s, k*s)[0]
        int_prob_inf[k] = quad(fx, k*s, np.inf)[0]

    # 3. Listas para retornar os dados do gráfico 3D
    lista_W, lista_M, lista_taxa, lista_indisp, lista_confiab = [], [], [], [], []
    
    menortaxa = float('inf')
    Wotm = Motm = menorinatividade = maiorconfiabilidade = 0
    
    # 4. Busca da melhor política
    for W in range(1, W_max + 1):
        for M in range(W, M_max + 1):
            p11 = p12 = p2 = p31 = p32 = p4 = 0
            c11 = c12 = c2 = c31 = c32 = c4 = 0
            t11 = t12 = t2 = t31 = t32 = t4 = 0
            d11 = d12 = d2 = d31 = d32 = d4 = 0 
            
            # --- CASO 1 ---
            for i in range(1, W):
                for j in range(i, M):
                    prob_i = int_prob[i]
                    xfx_i = int_xfx[i]
                    fator = ((1-q)**(j-i)) * q
                    
                    p11 += prob_i * fator
                    c11 += ((cf + cd*j*s) * prob_i - cd * xfx_i) * fator
                    t11 += (j*s * prob_i) * fator
                    d11 += ((j*s * prob_i) - xfx_i) * fator

            for i in range(W, M):
                for j in range(i, M):
                    prob_i = int_prob[i]
                    xfx_i = int_xfx[i]
                    fator = ((1-q)**(j-W)) * q
                    
                    p12 += prob_i * fator
                    c12 += ((cf + cd*j*s) * prob_i - cd * xfx_i) * fator
                    t12 += (j*s * prob_i) * fator
                    d12 += ((j*s * prob_i) - xfx_i) * fator

            # --- CASO 2 ---
            for j in range(W, M):
                prob_inf_j = int_prob_inf[j]
                fator = ((1-q)**(j-W)) * q
                
                p2 += prob_inf_j * fator
                c2 += (cp * prob_inf_j) * fator
                t2 += (j*s * prob_inf_j) * fator

            # --- CASO 3 ---
            for i in range(1, W):
                prob_i = int_prob[i]
                xfx_i = int_xfx[i]
                fator = (1-q)**(M-i)
                
                p31 += prob_i * fator
                c31 += ((cf + cm + cd*M*s) * prob_i - cd * xfx_i) * fator
                t31 += (M*s * prob_i) * fator
                d31 += ((M*s * prob_i) - xfx_i) * fator

            for i in range(W, M + 1): 
                prob_i = int_prob[i]
                xfx_i = int_xfx[i]
                fator = (1-q)**(M-W)
                
                p32 += prob_i * fator
                c32 += ((cf + cm + cd*M*s) * prob_i - cd * xfx_i) * fator
                t32 += (M*s * prob_i) * fator
                d32 += ((M*s * prob_i) - xfx_i) * fator

            # --- CASO 4 ---
            prob_inf_M = int_prob_inf[M]
            fator = (1-q)**(M-W)
            
            p4 += prob_inf_M * fator
            c4 += ((cp + cm) * prob_inf_M) * fator
            t4 += (M*s * prob_inf_M) * fator

            # --- RESULTADOS GLOBAIS DESTA COMBINAÇÃO ---
            pe = p11 + p12 + p2 + p31 + p32 + p4
            ce = c11 + c12 + c2 + c31 + c32 + c4
            ve = t11 + t12 + t2 + t31 + t32 + t4
            de = d11 + d12 + d2 + d31 + d32 + d4
            
            tx = ce/ve if ve != 0 else 0
            dx = de/ve if ve != 0 else 0
            div_mu = (p11+p12+p31+p32)
            mu = ve/div_mu if div_mu != 0 else 0
            
            if tx < menortaxa:
                menortaxa = tx
                Wotm = W
                Motm = M
                menorinatividade = dx
                maiorconfiabilidade = mu
                
            lista_W.append(W)
            lista_M.append(M)
            lista_taxa.append(tx)
            lista_indisp.append(dx)
            lista_confiab.append(mu)
            
    return Wotm, Motm, menortaxa, menorinatividade, maiorconfiabilidade, lista_W, lista_M, lista_taxa, lista_indisp, lista_confiab


def main():
    #criando 3 colunas
    col1, col2, col3 = st.columns(3)
    try:
        foto = Image.open('foto.png')
        #inserindo na coluna 2
        col2.image(foto, use_column_width=True)
    except:
        pass # Ignora caso a foto não seja encontrada no momento
    
    st.title('Modelo de Manutenção para Sistemas de Difícil Acesso')

    menu = ["Simulador", "Otimizador", "Informações", "Grupo de Pesquisa"]
    choice = st.sidebar.selectbox("Selecione aqui", menu)
    
    # =============================================================================
    # MENU 0: SIMULADOR (Roda apenas uma vez com o W e M fixos fornecidos)
    # =============================================================================
    if choice == menu[0]:
        st.header(menu[0])
        st.subheader("Insira os valores dos parâmetros de entrada abaixo:")

        n2 = st.number_input("Insira o parâmetro de escala - {}".format(chr(945)), min_value = 0.0, value = 10.0) 
        b2 = st.number_input("Insira o parâmetro de forma - {}".format(chr(946)), min_value = 1.0, max_value = 6.0, value = 3.0) 

        cp = st.number_input("Insira o parâmetro de custo de substituição preventiva", min_value = 0.0, value = 1.0) 
        cd = st.number_input("Insira o parâmetro de custo de tempo de inatividade por unidade de tempo", min_value = 0.0, value = 0.5) 
        cm = st.number_input("Insira o parâmetro de custo adicional para ação de manutenção garantida", min_value = 0.0, value = 1.00) 
        cf = st.number_input("Insira o parâmetro de custo de substituição corretiva", min_value = 0.0, value = 1.0) 
        
        q = st.number_input("Insira o parâmetro de probabilidade de oportunidade em uma visita", min_value = 0.0, max_value = 1.0, value = 0.2) 
        s = st.number_input("Insira o parâmetro de intervalo de tempo entre visitas", min_value = 0.0, value = 1.0) 
        
        st.subheader("Insira os valores das variáveis de decisão abaixo:")
        
        W = int(st.number_input("Insira o limite inferior das ações de manutenção preventivas por oportunidades", min_value = 0, max_value = 50, step = 1, value = 6)) 
        M = int(st.number_input("Insira a idade de ação de manutenção garantida", min_value = 0, max_value = 50, step = 1, value = 14)) 
        
        st.subheader("Clique no botão abaixo para executar esta aplicação:")
        botao = st.button("Executar")

        if botao:
            # Como aqui o cálculo é apenas para um único par de (W, M), a função antiga pode ser mantida
            # ou simplesmente chamamos a otimizada fixando W_max e M_max no mesmo valor. 
            # Para manter a lógica antiga intacta que funciona bem para UMA execução:
            def otm():
                def fx(x):
                    return (b2/n2)*((x/n2)**(b2-1))*np.exp(-(x/n2)**b2)
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
              
                if pe < 0.999 or pe > 1.001:
                    print("Algum erro ocorreu na probabilidade!")
              
                return tx, ce, ve, pe, dx, mu

            taxadecusto = otm()
            
            st.markdown(
                """
                <style>
                    .box-container {
                        border: 2px solid black; border-radius: 10px; padding: 15px; 
                        text-align: center; width: 100%; display: flex; 
                        justify-content: space-around; align-items: center;
                    }
                    .box-item { flex: 1; }
                    .column-box {
                        border: 2px solid black; border-radius: 10px; padding: 15px;
                        text-align: center; width: 100%; margin-bottom: 10px;
                    }
                </style>
                """, unsafe_allow_html=True
            )
            
            st.markdown(
                f"""
                <div class="box-container">
                    <div class="box-item"><h3>Política de manutenção</h3></div>
                    <div class="box-item"><h3>W = {W}</h3></div>
                    <div class="box-item"><h3>M = {M}</h3></div>
                </div>
                """, unsafe_allow_html=True
            )

            col_res1, col_res2, col_res3 = st.columns(3)
            with col_res1:
                st.markdown(f'<div class="column-box"><h3>Taxa de custo</h3><h2>{round(taxadecusto[0], 3)}</h2></div>', unsafe_allow_html=True)
            with col_res2:
                st.markdown(f'<div class="column-box"><h3>Taxa de indisponibilidade</h3><h2>{round(taxadecusto[4], 3)}</h2></div>', unsafe_allow_html=True)
            with col_res3:
                st.markdown(f'<div class="column-box"><h3>Confiabilidade operacional</h3><h2>{round(taxadecusto[5], 2)}</h2></div>', unsafe_allow_html=True)

    # =============================================================================
    # MENU 1: OTIMIZADOR (Usa a nova rotina vetorizada que calcula muito mais rápido)
    # =============================================================================
    if choice == menu[1]:
        st.header(menu[1])
        st.subheader("Insira os valores dos parâmetros de entrada abaixo:")

        n2 = st.number_input("Insira o parâmetro de escala - {}".format(chr(945)), min_value = 0.0, value = 10.0) 
        b2 = st.number_input("Insira o parâmetro de forma - {}".format(chr(946)), min_value = 1.0, max_value = 6.0, value = 3.0) 

        cp = st.number_input("Insira o parâmetro de custo de substituição preventiva", min_value = 0.0, value = 1.0) 
        cd = st.number_input("Insira o parâmetro de custo de tempo de inatividade por unidade de tempo", min_value = 0.0, value = 0.5) 
        cm = st.number_input("Insira o parâmetro de custo adicional para ação de manutenção garantida", min_value = 0.0, value = 1.00) 
        cf = st.number_input("Insira o parâmetro de custo de substituição corretiva", min_value = 0.0, value = 1.0) 
        
        q = st.number_input("Insira o parâmetro de probabilidade de oportunidade em uma visita", min_value = 0.0, max_value = 1.0, value = 0.2) 
        s = st.number_input("Insira o parâmetro de intervalo de tempo entre visitas", min_value = 0.0, value = 1.0) 

        st.subheader("Clique no botão abaixo para executar esta aplicação:")
        botao = st.button("Executar")

        if botao:
            with st.spinner("Processando otimização... Aguarde alguns instantes (está muito mais rápido agora!)"):
                (Wotm, Motm, menortaxa, menorinatividade, maiorconfiabilidade, 
                 lista_W, lista_M, lista_taxa, lista_indisp, lista_confiab) = executar_otimizacao(
                    n2, b2, cp, cd, cm, cf, q, s
                )
            
            st.success("Execução concluída com sucesso!")
            
            st.markdown(
                """
                <style>
                    .box-container {
                        border: 2px solid black; border-radius: 10px; padding: 15px; 
                        text-align: center; width: 100%; display: flex; 
                        justify-content: space-around; align-items: center;
                    }
                    .box-item { flex: 1; }
                    .column-box {
                        border: 2px solid black; border-radius: 10px; padding: 15px;
                        text-align: center; width: 100%; margin-bottom: 10px;
                    }
                </style>
                """, unsafe_allow_html=True
            )
            
            st.markdown(
                f"""
                <div class="box-container">
                    <div class="box-item"><h3>Política de manutenção</h3></div>
                    <div class="box-item"><h3>W = {Wotm}</h3></div>
                    <div class="box-item"><h3>M = {Motm}</h3></div>
                </div>
                """, unsafe_allow_html=True
            )
    
            col_res1, col_res2, col_res3 = st.columns(3)
    
            delta = 5 
            w_min = max(Wotm - delta, 1)
            w_max = min(Wotm + delta, 50)
            m_min = max(Motm - delta, 1)
            m_max = min(Motm + delta, 50)
                
            indices_zoom = [i for i, (w, m) in enumerate(zip(lista_W, lista_M)) 
                            if abs(w - Wotm) <= delta and abs(m - Motm) <= delta]
            
            lista_W_zoom = [lista_W[i] for i in indices_zoom]
            lista_M_zoom = [lista_M[i] for i in indices_zoom]
            lista_taxa_zoom = [lista_taxa[i] for i in indices_zoom]
            lista_indisp_zoom = [lista_indisp[i] for i in indices_zoom]
            lista_confiab_zoom = [lista_confiab[i] for i in indices_zoom]
    
            def criar_grafico_3D(x, y, z, eixo_z):
                fig = plt.figure(figsize=(8, 6), dpi=150)
                ax = fig.add_subplot(111, projection='3d')
                sc = ax.scatter(x, y, z, c=z, cmap='viridis', s=20, alpha=0.85)
                ax.set_xlim(w_min, w_max)
                ax.set_ylim(m_min, m_max)
                ax.set_xlabel("W", fontsize=12, labelpad=10)
                ax.set_ylabel("M", fontsize=12, labelpad=10)
                ax.xaxis.pane.fill = False
                ax.yaxis.pane.fill = False
                ax.zaxis.pane.fill = False
                ax.grid(False)
                cbar = plt.colorbar(sc, shrink=0.6, aspect=10)
                cbar.set_label(eixo_z, fontsize=12)
                return fig

            with col_res1:
                st.markdown(f'<div class="column-box"><h3>Taxa de custo</h3><h2>{round(menortaxa, 3)}</h2></div>', unsafe_allow_html=True)
                fig_taxa = criar_grafico_3D(lista_W_zoom, lista_M_zoom, lista_taxa_zoom, "Taxa de Custo")
                st.pyplot(fig_taxa)
            
            with col_res2:
                st.markdown(f'<div class="column-box"><h3>Taxa de indisponibilidade</h3><h2>{round(menorinatividade, 3)}</h2></div>', unsafe_allow_html=True)
                fig_indisp = criar_grafico_3D(lista_W_zoom, lista_M_zoom, lista_indisp_zoom, "Taxa de Indisponibilidade")
                st.pyplot(fig_indisp)
            
            with col_res3:
                st.markdown(f'<div class="column-box"><h3>Confiabilidade operacional</h3><h2>{round(maiorconfiabilidade, 2)}</h2></div>', unsafe_allow_html=True)
                fig_confiab = criar_grafico_3D(lista_W_zoom, lista_M_zoom, lista_confiab_zoom, "Confiabilidade Operacional")
                st.pyplot(fig_confiab)
    
            st.markdown(
                """
                <style>
                    .justificado { text-align: justify; }
                </style>
                """, unsafe_allow_html=True
            )
    
            texto = '''Este protótipo possui restrições quanto ao espaço de busca de soluções, com W,M ∈ {1,...,50}. Se for do interesse do usuário utilizar uma gama maior de combinações de soluções ou se houver alguma dúvida sobre o estudo e/ou este protótipo, elas podem ser direcionadas para qualquer um dos endereços de e-mail abaixo. Por fim, se esta aplicação for utilizada para qualquer propósito, todos os autores devem ser informados.'''
            st.markdown(f'<p class="justificado">{texto}</p>', unsafe_allow_html=True)
            st.write('''y.r.melo@random.org.br''')
            st.write('''c.a.v.cavalcante@random.org.br''')

    # =============================================================================
    # MENU 2 e 3: INFORMAÇÕES E CONTATOS
    # =============================================================================
    if choice == menu[2]:
        st.header(menu[2])
        st.markdown(
                """
                <style>
                    .justificado { text-align: justify; }
                </style>
                """, unsafe_allow_html=True
            )
        
        texto1 = '''Este produto tecnológico, desenvolvido por (Melo; Cavalcante, 2025), para apoiar a tomada de decisão na manutenção de sistemas críticos, como geração e distribuição de energia, abastecimento de água e saneamento básico, especialmente aqueles localizados em áreas remotas ou com infraestrutura complexa. Esses sistemas apresentam desafios operacionais significativos, exigindo estratégias de manutenção que conciliem eficácia e viabilidade operacional. A solução proposta baseia-se em um modelo de manutenção estruturado em períodos discretos, permitindo a integração de ações preventivas e corretivas conforme a disponibilidade de recursos, caracterizada pela ocorrência de oportunidades. Essa abordagem periódica facilita o planejamento estratégico das intervenções e aproxima a teoria da  prática. A otimização da política de manutenção é realizada por algoritmos numéricos que avaliam diferentes cenários de renovação do sistema. O modelo considera fatores como os custos das manutenções corretiva e preventiva, o custo da inatividade após uma falha, o custo para garantir uma ação de manutenção, a probabilidade de surgimento de oportunidades em visitas, os intervalos entre os períodos de visita e a vida útil do sistema. A aplicação numérica do modelo, aliada a uma análise de sensibilidade, demonstra que a abordagem proposta é particularmente eficaz em cenários com recursos limitados e poucas oportunidades de manutenção. Além disso, a comparação com políticas de manutenção contínuas evidencia que a discretização dos períodos não compromete significativamente o desempenho do sistema, reforçando a viabilidade prática da metodologia. Esse produto tecnológico oferece uma ferramenta de suporte à decisão acessível e intuitiva, permitindo simulações e otimizações da política de manutenção para diferentes contextos, auxiliando gestores e engenheiros na definição de estratégias eficientes para a gestão de ativos críticos.'''
        texto2 = '''Este produto tecnológico possui restrições quanto ao espaço de busca de soluções, com W,M ∈ {1,...,50}. Se for do interesse do usuário utilizar uma gama maior de combinações de soluções ou se houver alguma dúvida sobre o estudo e/ou este protótipo, elas podem ser direcionadas para qualquer um dos endereços de e-mail abaixo. Por fim, se esta aplicação for utilizada para qualquer propósito, todos os autores devem ser informados.'''
        texto3 = '''Uma representação gráfica da política de manutenção proposta é apresentada abaixo, sendo s o intervalo de tempo entre visitas, W o limite inferior da janela de ações de manutenção preventiva por oportunidade, e M a idade de ação de manutenção garantida'''
        st.markdown(f'<p class="justificado">{texto1}</p>', unsafe_allow_html=True)
        st.markdown(f'<p class="justificado">{texto3}</p>', unsafe_allow_html=True)

        col1, col2, col3 = st.columns(3)
        try:
            foto3 = Image.open('foto3.png')
            col2.image(foto3, use_column_width=True)
        except:
            pass
        
        st.markdown(f'<p class="justificado">{texto2}</p>', unsafe_allow_html=True)
        st.write('''y.r.melo@random.org.br''')
        st.write('''c.a.v.cavalcante@random.org.br''')
        st.markdown(
            """
            <p style="text-align: justify;">
                <b>MELO, Yan Ribeiro.</b> 
                <i>Proposição de Modelo de Manutenção para Sistemas de Difícil Acesso.</i> 
                2025. 84 f. Dissertação (Mestrado em Engenharia de Produção) – Universidade Federal de Pernambuco, Recife, 2025.
            </p>
            """, unsafe_allow_html=True
        )

    if choice == menu[3]:
        st.header(menu[3])
        st.write("O Grupo de Pesquisa em Risco e Análise da Decisão em Operações e Manutenção foi criado em 2012 com o objetivo de reunir diferentes pesquisadores que atuam nas seguintes áreas: risco, modelagem de manutenção e operação. Saiba mais sobre o grupo através do nosso site.")
        st.markdown('[Clique aqui para ser redirecionado ao nosso site](https://sites.ufpe.br/random/#page-top)', False)

if __name__ == "__main__": 
    main()
