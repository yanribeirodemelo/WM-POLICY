# -*- coding: utf-8 -*-
"""
Código de Análise de Sensibilidade

@author: yanri
"""
import streamlit as st
import numpy as np
import sys
from scipy.integrate import quad
from PIL import Image
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

st.set_page_config(
    page_title="Política de Manutenção",
    page_icon="foto2.png",
    layout="wide"
)

# import streamlit.report_thread as ReportThread
# st.report_thread.add_report_ctx(thread)

def main():
    #criando 3 colunas
    col1, col2, col3 = st.columns(3)
    foto = Image.open('foto.png')
    #inserindo na coluna 2
    col2.image(foto, use_column_width=True)
    
    st.title('Modelo de Manutenção para Sistemas de Difícil Acesso')

    menu = ["Simulador", "Otimizador", "Informações", "Grupo de Pesquisa"]
    
    choice = st.sidebar.selectbox("Selecione aqui", menu)
    
    if choice == menu[0]:
        
        st.header(menu[0])

        st.subheader("Insira os valores dos parâmetros de entrada abaixo:")

        n2 = st.number_input("Insira o parâmetro de escala - {}".format(chr(945)), min_value = 0.0, value = 10.0) # escala fraca
        b2 = st.number_input("Insira o parâmetro de forma - {}".format(chr(946)), min_value = 1.0, max_value = 6.0, value = 3.0) # forma fraca

        cp = st.number_input("Insira o parâmetro de custo de substituição preventiva", min_value = 0.0, value = 1.0) # custo da preventiva
        cd = st.number_input("Insira o parâmetro de custo de tempo de inatividade por unidade de tempo", min_value = 0.0, value = 0.5) # custo de downtime por unidade de tempo
        cm = st.number_input("Insira o parâmetro de custo adicional para ação de manutenção garantida", min_value = 0.0, value = 1.00) # custo de inspeção
        cf = st.number_input("Insira o parâmetro de custo de substituição corretiva", min_value = 0.0, value = 1.0) # custo de falha
        
        q = st.number_input("Insira o parâmetro de probabilidade de oportunidade em uma visita", min_value = 0.0, max_value = 1.0, value = 0.2) # taxa de chegada de oportunidades
        s = st.number_input("Insira o parâmetro de intervalo de tempo entre visitas", min_value = 0.0, value = 1.0) # intervalo entre visitas
        
        st.subheader("Insira os valores das variáveis de decisão abaixo:")
        
        W = int(st.number_input("Insira o limite inferior das ações de manutenção preventivas por oportunidades", min_value = 0, max_value = 50, step = 1, value = 6)) # Limite inferior da janela de oportunidades
        M = int(st.number_input("Insira a idade de ação de manutenção garantida", min_value = 0, max_value = 50, step = 1, value = 14)) # Limite inferior da janela de oportunidades
        
        st.subheader("Clique no botão abaixo para executar esta aplicação:")
        botao = st.button("Executar")

        if botao:
        
            def otm():
              def fx(x):#weibull density func (for strong comp) for arrival of fails
                  return (b2/n2)*((x/n2)**(b2-1))*np.exp(-(x/n2)**b2)
            
              #%% VARIÁVEIS E CONSTANTES
            
              # #Parâmetros
              # b2 = b2[i] #Forma do componente forte
              # n2 = n2[i] #Escala do componente forte     
              # cd = cd[i] #custo de downtime
              # cm = cm[i] #custo logístico
              # cp = cp[i] #custo de manutenção preventiva
              # cf = cf[i] #custo de manutenção corretiva
                 
              # s = s[i] #Intervalo entre visitas
              # q = q[i] #Oportunidade de manutenção
            
              #%%Variáveis auxiliares
              p11 = p12 = p2 = p31 = p32 = p4 = 0
              c11 = c12 = c2 = c31 = c32 = c4 = 0
              t11 = t12 = t2 = t31 = t32 = t4 = 0
              d11 = d12 = d2 = d31 = d32 = d4 = 0 
            
              #%%CASO 1
              
              def fprob1(x):
                  return ((fx(x)))
              def fcusto1(x):
                  return ((cf+cd*(j*s-x))*(fx(x)))
              def fvida1(x):
                  return ((j*s)*(fx(x)))
              def fdown1(x):
                  return ((j*s-x)*(fx(x)))
            
              for i in range(1, W):
                  for j in range(i, M):
                      fp11 = quad(fprob1, (i-1)*s, i*s)[0]*((1-q)**(j-i))*(q)
                      p11 = p11 + fp11
                      
                      fc11 = quad(fcusto1, (i-1)*s, i*s)[0]*((1-q)**(j-i))*(q)
                      c11 = c11 + fc11
                      
                      ft11 = quad(fvida1, (i-1)*s, i*s)[0]*((1-q)**(j-i))*(q)
                      t11 = t11 + ft11
                      
                      fd11 = quad(fdown1, (i-1)*s, i*s)[0]*((1-q)**(j-i))*(q)
                      d11 = d11 + fd11
                      
              for i in range(W, M):
                  for j in range(i, M):
                      fp12 = quad(fprob1, (i-1)*s, i*s)[0]*((1-q)**(j-W))*(q)
                      p12 = p12 + fp12
                      
                      fc12 = quad(fcusto1, (i-1)*s, i*s)[0]*((1-q)**(j-W))*(q)
                      c12 = c12 + fc12
                      
                      ft12 = quad(fvida1, (i-1)*s, i*s)[0]*((1-q)**(j-W))*(q)
                      t12 = t12 + ft12
                      
                      fd12 = quad(fdown1, (i-1)*s, i*s)[0]*((1-q)**(j-W))*(q)
                      d12 = d12 + fd12
                
              #%%CASO 2
            
              def fprob2(x):
                  return ((fx(x)))
              def fcusto2(x):
                  return ((cp)*(fx(x)))
              def fvida2(x):
                  return ((j*s)*(fx(x)))
              def fdown2(x):
                  return ((0)*(fx(x)))
            
              for j in range(W, M):
                fp2 = quad(fprob2, j*s, np.inf)[0]*(((1-q)**(j-W))*q)
                p2 = p2 + fp2
                  
                fc2 = quad(fcusto2, j*s, np.inf)[0]*(((1-q)**(j-W))*q)
                c2 = c2 + fc2
                
                ft2 = quad(fvida2, j*s, np.inf)[0]*(((1-q)**(j-W))*q)
                t2 = t2 + ft2
                
                fd2 = quad(fdown2, j*s, np.inf)[0]*(((1-q)**(j-W))*q)
                d2 = d2 + fd2 
                
              #%%CASO 3
              
              def fprob3(x):
                  return ((fx(x)))
              def fcusto3(x):
                  return ((cf+cd*(M*s-x)+cm)*(fx(x)))
              def fvida3(x):
                  return ((M*s)*(fx(x)))
              def fdown3(x):
                  return ((M*s-x)*(fx(x)))
            
              for i in range(1, W):
                  fp31 = quad(fprob3, (i-1)*s, i*s)[0]*((1-q)**(M-i))
                  p31 = p31 + fp31
                  
                  fc31 = quad(fcusto3, (i-1)*s, i*s)[0]*((1-q)**(M-i))
                  c31 = c31 + fc31
                  
                  ft31 = quad(fvida3, (i-1)*s, i*s)[0]*((1-q)**(M-i))
                  t31 = t31 + ft31
                  
                  fd31 = quad(fdown3, (i-1)*s, i*s)[0]*((1-q)**(M-i))
                  d31 = d31 + fd31   
                  
            
              for i in range(W, M+1):
                  fp32 = quad(fprob3, (i-1)*s, i*s)[0]*((1-q)**(M-W))
                  p32 = p32 + fp32
                  
                  fc32 = quad(fcusto3, (i-1)*s, i*s)[0]*((1-q)**(M-W))
                  c32 = c32 + fc32
                  
                  ft32 = quad(fvida3, (i-1)*s, i*s)[0]*((1-q)**(M-W))
                  t32= t32 + ft32
                  
                  fd32 = quad(fdown3, (i-1)*s, i*s)[0]*((1-q)**(M-W))
                  d32 = d32 + fd32  
            
              #%%CASO 4
              
              def fprob4(x):
                  return ((fx(x)))
              def fcusto4(x):
                  return ((cp+cm)*(fx(x)))
              def fvida4(x):
                  return ((M*s)*(fx(x)))
              def fdown4(x):
                  return ((0)*(fx(x)))
              
              fp4 = quad(fprob4, M*s, np.inf)[0]*(((1-q)**(M-W)))
              p4 = p4 + fp4
              
              fc4 = quad(fcusto4, M*s, np.inf)[0]*(((1-q)**(M-W)))
              c4 = c4 + fc4
              
              ft4 = quad(fvida4, M*s, np.inf)[0]*(((1-q)**(M-W)))
              t4= t4 + ft4
              
              fd4 = quad(fdown4, M*s, np.inf)[0]*(((1-q)**(M-W)))
              d4 = d4 + fd4  
            
              #%%RESULTADOS

                
              pe = p11+p12+p2+p31+p32+p4 #probabilidade total
              ce = c11+c12+c2+c31+c32+c4 #custo total
              ve = t11+t12+t2+t31+t32+t4 #tamanho total
              de = d11+d12+d2+d31+d32+d4 #downtime total
              
              tx = ce/ve #taxa de custo
              dx = de/ve #taxa de downtime
              mu = ve/(p11+p12+p31+p32) #mtbof
            
              if pe < 0.999 or pe > 1.001:
                print("Algum erro ocorreu!")
            
              return tx, ce, ve, pe, dx, mu

            taxadecusto = otm()
            st.markdown(
                """
                <style>
                    .box-container {
                        border: 2px solid black; /* Cor da borda */
                        border-radius: 10px; /* Bordas arredondadas */
                        padding: 15px; /* Espaço interno */
                        text-align: center;
                        width: 100%; /* Garante que a largura ocupe toda a linha */
                        display: flex; /* Para organizar os elementos lado a lado */
                        justify-content: space-around; /* Espaço uniforme entre elementos */
                        align-items: center; /* Alinha verticalmente */
                    }
                    .box-item {
                        flex: 1; /* Permite que os itens fiquem do mesmo tamanho */
                    }
                </style>
                """,
                unsafe_allow_html=True,
            )
            
            # Criando a caixa com os três elementos dentro
            st.markdown(
                f"""
                <div class="box-container">
                    <div class="box-item"><h3>Política de manutenção</h3></div>
                    <div class="box-item"><h3>W = {W}</h3></div>
                    <div class="box-item"><h3>M = {M}</h3></div>
                </div>
                """,
                unsafe_allow_html=True,
            )

            # CSS para aplicar a borda em cada coluna
            st.markdown(
                """
                <style>
                    .column-box {
                        border: 2px solid black; /* Cor da borda */
                        border-radius: 10px; /* Bordas arredondadas */
                        padding: 15px; /* Espaço interno */
                        text-align: center;
                        width: 100%; /* Garante que ocupa toda a largura da coluna */
                        margin-bottom: 10px; /* Espaço entre as colunas */
                    }
                </style>
                """,
                unsafe_allow_html=True,
            )
            
            # Criando colunas
            col1, col2, col3 = st.columns(3)
            
            # Criando os blocos com borda em cada coluna
            with col1:
                st.markdown(
                    f"""
                    <div class="column-box">
                        <h3>Taxa de custo</h3>
                        <h2>{round(taxadecusto[0], 3)}</h2>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
            
            with col2:
                st.markdown(
                    f"""
                    <div class="column-box">
                        <h3>Taxa de indisponibilidade</h3>
                        <h2>{round(taxadecusto[4], 3)}</h2>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
            
            with col3:
                st.markdown(
                    f"""
                    <div class="column-box">
                        <h3>Confiabilidade operacional</h3>
                        <h2>{round(taxadecusto[5], 2)}</h2>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

            #col1, col2, col3 = st.columns(3)
            #col1.subheader("Política de manutenção")
            #col2.subheader("W={}" .format(W))
            #col3.subheader("M={}" .format(M))
            
            #col1, col2, col3 = st.columns(3)
            #col1.subheader("Taxa de custo")
            #col2.subheader("Taxa de indisponibilidade")
            #col3.subheader("Confiabilidade operacional")

            #col1.subheader(taxadecusto[0])
            #col2.subheader(taxadecusto[4])
            #col3.subheader(taxadecusto[5])

    if choice == menu[1]:
        
        st.header(menu[1])

        st.subheader("Insira os valores dos parâmetros de entrada abaixo:")

        n2 = st.number_input("Insira o parâmetro de escala - {}".format(chr(945)), min_value = 0.0, value = 10.0) # escala fraca
        b2 = st.number_input("Insira o parâmetro de forma - {}".format(chr(946)), min_value = 1.0, max_value = 6.0, value = 3.0) # forma fraca

        cp = st.number_input("Insira o parâmetro de custo de substituição preventiva", min_value = 0.0, value = 1.0) # custo da preventiva
        cd = st.number_input("Insira o parâmetro de custo de tempo de inatividade por unidade de tempo", min_value = 0.0, value = 0.5) # custo de downtime por unidade de tempo
        cm = st.number_input("Insira o parâmetro de custo adicional para ação de manutenção garantida", min_value = 0.0, value = 1.00) # custo de inspeção
        cf = st.number_input("Insira o parâmetro de custo de substituição corretiva", min_value = 0.0, value = 1.0) # custo de falha
        
        q = st.number_input("Insira o parâmetro de probabilidade de oportunidade em uma visita", min_value = 0.0, max_value = 1.0, value = 0.2) # taxa de chegada de oportunidades
        s = st.number_input("Insira o parâmetro de intervalo de tempo entre visitas", min_value = 0.0, value = 1.0) # intervalo entre visitas

        #opcoes = ["Taxa de custo", "Taxa de indisponibilidade", "Confiabilidade operacional"]
        
        #escolha = st.selectbox("Defina a medida de desempenho você deseja otimizar", opcoes)
        
        st.subheader("Clique no botão abaixo para executar esta aplicação:")
        botao = st.button("Executar")

        if botao:
        
            def otm():
              def fx(x):#weibull density func (for strong comp) for arrival of fails
                  return (b2/n2)*((x/n2)**(b2-1))*np.exp(-(x/n2)**b2)
            
              #%% VARIÁVEIS E CONSTANTES
            
              # #Parâmetros
              # b2 = b2[i] #Forma do componente forte
              # n2 = n2[i] #Escala do componente forte     
              # cd = cd[i] #custo de downtime
              # cm = cm[i] #custo logístico
              # cp = cp[i] #custo de manutenção preventiva
              # cf = cf[i] #custo de manutenção corretiva
                 
              # s = s[i] #Intervalo entre visitas
              # q = q[i] #Oportunidade de manutenção
            
              #%%Variáveis auxiliares
              p11 = p12 = p2 = p31 = p32 = p4 = 0
              c11 = c12 = c2 = c31 = c32 = c4 = 0
              t11 = t12 = t2 = t31 = t32 = t4 = 0
              d11 = d12 = d2 = d31 = d32 = d4 = 0 
            
              #%%CASO 1
              
              def fprob1(x):
                  return ((fx(x)))
              def fcusto1(x):
                  return ((cf+cd*(j*s-x))*(fx(x)))
              def fvida1(x):
                  return ((j*s)*(fx(x)))
              def fdown1(x):
                  return ((j*s-x)*(fx(x)))
            
              for i in range(1, W):
                  for j in range(i, M):
                      fp11 = quad(fprob1, (i-1)*s, i*s)[0]*((1-q)**(j-i))*(q)
                      p11 = p11 + fp11
                      
                      fc11 = quad(fcusto1, (i-1)*s, i*s)[0]*((1-q)**(j-i))*(q)
                      c11 = c11 + fc11
                      
                      ft11 = quad(fvida1, (i-1)*s, i*s)[0]*((1-q)**(j-i))*(q)
                      t11 = t11 + ft11
                      
                      fd11 = quad(fdown1, (i-1)*s, i*s)[0]*((1-q)**(j-i))*(q)
                      d11 = d11 + fd11
                      
              for i in range(W, M):
                  for j in range(i, M):
                      fp12 = quad(fprob1, (i-1)*s, i*s)[0]*((1-q)**(j-W))*(q)
                      p12 = p12 + fp12
                      
                      fc12 = quad(fcusto1, (i-1)*s, i*s)[0]*((1-q)**(j-W))*(q)
                      c12 = c12 + fc12
                      
                      ft12 = quad(fvida1, (i-1)*s, i*s)[0]*((1-q)**(j-W))*(q)
                      t12 = t12 + ft12
                      
                      fd12 = quad(fdown1, (i-1)*s, i*s)[0]*((1-q)**(j-W))*(q)
                      d12 = d12 + fd12
                
              #%%CASO 2
            
              def fprob2(x):
                  return ((fx(x)))
              def fcusto2(x):
                  return ((cp)*(fx(x)))
              def fvida2(x):
                  return ((j*s)*(fx(x)))
              def fdown2(x):
                  return ((0)*(fx(x)))
            
              for j in range(W, M):
                fp2 = quad(fprob2, j*s, np.inf)[0]*(((1-q)**(j-W))*q)
                p2 = p2 + fp2
                  
                fc2 = quad(fcusto2, j*s, np.inf)[0]*(((1-q)**(j-W))*q)
                c2 = c2 + fc2
                
                ft2 = quad(fvida2, j*s, np.inf)[0]*(((1-q)**(j-W))*q)
                t2 = t2 + ft2
                
                fd2 = quad(fdown2, j*s, np.inf)[0]*(((1-q)**(j-W))*q)
                d2 = d2 + fd2 
                
              #%%CASO 3
              
              def fprob3(x):
                  return ((fx(x)))
              def fcusto3(x):
                  return ((cf+cd*(M*s-x)+cm)*(fx(x)))
              def fvida3(x):
                  return ((M*s)*(fx(x)))
              def fdown3(x):
                  return ((M*s-x)*(fx(x)))
            
              for i in range(1, W):
                  fp31 = quad(fprob3, (i-1)*s, i*s)[0]*((1-q)**(M-i))
                  p31 = p31 + fp31
                  
                  fc31 = quad(fcusto3, (i-1)*s, i*s)[0]*((1-q)**(M-i))
                  c31 = c31 + fc31
                  
                  ft31 = quad(fvida3, (i-1)*s, i*s)[0]*((1-q)**(M-i))
                  t31 = t31 + ft31
                  
                  fd31 = quad(fdown3, (i-1)*s, i*s)[0]*((1-q)**(M-i))
                  d31 = d31 + fd31   
                  
            
              for i in range(W, M+1):
                  fp32 = quad(fprob3, (i-1)*s, i*s)[0]*((1-q)**(M-W))
                  p32 = p32 + fp32
                  
                  fc32 = quad(fcusto3, (i-1)*s, i*s)[0]*((1-q)**(M-W))
                  c32 = c32 + fc32
                  
                  ft32 = quad(fvida3, (i-1)*s, i*s)[0]*((1-q)**(M-W))
                  t32= t32 + ft32
                  
                  fd32 = quad(fdown3, (i-1)*s, i*s)[0]*((1-q)**(M-W))
                  d32 = d32 + fd32  
            
              #%%CASO 4
              
              def fprob4(x):
                  return ((fx(x)))
              def fcusto4(x):
                  return ((cp+cm)*(fx(x)))
              def fvida4(x):
                  return ((M*s)*(fx(x)))
              def fdown4(x):
                  return ((0)*(fx(x)))
              
              fp4 = quad(fprob4, M*s, np.inf)[0]*(((1-q)**(M-W)))
              p4 = p4 + fp4
              
              fc4 = quad(fcusto4, M*s, np.inf)[0]*(((1-q)**(M-W)))
              c4 = c4 + fc4
              
              ft4 = quad(fvida4, M*s, np.inf)[0]*(((1-q)**(M-W)))
              t4= t4 + ft4
              
              fd4 = quad(fdown4, M*s, np.inf)[0]*(((1-q)**(M-W)))
              d4 = d4 + fd4  
            
              #%%RESULTADOS

                
              pe = p11+p12+p2+p31+p32+p4 #probabilidade total
              ce = c11+c12+c2+c31+c32+c4 #custo total
              ve = t11+t12+t2+t31+t32+t4 #tamanho total
              de = d11+d12+d2+d31+d32+d4 #downtime total
              
              tx = ce/ve #taxa de custo
              dx = de/ve #taxa de downtime
              mu = ve/(p11+p12+p31+p32) #mtbof
            
              if pe < 0.999 or pe > 1.001:
                print("Algum erro ocorreu!")
            
              return tx, ce, ve, pe, dx, mu

            # Inicializa session_state se for a primeira execução
            progress_bar = st.progress(0)
            status_text = st.empty()
            total_iteracoes = 1275
            contador = 0
    
            lista_W = []
            lista_M = []
            lista_taxa = []
            lista_indisp = []
            lista_confiab = []
            
            menortaxa = 10000000000
            for W in range(1, 50+1):
                for M in range(W, 50+1):
                    resultado = otm()
                    contador += 1
                    progresso_atual = int((contador / total_iteracoes) * 100)
                    progress_bar.progress(progresso_atual)
                    status_text.text(f"Processando: W={W}, M={M} ({progresso_atual}%)")
                    if resultado[0] < menortaxa:
                        Wotm = W
                        Motm = M
                        menortaxa = resultado[0]
                        menorinatividade = resultado[4]
                        maiorconfiabilidade = resultado[5]
                    lista_W.append(W)
                    lista_M.append(M)
                    lista_taxa.append(resultado[0])
                    lista_indisp.append(resultado[4])
                    lista_confiab.append(resultado[5])
            def criar_grafico_3D(x, y, z, eixo_z):
                fig = plt.figure(figsize=(8, 6), dpi=150)  # Maior resolução
                ax = fig.add_subplot(111, projection='3d')
            
                # Criando o scatter plot com melhor visual
                sc = ax.scatter(x, y, z, c=z, cmap='viridis', s=20, alpha=0.85)
    
                # Aplicando os limites no gráfico
                ax.set_xlim(w_min, w_max)
                ax.set_ylim(m_min, m_max)
            
                # Labels dos eixos
                ax.set_xlabel("W", fontsize=12, labelpad=10)
                ax.set_ylabel("M", fontsize=12, labelpad=10)
                #ax.set_zlabel(eixo_z, fontsize=12, labelpad=10, rotation=90)  # Rota z para lateral
            
                # Melhorando visual do gráfico
                ax.xaxis.pane.fill = False  # Remove fundo cinza
                ax.yaxis.pane.fill = False
                ax.zaxis.pane.fill = False
                ax.grid(False)  # Remove grid para visual limpo
            
                # Barra de cores para indicar valores de Z
                cbar = plt.colorbar(sc, shrink=0.6, aspect=10)
                cbar.set_label(eixo_z, fontsize=12)
            
                return fig
            
            status_text.text("Execução concluída")
            st.markdown(
                """
                <style>
                    .box-container {
                        border: 2px solid black; /* Cor da borda */
                        border-radius: 10px; /* Bordas arredondadas */
                        padding: 15px; /* Espaço interno */
                        text-align: center;
                        width: 100%; /* Garante que a largura ocupe toda a linha */
                        display: flex; /* Para organizar os elementos lado a lado */
                        justify-content: space-around; /* Espaço uniforme entre elementos */
                        align-items: center; /* Alinha verticalmente */
                    }
                    .box-item {
                        flex: 1; /* Permite que os itens fiquem do mesmo tamanho */
                    }
                </style>
                """,
                unsafe_allow_html=True,
            )
            
            # Criando a caixa com os três elementos dentro
            st.markdown(
                f"""
                <div class="box-container">
                    <div class="box-item"><h3>Política de manutenção</h3></div>
                    <div class="box-item"><h3>W = {Wotm}</h3></div>
                    <div class="box-item"><h3>M = {Motm}</h3></div>
                </div>
                """,
                unsafe_allow_html=True,
            )
    
            # CSS para aplicar a borda em cada coluna
            st.markdown(
                """
                <style>
                    .column-box {
                        border: 2px solid black; /* Cor da borda */
                        border-radius: 10px; /* Bordas arredondadas */
                        padding: 15px; /* Espaço interno */
                        text-align: center;
                        width: 100%; /* Garante que ocupa toda a largura da coluna */
                        margin-bottom: 10px; /* Espaço entre as colunas */
                    }
                </style>
                """,
                unsafe_allow_html=True,
            )
            
            # Criando colunas
            col1, col2, col3 = st.columns(3)
    
            delta = 5  # ajuste conforme necessário
    
            # Para W
            w_min = max(Wotm - delta, 1)
            w_max = min(Wotm + delta, 50)
            
            # Para M
            m_min = max(Motm - delta, 1)
            m_max = min(Motm + delta, 50)
                
            # Filtrar os dados que estejam dentro do intervalo desejado
            indices_zoom = [i for i, (w, m) in enumerate(zip(lista_W, lista_M)) 
                            if abs(w - Wotm) <= delta and abs(m - Motm) <= delta]
            
            lista_W_zoom = [lista_W[i] for i in indices_zoom]
            lista_M_zoom = [lista_M[i] for i in indices_zoom]
            lista_taxa_zoom = [lista_taxa[i] for i in indices_zoom]
            lista_indisp_zoom = [lista_indisp[i] for i in indices_zoom]
            lista_confiab_zoom = [lista_confiab[i] for i in indices_zoom]
    
            # Criando os blocos com borda em cada coluna
            with col1:
                st.markdown(
                    f"""
                    <div class="column-box">
                        <h3>Taxa de custo</h3>
                        <h2>{round(menortaxa, 3)}</h2>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
                fig_taxa = criar_grafico_3D(lista_W_zoom, lista_M_zoom, lista_taxa_zoom, "Taxa de Custo")
                st.pyplot(fig_taxa)
            
            with col2:
                st.markdown(
                    f"""
                    <div class="column-box">
                        <h3>Taxa de indisponibilidade</h3>
                        <h2>{round(menorinatividade, 3)}</h2>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
                fig_indisp = criar_grafico_3D(lista_W_zoom, lista_M_zoom, lista_indisp_zoom, "Taxa de Indisponibilidade")
                st.pyplot(fig_indisp)
            
            with col3:
                st.markdown(
                    f"""
                    <div class="column-box">
                        <h3>Confiabilidade operacional</h3>
                        <h2>{round(maiorconfiabilidade, 2)}</h2>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
                fig_confiab = criar_grafico_3D(lista_W_zoom, lista_M_zoom, lista_confiab_zoom, "Confiabilidade Operacional")
                st.pyplot(fig_confiab)
    
            st.markdown(
                """
                <style>
                    .justificado {
                        text-align: justify;
                    }
                </style>
                """,
                unsafe_allow_html=True,
            )
    
            texto = '''Este protótipo possui restrições quanto ao espaço de busca de soluções, com W,M ∈ {1,...,50}. Se for do interesse do usuário utilizar uma gama maior de combinações de soluções ou se houver alguma dúvida sobre o estudo e/ou este protótipo, elas podem ser direcionadas para qualquer um dos endereços de e-mail abaixo. Por fim, se esta aplicação for utilizada para qualquer propósito, todos os autores devem ser informados.'''
            st.markdown(f'<p class="justificado">{texto}</p>', unsafe_allow_html=True)
            st.write('''y.r.melo@random.org.br''')
            st.write('''c.a.v.cavalcante@random.org.br''')
    
    if choice == menu[2]:
        
        st.header(menu[2])
        st.markdown(
                """
                <style>
                    .justificado {
                        text-align: justify;
                    }
                </style>
                """,
                unsafe_allow_html=True,
            )
        
        texto1 = '''Este produto tecnológico, desenvolvido por (Melo; Cavalcante, 2025), para apoiar a tomada de decisão na manutenção de sistemas críticos, como geração e distribuição de energia, abastecimento de água e saneamento básico, especialmente aqueles localizados em áreas remotas ou com infraestrutura complexa. Esses sistemas apresentam desafios operacionais significativos, exigindo estratégias de manutenção que conciliem eficácia e viabilidade operacional. A solução proposta baseia-se em um modelo de manutenção estruturado em períodos discretos, permitindo a integração de ações preventivas e corretivas conforme a disponibilidade de recursos, caracterizada pela ocorrência de oportunidades. Essa abordagem periódica facilita o planejamento estratégico das intervenções e aproxima a teoria da  prática. A otimização da política de manutenção é realizada por algoritmos numéricos que avaliam diferentes cenários de renovação do sistema. O modelo considera fatores como os custos das manutenções corretiva e preventiva, o custo da inatividade após uma falha, o custo para garantir uma ação de manutenção, a probabilidade de surgimento de oportunidades em visitas, os intervalos entre os períodos de visita e a vida útil do sistema. A aplicação numérica do modelo, aliada a uma análise de sensibilidade, demonstra que a abordagem proposta é particularmente eficaz em cenários com recursos limitados e poucas oportunidades de manutenção. Além disso, a comparação com políticas de manutenção contínuas evidencia que a discretização dos períodos não compromete significativamente o desempenho do sistema, reforçando a viabilidade prática da metodologia. Esse produto tecnológico oferece uma ferramenta de suporte à decisão acessível e intuitiva, permitindo simulações e otimizações da política de manutenção para diferentes contextos, auxiliando gestores e engenheiros na definição de estratégias eficientes para a gestão de ativos críticos.'''
        texto2 = '''Este produto tecnológico possui restrições quanto ao espaço de busca de soluções, com W,M ∈ {1,...,50}. Se for do interesse do usuário utilizar uma gama maior de combinações de soluções ou se houver alguma dúvida sobre o estudo e/ou este protótipo, elas podem ser direcionadas para qualquer um dos endereços de e-mail abaixo. Por fim, se esta aplicação for utilizada para qualquer propósito, todos os autores devem ser informados.'''
        texto3 = '''Uma representação gráfica da política de manutenção proposta é apresentada abaixo, sendo s o intervalo de tempo entre visitas, W o limite inferior da janela de ações de manutenção preventiva por oportunidade, e M a idade de ação de manutenção garantida'''
        st.markdown(f'<p class="justificado">{texto1}</p>', unsafe_allow_html=True)
        st.markdown(f'<p class="justificado">{texto3}</p>', unsafe_allow_html=True)

        #criando 3 colunas
        col1, col2, col3 = st.columns(3)
        foto3 = Image.open('foto3.png')
        #inserindo na coluna 2
        col2.image(foto3, use_column_width=True)
        
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
            """,
            unsafe_allow_html=True
        )

    if choice == menu[3]:
        
        st.header(menu[3])
        
        st.write("O Grupo de Pesquisa em Risco e Análise da Decisão em Operações e Manutenção foi criado em 2012 com o objetivo de reunir diferentes pesquisadores que atuam nas seguintes áreas: risco, modelagem de manutenção e operação. Saiba mais sobre o grupo através do nosso site.")
        st.markdown('[Clique aqui para ser redirecionado ao nosso site](https://sites.ufpe.br/random/#page-top)', False)

if __name__ == "__main__": main()
