import streamlit as st
import pandas as pd
import plotly.express as px
st.set_page_config(layout='wide')

#Importacao e tratamento da base
dados = pd.read_csv('/repository/tech-challange-4/dados/ipeadata[19-04-2024-09-47].csv',sep = ';')
dados['Data'] = pd.to_datetime(dados['Data'],format='%d/%m/%Y')
dados = dados.set_index('Data')
dados = dados.rename(columns={'Preço - petróleo bruto - Brent (FOB) - US$ - Energy Information Administration (EIA) - EIA366_PBRENT366':'Preço - Petróleo Bruto'})
dados = dados.drop(columns='Unnamed: 2')
dados['Preço - Petróleo Bruto'] = dados['Preço - Petróleo Bruto'].str.replace(',','.').astype(float)
dados = dados.fillna(0)

#Tabelas
dados_anuais = dados.groupby(pd.Grouper(freq = 'YE'))['Preço - Petróleo Bruto'].mean()

#Graficos
fig_evolutivo_brent = px.line(dados_anuais)

#Aplicacao
st.image('/repository/tech-challange-4/dados/petrobras.jpg_2090796243-scaled.jpg')
st.markdown("<h1 style='text-align: center; color: orange;'>Tech Challange - Fase 4</h1>", unsafe_allow_html=True)
st.write('Para um time multidisciplinar qualquer desafio é uma oportunidade de unir forças alcançar novas oportunidades :rocket:')
st.divider()

tab1, tab2, tab3= st.tabs(['Introdução', 'Base de Dados', 'Workflow'])
with tab1:  
    st.markdown("<h2 style='text-align: left; color: lightblue;'>O Desafio<h2>", unsafe_allow_html=True)
    st.write('''Como entrega final para esta fase, foi proposto... melhorar texto
                ''')
    st.markdown("<h2 style='text-align: left; color: lightblue;'>O que é 'Brent'<h2>", unsafe_allow_html=True)
    st.write('''
             Antes de iniciarmos nossas analises é muito importante que se explique o que é o Brent, que será analisado neste trabalho. O Brent é uma classificação de pretóleo não refinado que acabou sendo estipulado como padrão internacional para a cotação do preço mundialmente.\\
             O petróleo Brent foi batizado assim porque era extraído de uma base da Shell chamada Brent. Atualmente, a palavra Brent designa todo o petróleo extraído no Mar do Norte e comercializado na Bolsa de Londres.
                ''')
with tab2:
    st.markdown("<h2 style='text-align: left; color: lightblue;'>Time Series<h2>", unsafe_allow_html=True)
    st.write('''Texto falando da base...
                ''')
    st.plotly_chart(fig_evolutivo_brent,use_container_width = True)
    st.dataframe(dados.head(15))
    st.markdown(f'A tabela possui :blue[{dados.shape[0]}] linhas e :blue[{dados.shape[1]}] colunas')