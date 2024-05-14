import streamlit as st
import pandas as pd
import plotly.express as px
from prophet import Prophet
from sklearn.metrics import mean_squared_error, mean_absolute_error

#Importacao e tratamento da base
dados = pd.read_csv('https://raw.githubusercontent.com/Toniferson/tech_challange_4/main/dados/ipeadata%5B19-04-2024-09-47%5D.csv',sep = ';', index_col="Data", parse_dates=[0])
dados_raw = dados
dados = dados.rename(columns={"Preço - petróleo bruto - Brent (FOB) - US$ - Energy Information Administration (EIA) - EIA366_PBRENT366":"preco", "Data":"data"})
dados['preco'] = dados['preco'].str.replace(',','.').astype(float)
dados = dados.drop("Unnamed: 2", axis=1)
dados.index.name = 'data'
dados.index.name = "ds"
dados = dados.rename(columns={"preco":"y"})
dados.index = pd.to_datetime(dados.index)

# Specify the date after which you want to select rows
certain_date = pd.to_datetime('2009-01-01')  # Change the date as needed

# Use boolean indexing to select rows after the certain date
dados_novo = dados[dados.index >= certain_date]
dados_novo.reset_index(inplace=True)

# Creating and fitting the Prophet model
model = Prophet()
model.fit(dados_novo)

# Making future predictions
future = model.make_future_dataframe(periods=30)  # Forecasting 30 days into the future
forecast = model.predict(future)

# Assuming you have the forecasted values and the actual values from your Prophet model
forecasted_values = forecast['yhat']
actual_values = dados_novo['y']

# Calculate Mean Squared Error (MSE)
mse = ((actual_values - forecasted_values) ** 2).mean()

# Calculate Mean Absolute Error (MAE)
mae = abs(actual_values - forecasted_values).mean()

# Calculate Mean Percentage Error (MPE)
mpe = ((actual_values - forecasted_values) / actual_values).mean() * 100

#Aplicacao
st.image('dados/petrobras.jpg_2090796243-scaled.jpg')
st.markdown("<h1 style='text-align: center; color: orange;'>Machine Learning</h1>", unsafe_allow_html=True)
st.write('Analisando a nossa base, passamos por alguns processos de trabalho, dentre eles o processamento dos dados e análise das informações obtidas a partir destes dados com isso, notamos a possibilidade de trabalhar com uma Time Series e podendo aplicar um modelo de Machine Learning para uma análise preditiva.')
st.divider()

#criar tabs para processamento, analise exploratória e modelo de ml de fato

st.markdown("<h2 style='text-align: left; color: lightblue;'>Pré Processamento dos Dados<h2>", unsafe_allow_html=True)

tab1,tab2,tab3 = st.tabs(['Pré Processamento', 'Modelo','Resultado'])
with tab1:
    col1,col2 = st.columns(2)
    with col1:
        st.write('A primeiro momento quando baixamos e carregamos a base podemos notar a presença de apenas duas colunas, uma de data e outra com o valor do Preço do Petroléo Brent em Dólar (US$), o que já nos indica que podemos tratar esta tabela como uma série temporal.')
        st.write('Podemos perceber que o nome da coluna de valor não esta nada amigável para ser trabalhada em um código, assim como também os valores estão sendo separados por vírgula ao invés de ponto, o que pode ocasionar futuros problemas em nosso modelos, por isso tivemos que tratar estes e outros pontos em nosso pré processamento dos dados, além de também  criar um filtro para pegar apenas um período específico a ser analisado')
    with col2:
        st.dataframe(dados_raw.head(10))

    st.write('Tratamento da base:')
    st.code('''
    dados = dados.rename(columns={"Preço - petróleo bruto - Brent (FOB) - US$ - Energy Information Administration (EIA) - EIA366_PBRENT366":"preco", "Data":"data"})
    dados['preco'] = dados['preco'].str.replace(',','.').astype(float)
    dados = dados.drop("Unnamed: 2", axis=1)
    dados.index.name = 'data'
    dados.index.name = "ds"
    dados = dados.rename(columns={"preco":"y"})
    
            
    certain_date = pd.to_datetime('2009-01-01')  # Change the date as needed                ''')
with tab2:
    st.markdown("<h2 style='text-align: left; color: lightblue;'>Modelo de Machine Learning<h2>", unsafe_allow_html=True)
    st.write('Já com a base tratada, chegou o momento de começar a criar o modelo usando a biblioteca do Prophet')

    st.write('Instanciando e treinando o modelo:')
    st.code('''model = Prophet()
model.fit(dados)''')
    st.write('Criando as predições')
    st.code('''future = model.make_future_dataframe(periods=30)  
forecast = model.predict(future)''')
    
    st.write('Descrever esse passo')
    st.code('''forecasted_values = forecast['yhat']
    actual_values = dados_novo['y']''')

    st.write('Calculando Erros')
    st.code('''# Calculate Mean Squared Error (MSE)
mse = ((actual_values - forecasted_values) ** 2).mean()

# Calculate Mean Absolute Error (MAE)
mae = abs(actual_values - forecasted_values).mean()

# Calculate Mean Percentage Error (MPE)
mpe = ((actual_values - forecasted_values) / actual_values).mean() * 100
''')

with tab3:
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric('Mean Squared Error (MSE)',mse.round(2))
    with col2:
        st.metric('Mean Absolute Error (MAE)',mae.round(2))
    with col3:
        st.metric('Mean Percentage Error (MPE)',mpe.round(2))
    st.image('dados/brent_model.png')
    st.write('''Após a limpeza inicial, geramos o gráfico acima com Prophet. Alguns pontos se evidenciaram a partir disso:
pudemos notar alguns pontos interessantes de picos(outliers altos) e vales(outliers baixos) no preço do petróleo Brent em alguns anos históricos;
o modelo acertou bastante nas suas previsões(linhas azuis), errando somente nesses momentos fora do comum;
como consequência dessas variações bruscas em certos momentos nos últimos 35 anos, as métricas de acertos e erros do modelo foram muito discrepantes, mostrando que algo deveria ser modificado para a melhora da previsão.
A partir disso, fizemos duas modificações: 
restringimos o período para somente os últimos 15 anos(ou seja, desde 01 de janeiro de 2009); e
restringimos o período para os últimos 15 anos até o início da pandemia.
O objetivo dessas modificações era reduzir bruscamente o valor dos erros calculados. E de fato, isso aconteceu! O nosso melhor resultado foi alcançado com os últimos 15 anos e, como excluindo a pandemia a variação no erro foi ínfima, decidimos não excluí-la da análise no final das contas. Os erros finais foram:
Mean Squared Error(MSE): 105.4
Mean Absolute Error(MAE): 7.5
Mean Percentage Error(MPE): -5.2
''')