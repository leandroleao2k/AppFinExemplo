#Este módulo implementa um dashboard interativo de análise financeira utilizando Streamlit, pandas, Plotly e Prophet.
#Permite ao usuário fazer upload de um extrato bancário em CSV, categoriza automaticamente as transações, exibe resumos e gráficos interativos, e realiza previsão de despesas futuras.
# >python -m streamlit run app.py  

import streamlit as st
import pandas as pd
import plotly.graph_objs as go
from prophet import Prophet
import numpy as np

st.set_page_config(page_title="Análise Financeira Interativa", layout="wide")
st.title("Dashboard de Análise Financeira")

st.sidebar.header("Upload do Extrato Bancário")
uploaded_file = st.sidebar.file_uploader("Escolha o arquivo CSV do extrato", type=["csv"])

# Categoria inteligente
def extrair_categoria(desc):
    desc = str(desc).lower()
    categorias = {
        'supermercado': ['supermercado', 'mercado', 'carrefour', 'extra', 'pao de acucar', 'atacadao', 'pao'],
        'restaurante': ['restaurante', 'bar', 'cafe', 'padaria', 'mcdonald', 'burger king', 'pizza'],
        'transporte': ['uber', '99', 'taxi', 'combustivel', 'posto', 'gasolina', 'metro', 'onibus'],
        'saude': ['farmacia', 'drogaria', 'medico', 'hospital', 'clinica', 'dentista'],
        'lazer': ['cinema', 'show', 'teatro', 'parque', 'viagem', 'hotel'],
        'educacao': ['escola', 'faculdade', 'curso', 'livro', 'material escolar'],
        'moradia': ['aluguel', 'condominio', 'energia', 'luz', 'agua', 'internet', 'telefone', 'TIT'],
        'servicos': ['seguro', 'cartao', 'banco', 'tarifa', 'taxa', 'conta', 'black'],
        'telefonia': ['internet', 'claro', 'vivo', 'net'],
        'outros': []
    }
    for cat, palavras in categorias.items():
        if any(p in desc for p in palavras):
            return cat.capitalize()
    return 'Outros'

if uploaded_file:
    df = pd.read_csv(uploaded_file, sep=';', header=0)
    df.rename(columns={'DATA': 'Data', 'VALOR': 'Valor', 'DESC': 'Desc'}, inplace=True)
    df['Data'] = pd.to_datetime(df['Data'], format='%m/%d/%Y', dayfirst=False, errors='coerce')
    df['Valor'] = df['Valor'].astype(str).str.replace('.', '')
    df['Valor'] = df['Valor'].astype(str).str.replace(',', '.').str.replace(' ', '')
    df['Valor'] = pd.to_numeric(df['Valor'], errors='coerce')
    df['Tipo'] = np.where(df['Valor'] < 0, 'Despesa', 'Receita')
   
    df['Categoria'] = df['Desc'].apply(extrair_categoria)
    df['AnoMes'] = df['Data'].dt.to_period('M')
    df['Ano'] = df['Data'].dt.year

    st.subheader("Resumo do Extrato")
    st.dataframe(df.tail(5))

    # Gráfico de saldo mensal
    mensal = df.groupby('AnoMes')['Valor'].sum()
    receita = df[df['Valor'] > 0].groupby('AnoMes')['Valor'].sum()
    despesa = df[df['Valor'] < 0].groupby('AnoMes')['Valor'].sum()
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=mensal.index.astype(str), y=mensal.values, name='Saldo Mensal', mode='lines+markers'))
    fig.add_trace(go.Bar(x=receita.index.astype(str), y=receita.values, name='Receita'))
    fig.add_trace(go.Bar(x=despesa.index.astype(str), y=despesa.values, name='Despesa'))
    fig.update_layout(title='Saldo, Receita e Despesa Mensal', xaxis_title='Ano/Mês', yaxis_title='Valor (R$)', barmode='group', legend_title='Tipo', hovermode='x unified')
    st.plotly_chart(fig, use_container_width=True)

    # Gráfico por tipo ano a ano
    soma_tipo_ano = df.groupby(['Ano', 'Tipo'])['Valor'].sum().unstack(fill_value=0)
    fig_tipo_ano = go.Figure()
    fig_tipo_ano.add_trace(go.Bar(x=soma_tipo_ano.index.astype(str), y=soma_tipo_ano['Despesa'], name='Despesa'))
    fig_tipo_ano.add_trace(go.Bar(x=soma_tipo_ano.index.astype(str), y=soma_tipo_ano['Receita'], name='Receita'))
    fig_tipo_ano.update_layout(title='Total de Gastos por Tipo (Ano a Ano)', xaxis_title='Ano', yaxis_title='Valor (R$)', barmode='group', legend_title='Tipo', hovermode='x unified')
    st.plotly_chart(fig_tipo_ano, use_container_width=True)

    # Gráfico por categoria ano a ano
    gastos_categoria_ano = df[df['Valor'] < 0].groupby(['Ano', 'Categoria'])['Valor'].sum().unstack(fill_value=0)
    fig_categoria_ano = go.Figure()
    for categoria in gastos_categoria_ano.columns:
        fig_categoria_ano.add_trace(go.Bar(x=gastos_categoria_ano.index.astype(str), y=gastos_categoria_ano[categoria], name=categoria))
    fig_categoria_ano.update_layout(title='Total de Gastos por Categoria (Ano a Ano)', xaxis_title='Ano', yaxis_title='Valor (R$)', barmode='stack', legend_title='Categoria', hovermode='x unified')
    st.plotly_chart(fig_categoria_ano, use_container_width=True)

    # Previsão de despesas com Prophet
    st.subheader('Previsão de Despesas para os Próximos Meses (Prophet)')
    df_despesa = df[df['Valor'] < 0].groupby('Data')['Valor'].sum().reset_index()
    df_despesa['Valor'] = -1*df_despesa['Valor']
    df_despesa.rename(columns={'Data': 'ds', 'Valor': 'y'}, inplace=True)
    if len(df_despesa) > 2:
        m = Prophet()
        m.fit(df_despesa)
        future = m.make_future_dataframe(periods=6, freq='M')
        forecast = m.predict(future)
        fig_prophet = go.Figure()
        fig_prophet.add_trace(go.Scatter(x=df_despesa['ds'], y=df_despesa['y'], name='Despesa Real', mode='lines+markers'))
        fig_prophet.add_trace(go.Scatter(x=forecast['ds'], y=forecast['yhat'], name='Previsão', mode='lines'))
        fig_prophet.update_layout(title='Previsão de Despesas (Prophet)', xaxis_title='Data', yaxis_title='Despesa (R$)', hovermode='x unified')
        st.plotly_chart(fig_prophet, use_container_width=True)
        st.write('Próximos meses previstos:')
        st.dataframe(forecast[['ds', 'yhat']].tail(6).rename(columns={'ds': 'Data', 'yhat': 'Despesa Prevista'}))
    else:
        st.warning('Não há dados suficientes para previsão com Prophet.')

    # Resumo financeiro
    total_receita = df[df['Tipo'] == 'Receita']['Valor'].sum()
    total_despesa = df[df['Tipo'] == 'Despesa']['Valor'].sum()
    saldo_final = total_receita + total_despesa
    st.markdown(f"""
    ### Resumo Financeiro Total
    - **Total de Receitas:** R$ {total_receita:,.2f}
    - **Total de Gastos:**   R$ {total_despesa:,.2f}
    - **Saldo Final:**       R$ {saldo_final:,.2f}
    """)
else:
    st.info('Faça upload de um arquivo CSV para iniciar a análise financeira.')

# Fim do código