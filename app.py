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

# Controle de meses de previsão na barra lateral
st.sidebar.header("Configuração de Previsão")
num_meses_previsao = st.sidebar.slider("Quantos meses para prever?", min_value=1, max_value=12, value=3)

# Categoria inteligente
def extrair_categoria(desc):
    desc = str(desc).lower()
    categorias = {
        'supermercado': ['supermercado', 'mercado', 'carrefour', 'extra', 'pao de acucar', 'atacadao', 'pao'],
        'restaurante': ['restaurante', 'bar', 'cafe', 'padaria', 'mcdonald', 'burger king', 'pizza'],
        'transporte': ['uber', '99', 'taxi', 'combustivel', 'posto', 'gasolina', 'metro', 'onibus', 'gas', 'passagem'],
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
    df['Data'] = pd.to_datetime(df['Data'], format='%d/%m/%Y', dayfirst=False, errors='coerce')
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
    # Agrupa despesas por mês antes de alimentar o Prophet
    df_despesa = df[df['Valor'] < 0].copy()
    df_despesa['Data'] = pd.to_datetime(df_despesa['Data'])
    df_despesa['AnoMes'] = df_despesa['Data'].dt.to_period('M').dt.to_timestamp()
    df_despesa_mensal = df_despesa.groupby('AnoMes')['Valor'].sum().reset_index()
    df_despesa_mensal['Valor'] = -1 * df_despesa_mensal['Valor']  # Torna positivo para previsão
    df_despesa_mensal.rename(columns={'AnoMes': 'ds', 'Valor': 'y'}, inplace=True)
    if len(df_despesa_mensal) > 2:
        m = Prophet()
        m.fit(df_despesa_mensal)
        future = m.make_future_dataframe(periods=num_meses_previsao, freq='M')
        forecast = m.predict(future)
        fig_prophet = go.Figure()
        fig_prophet.add_trace(go.Scatter(x=df_despesa_mensal['ds'], y=df_despesa_mensal['y'], name='Despesa Real', mode='lines+markers'))
        fig_prophet.add_trace(go.Scatter(x=forecast['ds'], y=forecast['yhat'], name='Previsão', mode='lines'))
        fig_prophet.update_layout(title='Previsão de Despesas (Prophet)', xaxis_title='Data', yaxis_title='Despesa (R$)', hovermode='x unified')
        st.plotly_chart(fig_prophet, use_container_width=True)
        # Somatório do total previsto para os meses futuros agrupado por mês
        previsao_futura = forecast[['ds', 'yhat']].tail(num_meses_previsao)
        previsao_futura['Mes'] = previsao_futura['ds'].dt.strftime('%Y-%m')
        previsao_por_mes = previsao_futura.groupby('Mes')['yhat'].sum().reset_index()
        previsao_por_mes.rename(columns={'yhat': 'Despesa Prevista'}, inplace=True)
        st.markdown(f"**Previsão de gastos agrupada por mês:**")
        st.dataframe(previsao_por_mes)
        total_previsto = previsao_futura['yhat'].sum()
        st.markdown(f"**Total previsto para os próximos {num_meses_previsao} meses:** R$ {total_previsto:,.2f}")
        st.write('Próximos meses previstos:')
        st.dataframe(previsao_futura.rename(columns={'ds': 'Data', 'yhat': 'Despesa Prevista'}))
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