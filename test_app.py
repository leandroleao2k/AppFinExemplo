from typing import Literal
from app import extrair_categoria
import pandas as pd
from prophet import Prophet
import pytest

def test_previsao_despesa_proximo_mes():
    # Cria um DataFrame de despesas simuladas
    data = {
        'Data': pd.date_range(start='2023-01-01', periods=12, freq='M'),
        'Valor': [-100, -120, -110, -130, -125, -140, -150, -160, -155, -170, -165, -180]
    }
    df = pd.DataFrame(data)
    df_despesa = df.groupby('Data')['Valor'].sum().reset_index()
    df_despesa.rename(columns={'Data': 'ds', 'Valor': 'y'}, inplace=True)

    # Treina o modelo Prophet
    m = Prophet()
    m.fit(df_despesa)
    future = m.make_future_dataframe(periods=1, freq='M')
    forecast = m.predict(future)

    # O último valor previsto deve estar presente e ser um número
    proxima_previsao = forecast.iloc[-1]['yhat']
    assert isinstance(proxima_previsao, float)
    # A previsão deve ser negativa (pois são despesas)
    assert proxima_previsao < 0
    @pytest.mark.parametrize(
        "desc,expected",
        [
            ("Supermercado Carrefour", "Supermercado"),
            ("Pagamento Uber", "Transporte"),
            ("Consulta Medico", "Saude"),
            ("Cinema Center", "Lazer"),
            ("Mensalidade Escola", "Educacao"),
            ("Aluguel Apartamento", "Moradia"),
            ("Tarifa Banco", "Servicos"),
            ("Compra Aleatoria", "Outros"),
            ("Padaria do Bairro", "Restaurante"),
            ("Livraria Cultura", "Educacao"),
        ]
    )
    def test_extrair_categoria(desc: Literal['Supermercado Carrefour'] | Literal['Pagamento Uber'] | Literal['Consulta Medico'] | Literal['Cinema Center'] | Literal['Mensalidade Escola'] | Literal['Aluguel Apartamento'] | Literal['Tarifa Banco'] | Literal['Compra Aleatoria'] | Literal['Padaria do Bairro'] | Literal['Livraria Cultura'], expected: Literal['Supermercado'] | Literal['Transporte'] | Literal['Saude'] | Literal['Lazer'] | Literal['Educacao'] | Literal['Moradia'] | Literal['Servicos'] | Literal['Outros'] | Literal['Restaurante']):
        """Testa a categorização automática de descrições."""
        assert extrair_categoria(desc) == expected

    def test_extrair_categoria_case_insensitive():
        """Testa se a função é case-insensitive."""
        assert extrair_categoria("SUPERMERCADO carrefour") == "Supermercado"
        assert extrair_categoria("restaurante mcdonald") == "Restaurante"

    def test_extrair_categoria_empty():
        """Testa se retorna 'Outros' para string vazia."""
        assert extrair_categoria("") == "Outros"

    def test_previsao_despesa_proximo_mes():
        """Testa previsão de despesas com Prophet."""
        data = {
            'Data': pd.date_range(start='2023-01-01', periods=12, freq='M'),
            'Valor': [-100, -120, -110, -130, -125, -140, -150, -160, -155, -170, -165, -180]
        }
        df = pd.DataFrame(data)
        df_despesa = df.groupby('Data')['Valor'].sum().reset_index()
        df_despesa.rename(columns={'Data': 'ds', 'Valor': 'y'}, inplace=True)

        m = Prophet()
        m.fit(df_despesa)
        future = m.make_future_dataframe(periods=1, freq='M')
        forecast = m.predict(future)

        proxima_previsao = forecast.iloc[-1]['yhat']
        assert isinstance(proxima_previsao, float)
        assert proxima_previsao < 0

    def test_previsao_despesa_dados_insuficientes():
        """Testa Prophet com dados insuficientes."""
        data = {
            'Data': pd.date_range(start='2023-01-01', periods=2, freq='M'),
            'Valor': [-100, -120]
        }
        df = pd.DataFrame(data)
        df_despesa = df.groupby('Data')['Valor'].sum().reset_index()
        df_despesa.rename(columns={'Data': 'ds', 'Valor': 'y'}, inplace=True)

        with pytest.raises(Exception):
            m = Prophet()
            m.fit(df_despesa)
            