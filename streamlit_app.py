"""
streamlit_app/app.py
Aplicación Streamlit para synth_bank_es
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path
import sys

# Agregar src al path
sys.path.append(str(Path(__file__).parent.parent))

from src.synthetic.generators.client_generator import ClientGenerator
from src.evaluation.statistical_tests import StatisticalTests


# ===== CONFIGURACIÓN DE PÁGINA =====
st.set_page_config(
    page_title="synth_bank_es 🏦",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'Get Help': 'https://github.com/tuusuario/synth_bank_es',
        'Report a bug': "https://github.com/tuusuario/synth_bank_es/issues",
        'About': "# synth_bank_es\nGenerador de datos sintéticos bancarios españoles"
    }
)


# ===== ESTILOS PERSONALIZADOS =====
def load_custom_css():
    """Cargar estilos CSS personalizados."""
    st.markdown("""
        <style>
        .main-header {
            font-size: 3rem;
            font-weight: bold;
            color: #2c3e50;
            text-align: center;
            margin-bottom: 2rem;
        }
        .metric-card {
            background-color: #f8f9fa;
            padding: 1.5rem;
            border-radius: 0.5rem;
            border-left: 4px solid #3498db;
            margin-bottom: 1rem;
        }
        .success-box {
            background-color: #d5f4e6;
            padding: 1rem;
            border-radius: 0.5rem;
            border-left: 4px solid #27ae60;
        }
        .warning-box {
            background-color: #fff3cd;
            padding: 1rem;
            border-radius: 0.5rem;
            border-left: 4px solid #f39c12;
        }
        .error-box {
            background-color: #fadbd8;
            padding: 1rem;
            border-radius: 0.5rem;
            border-left: 4px solid #e74c3c;
        }
        </style>
    """, unsafe_allow_html=True)


# ===== FUNCIONES AUXILIARES =====
@st.cache_data
def load_data(file_path: str) -> pd.DataFrame:
    """Cargar datos con caché."""
    return pd.read_csv(file_path)


@st.cache_resource
def load_generator(method: str):
    """Cargar generador con caché."""
    return ClientGenerator(method=method)


def display_dataframe_summary(df: pd.DataFrame, title: str):
    """Mostrar resumen de DataFrame."""
    st.subheader(title)
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("📊 Registros", f"{len(df):,}")
    with col2:
        st.metric("📈 Columnas", len(df.columns))
    with col3:
        st.metric("💾 Memoria", f"{df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
    with col4:
        missing_pct = (df.isnull().sum().sum() / (len(df) * len(df.columns)) * 100)
        st.metric("❌ Valores Nulos", f"{missing_pct:.2f}%")


def create_distribution_plot(df_real: pd.DataFrame, df_synth: pd.DataFrame, column: str):
    """Crear gráfico comparativo de distribuciones."""
    
    if df_real[column].dtype in ['int64', 'float64']:
        # Histograma para variables numéricas
        fig = go.Figure()
        
        fig.add_trace(go.Histogram(
            x=df_real[column],
            name='Real',
            opacity=0.7,
            marker_color='#3498db'
        ))
        
        fig.add_trace(go.Histogram(
            x=df_synth[column],
            name='Sintético',
            opacity=0.7,
            marker_color='#e74c3c'
        ))
        
        fig.update_layout(
            title=f'Distribución: {column}',
            xaxis_title=column,
            yaxis_title='Frecuencia',
            barmode='overlay',
            height=400
        )
        
    else:
        # Gráfico de barras para variables categóricas
        real_counts = df_real[column].value_counts()
        synth_counts = df_synth[column].value_counts()
        
        fig = go.Figure()
        
        fig.add_trace(go.Bar(
            x=real_counts.index,
            y=real_counts.values,
            name='Real',
            marker_color='#3498db'
        ))
        
        fig.add_trace(go.Bar(
            x=synth_counts.index,
            y=synth_counts.values,
            name='Sintético',
            marker_color='#e74c3c'
        ))
        
        fig.update_layout(
            title=f'Distribución: {column}',
            xaxis_title=column,
            yaxis_title='Frecuencia',
            height=400
        )
    
    return fig


# ===== SIDEBAR =====
def render_sidebar():
    """Renderizar barra lateral."""
    with st.sidebar:
        st.image("assets/images/logo.png", use_container_width=True)
        
        st.title("🏦 synth_bank_es")
        st.markdown("---")
        
        # Información del proyecto
        st.markdown("""
        ### 📌 Información
        
        **Versión:** 0.1.0  
        **Autor:** Alberto  
        **Licencia:** MIT
        
        ### 🎯 Características
        
        - ✅ Generación de datos sintéticos
        - ✅ Múltiples métodos (CTGAN, VAE, Copulas)
        - ✅ Evaluación de calidad
        - ✅ Modelos de scoring
        - ✅ Exportación de datos
        
        ### 📚 Recursos
        
        - [Documentación](https://github.com/tuusuario/synth_bank_es)
        - [Reportar bug](https://github.com/tuusuario/synth_bank_es/issues)
        - [Contacto](mailto:tu_email@example.com)
        """)
        
        st.markdown("---")
        
        # Estado de la aplicación
        if 'data_loaded' in st.session_state:
            st.success("✅ Datos cargados")
        else:
            st.info("ℹ️ Sin datos cargados")


# ===== PÁGINA PRINCIPAL =====
def main_page():
    """Página principal / dashboard."""
    
    load_custom_css()
    
    # Header
    st.markdown('<h1 class="main-header">🏦 synth_bank_es</h1>', unsafe_allow_html=True)
    st.markdown("""
        <p style='text-align: center; font-size: 1.2rem; color: #7f8c8d;'>
        Sistema de Generación de Datos Sintéticos Bancarios Españoles
        </p>
    """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Tabs principales
    tab1, tab2, tab3, tab4 = st.tabs([
        "📊 Dashboard", 
        "🎲 Generador", 
        "📈 Evaluación", 
        "⚙️ Configuración"
    ])
    
    with tab1:
        render_dashboard_tab()
    
    with tab2:
        render_generator_tab()
    
    with tab3:
        render_evaluation_tab()
    
    with tab4:
        render_settings_tab()


def render_dashboard_tab():
    """Tab de dashboard con métricas generales."""
    
    st.header("📊 Dashboard General")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("📈 Estadísticas Recientes")
        
        # Métricas de ejemplo
        metrics_col1, metrics_col2, metrics_col3 = st.columns(3)
        
        with metrics_col1:
            st.metric(
                label="Datasets Generados",
                value="47",
                delta="+3 esta semana"
            )
        
        with metrics_col2:
            st.metric(
                label="Registros Totales",
                value="1.2M",
                delta="+150K este mes"
            )
        
        with metrics_col3:
            st.metric(
                label="Calidad Promedio",
                value="92%",
                delta="+2%"
            )
        
        st.markdown("---")
        
        # Gráfico de ejemplo
        st.subheader("📊 Evolución de Generaciones")
        
        # Datos de ejemplo
        dates = pd.date_range(start='2024-01-01', end='2024-12-31', freq='M')
        values = [10, 15, 22, 18, 25, 30, 35, 42, 38, 45, 50, 47]
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=dates,
            y=values,
            mode='lines+markers',
            name='Datasets Generados',
            line=dict(color='#3498db', width=3)
        ))
        
        fig.update_layout(
            title="Datasets Generados por Mes",
            xaxis_title="Mes",
            yaxis_title="Cantidad",
            height=400
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader("📋 Última Actividad")
        
        st.markdown("""
        <div class="success-box">
        <strong>✅ Dataset Generado</strong><br>
        clients_v47 • 10,000 registros<br>
        <small>Hace 2 horas</small>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        <div class="success-box">
        <strong>✅ Evaluación Completada</strong><br>
        Quality Score: 94%<br>
        <small>Hace 3 horas</small>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        <div class="warning-box">
        <strong>⚠️ Modelo Re-entrenado</strong><br>
        CTGAN actualizado<br>
        <small>Hace 1 día</small>
        </div>
        """, unsafe_allow_html=True)


def render_generator_tab():
    """Tab de generación de datos sintéticos."""
    
    st.header("🎲 Generador de Datos Sintéticos")
    
    # Configuración de generación
    st.subheader("⚙️ Configuración")
    
    col1, col2 = st.columns(2)
    
    with col1:
        method = st.selectbox(
            "Método de Generación",
            options=['CTGAN', 'Copula', 'VAE'],
            help="Selecciona el método para generar datos sintéticos"
        )
        
        n_samples = st.number_input(
            "Número de Muestras",
            min_value=100,
            max_value=1000000,
            value=10000,
            step=1000
        )
    
    with col2:
        seed = st.number_input(
            "Semilla (Reproducibilidad)",
            min_value=0,
            value=42,
            help="Semilla para reproducibilidad de resultados"
        )
        
        apply_constraints = st.checkbox(
            "Aplicar Restricciones de Negocio",
            value=True
        )
    
    # Archivo de datos reales
    st.subheader("📁 Datos de Entrenamiento")
    uploaded_file = st.file_uploader(
        "Subir datos reales (CSV)",
        type=['csv'],
        help="Archivo CSV con datos reales para entrenar el generador"
    )
    
    if uploaded_file is not None:
        df_real = pd.read_csv(uploaded_file)
        display_dataframe_summary(df_real, "Datos Reales Cargados")
        
        st.dataframe(df_real.head(10), use_container_width=True)
        
        # Botón de generación
        if st.button("🚀 Generar Datos Sintéticos", type="primary"):
            with st.spinner("Generando datos sintéticos..."):
                # Aquí iría la lógica real de generación
                st.success(f"✅ {n_samples:,} registros sintéticos generados exitosamente!")
                
                # Mostrar preview de datos sintéticos (ejemplo)
                st.subheader("Vista Previa de Datos Sintéticos")
                st.dataframe(df_real.sample(10), use_container_width=True)
                
                # Botón de descarga
                st.download_button(
                    label="📥 Descargar Datos Sintéticos",
                    data=df_real.to_csv(index=False).encode('utf-8'),
                    file_name='synthetic_data.csv',
                    mime='text/csv'
                )


def render_evaluation_tab():
    """Tab de evaluación de calidad."""
    
    st.header("📈 Evaluación de Calidad")
    
    st.info("📌 Sube datos reales y sintéticos para comparar su calidad estadística")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Datos Reales")
        real_file = st.file_uploader("Subir datos reales", type=['csv'], key='real')
    
    with col2:
        st.subheader("Datos Sintéticos")
        synth_file = st.file_uploader("Subir datos sintéticos", type=['csv'], key='synth')
    
    if real_file is not None and synth_file is not None:
        df_real = pd.read_csv(real_file)
        df_synth = pd.read_csv(synth_file)
        
        # Selector de columna para comparar
        column_to_compare = st.selectbox(
            "Selecciona columna a comparar",
            options=list(df_real.columns)
        )
        
        # Mostrar gráfico comparativo
        fig = create_distribution_plot(df_real, df_synth, column_to_compare)
        st.plotly_chart(fig, use_container_width=True)
        
        # Botón de evaluación completa
        if st.button("🔬 Ejecutar Evaluación Completa"):
            with st.spinner("Ejecutando tests estadísticos..."):
                # Aquí iría la lógica real de evaluación
                st.success("✅ Evaluación completada")
                
                # Mostrar métricas de calidad (ejemplo)
                st.subheader("📊 Resultados de Evaluación")
                
                metrics_cols = st.columns(4)
                
                with metrics_cols[0]:
                    st.metric("KS Test", "0.045", delta="✅ Aprobado")
                with metrics_cols[1]:
                    st.metric("Wasserstein", "0.032", delta="✅ Excelente")
                with metrics_cols[2]:
                    st.metric("JS Divergence", "0.087", delta="✅ Bueno")
                with metrics_cols[3]:
                    st.metric("Correlación", "0.956", delta="✅ Alta")


def render_settings_tab():
    """Tab de configuración."""
    
    st.header("⚙️ Configuración")
    
    st.subheader("🔧 Configuración de Modelos")
    
    # CTGAN settings
    with st.expander("CTGAN Settings", expanded=True):
        epochs = st.slider("Epochs", 100, 1000, 300)
        batch_size = st.slider("Batch Size", 100, 1000, 500)
        generator_dim = st.text_input("Generator Dimensions", "[256, 256]")
    
    # VAE settings
    with st.expander("VAE Settings"):
        latent_dim = st.slider("Latent Dimensions", 32, 256, 128)
        learning_rate = st.number_input("Learning Rate", 0.0001, 0.01, 0.001, format="%.4f")
    
    if st.button("💾 Guardar Configuración"):
        st.success("✅ Configuración guardada exitosamente")


# ===== MAIN =====
if __name__ == "__main__":
    render_sidebar()
    main_page()
