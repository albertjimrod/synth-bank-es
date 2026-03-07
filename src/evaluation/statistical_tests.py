"""
src/evaluation/statistical_tests.py
Evaluación estadística de la calidad de datos sintéticos
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple
from scipy import stats
from scipy.spatial.distance import jensenshannon
import matplotlib.pyplot as plt
import seaborn as sns
from loguru import logger


class StatisticalTests:
    """
    Batería de tests estadísticos para evaluar calidad de datos sintéticos.
    """
    
    def __init__(
        self,
        real_data: pd.DataFrame,
        synthetic_data: pd.DataFrame,
        alpha: float = 0.05
    ):
        """
        Args:
            real_data: DataFrame con datos reales
            synthetic_data: DataFrame con datos sintéticos
            alpha: Nivel de significancia para tests
        """
        self.real_data = real_data
        self.synthetic_data = synthetic_data
        self.alpha = alpha
        self.results = {}
        
        logger.info(f"Evaluador inicializado - Real: {len(real_data)}, Sintético: {len(synthetic_data)}")
    
    def kolmogorov_smirnov_test(self, column: str) -> Dict:
        """
        Test de Kolmogorov-Smirnov para variables numéricas.
        
        Compara las distribuciones acumuladas de datos reales y sintéticos.
        Un p-valor alto (> alpha) indica que las distribuciones son similares.
        """
        real_values = self.real_data[column].dropna()
        synth_values = self.synthetic_data[column].dropna()
        
        statistic, p_value = stats.ks_2samp(real_values, synth_values)
        
        result = {
            'test': 'Kolmogorov-Smirnov',
            'column': column,
            'statistic': statistic,
            'p_value': p_value,
            'passed': p_value > self.alpha,
            'interpretation': 'Similar' if p_value > self.alpha else 'Diferente'
        }
        
        logger.info(f"KS Test [{column}]: statistic={statistic:.4f}, p={p_value:.4f}")
        return result
    
    def chi_square_test(self, column: str) -> Dict:
        """
        Test Chi-cuadrado para variables categóricas.
        
        Compara las frecuencias observadas en datos sintéticos
        con las esperadas según datos reales.
        """
        # Frecuencias reales
        real_counts = self.real_data[column].value_counts()
        synth_counts = self.synthetic_data[column].value_counts()
        
        # Alinear categorías
        categories = real_counts.index.union(synth_counts.index)
        real_freq = [real_counts.get(cat, 0) for cat in categories]
        synth_freq = [synth_counts.get(cat, 0) for cat in categories]
        
        # Normalizar a proporciones
        real_prop = np.array(real_freq) / sum(real_freq)
        expected_counts = real_prop * sum(synth_freq)
        
        # Chi-square test
        statistic, p_value = stats.chisquare(synth_freq, expected_counts)
        
        result = {
            'test': 'Chi-Square',
            'column': column,
            'statistic': statistic,
            'p_value': p_value,
            'passed': p_value > self.alpha,
            'categories': len(categories),
            'interpretation': 'Similar' if p_value > self.alpha else 'Diferente'
        }
        
        logger.info(f"Chi-Square [{column}]: statistic={statistic:.4f}, p={p_value:.4f}")
        return result
    
    def wasserstein_distance(self, column: str) -> Dict:
        """
        Distancia de Wasserstein (Earth Mover's Distance).
        
        Mide la "distancia" entre dos distribuciones.
        Valores más bajos indican mayor similitud.
        """
        real_values = self.real_data[column].dropna()
        synth_values = self.synthetic_data[column].dropna()
        
        distance = stats.wasserstein_distance(real_values, synth_values)
        
        # Normalizar por rango de datos
        data_range = real_values.max() - real_values.min()
        normalized_distance = distance / data_range if data_range > 0 else 0
        
        result = {
            'test': 'Wasserstein Distance',
            'column': column,
            'distance': distance,
            'normalized_distance': normalized_distance,
            'quality': self._interpret_wasserstein(normalized_distance)
        }
        
        logger.info(f"Wasserstein [{column}]: distance={distance:.4f}")
        return result
    
    def _interpret_wasserstein(self, norm_distance: float) -> str:
        """Interpretar distancia de Wasserstein normalizada."""
        if norm_distance < 0.05:
            return 'Excelente'
        elif norm_distance < 0.1:
            return 'Buena'
        elif norm_distance < 0.2:
            return 'Aceptable'
        else:
            return 'Pobre'
    
    def jensen_shannon_divergence(self, column: str, bins: int = 30) -> Dict:
        """
        Divergencia de Jensen-Shannon.
        
        Mide la similitud entre dos distribuciones de probabilidad.
        Rango: [0, 1], donde 0 = idénticas.
        """
        real_values = self.real_data[column].dropna()
        synth_values = self.synthetic_data[column].dropna()
        
        # Crear bins comunes
        min_val = min(real_values.min(), synth_values.min())
        max_val = max(real_values.max(), synth_values.max())
        bin_edges = np.linspace(min_val, max_val, bins + 1)
        
        # Histogramas normalizados
        real_hist, _ = np.histogram(real_values, bins=bin_edges)
        synth_hist, _ = np.histogram(synth_values, bins=bin_edges)
        
        real_hist = real_hist / real_hist.sum()
        synth_hist = synth_hist / synth_hist.sum()
        
        # Calcular divergencia
        divergence = jensenshannon(real_hist, synth_hist)
        
        result = {
            'test': 'Jensen-Shannon Divergence',
            'column': column,
            'divergence': divergence,
            'similarity': 1 - divergence,
            'quality': self._interpret_jsd(divergence)
        }
        
        logger.info(f"JS Divergence [{column}]: {divergence:.4f}")
        return result
    
    def _interpret_jsd(self, divergence: float) -> str:
        """Interpretar divergencia de Jensen-Shannon."""
        if divergence < 0.1:
            return 'Excelente'
        elif divergence < 0.2:
            return 'Buena'
        elif divergence < 0.3:
            return 'Aceptable'
        else:
            return 'Pobre'
    
    def correlation_comparison(self) -> Dict:
        """
        Comparar matrices de correlación.
        
        Calcula la diferencia entre correlaciones de datos reales y sintéticos.
        """
        # Seleccionar solo columnas numéricas
        numeric_cols = self.real_data.select_dtypes(include=[np.number]).columns
        
        real_corr = self.real_data[numeric_cols].corr()
        synth_corr = self.synthetic_data[numeric_cols].corr()
        
        # Diferencia absoluta media
        corr_diff = np.abs(real_corr - synth_corr)
        mean_diff = corr_diff.mean().mean()
        max_diff = corr_diff.max().max()
        
        result = {
            'test': 'Correlation Comparison',
            'mean_difference': mean_diff,
            'max_difference': max_diff,
            'quality': self._interpret_corr_diff(mean_diff)
        }
        
        logger.info(f"Correlation Diff: mean={mean_diff:.4f}, max={max_diff:.4f}")
        return result
    
    def _interpret_corr_diff(self, mean_diff: float) -> str:
        """Interpretar diferencia de correlación."""
        if mean_diff < 0.05:
            return 'Excelente'
        elif mean_diff < 0.1:
            return 'Buena'
        elif mean_diff < 0.15:
            return 'Aceptable'
        else:
            return 'Pobre'
    
    def statistical_summary_comparison(self) -> pd.DataFrame:
        """
        Comparar estadísticas descriptivas.
        
        Returns:
            DataFrame con comparación lado a lado
        """
        numeric_cols = self.real_data.select_dtypes(include=[np.number]).columns
        
        real_stats = self.real_data[numeric_cols].describe()
        synth_stats = self.synthetic_data[numeric_cols].describe()
        
        # Calcular diferencias porcentuales
        comparison = pd.DataFrame({
            'Metric': real_stats.index
        })
        
        for col in numeric_cols:
            comparison[f'{col}_real'] = real_stats[col]
            comparison[f'{col}_synth'] = synth_stats[col]
            comparison[f'{col}_diff_%'] = (
                (synth_stats[col] - real_stats[col]) / real_stats[col] * 100
            ).round(2)
        
        return comparison
    
    def run_all_tests(self) -> Dict:
        """
        Ejecutar todos los tests disponibles.
        
        Returns:
            Diccionario con resultados de todos los tests
        """
        logger.info("🔬 Ejecutando batería completa de tests...")
        
        results = {
            'numerical_tests': [],
            'categorical_tests': [],
            'correlation': None,
            'summary': None
        }
        
        # Tests para columnas numéricas
        numeric_cols = self.real_data.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            try:
                results['numerical_tests'].append({
                    'ks_test': self.kolmogorov_smirnov_test(col),
                    'wasserstein': self.wasserstein_distance(col),
                    'js_divergence': self.jensen_shannon_divergence(col)
                })
            except Exception as e:
                logger.error(f"Error en tests numéricos para {col}: {e}")
        
        # Tests para columnas categóricas
        categorical_cols = self.real_data.select_dtypes(include=['object', 'category']).columns
        for col in categorical_cols:
            try:
                results['categorical_tests'].append(
                    self.chi_square_test(col)
                )
            except Exception as e:
                logger.error(f"Error en test categórico para {col}: {e}")
        
        # Comparación de correlaciones
        try:
            results['correlation'] = self.correlation_comparison()
        except Exception as e:
            logger.error(f"Error en comparación de correlaciones: {e}")
        
        # Resumen estadístico
        try:
            results['summary'] = self.statistical_summary_comparison()
        except Exception as e:
            logger.error(f"Error en resumen estadístico: {e}")
        
        self.results = results
        logger.info("✓ Tests completados")
        
        return results
    
    def generate_report(self, output_dir: str = 'reports/evaluation_reports') -> str:
        """
        Generar informe HTML con resultados.
        
        Args:
            output_dir: Directorio de salida
            
        Returns:
            Ruta al archivo HTML generado
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        timestamp = pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')
        report_file = output_path / f'evaluation_report_{timestamp}.html'
        
        # Crear visualizaciones
        self._create_visualizations(output_path)
        
        # Generar HTML
        html_content = self._generate_html_report()
        
        with open(report_file, 'w') as f:
            f.write(html_content)
        
        logger.info(f"✓ Informe guardado en: {report_file}")
        return str(report_file)
    
    def _create_visualizations(self, output_dir: Path) -> None:
        """Crear gráficos comparativos."""
        # Implementar visualizaciones específicas
        pass
    
    def _generate_html_report(self) -> str:
        """Generar contenido HTML del informe."""
        html = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Informe de Evaluación - Datos Sintéticos</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 40px; }}
                h1 {{ color: #2c3e50; }}
                table {{ border-collapse: collapse; width: 100%; margin: 20px 0; }}
                th, td {{ border: 1px solid #ddd; padding: 12px; text-align: left; }}
                th {{ background-color: #3498db; color: white; }}
                .pass {{ background-color: #d5f4e6; }}
                .fail {{ background-color: #fadbd8; }}
            </style>
        </head>
        <body>
            <h1>📊 Informe de Evaluación de Datos Sintéticos</h1>
            <p><strong>Fecha:</strong> {pd.Timestamp.now()}</p>
            <p><strong>Registros Reales:</strong> {len(self.real_data)}</p>
            <p><strong>Registros Sintéticos:</strong> {len(self.synthetic_data)}</p>
            
            <h2>Resumen de Tests</h2>
            <p>Resumen de resultados...</p>
            
            <!-- Agregar más secciones según necesidad -->
        </body>
        </html>
        """
        return html


def main():
    """Ejemplo de uso."""
    
    # Cargar datos
    real_data = pd.read_csv('data/processed/clients_real.csv')
    synthetic_data = pd.read_csv('data/synthetic/clients/latest/clients_synthetic.csv')
    
    # Inicializar evaluador
    evaluator = StatisticalTests(real_data, synthetic_data, alpha=0.05)
    
    # Ejecutar tests
    results = evaluator.run_all_tests()
    
    # Generar informe
    report_path = evaluator.generate_report()
    
    print(f"\n✅ Evaluación completada!")
    print(f"📄 Informe: {report_path}")


if __name__ == "__main__":
    main()
