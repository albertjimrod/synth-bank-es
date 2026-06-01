"""
src/synthetic/generators/client_generator.py
Generador de perfiles sintéticos de clientes bancarios españoles
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, Optional, Literal
from datetime import datetime
import yaml
import json
from loguru import logger

from sdv.single_table import CTGANSynthesizer, GaussianCopulaSynthesizer, TVAESynthesizer
from sdv.metadata import SingleTableMetadata


class ClientGenerator:
    """
    Generador de datos sintéticos de clientes bancarios.
    
    Attributes:
        n_samples: Número de clientes sintéticos a generar
        method: Método de generación ('ctgan', 'copula', 'vae')
        config_path: Ruta al archivo de configuración
        seed: Semilla para reproducibilidad
    """
    
    def __init__(
        self,
        n_samples: int = 10000,
        method: Literal['ctgan', 'copula', 'vae'] = 'ctgan',
        config_path: Optional[str] = None,
        seed: int = 42
    ):
        self.n_samples = n_samples
        self.method = method
        self.seed = seed
        self.config = self._load_config(config_path)
        self.synthesizer = None
        self.metadata = None
        
        # Configurar logger
        Path("logs/model_training").mkdir(parents=True, exist_ok=True)
        logger.add(
            f"logs/model_training/client_generator_{datetime.now():%Y%m%d_%H%M%S}.log",
            rotation="500 MB"
        )
        
        logger.info(f"Inicializado ClientGenerator con método '{method}'")
    
    def _load_config(self, config_path: Optional[str]) -> Dict:
        """Cargar configuración desde archivo YAML."""
        if config_path is None:
            config_path = "configs/model_config.yaml"
        
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        return config['synthetic_generation']['methods'][self.method]
    
    def load_real_data(self, data_path: str) -> pd.DataFrame:
        """
        Cargar datos reales para entrenar el generador.
        
        Args:
            data_path: Ruta al archivo CSV con datos reales
            
        Returns:
            DataFrame con datos reales procesados
        """
        logger.info(f"Cargando datos reales desde: {data_path}")
        
        path = Path(data_path)
        if path.suffix == '.parquet':
            df = pd.read_parquet(data_path)
        else:
            df = pd.read_csv(data_path)
        logger.info(f"Datos cargados: {len(df)} registros, {len(df.columns)} columnas")
        
        # Validación básica
        self._validate_data(df)
        
        return df
    
    def _validate_data(self, df: pd.DataFrame) -> None:
        """Validar que los datos cumplen requisitos mínimos."""
        required_columns = [
            'edad', 'nivel_estudios', 'situacion_laboral',
            'ingresos_anuales', 'saldo_cuenta'
        ]
        # 'provincia' o 'ccaa' son válidos como columna geográfica
        if 'provincia' not in df.columns and 'ccaa' not in df.columns:
            required_columns.append('provincia')
        
        missing = set(required_columns) - set(df.columns)
        if missing:
            raise ValueError(f"Faltan columnas requeridas: {missing}")
        
        # Verificar valores nulos
        null_counts = df[required_columns].isnull().sum()
        if null_counts.any():
            logger.warning(f"Columnas con valores nulos:\n{null_counts[null_counts > 0]}")
        
        logger.info("✓ Validación de datos completada")
    
    def prepare_metadata(self, df: pd.DataFrame) -> SingleTableMetadata:
        """Preparar metadatos para SDV."""
        metadata = SingleTableMetadata()
        metadata.detect_from_dataframe(df)
        
        # Ajustes manuales si es necesario
        metadata.update_column('edad', sdtype='numerical')
        metadata.update_column('provincia', sdtype='categorical')
        metadata.update_column('nivel_estudios', sdtype='categorical')
        metadata.update_column('situacion_laboral', sdtype='categorical')
        metadata.update_column('ingresos_anuales', sdtype='numerical')
        metadata.update_column('saldo_cuenta', sdtype='numerical')
        
        self.metadata = metadata
        logger.info("✓ Metadatos preparados")
        
        return metadata
    
    def fit(self, df: pd.DataFrame) -> None:
        """
        Entrenar el modelo generativo.
        
        Args:
            df: DataFrame con datos reales
        """
        logger.info(f"Entrenando modelo {self.method.upper()}...")
        
        # Preparar metadatos
        if self.metadata is None:
            self.prepare_metadata(df)
        
        # Inicializar synthesizer según método
        if self.method == 'ctgan':
            self.synthesizer = CTGANSynthesizer(
                metadata=self.metadata,
                epochs=self.config['hyperparameters']['epochs'],
                verbose=self.config['hyperparameters']['verbose']
            )
        
        elif self.method == 'copula':
            self.synthesizer = GaussianCopulaSynthesizer(
                metadata=self.metadata,
                default_distribution=self.config['hyperparameters']['default_distribution']
            )
        
        elif self.method == 'vae':
            self.synthesizer = TVAESynthesizer(
                metadata=self.metadata,
                epochs=self.config.get('hyperparameters', {}).get('epochs', 100),
                verbose=self.config.get('hyperparameters', {}).get('verbose', True)
            )
        else:
            raise ValueError(f"Método no soportado: {self.method}. Opciones: 'ctgan', 'copula', 'vae'")
        
        # Entrenar
        start_time = datetime.now()
        self.synthesizer.fit(df)
        training_time = (datetime.now() - start_time).total_seconds()
        
        logger.info(f"✓ Entrenamiento completado en {training_time:.2f} segundos")
    
    def generate(
        self,
        apply_constraints: bool = True,
        apply_post_processing: bool = True
    ) -> pd.DataFrame:
        """
        Generar datos sintéticos.
        
        Args:
            apply_constraints: Si aplicar restricciones de negocio
            apply_post_processing: Si aplicar post-procesamiento
            
        Returns:
            DataFrame con datos sintéticos
        """
        if self.synthesizer is None:
            raise ValueError("Debe entrenar el modelo primero con fit()")
        
        logger.info(f"Generando {self.n_samples} clientes sintéticos...")
        
        # Generar
        synthetic_df = self.synthesizer.sample(num_rows=self.n_samples)
        
        # Aplicar restricciones
        if apply_constraints:
            synthetic_df = self._apply_constraints(synthetic_df)
        
        # Post-procesamiento
        if apply_post_processing:
            synthetic_df = self._post_process(synthetic_df)
        
        logger.info(f"✓ Generación completada: {len(synthetic_df)} registros")
        
        return synthetic_df
    
    def _apply_constraints(self, df: pd.DataFrame) -> pd.DataFrame:
        """Aplicar restricciones de negocio."""
        logger.info("Aplicando restricciones de negocio...")
        
        # Edad mínima
        df['edad'] = df['edad'].clip(lower=18)
        
        # Saldo no negativo
        df['saldo_cuenta'] = df['saldo_cuenta'].clip(lower=0)
        
        # Ingresos positivos
        df['ingresos_anuales'] = df['ingresos_anuales'].clip(lower=0)
        
        # Ratio deuda/ingreso
        if 'deuda_total' in df.columns:
            max_deuda = df['ingresos_anuales'] * 0.4
            df['deuda_total'] = np.minimum(df['deuda_total'], max_deuda)
        
        logger.info("✓ Restricciones aplicadas")
        return df
    
    def _post_process(self, df: pd.DataFrame) -> pd.DataFrame:
        """Post-procesamiento de datos generados."""
        logger.info("Aplicando post-procesamiento...")
        
        # Redondeos
        df['edad'] = df['edad'].round().astype(int)
        df['ingresos_anuales'] = (df['ingresos_anuales'] / 100).round() * 100
        df['saldo_cuenta'] = df['saldo_cuenta'].round(2)
        
        # Categorías válidas
        valid_provinces = ["Madrid", "Barcelona", "Valencia", "Sevilla", "Zaragoza"]
        df['provincia'] = df['provincia'].apply(
            lambda x: x if x in valid_provinces else np.random.choice(valid_provinces)
        )
        
        logger.info("✓ Post-procesamiento completado")
        return df
    
    def save(
        self,
        output_dir: str,
        synthetic_df: pd.DataFrame,
        save_model: bool = True
    ) -> Dict[str, str]:
        """
        Guardar datos sintéticos y modelo.
        
        Args:
            output_dir: Directorio de salida
            synthetic_df: DataFrame con datos sintéticos
            save_model: Si guardar el modelo entrenado
            
        Returns:
            Diccionario con rutas de archivos guardados
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Guardar datos
        data_path = output_path / "clients_synthetic.csv"
        synthetic_df.to_csv(data_path, index=False)
        logger.info(f"✓ Datos guardados en: {data_path}")
        
        # Guardar modelo
        model_path = None
        if save_model and self.synthesizer is not None:
            model_path = output_path / f"model_{self.method}.pkl"
            self.synthesizer.save(str(model_path))
            logger.info(f"✓ Modelo guardado en: {model_path}")
        
        # Guardar metadatos
        metadata_dict = {
            'timestamp': datetime.now().isoformat(),
            'method': self.method,
            'n_samples': self.n_samples,
            'seed': self.seed,
            'shape': synthetic_df.shape,
            'columns': list(synthetic_df.columns),
            'config': self.config
        }
        
        metadata_path = output_path / "metadata.json"
        with open(metadata_path, 'w') as f:
            json.dump(metadata_dict, f, indent=2)
        logger.info(f"✓ Metadatos guardados en: {metadata_path}")
        
        return {
            'data': str(data_path),
            'model': str(model_path) if model_path else None,
            'metadata': str(metadata_path)
        }
    
    @classmethod
    def from_pretrained(cls, model_path: str) -> 'ClientGenerator':
        """Cargar modelo pre-entrenado."""
        logger.info(f"Cargando modelo desde: {model_path}")
        
        # Detectar método desde nombre de archivo
        method = 'ctgan' if 'ctgan' in model_path else 'copula'
        
        generator = cls(method=method)
        
        if method == 'ctgan':
            generator.synthesizer = CTGANSynthesizer.load(model_path)
        elif method == 'copula':
            generator.synthesizer = GaussianCopulaSynthesizer.load(model_path)
        
        logger.info("✓ Modelo cargado exitosamente")
        return generator


def main():
    """Ejemplo de uso."""
    
    # Inicializar generador
    generator = ClientGenerator(
        n_samples=10000,
        method='ctgan',
        seed=42
    )
    
    # Cargar datos reales
    real_data = generator.load_real_data('data/processed/clients_real.csv')
    
    # Entrenar
    generator.fit(real_data)
    
    # Generar sintéticos
    synthetic_data = generator.generate()
    
    # Guardar
    output_dir = f"data/synthetic/clients/v1_{datetime.now():%Y%m%d}"
    paths = generator.save(output_dir, synthetic_data)
    
    print(f"\n✅ Generación completada!")
    print(f"📁 Datos: {paths['data']}")
    print(f"🤖 Modelo: {paths['model']}")
    print(f"📋 Metadatos: {paths['metadata']}")


if __name__ == "__main__":
    main()
