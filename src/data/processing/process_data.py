#!/usr/bin/env python3
"""
Procesador de microdatos INE para dataset bancario sintético.
Estructura esperada en data/raw/:
  - datos_2024/ (ECV)
  - datos_2024_epf/ (EPF)
  - datos_3t25/ (EPA)
"""
import pandas as pd
import numpy as np
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

BASE_DIR = Path(__file__).parent.parent
RAW_DIR = BASE_DIR / "data" / "raw"
OUT_DIR = BASE_DIR / "data" / "processed"

# Configuración de archivos CSV
CONFIG = {
    'ecv': {
        'path': RAW_DIR / "datos_2024",
        'files': {
            'personas': 'ECV_Tp_2024/CSV',
            'hogares': 'ECV_Th_2024/CSV', 
            'renta': 'ECV_Tr_2024/CSV',
            'datos': 'ECV_Td_2024/CSV'
        }
    },
    'epf': {
        'path': RAW_DIR / "datos_2024_epf",
        'files': {
            'hogar': 'EPFhogar/CSV',
            'gastos': 'EPFgastos/CSV',
            'miembros': 'EPFmhogar/CSV'
        }
    },
    'epa': {
        'path': RAW_DIR / "datos_3t25" / "CSV",
        'file': 'EPA_2025T3.tab'
    }
}

def load_csv_from_dir(dir_path, encoding='latin-1'):
    """Carga CSV/TAB desde directorio (detecta separador automáticamente)."""
    dir_path = Path(dir_path)
    if not dir_path.exists():
        print(f"  [WARN] No existe: {dir_path}")
        return None
    
    files = list(dir_path.glob('*.csv')) + list(dir_path.glob('*.tab'))
    if not files:
        print(f"  [WARN] Sin CSV/TAB en: {dir_path}")
        return None
    
    # Probar diferentes separadores
    for sep in ['\t', ',', ';']:
        try:
            df = pd.read_csv(files[0], sep=sep, encoding=encoding, low_memory=False)
            if len(df.columns) > 1:  # Separador correcto si hay más de 1 columna
                print(f"  [OK] {files[0].name}: {len(df)} registros, {len(df.columns)} cols")
                return df
        except:
            continue
    
    # Si ninguno funcionó, reportar error
    print(f"  [ERR] No se pudo leer: {files[0].name}")
    return None

def process_ecv():
    """Procesa Encuesta de Condiciones de Vida."""
    print("\n=== Procesando ECV 2024 ===")
    cfg = CONFIG['ecv']
    
    dfs = {}
    for key, subpath in cfg['files'].items():
        full_path = cfg['path'] / subpath
        dfs[key] = load_csv_from_dir(full_path)
    
    # Combinar personas + renta si ambos existen
    if dfs['personas'] is not None:
        df = dfs['personas'].copy()
        
        # Identificar columnas de merge (ID hogar/persona)
        id_cols = [c for c in df.columns if 'ID' in c.upper() or 'HOG' in c.upper()[:3]]
        
        # Merge con renta si existe y hay columnas comunes
        if dfs['renta'] is not None:
            common = set(df.columns) & set(dfs['renta'].columns)
            merge_on = [c for c in common if 'ID' in c.upper() or c in id_cols]
            if merge_on:
                df = df.merge(dfs['renta'], on=merge_on, how='left', suffixes=('', '_r'))
        
        # Mapear columnas - solo primera coincidencia por target
        col_map = {}
        mapped_targets = set()
        
        for col in df.columns:
            col_up = col.upper()
            # Columnas específicas EU-SILC
            if col_up == 'PB140' and 'edad' not in mapped_targets:
                col_map[col] = 'edad'
                mapped_targets.add('edad')
            elif col_up == 'PB150' and 'sexo' not in mapped_targets:
                col_map[col] = 'sexo'
                mapped_targets.add('sexo')
            elif col_up == 'DB040' and 'ccaa' not in mapped_targets:
                col_map[col] = 'ccaa'
                mapped_targets.add('ccaa')
            elif col_up == 'PE040' and 'nivel_estudios' not in mapped_targets:
                col_map[col] = 'nivel_estudios'
                mapped_targets.add('nivel_estudios')
            elif col_up == 'HY020' and 'renta_disponible' not in mapped_targets:
                col_map[col] = 'renta_disponible'
                mapped_targets.add('renta_disponible')
            elif col_up == 'PL031' and 'situacion_laboral' not in mapped_targets:
                col_map[col] = 'situacion_laboral'
                mapped_targets.add('situacion_laboral')
            elif col_up == 'PY010G' and 'ingresos_trabajo' not in mapped_targets:
                col_map[col] = 'ingresos_trabajo'
                mapped_targets.add('ingresos_trabajo')
        
        if col_map:
            df = df.rename(columns=col_map)
        
        # Eliminar columnas duplicadas
        df = df.loc[:, ~df.columns.duplicated()]
        
        df['fuente'] = 'ECV'
        return df
    return None

def process_epf():
    """Procesa Encuesta de Presupuestos Familiares."""
    print("\n=== Procesando EPF 2024 ===")
    cfg = CONFIG['epf']
    
    dfs = {}
    for key, subpath in cfg['files'].items():
        full_path = cfg['path'] / subpath
        dfs[key] = load_csv_from_dir(full_path)
    
    if dfs['hogar'] is not None:
        df = dfs['hogar'].copy()
        
        # Mapear columnas conocidas ANTES del merge
        col_map = {}
        for col in df.columns:
            col_up = col.upper()
            if 'CCAA' in col_up and 'ccaa' not in col_map.values():
                col_map[col] = 'ccaa'
            elif ('NMIEMB' in col_up or col_up == 'MIEMB') and 'n_miembros' not in col_map.values():
                col_map[col] = 'n_miembros'
            elif 'IMPEXAC' in col_up and 'gasto_total' not in col_map.values():
                col_map[col] = 'gasto_total'
            elif 'GASTMON' in col_up and 'gasto_monetario' not in col_map.values():
                col_map[col] = 'gasto_monetario'
            elif 'REGTEN' in col_up and 'regimen_vivienda' not in col_map.values():
                col_map[col] = 'regimen_vivienda'
        
        if col_map:
            df = df.rename(columns=col_map)
        
        # Eliminar columnas duplicadas si existen
        df = df.loc[:, ~df.columns.duplicated()]
        
        df['fuente'] = 'EPF'
        return df
    return None

def process_epa():
    """Procesa Encuesta de Población Activa."""
    print("\n=== Procesando EPA 3T2025 ===")
    cfg = CONFIG['epa']
    
    file_path = cfg['path'] / cfg['file']
    if not file_path.exists():
        # Buscar cualquier archivo en el directorio
        df = load_csv_from_dir(cfg['path'])
    else:
        try:
            df = pd.read_csv(file_path, sep='\t', encoding='latin-1', low_memory=False)
            print(f"  [OK] {cfg['file']}: {len(df)} registros")
        except:
            df = pd.read_csv(file_path, sep=',', encoding='latin-1', low_memory=False)
    
    if df is not None:
        # Mapear columnas EPA - solo primera coincidencia
        col_map = {}
        mapped_targets = set()
        
        for col in df.columns:
            col_up = col.upper()
            # Solo mapear si el target no está ya asignado
            if 'EDAD1' == col_up and 'edad' not in mapped_targets:
                col_map[col] = 'edad'
                mapped_targets.add('edad')
            elif 'SEXO1' == col_up and 'sexo' not in mapped_targets:
                col_map[col] = 'sexo'
                mapped_targets.add('sexo')
            elif 'CCAA' == col_up and 'ccaa' not in mapped_targets:
                col_map[col] = 'ccaa'
                mapped_targets.add('ccaa')
            elif 'AOI' == col_up and 'situacion_laboral' not in mapped_targets:
                col_map[col] = 'situacion_laboral'
                mapped_targets.add('situacion_laboral')
            elif 'CNO11' == col_up and 'ocupacion' not in mapped_targets:
                col_map[col] = 'ocupacion'
                mapped_targets.add('ocupacion')
            elif 'SPTS' == col_up and 'sector' not in mapped_targets:
                col_map[col] = 'sector'
                mapped_targets.add('sector')
            elif 'SALAM1' == col_up and 'tramo_salario' not in mapped_targets:
                col_map[col] = 'tramo_salario'
                mapped_targets.add('tramo_salario')
        
        if col_map:
            df = df.rename(columns=col_map)
        
        # Eliminar columnas duplicadas si existen
        df = df.loc[:, ~df.columns.duplicated()]
        
        df['fuente'] = 'EPA'
        return df
    return None

def create_synthetic_base():
    """Combina fuentes en dataset base para síntesis."""
    print("\n=== Creando dataset combinado ===")
    
    ecv = process_ecv()
    epf = process_epf()
    epa = process_epa()
    
    datasets = []
    
    # Procesar ECV (datos de renta e ingresos)
    if ecv is not None:
        cols_keep = ['edad', 'sexo', 'ccaa', 'nivel_estudios', 'renta_hogar', 
                     'renta_disponible', 'situacion_laboral', 'fuente']
        cols_exist = [c for c in cols_keep if c in ecv.columns]
        if cols_exist:
            datasets.append(ecv[cols_exist].copy())
            print(f"  ECV: {len(ecv)} registros, cols: {cols_exist}")
    
    # Procesar EPF (datos de gasto)
    if epf is not None:
        cols_keep = ['ccaa', 'n_miembros', 'gasto_total', 'gasto_monetario',
                     'regimen_vivienda', 'fuente']
        cols_exist = [c for c in cols_keep if c in epf.columns]
        if cols_exist:
            datasets.append(epf[cols_exist].copy())
            print(f"  EPF: {len(epf)} registros, cols: {cols_exist}")
    
    # Procesar EPA (datos laborales)
    if epa is not None:
        cols_keep = ['edad', 'sexo', 'ccaa', 'situacion_laboral', 'ocupacion',
                     'sector', 'tramo_salario', 'fuente']
        cols_exist = [c for c in cols_keep if c in epa.columns]
        if cols_exist:
            datasets.append(epa[cols_exist].copy())
            print(f"  EPA: {len(epa)} registros, cols: {cols_exist}")
    
    # Guardar datasets procesados individualmente
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    
    if ecv is not None:
        ecv.to_parquet(OUT_DIR / "ecv_processed.parquet", index=False)
    if epf is not None:
        epf.to_parquet(OUT_DIR / "epf_processed.parquet", index=False)
    if epa is not None:
        epa.to_parquet(OUT_DIR / "epa_processed.parquet", index=False)
    
    # Crear dataset combinado (solo si hay datos)
    if datasets:
        # Concatenar manteniendo todas las columnas
        combined = pd.concat(datasets, ignore_index=True, sort=False)
        combined.to_parquet(OUT_DIR / "combined_base.parquet", index=False)
        print(f"\n[OK] Dataset combinado: {len(combined)} registros")
        print(f"    Columnas: {list(combined.columns)}")
        print(f"    Guardado en: {OUT_DIR}/")
        return combined
    else:
        print("\n[ERR] No se pudieron procesar datos")
        return None

def main():
    print("="*50)
    print("PROCESADOR DE MICRODATOS INE")
    print("="*50)
    
    # Verificar estructura
    print(f"\nDirectorio base: {RAW_DIR}")
    if not RAW_DIR.exists():
        print(f"[ERR] No existe {RAW_DIR}")
        print("Coloca los microdatos descomprimidos en data/raw/")
        return
    
    # Listar contenido
    print("Contenido encontrado:")
    for item in RAW_DIR.iterdir():
        print(f"  - {item.name}/")
    
    # Procesar
    df = create_synthetic_base()
    
    if df is not None:
        print("\n=== Resumen estadístico ===")
        print(df.describe(include='all').T.head(10))

if __name__ == "__main__":
    main()