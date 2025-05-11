import os
from dotenv import load_dotenv
import pandas as pd
from sqlalchemy import create_engine

# Cargar las variables de entorno desde un archivo .env local
load_dotenv()

# Leer el archivo CSV
df = pd.read_csv("dataset_final.csv", encoding='latin1')
print("CSV cargado correctamente.")
print("Primeros registros del archivo:")
print(df.head())

# Obtener variables de entorno
DB_USER = os.environ.get("DB_USER")
DB_PASS = os.environ.get("DB_PASS")
DB_HOST = os.environ.get("DB_HOST")
DB_PORT = os.environ.get("DB_PORT")
DB_NAME = os.environ.get("DB_NAME")

# Crear la conexión
engine = create_engine(f"postgresql+psycopg2://{DB_USER}:{DB_PASS}@{DB_HOST}:{DB_PORT}/{DB_NAME}")

# Subir a PostgreSQL en Render
df.to_sql("dataset_final", engine, if_exists="replace", index=False)
print("\nDatos insertados correctamente en la tabla 'dataset_final'.")
