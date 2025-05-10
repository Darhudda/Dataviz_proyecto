import dash
from dash import dcc, html
from dash.dependencies import Input, Output
import dash_bootstrap_components as dbc
import os
from sqlalchemy import create_engine
import pandas as pd


df = pd.read_csv("dataset_final.csv", encoding='latin1') 
print("✅ CSV cargado correctamente.")
print("Primeros registros del archivo:")
print(df.head())

engine = create_engine("postgresql+pg8000://postgres:eliana20062004@localhost:5432/proyecto_grupal")


df.to_sql("dataset_final", engine, if_exists="replace", index=False)
print("\n Datos insertados correctamente en la tabla 'dataset_final'.")

