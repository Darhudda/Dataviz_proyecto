import dash
from dash import dcc, html
import dash_bootstrap_components as dbc
import os
import pandas as pd
from sqlalchemy import create_engine
from dash import Input, Output
import plotly.express as px
import joblib
from dash import dash_table


# Configuración de conexión a PostgreSQL
from sqlalchemy import create_engine
import os
from dotenv import load_dotenv
load_dotenv()


usuario = os.environ.get("DB_USER")
contraseña = os.environ.get("DB_PASS")
host = os.environ.get("DB_HOST")
puerto = os.environ.get("DB_PORT")
base_datos = os.environ.get("DB_NAME")

engine = create_engine(f"postgresql+psycopg2://{usuario}:{contraseña}@{host}:{puerto}/{base_datos}")


# Leer tabla desde la base de datos
df = pd.read_sql("SELECT * FROM dataset_final", con=engine)

query = """
SELECT EXISTS (
    SELECT 1
    FROM information_schema.tables
    WHERE table_schema = 'public'
    AND table_name = 'dataset_final'
);
"""

# INCIO DEL DASHBOARD
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])
html.Div([
    html.H1("Análisis y Detección de Fraudes en Transacciones Electrónicas", style={
        'color': 'white',
        'textAlign': 'center',
        'padding': '20px'
    })
], style={'backgroundColor': '#111111'})

app.title = "Dashboard del Proyecto Final "
server = app.server  


subtabs_metodologia = dcc.Tabs([
    dcc.Tab(label='a. Definición del Problema', children=[
        html.H4('a. Definición del Problema a Resolver'),
        html.Ul([
            html.Li('Tipo de problema: clasificación / regresión / agrupamiento / series de tiempo'),
            html.Li('Variable objetivo o de interés: Nombre de la variable')
        ])
    ]),
    dcc.Tab(label='b. Preparación de Datos', children=[
        html.H4('b. Preparación de los Datos'),
        html.Ul([
            html.Li('Limpieza y transformación de datos'),
            html.Li('División del dataset en entrenamiento y prueba o validación cruzada')
        ])
    ]),
    dcc.Tab(label='c. Selección del Modelo', children=[
        html.H4('c. Selección del Modelo o Algoritmo'),
        html.Ul([
            html.Li('Modelo(s) seleccionados'),
            html.Li('Justificación de la elección'),
            html.Li('Ecuación o representación matemática si aplica')
        ])
    ]),
    dcc.Tab(label='d. Evaluación del Modelo', children=[
        html.H4('d. Entrenamiento y Evaluación del Modelo'),
        html.Ul([
            html.Li('Proceso de entrenamiento'),
            html.Li('Métricas de evaluación: RMSE, MAE, Accuracy, etc.'),
            html.Li('Validación utilizada')
        ])
    ])
])

# Variables numéricas realmente interpretables
# Detectar columnas numéricas interpretables automáticamente
columnas_numericas_validas = [
    'transactionamt', 'd1', 'd8', 'd9', 'c14', 'v98', 'v160', 'v161', 'v162', 'v164',
    'v172', 'v173', 'v174', 'v175', 'v177', 'v184', 'v185', 'v223', 'v224', 'v226',
    'v229', 'v238', 'v250'
]

options=[{'label': col, 'value': col} for col in columnas_numericas_validas if col in df.columns]


# Crear figura de balance de clases
conteo = df['isfraud'].value_counts().sort_index()
proporcion = df['isfraud'].value_counts(normalize=True).sort_index()

fig_balance_clases = px.pie(
    names=['No Fraude', 'Fraude'],
    values=conteo.values,
    title='Distribución de Clases (isfraud)',
    hole=0.3,
    color_discrete_sequence=["#2ca02c", "#d62728"]
)
fig_balance_clases.update_traces(textinfo='percent+label')

# crear modelo
metrics_df = joblib.load('metrics_df.pkl')
fig_roc = joblib.load('fig_roc.pkl')
fig_cm = joblib.load('fig_cm.pkl')
fig_imp = joblib.load('fig_imp.pkl')
fig_disp = joblib.load('fig_disp.pkl')


# Crear columna de hora virtual si no existe
if 'hora_virtual' not in df.columns:
    df['hora_virtual'] = df['transactiondt'] // 3600

# Agrupar por hora e isfraud
hora_stats = df.groupby(['hora_virtual', 'isfraud']).size().unstack(fill_value=0)
hora_stats.columns = ['No Fraude', 'Fraude']

# Crear figura
import plotly.graph_objects as go

fig_transacciones_hora = go.Figure()
fig_transacciones_hora.add_trace(go.Scatter(
    x=hora_stats.index, y=hora_stats['No Fraude'], mode='lines', name='No Fraude'))
fig_transacciones_hora.add_trace(go.Scatter(
    x=hora_stats.index, y=hora_stats['Fraude'], mode='lines', name='Fraude'))

fig_transacciones_hora.update_layout(
    title='Transacciones por Hora Virtual',
    xaxis_title='Hora virtual (desde el inicio del registro)',
    yaxis_title='Número de transacciones',
    template='plotly_white',
    height=400
)


subtabs_resultados = dcc.Tabs([
    dcc.Tab(label='a. EDA', children=[
        html.H4('a. Análisis Exploratorio de Datos (EDA)'),
        
        dbc.Row([
            dbc.Col(html.Div([
                html.H5("Total de transacciones", className="text-white text-center"),
                html.H4(f"{len(df):,}", className="text-center text-white")
            ], className="p-3 rounded",style={"backgroundColor": "#636efa"}), width=4),

            dbc.Col(html.Div([
                html.H5("Fraude", className=" text-white text-center"),
                html.H4(f"{df['isfraud'].sum():,}", className="text-center text-white")
            ], className="p-3 rounded", style={"backgroundColor": "#ef553b"}), width=4),

            dbc.Col(html.Div([
                html.H5("No Fraude", className="text-white text-center"),
                html.H4(f"{(df['isfraud'] == 0).sum():,}", className="text-center text-white")
            ], className="p-3 rounded", style={"backgroundColor": "#00cc96"}), width=4)
        ], className="mb-4"),

        
        html.Label('Selecciona variable numérica:'),
        dcc.Dropdown(
            id='eda-variable-dropdown',
           options=[{'label': col, 'value': col} for col in columnas_numericas_validas if col in df.columns],
            value='transactionamt',  # o cualquier otra variable que te interese por defecto
            style={'width': '50%'}
        ),

        html.Br(),
        html.Label('Filtrar por tipo de transacción:'),
        dcc.RadioItems(
            id='eda-fraude-radio',
            options=[
                {'label': 'Todas', 'value': 'all'},
                {'label': 'Fraude', 'value': 1},
                {'label': 'No Fraude', 'value': 0}
            ],
            value='all',
            inline=True
        ),

        dbc.Row([
            dbc.Col(dcc.Graph(id='eda-histograma'), width=4),
            dbc.Col(dcc.Graph(id='eda-boxplot'), width=4),
            dbc.Col(dcc.Graph(id='eda-card6-pie'), width=4)
        ]),

        html.Br(),
        #html.Hr(),
        dbc.Row([
            dbc.Col(
                dbc.Card(
                    dbc.CardBody(
                        html.Div(id='eda-stats-output'),
                        className="text-start",
                        style={
                            "backgroundColor": "#e7f1fb",
                            "padding": "10px 15px",
                            "border": "1px solid #cfe2ff",
                            "borderRadius": "8px",
                            "boxShadow": "0 0 4px rgba(0,0,0,0.1)",
                            "fontSize": "0.9rem",
                            "maxWidth": "100%"
                        }
                    ),
                    style={"backgroundColor": "transparent", "border": "none"}
                ),
                width=2
            ),
            dbc.Col([
                html.H5("Distribución temporal de transacciones por hora virtual", className="mt-4"),
                dcc.Graph(figure=fig_transacciones_hora, id='grafico-transacciones-hora')], width=10)

        ]),


        
    ]),
    dcc.Tab(label='b. EDA 2', children=[
        html.H4('b. EDA 2 - Análisis adicional'),
        html.H5("Distribución de clases en la variable objetivo"),
        dcc.Graph(figure=fig_balance_clases),
    ]),
    dcc.Tab(label='c. Visualización del Modelo', children=[
        html.H4('c. Visualización de Resultados del Modelo'),

        # Mostrar tabla de métricas
        dash_table.DataTable(
            data=metrics_df.to_dict('records'),
            columns=[{"name": i, "id": i} for i in metrics_df.columns]
        ),

        html.Br(),
        html.Div([
            dcc.Graph(figure=fig_cm, style={'width': '48%', 'display': 'inline-block'}),
            dcc.Graph(figure=fig_disp, style={'width': '48%', 'display': 'inline-block'})
        ], style={'display': 'flex', 'justifyContent': 'space-between'}),

        html.Div([
            dcc.Graph(figure=fig_roc, style={'width': '48%', 'display': 'inline-block'}),
            dcc.Graph(figure=fig_imp, style={'width': '48%', 'display': 'inline-block'})
        ], style={'display': 'flex', 'justifyContent': 'space-between'}),


        html.Ul([
            html.Li('Gráficas de comparación: valores reales vs. predichos'),
            html.Li('Análisis de residuales')
        ])
    ]),
    dcc.Tab(label='d. Indicadores del Modelo', children=[
        html.H4('d. Indicadores de Evaluación del Modelo'),
        html.Ul([
            html.Li(' Tabla de errores: RMSE, MAE, MSE, etc.'),
            html.Li(' Interpretación de los valores para comparar modelos')
        ])
    ]),
    dcc.Tab(label='e. Limitaciones', children=[
        html.H4('e. Limitaciones y Consideraciones Finales'),
        html.Ul([
            html.Li('Restricciones del análisis'),
            html.Li('Posibles mejoras futuras')
        ])
    ])
])


tabs = [
    dcc.Tab(label='1. Introducción', children=[
        html.H2('Introducción'),
        html.P('Aquí se presenta una visión general del contexto de la problemática, el análisis realizado y los hallazgos encontrados.'),
        html.P('De manera resumida, indicar lo que se pretende lograr con el proyecto')
    ]),
    dcc.Tab(label='2. Contexto', children=[
        html.H2('Contextualización de los datos'),

        html.Div([
                html.Img(src='/assets/fraud.png', style={
                    'width': '400px',
                    'marginBottom': '10px'
                }),
            html.P('Los datos provienen de transacciones reales de comercio electrónico registradas por la empresa Vesta.'),
            html.P('El objetivo es identificar patrones de comportamiento que permitan detectar posibles fraudes, optimizando la eficacia de los sistemas de alerta y reduciendo falsos positivos.'),
            html.P('Esto no solo mejora la seguridad financiera de los usuarios, sino que también incrementa la eficiencia operativa de las plataformas comerciales.'),

            html.Br(),
            html.H4('Descripción de Variables Clave'),
            html.Ul([
                html.Li([html.B('TransactionDT'), ': Tiempo relativo de la transacción.']),
                html.Li([html.B('TransactionAMT'), ': Monto total en USD.']),
                html.Li([html.B('ProductCD'), ': Código del producto asociado.']),
                html.Li([html.B('card1 - card6'), ': Información de la tarjeta utilizada.']),
                html.Li([html.B('addr'), ': Dirección postal.']),
                html.Li([html.B('dist'), ': Distancia entre direcciones.']),
                html.Li([html.B('emaildomain'), ': Dominio del correo del comprador.']),
                html.Li([html.B('C1 – C14'), ': Recuentos y frecuencias codificadas.']),
                html.Li([html.B('D1 – D15'), ': Intervalos temporales codificados.']),
                html.Li([html.B('M1 – M9'), ': Indicadores de coincidencia.']),
                html.Li([html.B('Vxxx'), ': Variables generadas por Vesta.']),
                html.Li([html.B('DeviceType / DeviceInfo'), ': Información técnica del dispositivo.']),
                html.Li([html.B('id_12 – id_38'), ': Identificadores de red e identidad.']),
            ])

        ], style={
            'backgroundColor': '#f0f2f5',
            'padding': '20px',
            'borderRadius': '10px',
            'boxShadow': '0 2px 5px rgba(0,0,0,10)',
            'marginTop': '20px'
        }),
    ]),
    dcc.Tab(label='3. Planteamiento del Problema', children=[
        html.H2('Planteamiento del Problema'),
        html.P('Describe en pocas líneas la problemática abordada.'),
        html.P('Pregunta problema: ¿Cuál es la pregunta que intenta responder el análisis?')
    ]),
    dcc.Tab(label='4. Objetivos y Justificación', children=[
        html.H2('Objetivos y Justificación'),
        html.H4('Objetivo General'),
        html.Ul([html.Li('Objetivo general del proyecto')]),
        html.H4('Objetivos Específicos'),
        html.Ul([
            html.Li('Objetivo específico 1'),
            html.Li('Objetivo específico 2'),
            html.Li('Objetivo específico 3')
        ]),
        html.H4('Justificación'),
        html.P('Explicación breve sobre la importancia de abordar el problema planteado y los beneficios esperados.')
    ]),
    dcc.Tab(label='5. Marco Teórico', children=[
        html.H2('Marco Teórico'),
        html.P('Resumen de conceptos teóricos (definiciones formales) claves relacionados con el proyecto. Se pueden incluir referencias o citas.')
    ]),
    dcc.Tab(label='6. Metodología', children=[
        html.H2('Metodología'),
        subtabs_metodologia
    ]),
    dcc.Tab(label='7. Resultados y Análisis Final', children=[
        html.H2('Resultados y Análisis Final'),
        subtabs_resultados
    ]),
    dcc.Tab(label='8. Conclusiones', children=[
        html.H2('Conclusiones'),
        html.Ul([
            html.Li('Listar los principales hallazgos del proyecto'),
            html.Li('Relevancia de los resultados obtenidos'),
            html.Li('Aplicaciones futuras y recomendaciones')
        ])
    ])
]


app.layout = dbc.Container([
    html.Div([
        html.H1("Análisis y Detección de Fraudes en Transacciones Electrónicas", style={
            'color': 'white',
            'textAlign': 'center',
            'padding': '20px',
            'fontWeight': 'bold'
        })
    ], style={'backgroundColor': '#111111'}),

    dcc.Tabs(tabs)
], fluid=True)

# === Callback interactividad EDA ===
@app.callback(
    [Output('eda-histograma', 'figure'),
     Output('eda-boxplot', 'figure'),
     Output('eda-stats-output', 'children'),
     Output('eda-card6-pie', 'figure')],
    [Input('eda-variable-dropdown', 'value'),
     Input('eda-fraude-radio', 'value')]
)
def actualizar_eda(variable, filtro):
    if filtro == 'all':
        df_filtrado = df.copy()
    else:
        df_filtrado = df[df['isfraud'] == int(filtro)]

    df_filtrado[variable] = pd.to_numeric(df_filtrado[variable], errors='coerce')

    if df_filtrado[variable].dropna().empty:
        return {}, {}, html.Ul([html.Li("No hay datos numéricos válidos para esta variable.")])

    fig_hist = px.histogram(df_filtrado, x=variable, nbins=40,
                            title=f'Distribución de {variable}')
    fig_box = px.box(df_filtrado, y=variable, points='outliers',
                     title=f'Boxplot de {variable}')
    
    fig_violin = px.violin(
        df_filtrado,
        y=variable,
        color="isfraud",
        box=True,
        points="all",
        title=f'Distribución de {variable} según tipo de transacción',
        color_discrete_map={0: "blue", 1: "red"})
    

    # Pie chart de card6
    card6_counts = df_filtrado['card6'].value_counts(normalize=True).reset_index()
    card6_counts.columns = ['card6', 'proporcion']

    fig_card6 = px.pie(card6_counts, names='card6', values='proporcion',
                    title='Distribución de tipo de tarjeta (card6)',
                    hole=0.3)
    fig_card6.update_traces(textinfo='percent+label')



    stats = df_filtrado[variable].describe().round(2)
    resumen = dbc.Card(
    dbc.CardBody([
        html.H6("Resumen estadístico", className="fw-bold mb-3"),
        html.P(f"Cuenta: {stats['count']}"),
        html.P(f"Media: {stats['mean']}"),
        html.P(f"Desviación estándar: {stats['std']}"),
        html.P(f"Mínimo: {stats['min']}"),
        html.P(f"Q1: {stats['25%']}"),
        html.P(f"Mediana: {stats['50%']}"),
        html.P(f"Q3: {stats['75%']}"),
        html.P(f"Máximo: {stats['max']}")
    ]),
    style={
        "backgroundColor": "#e7f1fb",
        "borderRadius": "8px",
        "fontSize": "14px",
        "lineHeight": "1.6",
        "boxShadow": "none",
        "border": "1px solid #cfe2ff"
    }
)




    return fig_hist, fig_box, resumen, fig_card6





if __name__ == '__main__':
    port = int(os.environ.get('PORT', 8050))
    app.run(debug=False, host="0.0.0.0", port=port)

