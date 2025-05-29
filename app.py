import dash
from dash import dcc, html
import dash_bootstrap_components as dbc
import os
import pandas as pd
from sqlalchemy import create_engine
from dash import Input, Output
import plotly.express as px

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
app.title = "Dashboard Detección de Fraude"
server = app.server  


subtabs_metodologia = dcc.Tabs([
    dcc.Tab(label='a. Definición del Problema', children=[
        dbc.Card([
            dbc.CardBody([
                html.H4("🔍 Tipo de problema y variable objetivo", className="mb-3"),
                html.P("""
                    El presente estudio enfrenta un problema de clasificación supervisada. El objetivo es construir un modelo que prediga 
                    si una transacción en línea es fraudulenta (`isfraud=1`) o legítima (`isfraud=0`). El reto fundamental radica en la 
                    detección precisa de la clase minoritaria, que representa apenas el 3.5% de las observaciones totales.
                """, style={'fontSize': '1.2rem', 'lineHeight': '2'}),
                html.P("""
                    La relevancia de este problema es crítica, ya que permite anticipar pérdidas económicas y mejorar la seguridad 
                    transaccional de los usuarios en plataformas de comercio electrónico como las gestionadas por Vesta.
                """, style={'fontSize': '1.2rem', 'lineHeight': '2'})
            ])
        ], className="mt-4 mb-4", style={'backgroundColor': '#f8f9fa', 'borderRadius': '10px'})
    ]),

    dcc.Tab(label='b. Preparación de Datos', children=[
        dbc.Card([
            dbc.CardBody([
                html.H4("🧹 Limpieza y transformación de datos", className="mb-3"),
                html.P("""
                    Se partió de la unión de los archivos `train_transaction.csv` y `train_identity.csv` mediante `TransactionID`, 
                    consolidando un dataset con 434 variables. Se eliminaron aquellas con más del 50% de datos nulos, reduciendo 
                    el total a 73 columnas útiles.
                """, style={'fontSize': '1.2rem', 'lineHeight': '2'}),
                html.P("""
                    La imputación se realizó en dos fases. Las variables numéricas se completaron mediante `KNNImputer`, que tomó en cuenta 
                    la similitud multivariable entre registros. Las variables categóricas, con menos del 2.5% de faltantes, se completaron 
                    con la moda. Finalmente, se aplicó la prueba VIF y chi-cuadrado para eliminar multicolinealidad y redundancia.
                """, style={'fontSize': '1.2rem', 'lineHeight': '2'}),
                html.H4("📊 División del dataset", className="mt-4 mb-3"),
                html.P("""
                    Se utilizó una división estratificada de 70/30 para entrenamiento y prueba, preservando la proporción de fraudes. 
                    Sobre el conjunto de entrenamiento se aplicó SMOTETomek para balancear la clase positiva. Luego, las variables numéricas 
                    se escalaron mediante `StandardScaler`.
                """, style={'fontSize': '1.2rem', 'lineHeight': '2'})
            ])
        ], className="mt-4 mb-4", style={'backgroundColor': '#f8f9fa', 'borderRadius': '10px'})
    ]),

    dcc.Tab(label='c. Selección del Modelo', children=[
        dbc.Card([
            dbc.CardBody([
                html.H4("📌 Modelos implementados", className="mb-3"),
                html.P("""
                    Se exploraron múltiples modelos de clasificación como benchmark inicial: Regresión Logística, KNN, Naive Bayes, Árboles 
                    de Decisión, Random Forest, XGBoost y SVM. Cada uno se entrenó sobre el conjunto balanceado y se evaluó con validación 
                    cruzada estratificada (5 folds).
                """, style={'fontSize': '1.2rem', 'lineHeight': '2'}),
                html.H4("💡 Modelo original propuesto", className="mt-4 mb-3"),
                html.P("""
                    Se construyó un `StackingClassifier` con Random Forest, XGBoost y KNN como clasificadores base, y Regresión Logística 
                    como meta-modelo. Esta arquitectura permite capturar la diversidad de los clasificadores base y mejorar la capacidad 
                    de generalización sobre nuevas transacciones.
                """, style={'fontSize': '1.2rem', 'lineHeight': '2'}),
                html.P("""
                    El modelo fue encapsulado en un pipeline completo con GridSearchCV para ajustar hiperparámetros y facilitar su 
                    despliegue. Este enfoque modular garantiza replicabilidad y escalabilidad del sistema en entornos reales.
                """, style={'fontSize': '1.2rem', 'lineHeight': '2'})
            ])
        ], className="mt-4 mb-4", style={'backgroundColor': '#f8f9fa', 'borderRadius': '10px'})
    ]),

    dcc.Tab(label='d. Evaluación del Modelo', children=[
        dbc.Card([
            dbc.CardBody([
                html.H4("📏 Métricas de desempeño y validación", className="mb-3"),
                html.P("""
                    Dada la fuerte desproporción entre clases, se utilizaron métricas robustas a desbalance como:
                """, style={'fontSize': '1.2rem', 'lineHeight': '2'}),
                html.Ul([
                    html.Li("🔹 Precisión (Precision): ¿Qué tan fiables son los positivos predichos?"),
                    html.Li("🔹 Exhaustividad (Recall): ¿Qué tanto del fraude real fue detectado?"),
                    html.Li("🔹 F1-Score: Equilibrio entre precisión y recall."),
                    html.Li("🔹 AUC-ROC: Discriminación global del modelo.")
                ], style={'fontSize': '1.2rem', 'lineHeight': '2'}),
                html.P("""
                    El modelo final de stacking alcanzó un AUC superior a 0.95, superando los benchmarks individuales. 
                    Las curvas ROC, matrices de confusión y reportes de clasificación se incluyeron en el apartado de resultados 
                    para demostrar la solidez del enfoque.
                """, style={'fontSize': '1.2rem', 'lineHeight': '2'})
            ])
        ], className="mt-4 mb-4", style={'backgroundColor': '#f8f9fa', 'borderRadius': '10px'})
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

# para crear la figura de las horas

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
        html.P('Aquí puedes incluir análisis exploratorios complementarios como segmentaciones, boxplots, histogramas comparativos o mapas si aplica.')
    ]),
    dcc.Tab(label='c. Visualización del Modelo', children=[
        html.H4('c. Visualización de Resultados del Modelo'),
        html.P('Aquí se mostrarán las métricas de evaluación de los modelos en forma de tabla.'),
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
    dcc.Tab(label='🧠 Introducción', children=[
        dbc.Row([
            dbc.Col(html.Img(
                src='/assets/imgIntroduccion_1.png',
                style={
                    'width': '100%',
                    'height': '100%',
                    'objectFit': 'cover',
                    'borderRadius': '10px',
                    'boxShadow': '0 4px 8px rgba(0,0,0,0.1)'
                }
            ), width=6),

            dbc.Col(
                dbc.Card([
                    dbc.CardBody([
                        html.P("""
                            Este proyecto tiene como objetivo desarrollar un sistema de detección de fraudes en transacciones de comercio electrónico, 
                            utilizando técnicas de aprendizaje automático y análisis exploratorio de datos. 
                            A partir de un conjunto de datos reales proporcionado por la plataforma Vesta, se aplica una metodología rigurosa 
                            que abarca desde la limpieza y exploración de los datos hasta la construcción y evaluación de modelos predictivos.
                        """),

                        html.P("""
                            La estructura del análisis incluye etapas clave como la depuración del conjunto de datos, análisis exploratorio (EDA), 
                            implementación de modelos de clasificación benchmark, comparación de pipelines y la propuesta final de un modelo original.
                        """),

                        html.P("""
                            El resultado se presenta en este dashboard interactivo, diseñado para facilitar la comprensión de los datos, 
                            explorar las características relevantes y visualizar los resultados del modelado, brindando así una herramienta 
                            efectiva para la toma de decisiones frente al fraude en entornos financieros digitales.
                        """)
                    ], style={
                        'fontSize': '1.2rem',
                        'lineHeight': '2',
                        'textAlign': 'justify'
                    })
                ], style={
                    'height': '100%',
                    'backgroundColor': '#f8f9fa',
                    'borderRadius': '10px',
                    'padding': '20px',
                    'boxShadow': '0 2px 8px rgba(0,0,0,0.1)'
                }),
                width=6
            )
        ], align='center', style={'minHeight': '500px'})
    ]),
    dcc.Tab(label='📊 Contexto', children=[
        dbc.Row([
            dbc.Col(html.Img(
                src='/assets/imgContexto_1.png',
                style={
                    'width': '100%',
                    'height': '100%',
                    'objectFit': 'cover',
                    'borderRadius': '10px',
                    'boxShadow': '0 4px 8px rgba(0,0,0,0.1)'
                }
            ), md=6),

            dbc.Col(
                dbc.Card([
                    dbc.CardBody([
                        html.H5("📌 Descripción del contexto del proyecto", className="mb-2"),
                        html.P("""
                            Este proyecto tiene como objetivo desarrollar un sistema de detección de fraudes en transacciones de comercio electrónico, 
                            utilizando técnicas de aprendizaje automático y análisis exploratorio de datos. A partir de un conjunto de datos reales 
                            proporcionado por la plataforma Vesta, se aplica una metodología rigurosa que incluye limpieza de datos, análisis descriptivo, 
                            modelado supervisado y propuesta de modelado original.
                        """),

                        html.H5("📁 Fuente de los datos", className="mt-4 mb-2"),
                        html.P("Plataforma Vesta a través de la competencia 'IEEE-CIS Fraud Detection' publicada en Kaggle."),

                        html.H5("📊 Variables de interés", className="mt-4 mb-2"),
                        html.Ul([
                            html.Li("transactionAmt → Monto de la transacción."),
                            html.Li("transactionDT → Tiempo relativo desde el inicio del registro."),
                            html.Li("isFraud → Variable objetivo: 1 = fraude, 0 = no fraude."),
                            html.Li("card1–card6 → Identificadores anonimizados de medios de pago."),
                            html.Li("addr1, addr2 → Ubicación aproximada del usuario."),
                            html.Li("DeviceType / DeviceInfo → Dispositivo utilizado."),
                            html.Li("emaildomain → Dominio de correo del comprador o vendedor.")
                        ])
                    ], style={'fontSize': '1.1rem', 'lineHeight': '2', 'textAlign': 'justify'})
                ], style={
                    'height': '100%',
                    'backgroundColor': '#f8f9fa',
                    'borderRadius': '10px',
                    'padding': '20px',
                    'boxShadow': '0 2px 8px rgba(0,0,0,0.1)'
                }),
                md=6
            )
        ], align='center', style={'minHeight': '550px'})
    ]),
    dcc.Tab(label='📌 Planteamiento del Problema', children=[
        dbc.Container([
            dbc.Card([
                dbc.CardBody([
                    html.H3("Planteamiento del Problema", className="mb-4"),

                    html.P("""
                        El creciente volumen de transacciones electrónicas en el comercio digital ha generado un entorno propicio 
                        para que actores maliciosos desarrollen estrategias cada vez más sofisticadas de fraude financiero. 
                        Este fenómeno representa una amenaza directa tanto para usuarios como para instituciones bancarias 
                        y plataformas de pago.
                    """),

                    html.P("""
                        Los métodos tradicionales de detección de fraude suelen ser insuficientes, debido a su rigidez frente 
                        al dinamismo de los esquemas fraudulentos. Por esta razón, surge la necesidad de aplicar técnicas más robustas 
                        y adaptativas, como el aprendizaje automático, que permitan identificar patrones sutiles en los datos 
                        y distinguir de manera efectiva entre transacciones legítimas y fraudulentas.
                    """),

                    html.H5("❓ Pregunta problema", className="mt-4"),
                    html.P(
                        "¿Cómo identificar con precisión transacciones electrónicas fraudulentas utilizando técnicas de aprendizaje automático aplicadas sobre datos reales de comercio electrónico?",
                        style={
                            'fontStyle': 'italic',
                            'fontSize': '1.15rem',
                            'color': '#333'
                        }
                    )
                ], style={
                    'fontSize': '1.2rem',
                    'lineHeight': '2',
                    'textAlign': 'justify',
                    'padding': '25px'
                })
            ], style={
                'backgroundColor': '#f8f9fa',
                'borderRadius': '10px',
                'boxShadow': '0 2px 8px rgba(0,0,0,0.1)',
                'marginTop': '20px'
            })
        ])
    ]),
    dcc.Tab(label='🎯 Objetivos y Justificación', children=[
        dbc.Row([
            dbc.Col(
                dbc.Card([
                    dbc.CardBody([
                        html.H4("Objetivo General", className="mb-2"),
                        html.P("""
                            Diseñar un modelo de clasificación que permita detectar transacciones electrónicas fraudulentas utilizando técnicas de aprendizaje automático aplicadas a un conjunto de datos reales.
                        """),

                        html.H4("Objetivos Específicos", className="mt-4 mb-2"),
                        html.Ul([
                            html.Li("Explorar y preparar el conjunto de datos mediante limpieza, imputación y transformación de variables."),
                            html.Li("Entrenar un modelo de clasificación supervisado para predecir si una transacción es fraudulenta."),
                            html.Li("Comparar el desempeño de distintos algoritmos (Random Forest, XGBoost, LightGBM) usando métricas de evaluación."),
                            html.Li("Implementar un dashboard interactivo que facilite la visualización del modelo y sus resultados.")
                        ]),

                        html.H4("Justificación", className="mt-4 mb-2"),
                        html.P("""
                            La detección de fraude en transacciones electrónicas es un reto prioritario en la seguridad financiera digital. 
                            La aplicación de modelos de aprendizaje automático permite automatizar esta tarea con alta precisión, adaptándose 
                            al comportamiento cambiante de los defraudadores. Este proyecto busca ofrecer una solución práctica y escalable 
                            para anticipar riesgos mediante análisis predictivo.
                        """)
                    ], style={
                        'fontSize': '1.15rem',
                        'lineHeight': '2',
                        'textAlign': 'justify',
                        'padding': '25px'
                    })
                ], style={
                    'backgroundColor': '#f8f9fa',
                    'borderRadius': '10px',
                    'boxShadow': '0 2px 8px rgba(0,0,0,0.1)',
                    'height': '100%'
                }),
                md=6
            ),

            dbc.Col(html.Img(
                src='/assets/imgObjetivos_1.png',
                style={
                    'width': '100%',
                    'height': '100%',
                    'objectFit': 'contain',
                    'borderRadius': '10px',
                    'boxShadow': '0 4px 8px rgba(0,0,0,0.1)',
                    'padding': '10px'
                }
            ), md=6)
        ], align='center', style={'minHeight': '550px'})
    ]),
    dcc.Tab(label='📚 Marco Teórico', children=[
        dbc.Container([

            # Bloque 1: Fraude financiero
            dbc.Card([
                dbc.CardBody([
                    html.H4("🕵️‍♂️ Fraude financiero y su impacto", className="mb-3"),
                    html.P("""
                        El fraude financiero en plataformas digitales representa una amenaza significativa para usuarios, entidades 
                        comerciales y sistemas de pago. Este tipo de fraude ocurre cuando se realizan transacciones de manera ilegítima, 
                        generalmente con tarjetas robadas o suplantación de identidad. La detección oportuna de estas acciones permite 
                        mitigar pérdidas económicas y preservar la confianza de los usuarios en los servicios digitales.
                    """, style={'fontSize': '1.2rem', 'lineHeight': '2'})
                ])
            ], className="mb-4", style={'backgroundColor': '#f8f9fa', 'borderRadius': '10px', 'padding': '15px'}),

            # Bloque 2: Aprendizaje automático
            dbc.Card([
                dbc.CardBody([
                    html.H4("🤖 Aprendizaje automático", className="mb-3"),
                    html.P("""
                        El aprendizaje automático es un subcampo de la inteligencia artificial que busca construir algoritmos capaces 
                        de aprender patrones a partir de los datos, sin estar explícitamente programados para cada tarea. 
                        Estos algoritmos identifican regularidades estadísticas y las utilizan para realizar predicciones o 
                        clasificaciones sobre nuevas observaciones.
                    """, style={'fontSize': '1.2rem', 'lineHeight': '2'}),
                    html.P("""
                        En este proyecto, se aplican técnicas de aprendizaje supervisado, donde el modelo aprende a partir 
                        de un conjunto de datos etiquetado en el que se conoce si la transacción fue fraudulenta o no.
                    """, style={'fontSize': '1.2rem', 'lineHeight': '2'})
                ])
            ], className="mb-4", style={'backgroundColor': '#f8f9fa', 'borderRadius': '10px', 'padding': '15px'}),

            # Bloque 3: Modelos de clasificación
            dbc.Card([
                dbc.CardBody([
                    html.H4("📊 Modelos de clasificación", className="mb-3"),
                    html.P("""
                        Los modelos de clasificación se utilizan para predecir categorías discretas. En el caso del fraude, 
                        se trata de un problema binario: clasificar si una transacción es o no fraudulenta.
                    """, style={'fontSize': '1.2rem', 'lineHeight': '2'}),
                    html.P("""
                        Este proyecto emplea varios algoritmos como Regresión Logística, K-Nearest Neighbors (KNN), 
                        Árboles de Decisión, Random Forest, XGBoost y Máquinas de Soporte Vectorial (SVM).
                    """, style={'fontSize': '1.2rem', 'lineHeight': '2'}),
                    html.P("""
                        Dado el alto desbalance del conjunto de datos (solo el 8% de las transacciones son fraude), 
                        se aplicaron técnicas como SMOTE y SMOTETomek para equilibrar las clases antes del entrenamiento.
                    """, style={'fontSize': '1.2rem', 'lineHeight': '2'})
                ])
            ], className="mb-4", style={'backgroundColor': '#f8f9fa', 'borderRadius': '10px', 'padding': '15px'}),

            # Bloque 4: Evaluación de modelos
            dbc.Card([
                dbc.CardBody([
                    html.H4("📐 Evaluación del desempeño de los modelos", className="mb-3"),
                    html.P("""
                        La evaluación de modelos en problemas desbalanceados debe ir más allá de la simple precisión. 
                        En este proyecto se utilizan métricas clave como:
                    """, style={'fontSize': '1.2rem', 'lineHeight': '2'}),
                    html.Ul([
                        html.Li("Precisión (Precision): proporción de verdaderos fraudes entre los predichos como fraudes."),
                        html.Li("Recall (Sensibilidad): proporción de fraudes correctamente detectados."),
                        html.Li("F1-Score: media armónica entre precisión y recall."),
                        html.Li("AUC-ROC: área bajo la curva ROC, que mide la capacidad del modelo para discriminar entre clases.")
                    ], style={'fontSize': '1.2rem', 'lineHeight': '2'}),
                    html.P("""
                        Estas métricas permiten evaluar si el modelo no solo acierta en general, sino si detecta eficazmente 
                        los casos positivos de fraude.
                    """, style={'fontSize': '1.2rem', 'lineHeight': '2'})
                ])
            ], className="mb-5", style={'backgroundColor': '#f8f9fa', 'borderRadius': '10px', 'padding': '15px'})
        ])
    ]),
    dcc.Tab(label='⚙️ Metodología', children=[
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
    html.H1("Dashboard Detección de Fraude ", className="text-center my-4"),
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

