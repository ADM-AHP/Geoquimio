# Bibliotecas padrão do Python
import io
import json
from io import StringIO
import os
import traceback

# Visualização e Gráficos
import plotly.express as px
import plotly.graph_objects as go
import plotly.figure_factory as ff
import dash_ag_grid as dag
import dash_bootstrap_components as dbc

# Manipulação de Dados
import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.stats import pearsonr
from scipy.stats.mstats import zscore, winsorize

# Machine Learning e Modelagem
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.metrics import silhouette_score, silhouette_samples, confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.mixture import GaussianMixture
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.svm import SVC
from umap import UMAP
from imblearn.over_sampling import SMOTE, RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler
from sklearn.model_selection import cross_val_score


# Framework Dash
import dash
from dash import html, dcc, callback, dash_table
from dash.dependencies import Input, Output, State
from dash.exceptions import PreventUpdate
import dash_ag_grid as dag
from dash_bootstrap_templates import ThemeSwitchAIO




import json

from io import StringIO
import io
import base64
import joblib

# Módulos Locais
from app import *






# ========== Styles ============ #
tab_card = {'height': '100%'}

main_config = {
    "hovermode": "x unified",
    "legend": {"yanchor": "top", 
               "y": 0.9, 
               "xanchor": "left",
               "x": 0.1,
               "title": {"text": None},
               "font": {"color": "white"},
               "bgcolor": "rgba(0,0,0,0.5)"},
    "margin": {"l": 10, "r": 10, "t": 10, "b": 10}
}

config_graph = {"displayModeBar": False, "showTips": False}

template_theme1 = "flatly"
template_theme2 = "darkly"
url_theme1 = dbc.themes.FLATLY
url_theme2 = dbc.themes.DARKLY

material_palette = [
    "#4285F4",  # Azul Google
    "#DB4437",  # Vermelho
    "#F4B400",  # Amarelo
    "#0F9D58",  # Verde
    "#AB47BC",  # Roxo
    "#FF7043",  # Laranja
    "#03A9F4"   # Azul claro
]

custom_palette = [
    "#1f77b4",  # Azul
    "#ff7f0e",  # Laranja
    "#2ca02c",  # Verde
    "#d62728",  # Vermelho
    "#9467bd",  # Roxo
    "#8c564b",  # Marrom
    "#e377c2",  # Rosa
    "#7f7f7f",  # Cinza
    "#bcbd22",  # Verde-amarelado
    "#17becf"   # Azul-ciano
]

outra = [
    "#1F77B4", 
    "#FF7F0E", 
    "#2CA02C", 
    "#D62728", 
    "#9467BD", 
    "#8C564B", 
    "#E377C2",
    "#7F7F7F", 
    "#BCBD22", 
    "#17BECF"]

top_shap_features = [
    'TRIC/STER', '26/25TRI', 'DIA/C27AA', 'TS/TM', 'TET24/26TRI',
    '%H34', 'Pri/nC17', '29/30H', 'Total Height', '%Saturated'
]
other_important_features = [
    '%13C', '24/25TRI', 'DIA/C27AA', 'DITERP/H30', 'GAM/H30', 'H29/C29TS', 'TPP'
]

# Criar opções para o dropdown
feature_options = [
    {'label': 'Todas as Features', 'value': 'all'},
    {'label': 'Top SHAP Features', 'value': 'shap'},
    {'label': 'Outras Features Importantes', 'value': 'other'}
]

# Carregando os dados
df = pd.read_excel("/Users/fabri/Documents/Codigo/Fabão/TabelaErica_Final.xlsx", decimal= ",")

#missing_columns = df.columns[df.isnull().sum() > 0]
#imputer = SimpleImputer(strategy='median')
#df[missing_columns] = imputer.fit_transform(df[missing_columns])

df
missing_columns = df.columns[df.isnull().sum() > 0]
imputer = SimpleImputer(strategy='median')
df[missing_columns] = imputer.fit_transform(df[missing_columns])
df1 = df[df['%Sulfur'] != df['%Sulfur'].max()]
df1= df1[df1['%Aromatic'] != df1['%Aromatic'].max()]
corr = df1.corr(numeric_only=True)
df1_corr_forte = corr[(corr > 0.6) | (corr < -0.6)]
df1_corr_forte = df1_corr_forte.dropna(axis=0, how='all').dropna(axis=1, how='all')


# ========== Layout ============ #


# Layout da página
layout = dbc.Container(children=[
    dcc.Store(id="normalizacao-data"),
    dcc.Store(id='stored-model-path', storage_type='memory'),  # Armazena o caminho do modelo selecionado
    dcc.Store(id='stored-production-data', storage_type='memory'),  # Armazena os dados de produção carregados
    
    
    # Linha 1
    dbc.Row([
        # Coluna 1: Filtros
        dbc.Col([
            dbc.Card([
                dbc.CardHeader(html.H5("Filtros e Controles")),
                 
                dbc.CardBody([                                   
                # Transformação de Dados
                    dbc.Label("Normalização:", className="fw-bold mb-2"),
                    dbc.RadioItems(
                        id="scale-type2",
                        options=[
                            {"label": " Nenhuma", "value": "none"},
                            {"label": " StandardScaler", "value": "standard"},
                            {"label": " MinMaxScaler", "value": "minmax"},
                            {"label": " RobustScaler", "value": "robust"},
                        ],
                        value="standard",  
                        class_name="mb-3",
                    ),
                      # Armazena os dados transformados
                
                
                    # Tratamento de Outliers
                    dbc.Label("Outliers:", className="fw-bold mb-2"),
                    dbc.RadioItems(
                        id="outlier-treatment",
                        options=[
                            {"label": " Sem tratamento", "value": "none"},
                            {"label": " Remover via Z-Score (|Z| > 3)", "value": "zscore"},
                            {"label": " Winsorizar (1º/99º percentil)", "value": "winsorize"},
                        ],
                        value="none",
                        class_name="mb-3,"  
                    ),
                    # Upload do arquivo CSV
                    dbc.Label("Selecione Um Arquivo CSV:", className="fw-bold mb-2", style={"marginTop": "10px"}),
                    dcc.Upload(
                        id='upload-data',
                        children=html.Button('Carregar CSV', className="btn btn-primary"),
                        multiple=False,
                        className="mb-3",
                    ),
                    
                ], style={"maxHeight": "500px", "overflowY": "auto"}),
            ], style=tab_card)
        ], sm=12, lg=2),  # Ocupa 2 colunas em telas grandes

        
         # Área Principal
        dbc.Col([
            dbc.Tabs([
                # Tab 1: Redução Dimensional
                dbc.Tab(label="Redução Dimensional", children=[
                    dbc.Row([
                        
                        # PCA - Graficos
                        dbc.Col(dbc.Card([
                            dbc.CardHeader(html.H5("PCA - Principal Component Analysis", className="card-title")),
                            
                                dbc.CardBody([
                                    #Linha 1:Controle PCA
                                    dbc.Row([
                                        dbc.Col([
                                            html.Label("n_components (PCA):", className="fw-bold mb-2"),
                                            dcc.Slider(id='n-components-slider', min=2, max=5, step=1, value=2),
                                        ], width=8),   
                                        
                                        html.Label("PCA Explanation:", className="fw-bold mb-2"),
                                        html.Div(id='pca-explanation', className="mt-3 p-3 bg-light rounded") 
                                    ]) ,
                                    # Linha 2: Gráfico PCA 
                                    dbc.Row([
                                        dbc.Col([
                                            dcc.Graph(id='pca-scatter-plot', className='dbc', config=config_graph)
                                        ], width=12),
                                    ],class_name="mb-3"),
                                    
                                    dbc.Row([
                                        dbc.Col([
                                            
                                            dcc.Graph(id='pca-feature-importance', className='dbc',config=config_graph, style={'height': '400px'})  # Altura fixa
                                        ], width=12)
                                    ])                                    
                              
                                ]),
                        ], style=tab_card), width=12, lg=6, sm=12),  # Ocupa 6 colunas em telas grandes
                        
                        
                        
                        
                        # UMAM - Graficos
                        dbc.Col(dbc.Card([
                            dbc.CardHeader(html.H5("UMAP - Uniform Manifold Approximation and Projection", className="card-title")),
                            dbc.CardBody([
                                dbc.Col([
                                    html.Label("Número de Vizinhos (n_neighbors)"),
                                    dcc.Slider(
                                        id='n-neighbors-slider',
                                        min=2,
                                        max=100,
                                        step=1,
                                        value=15,
                                        marks={i: str(i) for i in range(2, 101, 10)}
                                    ),
                                    
                                    html.Label("Distância Mínima (min_dist)"),
                                    dcc.Slider(
                                        id='min-dist-slider',
                                        min=0.0,
                                        max=1.0,
                                        step=0.05,
                                        value=0.1,
                                        marks={i/10: str(i/10) for i in range(0, 11, 2)}
                                    ),
    
                                    
                                
                                    
                                
                                ], width=8),
                                
                                
                                
                                dbc.Row([
                                    dbc.Col(
                                        dcc.Graph(id='umap-graph', className='dbc', config=config_graph),
                                        sm=12, md=6, lg=12
                                    ),
                                ])
                            ]),
                        ], style=tab_card), width=12, lg=6, sm=12)  # Ocupa 6 colunas em telas grandes
                    ],className='g-2 my-auto', style={'margin-top': '7px'}),
                    
                ]),  # Fechamento do dbc.Tab
                
                # Tab 2: Clusterização
                dbc.Tab(label="Clusterização", children=[
                    dbc.Row([
                        dbc.Col([
                            dbc.Card([
                                dbc.CardHeader(html.H5("Dendograma", className="card-title")),
                                dbc.CardBody([
                                    dbc.Row([
                                        dbc.Col([
                                            dbc.Label("Selecione o  método de linkage:", className="fw-bold mb-2"),
                                            dbc.Select(
                                                id="dendrogram-filter",
                                                options=[
                                                    {"label": "ward", "value": "select1"},
                                                    {'label': 'Single', 'value': 'single'},
                                                    {"label": "complete", "value": "select2"},
                                                    {"label": " average", "value": "select3"}
                                                ],
                                                value="ward",
                                                className="mb-3"
                                            ),
                                            
                                            html.Label("Limiar (threshold) de corte:", className="fw-bold mb-2"),
                                            dcc.Slider(
                                            id='threshold-slider',
                                            min=0,
                                            max=70,  # Ajuste conforme a escala das distâncias no seu dendrograma
                                            #step=1,
                                            value=25,  # Valor inicial
                                            marks=None
                                            ),
                                        ], width=4),
                                        
                                        
                                    ]),
                                    dbc.Row([
                                        dbc.Col(
                                            dcc.Graph(id='dendrogram-graph', className='dbc', config=config_graph),
                                            sm=12, md=6, lg=12, width=12
                                        ),
                                    ])
                                ])
                            ],style=tab_card),
                        ], sm=12, lg=12), 
                        
                        dbc.Col([
                            dbc.Card([
                                dbc.CardHeader(html.H5("K-Means e Gaussian Mixture", className="card-title")),
                                dbc.CardBody([
                                    dbc.Row([
                                        dbc.Col([
                                            dbc.Label("Escolha o método de redução dimensional:",className="fw-bold mb-2"),
                                                dbc.RadioItems(
                                                id='dim-reduction-method',
                                                options=[
                                                    {'label': 'PCA', 'value': 'pca'},
                                                    {'label': 'UMAP', 'value': 'umap'}
                                                ],
                                                value='pca',  # Valor inicial
                                                inline=True,
                                                className="mb-3"
                                            ),
                                            dbc.Label("Selecione o  método:", className="fw-bold mb-2"),
                                            dbc.RadioItems(
                                                id="method-filter",
                                                options=[
                                                    {'label': 'K-Means', 'value': 'kmeans'},
                                                    {'label': 'Gaussian Mixture', 'value': 'gmm'},
                                                    
                                                ],
                                                value="kmeans",
                                                inline=True,
                                                className="mb-3"
                                            ),
                                            
                                            html.Label("Numero de Clusters:", className="fw-bold mb-2"),
                                            dcc.Slider(
                                            id='n-clusters-slider',
                                            min=2,
                                            max=10,  # Ajuste conforme a escala das distâncias no seu dendrograma
                                            step=1,
                                            value=7,  # Valor inicial
                                            #marks=
                                            )
                                        ], width=6),
                                        
                                        
                                    ]),
                                    dbc.Row([
                                        dbc.Col(
                                            dcc.Graph(id='cluster-graph', className='dbc', config=config_graph),
                                            sm=12, md=6, lg=6, width=12
                                        ),
                                        dbc.Col([
                                            html.Div('Silhouette', className='dbc'),
                                            dcc.Graph(id='elbow-graph', className='dbc', config=config_graph)
                                            
                                        ],sm=12, md=6, lg=6, width=12), 
                                        
                                    ]),
                                    
                                ])
                            ],style=tab_card),
                        ], sm=12, lg=12), # Ocupa 9 colunas em telas grandes
                    ],className='g-2 my-auto', style={'margin-top': '7px'}),
                ]),  # Fechamento do dbc.Tab
                
                
                # Tab 3: Classificação
                dbc.Tab(label="Classificação", children=[
                    dbc.Row([
                        dbc.Card([
                               dbc.Col(html.H3("Modelos Pré-Treinado para Análise de Dados", 
                                        style={'textAlign': 'center', 'color': '#2C3E50'}), width=12)
                                 
                            ]),
                        dbc.Col([
                            
                            dbc.Card([
                                dbc.CardHeader(html.H5("Modelos de Classificação", className="card-title")),
                                dbc.CardBody([
                                    dbc.Row([
                                        dbc.Col([
                                            html.Label("Selecione o Modelo:"),
                                            dcc.Dropdown(
                                                id='classification-method',
                                                options=[
                                                    {'label': 'Random Forest', 'value': 'rf'},
                                                    {'label': 'Arvore de Decisão', 'value': 'dt'},
                                                    {'label': 'SVM', 'value': 'svm'},
                                                    {'label': 'Rede Neural', 'value': 'neural'}
                                                ],
                                                value='rf',
                                                clearable=False
                                            ),
                                        ], width=3,style={"padding": "9px"}),
                                        
                                        dbc.Col([
                                            html.Label("Método de Balanceamento:"),
                                            dcc.Dropdown(
                                                id='balance-method',
                                                options=[
                                                    {'label': 'Sem balanceamento', 'value': 'none'},
                                                    {'label': 'SMOTE ', 'value': 'smote'},
                                                    {'label': 'Undersampling', 'value': 'undersample'},
                                                    {'label': 'Oversampling', 'value': 'oversample'}
                                                ],
                                                value='none',
                                                clearable=False
                                            ),
                                        ], width=3, style={"padding": "9px"}),
      
                                        
                                    ]),

                                    dbc.Col([
                                        dbc.Row([
                                            html.Label("Métricas do Modelo Pré-Treinado", className="fw-bold mb-2"),
                                                html.Div(
                                                    id="model-metrics",
                                                    style={
                                                        "padding": "9px"}
                                                     #   "border": "1px solid #ddd",
                                                       # "borderRadius": "5px",
                                                     #   "backgroundColor": "#f8f9fa",
                                                      #  "textAlign": "center",
                                                    #    "fontWeight": "bold",
                                                     #   "fontSize": "16px"
                                                    ),
                                                
                                                
                                        ]),
                                        dbc.Row([
                                            html.Label("Análise de Erros", className="fw-bold mb-2"),
                                                html.Div(
                                                    id="error-analysis",
                                                    style={
                                                        "padding": "9px"}
                                                     #   "border": "1px solid #ddd",
                                                       # "borderRadius": "5px",
                                                     #   "backgroundColor": "#f8f9fa",
                                                      #  "textAlign": "center",
                                                    #    "fontWeight": "bold",
                                                     #   "fontSize": "16px"
                                                    ),
                                                
                                                
                                        ]),
                          
                                                 
                                            
                                        
                                    ], width=12, ),

                                                                                                        
                                ])
                            ],style=tab_card),
                        ],width=7),

                        dbc.Col([
                            dbc.Card([
                                dbc.CardHeader(html.H5('Matriz de Confusão', className="card-title")),
                                    dbc.Col([
                                        dcc.Graph(id='confusion-matrix', className='dbc', config=config_graph),
                                       
                                        
                                     ])
                                
                            ], style=tab_card)
                        ], width=5),
                    # Seção de Upload
                    dbc.Col([
                        dbc.Card([
                            dbc.CardHeader(
                                html.H5("Seção de Upload de Dados de Produção", className="card-title"),
                                style={"backgroundColor": "#f7f7f7", "borderBottom": "1px solid #ddd"}
                            ),
                            dbc.CardBody([
                                dcc.Upload(
                                    id='upload-data',
                                    children=html.Div(['Arraste e Solte ou ', html.A('Selecione um Arquivo')]),
                                    style={
                                        'width': '100%',
                                        'height': '60px',
                                        'lineHeight': '60px',
                                        'borderWidth': '1px',
                                        'borderStyle': 'dashed',
                                        'borderRadius': '5px',
                                        'textAlign': 'center',
                                        'margin': '10px',
                                        'backgroundColor': '#f9f9f9'
                                    },
                                    multiple=False
                                ),
                                html.Div(id='output-data-upload', style={'marginTop': '10px'})
                            ])
                        ], className="shadow-sm")
                    ], sm=12, lg=12),  # Responsivo: ocupa 12 colunas em telas pequenas e grandes
                
                    
                    # Seção de Previsões
                    # Seção de Previsões
                    dbc.Row([
                        dbc.Col([
                            dbc.Card([
                                dbc.CardBody([
                                    html.Button(
                                        'Gerar Previsões', 
                                        id='predict-button', 
                                        n_clicks=0,
                                        style={
                                            'width': '100%',
                                            'padding': '10px',
                                            'backgroundColor': '#3498DB',
                                            'color': '#fff',
                                            'border': 'none',
                                            'borderRadius': '5px',
                                            'fontWeight': 'bold'
                                        }
                                    ),
                                    html.Div(id='prediction-output', style={'marginTop': '15px'})
                                ])
                            ], className="shadow-sm")
                        ], sm=12)
                    ], className="mb-4"),
                    
                    dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardHeader(
                    html.H5("Dados Filtrados", className="card-title"),
                    style={"backgroundColor": "#f7f7f7", "borderBottom": "1px solid #ddd"}
                ),
                dbc.CardBody(
                    dag.AgGrid(
                        id="data-table2",
                        columnDefs=[],  # Colunas serão definidas dinamicamente
                        rowData=[],     # Dados serão definidos dinamicamente
                        defaultColDef={"sortable": True, "filter": True, "resizable": True},
                        style={"height": "400px"},
                        dashGridOptions={"pagination": True}
                    )
                )
            ], className="mt-3 shadow-sm")
        ], sm=12)
    ])                  
                   
                   
                   
                    ],className="g-2 my-auto",style={'margin-top': '7px'})
                ])
            ])


        ], sm=12, lg=10),  # Ocupa 9 colunas em telas grandes
    ], className='g-2 my-auto', style={'margin-top': '7px'}),
  
  
    


], fluid=True, style={'height': '100vh'})



@callback(
    Output("normalizacao-data", "data"),
    Input("scale-type2", "value"),
    Input('outlier-treatment', 'value'),
    State('normalizacao-data', 'data'),
    
    
)
def apply_transformations(scale_type2, outlier, data):
    df_transformed = df.copy()
    
    # Identificar colunas numéricas (para aplicar transformações)
    numeric_cols = df_transformed.select_dtypes(include=np.number).columns.drop('ID')
    
    if outlier == 'zscore':
        z_scores = df_transformed[numeric_cols].apply(zscore)
        df_transformed = df_transformed[(z_scores.abs() < 3).all(axis=1)]
    elif outlier == 'winsorize':
        for col in numeric_cols:
            df_transformed[col] = winsorize(df_transformed[col], limits=[0.01, 0.01])
    
                 
    # Transformação de escala
    if scale_type2 == "standard":
        scaler = StandardScaler()
        df_transformed[numeric_cols] = scaler.fit_transform(df_transformed[numeric_cols])
        
    elif scale_type2 == "minmax":
        scaler_m = MinMaxScaler()
        df_transformed[numeric_cols] = scaler_m.fit_transform(df_transformed[numeric_cols]) 
    
    elif scale_type2 == "robust":
        scaler_r = RobustScaler()
        df_transformed[numeric_cols] = scaler_r.fit_transform(df_transformed[numeric_cols])
    else:  # original
        df_scaler = df_transformed
    
    return df_transformed.to_dict('records')



   # Redução dimensional - PCA
@callback(
    Output('pca-scatter-plot', 'figure'),
    Output('pca-feature-importance', 'figure'),
    Output('pca-explanation', 'children'),
    Input('n-components-slider', 'value'),
    Input('scale-type2', 'value'),
    Input('normalizacao-data', 'data'),
    
    
    
)
def update_pca_graphs(n_components, scale_type, normalized_data):
   
    
    
    filtered_df = pd.DataFrame.from_records(normalized_data)
    data_numeric = filtered_df.select_dtypes(include=np.number).drop('ID', axis=1, errors='ignore')

    # Aplicar PCA
    pca = PCA(n_components=n_components)
    pca_components = pca.fit_transform(data_numeric)

    # Variância explicada
    variance_ratio = pca.explained_variance_ratio_
    components = pca.components_

    # Criar DataFrame com os loadings dos componentes principais
    pca_loadings = pd.DataFrame(components, columns=data_numeric.columns, index=[f'PC{i+1}' for i in range(n_components)])

    # Gráfico de dispersão PCA
    pca_scatter = px.scatter(
        x=pca_components[:, 0],
        y=pca_components[:, 1],
        color=filtered_df['Proposed Class'] if 'Proposed Class' in filtered_df.columns else None
,  # Certifique-se de que `df` está definido globalmente
        title="PCA ",
        labels={'x': 'PCA1', 'y': 'PCA2'},
        hover_name=filtered_df.index,
        
    )

    explained_variance_ratio = pca.explained_variance_ratio_
    components = [f'PC{i+1}' for i in range(len(explained_variance_ratio))]
    cumulative_variance_ratio = np.cumsum(explained_variance_ratio)  # Variância acumulada

# Criar o scree plot com barras para variância individual
    scree_plot = px.bar(
    x=components,
    y=explained_variance_ratio,
    title="Variância Explicada por Componente Principal",
    labels={'x': 'Componente', 'y': 'Variância Explicada'},
    text=[f"{v:.2%}" for v in explained_variance_ratio],
    
)

# Adicionar a linha de variância acumulada
    scree_plot.add_trace(
    go.Scatter(
        x=components,
        y=cumulative_variance_ratio,
        mode='lines+markers+text',
        name='Variância Acumulada',
        line=dict(color='red', width=2),
        text=[f"{v:.2%}" for v in cumulative_variance_ratio],  # Percentuais na linha
        textposition='top center'
    )
)

    #Aplicar UMAP
    #umap = UMAP(n_components=2, n_neighbors=n_neighbors, min_dist=min_dist)
    #umap_components = umap.fit_transform(data_numeric)

    # Gráfico de dispersão UMAP
 #   umap_scatter = px.scatter(
   #     x=umap_components[:, 0],
   #     y=umap_components[:, 1],
   #     color=df['Proposed Class'] if 'Proposed Class' in filtered_df.columns else None,
   #     title="UMAP",
   #     labels={'x': 'UMAP1', 'y': 'UMAP2'},
        #hover_name=filtered_df.index
  #  )

    # Explicação da variância explicada
    explanation = f"Variância explicada pelos {n_components} componentes principais: {variance_ratio.round(3)}"

    return pca_scatter, scree_plot, explanation, #umap_scatter

@callback(
    Output('umap-graph', 'figure'),
    [Input('n-neighbors-slider', 'value'),
     Input('min-dist-slider', 'value'),
    Input('normalizacao-data', 'data'),]
    #Input(ThemeSwitchAIO.ids.switch("theme"), "value")]
)
def update_graph(n_neighbors, min_dist, normalized_data):
    
    #template = template_theme1 if trigg_thema else template_theme2
    
    filtered_df = pd.DataFrame.from_records(normalized_data)
    data_numeric = filtered_df.select_dtypes(include=np.number).drop('ID', axis=1, errors='ignore')
    data = data_numeric
    
    umap_model = UMAP(n_neighbors=n_neighbors, min_dist=min_dist, n_components=2)
    reduced_data = umap_model.fit_transform(data)
    fig = px.scatter(x=reduced_data[:, 0], 
                     y=reduced_data[:, 1], 
                     title="UMAP",
                     color=filtered_df['Proposed Class'] if 'Proposed Class' in filtered_df.columns else None,
                     labels={'x': 'UMAP1', 'y': 'UMAP2'},
                     hover_name=filtered_df.index,
                     #template=template
    )
    return fig

@callback(
    Output('dendrogram-graph', 'figure'),
    Output('cluster-graph', 'figure'),
    Output('elbow-graph', 'figure'),
    Input('dendrogram-filter', 'value'),
    Input('threshold-slider', 'value'),
    Input('normalizacao-data', 'data'),
    Input('method-filter', 'value'),
    Input('dim-reduction-method', 'value'),
    Input('n-clusters-slider', 'value'),
    
)


def update_dendrogram(dendrogram_filter, threshold, data, method, dim_reduction_method, n_clusters):
    
    
    # Converte os dados serializados de volta para DataFrame
    filtered_df = pd.DataFrame.from_records(data)
    index_names = filtered_df['Proposed Class'] #if 'Proposed Class' in filtered_df.columns else None
    # Seleciona apenas as colunas numéricas e remove a coluna 'ID' (se existir)
    data_numeric = filtered_df.select_dtypes(include=np.number).drop('ID', axis=1, errors='ignore')
    
    pca = PCA(n_components=3)
    pca_components = pca.fit_transform(data_numeric)
    
    umap_model = UMAP(n_components=3)
    umap_components = umap_model.fit_transform(data_numeric)
    
    
    
    
    
    color=filtered_df['Proposed Class'] if 'Proposed Class' in filtered_df.columns else None
    
    
    # Calcula o linkage para o dendrograma
    link= linkage(data_numeric, method=dendrogram_filter)
    
         
    # Cria o dendrograma usando plotly.figure_factory
    fig = ff.create_dendrogram(
        data_numeric,
        orientation='bottom',   # Pode ser 'left', 'right', 'bottom', 'top'
        labels=index_names.tolist(),
        linkagefun=lambda x: link,  # Usa o linkage pré-calculado
        color_threshold=threshold,
       
        
    )
    
    
   
    
    
    # Adiciona a linha horizontal de corte que atravessa todo o dendrograma
    fig.add_shape(
        type='line',
        x0=-0.5,  # Agora funciona porque ajustamos o range do eixo X
        x1=len(index_names.tolist()) + 1700,
        y0=threshold,
        y1=threshold,
        line=dict(color="red", width=2, dash="dash")
    )
    
    # Atualiza o layout do gráfico
    fig.update_layout(
        width=1800,
        height=850,
        title="Dendrograma",
        xaxis_title="Amostras",
        yaxis_title="Distância",
        xaxis=dict(
            tickangle=90,  # Rotaciona os rótulos 90 graus para melhor legibilidade
            tickfont=dict(size=10),  # Reduz o tamanho da fonte dos rótulos
            tickmode='array',
            
            #automargin=True,
            
        ),
        yaxis=dict(
            title_standoff=20,
            tickfont=dict(size=10)
        ),
        margin=dict(b=250, l=100, r=50, t=100),
        title_x=0.5,
        #template=template,
        showlegend=False  # Aumenta a margem inferior para acomodar rótulos
    )
    
    # Ajusta as linhas do dendrograma para maior clareza
    for trace in fig['data']:
        trace['line']['width'] = 1.5
        
    #===========================================+======================================== 
    
    
    if dim_reduction_method == 'pca':
        data = pca_components
        df_data = pd.DataFrame(pca_components[:, :3], columns=['PCA1', 'PCA2', 'PCA3'])
        axis_labels = ['PCA1', 'PCA2', 'PCA3']
        
    elif dim_reduction_method == 'umap':
        data = umap_components
        df_data = pd.DataFrame(umap_components[:, :3], columns=['UMAP1', 'UMAP2', 'UMAP3'], index=index_names)
        axis_labels = ['UMAP1', 'UMAP2', 'UMAP3']
    
     # Criação do gráfico de clusters
    fig_clusters = go.Figure()
     
    if method == 'kmeans': 
        # Análise de KMeans
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        kmeans_labels = kmeans.fit_predict(data)  # Usar 3 dimensões
        df_data['Cluster'] = kmeans_labels
        
        # Adicionar uma trace para cada cluster
        for cluster in df_data['Cluster'].unique():
            cluster_data = df_data[df_data['Cluster'] == cluster]
            
            #customdata = cluster_data['Classe'].unique() # Índices correspondentes aos pontos do cluster
            fig_clusters.add_trace(
                go.Scatter3d(
                x=cluster_data[axis_labels[0]],
                y=cluster_data[axis_labels[1]],
                z=cluster_data[axis_labels[2]],
                mode='markers',
                #marker=dict(size=5),  # Tamanho reduzido para 3D
                name=f'Cluster {cluster}',
                opacity=0.7,
                marker=dict(color=custom_palette[int(cluster) % len(custom_palette)], size=5),
                
                
                #customdata=customdata,  # Passa os índices como customdata
                #hovertemplate=(
                    #f"<b>Índice:</b> %{customdata}<br>" +  # Exibe o índice
                  #  "<extra></extra>"  # Remove a caixa cinza padrão
               # )
            )
        )
            
        
        
        # Adicionar os centróides em 3D
        centroids = kmeans.cluster_centers_
        fig_clusters.add_trace(
            go.Scatter3d(
                x=centroids[:, 0],
                y=centroids[:, 1],
                z=centroids[:, 2],
                mode='markers+text',
                marker=dict(color='red', size=5, symbol='x', line=dict(width=2, color='black')),
                text=[f'C{int(i)}' for i in range(len(centroids))],  # Rótulos como C0, C1, etc.
                textposition='top center',
                name='Centróides'
            )
        )
        title = "Clusters 3D - KMeans com Centróides"

    elif method == 'gmm':
        # Análise de Gaussian Mixture Model (GMM)
        gmm = GaussianMixture(n_components=n_clusters, random_state=42)
        gmm_labels = gmm.fit_predict(data)  # Usar 3 dimensões
        df_data['Cluster'] = gmm_labels
        
        # Adicionar uma trace para cada cluster
        for cluster in df_data['Cluster'].unique():
            cluster_data = df_data[df_data['Cluster'] == cluster]
            fig_clusters.add_trace(
                go.Scatter3d(
                    x=cluster_data[axis_labels[0]],
                    y=cluster_data[axis_labels[1]],
                    z=cluster_data[axis_labels[2]],
                    mode='markers',
                    marker=dict(size=5),
                    name=f'Cluster {cluster}',
                    opacity=0.7
                )
            )
        
        # Adicionar as médias das gaussianas com rótulos em 3D
        gmm_means = gmm.means_
        fig_clusters.add_trace(
            go.Scatter3d(
                x=gmm_means[:, 0],
                y=gmm_means[:, 1],
                z=gmm_means[:, 2],
                mode='markers+text',
                marker=dict(color='red', size=5, symbol='x', line=dict(width=2, color='black')),
                text=[f'M{int(i)}' for i in range(len(gmm_means))],  # Rótulos como M0, M1, etc.
                textposition='top center',
                name='Médias GMM',
            )
                
        )
        title = "Clusters 3D - GMM com Médias"
    
    # Atualiza o layout do gráfico de clusters em 3D
    fig_clusters.update_layout(
        width=800,
        height=800,
        title=title,
        template='plotly_white',
        scene=dict(
            xaxis_title=axis_labels[0],
            yaxis_title=axis_labels[1],
            zaxis_title=axis_labels[2],
            xaxis=dict(showgrid=True, gridcolor='lightgray', gridwidth=2),
            yaxis=dict(showgrid=True, gridcolor='lightgray', gridwidth=2),
            zaxis=dict(showgrid=True, gridcolor='lightgray', gridwidth=2)
        ),
        
    ) 
    
    # Gráfico do Cotovelo (Elbow Method)
    #inertias = []
    #k_range = range(1, 11)  # Testa de 1 a 10 clusters
    #for k in k_range:
    
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    if dim_reduction_method == 'pca':
        kmeans.fit_predict(pca_components)
        
    else:
        kmeans.fit_predict(umap_components)
    #inertias.append(kmeans.inertia_)
        
    

#    fig_elbow = go.Figure()
##       go.Scatter(
  #          x=list(k_range),
   #         y=inertias,
    #        mode='lines+markers',
     #       name='Inércia',
      #      marker=dict(color='blue', size=8)
       # )
   # )
   # fig_elbow.update_layout(
    #    title="Gráfico do Cotovelo (Elbow Method)",
     #   xaxis_title="Número de Clusters (k)",
      #  yaxis_title="Inércia",
       # template='plotly_white',
       # width=800,
        #height=600
   # )
    
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    if dim_reduction_method == 'pca':
        kmeans.fit_predict(pca_components)
        silhouette_avg = silhouette_score(pca_components, kmeans.labels_)
        silhouette_values = silhouette_samples(pca_components, kmeans.labels_)
        
    else:
        kmeans.fit_predict(umap_components)
        silhouette_avg = silhouette_score(umap_components, kmeans.labels_)
        silhouette_values = silhouette_samples(umap_components, kmeans.labels_)
        
        # Calcular o Silhouette Score
        
        
    # Criar DataFrame para o gráfico de faca
    silhouette_df = pd.DataFrame({
    'Silhouette': silhouette_values,
    'Classe': kmeans.labels_
    })
    
    # Ordenar por classe e valor de Silhouette (para melhor visualização)
    silhouette_df = silhouette_df.sort_values(['Classe', 'Silhouette'])

# Adicionar índice cumulativo para o eixo y
    silhouette_df['Index'] = np.arange(len(silhouette_df))
    
    # Criar o gráfico de faca
    fig_silhouette = go.Figure()
    
    # Plotar cada classe
    for classe in silhouette_df['Classe'].unique():
        class_data = silhouette_df[silhouette_df['Classe'] == classe]
        fig_silhouette.add_trace(go.Bar(
            y=class_data['Index'],
            x=class_data['Silhouette'],
            orientation='h',
            name=str(classe),
            marker=dict(color=custom_palette[int(classe) % len(custom_palette)]))  # Cor padrão caso a classe não esteja no dicionário
        )
        
    # Adicionar linha da média do Silhouette Score
    fig_silhouette.add_shape(
        type="line",
        x0=silhouette_avg,
        x1=silhouette_avg,
        y0=0,
        y1=len(silhouette_df),
        line=dict(color="red", width=2, dash="dash"),
        name="Média Silhouette"
    )

    # Configurar o layout do gráfico
    fig_silhouette.update_layout(
        title="Gráfico de Silhouette Score",
        xaxis_title="Coeficiente de Silhouette",
        yaxis_title="Clusters",
        showlegend=True,
        barmode='stack',
        bargap=0,
        plot_bgcolor='white',
        paper_bgcolor='white',
        font=dict(size=12),
        yaxis=dict(showticklabels=False)  # Oculta rótulos do eixo y para melhor legibilidade
    )
    
    return fig, fig_clusters, fig_silhouette
    
#======================Classificação===========================#
@callback(
    Output('model-metrics', 'children'),
    Output('error-analysis', 'children'),
    Output('confusion-matrix', 'figure'),
    Output('stored-model-path', 'data'),
    Input('classification-method', 'value'),
    Input('balance-method', 'value'),
    Input('normalizacao-data', 'data'),
    # Prevenção de concorrência
    prevent_initial_call=True
    

)


def update_classification(method, balance_method, data):
    try:
        filtered_df = pd.DataFrame.from_records(data)
        
        if 'Proposed Class' not in filtered_df.columns:
            raise ValueError("Coluna 'Proposed Class' não encontrada nos dados")

        data_numeric = filtered_df.select_dtypes(include=np.number).drop('ID', axis=1, errors='ignore')
        X = data_numeric
        y = filtered_df['Proposed Class']

        le = LabelEncoder()
        y_encoded = le.fit_transform(y)
        class_names = le.classes_

        X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.10, random_state=42)
        X_train_final, X_val, y_train_final, y_val = train_test_split(X_train, y_train, test_size=0.20, random_state=42)
        
        # lógica de balanceamento aqui
        if balance_method != 'none':
            from imblearn.over_sampling import SMOTE
            from imblearn.under_sampling import RandomUnderSampler
            
            if balance_method == 'smote':
                smote = SMOTE(k_neighbors=4, random_state=42)
                X_train_final, y_train_final = smote.fit_resample(X_train_final, y_train_final)
            elif balance_method == 'undersample':
                rus = RandomUnderSampler(random_state=42)
                X_train_final, y_train_final = rus.fit_resample(X_train_final, y_train_final)
            elif balance_method == 'oversample':
                ros = RandomOverSampler(random_state=42)
                X_train_final, y_train_final = ros.fit_resample(X_train_final, y_train_final)

        #seleção de modelos
        if method == 'rf':
            model = RandomForestClassifier(n_estimators=100, random_state=42)
        elif method == 'dt':  
            model = DecisionTreeClassifier(max_depth=5, max_leaf_nodes=20, random_state=42)
        elif method == 'svm':
            model = SVC(kernel='linear', C=1.0, random_state=42)
        else:
            model = MLPClassifier(max_iter=1000, random_state=42)

        # Cross-validation
        cv_scores = cross_val_score(model, X_train_final, y_train_final, cv=5, scoring='accuracy')
        cv_accuracy_mean = np.mean(cv_scores)
        cv_accuracy_std = np.std(cv_scores)

        model.fit(X_train_final, y_train_final)
        y_pred = model.predict(X_val)

        metrics = [
            html.P(f"Acurácia: {accuracy_score(y_val, y_pred):.2f}"),
            html.P(f"Precisão: {precision_score(y_val, y_pred, average='weighted'):.2f}"),
            html.P(f"Recall: {recall_score(y_val, y_pred, average='weighted'):.2f}"),
            html.P(f"F1-Score: {f1_score(y_val, y_pred, average='weighted'):.2f}"),
            html.P(f"Validação Cruzada (5 folds): {cv_accuracy_mean:.2f} ± {cv_accuracy_std:.2f}")
        ]

        cm = confusion_matrix(y_val, y_pred)
        fig = px.imshow(
            cm,
            labels=dict(x="Previsto", y="Real"),
            x=class_names,
            y=class_names,
            color_continuous_scale='Blues',
            text_auto=True
        )
        
        errors = y_val != y_pred
        # Correção na indexação dos dados
        df_errors = pd.DataFrame(
            X_val.iloc[errors],
            columns=X.columns
        )
        df_errors['True Label'] = le.inverse_transform(y_val[errors])
        df_errors['Predicted Label'] = le.inverse_transform(y_pred[errors])
        
        error_count = len(df_errors)
        total_samples = len(y_val)
        error_rate = (error_count / total_samples) * 100 if total_samples > 0 else 0
        
        alert = dbc.Alert(
            f"Erros encontrados: {error_count} ({error_rate:.1f}% do total)",
            color="danger" if error_rate > 10 else "warning",
            className='mb-3'
        )
        
        # Salvar o modelo treinado
        model_path = f"pretrained_model_{method}.joblib"
        joblib.dump(model, model_path)
        joblib.dump(le, f"label_encoder_{method}.joblib")

        return metrics, alert, fig, model_path  # Retornar 4 valores

    except Exception as e:
        error_msg = html.Div(f"Erro: {str(e)}")
        return error_msg, error_msg, None, None
    
    
@callback(
    [Output('stored-production-data', 'data'),
     Output('data-table2', 'columnDefs'),
     Output('data-table2', 'rowData')],
    [Input('upload-data', 'contents')],
    [State('upload-data', 'filename')],
    
)
def update_output(contents, filename):
    if contents is None:
        return None, [], []  # Retornar 3 valores

    try:
        content_type, content_string = contents.split(',')
        decoded = base64.b64decode(content_string)

        if filename.endswith('.csv'):
            encodings = ['utf-8', 'ISO-8859-1', 'latin1']
            for enc in encodings:
                try:
                    df = pd.read_csv(io.StringIO(decoded.decode(enc)))
                    break
                except UnicodeDecodeError:
                    continue
        elif filename.endswith(('.xls', '.xlsx')):
            df = pd.read_excel(io.BytesIO(decoded))
        else:
            return None, [], []  # Retornar 3 valores para tipo de arquivo inválido

        column_defs = [{"field": col} for col in df.columns]
        row_data = df.to_dict('records')

        return row_data, column_defs, row_data  # Retornar 3 valores

    except Exception as e:
        return None, [], []  # Retornar 3 valores no caso de erro
        
            

@callback(
    [Output('data-table2', 'columnDefs', allow_duplicate=True),
    Output('data-table2', 'rowData', allow_duplicate=True),  # Permitir duplicação
    Output('prediction-output', 'children')],
    [Input('predict-button', 'n_clicks'),
     Input('scale-type2', 'value')],
    [State('classification-method', 'value'),
    State('stored-production-data', 'data'),
    State('stored-model-path', 'data')],
    prevent_initial_call=True
)

def generate_predictions(n_clicks, scale_type, method, data, model_path):
    if method is None:
        return dash.no_update, dash.no_update, html.Div("Selecione um método.")
        
    try:
        if not data:
           return dash.no_update, dash.no_update, html.Div("Nenhum dado carregado.")
        
        if not model_path or not os.path.exists(model_path):
            return dash.no_update, dash.no_update, html.Div(f"Modelo {method} não encontrado. Treine-o primeiro.")

        # Carregar dados e pré-processar
        df_new = pd.DataFrame.from_records(data)
        # Check if 'Proposed Class' exists
        
        
        data_numeric = df_new.select_dtypes(include=np.number).drop('ID', axis=1, errors='ignore')
        X_new = data_numeric
        
     
        
        if scale_type == 'standard':
            scaler = StandardScaler()
            X_new = scaler.fit_transform(X_new)
        elif scale_type == 'minmax':
            scaler = MinMaxScaler()
            X_new = scaler.fit_transform(X_new)
        elif scale_type == 'robust':
            scaler = RobustScaler()
            X_new = scaler.fit_transform(X_new)

        # Carregar modelo e encoder
        model = joblib.load(model_path)
        encoder_path = f"label_encoder_{method}.joblib"
        
        if not os.path.exists(encoder_path):
            return dash.no_update, dash.no_update, html.Div(f"Encoder {encoder_path} não encontrado.")
        
        le = joblib.load(encoder_path)
        
        
        

        
        
        # Fazer previsões
        y_pred_encoded = model.predict(X_new)
        df_new['Predicted Class'] = le.inverse_transform(y_pred_encoded)
        
        if 'Proposed Class' in df_new.columns:
            try:
                y_true_encoded = le.transform(df_new['Proposed Class'])
                # Calculate metrics
                accuracy = accuracy_score(y_true_encoded, y_pred_encoded)
                precision = precision_score(y_true_encoded, y_pred_encoded, average='weighted')
                recall = recall_score(y_true_encoded, y_pred_encoded, average='weighted')
                f1 = f1_score(y_true_encoded, y_pred_encoded, average='weighted')
                
                
                metrics_text = [f"Acurácia: {accuracy:.4f}", f"Precision: {precision:.4f}", f"Recall: {recall:.4f}", f"F1 Score: {f1:.4f}"]
                output_text = html.Div(
                [
                    html.H5("Métricas de Desempenho:", style={'textAlign': 'center', 'marginBottom': '15px', 'color': '#4A90E2'}),
                    html.Ul(
                        [html.Li(metric, style={'fontSize': '16px', 'color': '#333', 'marginBottom': '5px'}) for metric in metrics_text],
                        style={'listStyleType': 'disc', 'paddingLeft': '20px'}
                    )
    ],
    style={
        'border': '1px solid #ddd',
        'borderRadius': '8px',
        'padding': '20px',
        'backgroundColor': '#f9f9f9',
        'boxShadow': '0 2px 4px rgba(0, 0, 0, 0.1)',
        'marginTop': '10px'
    }
)
            except ValueError:
                output_text = html.Div("Atenção: Classes não vistas no treinamento encontradas nos dados. Métricas não calculadas.", style={'marginTop': '10px', 'color': 'orange'})
        else:
            output_text = html.Div("Previsões geradas com sucesso! Métricas não calculadas pois 'Proposed Class' está ausente.", style={'marginTop': '10px'})
        
        # Atualize a tabela
        column_defs = [{"field": col} for col in df_new.columns]
        row_data = df_new.to_dict('records')
        
        return column_defs, row_data, output_text
    
    except Exception as e:
        return dash.no_update, dash.no_update, html.Div(f"Erro: {str(e)}", style={'color': 'red'})