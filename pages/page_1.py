import dash
from dash import html, dcc
from dash.dependencies import Input, Output, State
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import dash_bootstrap_components as dbc
from sklearn.impute import SimpleImputer
from scipy.stats import pearsonr
#from app import app
from app import *
from dash import callback
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import numpy as np
import dash_ag_grid as dag
from dash.exceptions import PreventUpdate
from io import StringIO
import io
import base64
from scipy.stats.mstats import zscore, winsorize
#from dash_bootstrap_templates import ThemeSwitchAIO

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


# Carregando os dados
df = pd.read_excel("/Users/fabri/Documents/Codigo/Fabão/TabelaErica_Final.xlsx", decimal= ",")

#missing_columns = df.columns[df.isnull().sum() > 0]
#imputer = SimpleImputer(strategy='median')
#df[missing_columns] = imputer.fit_transform(df[missing_columns])

df
missing_columns = df.columns[df.isnull().sum() > 0]
imputer = SimpleImputer(strategy='median')
df[missing_columns] = imputer.fit_transform(df[missing_columns])
#df = df[df['%Sulfur'] != df['%Sulfur'].max()]
#df = df[df['%Aromatic'] != df['%Aromatic'].max()]
initial_data = df.to_dict('records')
df1 = df[df['%Sulfur'] != df['%Sulfur'].max()]
df1= df1[df1['%Aromatic'] != df1['%Aromatic'].max()]
corr = df1.corr(numeric_only=True)
df1_corr_forte = corr[(corr > 0.6) | (corr < -0.6)]
df1_corr_forte = df1_corr_forte.dropna(axis=0, how='all').dropna(axis=1, how='all')


# ========== Layout ============ #


# Layout da página
layout = dbc.Container(children=[
    dcc.Store(id="transformed-data", data=initial_data), # Armazena os dados transformados
    
    
    # Linha 1
    dbc.Row([
        # Coluna 1: Filtros
        dbc.Col([
            dbc.Card([
                dbc.CardHeader(html.H5("Filtros e Controles")),
                   
                dbc.CardBody([
                    # Classe de Origem
                    dbc.Label("Classe de Origem:", className="fw-bold mb-2"),
                    html.I(className="fa-brands fa-suse"),  
                    dbc.Checklist(
                        id="classe-filter",
                        options=[{"label": classe, "value": classe} for classe in df["Proposed Class"].unique()],
                        value=df["Proposed Class"].unique(),
                        inline=False,
                        switch=True,
                        className="mb-3"
                    ),
                                   
                # Transformação de Dados
                    dbc.Label("Transformação:", className="fw-bold mb-2"),
                    dbc.RadioItems(
                        id="scale-type",
                        options=[
                            {"label": " Escala Original", "value": "original"},
                            {"label": " Escala Logarítmica", "value": "log"},
                            {"label": " Normalização (Z-Score)", "value": "zscore"},
                        ],
                        value="original",  # Padrão: dados originais
                        class_name="mb-3",
                    ),
                      
                
                
                    # Tratamento de Outliers
                    dbc.Label("Outliers:", className="fw-bold mb-2"),
                    dbc.RadioItems(
                        id="outlier",
                        options=[
                            {"label": " Sem tratamento", "value": "none"},
                            {"label": " Remover via Z-Score (|Z| > 3)", "value": "zscore"},
                            {"label": " Winsorizar (1º/99º percentil)", "value": "winsorize"},
                        ],
                        value="none",
                        class_name="mb-3,"  
                    )
                ], style={"maxHeight": "500px", "overflowY": "auto"}),
            ], style=tab_card)
        ], sm=12, lg=2),  # Ocupa 3 colunas em telas grandes

        
         # Área Principal
        dbc.Col([
            dbc.Tabs([
                # Tab 1: Análise Geral
                dbc.Tab(label="Visão Geral", children=[
                    dbc.Row([
                        # Gráfico de Distribuição (primeiro gráfico)
                        dbc.Col(dbc.Card([
                            dbc.CardHeader(html.H5("Distribuição por Classe")),
                            dbc.CardBody(dcc.Graph(id='graph1', className='dbc', config=config_graph)),
                        ], style=tab_card), width=12, lg=6, sm=12),  # Ocupa 6 colunas em telas grandes
                        
                        # Gráfico de Origem (segundo gráfico)
                        dbc.Col(dbc.Card([
                            dbc.CardHeader(html.H5("Origem das Amostras de Óleo")),
                            dbc.CardBody([
                                dbc.Col([
                                    html.Label("Selecione a relação para análise:", className="fw-bold mb-2"),
                                    dbc.Select(
                                        id="origem-filter",
                                        options=[
                                            {"label": "Pristano/n-C17 x Fitano/n-C18 (Shanmugam, 1985)", "value": "opcao1"},
                                            {"label": "Tr24/Tr21 x Tr26/Tr25 foi utilizada (Chang et al., 2008)", "value": "opcao2"}
                                        ],
                                        value="opcao1",
                                        className="mb-2"
                                    )
                                ], width=6),
                                
                                dbc.Row([
                                    dbc.Col(
                                        dcc.Graph(id='graph2', className='dbc', config=config_graph),
                                        sm=12, md=6, lg=12
                                    ),
                                ])
                            ]),
                        ], style=tab_card), width=12, lg=6, sm=12)  # Ocupa 6 colunas em telas grandes
                    ],className='g-2 my-auto', style={'margin-top': '7px'}),
                    # Gráficos Ternários
                    dbc.Row([
                        dbc.Col([
                            dbc.Card([
                                dbc.CardHeader(html.H5("Composição Ternária")),
                                dbc.CardBody([
                                    dbc.Row([
                                        dbc.Col(
                                            dcc.Graph(id='graph3', className='dbc', config=config_graph),
                                            sm=12, md=6, lg=6
                                        ),
                                        dbc.Col(
                                            dcc.Graph(id='graph4', className='dbc', config=config_graph),
                                            sm=12, md=6, lg=6
                                        )
                                    ]),
                                ]),
                            ]),
                        ], style=tab_card, sm=12, lg=12),
                    ],className='g-2 my-auto', style={'margin-top': '7px'}),
                ]),  # Fechamento do dbc.Tab
                
                # Tab 2: Indicadores de Qualidade
                dbc.Tab(label="Qualidade do Óleo", children=[
                    dbc.Row([
                        dbc.Col([
                            dbc.Card([
                                dbc.CardHeader(html.H5("Indicadores de Qualidade do Óleo")),
                                dbc.CardBody([
                                    dbc.Row([
                                        dbc.Col([
                                            dbc.Label("Selecione a relação para análise:", className="fw-bold mb-2"),
                                            dbc.Select(
                                                id="quality-filter",
                                                options=[
                                                    {"label": "Grau API vs. % Enxofre", "value": "select1"},
                                                    {"label": "Grau API vs. % NOS (Nitrogênio, Oxigênio e Enxofre)", "value": "select2"},
                                                    {"label": "Grau API vs. % Hidrocarbonetos Saturados", "value": "select3"}
                                                ],
                                                value="select3",
                                                className="mb-3"
                                            )
                                        ], width=6)
                                    ]),
                                    dbc.Row([
                                        dbc.Col(
                                            dcc.Graph(id='graph5', className='dbc', config=config_graph),
                                            sm=12, md=6, lg=12, width=12
                                        ),
                                    ])
                                ])
                            ],style=tab_card),
                        ], sm=12, lg=12),  # Ocupa 9 colunas em telas grandes
                    ],className='g-2 my-auto', style={'margin-top': '7px'}),
                ]),  # Fechamento do dbc.Tab
                
                
                # Tab 3: Estatísticas
                dbc.Tab(label="Estatísticas", children=[
                    dbc.Row([
                        dbc.Col([
                            dbc.Card([
                                dbc.CardHeader(html.H5("Análise de Correlação")),
                                dbc.CardBody([
                                    dbc.Row([
                                        dbc.Col([
                                            html.Label("Selecione o Eixo X:"),
                                            dcc.Dropdown(
                                                id="x-axis",
                                                options=[{'label': col, 'value': col} for col in df.select_dtypes(include=np.number).columns],
                                                value=df.select_dtypes(include=np.number).columns[0],
                                                style={ 
                                                    'backgroundColor': '#f8f9fa',
                                                    'color': 'black'
                                                }
                                            )
                                        ], width=3),

                                        dbc.Col([
                                            html.Label("Selecione o Eixo Y:"),
                                            dcc.Dropdown(
                                                id="y-axis",
                                                options=[{"label": col, "value": col} for col in df.select_dtypes(include=np.number).columns],
                                                value=df.select_dtypes(include=np.number).columns[1],
                                                style={  
                                                    'backgroundColor': '#f8f9fa',
                                                    'color': 'black'
                                                }
                                            )
                                        ], width=3),
                                    ]),

                                    dbc.Col([
                                            html.Label("Correlação (R):"),
                                            html.Div(
                                                id="correlation-value",
                                                style={
                                                    "padding": "9px",
                                                    "border": "1px solid #ddd",
                                                    "borderRadius": "5px",
                                                    "backgroundColor": "#f8f9fa",
                                                    "textAlign": "center",
                                                    "fontWeight": "bold",
                                                    "fontSize": "16px"
                                                },
                                                children="0.00"  # Valor inicial
                                            )
                                        ], width=1, style={"display": "flex", "flexDirection": "column", "justifyContent": "flex-end"}),

                                    
                                    dbc.Col(
                                        dcc.Graph(id='graph8', className='dbc', config=config_graph),
                                        sm=12, md=6, lg=12
                                    )
                                    
                                ])
                            ]),
                        ],width=7),

                        dbc.Col([
                            dbc.Card([
                                dbc.CardHeader(html.H5('Box-Plot por Classe ')),
                                dbc.CardBody([
                                    html.Label("Selecione o Elemento:"),
                                    dcc.Dropdown(
                                        id="element-stats",
                                        options=[{"label": col, "value": col} for col in df.select_dtypes(include=np.number).columns],
                                        value=df.select_dtypes(include=np.number).columns[0],
                                        style={
                                            'backgroundColor': '#f8f9fa',
                                            'color': 'black'
                                        }
                                    ),
                                    dbc.Col([
                                        dcc.Graph(id='graph7', className='dbc', config=config_graph),
                                       
                                        
                                     ])
                                ])
                            ], style=tab_card)
                        ], width=5),
                    # Gráfico de Correlação
                    dbc.Row([
                        dbc.Col([
                            dbc.Card([
                                dbc.CardHeader(html.H5("Mapa de Correlação")),
                                dbc.CardBody([
                                    dcc.Graph(id='graph9', className='dbc', config=config_graph),
                                     dcc.Dropdown(
                                            id='variable-selector',
                                            options=[{'label': col, 'value': col} for col in df1_corr_forte.columns],
                                            multi=True,
                                            placeholder="Selecione variáveis para exibir"
                                        ),
                                    
                                ])
                            ]),
                        ],style=tab_card, sm=12, lg=12),  # Ocupa 9 colunas em telas grandes
                    ],className="g-2 my-auto",style={'margin-top': '7px'})
                   
                   
                    ],className="g-2 my-auto",style={'margin-top': '7px'})
                ])
            ])


        ], sm=12, lg=10),  # Ocupa 9 colunas em telas grandes
    ], className='g-2 my-auto', style={'margin-top': '7px'}),
  
    # Tabela de dados
    dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardHeader(html.H5("Dados Filtrados")),
                dbc.CardBody(dag.AgGrid(
                    id="data-table",
                    columnDefs=[{"field": col} for col in df.columns],  # Use as colunas do seu DataFrame original
                    defaultColDef={"sortable": True, "filter": True, "resizable": True},
                    style={"height": "400px"},
                    dashGridOptions={"pagination": True}
                ))
            ], className="mt-3 shadow-sm")
        ], width=12)
    ])
    #], className='g-2 my-auto', style={'margin-top': '7px'}),


], fluid=True, style={'height': '100vh'})

# ========== Callbacks ============ #

#Callback para Tratamento de Outliers
   

@callback(
    Output("transformed-data", "data"),
    Input("scale-type", "value"),
    Input('outlier', 'value'),
    State('transformed-data', 'data'),
    
)
def apply_transformations(scale_type, outlier, data):
    df_transformed = df.copy()
    
    # Identificar colunas numéricas (para aplicar transformações)
    numeric_cols = df_transformed.select_dtypes(include=np.number).columns
    
    if outlier == 'zscore':
        z_scores = df_transformed[numeric_cols].apply(zscore)
        df_transformed = df_transformed[(z_scores.abs() < 3).all(axis=1)]
    elif outlier == 'winsorize':
        for col in numeric_cols:
            df_transformed[col] = winsorize(df_transformed[col], limits=[0.01, 0.01])
    
                 
    # Transformação de escala
    if scale_type == "log":
        df_transformed[numeric_cols] = np.log1p(df_transformed[numeric_cols])
    elif scale_type == "zscore":
        scaler = StandardScaler()
        df_transformed[numeric_cols] = scaler.fit_transform(df_transformed[numeric_cols])
        
    else:  # original
        df_scaler = df_transformed
    
    return df_transformed.to_dict('records')





# Callback 1: Atualiza os gráficos principais (graph1 e graph2) com base nos filtros

@callback(
    Output('graph1', 'figure'),
    Output('graph2', 'figure'),
    Input('origem-filter', 'value'),
    Input('classe-filter', 'value'),
    Input(ThemeSwitchAIO.ids.switch("theme"), "value"),
    Input("transformed-data", "data")
)
def update_graphs(selected_origem, selected_classes, toggle_theme, transformed_data):

    # Define o template com base no tema selecionado
    template = template_theme1 if toggle_theme else template_theme2
    # Filtra o DataFrame com base nas classes selecionadas
    filtered_df = pd.DataFrame.from_records(transformed_data)
    filtered_df = filtered_df[filtered_df["Proposed Class"].isin(selected_classes)]
    

    # Gráfico 1 - Barras horizontais (mantido igual)
    class_counts = filtered_df['Proposed Class'].value_counts()
    class_percentages = (class_counts / len(filtered_df)) * 100
    
    fig1 = go.Figure()

    fig1.add_trace(go.Bar(
        y=class_counts.index,
        x=class_counts.values,
        orientation='h',
        marker_color=custom_palette,
        text=[f'{count} ({percentage:.1f}%)' for count, percentage in zip(class_counts.values, class_percentages)],
        textposition='auto',
        #marker_color = px.colors.qualitative.T10,
        
        
    ))

    fig1.update_layout(template=template)
        #title="Distribuição das Classes (com percentuais)",
       # xaxis_title="Quantidade",
      #  yaxis_title="Classes Propostas",
    #    template=template,  # Aplica o template de acordo com o tema,
   #     height=500,
   #     width=400,
   #     plot_bgcolor='rgba(240, 240, 240, 0.5)'
    #)

    # Configurações adicionais para melhorar legibilidade
    fig1.update_traces(
        textfont=dict(size=12),
        hovertemplate='<b>%{y}</b><br>Quantidade: %{x}<br>Percentual: %{text}<extra></extra>'
    )

    # Gráfico 2 - Baseado na seleção de origem
    if selected_origem == "opcao1":
        fig2 = px.scatter(
            filtered_df,  # Usar filtered_df em vez de df para aplicar os filtros
            x="PHY/nC18", 
            y="Pri/nC17", 
            color="Proposed Class",
            labels={"PHY/nC18": "PHY/nC18", "Pri/nC17": "Pri/nC17"},
            color_discrete_sequence=custom_palette,
            
            
            #width=890,
        )
        
                
        # Adicionando a linha diagonal tracejada
        fig2.add_trace(
            go.Scatter(
                x=[0.0, 0.8], 
                y=[0.0, 0.8], 
                mode='lines', 
                line=dict(color='blue', width=1, dash='dash'),
                name='Linha de referência',
                showlegend=True
            )
        )

        # Adicionando as anotações de texto para os tipos de matéria orgânica (MO)
        fig2.add_annotation(
            x=0.1, 
            y=0.35,
            text="MO Não Marinha",
            showarrow=False,
            font=dict(color="red", size=14)
        )

        fig2.add_annotation(
            x=0.35, 
            y=0.10,
            text="MO Marinha",
            showarrow=False,
            font=dict(color="blue", size=14)
        )

        # Configurando os limites dos eixos
        fig2.update_xaxes(range=[0.0, 0.8])
        fig2.update_yaxes(range=[0.0, 0.8])

        # Melhorando o layout
        fig2.update_layout(template=template)
          #  legend_title_text="Proposed Class",
          #  font=dict(family="Arial", size=12),
           # plot_bgcolor='rgba(240, 240, 240, 0.5)',
          #  margin=dict(l=40, r=40, t=60, b=40),
           # hovermode='closest',
            #template=template
       # )

        # Ajustando o tamanho dos pontos
        fig2.update_traces(
            marker=dict(size=10, opacity=0.8),
            selector=dict(mode='markers')
        )      
    elif selected_origem == "opcao2":
        fig2 = px.scatter(
            filtered_df,  # Usar filtered_df em vez de df para aplicar os filtros
            x="26/25TRI", 
            y="24/25TRI", 
            color="Proposed Class",
            labels={"Tr24/Tr21": "Tr24/Tr21", "Tr26/Tr25": "Tr26/Tr25"},
            color_discrete_sequence=custom_palette,
            
        )

        # Adicionando a linha diagonal tracejada# Adicionando linhas verticais de corte
        # Limites Marinhos (azuis)
        fig2.add_shape(
            type="line",
            x0=0.5, y0=0, x1=0.5, y1=2,
            line=dict(color="blue", width=1, dash="dash"),
        )
        fig2.add_shape(
            type="line",
            x0=1.0, y0=0, x1=1.0, y1=2,
            line=dict(color="blue", width=1, dash="dash"),
        )

        # Limites Lacustres (vermelhos)
        fig2.add_shape(
            type="line",
            x0=1.25, y0=0, x1=1.25, y1=2,
            line=dict(color="red", width=1, dash="dash"),
        )
        fig2.add_shape(
            type="line",
            x0=1.99, y0=0, x1=1.99, y1=2,
            line=dict(color="red", width=1, dash="dash"),
        )

        # Adicionando textos para categorias
        fig2.add_annotation(
            x=0.6, y=0.1,
            text="Marinho",
            showarrow=False,
            font=dict(color="blue", size=14)
        )
        fig2.add_annotation(
            x=1.5, y=0.1,
            text="Lacustre",
            showarrow=False,
            font=dict(color="red", size=14)
        )

        # Adicionando legendas das linhas
        fig2.add_trace(
            go.Scatter(
                x=[None], y=[None],
                mode='lines',
                line=dict(color='blue', width=1, dash='dash'),
                name='Limite Marinho',
                showlegend=True
            )
        )
        fig2.add_trace(
            go.Scatter(
                x=[None], y=[None],
                mode='lines',
                line=dict(color='red', width=1, dash='dash'),
                name='Limite Lacustre',
                showlegend=True
            )
        )

        # Configurando os limites dos eixos
        fig2.update_xaxes(range=[0.0, 2.0])
        fig2.update_yaxes(range=[0.0, 2.0])

        # Ajustando o layout
      #  fig2.update_layout(
        #    legend_title_text="Proposed Class",
        #    font=dict(family="Arial", size=12),
        #    plot_bgcolor='rgba(240, 240, 240, 0.5)',
        #    margin=dict(l=40, r=40, t=60, b=40),
           # hovermode='closest'
        #)

        # Ajustando o tamanho dos pontos
        fig2.update_traces(
            marker=dict(size=10, opacity=0.8),
            selector=dict(mode='markers')
        )

    

    return fig1, fig2
    
# Callback 2: Atualiza os gráficos ternários (graph3, graph4)
@callback(
    Output('graph3', 'figure'),
    Output('graph4', 'figure'),
    Input('classe-filter', 'value'),
    Input("transformed-data", "data"),
    Input(ThemeSwitchAIO.ids.switch("theme"), "value")

)
def ternary_graphs(selected_classes, transformed_data, toggle_theme):

    template = template_theme1 if toggle_theme else template_theme2

    filtered_df = pd.DataFrame.from_records(transformed_data)
    filtered_df = filtered_df[filtered_df["Proposed Class"].isin(selected_classes)]
    

    fig3 = px.scatter_ternary(
        filtered_df,
        a='%27BBS218', b='%28BBS218', c='%29BBS218',  # Colunas que representam os componentes
        color='Proposed Class',           
        #title='Distribuição dos esteranos regulares (Huang e Meinschein, 1979)',
        hover_name='Proposed Class',       # Informação ao passar o mouse
        color_discrete_sequence=custom_palette,  # Paleta de cores personalizada
        template=template,  # Aplicar o template de acordo com o tema            
    )

    # Ajustar opacidade, contorno e tamanho dos pontos
    fig3.update_traces(marker=dict(opacity=0.8, 
                                line=dict(width=0.5, color='black'),
                                size=8)) 

    # Ajustar layout
    fig3.update_layout(
        width=600, height=700,             # Ajustar largura e altura
        title_x=0.5,                       # Centralizar o título
        font=dict(family="Arial, sans-serif", size=12),  # Definir fonte
        ternary=dict(
            #bgcolor="white",              # Fundo do gráfico ternário
            sum=100,                       # A soma dos eixos deve ser 100%
            aaxis=dict(title='C27'),  # Títulos dos eixos
            baxis=dict(title='C28'),
            caxis=dict(title='C29')
        ),
        hovermode="closest",
                       
    )

    # Gráfico 4 - Ternário
    fig4 = px.scatter_ternary(
        filtered_df,
        a='%Saturated', b='%Aromatic', c='%NSO',  # Colunas que representam os componentes
        color='Proposed Class',           
        #title='Composição Ternária dos Óleos da Bacia Potiguar',
        hover_name='Proposed Class',       # Informação ao passar o mouse
        color_discrete_sequence=custom_palette, # Paleta de cores personalizada
        template=template            
    )

    # Ajustar opacidade, contorno e tamanho dos pontos
    fig4.update_traces(marker=dict(opacity=0.8, 
                                line=dict(width=0.5, color='black'),
                                size=8)) 

    # Ajustar layout
    fig4.update_layout(
        width=600, height=700,             # Ajustar largura e altura
        title_x=0.5,                       # Centralizar o título
        font=dict(family="Arial, sans-serif", size=12),  # Definir fonte
        ternary=dict(
            #bgcolor="white",              # Fundo do gráfico ternário
            sum=100,                       # A soma dos eixos deve ser 100%
            aaxis=dict(title='% Saturados'),  # Títulos dos eixos
            baxis=dict(title='% Aromáticos'),
            caxis=dict(title='% NSO')
        ),
        hovermode="closest"               
    )

    return fig3, fig4

# Callback 3: Atualiza os Indicadores de Qualidade (graph5)
@callback(
    Output('graph5', 'figure'),
    Input('quality-filter', 'value'),
    Input('classe-filter', 'value'),
    Input("transformed-data", "data"),
    Input(ThemeSwitchAIO.ids.switch("theme"), "value")
)
def quality_graph(selected_quality, selected_classes, transformed_data, toggle_theme):

    template = template_theme1 if toggle_theme else template_theme2

    filtered_df = pd.DataFrame.from_records(transformed_data)
    filtered_df = filtered_df[filtered_df["Proposed Class"].isin(selected_classes)]

    if selected_quality == "select1":
        fig5 = px.scatter(
            filtered_df,
            x='API',
            y='%Sulfur',
            color='Proposed Class',
            #height=500,
            #width=700,  # Ajustado para um quarto do tamanho original, já que estava em subplot(2,2,4)
            labels={'API': 'API', '%Sulfur': '%Sulfur'},
            hover_data=['Proposed Class', 'API', '%Sulfur'],
            template=template,
            color_discrete_sequence=custom_palette,
        
        )

        # Ajustando o layout
        fig5.update_layout(
            font=dict(family="Arial", size=12),
            legend_title_text='Proposed Class',
            plot_bgcolor='rgba(240, 240, 240, 0.5)',  # Fundo cinza claro
            margin=dict(l=40, r=40, t=60, b=40),
            hovermode='closest'
        )

        # Melhorando a aparência dos pontos
        fig5.update_traces(
            marker=dict(size=10, opacity=0.8, line=dict(width=1, color='DarkSlateGrey')),
        )
                
    elif selected_quality == "select2":
        fig5 = px.scatter(
        filtered_df,
        x='API',
        y='%NSO',
        color='Proposed Class',
        #height=500,
        #width=600,  # Dimensionado para corresponder ao subplot original
        labels={'API': 'API', '%NSO': '%NSO'},
        hover_data=['Proposed Class', 'API', '%NSO'],
        title='API vs %NSO por Classe Proposta',
        template = template,
        color_discrete_sequence=custom_palette,
        )

    # Ajustando o layout
        fig5.update_layout(
            font=dict(family="Arial", size=12),
            legend_title_text='Proposed Class',
            plot_bgcolor='rgba(240, 240, 240, 0.5)',
            margin=dict(l=40, r=40, t=60, b=40),
            hovermode='closest'
        )

    # Melhorando a aparência dos pontos
        fig5.update_traces(
            marker=dict(size=10, opacity=0.8, line=dict(width=1, color='DarkSlateGrey')),
        )

    elif selected_quality == "select3":
        fig5 = px.scatter(
            filtered_df,
            x='API', 
            y='%Saturated',
            color='Proposed Class',
            #height=500,
            #width=600,  # Dimensionado para corresponder ao subplot original
            labels={'API': 'API', '%Saturated': '%Saturated'},
            hover_data=['Proposed Class', 'API', '%Saturated'],
            title='API vs % Saturados por Classe',
            template = template,
            color_discrete_sequence=custom_palette,
        )

        # Ajustando o layout
#        fig5.update_layout(
#            font=dict(family="Arial", size=12),
#            legend_title_text='Proposed Class',
#            plot_bgcolor='rgba(240, 240, 240, 0.5)',
#            margin=dict(l=40, r=40, t=60, b=40),
#           hovermode='closest'
#        )
        

        # Melhorando a aparência dos pontos
        fig5.update_traces(
            marker=dict(size=10, opacity=0.8, line=dict(width=1, color='DarkSlateGrey')),
        )

    return fig5
          
# Callback 4: Atualiza scatter (graph6)
@callback(
    Output('graph8', 'figure'),
    Output("correlation-value", "children"),
    Input('classe-filter', 'value'),
    Input('x-axis', 'value'),
    Input('y-axis', 'value'),
    Input("transformed-data", "data"),
    Input(ThemeSwitchAIO.ids.switch("theme"), "value")
    
)                 
                
def scatter_graph(selected_classes, x_axis, y_axis, transformed_data, toggle_theme):

    template = template_theme1 if toggle_theme else template_theme2

    filtered_df = pd.DataFrame.from_records(transformed_data)
    filtered_df = filtered_df[filtered_df["Proposed Class"].isin(selected_classes)]

    if x_axis and y_axis:
        correlation = filtered_df[x_axis].corr(filtered_df[y_axis])
        color = "green" if correlation >= 0 else "red"
        corr_p= html.Span(f"{correlation:.2f}", style={"color": color})
        
    
    fig6 = px.scatter(
            filtered_df,
            x=x_axis,
            y=y_axis,
            color='Proposed Class',
            #height=600,
            #width=1000,
            labels={x_axis: x_axis, y_axis: y_axis},
            #title=(f'correlação {x_axis} x {y_axis} '),
            hover_data=['Proposed Class', x_axis, y_axis],
            template=template,
            color_discrete_sequence=custom_palette,
        )

        # Ajustando o layout
   # fig6.update_layout(
#            font=dict(family="Arial", size=12),
#            legend_title_text='Proposed Class',
#           plot_bgcolor='rgba(240, 240, 240, 0.5)',
#          margin=dict(l=40, r=40, t=60, b=40),
#           hovermode='closest'
#       )

        # Melhorando a aparência dos pontos
    fig6.update_traces(
            marker=dict(size=10, opacity=0.8, line=dict(width=1, color='DarkSlateGrey')),
        )
    
    

    return fig6, corr_p

# Callback 5: Atualiza Box-Plot (graph7)

@callback(
    Output('graph7', 'figure'),
    Input('element-stats', 'value'),
    Input('classe-filter', 'value'),
    Input("transformed-data", "data"),
    Input(ThemeSwitchAIO.ids.switch("theme"), "value")
)
def box_plot(selected_element, selected_classes, transformed_data, toggle_theme):

    template = template_theme1 if toggle_theme else template_theme2
    # Define o template com base no tema selecionado
    filtered_df = pd.DataFrame.from_records(transformed_data)
    filtered_df = filtered_df[filtered_df["Proposed Class"].isin(selected_classes)]

    fig7 = px.box(
        filtered_df,
        x='Proposed Class',
        y=selected_element,
        color='Proposed Class',
        labels={'Proposed Class': 'Classe Proposta', selected_element: selected_element},
        #title=f'Distribuição de {selected_element} por Classe',
        #height=350,
        #width=600,
        template=template,
        color_discrete_sequence=custom_palette,
    )

    # Ajustando o layout
#    fig7.update_layout(
#        font=dict(family="Arial", size=12),
#        legend_title_text='Proposed Class',
#        plot_bgcolor='rgba(240, 240, 240, 0.5)',
#        margin=dict(l=40, r=40, t=60, b=40),
#        hovermode='closest'
#   )

    # Melhorando a aparência dos pontos
    fig7.update_traces(
        marker=dict(size=10, opacity=0.8, line=dict(width=1, color='DarkSlateGrey')),
    )

    return fig7


@callback(
    Output("data-table", "rowData"),  # Alterando para saída direta nos dados da tabela
    Input("classe-filter", "value"),
    Input("transformed-data", "data")
)
def update_table(selected_classes, transformed_data):
    filtered_df = pd.DataFrame.from_records(transformed_data)
    
    if selected_classes:  # Verifica se há classes selecionadas
        filtered_df = filtered_df[filtered_df["Proposed Class"].isin(selected_classes)]
    
    return filtered_df.to_dict('records')

@callback(
    Output('graph9', 'figure'),
    Input('variable-selector', 'value')
)
def update_heatmap(selected_variables):
    if not selected_variables:
        filtered_corr = df1_corr_forte
    else:
        filtered_corr = df1_corr_forte.loc[selected_variables, selected_variables]
    
    # Usar uma escala de cores personalizada semelhante à da imagem
    colorscale = [
        [0, 'rgb(75, 145, 190)'],     # Azul para correlações negativas
        [0.5, 'rgb(241, 237, 220)'],  # Bege claro para correlações próximas de zero
        [1, 'rgb(177, 89, 40)']       # Marrom/laranja para correlações positivas
    ]
    
    fig = px.imshow(
        filtered_corr,
        height=800,                 # Aumentar altura
        width=800,                  # Aumentar largura
        color_continuous_scale=colorscale,
        zmin=-1,
        zmax=1,
        text_auto='.2f',           # Formatar para duas casas decimais
        aspect="auto"
    )
    
    # Ajustes adicionais de layout
    fig.update_layout(
        title=None,                 # Remover título para combinar com a imagem
        coloraxis_colorbar=dict(
            title="Correlação",
            tickvals=[-1, -0.5, 0, 0.5, 1],
            ticktext=['-1.0', '-0.5', '0.0', '0.5', '1.0']
        ),
        margin=dict(l=40, r=40, t=40, b=40),
        xaxis=dict(side='top'),     # Mover eixo X para cima como na imagem
    )
    
    # Ajustar o tamanho e estilo do texto
    fig.update_traces(
        textfont=dict(size=10, color='black'),
        showscale=True
    )
    
    return fig

#if __name__ == "__main__":
    #app.run_server(port = 8050, debug=True)