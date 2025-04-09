import dash
from dash import html, dcc
from dash.dependencies import Input, Output

import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go


import dash_bootstrap_components as dbc



app = dash.Dash(__name__, external_stylesheets=[dbc.themes.MINTY])
server = app.server


# Adicionar Font Awesome
app.index_string = '''
<!DOCTYPE html>
<html>
    <head>
        {%metas%}
        <title>{%title%}</title>
        {%favicon%}
        {%css%}
        <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.0.0/css/all.min.css">
    </head>
    <body>
        {%app_entry%}
        <footer>
            {%config%}
            {%scripts%}
            {%renderer%}
        </footer>
    </body>
</html>
'''

df = pd.read_excel("/Users/fabri/Documents/Codigo/Fabão/TabelaErica_Final.xlsx", decimal= ",")

app.title = "Análise Geoquímica"

# Layout do Dashboard

app.layout = dbc.Container([
    # Título
    dbc.Row([
        dbc.Col(html.H1("Análise Exploratória e Aplicação de Inteligência Artificial na Classificação Genética de Óleos da Bacia Potiguar", style={"text-align": "center","font-family": "Voltaire", "font-size": "40px"}), className="mb-4")
    ]),

    #sub-título
    dbc.Row([
        dbc.Col(html.H2("Objetivo.", style={"font-family": "Voltaire", "font-size": "40px"}), className="mb-4")

    ]),

    # Texto
    dbc.Row([
        dbc.Col([
            html.P("Desenvolver um classificador automatizado para amostras de óleo utilizando técnicas avançadas de inteligência artificial. "
                "O modelo combina abordagens supervisionadas e não supervisionadas."
                "As principais técnicas utilizadas incluem:",style={"font-size": "18px"}),
                
    
           html.Ul([
                html.Li([html.I(className="fas fa-sitemap"), " Árvores de Decisão (AD): modelo interpretável e transparente."]),
                html.Li([html.I(className="fas fa-brain"), " Redes Neurais Artificiais (RNA): modelo de alta complexidade e capacidade de aprendizado."]),
                html.Li([html.I(className="fas fa-project-diagram"), " Análise Hierárquica de Cluster (Dendograma): técnica não supervisionada para agrupamento de dados."]),
                html.Li([html.I(className="fas fa-chart-area"), " UMAP: método para redução de dimensionalidade."]),
            ],style={"font-size": "18px"})

               
            
        ])
    ]),    

    #materias
    dbc.Row([
        dbc.Col([
                html.H2("Materias.", style={"font-family": "Voltaire", "font-size": "40px"}), 
                html.P("O estudo utilizou um conjunto de 200 amostras, previamente analisadas pelo Centro de Excelência em Geoquímica da Petrobras (CEGEQ/CENPES)",style={"font-size": "18px"}),
        ]),





    ]),

    #imagem
    dbc.Row([
        dbc.Col(html.Img(src="assets/mapa_bacia1.png", style={"width": "100%", "height": "auto"}), width=6),

        dbc.Col(html.Img(src="assets/mapa_classificacao.png", style={"width": "100%", "height": "auto"}), width=6)
            
    ], class_name="mb-4"),

    dbc.Row([
        dbc.Col(html.Footer(
            "Referência: MORAIS, Érica Tavares de. Aplicações de Técnicas de Inteligência Artificial para Classificação Genética de Amostras de Óleo da Porção Terrestre, "
            "da Bacia Potiguar, Brasil. Rio de Janeiro, COPPE/UFRJ, 2007. Dissertação de Mestrado em Engenharia Civil.",
            style={
                "text-align": "center",
                "background-color": "#f8f9fa",
                "padding": "10px",
                "color": "#7f8c8d",
                "border-top": "1px solid #ddd",
                "width": "100%",
                "position": "fixed",
                "bottom": "0",
                "left": "0",
            
            }
        ))
    ])
])



   









if __name__ == "__main__":
    app.run_server(port = 8051, debug=True)