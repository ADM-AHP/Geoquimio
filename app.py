import dash
import dash_bootstrap_components as dbc
from dash import Input, Output, dcc, html
from pages import Home, page_1, page_2

# import from folders/theme changer
from tamplate import *
from dash_bootstrap_templates import ThemeSwitchAIO



# ========== Styles ============ #
tab_card = {'height': '100%'}

main_config = {
    "hovermode": "x unified",
    "legend": {"yanchor":"top", 
                "y":0.9, 
                "xanchor":"left",
                "x":0.1,
                "title": {"text": None},
                "font" :{"color":"white"},
                "bgcolor": "rgba(0,0,0,0.5)"},
    "margin": {"l":10, "r":10, "t":10, "b":10}
}

config_graph={"displayModeBar": False, "showTips": False}

template_theme1 = "flatly"
template_theme2 = "darkly"
url_theme1 = dbc.themes.BOOTSTRAP
url_theme2 = dbc.themes.DARKLY

FONT_AWESOME = ["https://use.fontawesome.com/releases/v5.10.2/css/all.css"]

app = dash.Dash(__name__, pages_folder="pages", use_pages=True, external_stylesheets=FONT_AWESOME, suppress_callback_exceptions=True)
server = app.server

# Configuração global para adicionar Font Awesome
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

# the style arguments for the sidebar. We use position:fixed and a fixed width
SIDEBAR_STYLE = {
    "position": "fixed",
    "top": "7px",
    "left": 0,
    "bottom": 0,
    "width": "16rem",
    "padding": "1.5rem 1rem",
    #"background-color": "#f8f9fa",
}

# the styles for the main content position it to the right of the sidebar and
# add some padding.
CONTENT_STYLE = {
    "margin-left": "16rem",
    "margin-right": "1rem",
    "padding": "2rem 1rem",
}

sidebar = html.Div(
    [
        dbc.Col([
            dbc.Card(
                [
                    html.H2("GeoQuimio", style={"font-family": "Voltaire", "font-size": "30px","display": "flex", "alignItems": "center"}, 
                            className="sidbar-header"),
                    dbc.Row([
                            ThemeSwitchAIO(aio_id="theme", themes=[url_theme1, url_theme2]),
                            
                            
                            
                    ],style={"marginTop": "0.5rem", "marginBottom": "0.5rem"}),

                    html.Hr(),

                    html.P("Análise Exploratória de Dados Geoquímicos", 
                           className="lead sidebar-text",
                           style={"fontSize": "0.9rem", "marginTop": "10px", "marginBottom": "1.5rem"}),
                        
                    dbc.Nav(
                [
                        dbc.NavLink(
                            [
                                html.I(className="fas fa-home", style={"marginRight": "8px"}),
                                html.Span("Introdução")
                            ],
                            href="/",
                            active="exact",
                            style={"display": "flex", "alignItems": "center"}
                        ),
                        dbc.NavLink(
                            [
                                html.I(className="fas fa-chart-bar", style={"marginRight": "8px"}),
                                html.Span("Visualização")
                            ],
                            href="/page-1",
                            active="exact",
                            style={"display": "flex", "alignItems": "center"}
                        ),
                        dbc.NavLink(
                            [
                                html.I(className="fas fa-cogs", style={"marginRight": "8px"}),
                                html.Span("Algoritmos")
                            ],
                            href="/page-2",
                            active="exact",
                            style={"display": "flex", "alignItems": "center"}
                        ),
                    ],
                    vertical=True,
                    pills=True,
                    className="sidebar-nav"
                    ),
                ], style={"height": "100%", "width": "250px", "margin": "7px", "padding": "20px", "marginTop": "7px"}, 
            ),
        ], className='g-2 my-auto', style={"height": "100%",'margin-top': '7px'}, sm=3, md=1, lg=1),
    ],
    style=SIDEBAR_STYLE
)

content = html.Div(id="page-content", style=CONTENT_STYLE)

app.layout = html.Div([dcc.Location(id="url"), sidebar, content])



@app.callback(Output("page-content", "children"), [Input("url", "pathname")])
def render_page_content(pathname):
    if pathname == "/":
        return Home.app.layout
    elif pathname == "/page-1":
        return page_1.layout
    elif pathname == "/page-2":
        return page_2.layout
    # If the user tries to reach a different page, return a 404 message
    return html.Div(
        [
            html.H1("404: Not found", className="text-danger"),
            html.Hr(),
            html.P(f"The pathname {pathname} was not recognised..."),
        ],
        className="p-3 bg-light rounded-3",
    )


if __name__ == "__main__":
    app.run_server(debug=False)