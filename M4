import dash
from dash import dcc, html, Input, Output, State, dash_table
import pandas as pd
import plotly.express as px
import io
import base64
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
import numpy as np

app = dash.Dash(__name__)
app.title = "CS301 Milestone 4 Group 12"
global_df = None
model = None
features = []
target_variable = None

# Layout
app.layout = html.Div([
    html.H1("Upload File", style={'textAlign': 'center'}),

    # Upload
    dcc.Upload(
        id='upload-data',
        children=html.Div(['Drag and Drop or ', html.A('Select Files')]),
        style={
            'width': '100%', 'height': '60px', 'lineHeight': '60px',
            'borderWidth': '1px', 'borderStyle': 'dashed', 'borderRadius': '5px',
            'textAlign': 'center', 'margin': '10px'
        },
        multiple=False
    ),
    # Select Target
    html.Div([
        html.Label("Select Target:"),
        dcc.Dropdown(id='target-dropdown')
    ], style={'width': '50%', 'margin': '10px auto'}),

    # Barcharts
    html.Div([
        # Average target value by category
        html.Div([
            html.H4("Average target by category"),
            dcc.RadioItems(id='category-radio', inline=True),
            dcc.Graph(id='bar-chart-category')
        ], style={'width': '48%', 'display': 'inline-block'}),

        # Correlation Strength
        html.Div([
            html.H4("Correlation Strength of Numerical Variables with Target"),
            dcc.Graph(id='bar-chart-correlation')
        ], style={'width': '48%', 'display': 'inline-block'})
    ]),
    html.Hr(),

    # Feature Selection & Training
    html.Div([
        html.H4("Select Features and Train Model"),
        dcc.Checklist(id='feature-checklist', inline=True),
        html.Button("Train", id='train-button', n_clicks=0, style={'margin': '10px'}),
        html.Div(id='r2-score-output', style={'fontSize': '18px', 'margin': '10px'})
    ]),
    html.Hr(),

    # Predict
    html.Div([
        html.H4("Predict Target Value"),
        dcc.Input(id='predict-input', type='text', placeholder="Enter feature values, comma-separated"),
        html.Button("Predict", id='predict-button', n_clicks=0, style={'margin': '10px'}),
        html.Div(id='predict-output', style={'fontSize': '18px'})
    ])
], style={'backgroundColor': 'white', 'padding': '20px'})


# File Upload
@app.callback(
    [Output('target-dropdown', 'options'),
     Output('category-radio', 'options'),
     Output('feature-checklist', 'options')],
    Input('upload-data', 'contents'),
    State('upload-data', 'filename')
)
def upload_file(contents, filename):
    global global_df
    if contents is None:
        return [], [], []

    # Would like to try and change from io if possible, todo later. Generally needs to be reworked
    content_type, content_string = contents.split(',')
    decoded = io.StringIO(io.BytesIO(base64.b64decode(content_string)).read().decode('utf-8'))
    global_df = pd.read_csv(decoded)

    numerical_cols = global_df.select_dtypes(include=['number']).columns.tolist()
    categorical_cols = global_df.select_dtypes(exclude=['number']).columns.tolist()

    return [{'label': col, 'value': col} for col in numerical_cols], \
           [{'label': col, 'value': col} for col in categorical_cols], \
           [{'label': col, 'value': col} for col in global_df.columns if col != target_variable]


# Bar Charts
@app.callback(
    [Output('bar-chart-category', 'figure'),
     Output('bar-chart-correlation', 'figure')],
    [Input('target-dropdown', 'value'),
     Input('category-radio', 'value')]
)
def update_charts(target, category):
    global target_variable
    if target is None or global_df is None:
        return {}, {}

    target_variable = target

    if category:
        avg_df = global_df.groupby(category)[target].mean().reset_index()
        fig1 = px.bar(avg_df, x=category, y=target, title=f"Average {target} by {category}")
    else:
        fig1 = {}

    numerical_cols = global_df.select_dtypes(include=['number']).drop(columns=target).columns
    corr_values = global_df[numerical_cols].corrwith(global_df[target]).abs().reset_index()
    corr_values.columns = ['Variable', 'Correlation']
    fig2 = px.bar(corr_values, x='Variable', y='Correlation',
                  title=f"Correlation Strength with {target}", text='Correlation')

    return fig1, fig2


# Train Model
@app.callback(
    Output('r2-score-output', 'children'),
    Input('train-button', 'n_clicks'),
    State('feature-checklist', 'value')
)
def train_model(n_clicks, selected_features):
    global model
    if n_clicks == 0 or not selected_features or target_variable is None:
        return ""

    X = global_df[selected_features]
    y = global_df[target_variable]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    num_features = X.select_dtypes(include=['number']).columns
    cat_features = X.select_dtypes(exclude=['number']).columns

    preprocessor = ColumnTransformer(transformers=[
        ('num', StandardScaler(), num_features),
        ('cat', Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('onehot', OneHotEncoder(handle_unknown='ignore'))
        ]), cat_features)
    ])

    model = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('regressor', LinearRegression())
    ])

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    r2 = r2_score(y_test, y_pred)

    return f"The R² score is: {r2:.2f}"


# Predict Callback
@app.callback(
    Output('predict-output', 'children'),
    Input('predict-button', 'n_clicks'),
    State('predict-input', 'value'),
    State('feature-checklist', 'value')
)
def predict(n_clicks, input_values, selected_features):
    if n_clicks == 0 or not input_values or model is None:
        return ""

    try:
        input_list = input_values.split(',')
        input_data = pd.DataFrame([input_list], columns=selected_features)

        for col in input_data.select_dtypes(include=['object']).columns:
            if col in global_df.select_dtypes(include=['number']).columns:
                input_data[col] = input_data[col].astype(float)

        prediction = model.predict(input_data)[0]

        return f"Predicted target value: {prediction:.2f}"
    except Exception as e:
        return f"Error: {str(e)}"

if __name__ == '__main__':
    app.run_server(debug=True)
