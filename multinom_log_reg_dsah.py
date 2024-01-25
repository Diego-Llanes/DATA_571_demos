import dash
from dash import dcc, html
from dash.dependencies import Input, Output, State
import plotly.graph_objs as go
import numpy as np

# Initialize the Dash app
app = dash.Dash(__name__)

# App layout
app.layout = html.Div([
    html.Div([
        html.Div([
            dcc.Input(id=f'weight-{i}-{j}', type='number', value=0, style={'width': '60px'})
            for j in range(3)
        ], style={'padding': '10px'}) for i in range(2)
    ], style={'display': 'inline-block', 'verticalAlign': 'top'}),

    html.Div([
        dcc.Input(id=f'bias-{j}', type='number', value=0, style={'width': '60px'})
        for j in range(3)
    ], style={'display': 'inline-block', 'paddingLeft': '20px'}),

    html.Button('Update', id='update-button', n_clicks=0),

    dcc.Graph(id='graph')
])

def softmax(z):
    exp_z = np.exp(z - np.max(z, axis=0))
    return exp_z / exp_z.sum(axis=0)

@app.callback(
    Output('graph', 'figure'),
    [Input('update-button', 'n_clicks')],
    [State(f'weight-{i}-{j}', 'value') for i in range(2) for j in range(3)] +
    [State(f'bias-{j}', 'value') for j in range(3)]
)
def update_graph(n_clicks, *args):
    weights = np.array(args[:6]).reshape(2, 3)
    biases = np.array(args[6:])

    # Generate a grid of points
    x1, x2 = np.meshgrid(np.linspace(-10, 10, 100), np.linspace(-10, 10, 100))
    grid = np.c_[x1.ravel(), x2.ravel()]

    # Compute model output (z) for each class
    z = np.dot(grid, weights) + biases

    # Apply softmax to get probabilities
    probabilities = softmax(z.T).T

    # Determine the predicted class based on max probability
    predicted_class = np.argmax(probabilities, axis=1).reshape(x1.shape)

    # Plot decision boundaries and color map
    fig = go.Figure()

    # Add contour for softmax probabilities
    fig.add_trace(go.Contour(x=np.linspace(-10, 10, 100), y=np.linspace(-10, 10, 100), z=predicted_class, colorscale='Jet', opacity=0.5, showscale=False))

    # Plot hyperplanes for each class
    x_vals = np.linspace(-10, 10, 100)
    for i in range(3):
        if weights[1, i] != 0:  # Avoid division by zero
            y_vals = (-weights[0, i] * x_vals - biases[i]) / weights[1, i]
            fig.add_trace(go.Scatter(x=x_vals, y=y_vals, mode='lines', name=f'Class {i+1} Hyperplane'))

    fig.update_layout(xaxis_title='X1', yaxis_title='X2', title='Decision Boundaries and Softmax Probabilities')
    return fig

# Run the app
if __name__ == '__main__':
    app.run_server(debug=True)
