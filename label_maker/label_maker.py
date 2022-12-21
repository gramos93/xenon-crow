import warnings
import numpy as np
import base64
import io
from dash_canvas import DashCanvas

import dash
from dash import Dash, dcc, html
from dash_extensions.enrich import Output, DashProxy, Input, MultiplexerTransform, State
from dash_canvas.utils import array_to_data_url, parse_jsonstring

from PIL import Image
from skimage.segmentation import slic
from skimage.segmentation import mark_boundaries
#from skimage.util import img_as_float
from skimage import img_as_ubyte
from skimage import io as skio
import plotly.express as px


warnings.filterwarnings('ignore')
dims = (240,320,3)

'''
Dash app layout. Python wrappers for html code generation.
'''
app = DashProxy(__name__, transforms=[MultiplexerTransform()])
app.layout = html.Div(
    [
        dcc.Upload(
            id='upload',
            children=html.Div([
                'Drag and Drop or ',
                html.A('Select Files')
            ]),
            style={
            'width': '100%',
            'height': '60px',
            'lineHeight': '60px',
            'borderWidth': '1px',
            'borderStyle': 'dashed',
            'borderRadius': '5px',
            'textAlign': 'center',
            'margin': '10px'
            },
        ),
        html.Div([
            dcc.Graph(id='canvas',
                       figure=px.imshow(np.zeros(dims))), 
            html.Div([
                html.H5('Enter Number of Superpixels:'),
                dcc.Input(id="num_pixels", type='number', style={'width': 190, 'height': 30}),
                html.Button('Apply SLIC segmentation', id='segment', n_clicks=0, style={'width': 200, 'height': 50}),
                html.Button('Create Dataset', id='d_set', n_clicks=0, style={'width': 200, 'height': 50}),
            ], style={"display":'flex', "align-items":'center', "flex-direction":'column'}),
        ], style={"display":'flex', "align-items":'center', "flex-direction":'row'}),
        html.Div(
            id='show_masks',
            children=[
                html.Div("Superpixel Masks:")
            ]
        ),
        # Used to store our states/actions in the web page before each update
        dcc.Store(id='image_string'),
        dcc.Store(id='selects'),
        dcc.Store(id='clusters'),
        dcc.Store(id='gt'),
        dcc.Store(id='filename')
    ],
)

@app.callback(Output('image_string', 'data'),
              Input('upload', 'contents'),
              prevent_initial_call=True)
def update_image_string(contents):
    return contents

@app.callback(Output('filename', 'data'),
              Input('upload', 'filename'),
              prevent_initial_call=True)
def update_image_string(filename):
    return filename


@app.callback(Output('gt', 'data'),
              Input('d_set', 'n_clicks'),
              State('selects', 'data'),
              State('clusters', 'data'),
              State('filename', 'data'),
              prevent_initial_call=True)
def generate_dataset(n_clicks, selects, clusters, filename):
    if clusters:
        clusters = np.array(clusters).astype(np.uint8)
        if selects:
            gt = np.zeros((240,320))
            for sel in selects:
                val = clusters[sel[1]][sel[0]]
                bool_mask = clusters == val
                gt = np.where(clusters==val, 255, gt)
                gt = gt.astype(np.uint8)
            im = Image.fromarray(gt)
            im.save("out/gt_"+str(filename)+".png")

            idx = 1
            for c in np.unique(clusters):
                mask = np.where(clusters==c, 255, 0)
                mask = mask.astype(np.uint8)
                im = Image.fromarray(mask)
                im.save("out/state"+str(idx)+"_"+str(filename)+".png")
                idx += 1
                

@app.callback(Output('clusters', 'data'),
            Input('upload', 'contents'),
            Input('segment', 'n_clicks'),
            State('num_pixels', 'value'),
            prevent_initial_call=True)
def update_clusters(contents, clickData, num_pixels):
    changed_id = [p['prop_id'] for p in dash.callback_context.triggered][0]
    content_type, content_string = contents.split(',')
    decoded = base64.b64decode(content_string)
    image = np.array(Image.open(io.BytesIO(decoded)))
    if 'segment' in changed_id:
        clusters = slic(image, n_segments = num_pixels, sigma = 5, enforce_connectivity = True)
        return clusters

@app.callback(Output('canvas', 'figure'),
              Input('clusters', 'data'),
              Input('selects', 'data'),
              State('image_string', 'data'),
              State('num_pixels', 'value'),
              State('upload', 'contents'),
              prevent_initial_call=True)
def update_canvas(clusters, selects, image_string, num_pixels, contents):
    content_type, content_string = contents.split(',')
    decoded = base64.b64decode(content_string)
    image = np.array(Image.open(io.BytesIO(decoded)))
    if clusters:
        clusters = np.array(clusters).astype(np.uint8)
        segmented_img = mark_boundaries(image, clusters)
        if selects:
            for sel in selects:
                val = clusters[sel[1]][sel[0]]
                bool_mask = clusters == val
                segmented_img[:,:,0][bool_mask] = 0.0
                segmented_img[:,:,1][bool_mask] = 1.0
                segmented_img[:,:,2][bool_mask] = 0.0
        return px.imshow(segmented_img)
    return px.imshow(image)

@app.callback(Output('show_masks', 'children'),
              Input('selects', 'data'),
              State('clusters', 'data'),
              State('show_masks', 'children'),
              prevent_initial_call=True)
def update_image_string(selects, clusters, child):
    if clusters:
        clusters = np.array(clusters).astype(np.uint8)
        if selects:
            for sel in selects:
                val = clusters[sel[1]][sel[0]]
                bool_mask = clusters == val
                mask = np.where(clusters==val, 255, 0)
                mask = mask.astype(np.uint8)
                image_string = array_to_data_url(mask)
    return child + [html.Img(src=image_string, alt='image')]

@app.callback(Output('selects', 'data'),
            Input('canvas', 'clickData'),
            State('selects', 'data'),
            prevent_initial_call=True)
def update_selects(clickData, data):
    if clickData:
        x = clickData["points"][0]["x"]
        y = clickData["points"][0]["y"]
        if data is None:
            data = list()
        else:
            data = list(data)
        data.append([x,y])
        return data

if __name__ == "__main__":
    app.run_server(debug=True)