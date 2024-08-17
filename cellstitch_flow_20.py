import os
import io
import torch
import dash
import numpy as np
import flask
import base64
import tifffile
import uuid;

from PIL import Image  # Import Image from PIL

from cellpose.models import Cellpose
from dash import dcc, html, Input, Output, State, exceptions
from dash.exceptions import PreventUpdate

# Initialize the Dash app
app = dash.Dash(__name__)
server = app.server

# Define layout of the app
app.layout = html.Div([
    html.H1("CellStitch Interface", style={'text-align': 'center'}),
    dcc.Upload(
        id='upload-image',
        children=html.Div([
            'Drag and Drop or ',
            html.A('Select Image Files')
        ]),
        style={
            'width': '50%',
            'height': '40px',
            'lineHeight': '40px',
            'borderWidth': '1px',
            'borderStyle': 'dashed',
            'borderRadius': '5px',
            'textAlign': 'center',
            'margin': '10px auto'
        },
        # Allow multiple files to be uploaded
        multiple=False
    ),
    html.Div(id='upload-success-msg', style={'text-align': 'center'}),
    html.Div([
        dcc.Input(
            id='input-shape',
            type='text',
            placeholder='Enter image shape in X,Y,Z dimension: (e.g. 20,224,224)',
            style={'width': '50%', 'margin': '10px auto', 'display': 'block', 'margin-left': 'auto', 'margin-right': 'auto'}
        )
    ]),
    html.Div([
        dcc.Input(
            id='flow-threshold',
            type='text',
            placeholder='Enter flow threshold between 0.0 and 20.0 (default 1.0)',
            style={'width': '50%', 'margin': '10px auto', 'display': 'block', 'margin-left': 'auto', 'margin-right': 'auto'}
        )
    ]),
    html.Button('Run CellStitch', id='run-button', n_clicks=0, style={'display': 'block', 'margin': '20px auto'}),
    html.Div(id='output-message', style={'text-align': 'center'}),
    html.Div(id='output-download', style={'text-align': 'center'}),

    dcc.Interval(
        id='message-timer',
        interval=2000,  # in milliseconds
        n_intervals=0,
        disabled=True  # Start with Disabled
    ),
    html.Div(id='enable-message-timer'),
])

# Callback to display upload success message
@app.callback(
    Output('upload-success-msg', 'children'),
    [Input('upload-image', 'contents')],
    [State('upload-image', 'filename')]
)
def display_upload_success(contents, filename):
    if contents is not None:
        return html.Div('File successfully uploaded:' + filename, style={'margin': '10px auto'})
    else:
        return None
    

# Function to process the uploaded file
# TODO (YJ): find a way to either pass in or extract the actual image size 
# (E.g. img.shape = (Z, Y, X))

# Function to process the uploaded file
def process_file(base64String, filename, shape, out_filename, in_flow_threshold):
    if filename is not None:
        try:
            # Convert the base64 encoded content to a numpy array
            print("process_file: Begin.")
            decoded_string = base64.b64decode(base64String)
            decoded_bytes = io.BytesIO(decoded_string)
            if filename[-3:] == 'tif':
                img = tifffile.imread(decoded_bytes)
            else:
                img = io.imread(decoded_bytes)
            img_array = np.array(img)
            print("process_file: Finished reading file.")
            # Ensure image has the correct shape
            if img.shape != shape:
                img = img.resize(shape)
            print("process_file: Running Cellpose algorithm.")
            # Run Cellpose algorithm
            flow_threshold = float(in_flow_threshold)
            use_gpu = torch.cuda.is_available()
            model = Cellpose(model_type='cyto2', gpu=use_gpu)
            # Ensure input image has consistent shape
            if len(img.shape) == 2:  # If it's a single 2D image
                img = img[np.newaxis, :, :]  # Add a new axis to represent the batch dimension
            # Process the image
            masks = []
            for i in range(img.shape[0]):
                mask, _, _, _ = model.eval([img[i]], flow_threshold=flow_threshold, channels=[0, 0])
                masks.append(mask[0])  # Append the first channel of the mask
            cellstitch_masks = np.array(masks)
            print("process_file: Completed Cellpose algorithm.")
            current_directory = os.path.dirname(__file__)
            output_filename = current_directory + "/" + out_filename
            np.save(output_filename, cellstitch_masks)
            print("process_file: Creatied output file ", output_filename)
            return cellstitch_masks, None
        except Exception as e:
            print("process_file: Error - ", e)
            return None, html.Div([
                'An error occurred while processing the image:',
                html.Pre(str(e))
            ])
    else:
        return None, None

# Callback to handle file upload and processing
@app.callback(
    [Output('output-message', 'children'),
     Output('output-download', 'children'),
     Output('message-timer', 'disabled')],
    [Input('run-button', 'n_clicks')],
    [State('upload-image', 'contents'),
     State('upload-image', 'filename'),
     State('input-shape', 'value'),
     State('flow-threshold', 'value')],  # Include input shape as State
     prevent_initial_call=True,
)
def update_output(n_clicks, contents, filename, shape_value, flow_threshold):
    
    if contents is not None:
        try:
            # Set default value to flow threshold if not entered
            if flow_threshold is None:
                flow_threshold = 1.0
            if (float(flow_threshold) < 0):
                flow_threshold = 1.0
            if (float(flow_threshold) > 20):
                flow_threshold = 20.0 
            print("RunProcess- file:" + str(filename) + " shape:" + str(shape_value) + " flow threshold:" + str(flow_threshold))
            # Show message while algorithm is running
            running_msg = html.Div('CellStitch algorithm is running... Please wait', style={'margin': '20px auto'})
            # Run the algorithm
            shape = tuple(map(int, shape_value.split(',')))  # Convert shape_value to tuple of integers
            # create randon unique output file for each request
            output_filename = str(uuid.uuid4()) + ".npy"
            #output_filename = 'cellstitch_masks.npy'
            cellstitch_masks = process_file(contents.encode("utf8").split(b";base64,")[1],
                                            filename,
                                            shape, output_filename, flow_threshold)
            running_msg = html.Div('CellStitch algorithm completed ...', style={'margin': '20px auto'})
            if cellstitch_masks is not None:
                download_button = html.A('Download Result: ' + output_filename, href='/download/{}'.format(output_filename))
                return running_msg, download_button, True
            else:
                raise PreventUpdate
        except Exception as e:
            print("Error: ", e)
            running_msg =  html.Div(['An error occurred while processing the image:',html.Pre(str(e))])
            return running_msg, None, True
    else:
        print("RunProcess-No file.")
        raise PreventUpdate

@app.callback(
    [Output('enable-message-timer', 'children'),
     Output('message-timer', 'disabled', allow_duplicate=True)],
    [Input('run-button', 'n_clicks')],
    prevent_initial_call=True,
)
def enable_message_timer(n_clicks):
    print("Running-Message-Enabled", str(n_clicks))
    return "", False


@app.callback(
    Output('output-message', 'children', allow_duplicate=True),
    [Input('message-timer', 'n_intervals')],
    [State('message-timer', 'disabled'),
     State('upload-image', 'filename')],
     prevent_initial_call=True,
)
def running_message(n_intervals, disabled, filename):
    if filename == None :
        running_msg = html.Div('', style={'margin': '20px auto'})
    elif n_intervals % 2 == 0:
        running_msg = html.Div('CellStitch algorithm is running...', style={'margin': '20px auto'})
    else:
        running_msg = html.Div('CellStitch algorithm is running... Please wait...', style={'margin': '20px auto'})
    return running_msg

# Callback to handle file download
@app.server.route('/download/<filename>')
def download_output(filename):
    return flask.send_file(filename, as_attachment=True)

if __name__ == '__main__':
    app.run_server(debug=True)
