import PySimpleGUI as sg
import pandas as pd
import numpy as np
from glob import glob
import sys
import os

"""
HELP

This script imports the CSV data from the peri-events analysis and removes 
columns of data where there was no activity (defined using a minimum threshold).
If x is the mean of one part of the data and y is mean of another part of the 
data, the threshold is abs(x - y).

The time period for defining each mean is the proportion of the event window.
Thus, 0 is the start of the event window and 1 is the end of the event window.
You can also put in fractions like 1/3.
"""

# Create a GUI.
import_location = r"C:\Users\hazza\Downloads\newfiphotanks\newfiphotanks\Agrp_2965_3220f-230601-011335\New folder"
export_location = r"C:\Users\hazza\Downloads\newfiphotanks\newfiphotanks\Agrp_2965_3220f-230601-011335\New folder"
mean1_start     = '0'
mean1_end       = '1/3'
mean2_start     = '1/3'
mean2_end       = '2/3'
diff_thresh     = 0.5
sg.theme("DarkTeal2")
layout = [[sg.T("")], 
    [sg.Text("Choose a folder for the import location"), 
     sg.Input(key="Import",enable_events=True,default_text=import_location),
     sg.FolderBrowse(key="Import2")], [sg.T("")], 
    [sg.Text("Choose a folder for the export location"),
     sg.Input(key="Export",enable_events=True,default_text=export_location),
     sg.FolderBrowse(key="Export2")], [sg.T("")], 
    [sg.Text("Choose the time period to define the first mean",size=(37,1)),
     sg.Input(key="Mean1_start",enable_events=True,default_text=mean1_start,size=(10,1)),
     sg.Input(key="Mean1_end",  enable_events=True,default_text=mean1_end,  size=(10,1))], [sg.T("")], 
    [sg.Text("Choose the time period to define the second mean",size=(37,1)),
     sg.Input(key="Mean2_start",enable_events=True,default_text=mean2_start,size=(10,1)),
     sg.Input(key="Mean2_end",  enable_events=True,default_text=mean2_end,  size=(10,1))], [sg.T("")], 
    [sg.Text("Choose the minimum threshold for keeping a column of data"),
     sg.Input(key="Threshold",enable_events=True,default_text=diff_thresh,size=(10,1))], [sg.T("")], 
    [sg.Button("Submit")]]
window  = sg.Window('Removing columns with no signal', layout)
while True:
    event, values = window.read()
    if event == sg.WIN_CLOSED or event=="Exit":
        window.close()
        sys.exit()
    elif event == "Submit":
        import_location = values['Import']
        export_location = values['Export']
        diff_thresh     = values['Threshold']
        mean1_start     = float(eval(values['Mean1_start']))
        mean1_end       = float(eval(values['Mean1_end']))
        mean2_start     = float(eval(values['Mean2_start']))
        mean2_end       = float(eval(values['Mean2_end']))
        window.close()
        break

# Create a list of all CSV paths in the import location.
import_paths  = os.path.join(import_location, "*.csv")
import_paths  = glob(import_paths)
exclude_paths = []

# For every CSV in this list of paths:
for path in import_paths:
    
    # Import the CSV and check whether an already cleaned file is imported.
    df = pd.read_csv(path, header=None)
    if 'Direction between 2 means' in df[0].tolist():
        print('Skipping an already cleaned file.')
        exclude_paths += [path]
        continue
    
    # Split the header and data to manipulate the data.
    num_header_rows = df.index[df[0] == 'Custom name'][0] + 1
    header = df.loc[:num_header_rows-1]
    data   = df.loc[num_header_rows:].astype(float)
    
    # Remove columns where the absolute difference between the mean1 and mean2
    # is less than `diff_thresh`.
    keep_cols  = []
    directions = []
    data_cols    = data.columns[3:].tolist()
    nondata_cols = data.columns[:3].tolist()
    mean_col     = nondata_cols[2]
    mean1_start_ind = int(len(data)*mean1_start)
    mean1_end_ind   = int(len(data)*mean1_end)
    mean2_start_ind = int(len(data)*mean2_start)
    mean2_end_ind   = int(len(data)*mean2_end)
    for col in data_cols:
        mean1 = data[col][mean1_start_ind:mean1_end_ind].mean()
        mean2 = data[col][mean2_start_ind:mean2_end_ind].mean()
        diff  = mean1 - mean2
        if abs(diff) >= 0.5:
            keep_cols += [col]
        if diff > 0:
            directions += ['Decreasing']
        elif diff == 0:
            directions += ['Flat']
        elif diff < 0:
            directions += ['Increasing']
    data[mean_col] = data[keep_cols].mean(axis=1)
    new_cols = nondata_cols + keep_cols
    
    # Add a header row with the increasing/decreasing info.
    new_header = ['Direction between 2 means', np.nan, np.nan] + directions
    new_header = {i:[new_header[i]] for i in range(len(new_header))}
    new_header = pd.DataFrame(new_header)
    header = pd.concat([header, new_header])
    
    # Sort the columns with signal by increasing/decreasing info.
    directions_row = ['#','#','#'] + directions
    sorted_cols    = {new_cols[i]:directions_row[i] for i in range(len(new_cols))}
    new_cols       = sorted(new_cols, key=lambda x:sorted_cols[x])

    # Remove the columns without a signal.
    df = pd.concat([header, data])
    df = df[new_cols]
    df.index = range(len(df))
    
    # Export this new CSV with "_cleaned" added to the file name.
    export_name = os.path.basename(path)[:-4] + "_cleaned.csv"
    export_path = os.path.join(export_location, export_name)
    df.to_csv(export_path, index=False, header=False)
    
# Export the settings for analysis.
text_export_name = os.path.basename(import_location) + ' cleaning settings.txt'
text_export_path = os.path.join(export_location, text_export_name)
list_files = []
with open(text_export_path, 'w') as file:
    file.write(f'Import location was {import_location}\n')
    file.write(f'Export location was {export_location}\n')
    file.write(f'Mean 1 time period was [{round(mean1_start,2)}, {round(mean1_end,2)}]\n')
    file.write(f'Mean 2 time period was [{round(mean2_start,2)}, {round(mean2_end,2)}]\n')
    file.write(f'Minimum threshold was {diff_thresh}\n\n')
    file.write('Cleaned files were:\n')
    for path in import_paths:
        if path in exclude_paths:
            continue
        file.write(os.path.basename(path)+'\n')
        