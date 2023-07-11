import PySimpleGUI as sg
import pandas as pd
from glob import glob
import sys
import os

"""
This script imports the CSV data from the peri-events analysis and removes 
columns of data where there was no activity (defined using a minimum threshold).
If x is the mean of the first third of data and y is mean of the second third of
data, the minimum threshold is abs(x - y).
"""

# Create a GUI.
import_location = r"C:\Users\hazza\Downloads\newfiphotanks\newfiphotanks\Agrp_2965_3220f-230601-011335\New folder"
export_location = r"C:\Users\hazza\Downloads\newfiphotanks\newfiphotanks\Agrp_2965_3220f-230601-011335\New folder"
diff_thresh     = 0.5
sg.theme("DarkTeal2")
layout = [[sg.T("")], 
    [sg.Text("Choose a folder for the import location"), 
     sg.Input(key="Import",enable_events=True,default_text=import_location),
     sg.FolderBrowse(key="Import2")], [sg.T("")], 
    [sg.Text("Choose a folder for the export location"),
     sg.Input(key="Export",enable_events=True,default_text=export_location),
     sg.FolderBrowse(key="Export2")], [sg.T("")], 
    [sg.Text("Choose the minimum threshold for keeping a column of data."),
     sg.Input(key="Threshold",enable_events=True,default_text=diff_thresh,size=(10,1))],
    [sg.T("")], [sg.Button("Submit")]]
window  = sg.Window('Removing columns with no signal', layout)
while True:
    event, values = window.read()
    if event == sg.WIN_CLOSED or event=="Exit":
        window.close()
        sys.exit()
    elif event == "Submit":
        import_location = values['Import']
        export_location = values['Export']
        diff_thresh = values['Threshold']
        window.close()
        break

# Create a list of all CSV paths in the import location.
import_paths = os.path.join(import_location, "*.csv")
import_paths = glob(import_paths)

# For every CSV in this list of paths:
for path in import_paths:
    
    # Import the CSV.
    df = pd.read_csv(path, header=None)
    num_header_rows = df.index[df[0] == 'Custom name'][0] + 1
    header = df.loc[:num_header_rows-1]
    data   = df.loc[num_header_rows:].astype(float)
    
    # Remove columns where the absolute difference between the mean of the first 
    # third of data and the mean of the second third of data is less than
    # `diff_thresh`.
    keep_cols = []
    data_cols    = data.columns[3:].tolist()
    nondata_cols = data.columns[:3].tolist()
    mean_col     = nondata_cols[2]
    first_third  = int(len(data)*(1/3))
    second_third = int(len(data)*(2/3))
    for col in data_cols:
        mean_first_third  = data[col][:first_third].mean()
        mean_second_third = data[col][first_third:second_third].mean()
        diff              = abs(mean_first_third - mean_second_third)
        if diff >= 0.5:
            keep_cols += [col]
    data[mean_col] = data[keep_cols].mean(axis=1)
    new_cols = nondata_cols + keep_cols
    
    # Remove columns with no signal.
    df = pd.concat([header, data])
    df = df[new_cols]
    
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
    file.write(f'Minimum threshold was {diff_thresh}\n\n')
    file.write('Cleaned files were:\n')
    for path in import_paths:
        file.write(os.path.basename(path)+'\n')
        