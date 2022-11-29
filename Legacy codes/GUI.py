import PySimpleGUI as sg
import pandas as pd
import numpy as np
import sys
import os
import webbrowser
from tdt import read_block
from copy import deepcopy

from PhotometryPeriEventv3                import PhotometryPeriEventv3
from PhotometryPreprocessingv3            import PhotometryPreprocessingv3
from PeriEventTTL                         import PeriEventTTL
from FED3_one_active_poke                 import FED3_one_active_poke
from FED3_both_active_pokes               import FED3_both_active_pokes
from FED3_pellet_retrievals               import FED3_pellet_retrievals
from Whole_recording_1_TTL                import Whole_recording_1_TTL
from Between_TTLs                         import Between_TTLs
from Convert_NPM_to_TDT_data              import Convert_NPM_to_TDT_data

default = {}
inputs  = {}

def camera(value):
    dict1 = {'Camera 1':'Cam1', 'Camera 2':'Cam2'}
    return(dict1[value])
def recognise_ZTM(value):
    return([] if value=='' else [int(value)])
def recognise_artifact(value):
    return(np.inf if value=='' else float(value))
def recognise_bool(value):
    dict1 = {'True':True, 'False':False}
    return(dict1[value])
def setups(name, letter):
    if ',' in letter:
        return(name)
    dict1 = {'ISOS':  {'Setup A':'_405A', 'Setup B':'_415A'},
             'GCaMP': {'Setup A':'_465A', 'Setup B':'_475A'}}
    return(dict1[name][letter])

# Check whether there is a settings excel file.

default['Settings'] = 'False'
sg.theme("DarkTeal2")
layout  = []
layout += [[sg.T("")],[sg.Text("Use settings from an excel file."), 
            sg.Combo(['True','False'],key="Settings",
            enable_events=True,default_value=default['Settings'])]]
layout += [[sg.T("")],[sg.Button("Submit")]]
window  = sg.Window('Photometry Analysis', layout)
while True:
    event, values = window.read()
    if event == sg.WIN_CLOSED or event=="Exit":
        window.close()
        sys.exit()
    elif event == "Submit":
        inputs["Settings"] = recognise_bool(values["Settings"])
        window.close()
        break
print('Do not use settings excel file' if inputs['Settings']=='False' else 'Use settings excel file')
    
# Put in the options from an excel file.

if inputs["Settings"] == True:
    
    inputs = {}
    default['Import settings'] = 'C:/Users/hazza/Desktop/Fibre photometry GUI/Settings0.xlsx'
    sg.theme("DarkTeal2")
    layout  = []
    layout += [[sg.T("")],[sg.Text("Choose the location of the excel file."), 
                sg.Input(key="Import settings",enable_events=True,default_text=default["Import settings"]),
                sg.FileBrowse(key="Import2")]]
    layout += [[sg.T("")],[sg.Button("Submit")]]
    window  = sg.Window('Photometry Analysis', layout)
    while True:
        event, values = window.read()
        if event == sg.WIN_CLOSED or event=="Exit":
            window.close()
            sys.exit()
        elif event == "Submit":
            inputs["Import settings"] = values["Import settings"]
            window.close()
            break
    
    df = pd.read_excel(inputs["Import settings"],header=None)
    df.columns = ['Options','Values']
    
    def clean_inputs(value):
        if type(value) == str:
            if ((value[0] == "[" and value[-1] == "]") or
                (value[0] == "(" and value[-1] == ")") or
                (value[0] == "{" and value[-1] == "}")):
                return(eval(value))
            elif value == 'inf':
                return(np.inf)
            elif value == 'Camera 1':
                return('Cam1')
            elif value == 'Camera 2':
                return('Cam2')
            else:
                return(value)
        else:
            return(value)
    
    df['Values'] = df['Values'].apply(clean_inputs)
    df.index = list(range(len(df)))
    num_lines = len(df)
    
    # Convert multiple inputs into a dictionary.
    i = 0
    while i < num_lines:
        list_TTL_dict = {}
        list_TTL_indices = []
        list_Barcode_dict = {}
        list_Barcode_indices = []
        list_Video_dict = {}
        list_Video_indices = []
        while i < num_lines:
            if pd.isna(df.loc[i]).all():
                break
            option = df.at[i,'Options']
            value = df.at[i,'Values']
            if ('fc TTL' in option) and ('CSV' not in option):
                list_TTL_indices += [i]
                list_TTL_dict[option[7:]] = value
            if ('fc Create barcode' in option):
                list_Barcode_indices += [i]
                list_Barcode_dict[option[18:]] = value
            if ('fc Create snippets' in option):
                list_Video_indices += [i]
                list_Video_dict[option[19:]] = value
            if option == 'fc Zero time point':
                if pd.isna(value):
                    df.at[i,'Values'] = []
                else:
                    df.at[i,'Values'] = [int(value)]
            i += 1
        if len(list_TTL_indices) != 0:
            df.at[list_TTL_indices[0], 'Values'] = list_TTL_dict
            df.at[list_TTL_indices[0], 'Options'] = 'fc TTL'
            df = df.drop(list_TTL_indices[1:])
        if len(list_Barcode_indices) != 0:
            df.at[list_Barcode_indices[0], 'Values'] = list_Barcode_dict
            df.at[list_Barcode_indices[0], 'Options'] = 'fc Create barcode'
            df = df.drop(list_Barcode_indices[1:])
        if len(list_Video_indices) != 0:
            df.at[list_Video_indices[0], 'Values'] = list_Video_dict
            df.at[list_Video_indices[0], 'Options'] = 'fc Create snippets'
            df = df.drop(list_Video_indices[1:])
        i += 1
    
    # Add blank rows at the start and end.
    blank_line = pd.DataFrame({'Options':[np.nan],'Values':[np.nan]})
    if pd.isna(df.iloc[0]).all() == False:
        df = pd.concat([blank_line, df])
    if pd.isna(df.iloc[-1]).all() == False:
        df = pd.concat([df, blank_line])
    
    # Find all the blank row indices.
    list_blanks = [i for i in range(len(df['Options'])) if pd.isna(df['Options'].iloc[i])]
    # Find the corresponding start and end pairs of the data.
    list_coord = [(list_blanks[i-1]+1, list_blanks[i]) for i in range(1,len(list_blanks))]
    # Create a dictionary for each data section separated by blanks.
    list_tanks = [dict(zip(df['Options'][cd[0]:cd[1]], df['Values'][cd[0]:cd[1]])) for cd in list_coord]
    
    # Some tanks in list_tanks incidicate to import subfolders.
    # In these cases, create many of these tanks and only change the import and export locations.
    # Add these to list_tanks_full.
    list_tanks_full = []
    
    for inputs in list_tanks:
        
        # Older versions of the settings files don't have settings for creating
        # barcodes or video snippets.
        # Add these in, so the code can reference these values.
        if (inputs['Analysis'] == 'Whole recording' and 
            'fc Create barcode Create?' not in inputs.keys()):
            inputs['fc Create barcode'] = {'Create?':False}
        if (inputs['Analysis'] == 'Peri-event TTL' and
            'fc Create snippets Create?' not in inputs.keys()):
            inputs['fc Create snippets'] = {'Create?':False}
        if ('fc ISOS' not in inputs.keys() and 'fc GCaMP' not in inputs.keys()):
            inputs['fc ISOS']  = False
            inputs['fc GCaMP'] = False
        # The older version of the settings files don't have an active poke entry.
        if (inputs['Analysis'] == 'FED3' and 'fc Active poke' not in inputs.keys()):
            # Add an active poke entry betweeen 'fc Poke to analyse' and 
            # 'fc Zero time point'.
            options     = list(inputs.keys())
            ind         = options.index('fc Zero time point')
            options_upd = options[:ind] + ['Active poke'] + options[ind:]
            updated = {}
            for key in options_upd:
                if key == 'Active poke':
                    updated[key] = 'Changing'
                else:
                    updated[key] = inputs[key]
            inputs = deepcopy(updated)
            
        if inputs["Import subfolders"] == False:
            list_tanks_full.append(inputs)
            continue

        import_loc = inputs["fc Import location"]
        export_loc = inputs["fc Export location"]
        for folder in os.listdir(import_loc):
            if os.path.isfile(import_loc+'/'+folder) == True:
                continue
            list_tanks_full.append(deepcopy(inputs))
            list_tanks_full[-1]['fc Import location'] = import_loc+'/'+folder
            list_tanks_full[-1]['fc Export location'] = export_loc+'/'+folder
            
    # Check that the input variables/rows in the settings excel file are correct
    # for the given type of analysis (whole recording, peri-event, etc.)
    basic_options = ['Import subfolders', 'System', 'Analysis']
    peri_event_options = ['']
    
    # Run the correct code, based on the information in the settings excel file.
    for inputs in list_tanks_full:
        
        print('')
        
        # if inputs['System'] == 'NPM (Neurophotometrics)':
        
        #     if inputs['Analysis'] == 'Pre-processing':
        #         arguments = [inputs[key] for key in inputs.keys() if key[:2]=='fc']
        #         PhotometryPreprocessingv3(*arguments)
                
        #     if inputs['Analysis'] == 'Peri-event analysis':
        #         arguments = [inputs[key] for key in inputs.keys() if key[:2]=='fc']
        #         PhotometryPeriEventv3(*arguments)
            
        if inputs['System'] == 'TDT (Tucker-Davis Technologies)':
        
            if inputs['Analysis'] == 'Peri-event TTL':

                arguments = [inputs[key] for key in inputs.keys() if key[:2]=='fc']
                PeriEventTTL(*arguments) # Run the corresponding code.

            if inputs['Analysis'] == 'FED3':        

                arguments = [inputs[key] for key in inputs.keys() if key[:2]=='fc']
                if inputs['fc Poke to analyse'] == 'Left' or inputs['fc Poke to analyse'] == 'Right':
                    FED3_one_active_poke(*arguments)
                    FED3_pellet_retrievals(*arguments)
                elif inputs['fc Poke to analyse'] == 'Both':
                    FED3_both_active_pokes(*arguments)
                    FED3_pellet_retrievals(*arguments)

            if inputs['Analysis'] == 'Whole recording':         

                arguments = [inputs[key] for key in inputs.keys() if key[:2]=='fc']
                Whole_recording_1_TTL(*arguments)
                # if inputs['No. TTLs'] == 1:
                #     Whole_recording_1_TTL(*arguments)
                # elif inputs['No. TTLs'] == 2:
                #     Whole_recording_2_TTLs(*arguments)       
                    
            if inputs['Analysis'] == 'Between TTLs':         
                
                arguments = [inputs[key] for key in inputs.keys() if key[:2]=='fc']
                Between_TTLs(*arguments)
                
        print('Inputs are:')
        for key in inputs.keys():
            print(key+': '+str(inputs[key]))
            
    sys.exit()
                
# Put in the options manually with a GUI.

#-----------------------------------------------------------------------------#
# Choose whether to analyse TDT or NPM data.
#-----------------------------------------------------------------------------#

inputs = {}
default['System'] = 'TDT (Tucker-Davis Technologies)'
sg.theme("DarkTeal2")
layout  = []
layout += [[sg.T("")],[sg.Text("Choose the system"), 
            sg.Combo(['TDT (Tucker-Davis Technologies)','NPM (Neurophotometrics)'],
            key="System",enable_events=True,default_value=default["System"])]]
layout += [[sg.T("")],[sg.Button("Submit")]]
window  = sg.Window('Photometry Analysis', layout)
while True:
    event, values = window.read()
    if event == sg.WIN_CLOSED or event=="Exit":
        window.close()
        sys.exit()
    elif event == "Submit":
        inputs["System"] = values["System"]
        window.close()
        break
print('System is '+inputs['System'])

#-----------------------------------------------------------------------------#
# NPM -> choose the type of analysis
#-----------------------------------------------------------------------------#
    
if inputs['System'] == 'NPM (Neurophotometrics)':
    # Choose the import location, export location and setup.
    default["Import location"] = 'C:/Users/hazza/Documents/NAc_A3088W_B3089W-211213-101735'
    default["Export location"] = 'C:/Users/hazza/Documents/NAc_A3088W_B3089W-211213-101735'
    default["Subfolders"]      = 'False'
    default["ISOS wavelength"] = '415'
    default["ISOS color"]      = '0 green'
    default["GCaMP wavelength"]= '470'
    default["GCaMP color"]     = '0 green'
    default["Analysis"]        = 'Peri-event TTL'
    sg.theme("DarkTeal2")
    layout  = []
    layout += [[sg.T("")], [sg.Text("Choose a folder for the import location"), 
                sg.Input(key="Import" ,enable_events=True,default_text=default["Import location"]),
                sg.FolderBrowse(key="Import2")]]
    layout += [[sg.T("")], [sg.Text("Choose a folder for the export location"),
                sg.Input(key="Export" ,enable_events=True,default_text=default["Import location"]),
                sg.FolderBrowse(key="Export2")]]
    # layout += [[sg.T("")],[sg.Text("Use the subfolders within these import and export locations"), 
    #             sg.Combo(['True','False'],key="Subfolders",enable_events=True,default_value=default["Subfolders"])]]
    layout += [[sg.T("")],[sg.Text("Choose the wavelength and color for the ISOS channel", size=(41,1)), 
                sg.Combo(['415', '470', '560'],key="ISOS wavelength",enable_events=True,default_value=default["ISOS wavelength"]),
                sg.Combo(['0 green', '1 red', '2 green', '3 red'],key="ISOS color",enable_events=True,default_value=default["ISOS color"])]]
    layout += [[sg.T("")],[sg.Text("Choose the wavelength and color for the GCaMP channel", size=(41,1)), 
                sg.Combo(['415', '470', '560'],key="GCaMP wavelength",enable_events=True,default_value=default["GCaMP wavelength"]),
                sg.Combo(['0 green', '1 red', '2 green', '3 red'],key="GCaMP color",enable_events=True,default_value=default["GCaMP color"])]]
    layout += [[sg.T("")],[sg.Text("Choose the type of analysis."), 
                sg.Combo(['Peri-event TTL', 'FED3', 'Whole recording', 'Between TTLs'],
                key="Analysis",enable_events=True,default_value=default["Analysis"])]]
    layout += [[sg.T("")], [sg.Button("Submit")]]
    window  = sg.Window('Photometry Analysis', layout)
    while True:
        event, values = window.read()
        if event == sg.WIN_CLOSED or event=="Exit":
            window.close()
            sys.exit()
        elif event == "Submit":
            inputs['Import location'] = values["Import"]
            inputs['Export location'] = values["Export"]
            # inputs['Subfolders']      = recognise_bool(values["Subfolders"])
            inputs['Setup']           = (values["ISOS wavelength"] +"_"+values["ISOS color" ].replace(' ','_')+","+
                                         values["GCaMP wavelength"]+"_"+values["GCaMP color"].replace(' ','_'))
            inputs['Analysis']        = values["Analysis"]
            window.close()
            break
    print('Import location is '+inputs['Import location'])
    print('Export location is '+inputs['Export location'])
    print('Setup is '+inputs['Setup'])
    print('Type of analysis is '+inputs['Analysis'])
    
    # Import the NPM data.
    print('\nPlease wait while the NPM data is importing...')
    data = Convert_NPM_to_TDT_data(inputs['Import location'])
    print('')
    inputs['TTL'] = {}
    
#-----------------------------------------------------------------------------#
# NPM -> choose the type of analysis
#-----------------------------------------------------------------------------#
    
    if inputs['Analysis'] in ['Peri-event TTL', 'Whole recording']:
        # Choose the type of TTL.
        default["TTL"] = list(data.epocs.keys())[0]
        default["Custom"] = "TTL"
        sg.theme("DarkTeal2")
        layout  = []
        layout += [[sg.T("")],[sg.Text("Choose the name of the TTL event."), 
                    sg.Combo(list(data.epocs.keys()), key="TTL", enable_events=True, default_value=default["TTL"])]]
        layout += [[sg.T("")], [sg.Text("Choose a custom name for this event."), 
                    sg.Input(key="Custom",enable_events=True,default_text=default["Custom"])]]
        layout += [[sg.T("")], [sg.Button("Submit")]]
        window  = sg.Window('Photometry Analysis', layout)
        while True:
            event, values = window.read()
            if event == sg.WIN_CLOSED or event=="Exit":
                window.close()
                sys.exit()
            elif event == "Submit":
                inputs['TTL']['Name']   = values["TTL"]
                inputs['TTL']['Custom'] = values["Custom"]
                window.close()
                break
        print('The name of the TTL pulse is '+inputs['TTL']['Name'])
        
# #-----------------------------------------------------------------------------#
# # NPM -> choose the type of analysis
# #-----------------------------------------------------------------------------#
        
#         if inputs['Poke to analyse'] in ['Left', 'Right']:

#             # Choose the type of TTL.
#             default["Left_right"] = list(data.epocs.keys())[0]
#             default["Pellet"]     = list(data.epocs.keys())[0]
#             default["Custom"] = "TTL"
#             sg.theme("DarkTeal2")
#             layout  = []
#             layout += [[sg.T("")],[sg.Text("Choose the "+inputs['Poke to analyse']+" event."), 
#                         sg.Combo(list(data.epocs.keys()), key="Left_right", enable_events=True, default_value=default["Left_right"])]]
#             layout += [[sg.T("")],[sg.Text("Choose the pellet retrieval event."), 
#                         sg.Combo(list(data.epocs.keys()), key="Pellet", enable_events=True, default_value=default["Pellet"])]]
#             layout += [[sg.T("")], [sg.Button("Submit")]]
#             window  = sg.Window('Photometry Analysis', layout)
#             while True:
#                 event, values = window.read()
#                 if event == sg.WIN_CLOSED or event=="Exit":
#                     window.close()
#                     sys.exit()
#                 elif event == "Submit":
#                     inputs['TTL']['Name']   = [values["Left_right"], values['Pellet']]
#                     inputs['TTL']['Custom'] = [inputs['Poke to analyse'], 'Pellet']
#                     window.close()
#                     break
#             print('The name of the TTL pulse is '+inputs['TTL']['Name'])
            
# #-----------------------------------------------------------------------------#
# # NPM -> choose the type of analysis
# #-----------------------------------------------------------------------------#
            
#         elif inputs['Poke to analyse'] == 'Both':
        
#             # Choose the type of TTL.
#             default["Left"]   = list(data.epocs.keys())[0]
#             default["Right"]  = list(data.epocs.keys())[0]
#             default["Pellet"] = list(data.epocs.keys())[0]
#             default["Custom"] = "TTL"
#             sg.theme("DarkTeal2")
#             layout  = []
#             layout += [[sg.T("")],[sg.Text("Choose the left event."), 
#                         sg.Combo(list(data.epocs.keys()), key="Left", enable_events=True, default_value=default["Left"])]]
#             layout += [[sg.T("")],[sg.Text("Choose the right event."), 
#                         sg.Combo(list(data.epocs.keys()), key="Right", enable_events=True, default_value=default["Right"])]]
#             layout += [[sg.T("")],[sg.Text("Choose the pellet retrieval event."), 
#                         sg.Combo(list(data.epocs.keys()), key="Pellet", enable_events=True, default_value=default["Pellet"])]]
#             layout += [[sg.T("")], [sg.Button("Submit")]]
#             window  = sg.Window('Photometry Analysis', layout)
#             while True:
#                 event, values = window.read()
#                 if event == sg.WIN_CLOSED or event=="Exit":
#                     window.close()
#                     sys.exit()
#                 elif event == "Submit":
#                     inputs['TTL']['Name']   = [values["Left"], values["Right"], values['Pellet']]
#                     inputs['TTL']['Custom'] = ['Left', 'Right', 'Pellet']
#                     window.close()
#                     break
#             print('The name of the TTL pulse is '+inputs['TTL']['Name'])

# #-----------------------------------------------------------------------------#
# # NPM -> choose the type of analysis
# #-----------------------------------------------------------------------------#

# if inputs['System'] == 'NPM (Neurophotometrics)':
#     # Choose the type of analysis.
#     default['Analysis'] = 'Pre-processing'
#     sg.theme("DarkTeal2")
#     layout  = []
#     layout += [[sg.T("")],[sg.Text("Choose the type of analysis"), 
#                 sg.Combo(['Pre-processing','Peri-event analysis'],key="Analysis",
#                 enable_events=True,default_value=default['Analysis'])]]
#     layout += [[sg.T("")],[sg.Button("Submit")]]
#     window  = sg.Window('Photometry Analysis', layout)
#     while True:
#         event, values = window.read()
#         if event == sg.WIN_CLOSED or event=="Exit":
#             window.close()
#             sys.exit()
#         elif event == "Submit":
#             inputs["Analysis"] = values["Analysis"]
#             window.close()
#             break
#     print('Type of analysis is '+inputs['Analysis'])

# #-----------------------------------------------------------------------------#
# # NPM -> pre-processing -> select options before running the code
# #-----------------------------------------------------------------------------#

#     if inputs['Analysis'] == 'Pre-processing':
#         default["Import and export location"]  = 'C:/Users/hazza/Desktop/Fibre photometry GUI/Photometry tanks/NAc_3147K_2614K-211118-105517'
#         default['Start time']  = 180
#         default['Duration'] = 3600
#         default['Capture rate'] = 20
#         default['StartBaseline'] = 5155
#         default['DurationBaseline'] = 15
#         sg.theme("DarkTeal2")
#         layout = []
#         layout += [[sg.T("")], [sg.Text("Created by Nick Everett.")]]
#         layout += [[sg.T("")], [sg.Text("Choose a folder for the import and export location." + 
#                                         "It should contain the 5 files (470, 410, cameracsv, DLCtracking, behaviour)")]]
#         layout += [[sg.T("")], [sg.Input(key="Import" ,enable_events=True,default_text=default["Import and export location"]),
#                                 sg.FolderBrowse(key="Import2")]]
#         layout += [[sg.T("")], [sg.Text("Input start time and duration of TOTAL recording in seconds, for cropping.")]]
#         layout += [[sg.T("")], [sg.Text("Start time (if using this for peri-event, keep at 0. it will cut the file to this point)"), 
#                                 sg.Input(key="Start time",enable_events=True,default_text=default["Start time"],size=(20,1))]]
#         layout += [[sg.T("")], [sg.Text("Duration (make this at least the length of the session)"),
#                                 sg.Input(key="Duration",enable_events=True,default_text=default["Duration"],size=(20,1))]]
#         layout += [[sg.T("")], [sg.Text("Capture rate on NPM system for the 470nm channel"),
#                                 sg.Input(key="Capture rate",enable_events=True,default_text=default["Capture rate"],size=(20,1))]]
#         layout += [[sg.T("")], [sg.Text("Define the start of the baseline period."),
#                                 sg.Input(key="StartBaseline",enable_events=True,default_text=default["StartBaseline"],size=(20,1))]]
#         layout += [[sg.T("")], [sg.Text("Define the baseline duration (in minutes)."), 
#                                 sg.Input(key="DurationBaseline",enable_events=True,default_text=default["DurationBaseline"],size=(20,1))]]
#         layout += [[sg.T("")],[sg.Button("Submit")]]
#         window = sg.Window('Photometry Analysis', layout)
#         while True:
#             event, values = window.read()
#             if event == sg.WIN_CLOSED or event=="Exit":
#                 window.close()
#                 sys.exit()
#             elif event == "Submit":
#                 inputs['fc Start time']                 = float(values["Start time"])
#                 inputs['fc Duration']                   = float(values["Duration"])
#                 inputs['fc Capture rate']               = float(values["Capture rate"])
#                 inputs['fc Import and export location'] = values["Import"]   
#                 inputs['fc StartBaseline']              = float(values["StartBaseline"])
#                 inputs['fc DurationBaseline']           = float(values["DurationBaseline"])
#                 window.close()
#                 break
#         print('Import and export location is '+inputs['fc Import and export location'])
#         print('Start time is '+str(inputs['fc Start time']))
#         print('Duration is '+str(inputs['fc Duration']))
#         print('Capture rate is '+str(inputs['fc Capture rate']))
#         print('Start of baseline is '+str(inputs['fc StartBaseline']))
#         print('Duration of baseline is '+str(inputs['fc DurationBaseline']))
#         arguments = [inputs[key] for key in inputs.keys() if key[:2]=='fc']
#         PhotometryPreprocessingv3(*arguments)

# #-----------------------------------------------------------------------------#
# # NPM -> peri-event analysis -> select options before running the code
# #-----------------------------------------------------------------------------#
        
#     if inputs['Analysis'] == 'Peri-event analysis':
#         default["Import and export location"]  = 'C:/Users/hazza/Desktop/Fibre photometry GUI/Photometry tanks/NAc_3147K_2614K-211118-105517'
#         default['Pre-window']  = 5.0
#         default['Post-window'] = 5.0
#         default['Capture rate'] = 20
#         sg.theme("DarkTeal2")
#         layout = []
#         layout += [[sg.T("")], [sg.Text("Created by Nick Everett.")]]
#         layout += [[sg.T("")], [sg.Text("Choose a folder for the import and export location"), 
#                                 sg.Input(key="Import" ,enable_events=True,default_text=default["Import and export location"]), 
#                                 sg.FolderBrowse(key="Import2")]]
#         layout += [[sg.T("")], [sg.Text("Ensure that the values below have 1 decimal place (eg 5.1, 2.7).")]]
#         layout += [[sg.T("")], [sg.Text("Choose the pre-window in seconds (how long before the behaviour onset do you want to extract?)"),
#                                 sg.Input(key="Pre-window",enable_events=True,default_text=default["Pre-window"],size=(20,1))]]
#         layout += [[sg.T("")], [sg.Text("Choose the post-window in seconds (how long after the behaviour onset do you want to extract?)"),
#                                 sg.Input(key="Post-window",enable_events=True,default_text=default["Post-window"],size=(20,1))]]
#         layout += [[sg.T("")], [sg.Text("Choose the capture rate of the photometry system in Hz.")]]
#         layout += [[sg.T("")], [sg.Text("If NPM is capturing at 40fps, and you're 1:1 interleaved with 470/415, then put in 20 Hz."), 
#                                 sg.Input(key="Capture rate",enable_events=True,default_text=default["Capture rate"],size=(20,1))]]
#         layout += [[sg.T("")], [sg.Button("Submit")]]
#         window = sg.Window('Photometry Analysis', layout)
#         while True:
#             event, values = window.read()
#             if event == sg.WIN_CLOSED or event=="Exit":
#                 window.close()
#                 exit()
#             elif event == "Submit":    
#                 inputs['fc Pre-window']                 = float(values["Pre-window"])
#                 inputs['fc Post-window']                = float(values["Post-window"])
#                 inputs['fc Import and export location'] = values["Import"]   
#                 inputs['fc Capture rate']               = float(values["Capture rate"])
#                 window.close()
#                 break
#         print('Import and export location is '+inputs['fc Import'])
#         print('Start time is '+str(inputs['fc Start time']))
#         print('Duration is '+str(inputs['fc Duration']))
#         print('Capture rate is '+str(inputs['fc Capture rate']))
#         print('Start of baseline is '+str(inputs['fc StartBaseline']))
#         print('Duration of baseline is '+str(inputs['fc DurationBaseline']))
#         arguments = [inputs[key] for key in inputs.keys() if key[:2]=='fc']
#         PhotometryPeriEventv3(*arguments)
        
#-----------------------------------------------------------------------------#
# TDT -> choose the type of analysis and more options
#-----------------------------------------------------------------------------#
    
if inputs['System'] == 'TDT (Tucker-Davis Technologies)':
    # Choose the import location, export location and setup.
    default["Import location"] = 'C:/Users/hazza/Documents/NAc_A3088W_B3089W-211213-101735'
    default["Export location"] = 'C:/Users/hazza/Documents/NAc_A3088W_B3089W-211213-101735'
    default["Subfolders"]      = 'False'
    default["Setup"]           = 'Setup A'
    default["Analysis"]        = 'Peri-event TTL'
    sg.theme("DarkTeal2")
    layout  = []
    layout += [[sg.T("")], [sg.Text("Choose a folder for the import location"), 
                sg.Input(key="Import" ,enable_events=True,default_text=default["Import location"]),
                sg.FolderBrowse(key="Import2")]]
    layout += [[sg.T("")], [sg.Text("Choose a folder for the export location"),
                sg.Input(key="Export" ,enable_events=True,default_text=default["Import location"]),
                sg.FolderBrowse(key="Export2")]]
    # layout += [[sg.T("")],[sg.Text("Use the subfolders within these import and export locations"), 
    #             sg.Combo(['True','False'],key="Subfolders",enable_events=True,default_value=default["Subfolders"])]]
    layout += [[sg.T("")],[sg.Text("Choose setup A or setup B"), 
                sg.Combo(['Setup A','Setup B'],key="Setup",enable_events=True,default_value=default["Setup"])]]
    layout += [[sg.T("")],[sg.Text("Choose the type of analysis."), 
                sg.Combo(['Peri-event TTL', 'FED3', 'Whole recording', 'Between TTLs'],
                key="Analysis",enable_events=True,default_value=default["Analysis"])]]
    layout += [[sg.T("")], [sg.Button("Submit")]]
    window  = sg.Window('Photometry Analysis', layout)
    while True:
        event, values = window.read()
        if event == sg.WIN_CLOSED or event=="Exit":
            window.close()
            sys.exit()
        elif event == "Submit":
            inputs['Import location'] = values["Import"]
            inputs['Export location'] = values["Export"]
            # inputs['Subfolders']      = recognise_bool(values["Subfolders"])
            inputs['Setup']           = values["Setup"]
            inputs['Analysis']        = values["Analysis"]
            window.close()
            break
    print('Import location is '+inputs['Import location'])
    print('Export location is '+inputs['Export location'])
    print('Setup is '+inputs['Setup'])
    print('Type of analysis is '+inputs['Analysis'])

# list_tanks_full = []
# import_loc = inputs["Import location"]
# export_loc = inputs["Export location"]
# if inputs['Subfolders'] == True:
#     for folder in os.listdir(import_loc):
#         if os.path.isfile(import_loc+'/'+folder) == True:
#             continue
#         list_tanks_full.append(deepcopy(inputs))
#         list_tanks_full[-1]['Import location'] = import_loc+'/'+folder
#         list_tanks_full[-1]['Export location'] = export_loc+'/'+folder
#     inputs = list_tanks_full[0]
# else:
#     list_tanks_full.append(inputs)

#-----------------------------------------------------------------------------#
# TDT -> peri-event TTL -> choose the type of TTL pulse
#-----------------------------------------------------------------------------#

    if inputs['Analysis'] in ['Peri-event TTL', 'Whole recording']:
        # Choose the type of TTL.
        default["TTL"] = 'TTLM'
        sg.theme("DarkTeal2")
        layout  = []
        layout += [[sg.T("")],[sg.Text("Choose the type of TTL pulse."), 
                    sg.Combo(['TTLM','Note','Video timestamp','Ethovision'],
                    key="TTL",enable_events=True,default_value=default["TTL"])]]
        layout += [[sg.T("")], [sg.Button("Submit")]]
        window  = sg.Window('Photometry Analysis', layout)
        while True:
            event, values = window.read()
            if event == sg.WIN_CLOSED or event=="Exit":
                window.close()
                sys.exit()
            elif event == "Submit":
                inputs['TTL'] = {'Type':values["TTL"]}
                if inputs['TTL']['Type'] == 'TTLM':
                    # Find the list of notes using the read_block function.
                    print('\nPlease wait while the TDT tank is importing...')
                    data = read_block(inputs['Import location'])
                    print('')
                    # Create a list of system variables to exclude from possible TTLMs.
                    TTLM_exclude = ['PC0_','PC1_','PC2_','PC3_','PC0/','PC1/','PC2/','PC3/',
                                    'Cam1','Cam2','BGte','Gate','Note','Tick']
                    # If using setup A, exclude the score events from setup B and vice versa.
                    if inputs['Setup'] == 'Setup A':
                        TTLM_exclude += ['Blft','Brgt','BRgt','Bplt']
                    elif inputs['Setup'] == 'Setup B':
                        TTLM_exclude += ['Alft','Argt','ARgt','Aplt']
                    TTLM_list = data.epocs.keys()
                    TTLM_list = list(set(TTLM_list)) # Remove non-unique elements.
                    inputs['TTLM list'] = [TTLM for TTLM in TTLM_list if TTLM not in TTLM_exclude]
                    if len(inputs['TTLM list']) == 0:
                        print('There are no TTLMs to choose from. Try a note or video timestamp.')
                        window.close()
                        sys.exit()
                elif inputs['TTL']['Type'] == 'Note':
                    # Find the list of notes using the read_block function.
                    print('\nPlease wait while the TDT tank is importing...')
                    data = read_block(inputs['Import location'])
                    print('')
                    # If the notes are all 'none', go back into the Notes.txt file and find comments in "" marks.
                    if 'Note' in data.epocs.keys() and np.all(data.epocs.Note.notes == 'none'):
                        notes_txt_path = os.path.join(inputs['Import location'], 'Notes.txt')
                        with open(notes_txt_path, 'r') as notes_file:
                            notes_lines = notes_file.readlines()
                        def find_comment(note):
                            ind = [i for i in range(len(note)) if note[i]=='"']
                            return(note[ind[0]+1:ind[1]])
                        notes_lines = [find_comment(note) for note in notes_lines if note[:5]=='Note-']
                        data.epocs.Note.notes = np.array(notes_lines)
                    #####################################
                    # Create a list of notes.
                    inputs['Notes list'] = list(data.epocs.Note.notes)
                    inputs['Notes list'] = list(set(inputs['Notes list'])) # Remove non-unique elements.
                elif inputs['TTL']['Type'] == 'Video timestamp':
                    inputs['Video timestamp list'] = ['Camera 1', 'Camera 2']
                elif inputs['TTL']['Type'] == 'Ethovision':
                    print('\nPlease wait while the Ethovision excel file is importing...')
                    # Check all the excel files in the import tank, and only use the one from Ethovision.
                    excel_files = [file for file in os.listdir(inputs['Import location']) if file[-5:]=='.xlsx']
                    excel_file_found = False
                    for file in excel_files:
                        import_destination = inputs['Import location']+'/'+file
                        df = pd.read_excel(import_destination, sheet_name=0)
                        if list(df[:0])[0] == 'Number of header lines:':
                            excel_file_found = True
                            break
                    if excel_file_found == False:
                        print('File not found.')
                        print('Please put the raw data Ethovision excel file in the folder of the import tank.')
                        window.close()
                        sys.exit()
                    # Find all the column headings in the Ethovision files (that are not position, speed, etc.)
                    num_headers = int(list(df[:0])[1])
                    rows_skip = list(range(0,num_headers-2)) + [num_headers-1]
                    headings_to_exclude = ["Trial time", "Recording time", "X center", "Y center", "X nose", "Y nose", 
                                           "X tail", "Y tail", "Area", "Areachange", "Elongation", "Direction", 
                                           "Distance moved(nose-point)", "Distance moved(center-point)", "Velocity(nose-point)", 
                                           "Velocity(center-point)", "Result 1"]
                    # Exclude the columns from headings_to_exclude.
                    df = pd.read_excel(import_destination, sheet_name=0, skiprows=rows_skip, 
                                       usecols=lambda x: x not in headings_to_exclude)
                    inputs['Ethovision event list'] = list(df.columns)
                window.close()
                break
        print('Type of TTL pulse is '+inputs['TTL']['Type'])
        
    #-----------------------------------------------------------------------------#
    # TDT -> peri-event TTL -> TTLM -> choose the name of the TTL event
    #-----------------------------------------------------------------------------#
    
        if inputs['TTL']['Type'] == 'TTLM':
            # Choose the type of TTL.
            default["TTLM name"] = inputs['TTLM list'][0]
            default["Custom"] = "TTL"
            sg.theme("DarkTeal2")
            layout  = []
            layout += [[sg.T("")],[sg.Text("Choose the name of the TTL event."), 
                        sg.Combo(inputs["TTLM list"],key="TTLM name",enable_events=True,default_value=default["TTLM name"])]]
            layout += [[sg.T("")], [sg.Text("Choose a custom name for this event."), 
                        sg.Input(key="Custom",enable_events=True,default_text=default["Custom"])]]
            layout += [[sg.T("")], [sg.Button("Submit")]]
            window  = sg.Window('Photometry Analysis', layout)
            while True:
                event, values = window.read()
                if event == sg.WIN_CLOSED or event=="Exit":
                    window.close()
                    sys.exit()
                elif event == "Submit":
                    inputs['TTL']['Name'] = values["TTLM name"]
                    inputs['TTL']['Custom'] = values["Custom"]
                    window.close()
                    break
            print('The name of the TTL pulse is '+inputs['TTL']['Name'])
            
    #-----------------------------------------------------------------------------#
    # TDT -> peri-event TTL -> note -> choose the number of notes
    #-----------------------------------------------------------------------------#        
    
        elif inputs['TTL']['Type'] == 'Note':
            default["No. notes"] = 'All'
            default["Custom"] = 'Note'
            sg.theme("DarkTeal2")
            layout  = []
            layout += [[sg.T("")],[sg.Text("Choose the number of notes to count as the same event."), 
                        sg.Combo(['All']+list(range(1,len(inputs['Notes list']))),
                        key="No. notes",enable_events=True,default_value=default["No. notes"])]]
            layout += [[sg.T("")], [sg.Text("Choose a custom name for this event."), 
                        sg.Input(key="Custom",enable_events=True,default_text=default["Custom"])]]
            layout += [[sg.T("")], [sg.Button("Submit")]]
            window  = sg.Window('Photometry Analysis', layout)
            while True:
                event, values = window.read()
                if event == sg.WIN_CLOSED or event=="Exit":
                    window.close()
                    sys.exit()
                elif event == "Submit":
                    inputs['TTL']['No. notes'] = values["No. notes"]
                    inputs['TTL']['Custom'] = values["Custom"]
                    window.close()
                    break
            print('Number of notes to count as the same event: '+str(inputs['TTL']['No. notes']))
                
    #-----------------------------------------------------------------------------#
    # TDT -> peri-event TTL -> note -> 1,2,... notes -> choose which notes
    #-----------------------------------------------------------------------------#      
            
            if inputs['TTL']['No. notes'] != 'All':
                default['Notes list'] = inputs['Notes list'][0]
                list_combos = []
                for i in range(inputs['TTL']['No. notes']):
                    list_combos += [sg.Combo(inputs['Notes list'],key='Note'+str(i),enable_events=True,default_value=default["Notes list"])]
                sg.theme("DarkTeal2")
                layout  = []
                layout += [[sg.T("")], [sg.Text("Choose which notes should count as the same event."),*list_combos]]
                layout += [[sg.T("")], [sg.Button("Submit")]]                    
                window  = sg.Window('Photometry Analysis', layout)
                while True:
                    event, values = window.read()
                    if event == sg.WIN_CLOSED or event=="Exit":
                        window.close()
                        sys.exit()
                    elif event == "Submit":
                        inputs['TTL']['Name'] = [values["Note"+str(i)] for i in range(inputs['TTL']['No. notes'])]
                        window.close()
                        break
                print('Use these notes as the same event: '+', '.join(inputs['TTL']['Name']))            
    
    #-----------------------------------------------------------------------------#
    # TDT -> peri-event TTL -> video timestamp -> choose the camera            
    #-----------------------------------------------------------------------------#    
            
        elif inputs['TTL']['Type'] == 'Video timestamp':
            default["Camera"] = 'Camera 1'
            default["Custom"] = 'Video_timestamp'
            sg.theme("DarkTeal2")
            layout  = []
            layout += [[sg.T("")], [sg.Text("Choose the camera."),sg.Combo(['Camera 1', 'Camera 2'],
                                   key="Camera",enable_events=True,default_value=default["Camera"])]]
            layout += [[sg.T("")], [sg.Text("Choose a custom name for this event."), 
                        sg.Input(key="Custom",enable_events=True,default_text=default["Custom"])]]
            layout += [[sg.T("")], [sg.Button("Submit")]]
            window  = sg.Window('Photometry Analysis', layout)
            while True:
                event, values = window.read()
                if event == sg.WIN_CLOSED or event=="Exit":
                    window.close()
                    sys.exit()
                elif event == "Submit":
                    inputs['TTL']['Camera'] = camera(values["Camera"])
                    inputs['TTL']['Custom'] = values["Custom"]
                    window.close()
                    break
            print('Use '+str(inputs['TTL']['Camera']))
            
    #-----------------------------------------------------------------------------#
    # TDT -> peri-event TTL -> Ethovision -> choose the name of the Ethovision event
    #-----------------------------------------------------------------------------#
    
        if inputs['TTL']['Type'] == 'Ethovision':
            # Choose the type of TTL.
            default["Ethovision name"] = inputs['Ethovision event list'][0]
            default["Custom"] = "TTL"
            sg.theme("DarkTeal2")
            layout  = []
            layout += [[sg.T("")], [sg.Text("Ensure the raw data Ethovision files are in the import tank.\n"
                                            "Edit the filenames, so that the excel files start with 'Setup A' or 'Setup B'.\n"
                                            "Choose the column heading, which corresponds to your event.\n")]]
            layout += [[sg.Combo(inputs['Ethovision event list'],key='Ethovision name',enable_events=True,default_value=default["Ethovision name"])]]
            layout += [[sg.T("")], [sg.Text("Choose a custom name for this event."), 
                                    sg.Input(key="Custom",enable_events=True,default_text=default["Custom"])]]
            layout += [[sg.T("")], [sg.Button("Submit")]]
            window  = sg.Window('Photometry Analysis', layout)
            while True:
                event, values = window.read()
                if event == sg.WIN_CLOSED or event=="Exit":
                    window.close()
                    sys.exit()
                elif event == "Submit":
                    inputs['TTL']['Name'] = values["Ethovision name"]
                    inputs['TTL']['Custom'] = values["Custom"]
                    window.close()
                    break
            print('The name of the TTL pulse is '+inputs['TTL']['Name'])
    
#-----------------------------------------------------------------------------#
# TDT -> peri-event TTL -> ... -> select options before running the code     
#-----------------------------------------------------------------------------#

if inputs['Analysis'] == 'Peri-event TTL':
    
    default["Zero time point"]  = ['']
    default["TRANGE"]           = [-20,80]
    default["Baseline period"]  = [-20,-5]
    default["Artifact RL"]      = ''     
    default["zScore"]           = 'True'
    default["dFF"]              = 'False'
    default["Video"]            = 'False'
    sg.theme("DarkTeal2")
    layout = []
    layout += [[sg.T("")], [sg.Text("Choose the TRANGE window size (start time relative to epoc onset in seconds, window duration)"), 
                            sg.Input(key="TRANGE1",enable_events=True,default_text=default["TRANGE"][0],size=(10,1)), 
                            sg.Input(key="TRANGE2",enable_events=True,default_text=default["TRANGE"][1],size=(10,1))]]
    layout += [[sg.T("")], [sg.Text("Choose the baseline period within the window"), 
                            sg.Input(key="BASELINE1",enable_events=True,default_text=default["Baseline period"][0],size=(10,1)),
                            sg.Input(key="BASELINE2",enable_events=True,default_text=default["Baseline period"][1],size=(10,1))]]
    layout += [[sg.T("")], [sg.Text("Choose the artifact rejection level (optional)"), 
                            sg.Input(key="Artifact",enable_events=True,default_text=default["Artifact RL"],size=(20,1))]]
    layout += [[sg.T("")], [sg.Text("Choose the TTL number you want as the zero time point (optional)."), 
                            sg.Input(key="ZTP",enable_events=True,default_text=default["Zero time point"][0],size=(20,1))]]
    layout += [[sg.T("")], [sg.Text("Save Z-Score data to CSV?"), 
                            sg.Combo(['True','False'],key="zScore",enable_events=True,default_value=default['zScore'])]]
    layout += [[sg.T("")], [sg.Text("Save dFF data to CSV?"), 
                            sg.Combo(['True','False'],key="dFF",enable_events=True,default_value=default['dFF'])]]
    layout += [[sg.T("")], [sg.Text("Create video snippets of epochs?"), 
                            sg.Combo(['True','False'],key="Video",enable_events=True,default_value=default['Video'])]]
    layout += [[sg.T("")], [sg.Text("Save ISOS data to CSV?"), 
                            sg.Combo(['True','False'],key="ISOS",enable_events=True,default_value=default['dFF'])]]
    layout += [[sg.T("")], [sg.Text("Save GCaMP data to CSV?"), 
                            sg.Combo(['True','False'],key="GCaMP",enable_events=True,default_value=default['dFF'])]]
    layout += [[sg.T("")], [sg.Button("Submit")]]
    window = sg.Window('Photometry Analysis', layout)
    while True:
        event, values = window.read()
        if event == sg.WIN_CLOSED or event=="Exit":
            window.close()
            sys.exit()
        elif event == "Submit":
            inputs['fc Import location'] = inputs["Import location"]
            inputs['fc Export location'] = inputs["Export location"]
            inputs['fc TTL']             = inputs['TTL']
            inputs['fc Zero time point'] = recognise_ZTM(values["ZTP"])
            inputs['fc Setup']           = inputs['Setup']
            inputs['fc TRANGE']          = [float(values['TRANGE1']), float(values['TRANGE2'])]
            inputs['fc Baseline period'] = [float(values['BASELINE1']), float(values['BASELINE2'])]
            inputs['fc Artifact RL']     = recognise_artifact(values['Artifact'])
            inputs['fc zScore']          = recognise_bool(values['zScore'])
            inputs['fc dFF']             = recognise_bool(values['dFF'])
            inputs['fc Create snippets'] = {'Create?':recognise_bool(values['Video']),
                                            'Camera':'Camera 1'}
            inputs['fc ISOS']            = recognise_bool(values['ISOS'])
            inputs['fc GCaMP']           = recognise_bool(values['GCaMP'])
            window.close()
            break
        
    if inputs['fc Create snippets']['Create?'] == False:
        print('\nStarted analysis.')
        arguments = [inputs[key] for key in inputs.keys() if key[:2]=='fc']
        PeriEventTTL(*arguments) # Run the corresponding code.
        print('Finished analysis')

    #-------------------------------------------------------------------------------#
    # TDT -> peri-event TTL -> ... -> put in options for the video snippets.
    #-------------------------------------------------------------------------------#     
       
    elif inputs['fc Create snippets']['Create?'] == True:

        default["Camera"]  = ('Camera 1' if inputs['fc Setup']=='Setup A' else 'Camera 2')
        sg.theme("DarkTeal2")
        layout = []
        layout += [[sg.T("")], [sg.Text("Choose the camera for the video."), 
                                sg.Combo(['Camera 1','Camera 2'],key="Camera",enable_events=True,default_value=default['Camera'])]]
        layout += [[sg.T("")], [sg.Button("Submit")]]
        window = sg.Window('Photometry Analysis', layout)
        while True:
            event, values = window.read()
            if event == sg.WIN_CLOSED or event=="Exit":
                window.close()
                sys.exit()
            elif event == "Submit":
                inputs['fc Create snippets']['Camera'] = camera(values['Camera'])
                window.close()
                break
        print('\nStarted analysis.')
        arguments = [inputs[key] for key in inputs.keys() if key[:2]=='fc']
        PeriEventTTL(*arguments) # Run the corresponding code.
        print('Finished analysis')
        
#-----------------------------------------------------------------------------#
# TDT -> whole recording -> ... -> select options before running the code
#-----------------------------------------------------------------------------#

if inputs['Analysis'] == 'Whole recording':

    default["TRANGE"]  = [-20,30]
    default["Remove"]  = 4
    default["TTL CSV"] = 'True'
    default["Barcode"] = 'False'
    save_CSV_GUI = []
    i=1
    save_CSV_GUI += [[sg.T("")]]
    save_CSV_GUI += [[sg.Text("Save TTL"+str(i)+" time data to CSV?"), 
                      sg.Combo(['True','False'],key="TTL CSV"+str(i),enable_events=True,default_value=default['TTL CSV'])]]
    sg.theme("DarkTeal2")        
    layout = []
    layout += [[sg.T("")], [sg.Text("Choose the TRANGE window size (start time relative to epoc onset in seconds, window duration)"), 
                            sg.Input(key="TRANGE1",enable_events=True,default_text=default["TRANGE"][0],size=(10,1)), 
                            sg.Input(key="TRANGE2",enable_events=True,default_text=default["TRANGE"][1],size=(10,1))]]
    layout += [[sg.T("")], [sg.Text("Choose how much data from the start should be removed (in secs, to account for the artifact when turning on the LED)"), 
                            sg.Input(key="Remove",enable_events=True,default_text=default["Remove"],size=(20,1))]]
    layout += save_CSV_GUI
    layout += [[sg.T("")], [sg.Text("Create plot with Ethovision barcodes overlayed?"), 
                            sg.Combo(['True','False'],key="Barcode",enable_events=True,default_value=default['Barcode'])]]
    layout += [[sg.T("")], [sg.Button("Submit")]]
    window = sg.Window('Photometry Analysis', layout)
    while True:
        event, values = window.read()
        if event == sg.WIN_CLOSED or event=="Exit":
            window.close()
            sys.exit()
        elif event == "Submit":
            inputs['fc Import location'] = inputs["Import location"]
            inputs['fc Export location'] = inputs["Export location"]
            inputs['fc TTL']             = inputs['TTL']
            # inputs['fc TTL']             = {'TTL'+str(i):inputs['TTL epoch'][i-1]    for i in range(1,inputs['No. TTLs']+1)}
            # extra_entries                = {'Custom'+str(i):inputs['Custom'][i-1]    for i in range(1,inputs['No. TTLs']+1)}
            # inputs['fc TTL'].update(extra_entries)
            inputs['fc Setup']           = inputs['Setup']
            inputs['fc TRANGE']          = [float(values['TRANGE1']), float(values['TRANGE2'])]
            inputs['fc Remove']          = float(values['Remove'])
            # inputs['fc TTL CSV']         = [recognise_bool(values['TTL CSV'+str(i)]) for i in range(1,inputs['No. TTLs']+1)]
            # inputs['fc TTL CSV']         = {'TTL'+str(i):inputs['fc TTL CSV'][i-1]   for i in range(1,inputs['No. TTLs']+1)}
            inputs['fc TTL CSV']         = recognise_bool(values['TTL CSV'+str(i)])
            inputs['fc Create barcode']  = {'Create?':recognise_bool(values['Barcode'])}
            # Run the corresponding codes.
            window.close()
            break
    
    if inputs['fc Create barcode']['Create?'] == False:
        arguments = [inputs[key] for key in inputs.keys() if key[:2]=='fc']
        Whole_recording_1_TTL(*arguments)
    
#-------------------------------------------------------------------------------#
# TDT -> whole recording -> ... -> put in options for the barcodes.
#-------------------------------------------------------------------------------#
    
    elif inputs['fc Create barcode']['Create?'] == True:
        
        #-----------------------------------------------------------------------------

        # Choose the default values for the GUI (dv).

        dv = {} # Ignore this.

        # Choose the import and export locations.
        # Note that these locations should not have a slash at the end.
        # dv['Import location'] = 'C:/Users/hazza/Desktop/Andrews Lab Python Work/Task 27 - Manual scoring for Felicia 24-12-21/Raw data (10 mins, YT8 4 mins)/Grey Test Trial     1.xlsx'

        # Choose the number of behaviours and whether to find overlap with a zone.
        dv['No behaviours'] = 4
        dv['Find overlap'] = 'False' # Note that this should be a string of 'True' or 'False'.

        # Choose the excel names, custom names and barcode colours for each behaviour.
        dv['Excel  name for zone'] = 'In zone(Arena / center-point)'
        dv['Custom name for zone'] = 'Arena'
        dv['Excel  names for behaviours'] = ['Walking(Mutually exclusive)', 'Stationary(Mutually exclusive)', 
                                             'Rearing(Mutually exclusive)', 'Grooming(Mutually exclusive)']
        dv['Custom names for behaviours'] = ['Walking', 'Stationary', 'Rearing', 'Grooming']
        dv['Colours for behaviours']      = ['#aec7e8', '#ffbb78', '#98df8a', '#ff9896']
            
        #-----------------------------------------------------------------------------
    
        # Choose the import and export locations, number of behaviours and whether 
        # to measure overlap with a zone.
        
        # Create the GUI.
        sg.theme("DarkTeal2")
        layout = [
            # [sg.T("")], [sg.Text("Choose an Ethovision file to import"), 
            #              sg.Input(key="Import" ,enable_events=True,
            #                       default_text=dv['Import location']), 
            #              sg.FileBrowse(key="Import2")],
            [sg.T("")], [sg.Text("How many behaviours?"), 
                         sg.Combo(list(range(1,15+1)),key="Num",enable_events=True,
                                  default_value=dv['No behaviours'])],
            [sg.T("")], [sg.Text("Find overlap with a zone?"), 
                         sg.Combo(['True','False'],key="Overlap",enable_events=True,
                                  default_value=dv['Find overlap'])],
            [sg.T("")], [sg.Button("Submit")]
                  ]
        window = sg.Window('Manual scoring GUI', layout)
        
        # Convert 'True' to boolean True and 'False' to boolean False.
        def bool_check(value):
            if value == 'True':
                return(True)
            elif value == 'False':
                return(False)
            else:
                print('Make sure True or False is used.\n')
                exit()
        
        # Assign the values in the GUI to variables.
        val = {} # The input values, as opposed to the default values, are stored here.
        while True:
            event, values = window.read()
            if event == sg.WIN_CLOSED or event=="Exit":
                window.close()
                exit()
            elif event == "Submit":
                val['No behaviours']  = values["Num"]
                val['Find overlap'] = bool_check(values["Overlap"])
                val['Excel  name for zone'] = ''
                # Make the import location for the barcode the same Ethovision 
                # file imported at the step for choosing the TTL pulse.
                val['Import location'] = inputs['fc Import location']
                window.close()
                break
        
        print('Import location is ' + val['Import location'])
        print('The number of behaviours is ' + str(val['No behaviours']))
        if val['Find overlap'] == True:
            print('Find overlap with a zone')
        elif val['Find overlap'] == False:
            print('Do not find overlap with a zone')
        
        # Choose the names of the behaviours in excel and the names and colors for the output.
        
        # Create the GUI.
        sg.theme("DarkTeal2")
        message =  "Put the behaviours in the order of importance. "
        message += "If 2 happen at the same time, only the first one will be counted. "
        message += "If the behaviours are mutually exclusive, the order does not matter."
        layout = [[sg.Text(message)],
                  [sg.Text("Use the default colors or click this link to find HEX values. "
                           "Make sure you include the # at the start:"),
                   sg.Text("https://coolors.co/", key="Link", enable_events=True)]]
        
        for i in range(1,len(dv['Excel  names for behaviours'])+1):
            dv['Excel  name for behaviour'+str(i)] = dv['Excel  names for behaviours'][i-1]
            dv['Custom name for behaviour'+str(i)] = dv['Custom names for behaviours'][i-1]
            dv['Colour for behaviour'+str(i)]      = dv['Colours for behaviours'][i-1]
        
        if val['Find overlap'] == True:
            layout += [[sg.T("")], [sg.Text("Excel name for zone         "), 
                sg.Input(key="Zone",enable_events=True,default_text=dv['Excel  name for zone']),
                sg.Text("Custom name for zone         "), 
                sg.Input(key="cZone",enable_events=True,default_text=dv['Custom name for zone'])]]
        for i in range(1,val['No behaviours']+1):
            if i <= dv['No behaviours']:
                layout += [[sg.T("")], [sg.Text("Excel name for behaviour "+str(i)), 
                    sg.Input(key="Behaviour"+str(i),enable_events=True,
                             default_text=dv['Excel  name for behaviour'+str(i)]),
                    sg.Text("Custom name for behaviour "+str(i)), 
                    sg.Input(key="cBehaviour"+str(i),enable_events=True,
                             default_text=dv['Custom name for behaviour'+str(i)]),
                    sg.Text("Choose color"),
                    sg.Input(key="Color"+str(i),enable_events=True,
                             default_text=dv['Colour for behaviour'+str(i)])]]
            else:
                layout += [[sg.T("")], [sg.Text("Excel name for behaviour "+str(i)), 
                    sg.Input(key="Behaviour"+str(i),enable_events=True,default_text=''),
                    sg.Text("Custom name for behaviour "+str(i)), 
                    sg.Input(key="cBehaviour"+str(i),enable_events=True,default_text=''),
                    sg.Text("Choose color"),
                    sg.Input(key="Color"+str(i),enable_events=True,default_text='')]]        
                               
        layout += [[sg.T("")],[sg.Button("Submit")]]
        window_height = 100 + val['No behaviours']*60
        if val['Find overlap'] == True:
            window_height += 60
        window = sg.Window('Manual scoring GUI', layout)
        
        # Assign the values in the GUI to variables.
        while True:
            event, values = window.read()
            if event == sg.WIN_CLOSED or event=="Exit":
                window.close()
                exit()
            elif event == "Link":
                webbrowser.open(r'https://coolors.co/')
            elif event == "Submit":
                if val['Find overlap'] == True:
                    val['Excel  name for zone'] = values["Zone"]
                    val['Custom name for zone'] = values["cZone"]
                val['Excel  names for behaviours'] = []
                val['Custom names for behaviours'] = []
                val['Colours for behaviours']  = []
                for i in range(1,val['No behaviours']+1):
                    val['Excel  names for behaviours'].append(values["Behaviour"+str(i)])
                    val['Custom names for behaviours'].append(values["cBehaviour"+str(i)])
                    val['Colours for behaviours'].append(values["Color"+str(i)])
                val['Excel  names for behaviours'] = tuple(val['Excel  names for behaviours'])
                val['Custom names for behaviours'] = tuple(val['Custom names for behaviours'])
                val['Colours for behaviours']      = tuple(val['Colours for behaviours'])
                window.close()
                break
        
        print('Analyse the ' + str(val['No behaviours']) + ' behaviours in order of importance: ')
        for i in range(val['No behaviours']):
            print(str(i+1) + ') ' + val['Custom names for behaviours'][i] + 
                  ' with HEX color ' + val['Colours for behaviours'][i])
        if val['Find overlap'] == True:
            print('Find the overlap of these behaviours with ' + val['Custom name for zone'])
        
        # Choose the codes to run.
        args_list  = ['Import location', 'Excel  names for behaviours','Custom names for behaviours',
                      'Colours for behaviours','Find overlap','Excel  name for zone']
        
        for arg in args_list:
            inputs['fc Create barcode'][arg] = val[arg]
        
        arguments = [inputs[key] for key in inputs.keys() if key[:2]=='fc']
        Whole_recording_1_TTL(*arguments)
        
        window.close()

#-----------------------------------------------------------------------------#
# TDT -> FED3 -> choose the poke to analyse     
#-----------------------------------------------------------------------------#  
        
if inputs['Analysis'] == 'FED3':
    default["Poke to analyse"] = 'Left'
    sg.theme("DarkTeal2")
    layout  = []
    layout += [[sg.T("")],[sg.Text("Choose the poke to analyse."),
                           sg.Combo(['Left', 'Right', 'Both'],
                           key="Poke to analyse",enable_events=True,
                           default_value=default["Poke to analyse"])]]
    layout += [[sg.T("")], [sg.Button("Submit")]]
    window  = sg.Window('Photometry Analysis', layout)
    while True:
        event, values = window.read()
        if event == sg.WIN_CLOSED or event=="Exit":
            window.close()
            sys.exit()
        elif event == "Submit":
            inputs['Poke to analyse'] = values["Poke to analyse"]
            window.close()
            break
    print('Analyse this poke: '+inputs['Poke to analyse'])            
    
    #-----------------------------------------------------------------------------#
    # TDT -> FED3 -> choose the poke to analyse     
    #-----------------------------------------------------------------------------#  
            
    default["Active poke"] = 'Left'
    sg.theme("DarkTeal2")
    layout  = []
    layout += [[sg.T("")],[sg.Text('At the moment, "changing" just makes the '+
                                   'active pokes the ones that preceded\n'+
                                   'a pellet drop, even if this was not the poke '+
                                   'that caused the pellet drop.')], [sg.T("")], 
                          [sg.Text("Choose the active poke."),
                           sg.Combo(['Left', 'Right', 'Changing'],
                           key="Active poke",enable_events=True,
                           default_value=default["Active poke"])]]
    layout += [[sg.T("")], [sg.Button("Submit")]]
    window  = sg.Window('Photometry Analysis', layout)
    while True:
        event, values = window.read()
        if event == sg.WIN_CLOSED or event=="Exit":
            window.close()
            sys.exit()
        elif event == "Submit":
            inputs['Active poke'] = values["Active poke"]
            window.close()
            break
    print('The active poke is: '+inputs['Active poke'])         

#-----------------------------------------------------------------------------#
# TDT -> FED3 -> ... -> select options before running the code
#-----------------------------------------------------------------------------#

    default["Zero time point"]  = ['']
    default["TRANGE"]           = [-20,80]
    default["Baseline period"]  = [-20,-5]
    default["Artifact RL"]      = ''     
    default["zScore"]           = 'True'
    default["dFF"]              = 'False'
    default["AUC"]              = 'False'
    sg.theme("DarkTeal2")
    layout = []
    layout += [[sg.T("")], [sg.Text("Choose the TRANGE window size (start time relative to epoc onset in seconds, window duration)"), 
                            sg.Input(key="TRANGE1",enable_events=True,default_text=default["TRANGE"][0],size=(10,1)), 
                            sg.Input(key="TRANGE2",enable_events=True,default_text=default["TRANGE"][1],size=(10,1))]]
    layout += [[sg.T("")], [sg.Text("Choose the baseline period within the window"), 
                            sg.Input(key="BASELINE1",enable_events=True,default_text=default["Baseline period"][0],size=(10,1)),
                            sg.Input(key="BASELINE2",enable_events=True,default_text=default["Baseline period"][1],size=(10,1))]]
    layout += [[sg.T("")], [sg.Text("Choose the artifact rejection level (optional)"), 
                            sg.Input(key="Artifact",enable_events=True,default_text=default["Artifact RL"],size=(20,1))]]
    layout += [[sg.T("")], [sg.Text("Choose the TTL number you want as the zero time point (optional)."), 
                            sg.Input(key="ZTP",enable_events=True,default_text=default["Zero time point"][0],size=(20,1))]]
    layout += [[sg.T("")], [sg.Text("Save Z-Score data to CSV?"), 
                            sg.Combo(['True','False'],key="zScore",enable_events=True,default_value=default['zScore'])]]
    layout += [[sg.T("")], [sg.Text("Save dFF data to CSV?"), 
                            sg.Combo(['True','False'],key="dFF",enable_events=True,default_value=default['dFF'])]]
    # layout += [[sg.T("")], [sg.Text("Save AUC data to CSV?"), 
    #                         sg.Combo(['True','False'],key="AUC",enable_events=True,default_value=default['AUC'])]]
    layout += [[sg.T("")], [sg.Text("Save ISOS data to CSV?"), 
                            sg.Combo(['True','False'],key="ISOS",enable_events=True,default_value=default['dFF'])]]
    layout += [[sg.T("")], [sg.Text("Save GCaMP data to CSV?"), 
                            sg.Combo(['True','False'],key="GCaMP",enable_events=True,default_value=default['dFF'])]]
    layout += [[sg.T("")], [sg.Button("Submit")]]
    window = sg.Window('Photometry Analysis', layout)
    while True:
        event, values = window.read()
        if event == sg.WIN_CLOSED or event=="Exit":
            window.close()
            sys.exit()
        elif event == "Submit":
            inputs['fc Import location'] = inputs["Import location"]
            inputs['fc Export location'] = inputs["Export location"]
            inputs['fc Poke to analyse'] = inputs['Poke to analyse']
            inputs['fc Active poke']     = inputs['Active poke']
            inputs['fc Zero time point'] = recognise_ZTM(values["ZTP"])
            inputs['fc Setup']           = inputs['Setup']
            inputs['fc TRANGE']          = [float(values['TRANGE1']), float(values['TRANGE2'])]
            inputs['fc Baseline period'] = [float(values['BASELINE1']), float(values['BASELINE2'])]
            inputs['fc Artifact RL']     = recognise_artifact(values['Artifact'])
            inputs['fc zScore']          = recognise_bool(values['zScore'])
            inputs['fc dFF']             = recognise_bool(values['dFF'])
            # inputs['fc AUC']             = recognise_bool(values['AUC'])
            inputs['fc ISOS']            = recognise_bool(values['ISOS'])
            inputs['fc GCaMP']           = recognise_bool(values['GCaMP'])
            # Run the corresponding codes.
            arguments = [inputs[key] for key in inputs.keys() if key[:2]=='fc']
            if inputs['fc Poke to analyse'] == 'Left' or inputs['fc Poke to analyse'] == 'Right':
                print('\nStarted analysis.')
                FED3_one_active_poke(*arguments)
                FED3_pellet_retrievals(*arguments)
                print('Finished analysis')
            elif inputs['fc Poke to analyse'] == 'Both':
                print('\nStarted analysis.')
                FED3_both_active_pokes(*arguments)
                FED3_pellet_retrievals(*arguments)
                print('Finished analysis')
            window.close()
            break
        
#-----------------------------------------------------------------------------#
# TDT -> between TTLs -> choose the test
#-----------------------------------------------------------------------------#         
        
if inputs['Analysis'] == 'Between TTLs':         
    default["Test"] = '2 bottle choice'
    sg.theme("DarkTeal2")
    layout  = []
    layout += [[sg.T("")],[sg.Text("Choose the test."),
                           sg.Combo(['2 bottle choice','Open field','Elevated plus maze'],
                                    key="Test",enable_events=True,default_value=default["Test"])]]
    layout += [[sg.T("")], [sg.Button("Submit")]]
    window  = sg.Window('Photometry Analysis', layout)
    while True:
        event, values = window.read()
        if event == sg.WIN_CLOSED or event=="Exit":
            window.close()
            sys.exit()
        elif event == "Submit":
            inputs['Test'] = values["Test"]
            window.close()
            break
    print('Test is '+inputs['Test'])
    
#-----------------------------------------------------------------------------#
# TDT -> between TTLs -> ... -> select options before running the code.
#-----------------------------------------------------------------------------#
    
    if inputs['System'] == 'TDT (Tucker-Davis Technologies)':
        df_TTL = pd.DataFrame({'2 bottle choice':    ['Left lick',  'Left','Right lick','Rght'],
                                'Open field':        ['Centre zone','Cntr','Outer zone','Outr'],
                                'Elevated plus maze':['Open arm',   'Open','Closed arm','Clsd']},
                                index=['Type TTL1','Name TTL1','Type TTL2','Name TTL2'])
    elif inputs['System'] == 'NPM (Neurophotometrics)':
        events     = list(data.epocs.keys())
        left       = [event for event in events if ('left'  in event or 'Left'  in event)][0]
        right      = [event for event in events if ('right' in event or 'Right' in event)][0]
        centre     = [event for event in events if ('centre' in event or 'Centre' in event or 'center' in event or 'Center' in event)][0]
        outer      = [event for event in events if ('outer' in event or 'Outer' in event)][0]
        open_arm   = [event for event in events if ('open' in event or 'Open' in event)][0]
        closed_arm = [event for event in events if ('closed' in event or 'Closed' in event)][0]
        df_TTL = pd.DataFrame({'2 bottle choice':    ['Left lick',  left,    'Right lick',right],
                                'Open field':        ['Centre zone',centre,  'Outer zone',outer],
                                'Elevated plus maze':['Open arm',   open_arm,'Closed arm',closed_arm]},
                                index=['Type TTL1','Name TTL1','Type TTL2','Name TTL2'])        
        
    default["Zero time point"]  = ['']
    default["Artifact RL"]      = ''     
    default["zScore"]           = 'True'
    default["dFF"]              = 'False'
    default["AUC"]              = 'False'
    sg.theme("DarkTeal2")
    layout = []
    layout += [[sg.T("")], [sg.Text("Choose the artifact rejection level (optional)"), 
                            sg.Input(key="Artifact",enable_events=True,default_text=default["Artifact RL"],size=(20,1))]]
    layout += [[sg.T("")], [sg.Text("Choose the TTL number you want as the zero time point (optional)."), 
                            sg.Input(key="ZTP",enable_events=True,default_text=default["Zero time point"][0],size=(20,1))]]
    layout += [[sg.T("")], [sg.Text("Save Z-Score data to CSV?"), 
                            sg.Combo(['True','False'],key="zScore",enable_events=True,default_value=default['zScore'])]]
    layout += [[sg.T("")], [sg.Text("Save dFF data to CSV?"), 
                            sg.Combo(['True','False'],key="dFF",enable_events=True,default_value=default['dFF'])]]
    # layout += [[sg.T("")], [sg.Text("Save AUC data to CSV?"), 
    #                         sg.Combo(['True','False'],key="AUC",enable_events=True,default_value=default['AUC'])]]
    layout += [[sg.T("")], [sg.Button("Submit")]]
    window = sg.Window('Photometry Analysis', layout)
    while True:
        event, values = window.read()
        if event == sg.WIN_CLOSED or event=="Exit":
            window.close()
            sys.exit()
        elif event == "Submit":
            inputs['fc Import location'] = inputs["Import location"]
            inputs['fc Export location'] = inputs["Export location"]
            inputs['fc TTL'] = ''
            inputs['fc Zero time point'] = recognise_ZTM(values["ZTP"])
            inputs['fc Setup']           = inputs['Setup']
            inputs['fc Artifact RL']     = recognise_artifact(values['Artifact'])
            inputs['fc zScore']          = recognise_bool(values['zScore'])
            inputs['fc dFF']             = recognise_bool(values['dFF'])
            # inputs['fc AUC']             = recognise_bool(values['AUC'])
            print('\nStarted analysis.')
            inputs['fc TTL'] = {'Test':inputs['Test'],
                                'Type':df_TTL.at['Type TTL1',inputs['Test']],
                                'Name':df_TTL.at['Name TTL1',inputs['Test']]}
            inputs1 = deepcopy(inputs)
            arguments = [inputs[key] for key in inputs.keys() if key[:2]=='fc']
            Between_TTLs(*arguments)
            inputs['fc TTL'] = {'Test':inputs['Test'],
                                'Type':df_TTL.at['Type TTL2',inputs['Test']],
                                'Name':df_TTL.at['Name TTL2',inputs['Test']]}
            inputs2 = deepcopy(inputs)
            arguments = [inputs[key] for key in inputs.keys() if key[:2]=='fc']
            Between_TTLs(*arguments)
            print('Finished analysis')
            window.close()
            break

# Export the settings.
inputs_exclude = ['Import location', 'Export location', 'Setup', 'TTLM list', 'Notes list',
                  'Video timestamp list', 'Poke to analyse', 'TTL list', 'No. TTLs', 
                  'TTL epoch', 'TTL', 'Ethovision event list', 'Active poke']
inputs_dicts = ['fc TTL', 'fc Create barcode', 'fc Create snippets']

# For between TTLs, the analysis is run twice with different event names.
# Thus, two different sets of settings need to be added to the settings excel file.
list_inputs = []
if inputs['Analysis'] == 'Between TTLs':
    list_inputs += [inputs1, inputs2]
else:
    list_inputs += [inputs]

# Create the settings excel file.
first_loop = True

for inputs in list_inputs:
    df_ind    = {}
    
    for key in inputs.keys():
        if key in inputs_exclude:
            continue
        if key in inputs_dicts:
            for key2 in inputs[key]:
                df_ind[key+' '+key2] = inputs[key][key2]
            continue
        if key == 'fc Zero time point':
            if inputs[key] == []:
                df_ind[key] = ''
            else:
                df_ind[key] = inputs[key][0]
            continue
        df_ind[key] = inputs[key]
    
    df_ind = pd.DataFrame(df_ind.values(), index=df_ind.keys())
    extra_line = pd.DataFrame([False], index=['Import subfolders'])
    df_ind = pd.concat([extra_line, df_ind])
    
    blank_line = pd.DataFrame([''], index=[''])
    # Add a blank line between each set of analysis.
    # But skip adding a blank line to the top of the settings excel file.
    if first_loop == True:
        df_master  = df_ind
    else:
        df_master  = pd.concat([df_master, blank_line, df_ind])
    first_loop = False
    # If import subfolders was used, change the import/export locations.
    # if inputs['Subfolders'] == True:
    #     df.at['fc Import location',0] = import_loc
    #     df.at['fc Export location',0] = export_loc

# If a settings file already exists, create a new one with a number at the end.
export_name = 'Settings0.xlsx'
i = 1
while export_name in os.listdir(inputs['fc Export location']):
    export_name = export_name[:8] + str(i) + '.xlsx'
    i += 1
    
df_master.to_excel(inputs['fc Export location']+'/'+export_name,header=False)
print('Saved ' + export_name)
