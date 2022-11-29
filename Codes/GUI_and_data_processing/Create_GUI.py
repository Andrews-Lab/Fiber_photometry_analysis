import PySimpleGUI as sg
import numpy as np
import sys
from Analysis_types.Whole_recording import extract_event_data_from_cols

def camera(value):
    dict1 = {'Camera 1':'Cam1', 'Camera 2':'Cam2'}
    return(dict1[value])
def recognise_artifact(value):
    return(np.inf if value=='' else float(value))
def recognise_bool(value):
    dict1 = {'True':True, 'False':False}
    return(dict1[value])
def setups(name, letter):
    dict1 = {'ISOS':  {'Setup A':'_405A', 'Setup B':'_415A'},
             'GCaMP': {'Setup A':'_465A', 'Setup B':'_475A'}}
    return(dict1[name][letter])

def choose_settings_file_or_not(inputs):

    # Check whether there is a settings excel file.
    default = {}
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
            inputs['Import subfolders'] = False # This is just needed for the settings export file.
            window.close()
            break
    print('Do not use settings excel file' if inputs['Settings']=='False' else 'Use settings excel file')
    
    return(inputs)
    
# Put in the options from an excel file.

def choose_location_settings_file(inputs):
    
    default = {}
    default['Import settings'] = r"C:\Users\hazza\Desktop\Fibre photometry GUI\Photometry tanks\NAc_3147K_2614K-211118-105517\Settings_all.xlsx"
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
        
    return(inputs)
    
def choose_basic_TDT_options(inputs):

    # Choose the import location, export location and setup.
    default = {}
    default["Import location"] = r'C:\Users\hazza\Desktop\Fibre photometry GUI\Photometry tanks\NAc_3147K_2614K-211118-105517'
    default["Export location"] = r'C:\Users\hazza\Desktop\Fibre photometry GUI\Photometry tanks\NAc_3147K_2614K-211118-105517'
    default["Setup"]           = 'Setup A'
    default["Camera"]          = 'Camera 1'
    default["Analysis"]        = 'Peri-events'
    sg.theme("DarkTeal2")
    layout  = []
    layout += [[sg.T("")], [sg.Text("Choose a folder for the import location"), 
                sg.Input(key="Import" ,enable_events=True,default_text=default["Import location"]),
                sg.FolderBrowse(key="Import2")]]
    layout += [[sg.T("")], [sg.Text("Choose a folder for the export location"),
                sg.Input(key="Export" ,enable_events=True,default_text=default["Import location"]),
                sg.FolderBrowse(key="Export2")]]
    layout += [[sg.T("")],[sg.Text("Choose setup A or setup B", size=(22,1)), 
                sg.Combo(['Setup A','Setup B','Custom'],key="Setup",enable_events=True,default_value=default["Setup"])]]
    layout += [[sg.T("")],[sg.Text("Choose camera 1 or camera 2", size=(22,1)), 
                sg.Combo(['Camera 1','Camera 2'],key="Camera",enable_events=True,default_value=default["Camera"])]]
    layout += [[sg.T("")],[sg.Text("Choose the type of analysis.", size=(22,1)), 
                sg.Combo(['Peri-events', 'FED3', 'Between events', 'Whole recording'],
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
            inputs['Setup']           = values["Setup"]
            if values['Setup'] != 'Custom':
                inputs['ISOS']            = setups('ISOS',  inputs['Setup'])
                inputs['GCaMP']           = setups('GCaMP', inputs['Setup'])
            inputs['Camera']          = camera(values['Camera'])
            inputs['Analysis']        = values["Analysis"]
            inputs["N"]               = 100
            window.close()
            break
    print('Import location is '+inputs['Import location'])
    print('Export location is '+inputs['Export location'])
    print('Setup is '+inputs['Setup'])
    print('Camera is '+inputs['Camera'])
    print('Type of analysis is '+inputs['Analysis'])
    
    return(inputs)

def choose_ISOS_and_GCaMP_signals(inputs):

    # Choose the type of TTL.
    default = {}
    default["ISOS"]  = '_405A'
    default["GCaMP"] = '_465A'
    sg.theme("DarkTeal2")
    layout  = []
    layout += [[sg.T("")],[sg.Text("Choose the stream names in the TDT tanks.")], 
               [sg.T("")],[sg.Text("ISOS signal", size=(11,1)),
                sg.Input(key="ISOS",enable_events=True,default_text=default["ISOS"],
                         size=(20,1))],
               [sg.T("")],[sg.Text("GCaMP signal", size=(11,1)),
                sg.Input(key="GCaMP",enable_events=True,default_text=default["GCaMP"],
                         size=(20,1))]]
    layout += [[sg.T("")], [sg.Button("Submit")]]
    window  = sg.Window('Photometry Analysis', layout)
    while True:
        event, values = window.read()
        if event == sg.WIN_CLOSED or event=="Exit":
            window.close()
            sys.exit()
        elif event == "Submit":
            inputs['ISOS']  = values['ISOS']
            inputs['GCaMP'] = values['GCaMP']
            window.close()
            break
            
    return(inputs)

# TDT -> peri-event TTL -> choose the type of TTL pulse

def choose_type_TDT_event(inputs):

    # Choose the type of TTL.
    default = {}
    default["Event"] = 'Other'
    sg.theme("DarkTeal2")
    layout  = []
    layout += [[sg.T("")],[sg.Text("Choose the type of event."), 
                sg.Combo(['Other','Note','Video timestamp','Ethovision'],
                key="Event",enable_events=True,default_value=default["Event"])]]
    layout += [[sg.T("")], [sg.Button("Submit")]]
    window  = sg.Window('Photometry Analysis', layout)
    while True:
        event, values = window.read()
        if event == sg.WIN_CLOSED or event=="Exit":
            window.close()
            sys.exit()
        elif event == "Submit":
            inputs['Type'] = values['Event']
            window.close()
            break
    print('Type of event pulse is '+inputs['Type'])
            
    return(inputs)
        
# TDT -> peri-event TTL -> TTLM -> choose the name of the TTL event
    
def choose_name_TDT_TTLM_event(inputs):
    
    # Choose the type of TTL.
    default = {}
    default["TTLM name"] = inputs['Other list'][0]
    default["Custom"] = "Peri-event_other"
    sg.theme("DarkTeal2")
    layout  = []
    layout += [[sg.T("")],[sg.Text("Choose the name of the event."), 
                sg.Combo(inputs["Other list"],key="TTLM name",enable_events=True,default_value=default["TTLM name"])]]
    layout += [[sg.T("")], [sg.Text("Choose a custom name for this event."), 
                sg.Input(key="Custom",enable_events=True,default_text=default["Custom"],
                         size=(20,1))]]
    layout += [[sg.T("")], [sg.Button("Submit")]]
    window  = sg.Window('Photometry Analysis', layout)
    while True:
        event, values = window.read()
        if event == sg.WIN_CLOSED or event=="Exit":
            window.close()
            sys.exit()
        elif event == "Submit":
            inputs['Name']   = values["TTLM name"]
            inputs['Analysis name'] = values['Custom']
            window.close()
            break
    print('The name of the TTL pulse is '+inputs['Name'])
    
    return(inputs)
                
# TDT -> peri-event TTL -> note -> 1,2,... notes -> choose which notes    
          
def choose_name_TDT_note_event(inputs):

    default = {}
    default["Custom"] = 'Peri-event_note'
    default["All"] = 'Specific'
    checkbox_cols = []
    for i in range(len(inputs['Notes list'])):
        checkbox_cols += [[sg.Checkbox(inputs['Notes list'][i], default=False, 
                                        key=inputs['Notes list'][i])]]
    sg.theme("DarkTeal2")
    layout  = []
    layout += [[sg.T("")], [sg.Text("Choose a custom name for this event."), 
                sg.Input(key="Custom",enable_events=True,default_text=default["Custom"],
                         size=(20,1))]]
    layout += [[sg.T("")], [sg.Text("Use all or specific notes."),
                            sg.Combo(['All', 'Specific'], key="All",enable_events=True,
                                      default_value=default["All"])]]
    layout += [[sg.T("")], [sg.Text("Choose which notes should count as the same event.",
                                    key='Specific')]]
    layout += [*checkbox_cols]
    layout += [[sg.T("")], [sg.Button("Submit")]]                    
    window  = sg.Window('Photometry Analysis', layout)
    while True:
        event, values = window.read()
        if event == sg.WIN_CLOSED or event=="Exit":
            window.close()
            sys.exit()
        if values["All"] == "All":
            window.Element("Specific").Update(visible=False)
            for key in inputs['Notes list']:
                window.Element(key).Update(visible=False)
        if values["All"] == "Specific":
            window.Element("Specific").Update(visible=True)
            for key in inputs['Notes list']:
                window.Element(key).Update(visible=True)
        if event == "Submit":
            if values["All"] == 'All':
                inputs['Name'] = 'All'
            elif values["All"] == 'Specific':
                inputs['Name'] = [inputs['Notes list'][i]
                                  for i in range(len(inputs['Notes list']))
                                  if values[inputs['Notes list'][i]] == True]
            inputs['Analysis name'] = values['Custom']
            window.close()
            break
    print('Use these notes as the same event: '+str(inputs['Name']))

    return(inputs)       
    
# TDT -> peri-event TTL -> video timestamp -> choose the camera.

def choose_name_TDT_video_event(inputs):

    default = {}
    default["Custom"] = 'Peri-event_video_timestamp'
    default["All"]    = 'Specific'
    default["Timestamp"] = inputs['Video timestamp list'][0]
    checkbox_cols = []
    for i in range(len(inputs['Video timestamp list'])):
        checkbox_cols += [[sg.Checkbox(inputs['Video timestamp list'][i], default=False, 
                                        key=inputs['Video timestamp list'][i])]]
    sg.theme("DarkTeal2")
    layout  = []
    layout += [[sg.T("")], [sg.Text("Choose a custom name for this event."), 
                            sg.Input(key="Custom",enable_events=True,default_text=default["Custom"],
                                     size=(20,1))]]
    layout += [[sg.T("")], [sg.Text("Use all or specific video timestamps."),
                            sg.Combo(['All', 'Specific'], key="All",enable_events=True,
                                      default_value=default["All"])]]
    layout += [[sg.T("")], [sg.Text("Choose which video timestamps should count as the same event.", 
                                    key='Specific')]]
    layout += [*checkbox_cols]
    layout += [[sg.T("")], [sg.Button("Submit")]]
    window  = sg.Window('Photometry Analysis', layout)
    while True:
        event, values = window.read()
        if event == sg.WIN_CLOSED or event=="Exit":
            window.close()
            sys.exit()
        if values["All"] == "All":
            window.Element("Specific").Update(visible=False)
            for key in inputs['Video timestamp list']:
                window.Element(key).Update(visible=False)
        if values["All"] == "Specific":
            window.Element("Specific").Update(visible=True)
            for key in inputs['Video timestamp list']:
                window.Element(key).Update(visible=True)
        if event == "Submit":
            if values["All"] == 'All':
                inputs['Name'] = 'All'
            elif values["All"] == 'Specific':
                inputs['Name'] = [inputs['Video timestamp list'][i]
                                  for i in range(len(inputs['Video timestamp list']))
                                  if values[inputs['Video timestamp list'][i]] == True]
            inputs['Analysis name'] = values['Custom']
            window.close()
            break
    print('Use the video timestamps '+', '.join(inputs['Name']))
    
    return(inputs)
            
# TDT -> peri-event TTL -> Ethovision -> choose the name of the Ethovision event
    
def choose_name_TDT_Ethovision_event(inputs):
    
    # Choose the type of TTL.
    default = {}
    default["Ethovision name"] = inputs['Ethovision event list'][0]
    default["Custom"] = "Peri-event_Ethovision"
    sg.theme("DarkTeal2")
    layout  = []
    layout += [[sg.T("")], [sg.Text("Ensure the raw data Ethovision files are in the import tank.\n"+
                                    "Edit the filenames, so that the excel files start with 'Setup A' or 'Setup B'.\n"+
                                    "Choose the column heading, which corresponds to your event.\n")]]
    layout += [[sg.Combo(inputs['Ethovision event list'],key='Ethovision name',enable_events=True,default_value=default["Ethovision name"])]]
    layout += [[sg.T("")], [sg.Text("Choose a custom name for this event."), 
                            sg.Input(key="Custom",enable_events=True,default_text=default["Custom"],
                                     size=(20,1))]]
    layout += [[sg.T("")], [sg.Button("Submit")]]
    window  = sg.Window('Photometry Analysis', layout)
    while True:
        event, values = window.read()
        if event == sg.WIN_CLOSED or event=="Exit":
            window.close()
            sys.exit()
        elif event == "Submit":
            inputs['Name'] = values["Ethovision name"]
            inputs['Analysis name'] = values['Custom']
            window.close()
            break
    print('The name of the TTL pulse is '+inputs['Name'])
    
    return(inputs)

def choose_name_TDT_event(inputs):

    if inputs['Type'] == 'Other':
        inputs = choose_name_TDT_TTLM_event(inputs)
    elif inputs['Type'] == 'Note':
        inputs = choose_name_TDT_note_event(inputs)
    elif inputs['Type'] == 'Video timestamp':
        inputs = choose_name_TDT_video_event(inputs)
    elif inputs['Type'] == 'Ethovision':
        inputs = choose_name_TDT_Ethovision_event(inputs)
        
    return(inputs)
    
# TDT -> peri-event TTL -> ... -> select options before running the code     

def choose_peri_event_options(inputs):
    
    default = {}
    default["t-range"]           = [-20,80]
    default["Baseline period"]  = [-20,-5]
    default["Artifact RL"]      = ''
    default["Image"]            = 'True'
    default["Video"]            = 'False'
    default["zScore"]           = 'True'
    default["dFF"]              = 'False'
    default["ISOS"]             = 'False'
    default["GCaMP"]            = 'False'
    sg.theme("DarkTeal2")
    layout = [[sg.T("")],
        [sg.Text("Choose the t-range (time before event, duration of window) (secs)"), 
         sg.Input(key="TRANGE1",enable_events=True,
                  default_text=default["t-range"][0],size=(10,1)), 
         sg.Input(key="TRANGE2",enable_events=True,
                  default_text=default["t-range"][1],size=(10,1))],[sg.T("")],
        [sg.Text("Choose the baseline period within the window"), 
         sg.Input(key="BASELINE1",enable_events=True,
                  default_text=default["Baseline period"][0],size=(10,1)),
         sg.Input(key="BASELINE2",enable_events=True,
                  default_text=default["Baseline period"][1],size=(10,1))],[sg.T("")],
        [sg.Text("Choose the artifact rejection level (optional)"), 
         sg.Input(key="Artifact",enable_events=True,
                  default_text=default["Artifact RL"],size=(20,1))],[sg.T("")],
        [sg.Text("Create video snippets of epochs?"), 
         sg.Combo(['True','False'],key="Video",enable_events=True,
                  default_value=default['Video'])],[sg.T("")],
        [sg.Text("Save preview image of data?"), 
         sg.Combo(['True','False'],key="Image",enable_events=True,
                  default_value=default['Image'])],[sg.T("")],
        [sg.Text("Save Z-Score data to CSV?"), 
         sg.Combo(['True','False'],key="zScore",enable_events=True,
                  default_value=default['zScore'])],[sg.T("")],
        [sg.Text("Save dFF data to CSV?"), 
         sg.Combo(['True','False'],key="dFF",enable_events=True,
                  default_value=default['dFF'])],[sg.T("")],
        [sg.Text("Save ISOS data to CSV?"), 
         sg.Combo(['True','False'],key="ISOS",enable_events=True,
                  default_value=default['ISOS'])],[sg.T("")],
        [sg.Text("Save GCaMP data to CSV?"), 
         sg.Combo(['True','False'],key="GCaMP",enable_events=True,
                  default_value=default['GCaMP'])],[sg.T("")],
        [sg.Button("Submit")]]
    window = sg.Window('Photometry Analysis', layout)
    while True:
        event, values = window.read()
        if event == sg.WIN_CLOSED or event=="Exit":
            window.close()
            sys.exit()
        elif event == "Submit":
            inputs['t-range']         = [float(values['TRANGE1']),   
                                         float(values['TRANGE2'])]
            inputs['Baseline period'] = [float(values['BASELINE1']), 
                                         float(values['BASELINE2'])]
            inputs['Artifact RL']     = recognise_artifact(values['Artifact'])
            inputs['Image']           = recognise_bool(values["Image"])
            inputs['Create snippets'] = recognise_bool(values['Video'])
            inputs['Export ISOS']   = recognise_bool(values['ISOS'])
            inputs['Export GCaMP']  = recognise_bool(values['GCaMP'])
            inputs['Export dFF']    = recognise_bool(values['dFF'])
            inputs['Export zScore'] = recognise_bool(values['zScore'])
            window.close()
            break
        
    return(inputs)
    
def choose_video_snippet_options(inputs):
    
    # Choose the type of TTL.
    default = {}
    default["Signal"] = 'zScore'
    default["Window"] = 'Same as t-range'
    default["Window size"] = ([0,0] if 't-range' not in inputs.keys() else inputs['t-range'])
    sg.theme("DarkTeal2")
    layout  = [[sg.T("")],
        [sg.Text("Choose the signal to plot."), 
         sg.Combo(['ISOS', 'GCaMP', 'dFF', 'zScore'], key='Signal', 
                  enable_events=True, default_value=default["Signal"])], [sg.T("")],
        [sg.Text("Choose the time window to plot."), 
         sg.Combo(['Same as t-range', 'Custom'],key='Window',
                  enable_events=True, default_value=default["Window"])], [sg.T("")],
        [sg.Text("Choose the time before event and duration of window (secs). "+
                 "\nThis must be smaller than or equal to the t-range.", 
                 key="Window size", visible=False), 
         sg.Input(key="Window size 1",enable_events=True, 
                  default_text=default["Window size"][0], size=(10,1),visible=False),
         sg.Input(key="Window size 2",enable_events=True, 
                  default_text=default["Window size"][1], size=(10,1),visible=False)]]
    layout += [[sg.T("")], [sg.Button("Submit")]]
    window  = sg.Window('Photometry Analysis', layout)
    while True:
        event, values = window.read()
        if event == sg.WIN_CLOSED or event=="Exit":
            window.close()
            sys.exit()
        if values["Window"] == "Custom":
            window.Element("Window size").Update(visible=True)
            window.Element("Window size 1").Update(visible=True)
            window.Element("Window size 2").Update(visible=True)
        if values["Window"] == "Same as t-range":
            window.Element("Window size").Update(visible=False)
            window.Element("Window size 1").Update(visible=False)
            window.Element("Window size 2").Update(visible=False)
        if event == "Submit":
            inputs['Snippets signal'] = values['Signal']
            inputs['Snippets window'] = values['Window']
            inputs['Snippets window size'] = [float(values['Window size 1']),   
                                              float(values['Window size 2'])]
            window.close()
            break
    
    return(inputs)    

def choose_events_for_whole_recording(inputs):

    default = {}
    default['All'] = 'Specific'
    default['Event type'] = 'Point event'
    default['Analysis name'] = 'Whole recording'
    
    # List all event types that are available in the tank.
    possible_event_types = ['Other list', 'Notes list',
                            'Video timestamp list', 'Ethovision event list']
    event_types = [type1 for type1 in possible_event_types if type1 in inputs['Options'].keys()]
    # Column for selecting the event names.
    checkbox_cols = []
    for type1 in event_types:
        col1 = []
        col1 += [[sg.Text(type1)],
                 [sg.Combo(['All', 'Specific'], key=type1+" All",
                           enable_events=True, default_value=default["All"])],
                 [sg.Combo(['Point event', 'Start-stop event'], 
                           default_value=default['Event type'], 
                           key="Event type "+type1)]]
        for event1 in inputs['Options'][type1]:
            col1 += [[sg.Checkbox(event1, default=False, 
                                  key="Checkbox "+type1+event1)]]
        checkbox_cols += [sg.Column(col1, vertical_alignment='top')]
    
    sg.theme("DarkTeal2")
    layout  = [[sg.T("")]]
    layout += [[sg.Text("Choose a custom name for this analysis"),
                sg.Input(key="Analysis name",enable_events=True, size=(20,1),
                         default_text=default["Analysis name"])], [sg.T("")]] 
    layout += [[sg.Text("Choose events to analyse.")]]
    layout += [checkbox_cols]
    layout += [[sg.T("")], [sg.Button("Submit")]]
    window  = sg.Window('Photometry Analysis', layout)
    while True:
        event, values = window.read()
        if event == sg.WIN_CLOSED or event=="Exit":
            window.close()
            sys.exit()
        for type1 in event_types:
            if values[type1+" All"] == "All":
                for event1 in inputs['Options'][type1]:
                    window.Element("Checkbox "+type1+event1).Update(visible=False)
            if values[type1+" All"] == "Specific":
                for event1 in inputs['Options'][type1]:
                    window.Element("Checkbox "+type1+event1).Update(visible=True)
        if event == "Submit":
            inputs['Analysis name'] = values['Analysis name']
            inputs = extract_event_data_from_cols(inputs, values)
            window.close()
            break 
        
    return(inputs)

# TDT -> whole recording -> ... -> select options before running the code

def choose_whole_recording_options(inputs):

    default = {}
    default["Remove"]  = 4
    default['Raw data'] = 'False'
    default['zScore']  = 'True'
    default['dFF']     = 'False'
    default['ISOS']    = 'False'
    default['GCaMP']   = 'False'
    sg.theme("DarkTeal2")        
    layout = []
    layout += [[sg.T("")], [sg.Text("Choose how much data from the start\n"+
                                    "should be removed (in secs, to account\n"+
                                    "for the artifact when turning on the LED)"), 
                            sg.Input(key="Remove",enable_events=True,
                                     default_text=default["Remove"], size=(8,1))]]
    layout += [[sg.T("")], [sg.Text("Export the raw data"), 
                            sg.Combo(['True','False'],key="Raw data",enable_events=True,
                                     default_value=default['Raw data'])]]
    layout += [[sg.T("")], [sg.Text("Create Z-Score plot?"), 
                            sg.Combo(['True','False'],key="zScore",enable_events=True,
                                     default_value=default['zScore'])]]
    layout += [[sg.T("")], [sg.Text("Create dFF plot?"), 
                            sg.Combo(['True','False'],key="dFF",enable_events=True,
                                     default_value=default['dFF'])]]
    layout += [[sg.T("")], [sg.Text("Create ISOS plot?"), 
                            sg.Combo(['True','False'],key="ISOS",enable_events=True,
                                     default_value=default['ISOS'])]]
    layout += [[sg.T("")], [sg.Text("Create GCaMP plot?"), 
                            sg.Combo(['True','False'],key="GCaMP",enable_events=True,
                                     default_value=default['GCaMP'])]]
    layout += [[sg.T("")], [sg.Button("Submit")]]
    window = sg.Window('Photometry Analysis', layout)
    while True:
        event, values = window.read()
        if event == sg.WIN_CLOSED or event=="Exit":
            window.close()
            sys.exit()
        elif event == "Submit":
            inputs['Remove'] = float(values['Remove'])
            inputs['Export ISOS']   = recognise_bool(values['ISOS'])
            inputs['Export GCaMP']  = recognise_bool(values['GCaMP'])
            inputs['Export dFF']    = recognise_bool(values['dFF'])
            inputs['Export zScore'] = recognise_bool(values['zScore'])
            inputs['Raw data'] = recognise_bool(values['Raw data'])
            window.close()
            break
    
    return(inputs)

# TDT -> FED3 -> choose the poke to analyse.

def choose_FED3_options(inputs):
            
    default = {}
    default["Analysis name"]   = 'FED3'
    default["Active poke"]     = 'Left'
    default["Poke to analyse"] = 'Left'
    sg.theme("DarkTeal2")
    layout = [[sg.T("")]]
    layout += [[sg.Text("Choose a custom name for this analysis"),
                sg.Input(key="Analysis name",enable_events=True, size=(20,1),
                         default_text=default["Analysis name"])], [sg.T("")], 
               [sg.Text('At the moment, "changing" just makes the '+
                        'active pokes the ones that preceded\n'+
                        'a pellet drop, even if this was not the poke '+
                        'that caused the pellet drop.')], [sg.T("")], 
               [sg.Text("Choose the active poke", size=(20,1)),
                sg.Combo(['Left', 'Right', 'Changing'], size=(8,1),
                        key="Active poke",enable_events=True,
                        default_value=default["Active poke"])], [sg.T("")],
               [sg.Text("Choose the poke to analyse", size=(20,1)),
                sg.Combo(['Left', 'Right', 'Both'], size=(8,1), 
                         key="Poke to analyse",enable_events=True,
                         default_value=default["Poke to analyse"])], [sg.T("")],
               [sg.Text('Confirm the event names for the pokes and pellet events')], [sg.T("")], 
               [sg.Text("Left poke", size=(8,1)),
                sg.Combo(inputs['Options list'],
                        key="Left",enable_events=True,
                        default_value=inputs['Name'][0])], [sg.T("")],
               [sg.Text("Right poke", size=(8,1)),
                sg.Combo(inputs['Options list'],
                        key="Right",enable_events=True,
                        default_value=inputs['Name'][1])], [sg.T("")],
               [sg.Text("Pellet drop", size=(8,1)),
                sg.Combo(inputs['Options list'],
                        key="Pellet",enable_events=True,
                        default_value=inputs['Name'][2])], [sg.T("")]]
    layout += [[sg.Button("Submit")]]
    window  = sg.Window('Photometry Analysis', layout)
    while True:
        event, values = window.read()
        if event == sg.WIN_CLOSED or event=="Exit":
            window.close()
            sys.exit()
        elif event == "Submit":
            inputs['Analysis name']   = values['Analysis name']
            inputs['Active poke']     = values["Active poke"]
            inputs['Poke to analyse'] = values["Poke to analyse"]
            inputs['Name']            = [values['Left'], values['Right'], values['Pellet']]
            window.close()
            break
    print('The active poke is: '+inputs['Active poke'])   
    print('Analyse this poke: ' +inputs['Poke to analyse'])

    return(inputs)

# TDT -> between TTLs -> choose the test     
    
def choose_between_TTL_test(inputs):
    
    default = {}
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
            inputs['Test'] = values['Test']
            window.close()
            break
    print('Test is '+inputs['Test'])
    
    return(inputs)

# TDT -> peri-event TTL -> choose the type of TTL pulse

def choose_type_TDT_event_between_TTL(inputs):

    # Choose the type of TTL.
    default = {}
    default["Event"] = 'Other'
    sg.theme("DarkTeal2")
    layout  = []
    layout += [[sg.T("")],[sg.Text("Choose the type of event."), 
                sg.Combo(['Other','Note','Video timestamp','Ethovision'],
                key="Event",enable_events=True,default_value=default["Event"])]]
    layout += [[sg.T("")], [sg.Button("Submit")]]
    window  = sg.Window('Photometry Analysis', layout)
    while True:
        event, values = window.read()
        if event == sg.WIN_CLOSED or event=="Exit":
            window.close()
            sys.exit()
        elif event == "Submit":
            inputs['Type'] = values["Event"]
            window.close()
            break
    print('Type of event pulse is '+inputs['Type'])
            
    return(inputs)

def choose_name_TDT_event_between_events(inputs):
            
    default = {}
    default["Analysis name"] = inputs['Test']
    sg.theme("DarkTeal2")
    event1_name = inputs['Custom'][0]
    event2_name = inputs['Custom'][1]
    event1 = inputs['Name'][0]
    event2 = inputs['Name'][1]
    layout = [[sg.T("")]]
    layout += [[sg.Text("Choose a custom name for this analysis"),
                sg.Input(key="Analysis name",enable_events=True, size=(20,1),
                         default_text=default["Analysis name"])], [sg.T("")], 
               [sg.Text('Confirm the event names for '+event1+' and '+event2)], [sg.T("")], 
               [sg.Text(event1_name, size=(8,1)),
                sg.Combo(inputs['Options list'],
                        key=event1,enable_events=True,
                        default_value=event1)], [sg.T("")],
               [sg.Text(event2_name, size=(8,1)),
                sg.Combo(inputs['Options list'],
                        key=event2,enable_events=True,
                        default_value=event2)], [sg.T("")]]
    layout += [[sg.Button("Submit")]]
    window  = sg.Window('Photometry Analysis', layout)
    while True:
        event, values = window.read()
        if event == sg.WIN_CLOSED or event=="Exit":
            window.close()
            sys.exit()
        elif event == "Submit":
            inputs['Analysis name'] = values['Analysis name']
            inputs['Name'][0] = values[event1]
            inputs['Name'][1] = values[event2]
            window.close()
            break

    return(inputs)

# TDT -> between TTLs -> ... -> select options before running the code.

def choose_between_events_options(inputs):
    
    default = {}
    default["Artifact RL"]      = ''
    default["Image"]            = 'True'
    default["Video"]            = 'False'
    default["zScore"]           = 'True'
    default["dFF"]              = 'False'
    default["ISOS"]             = 'False'
    default["GCaMP"]            = 'False'
    sg.theme("DarkTeal2")
    layout = [[sg.T("")],
        [sg.Text("Choose the artifact rejection level (optional)"), 
         sg.Input(key="Artifact",enable_events=True,
                  default_text=default["Artifact RL"],size=(20,1))],[sg.T("")],
        [sg.Text("Create video snippets of epochs?"), 
         sg.Combo(['True','False'],key="Video",enable_events=True,
                  default_value=default['Video'])],[sg.T("")],
        [sg.Text("Save preview image of data?"), 
         sg.Combo(['True','False'],key="Image",enable_events=True,
                  default_value=default['Image'])],[sg.T("")],
        [sg.Text("Save Z-Score data to CSV?"), 
         sg.Combo(['True','False'],key="zScore",enable_events=True,
                  default_value=default['zScore'])],[sg.T("")],
        [sg.Text("Save dFF data to CSV?"), 
         sg.Combo(['True','False'],key="dFF",enable_events=True,
                  default_value=default['dFF'])],[sg.T("")],
        [sg.Text("Save ISOS data to CSV?"), 
         sg.Combo(['True','False'],key="ISOS",enable_events=True,
                  default_value=default['ISOS'])],[sg.T("")],
        [sg.Text("Save GCaMP data to CSV?"), 
         sg.Combo(['True','False'],key="GCaMP",enable_events=True,
                  default_value=default['GCaMP'])],[sg.T("")],
        [sg.Button("Submit")]]
    window = sg.Window('Photometry Analysis', layout)
    while True:
        event, values = window.read()
        if event == sg.WIN_CLOSED or event=="Exit":
            window.close()
            sys.exit()
        elif event == "Submit":
            inputs['Artifact RL']     = recognise_artifact(values['Artifact'])
            inputs['Image']           = recognise_bool(values["Image"])
            inputs['Create snippets'] = recognise_bool(values['Video'])
            inputs['Export ISOS']   = recognise_bool(values['ISOS'])
            inputs['Export GCaMP']  = recognise_bool(values['GCaMP'])
            inputs['Export dFF']    = recognise_bool(values['dFF'])
            inputs['Export zScore'] = recognise_bool(values['zScore'])
            window.close()
            break
        
    return(inputs)
