from GUI_and_data_processing.Data_processing import analyse_Ethovision_data
from Root_Morales_lab_codes.FibPhoEpocAveraging import FiPhoEpocAveraging
from tdt import StructType
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

def find_lists_of_events(inputs):
    
    inputs['Options'] = {}

    # OTHER
    
    # Create a list of system variables to exclude from possible TTLMs.
    TTLM_exclude = ['PC0_','PC1_','PC2_','PC3_','PC0/','PC1/','PC2/','PC3/',
                    'Cam1','Cam2','BGte','Gate','Note','Tick']
    # If using setup A, exclude the score events from setup B and vice versa.
    if inputs['Setup'] == 'Setup A':
        TTLM_exclude += ['Blft','Brgt','BRgt','Bplt']
    elif inputs['Setup'] == 'Setup B':
        TTLM_exclude += ['Alft','Argt','ARgt','Aplt']
    TTLM_list = inputs['Tank'].epocs.keys()
    TTLM_list = list(set(TTLM_list)) # Remove non-unique elements.
    TTLM_list = [TTLM for TTLM in TTLM_list if TTLM not in TTLM_exclude]
    if len(TTLM_list) > 0:
        inputs['Options']['Other list'] = TTLM_list
        
    # NOTE
            
    # If the notes are all 'none', go back into the Notes.txt file and find comments in "" marks.
    if 'Note' in inputs['Tank'].epocs.keys() and np.all(inputs['Tank'].epocs.Note.notes == 'none'):
        notes_txt_path = os.path.join(inputs['Import location'], 'Notes.txt')
        with open(notes_txt_path, 'r') as notes_file:
            notes_lines = notes_file.readlines()
        def find_comment(note):
            ind = [i for i in range(len(note)) if note[i]=='"']
            return(note[ind[0]+1:ind[1]])
        notes_lines = [find_comment(note) for note in notes_lines if note[:5]=='Note-']
        inputs['Tank'].epocs.Note.notes = np.array(notes_lines)
    # Create a list of notes.
    notes_list = list(inputs['Tank'].epocs.Note.notes)
    notes_list = list(set(notes_list)) # Remove non-unique elements.
    if len(notes_list) > 0:
        inputs['Options']['Notes list'] = notes_list
        
    # VIDEO TIMESTAMP

    if 'notes' in inputs['Tank'].epocs[inputs['Camera']].keys():
        list_video_events = zip(inputs['Tank'].epocs[inputs['Camera']].notes.index, 
                                inputs['Tank'].epocs[inputs['Camera']].notes.notes)
        list_video_events = dict(sorted(list_video_events))
        list_video_events = list(list_video_events.values())
        if len(list_video_events) > 0:
            inputs['Options']['Video timestamp list'] = list_video_events
            
    # ETHOVISION

    # Check all the excel files in the import tank, and only use the one from Ethovision.
    excel_files = [file for file in os.listdir(inputs['Import location']) if file[-5:]=='.xlsx']
    excel_file_found = False
    for file in excel_files:
        import_destination = inputs['Import location']+'/'+file
        df = pd.read_excel(import_destination, sheet_name=0)
        if list(df[:0])[0] == 'Number of header lines:':
            excel_file_found = True
            break
    if excel_file_found == True:
        # Find all the column headings in the Ethovision files (that are not position, speed, etc.)
        num_headers = int(list(df[:0])[1])
        rows_skip = list(range(0,num_headers-2)) + [num_headers-1]
        headings_to_exclude = ["Trial time", "Recording time", "X center", "Y center", "X nose", "Y nose", 
                               "X tail", "Y tail", "Area", "Areachange", "Elongation", "Direction", 
                               "Distance moved(nose-point)", "Distance moved(center-point)", "Velocity(nose-point)", 
                               "Velocity(center-point)", "Result 1", "Distance moved", "Velocity"]
        # Exclude the columns from headings_to_exclude.
        df = pd.read_excel(import_destination, sheet_name=0, skiprows=rows_skip, 
                           usecols=lambda x: x not in headings_to_exclude)
        if len(df.columns) > 0:
            inputs['Options']['Ethovision event list'] = list(df.columns)
    
    return(inputs)

def extract_event_data_from_cols(inputs, values):
    
    # List all event types that are available in the tank.
    possible_event_types = ['Other list', 'Notes list',
                            'Video timestamp list','Ethovision event list']
    event_types = [type1 for type1 in possible_event_types if type1 in inputs['Options'].keys()]
    
    # Put all the GUI inputs into a dictionary called 'data.'
    inputs['Name'] = []
    inputs['Epoch type'] = []
    inputs['All or specific'] = []
    inputs['Event type'] = []
    
    for type1 in event_types:
        
        names = [event1 for event1 in inputs['Options'][type1] 
                 if values["Checkbox "+type1+event1]==True]
        
        if values[type1+' All'] == 'All':
            inputs['Name']            += ['All']
            inputs['Epoch type']      += [type1]
            inputs['All or specific'] += [values[type1+' All']]
            inputs['Event type']      += [values['Event type '+type1]]
                                          
        if values[type1+' All'] != 'All' and len(names) > 0:
            inputs['Name'] += names
            inputs['Epoch type'] += len(names)*[type1]
            inputs['All or specific'] += len(names)*[values[type1+' All']]
            inputs['Event type'] += len(names)*[values['Event type '+type1]]
        
    return(inputs)

def define_all_whole_recording_events(inputs):
    
    SCORE_EVENT = 'Use_these_epochs'
    SCORE_DICT = {"onset":    [],
                  "offset":   [],
                  "notes":    [],
                  "event_type":[]}
    # This should be a proportion of the total recording time from 0 to 1.
    # This defines the width of the point events (rather than events with starts
    # and stops).
    total_duration = inputs['Tank'].info.duration.total_seconds()
    width_line = total_duration * 0.0015
    
    # Re-organise the data from this format:
    # inputs['Name'] = ['Left','Start']
    # inputs['Epoch type'] = ['Other list','Notes list']
    # inputs['All or specific'] = ['Specific','Specific']
    # inputs['Event type'] = ['Point event', 'Point event']
    # ... into this format:
    # inputs['Event'] = {'Name':{'Other list':['Left'], 'Notes list':['Start']}, 
    #                    'All or specific':{'Other list':'Specific', 'Notes list':'Specific'}, 
    #                    'Event type':{'Other list':'Point event', 'Notes list':'Point event'}}
    types_list = list(set(inputs['Epoch type']))
    indices = {type1:[] for type1 in types_list}
    for i in range(len(inputs['Epoch type'])):
        type1 = inputs['Epoch type'][i]
        indices[type1] += [i]
    inputs['Event'] = {'Name':           {type1:list(np.array(inputs['Name'])[indices[type1]]) 
                                          for type1 in types_list}, 
                       'All or specific':{type1:list(np.array(inputs['All or specific'])[indices[type1]])[0] 
                                          for type1 in types_list}, 
                       'Event type':     {type1:list(np.array(inputs['Event type'])[indices[type1]])[0] 
                                          for type1 in types_list}}
    
    # For every event type, add these events to the 'Use_these_epochs' event definition.
    for key in inputs['Event']['Name']:
    
        if key == 'Other list':
            # Find the event names selected.
            if inputs['Event']['Name'][key] == 'All':
                other_list = inputs['Options'][key]
            else:
                other_list = inputs['Event']['Name'][key] 
            # Add these into the event called 'Analyse_this_event'.
            for event in other_list:
                onsets  = list(inputs['Tank'].epocs[event].onset)
                offsets = list(inputs['Tank'].epocs[event].offset)
                if inputs['Event']['Event type'][key] == 'Point event':
                    SCORE_DICT['onset']  += onsets
                    SCORE_DICT['offset'] += list(np.array(onsets) + width_line)
                elif inputs['Event']['Event type'][key] == 'Start-stop event':   
                    SCORE_DICT['onset']  += onsets
                    SCORE_DICT['offset'] += offsets
                SCORE_DICT['notes']      += len(onsets)*[event]
                SCORE_DICT['event_type'] += len(onsets)*[inputs['Event']['Event type'][key]]
            
        elif key == 'Notes list':
            # Find the indices of the note events selected.
            if inputs['Event']['Name'][key] == 'All':
                indices = [i for i in range(len(inputs['Options'][key]))]
            else:
                indices = [i for i in range(len(inputs['Options'][key])) 
                           if inputs['Options'][key][i] in inputs['Event']['Name'][key]]
            # Put the note timestamps into the event called 'Analyse_this_event'.
            onsets  = list(inputs['Tank'].epocs.Note.onset[indices])
            offsets = list(inputs['Tank'].epocs.Note.offset[indices])
            if inputs['Event']['Event type'][key] == 'Point event':
                SCORE_DICT['onset']  += onsets
                SCORE_DICT['offset'] += list(np.array(onsets) + width_line)
            elif inputs['Event']['Event type'][key] == 'Start-stop event':   
                SCORE_DICT['onset']  += onsets
                SCORE_DICT['offset'] += offsets
            SCORE_DICT['notes']      += list(inputs['Tank'].epocs.Note.notes[indices])
            SCORE_DICT['event_type'] += len(onsets)*[inputs['Event']['Event type'][key]]
                
        elif key == 'Video timestamp list':
            # Find the indices of the video timestamps selected.
            if inputs['Event']['Name'][key] == 'All':
                indices = [i for i in range(len(inputs['Options'][key]))]
            else:
                indices = [i for i in range(len(inputs['Options'][key])) 
                           if inputs['Options'][key][i] in inputs['Event']['Name'][key]]
            # Put the video timestamps into the event called 'Analyse_this_event'.
            onsets  = list(inputs['Tank'].epocs[inputs['Camera']].notes.ts[indices])
            offsets = onsets[1:] + [np.inf]
            if inputs['Event']['Event type'][key] == 'Point event':
                SCORE_DICT['onset']  += onsets
                SCORE_DICT['offset'] += list(np.array(onsets) + width_line)
            elif inputs['Event']['Event type'][key] == 'Start-stop event':   
                SCORE_DICT['onset']  += onsets
                SCORE_DICT['offset'] += offsets
            SCORE_DICT['notes']      += list(inputs['Tank'].epocs[inputs['Camera']].notes.notes[indices])
            SCORE_DICT['event_type'] += len(onsets)*[inputs['Event']['Event type'][key]]
    
        elif key == 'Ethovision event list':      
            
            # Find the event names selected.
            if inputs['Event']['Name'][key] == 'All':
                ethovision_list = inputs['Options'][key]
            else:
                ethovision_list = inputs['Event']['Name'][key] 
            # Add these into the event called 'Analyse_this_event'.
            for event in ethovision_list:
                df_results = analyse_Ethovision_data(inputs, event)
                onsets  = df_results['Bout start time (secs)']
                offsets = df_results['Bout end time (secs)']
                if inputs['Event']['Event type'][key] == 'Point event':
                    SCORE_DICT['onset']  += onsets
                    SCORE_DICT['offset'] += list(np.array(onsets) + width_line)
                elif inputs['Event']['Event type'][key] == 'Start-stop event':   
                    SCORE_DICT['onset']  += onsets
                    SCORE_DICT['offset'] += offsets
                SCORE_DICT['notes']      += len(onsets)*[event]
                SCORE_DICT['event_type'] += len(onsets)*[inputs['Event']['Event type'][key]]
    
    # Sort the events by onset time.
    SCORE_DICT = pd.DataFrame(SCORE_DICT)
    SCORE_DICT = SCORE_DICT.sort_values(by=['onset'])
    SCORE_DICT = SCORE_DICT.to_dict('list')
    
    # Add the combined event data to the TDT tank.
    SCORE_DICT["name"]     = SCORE_EVENT,
    SCORE_DICT['type_str'] = 'epocs'
    SCORE_DICT['data']     = np.array(range(1,len(SCORE_DICT['onset'])+1))
    SCORE_DICT['onset']    = np.array(SCORE_DICT['onset'])
    SCORE_DICT['offset']   = np.array(SCORE_DICT['offset'])
    # if SCORE_DICT['offset'][-1] == np.inf:
    #     SCORE_DICT['offset'][-1] = total_duration
    SCORE_DICT['notes']    = np.array(SCORE_DICT['notes'])
    inputs['Tank'].epocs[SCORE_EVENT] = StructType(SCORE_DICT)
        
    return(inputs)

def whole_recording_analysis(inputs):
    
    # This creates the data to analyse the whole recording using the 
    # FibPhoEpocAveraging code.
    # This is normally intended for analysing short events.
    # Instead, I create an event called 'Whole_recording', which starts at a 
    # given timestamp and ends at the end of the recording.
    
    # Modify the 'inputs' dictionary, before it is run through the peri-events code.
    
    # Set the t-range and baseline periods.
    start_time = 0
    end_time   = int(inputs['Tank'].info.duration.total_seconds() - inputs['Remove'] - 1)
    whole_recording_interval = [start_time, end_time]
    inputs['t-range'] = whole_recording_interval
    inputs['Baseline period'] = whole_recording_interval
    
    # Remove the data at the start of the recording.
    for signal in ['ISOS', 'GCaMP']:
        stream = inputs['Tank'].streams[inputs[signal]].data
        stream_thresholded = [t for t in stream if t >= inputs['Remove']]
        inputs['Tank'].streams[inputs[signal]].data = np.array(stream_thresholded)
    
    # Add unimportant variables that are needed by the FibPhoEpocAveraging code.
    inputs['Artifact RL'] = np.inf
    inputs['Analysis name'] = 'Whole recording'
    inputs['Baseline type'] = 'Specific'
    
    # Add the event needed to analyse the whole recording to the TDT tank.
    SCORE_DICT = {"name":     'Analyse_this_event',
                  "onset":    np.array([inputs['Remove']]),
                  "offset":   np.array([inputs['Remove'] + 0.01]),
                  "type_str": 'epocs',
                  "data":     np.array([1])}
    inputs['Tank'].epocs['Analyse_this_event'] = StructType(SCORE_DICT)
    
    # Run these inputs through the peri-events code.
    inputs, outputs = FiPhoEpocAveraging(inputs)
    
    return(inputs, outputs)

def create_export_plots(inputs, outputs):
    
    outputs['Plots'] = {}
    outputs['Dataframe'] = {}
    
    for data_type in ['zScore', 'dFF', 'ISOS', 'GCaMP']:
        
        if inputs['Export '+data_type] == True:
    
            # Plot the signal data.
            fig = plt.figure(figsize=(20,4))
            ax  = fig.add_subplot()
            time_stamps = [t for t in outputs['Timestamps'] if t>=4]
            time_stamps = outputs['Timestamps']
            plt.plot(time_stamps, outputs[data_type][0], linewidth=2, color='green', label=data_type)
            
            # Create a dictionary that converts event names to unique colors.
            notes = inputs['Tank'].epocs['Use_these_epochs'].notes
            notes = list(set(notes))
            cmap = plt.cm.get_cmap('tab20')
            event_color_point = {notes[i]:cmap(0.1*i) for i in range(len(notes))}
            event_color_inter = {notes[i]:cmap(0.05+0.1*i) for i in range(len(notes))}
            
            # Mark the events over the signal data.
            starts = inputs['Tank'].epocs['Use_these_epochs'].onset
            ends   = inputs['Tank'].epocs['Use_these_epochs'].offset
            events = inputs['Tank'].epocs['Use_these_epochs'].notes
            etypes = inputs['Tank'].epocs['Use_these_epochs'].event_type
            for i in range(len(starts)):
                if etypes[i] == 'Point event':
                    plt.axvspan(starts[i], ends[i], color=event_color_point[events[i]], 
                                label=events[i])
                else:
                    plt.axvspan(starts[i], ends[i], color=event_color_inter[events[i]], 
                                label=events[i])
                
            # Create a legend.
            handles, labels = ax.get_legend_handles_labels()
            dict1 = {}
            for i in range(len(handles)):
                dict1[labels[i]] = handles[i]
            plt.legend(dict1.values(), dict1.keys(), loc='center left', bbox_to_anchor=(1, 0.5))
            
            # Choose the time axis range to be the earliest start event and the latest end event.
            ax.set_ylabel(data_type)
            ax.set_xlabel('Time (secs)')
            ax.set_title('Whole recording with events highlighted')
            plt.tight_layout()
            
            # Create raw data file.
            raw_data = pd.DataFrame()
            raw_data['Timestamps (secs)'] = time_stamps
            raw_data['Timestamps (shifted)'] = raw_data['Timestamps (secs)'].shift()
            raw_data.at[0,'Timestamps (shifted)'] = raw_data.at[0,'Timestamps (secs)']
            raw_data['Intervals'] = list(zip(raw_data['Timestamps (shifted)'], 
                                             raw_data['Timestamps (secs)']))
            raw_data[data_type] = outputs[data_type][0]
            def add_event_info(time, onsets, offsets, notes):
                int2 = pd.Interval(left=time[0], right=time[1])
                for i in range(len(onsets)):
                    int1 = pd.Interval(left=onsets[i], right=offsets[i])
                    if int1.overlaps(int2) == True:
                        return(notes[i])
                return(np.nan)
            raw_data['Events'] = raw_data['Intervals'].apply(add_event_info, 
                                    onsets=starts, offsets=ends, notes=events)
            raw_data = raw_data.drop(['Timestamps (shifted)', 'Intervals'], axis=1)
            
            # Save the plots and raw data.
            outputs['Plots'][data_type] = fig
            outputs['Dataframe'][data_type] = raw_data
            return(outputs)
            
def export_whole_recording_plots(inputs, outputs):

    for data_type in ['zScore', 'dFF', 'ISOS', 'GCaMP']:
        
        if inputs['Export '+data_type] == True:

            export_name = (os.path.basename(inputs['Import location']) + "_"+data_type+"_" + 
                           inputs['Analysis name'] + "_" + inputs['Setup'].replace(' ','_') + ".png")
            export_destination = os.path.join(inputs['Export location'], export_name)
            outputs['Plots'][data_type].savefig(export_destination)
            
def export_whole_recording_data(inputs, outputs):
    
    for data_type in ['zScore', 'dFF', 'ISOS', 'GCaMP']:
        
        if inputs['Raw data'] == True and inputs['Export '+data_type] == True:
    
            export_name = (os.path.basename(inputs['Import location']) + "_"+data_type+"_" + 
                           inputs['Analysis name'] + "_" + inputs['Setup'].replace(' ','_') + ".csv")
            export_destination = os.path.join(inputs['Export location'], export_name)
            outputs['Dataframe'][data_type].to_csv(export_destination, index=False)
    