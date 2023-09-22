import numpy as np
from tdt import StructType
import pandas as pd
import sys
from copy import deepcopy
import os

def find_options_FED3(inputs):

    # Create a list of system variables to exclude from possible TTLMs.
    TTLM_exclude = ['PC0_','PC1_','PC2_','PC3_','PC0/','PC1/','PC2/','PC3/',
                    'Cam1','Cam2','BGte','Gate','Note','Tick']
    # If using setup A, exclude the score events from setup B and vice versa.
    TTLM_list = inputs['Tank'].epocs.keys()
    TTLM_list = list(set(TTLM_list)) # Remove non-unique elements.
    TTLM_list = [TTLM for TTLM in TTLM_list if TTLM not in TTLM_exclude]
    inputs['Options list'] = TTLM_list
    
    if len(inputs['Options list']) == 0:
        print('There are no TDT epoc events to find FED3 events from in this tank.')
        sys.exit()
    
    return(inputs)

def find_event_names(inputs):
    
    if inputs['Setup'] == 'Setup A':
        possible_rewards = ['Arwd','ARwd']
        possible_pellets = ['Aplt','APlt']
        possible_lefts   = ['Alft','ALft']
        possible_rights  = ['Argt','ARgt']
    
    elif inputs['Setup'] == 'Setup B':
        possible_rewards = ['Brwd','BRwd']
        possible_pellets = ['Bplt','BPlt']
        possible_lefts   = ['Blft','BLft']
        possible_rights  = ['Brgt','BRgt']
            
    # Find the element from data.epocs that matches one of the possible pellet and active poke names.
    REWARD = list(set(possible_rewards).intersection(inputs['Options list']))
    PELLET = list(set(possible_pellets).intersection(inputs['Options list']))
    LEFT   = list(set(possible_lefts).intersection(inputs['Options list']))
    RIGHT  = list(set(possible_rights).intersection(inputs['Options list']))
    
    # If the event names cannot be found, try the generic "Left", "Right", "Pelt" or "Rewd".
    # They could refer to setup A or setup B, so those were not used first.
    if len(REWARD) == 0:
        possible_rewards += ['Rewd']
        REWARD = list(set(possible_rewards).intersection(inputs['Options list']))
    if len(PELLET) == 0:
        possible_pellets += ['Pelt']
        PELLET = list(set(possible_pellets).intersection(inputs['Options list']))
    if len(LEFT) == 0:
        possible_lefts += ['Left']
        LEFT = list(set(possible_lefts).intersection(inputs['Options list']))
    if len(RIGHT) == 0:
        possible_rights += ['Rght','RGht']
        RIGHT = list(set(possible_rights).intersection(inputs['Options list']))
    
    # If we can automatically detect the event names for left poke, right poke 
    # and pellet, make those appear automatically.
    inputs['Custom'] = ['Left', 'Right', 'Pellet', 'Rewarded', 'Extra event']
    inputs['Name'] = []
    if len(LEFT) == 1:
        inputs['Name'] += [LEFT[0]]
    else:
        inputs['Name'] += [inputs['Options list'][0]]
    if len(RIGHT) == 1:
        inputs['Name'] += [RIGHT[0]]
    else:
        inputs['Name'] += [inputs['Options list'][0]]
    if len(PELLET) == 1:
        inputs['Name'] += [PELLET[0]]
    else:
        inputs['Name'] += [inputs['Options list'][0]]    
    if len(REWARD) == 1:
        inputs['Name'] += [REWARD[0]]
    else:
        inputs['Name'] += [inputs['Options list'][0]]    
    inputs['Name'] += [inputs['Options list'][0]] # For the extra event.

    return(inputs)
    
def create_unique_TDT_event_FED3(inputs):
    
    # Re-organise the event data about left pokes, right pokes and pellet drops.
    events = {'Onsets':[], 'Notes':[]}
    
    for i in range(len(inputs['Custom'])):
        event_name = inputs['Name'][i]
        event_tag  = inputs['Custom'][i]
        onsets = list(inputs['Tank'].epocs[event_name].onset)
        events['Onsets'] += onsets
        events['Notes']  += len(onsets)*[event_tag]
        
    # Convert these lists of dictionaries to a dataframe and sort all events by time.
    events = pd.DataFrame(events)
    events = events.sort_values(by=['Onsets'])
    
    # Create a unique TDT event.
    onsets   = np.array(events['Onsets'])
    offsets  = np.array(list(events['Onsets'][1:])+[np.inf])
    data     = np.array(range(1,len(onsets)+1))
    notes    = np.array(events['Notes'])
    
    # Create a new event.
    SCORE_EVENT = 'Analyse_this_event'
    SCORE_DICT = {"name":     SCORE_EVENT,
                  "onset":    onsets,
                  "offset":   offsets,
                  "data":     data,
                  "notes":    notes}
    inputs['Tank'].epocs[SCORE_EVENT] = StructType(SCORE_DICT)
    
    return(inputs)

def convert_multiindex_headings_to_rows(df):
    # Convert the multiindex headings to rows.
    df_cols = df.columns.to_frame().T
    df_cols.columns = range(len(df_cols.columns))
    df.columns = range(len(df.columns))
    df = pd.concat([df_cols,df])
    return(df)

def create_export_data_FED3(inputs, outputs):
    
    pd.options.mode.chained_assignment = None  # default='warn'
    outputs['Overall'] = {}
    for i in range(len(inputs['Custom'])):
        event_tag = inputs['Custom'][i]
        outputs[event_tag] = {}
    
    for stat in ['zScore', 'dFF', 'ISOS', 'GCaMP']:
        df = outputs['Dataframe'][stat]
        outputs['Overall'][stat] = convert_multiindex_headings_to_rows(df.copy())
        
        for i in range(len(inputs['Custom'])):
            
            # Remove e.g. the 'Left poke' columns from the sheet with 'Left poke', 'Right poke', ...
            event_tag    = inputs['Custom'][i]
            cols_subset  = list(tuple(df.columns[0:3]))
            cols_subset += [col for col in df.columns if col[2] == event_tag]
            df_subset = df[cols_subset].copy()
            
            # Redo the mean columns, for the new shortened dataframes. 
            mean_events_col = df_subset.columns[2]
            data_cols = df_subset.columns[3:]
            new_mean_col = df_subset[data_cols].mean(axis=1)
            df_subset[mean_events_col] = new_mean_col
            
            # Convert the multiindex headings to rows.
            outputs[event_tag][stat] = convert_multiindex_headings_to_rows(df_subset)
            
    return(outputs)

def export_analysed_data_FED3(inputs, outputs):
    
    for data_type in ['zScore', 'dFF', 'ISOS', 'GCaMP']:
        
        if inputs['Export '+data_type] == True:

            # Export the data.
            export_name = (os.path.basename(inputs['Import location']) + "_"+data_type+"_" + 
                           inputs['Analysis name'] + "_" + inputs['Setup'].replace(' ','_') + ".xlsx")
            export_destination = os.path.join(inputs['Export location'], export_name)
            with pd.ExcelWriter(export_destination) as writer:
                outputs['Overall'][data_type].to_excel(writer, sheet_name='Overall', index=False, header=False)
                for sheet_name in inputs['Custom']:
                    outputs[sheet_name][data_type].to_excel(writer, sheet_name=sheet_name, index=False, header=False)
            
def export_preview_image_FED3(inputs, outputs):

    export_name = (os.path.basename(inputs['Import location']) + "_" + 
                   inputs['Analysis name'] + "_" + inputs['Setup'].replace(' ','_') + ".png")
    export_destination = os.path.join(inputs['Export location'], export_name)
    outputs['Figure'].savefig(export_destination)
    