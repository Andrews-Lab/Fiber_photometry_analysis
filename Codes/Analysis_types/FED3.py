import numpy as np
from tdt import read_block, StructType
import pandas as pd
import sys
from copy import deepcopy
import os

def find_options_FED3(inputs):
    
    print('\nPlease wait while the TDT tank is importing...')
    inputs['Tank'] = read_block(inputs['Import location'])
    print('')

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
        possible_pellets = ['Aplt','APlt']
        possible_lefts   = ['Alft','ALft']
        possible_rights  = ['Argt','ARgt']
    
    elif inputs['Setup'] == 'Setup B':
        possible_pellets = ['Bplt','BPlt']
        possible_lefts   = ['Blft','BLft']
        possible_rights  = ['Brgt','BRgt']
            
    # Find the element from data.epocs that matches one of the possible pellet and active poke names.
    PELLET = list(set(possible_pellets).intersection(inputs['Options list']))
    LEFT   = list(set(possible_lefts).intersection(inputs['Options list']))
    RIGHT  = list(set(possible_rights).intersection(inputs['Options list']))
    
    # If the event names cannot be found, try the generic "Left", "Right" or "Pelt".
    # They could refer to setup A or setup B, so those were not used first.
    if len(PELLET) == 0 or len(LEFT) == 0 or len(RIGHT) == 0:
        possible_lefts   += ['Left']
        possible_rights  += ['Rght','RGht']
        possible_pellets += ['Pelt']
        PELLET = list(set(possible_pellets).intersection(inputs['Options list']))
        LEFT   = list(set(possible_lefts).intersection(inputs['Options list']))
        RIGHT  = list(set(possible_rights).intersection(inputs['Options list']))
    
    # If we can automatically detect the event names for left poke, right poke 
    # and pellet, make those appear automatically.
    inputs['Custom'] = ['Left', 'Right', 'Pellet']
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

    return(inputs)
    
def create_unique_TDT_event_FED3(inputs):
    
    # Re-organise the event data about left pokes, right pokes and pellet drops.
    events = {'Onsets':[], 'Notes':[]}
    
    if inputs['Poke to analyse'] == 'Left':
        onsets = list(inputs['Tank'].epocs[inputs['Name'][0]].onset)
        events['Onsets'] += onsets
        events['Notes']  += len(onsets)*['Left']
        
    elif inputs['Poke to analyse'] == 'Right':
        onsets = list(inputs['Tank'].epocs[inputs['Name'][1]].onset)
        events['Onsets'] += onsets
        events['Notes']  += len(onsets)*['Right']
        
    elif inputs['Poke to analyse'] == 'Both':
        onsets = list(inputs['Tank'].epocs[inputs['Name'][0]].onset)
        events['Onsets'] += onsets
        events['Notes']  += len(onsets)*['Left']
        onsets = list(inputs['Tank'].epocs[inputs['Name'][1]].onset)
        events['Onsets'] += onsets
        events['Notes']  += len(onsets)*['Right']
        
    onsets = list(inputs['Tank'].epocs[inputs['Name'][2]].onset)
    events['Onsets'] += onsets
    events['Notes']  += len(onsets)*['Pellet']
        
    # Convert these lists of dictionaries to a dataframe and sort all events by time.
    events = pd.DataFrame(events)
    events = events.sort_values(by=['Onsets'])
    
    # Add a column to this dataframe with the rewarded status of pokes.
    if inputs['Active poke'] == 'Changing':
        events['Rewarded'] = ''
        # If a pellet drop happens after a nose poke, that nose poke is rewarded.
        # Otherwise, the poke is non-rewarded.
        for i in range(len(events)):
            if events.at[i,'Notes'] in ['Left', 'Right']:
                if i != len(events)-1 and events.at[i+1,'Notes'] == 'Pellet':
                    events.at[i,'Rewarded'] = 'Rewarded'
                else:
                    events.at[i,'Rewarded'] = 'Non-rewarded'
    else:
        def is_rewarded(current_poke, active_poke):
            # Check whether the current poke is active.
            # The values for current_poke can only be 'Left' or 'Right'.
            if current_poke == 'Pellet':
                return('')
            elif current_poke == active_poke:
                return('Rewarded')
            else:
                return('Non-rewarded')
        events['Rewarded'] = events['Notes'].apply(is_rewarded, 
                                    active_poke=inputs['Active poke'])
    
    # Create a unique TDT event.
    onsets   = np.array(events['Onsets'])
    offsets  = np.array(list(events['Onsets'][1:])+[np.inf])
    data     = np.array(range(1,len(onsets)+1))
    notes    = np.array(events['Notes'])
    rewarded = np.array(events['Rewarded'])
    
    # Create a new event.
    SCORE_EVENT = 'Analyse_this_event'
    SCORE_DICT = {"name":     SCORE_EVENT,
                  "onset":    onsets,
                  "offset":   offsets,
                  "data":     data,
                  "notes":    notes,
                  "rewarded": rewarded}
    inputs['Tank'].epocs[SCORE_EVENT] = StructType(SCORE_DICT)
    
    return(inputs)

def create_export_data_FED3(inputs, outputs):
    
    # Check whether a t-range extended beyond the recording and was therefore excluded.
    # If so, exclude the corresponding notes.
    event_onsets = inputs['Tank'].epocs['Analyse_this_event'].onset
    if len(event_onsets) != len(outputs['zScore']):
        print('\nPLEASE NOTE: the last '+str(len(event_onsets)-len(outputs['zScore'])) + ' events '+
              'have been excluded, because their window durations go past the end of '+
              'the recording.\n')
    
    # Create the pandas dataframes to export as CSV files.
    # Only include the keys 'zScore', 'dFF', 'ISOS' and 'GCaMP'.
    remove_data = ['Timestamps', 'Figure']
    result_arrays = deepcopy(outputs) # Numpy arrays, rather than lists.
    results = deepcopy(outputs) # Eventually dataframes with headers
    for key in remove_data:
        result_arrays.pop(key)
        results.pop(key)
    
    Zscore_max_threshold = 1
    
    for stat in results.keys(): # zScore, dFF, ISOS and GCaMP data types.

        # Create a header for the results.
        header = {}
        header['Names'] = [str(i) for i in range(1,len(result_arrays[stat])+1)]
        header['Times'] = list(inputs['Tank'].epocs['Analyse_this_event'].onset)
        # Make sure the analysed data columns and event names are the same length.
        header['Times'] = header['Times'][:len(header['Names'])]
        all_headers = list(zip(*[header[key] for key in header.keys()]))
        all_headers = pd.MultiIndex.from_tuples(all_headers)
        
        # Create a pandas dataframe for the poke names and Z-score data.
        results[stat] = pd.DataFrame(np.transpose(results[stat]), columns=all_headers)
        
        # Add in extra columns.
        results[stat].insert(0, ('Mean of events',''), results[stat].mean(axis=1))
        results[stat].insert(0, ('Time stamps (secs)',''), outputs['Timestamps'])
        results[stat].insert(0, ('','Time of event onset (secs)'), ['']*len(outputs['Timestamps']))
            
        # Define all the extra rows as entries in a dictionary.
        extra_rows = {}
            
        # If there is notes information, add many more header rows.
        if 'notes' in inputs['Tank'].epocs['Analyse_this_event'].keys():
    
            list_notes = list(inputs['Tank'].epocs['Analyse_this_event'].notes)
            # Make sure the analysed data columns and event names are the same length.
            list_notes = list_notes[:len(header['Names'])]
            list_differences = ['']
            for i in range(len(list_notes)-1):
                if list_notes[i] != list_notes[i+1]:
                    list_differences += ['Different']
                else:
                    list_differences += ['Same']
            extra_rows['Event note'] = ['',''] + list_notes
            extra_rows['Preceding note'] = ['',''] + list_differences
            
        # # SLOPE OF Z-SCORE DATA.
        # # UNCOMMENT THIS TO RE-INCLUDE THESE STATS.
        # list_slopes = [linregress(outputs['Timestamps'],col)[0] for col in result_arrays[stat]]
        # extra_rows['Slope'] = ['',''] + list_slopes
        # slope_cutoff = 0.1 # Ensure this is a positive number.
        # list_directions = []
        # for i in range(len(list_slopes)):
        #     if list_slopes[i] >= slope_cutoff:
        #         list_directions += ['Positive']
        #     elif list_slopes[i] <= -slope_cutoff:
        #         list_directions += ['Negative']
        #     else:
        #         list_directions += ['Flat']
        # extra_rows['Slope direction where [0.1,-0.1] is flat'] = ['',''] + list_directions
        
        # Find the time to baseline after the event.
        if stat == 'zScore':
            BL_thresh = 0.1
            post_event_ind = [i for i in range(len(outputs['Timestamps']))
                              if outputs['Timestamps'][i] >= 0]
            list_baseline_times = []
            for col in result_arrays[stat]:
                found_baseline = False
                for i in post_event_ind:
                    if (-BL_thresh <= col[i] <= BL_thresh):
                        list_baseline_times += [outputs['Timestamps'][i]]
                        found_baseline = True
                        break
                # If the signal does not return to baseline after the event, 
                # list the last possible time point.
                if found_baseline == False:
                    list_baseline_times += [outputs['Timestamps'][-1]]
            extra_rows[f'Time to baseline (between -{BL_thresh} and {BL_thresh}) after event'] = (
                ['',''] + list(list_baseline_times))

        list_max_Zscores = [max(col) for col in result_arrays[stat]]
        extra_rows['Max values'] = ['',''] + list_max_Zscores
        # Find the time point at which the max Z-score values occur.
        list_time_Zscores = [outputs['Timestamps'][np.argmax(col)] for col in result_arrays[stat]]
        extra_rows['Time of max values'] = ['',''] + list_time_Zscores
        list_thresh_Zscores = [('Yes' if max1>Zscore_max_threshold else 'No') 
                               for max1 in list_max_Zscores]
        extra_rows['Max value above '+str(Zscore_max_threshold)+'?'] = (
                   ['',''] + list_thresh_Zscores)
        extra_rows['Rewarded/non-rewarded'] =  ['',''] + list(inputs['Tank'].epocs['Analyse_this_event'].rewarded)
        extra_rows['Filename'] = ['',''] + [os.path.basename(inputs['Import location'])]*len(list_max_Zscores)
        extra_rows['Custom name'] = ['',''] + [inputs['Analysis name']]*len(list_max_Zscores)
        
        # Convert the current multi-header into a dataframe, do the same thing 
        # with the extra rows, combine them and then use that as the multi-header.
        current_rows          = results[stat].columns.to_frame()
        current_rows.index    = range(len(current_rows))
        extra_rows            = pd.DataFrame(extra_rows)
        # Make the header row here the first row in the dataframe.
        extra_rows            = pd.concat([extra_rows.columns.to_frame().T, extra_rows])
        extra_rows.index      = range(len(extra_rows))
        new_rows              = pd.concat([current_rows, extra_rows], axis=1)
        results[stat].columns = pd.MultiIndex.from_frame(new_rows)
    
    # Create dataframes with the results.
    outputs['Overall'] = results
    outputs['All pokes'] = {}
    outputs['Pellets'] = {}
    outputs['Rewarded pokes'] = {}
    outputs['Non-rewarded pokes'] = {}
    pd.options.mode.chained_assignment = None  # default='warn'
    
    for stat in ['zScore', 'dFF', 'ISOS', 'GCaMP']:
        
        # Create dataframes for only nose pokes, pellet drops, rewarded pokes 
        # and non-rewarded pokes.
        df = outputs['Overall'][stat]
        nose_poke_cols   = list(tuple(df.columns[0:3])) + [col for col in df.columns if col[2] in ['Left','Right']]
        pellet_cols      = list(tuple(df.columns[0:3])) + [col for col in df.columns if col[2] == 'Pellet']
        rewarded_cols    = list(tuple(df.columns[0:3])) + [col for col in df.columns if col[7] == 'Rewarded']
        nonrewarded_cols = list(tuple(df.columns[0:3])) + [col for col in df.columns if col[7] == 'Non-rewarded']
        outputs['All pokes'][stat]          = df[nose_poke_cols]
        outputs['Pellets'][stat]            = df[pellet_cols]
        outputs['Rewarded pokes'][stat]     = df[rewarded_cols]
        outputs['Non-rewarded pokes'][stat] = df[nonrewarded_cols]
        
        for type1 in ['Overall', 'All pokes', 'Pellets', 'Rewarded pokes', 'Non-rewarded pokes']:
            
            df = outputs[type1][stat].copy()
            
            if type1 != 'Overall':
                # Redo the mean columns, for the new shortened dataframes.
                mean_events_col = df.columns[2]
                data_cols = df.columns[3:]
                new_mean_col = df[data_cols].mean(axis=1)
                df[mean_events_col] = new_mean_col
            
            # Convert the multiindex headings to rows.
            df_cols = df.columns.to_frame().T
            df_cols.columns = range(len(df_cols.columns))
            df.columns = range(len(df.columns))
            df = pd.concat([df_cols,df])
            outputs[type1][stat] = df
            
    return(outputs)

def export_analysed_data_FED3(inputs, outputs):
    
    for data_type in ['zScore', 'dFF', 'ISOS', 'GCaMP']:
        
        if inputs['Export '+data_type] == True:

            # Export the data.
            export_name = (os.path.basename(inputs['Import location']) + "_"+data_type+"_" + 
                           inputs['Analysis name'] + "_" + inputs['Setup'].replace(' ','_') + ".xlsx")
            export_destination = os.path.join(inputs['Export location'], export_name)
            with pd.ExcelWriter(export_destination) as writer:
                for sheet_name in ['Overall', 'All pokes', 'Pellets', 'Rewarded pokes', 'Non-rewarded pokes']:
                    outputs[sheet_name][data_type].to_excel(writer, sheet_name=sheet_name, index=False, header=False)
            
def export_preview_image_FED3(inputs, outputs):

    export_name = (os.path.basename(inputs['Import location']) + "_" + 
                   inputs['Analysis name'] + "_" + inputs['Setup'].replace(' ','_') + ".png")
    export_destination = os.path.join(inputs['Export location'], export_name)
    outputs['Figure'].savefig(export_destination)
    