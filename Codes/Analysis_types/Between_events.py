from GUI_and_data_processing.Data_processing import analyse_Ethovision_data
import pandas as pd
import numpy as np
from tdt import StructType
from copy import deepcopy
import os

def find_event_names_between_events(inputs):
    
    # Define a list of all possible event names.
    if inputs['Type'] == 'Other':
        inputs['Options list'] = inputs['Other list']
    elif inputs['Type'] == 'Note':
        inputs['Options list'] = inputs['Notes list']
    elif inputs['Type'] == 'Video timestamp':
        inputs['Options list'] = inputs['Video timestamp list']
    elif inputs['Type'] == 'Ethovision':
        inputs['Options list'] = inputs['Ethovision event list']
        
    events = {}
    if inputs['Test'] == '2 bottle choice':
        event1 = 'Left lick'
        event2 = 'Right lick'
        events[event1] = ['Left lick', 'Left', 'left']
        events[event2] = ['Right lick', 'Right', 'right', 'Rght', 'RGht', 'rght']
    elif inputs['Test'] == 'Open field':
        event1 = 'Centre zone'
        event2 = 'Outer zone'
        events[event1] = ['Centre zone', 'Cntr', 'Centre', 'centre', 
                          'Center zone', 'Center', 'center']
        events[event2] = ['Outer zone', 'Outr', 'Outer', 'outer']
    elif inputs['Test'] == 'Elevated plus maze':
        event1 = ['Open arm']
        event2 = ['CLosed arm']
        events[event1] = ['Open arm', 'Open', 'open']
        events[event2] = ['Closed arm', 'Clsd', 'Closed', 'closed']
        
    # Find the element from data.epocs that matches one of the possible pellet and active poke names.
    events[event1] = list(set(events[event1]).intersection(inputs['Options list']))
    events[event2] = list(set(events[event2]).intersection(inputs['Options list']))
    
    # If we can automatically detect the event names for left poke, right poke 
    # and pellet, make those appear automatically.
    inputs['Custom'] = [event1, event2]
    inputs['Name'] = []
    if len(events[event1]) == 1:
        inputs['Name'] += [events[event1][0]]
    else:
        # inputs['Name'] += [inputs['Options list'][0]]
        inputs['Name'] += ['Event1']
    if len(events[event2]) == 1:
        inputs['Name'] += [events[event2][0]]
    else:
        # inputs['Name'] += [inputs['Options list'][0]] 
        inputs['Name'] += ['Event2']
        
    return(inputs)

def create_unique_TDT_event_between_events(inputs):
    
    # Re-organise the event data about open/closed arms, outer/centre zones, etc.
    events = {'Onsets':[], 'Notes':[]}
    event1 = inputs['Name'][0]
    event2 = inputs['Name'][1]
    
    if inputs['Type'] == 'Other':
        onsets1 = list(inputs['Tank'].epocs[event1].onset)
        onsets2 = list(inputs['Tank'].epocs[event2].onset)
        notes1  = len(onsets1)*[event1]
        notes2  = len(onsets2)*[event2]
        events['Onsets'] = onsets1 + onsets2
        events['Notes']  = notes1 + notes2
        
    elif inputs['Type'] == 'Note':
        list_notes = inputs['Tank'].epocs.Note.notes
        indices = [i for i in range(len(list_notes)) if list_notes[i] in [event1, event2]]
        events['Onsets'] = list(inputs['Tank'].epocs.Note.onset[indices])
        events['Notes']  = list(inputs['Tank'].epocs.Note.notes[indices])
        
    elif inputs['Type'] == 'Video timestamp':
        list_TS = inputs['Tank'].epocs[inputs['Camera']].notes.notes
        indices = [i for i in range(len(list_TS)) if list_TS[i] in [event1, event2]]
        events['Onsets'] = list(inputs['Tank'].epocs[inputs['Camera']].notes.ts[indices])
        events['Notes']  = list(inputs['Tank'].epocs[inputs['Camera']].notes.notes[indices])
        
    elif inputs['Type'] == 'Ethovision':        
        df_results1 = analyse_Ethovision_data(inputs, event1)
        df_results2 = analyse_Ethovision_data(inputs, event2)
        onsets1 = list(df_results1['Bout start time (secs)'])
        onsets2 = list(df_results2['Bout start time (secs)'])
        notes1 = len(onsets1)*[event1]
        notes2 = len(onsets2)*[event2]
        events['Onsets'] = onsets1 + onsets2
        events['Notes']  = notes1 + notes2
        
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
    # inputs['Tank'].epocs[SCORE_EVENT] = StructType(SCORE_DICT)
    
    # Remove the other epocs, to avoid issues with filtering in 
    # "FibPhoEpocAveraging_between_events.py".
    inputs['Tank'].epocs = StructType({'Analyse_this_event':StructType(SCORE_DICT)})
    
    return(inputs)

def create_export_data_between_events(inputs, outputs):
    
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
        
        # Add nans to each fill each column.
        def add_nans(list_of_lists):
            # Find the length of the longest list.
            max_len = max([len(list1) for list1 in list_of_lists])
            # Fill each smaller sublist with nans until they are all the same length.
            list_of_lists = [np.concatenate((list1, np.array((max_len-len(list1))*[np.nan]))) for list1 in list_of_lists]
            return(list_of_lists)
        results[stat] = add_nans(results[stat])
        
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
            list_baseline_times = []
            for col in result_arrays[stat]:
                found_baseline = False
                for i in range(len(col)):
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
        list_means = [np.nanmean(col) for col in result_arrays[stat]]
        extra_rows['Mean of column'] =  ['',''] + list_means
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
    event1 = inputs['Name'][0]
    event2 = inputs['Name'][1]
    outputs['Overall'] = results
    outputs[event1] = {}
    outputs[event2] = {}
    pd.options.mode.chained_assignment = None  # default='warn'
    
    for stat in ['zScore', 'dFF', 'ISOS', 'GCaMP']:
        
        # Create dataframes for only nose pokes, pellet drops, rewarded pokes 
        # and non-rewarded pokes.
        df = outputs['Overall'][stat]
        event1_cols = list(tuple(df.columns[0:3])) + [col for col in df.columns if col[2] == event1]
        event2_cols = list(tuple(df.columns[0:3])) + [col for col in df.columns if col[2] == event2]
        outputs[event1][stat] = df[event1_cols]
        outputs[event2][stat] = df[event2_cols]
        
        for type1 in ['Overall', event1, event2]:
            
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

def export_analysed_data_between_events(inputs, outputs):
    
    for data_type in ['zScore', 'dFF', 'ISOS', 'GCaMP']:
        
        if inputs['Export '+data_type] == True:

            # Export the data.
            event1 = inputs['Name'][0]
            event2 = inputs['Name'][1]
            export_name = (os.path.basename(inputs['Import location']) + "_"+data_type+"_" + 
                           inputs['Analysis name'] + "_" + inputs['Setup'].replace(' ','_') + ".xlsx")
            export_destination = os.path.join(inputs['Export location'], export_name)
            with pd.ExcelWriter(export_destination) as writer:
                for sheet_name in ['Overall', event1, event2]:
                    outputs[sheet_name][data_type].to_excel(writer, sheet_name=sheet_name, index=False, header=False)
            
def export_preview_image_between_events(inputs, outputs):

    export_name = (os.path.basename(inputs['Import location']) + "_" + 
                   inputs['Analysis name'] + "_" + inputs['Setup'].replace(' ','_') + ".png")
    export_destination = os.path.join(inputs['Export location'], export_name)
    outputs['Figure'].savefig(export_destination)
