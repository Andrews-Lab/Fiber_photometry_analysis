from GUI_and_data_processing.Data_processing import analyse_Ethovision_data
from tdt import read_block, StructType
from copy import deepcopy
import sys
import numpy as np
import os
import pandas as pd

def find_possible_TDT_event_names(inputs):
    
    print('\nPlease wait while the TDT tank is importing...')
    inputs['Tank'] = read_block(inputs['Import location'])
    print('')
    
    if inputs['Type'] == 'Other':
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
        inputs['Other list'] = [TTLM for TTLM in TTLM_list if TTLM not in TTLM_exclude]
        if len(inputs['Other list']) == 0:
            print('There are no other events to choose from. Try a note or video timestamp.')
            sys.exit()
            
    elif inputs['Type'] == 'Note':
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
        inputs['Notes list'] = list(inputs['Tank'].epocs.Note.notes)
        inputs['Notes list'] = list(set(inputs['Notes list'])) # Remove non-unique elements.
        
    elif inputs['Type'] == 'Video timestamp':
        if 'notes' not in inputs['Tank'].epocs[inputs['Camera']].keys():
            print('There are no video timestamps for this camera. '+
                  'Try another camera or another data type.')
            sys.exit()
        list_video_events = zip(inputs['Tank'].epocs[inputs['Camera']].notes.index, 
                                inputs['Tank'].epocs[inputs['Camera']].notes.notes)
        list_video_events = dict(sorted(list_video_events))
        list_video_events = list(list_video_events.values())
        inputs['Video timestamp list'] = list_video_events
        
    elif inputs['Type'] == 'Ethovision':
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
            print('Also ensure that "Setup A" or "Setup B" is at the start of the filename.')
            sys.exit()
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
        inputs['Ethovision event list'] = list(df.columns)
    
    return(inputs)

def define_unique_TDT_event(inputs):
    
    SCORE_EVENT = 'Analyse_this_event'
    
    if inputs['Type'] == 'Other':
        SCORE_DICT = {"name":     SCORE_EVENT,
                      "onset":    inputs['Tank'].epocs[inputs['Name']].onset,
                      "offset":   inputs['Tank'].epocs[inputs['Name']].offset,
                      "type_str": inputs['Tank'].epocs[inputs['Name']].type_str,
                      "data":     inputs['Tank'].epocs[inputs['Name']].data}
        
    elif inputs['Type'] == 'Note':
        list_notes = inputs['Tank'].epocs.Note.notes
        if inputs['Name'] == 'All':
            indices = [i for i in range(len(list_notes))]
        else:
            indices = [i for i in range(len(list_notes)) if list_notes[i] in inputs['Name']]
        SCORE_DICT = {"name":     SCORE_EVENT,
                      "onset":    inputs['Tank'].epocs.Note.onset[indices],
                      "offset":   inputs['Tank'].epocs.Note.offset[indices],
                      "type_str": inputs['Tank'].epocs.Note.type_str,
                      "data":     inputs['Tank'].epocs.Note.data[indices],
                      "notes":    inputs['Tank'].epocs.Note.notes[indices]}
            
    elif inputs['Type'] == 'Video timestamp':
        list_TS = inputs['Tank'].epocs[inputs['Camera']].notes.notes
        if inputs['Name'] == 'All':
            indices = [i for i in range(len(list_TS))]
        else:
            indices = [i for i in range(len(list_TS)) if list_TS[i] in inputs['Name']]
        onset  = inputs['Tank'].epocs[inputs['Camera']].notes.ts[indices]
        offset = np.append(onset[1:], np.inf)
        SCORE_DICT = {"name":     SCORE_EVENT,
                      "onset":    onset,
                      "offset":   offset,
                      "type_str": inputs['Tank'].epocs[inputs['Camera']].type_str,
                      "data":     inputs['Tank'].epocs[inputs['Camera']].notes.index[indices],
                      "notes":    inputs['Tank'].epocs[inputs['Camera']].notes.notes[indices]}

    elif inputs['Type'] == 'Ethovision':        
        df_results = analyse_Ethovision_data(inputs, inputs['Name'])
        SCORE_DICT = {"name":     SCORE_EVENT,
                      "onset":    np.array(df_results['Bout start time (secs)']),
                      "offset":   np.array(df_results['Bout end time (secs)']),
                      "type_str": 'epocs',
                      "data":     np.array(range(1,len(df_results['Bout start time (secs)'])+1))}
        
    inputs['Tank'].epocs[SCORE_EVENT] = StructType(SCORE_DICT)
        
    return(inputs)

def create_export_data_peri_events(inputs, outputs):
    
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
    
    for stat in results.keys():
        
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

        list_max_Zscores = [max(col) for col in result_arrays[stat]]
        extra_rows['Max values'] = ['',''] + list_max_Zscores
        # Find the time point at which the max Z-score values occur.
        list_time_Zscores = [outputs['Timestamps'][np.argmax(col)] for col in result_arrays[stat]]
        extra_rows['Time of max values'] = ['',''] + list_time_Zscores
        list_thresh_Zscores = [('Yes' if max1>Zscore_max_threshold else 'No') 
                               for max1 in list_max_Zscores]
        extra_rows['Max value above '+str(Zscore_max_threshold)+'?'] = (
                   ['',''] + list_thresh_Zscores)
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
    
    outputs['Dataframe'] = results
    return(outputs)

def export_preview_image_peri_events(inputs, outputs):
            
    export_name = (os.path.basename(inputs['Import location']) + "_" + 
                   inputs['Analysis name'] + "_" + inputs['Setup'].replace(' ','_') + ".png")
    export_destination = os.path.join(inputs['Export location'], export_name)
    outputs['Figure'].savefig(export_destination)

def export_analysed_data_peri_events(inputs, outputs):
    
    for data_type in ['zScore', 'dFF', 'ISOS', 'GCaMP']:
        
        if inputs['Export '+data_type] == True:
    
            export_name = (os.path.basename(inputs['Import location']) + "_"+data_type+"_" + 
                           inputs['Analysis name'] + "_" + inputs['Setup'].replace(' ','_') + ".csv")
            export_destination = os.path.join(inputs['Export location'], export_name)
            outputs['Dataframe'][data_type].to_csv(export_destination, index=False)
