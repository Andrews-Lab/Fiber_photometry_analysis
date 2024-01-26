import pandas as pd
import numpy as np
import sys
import os
from copy import deepcopy
import cv2 as cv
import matplotlib
matplotlib.use('agg')
# plt = matplotlib.pyplot
import matplotlib.pyplot as plt
from tqdm import tqdm
from tdt import read_block
from GUI_and_data_processing.Custom_TDT_file_reader import find_start_and_end_times

def check_package_versions():

    if matplotlib.__version__ < '3.5.0':
        print("Your matplotlib version is less than 3.5.0,\n"+
              "which means you might run into some errors.\n"+
              "Please update this by running:\n"+
              "pip install --upgrade matplotlib")  

def create_values_col(df):
    
    # Convert value entries to lists, so they can be combined together.
    value_cols = df.columns[1:]
    def enlist(value):
        return([value])
    for col in value_cols:
        df[col] = df[col].apply(enlist)
    # Create a column with all value columns combined into a list.
    df['Values'] = df[value_cols[0]]
    rest_value_cols = value_cols[1:]
    for col in rest_value_cols:
        df['Values'] = df['Values'] + df[col]
    # Remove nans from these lists.
    def remove_nans(list1):
        list1 = [val for val in list1 if pd.isna(val)==False]
        return(list1)
    df['Values'] = df['Values'].apply(remove_nans)
    # Convert keep the multi-item lists as lists and convert the single-item
    # lists to the item.
    def deenlist(list1):
        if len(list1) == 1:
            return(list1[0])
        else:
            return(list1)
    df['Values'] = df['Values'].apply(deenlist)
    
    return(df['Values'])

def setups(name, letter):
    dict1 = {'ISOS':  {'Setup A':'_405A', 'Setup B':'_415A'},
             'GCaMP': {'Setup A':'_465A', 'Setup B':'_475A'}}
    return(dict1[name][letter])

def import_settings_excel_file(inputs):
    
    df = pd.read_excel(inputs["Import settings"], header=None)
    # df = pd.read_excel(path, header=None)
    
    # Change the dataframe into 2 columns with options and values as columns.
    df['Values']  = create_values_col(df)
    df['Options'] = df[0]
    df = df[['Options', 'Values']]
    
    # Split the dataframes by spaces in the options column.
    df_list = np.split(df, df[pd.isna(df['Options'])].index)
    
    # Convert these dataframes to dictionaries.
    dict_list = []
    for df1 in df_list:
        dict1 = {}
        for i in df1.index:
            # Exclude the nan rows used to separate the dataframes.
            if pd.isna(df1.at[i,'Options']) == False:
                option = df1.at[i,'Options']
                value  = df1.at[i,'Values']
                # Replace 'inf' with np.inf for the artifact rejection level section.
                if option == 'Artifact RL' and value == 'inf':
                    value = np.inf
                # Add this option, value pair to the dictionary.
                dict1[option] = value
        if dict1['Setup'] != 'Custom' and ('ISOS' not in dict1.keys() or 'GCaMP' not in dict1.keys()):
            # Work out the ISOS and GCaMP values for 'Setup A' and 'Setup B'.
            # These are needed for the FibPhoEpocAveraging code.
            dict1['ISOS']  = setups('ISOS',  dict1['Setup'])
            dict1['GCaMP'] = setups('GCaMP', dict1['Setup'])
        if dict1['Analysis'] == 'Between events' and 'Baseline type' not in dict1.keys():
            dict1['Baseline type'] = 'Whole recording'
            dict1['Baseline period'] = [0,10] # This is arbitrary as whole recording is the default.
        if 'Baseline type' not in dict1.keys():
            dict1['Baseline type'] = 'Specific'
        # Enter the sampling rate.
        dict1["N"] = 100
        dict_list += [dict1]
    
    # Some tanks in list_tanks indicate to import subfolders.
    # In these cases, create many of these tanks and only change the import and export locations.
    # Add these to list_tanks_full.
    list_tanks_full = []
    
    for inputs in dict_list:
        
        if inputs["Import subfolders"] == False:
            list_tanks_full += [deepcopy(inputs)]
            continue
        import_loc = inputs["Import location"]
        export_loc = inputs["Export location"]
        for folder in os.listdir(import_loc):
            import_new = os.path.join(import_loc, folder)
            export_new = os.path.join(export_loc, folder)
            if os.path.isfile(import_new) == True:
                continue
            list_tanks_full += [deepcopy(inputs)]
            list_tanks_full[-1]['Import location'] = import_new
            list_tanks_full[-1]['Export location'] = export_new
            
    print('Start analysing each tank within the settings excel file.')
            
    return(list_tanks_full)

def find_1dp_without_rounding(num):
    decimal           = num - int(num)
    decimal_truncated = float(str(decimal)[:3])
    num_1dp           = int(num) + decimal_truncated
    return(num_1dp)

def import_tank(inputs):
    
    # Import the tank, so the stream names can be checked.
    print('\nPlease wait while the TDT tank is importing...')
    try:
        inputs['Tank'] = read_block(inputs['Import location'])
    except ValueError:
        # Sometimes users will get the error
        # Warning: Block end marker not found, block did not end cleanly. Try 
        # setting T2 smaller if errors occur
        # Use a custom TDT file reader code to find the proper start and end times
        # and re-import the tank using these numbers.
        start, end = find_start_and_end_times(inputs['Import location'])
        start, end = find_1dp_without_rounding(start), find_1dp_without_rounding(end)
        print(f'Re-importing the tank data from {start} secs to {end} secs')
        inputs['Tank'] = read_block(inputs['Import location'], t1=start, t2=end)
        
    print('')
    
    return(inputs)

def analyse_Ethovision_data(inputs, event_name):
    
    # If "Setup A" or "Setup B" is included at the start of the excel file name, read it.
    excel_file = [file for file in os.listdir(inputs['Import location']) 
                  if (file.endswith('.xlsx') and file.startswith(inputs['Setup']))]
    if len(excel_file) == 0:
        print('Error: check whether')
        print('- The Ethovision excel file is in '+inputs['Import location'])
        print('- The start of the excel file name is "Setup A", "Setup B" or "Custom"')
        sys.exit()
    import_destination = os.path.join(inputs['Import location'], excel_file[0])
    df = pd.read_excel(import_destination, sheet_name=0)
    
    if list(df[:0])[0] == 'Number of header lines:':
        num_headers = int(list(df[:0])[1])
    rows_skip = list(range(0,num_headers-2)) + [num_headers-1]
    headings = ['Trial time', event_name]
    df = pd.read_excel(import_destination, sheet_name=0, usecols=headings, skiprows=rows_skip)
    df = df.replace('-', np.nan)
    df = df.fillna(method='ffill')
    df = df.fillna(0)
    
    # Create the bout data.
    df_results = {}
    behaviour = event_name
    df_results['Bout start time (secs)'] = []
    df_results['Bout end time (secs)']   = []
        
    for i in range(len(df['Trial time'])):

        # Record the start time of a new behaviour.
        if df.at[i, behaviour] == 1 and (i==0 or df.at[i-1, behaviour]==0):
            start_point = df.at[i,'Trial time']
            df_results['Bout start time (secs)'].append(start_point)
        
        # Record the end time of a previous behaviour.
        # A second 'if' is needed here, because a single 1 at the end is both
        # the start and end time of the behaviour
        if ((df.at[i, behaviour]==1 and i==len(df['Trial time'])-1) or 
            (df.at[i, behaviour]==0 and (i!=0 and df.at[i-1, behaviour]==1))):
            end_point = df.at[i,'Trial time']
            df_results['Bout end time (secs)'].append(end_point)
            
    return(df_results)

def check_for_excluded_data(inputs, outputs):
    
    # If the t-range for some events extend beyond the length of the recording,
    # then those events will be excluded after running epoc_filter...
    
    # Check whether a t-range extended beyond the recording.
    event_onsets = inputs['Tank'].epocs['Analyse_this_event'].onset
    if len(event_onsets) != len(outputs['zScore']):
        print('\nPLEASE NOTE: '+str(len(event_onsets)-len(outputs['zScore'])) + ' events '+
              'have been excluded, because their window durations go outside '+
              'the recording.\n')

        # Create a list of epoch intervals for each event onset.
        event_intervals = []
        for onset in event_onsets:
            left_bound  = onset + inputs['t-range'][0]
            right_bound = onset + (inputs['t-range'][1] + inputs['t-range'][0])
            event_intervals += [(left_bound, right_bound)]
            
        # Create an interval for the entire experiment duration.
        start_time   = inputs['Tank'].streams[inputs['GCaMP']].start_time
        exp_length   = inputs['Tank'].info.duration.total_seconds()
        exp_interval = (start_time, start_time + exp_length)
        
        # Find which event indices are excluded for extending beyond the recording
        # length.
        nonexcluded_events = [i for i in range(len(event_intervals))
                              if event_intervals[i][0] >= exp_interval[0] and
                                 event_intervals[i][1] <= exp_interval[1]]
        
        # Change the inputs['Tank'].epocs['Analyse_this_event'] info to exclude 
        # these event indices.
        possible_data_types = ['onset', 'offset', 'data', 'notes', 'rewarded']
        event_keys = set(inputs['Tank'].epocs['Analyse_this_event'].keys())
        data_types = event_keys.intersection(possible_data_types)
        for data_type in data_types:
            original_data = inputs['Tank'].epocs['Analyse_this_event'][data_type]
            inputs['Tank'].epocs['Analyse_this_event'][data_type] = (
                original_data[nonexcluded_events])
        
    return(inputs)

def create_annotated_video(inputs, outputs):
    
    # Create a variable for the signal.
    signal = outputs[inputs['Snippets signal']]

    # Import the video by looking for "Cam1" or "Cam2".
    video_file = [file for file in os.listdir(inputs['Import location']) if 
                  (inputs['Camera'] in file)]
    if len(video_file) == 0:
        print('Error: check whether')
        print('- The video file is in '+inputs['Import location'])
        print('- The video filename contains "Cam1" or "Cam2"')
        sys.exit()
    import_destination = os.path.join(inputs['Import location'],video_file[0])

    print('Creating video snippets for '+str(len(signal))+' epochs.')
    
    # Create a folder in the export location with the snipped videos.
    folder_name = 'Video snippets0'
    i = 1
    while folder_name in os.listdir(inputs['Export location']):
        folder_name = folder_name[:-1] + str(i)
        i += 1
    export_path = os.path.join(inputs['Export location'], folder_name)
    os.makedirs(export_path)
    
    for i in tqdm(range(len(signal)), ncols=70):

        warning = "!!! Failed cap.read()"
        cap = cv.VideoCapture(import_destination)
        
        fps = cap.get(cv.CAP_PROP_FPS)
        start = inputs['Tank'].time_ranges[0][i]
        end   = inputs['Tank'].time_ranges[1][i]
        # If a custom time window is selected, use that instead of the t-range.
        # Find the difference between the t-range and the custom window size
        # parameters, and add that to 'start' and 'end'.
        if inputs['Snippets window'] == 'Custom':
            start += inputs['Snippets window size'][0] - inputs['t-range'][0]
            end   += inputs['Snippets window size'][1] - inputs['t-range'][1]
        # Convert seconds to frames.
        start = int(start*fps)
        end   = int(end*fps)
        # If the start/end time of the window is before the start of the video,
        # make that time the start of the video.
        if start < 0:
            start = 0
        if end < 0:
            end = 0
    
        video_width   = int(cap.get(cv.CAP_PROP_FRAME_WIDTH))
        video_height  = int(cap.get(cv.CAP_PROP_FRAME_HEIGHT))
        border_top    = int(video_width*0.02)
        border_bottom = int(video_width*0.3)
        border_left   = int(video_width*0.02)
        border_right  = int(video_width*0.02)
        graph_height  = border_bottom - border_top*2
        graph_width   = video_width
        final_width   = border_left + video_width + border_right
        final_height  = border_top + video_height + border_bottom
        graph_x1 = border_left
        graph_x2 = border_left + video_width
        graph_y1 = border_top  + video_height + border_top
        graph_y2 = border_top  + video_height + border_top + graph_height
        
        # Create a list of the time stamps from TDT and Z-scores.
        x1 = outputs['Timestamps']
        y1 = signal[i]
        df1 = pd.DataFrame({'x':x1,'y':y1,'Type':len(x1)*['TDT']})
        # Create a list of the time stamps from the video frames and Z-scores.
        x2 = list(np.arange(inputs['t-range'][0]+1/fps, inputs['t-range'][0]+inputs['t-range'][1]+1/fps, 1/fps))
        y2 = len(x2)*[np.nan]
        df2 = pd.DataFrame({'x':x2,'y':y2,'Type':len(x2)*['Video']})
        # Combine the 2 datasets into a dataframe, sort by time and forward-fill the nans.
        df = pd.concat([df1, df2])
        df = df.sort_values('x')
        df.index = range(len(df))
        df['y'] = df['y'].fillna(method='ffill').fillna(method='bfill')
        # Extract the time and Z-score data for the video frames.
        df = df[df['Type'] == 'Video']
        x_data = df['x'].to_list()
        y_data = df['y'].to_list()
    
        plt.rcParams.update({'font.size': 10})
        fig = plt.figure(figsize=(graph_width/72, graph_height/72), dpi=72)
        # plt.subplots_adjust(left=0.3)
        # plt.subplots_adjust(bottom=0.3)
        plt.xlim(inputs['t-range'][0],inputs['t-range'][0]+inputs['t-range'][1])
        y_min = min([min(signal[i]) for i in range(len(signal))])
        y_max = max([max(signal[i]) for i in range(len(signal))])
        plt.ylim(y_min,y_max)
        plt.xlabel('Time (secs)')
        plt.ylabel(inputs['Snippets signal'])
        
        plt.tight_layout(h_pad=0)
        plt.subplots_adjust(left=0.07)
        line1, = plt.plot([], [], 'g-', lw=1.5)
        plt.axvline(x = 0, color = 'lightgray', linestyle='dashed')
    
        cap.set(cv.CAP_PROP_POS_FRAMES, start)
        result = cv.VideoWriter(os.path.join(export_path, 'Event '+str(i)+'.mp4'), 
                                cv.VideoWriter_fourcc(*'mp4v'),
                                fps, (final_width, final_height))
    
        while cap.isOpened():
            
            frame_no = int(cap.get(cv.CAP_PROP_POS_FRAMES))
            ret, frame = cap.read()
            frame = cv.copyMakeBorder(frame,
                                      border_top,border_bottom,border_left,border_right,
                                      cv.BORDER_CONSTANT,value=[0,0,0])
            
            # update data
            line1.set_data(x_data[:frame_no-start], y_data[:frame_no-start])
            # redraw the canvas
            fig.canvas.draw()
            # convert canvas to image
            img = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
            img = img.reshape(fig.canvas.get_width_height()[::-1] + (3,))
            # img is rgb, convert to opencv's default bgr
            img = cv.cvtColor(img,cv.COLOR_RGB2BGR)
            img = cv.resize(img, (graph_width, graph_height))
            frame[ graph_y1:graph_y2, graph_x1:graph_x2 ] = img
            
            if ret == False:
                print(warning)
                break
            result.write(frame)
            
            if int(cap.get(cv.CAP_PROP_POS_FRAMES)) == end:
                break
                
        # When everything done, release the video capture and write objects
        cap.release()
        result.release()
            
        # Closes all the frames
        cv.destroyAllWindows()
        
        plt.close()
        
def export_settings_excel_file(inputs):
    
    # Create a list of inputs that are not needed to re-create the analysis.
    inputs_exclude = ['Settings', 'N', 'Tank', 'Other list', 
                      'Notes list', 'Video timestamp list', 'Ethovision event list', 
                      'Options list', 'Options', 'Event']
    if inputs['Analysis'] == 'Whole recording':
        inputs_exclude += ['t-range', 'Baseline period', 'Create snippets']
    if inputs['Analysis'] == 'Between events':
        inputs_exclude += ['Create snippets']
    # if inputs['Setup'] == 'Custom':
    #     inputs_exclude += ['Setup']
    # if inputs['Setup'] in ['Setup A', 'Setup B']:
    #     inputs_exclude += ['ISOS', 'GCaMP']
    
    # Remove these inputs.
    export = deepcopy(inputs)
    for input1 in inputs_exclude:
        if input1 in export.keys():
            export.pop(input1)
        
    # Convert every input into a list.
    for key in export.keys():
        if type(export[key]) != list:
            export[key] = [export[key]]
            
    # Make all lists the same length.
    max_len = max([len(export[key]) for key in export.keys()])
    for key in export.keys():
        cur_len = len(export[key])
        export[key] = export[key] + (max_len - cur_len)*[np.nan]
    
    # Convert this to a dataframe.
    export = pd.DataFrame(export).T
    
    # Expor this datarame.
    # If a settings file already exists, create a new one with a number at the end.
    export_name = 'Settings0.xlsx'
    i = 1
    while export_name in os.listdir(inputs['Export location']):
        export_name = export_name[:8] + str(i) + '.xlsx'
        i += 1
    export_path = os.path.join(inputs['Export location'], export_name)
    export.to_excel(export_path, header=False)
    print('Saved ' + export_name)
