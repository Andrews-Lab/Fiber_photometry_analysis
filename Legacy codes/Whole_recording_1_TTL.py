# # To run this file manually, uncomment this section, comment the function definition and un-indent the rest of the code.
# import_location = 'S:/MNHS-SOBS-Physiology/andrews-lab/00 Romana Stark/Photometry/Romana_Agrppreleap2-220614-124335/Agrp_A2451_B2619-220615-091605'
# export_location = 'S:/MNHS-SOBS-Physiology/andrews-lab/00 Romana Stark/Photometry/Romana_Agrppreleap2-220614-124335/Agrp_A2451_B2619-220615-091605'
# TTL = {'Type': 'Note', 'Name': ['Ghrelin A'], 'No. notes': 1, 'Custom':'Ghrelin A'}
# setup = 'Setup A'
# TRANGE = [-950, 2700]
# Remove = 4
# save_to_csv = True
# create_barcode = {'Create?': False, 'Import location': 'C:/Users/hazza/Documents/Alex videos/Tanks/Agrp_A2817_B2837-220323-120339/Setup A Raw data-Alex_FiPho-Trial     9.xlsx', 'Excel  names for behaviours': ('Novel zone(Any of Novel zone, Intruder, Transition zone / nose-point)', 'Familiar zone(Familiar zone / nose-point)', 'Intruder zone(Intruder / nose-point)'), 'Custom names for behaviours': ('Novel zone', 'Familiar zone', 'Intruder zone'), 'Colours for behaviours': ('#aec7e8', '#ffbb78', '#98df8a'), 'Find overlap': False, 'Excel  name for zone': ''}

# import numpy as np
# import_location = 'C:/Users/hazza/Desktop/Fibre photometry GUI/Alex Mouse 1'
# export_location = 'C:/Users/hazza/Desktop/Fibre photometry GUI/Alex Mouse 1'
# TTL = {'Name': 'Left_poke2022-02-22T12_42_05', 'Custom':'Left'}
# setup = '415_0_green,470_0_green'
# TRANGE = [-950, 2700]
# Remove = 100
# save_to_csv = False
# create_barcode = {'Create?': False, 'Import location': 'C:/Users/hazza/Documents/Alex videos/Tanks/Agrp_A2817_B2837-220323-120339/Setup A Raw data-Alex_FiPho-Trial     9.xlsx', 'Excel  names for behaviours': ('Novel zone(Any of Novel zone, Intruder, Transition zone / nose-point)', 'Familiar zone(Familiar zone / nose-point)', 'Intruder zone(Intruder / nose-point)'), 'Custom names for behaviours': ('Novel zone', 'Familiar zone', 'Intruder zone'), 'Colours for behaviours': ('#aec7e8', '#ffbb78', '#98df8a'), 'Find overlap': False, 'Excel  name for zone': ''}

def Whole_recording_1_TTL(import_location, export_location, TTL, setup, TRANGE, Remove, save_to_csv, create_barcode):
    
    #import the read_block function from the tdt package
    #also import other python packages we care about
    from Convert_NPM_to_TDT_data import Convert_NPM_to_TDT_data
    from tdt import read_block, download_demo_data, StructType
    import numpy as np
    import matplotlib.pyplot as plt  # standard Python plotting library
    import os
    import pandas as pd
    import sys
    
    if ',' in setup:
        data_type = 'NPM'
    else:
        data_type = 'TDT'
    
    def setups(name, letter):
        if data_type == 'NPM':
            letter = letter.split(',')
            if name == 'ISOS':
                return(letter[0])
            elif name == 'GCaMP':
                return(letter[1])
        elif data_type == 'TDT':
            dict1 = {'ISOS':  {'Setup A':'_405A', 'Setup B':'_415A'},
                     'GCaMP': {'Setup A':'_465A', 'Setup B':'_475A'}}
            return(dict1[name][letter])
    ISOS  = setups('ISOS', setup)
    GCaMP = setups('GCaMP',setup)
    
    #if you want to learn about all the available options of read_block, uncomment 
    #the print() line below:
    #print(read_block.__doc__)
    
    #declare a string that will be used as an argument for read_block
    #change this to be the full file path to your block files
    #import_location = 'C:/Users/hazza/Desktop/Photometry/NAc_3147K_2614K-211118-105517'
    
    #call read block - new variable 'data' is the full data structure
    # data = read_block(import_location)
    
    if data_type == 'NPM':
        data = Convert_NPM_to_TDT_data(import_location)
        SCORE_EVENT = TTL['Name']
        
    elif data_type == 'TDT':
        data = read_block(import_location)
    
        #what is inside data? print to list out objects of 'data'
        #print(data)
        
        #where are my demodulated streams? 
        #data.streams contains your response traces and the raw photodetector sigs
        #print(data.streams)
        
        #print out values from my YCRo stream. This is a numpy array for reference, 
        #which is a python array that you can do easy math on
        #print("data stream _475A")
        #print(data.streams._475A.data)
        
        #make some variables up here to so if they change in new recordings you won't
        #have to change everything downstream
        
        """
        Depending on the type of peri-event selected, create a new score event.
        """
        
        if TTL['Type'] == 'TTLM':
            SCORE_EVENT = TTL['Name']
        
        elif TTL['Type'] == 'Note':
            # Check that the note comments are included in data.epocs.Note.notes.
            # If not, go back to the original Notes.txt file and find the phrases in "" marks.
            # Assign these comments to data.epocs.Note.notes.
            if 'Note' in data.epocs.keys() and np.all(data.epocs.Note.notes == 'none'):
                notes_txt_path = os.path.join(import_location, 'Notes.txt')
                with open(notes_txt_path, 'r') as notes_file:
                    notes_lines = notes_file.readlines()
                def find_comment(note):
                    ind = [i for i in range(len(note)) if note[i]=='"']
                    return(note[ind[0]+1:ind[1]])
                notes_lines = [find_comment(note) for note in notes_lines if note[:5]=='Note-']
                data.epocs.Note.notes = np.array(notes_lines)
            
            if TTL['No. notes'] == 'All':
                SCORE_EVENT = 'Note'
            else:
                SCORE_EVENT = 'Comment' # This refers to individual comments.
                indices = [i for i in range(len(data.epocs.Note.notes)) if data.epocs.Note.notes[i] in TTL['Name']]
                SCORE_DICT = {"name":     SCORE_EVENT,
                              "onset":    data.epocs.Note.onset[indices],
                              "offset":   data.epocs.Note.offset[indices],
                              "type_str": data.epocs.Note.type_str,
                              "data":     data.epocs.Note.data[indices]}
                data.epocs[SCORE_EVENT] = StructType(SCORE_DICT)
        
        elif TTL['Type'] == 'Video timestamp':
            SCORE_EVENT = 'Time'
            SCORE_DICT = {"name":     SCORE_EVENT,
                          "onset":    data.epocs[TTL['Camera']].notes.ts,
                          "offset":   data.epocs[TTL['Camera']].notes.ts + .01,
                          "type_str": data.epocs[TTL['Camera']].type_str,
                          "data":     data.epocs[TTL['Camera']].notes.index}
            data.epocs[SCORE_EVENT] = StructType(SCORE_DICT)
            
        elif TTL['Type'] == 'Ethovision':
            
            # If "Setup A" or "Setup B" is included at the start of the excel file name, read it.
            excel_file = [file for file in os.listdir(import_location) if (file[-5:]=='.xlsx' and file[:7]==setup)]
            if len(excel_file) == 0:
                print('Error: check whether')
                print('- The EthoVision excel file is in '+import_location)
                print('- The start of the excel file name is "Setup A" or "Setup B"')
                sys.exit()
            import_destination = import_location+'/'+excel_file[0]
            df = pd.read_excel(import_destination, sheet_name=0)
            
            if list(df[:0])[0] == 'Number of header lines:':
                num_headers = int(list(df[:0])[1])
            rows_skip = list(range(0,num_headers-2)) + [num_headers-1]
            headings = ['Trial time', TTL['Name']]
            df = pd.read_excel(import_destination, sheet_name=0, usecols=headings, skiprows=rows_skip)
            df = df.replace('-', np.nan)
            df = df.fillna(method='ffill')
            df = df.fillna(0)
            
            # Create the bout data.
            df_results = {}
            for behaviour in [TTL['Name']]:
                df_results[behaviour+' (bout start time in secs)']     = []
                df_results[behaviour+' (bout end time in secs)']       = []
                df_results[behaviour+' (bout lengths in secs)']        = []
                df_results[behaviour+' (bout frequency)']              = [0]
            df_results['Number of transitions (sum of all frequencies)'] = [0]
            df_results['']                                               = ['']
            for behaviour in [TTL['Name']]:
                df_results[behaviour+' (sum of bout lengths in secs)'] = [0]
            df_results['Total time (sum of all bout lengths in secs)'] = [0]
                
            for i in range(len(df['Trial time'])):
                
                for behaviour in [TTL['Name']]:
        
                    # Record the start time of a new behaviour.
                    if df.at[i, behaviour] == 1 and (i==0 or df.at[i-1, behaviour]==0):
                        start_point = df.at[i,'Trial time']
                        df_results[behaviour+' (bout start time in secs)'].append(start_point)
                    
                    # Record the end time of a previous behaviour.
                    # A second 'if' is needed here, because a single 1 at the end is both
                    # the start and end time of the behaviour
                    if (df.at[i, behaviour]==1 and i==len(df['Trial time'])-1) or (df.at[i, behaviour]==0 and (i!=0 and df.at[i-1, behaviour]==1)):
                        end_point = df.at[i,'Trial time']
                        df_results[behaviour+' (bout end time in secs)'].append(end_point)
                        start_point = df_results[behaviour+' (bout start time in secs)'][-1]
                        df_results[behaviour+' (bout lengths in secs)'].append(end_point - start_point)
                        df_results[behaviour+' (bout frequency)'][0] += 1
                        df_results['Number of transitions (sum of all frequencies)'][0] += 1
                        df_results[behaviour+' (sum of bout lengths in secs)'][0] += (end_point - start_point)
                        df_results['Total time (sum of all bout lengths in secs)'][0] += (end_point - start_point)
        
            SCORE_EVENT = TTL['Name']
            SCORE_DICT = {"name":     SCORE_EVENT,
                          "onset":    np.array(df_results[SCORE_EVENT+' (bout start time in secs)']),
                          "offset":   np.array(df_results[SCORE_EVENT+' (bout end time in secs)']),
                          "type_str": 'epocs',
                          "data":     np.array(list(range(1,len(df_results[SCORE_EVENT+' (bout start time in secs)'])+1)))}
            data.epocs[SCORE_EVENT] = StructType(SCORE_DICT)
    
    # elif TTL['Type'] == 'Neurophotometrics':
        
    #     excel_files = [file for file in os.listdir(import_location) if file[-5:]=='.xlsx']
    #     import_destination = import_location+'/'+excel_files[0]
    #     df = pd.read_excel(import_destination, sheet_name=0)
        
    #     SCORE_EVENT = TTL['Name']
    #     SCORE_DICT = {"name":     SCORE_EVENT,
    #                   "onset":    np.array(df_results[SCORE_EVENT+' (bout start time in secs)']),
    #                   "offset":   np.array(df_results[SCORE_EVENT+' (bout end time in secs)']),
    #                   "type_str": 'epocs',
    #                   "data":     np.array(list(range(1,len(df_results[SCORE_EVENT+' (bout start time in secs)'])+1)))}
    #     data.epocs[SCORE_EVENT] = StructType(SCORE_DICT)
        
    if create_barcode['Create?'] == True:
        
        if TTL['Type'] == 'Ethovision':
            
            ethovision_events = list(create_barcode['Excel  names for behaviours'])
            ethovision_custom = list(create_barcode['Custom names for behaviours'])
            # # Do not use the commented line below. I have made the excel file the 
            # # same as already imported for the TTL analysis.
            # import_destination = create_barcode['Import location']
            
            df = pd.read_excel(import_destination, sheet_name=0)
            
            if list(df[:0])[0] == 'Number of header lines:':
                num_headers = int(list(df[:0])[1])
            rows_skip = list(range(0,num_headers-2)) + [num_headers-1]
            headings = ['Trial time'] + ethovision_events
            df = pd.read_excel(import_destination, sheet_name=0, usecols=headings, skiprows=rows_skip)
            df = df.replace('-', np.nan)
            df = df.fillna(method='ffill')
            df = df.fillna(0)
            
            # Create the bout data.
            df_results = {}
            for behaviour in ethovision_events:
                df_results[behaviour+' (bout start time in secs)']     = []
                df_results[behaviour+' (bout end time in secs)']       = []
                df_results[behaviour+' (bout lengths in secs)']        = []
                df_results[behaviour+' (bout frequency)']              = [0]
            df_results['Number of transitions (sum of all frequencies)'] = [0]
            df_results['']                                               = ['']
            for behaviour in ethovision_events:
                df_results[behaviour+' (sum of bout lengths in secs)'] = [0]
            df_results['Total time (sum of all bout lengths in secs)'] = [0]
                
            for i in range(len(df['Trial time'])):
                
                for behaviour in ethovision_events:
        
                    # Record the start time of a new behaviour.
                    if df.at[i, behaviour] == 1 and (i==0 or df.at[i-1, behaviour]==0):
                        start_point = df.at[i,'Trial time']
                        df_results[behaviour+' (bout start time in secs)'].append(start_point)
                    
                    # Record the end time of a previous behaviour.
                    # A second 'if' is needed here, because a single 1 at the end is both
                    # the start and end time of the behaviour
                    if (df.at[i, behaviour]==1 and i==len(df['Trial time'])-1) or (df.at[i, behaviour]==0 and (i!=0 and df.at[i-1, behaviour]==1)):
                        end_point = df.at[i,'Trial time']
                        df_results[behaviour+' (bout end time in secs)'].append(end_point)
                        start_point = df_results[behaviour+' (bout start time in secs)'][-1]
                        df_results[behaviour+' (bout lengths in secs)'].append(end_point - start_point)
                        df_results[behaviour+' (bout frequency)'][0] += 1
                        df_results['Number of transitions (sum of all frequencies)'][0] += 1
                        df_results[behaviour+' (sum of bout lengths in secs)'][0] += (end_point - start_point)
                        df_results['Total time (sum of all bout lengths in secs)'][0] += (end_point - start_point)
            
            for i in range(len(ethovision_events)):
            
                SCORE_EVENT = ethovision_events[i]
                SCORE_DICT = {"name":     ethovision_custom[i],
                              "onset":    np.array(df_results[SCORE_EVENT+' (bout start time in secs)']),
                              "offset":   np.array(df_results[SCORE_EVENT+' (bout end time in secs)']),
                              "type_str": 'epocs',
                              "data":     np.array(list(range(1,len(df_results[SCORE_EVENT+' (bout start time in secs)'])+1)))}
                data.epocs[ethovision_custom[i]] = StructType(SCORE_DICT)
    
    PELLET = SCORE_EVENT
    # GCaMP = '_475A'
    # ISOS = '_415A'
    # PELLET = 'Bplt'
    # active = 'Blft'
    #inactive = 'Rght'
    
    
    #print out values from my YCRo stream. This is a numpy array for reference, 
    #which is a python array that you can do easy math on
    #---------------------------------------------------------------------
    # for setupA un-comment these lines
    
    #print("data stream _465A")
    #print(data.streams._465A.data)
    #make some variables up here to so if they change in new recordings you won't
    #have to change everything downstream
    # GCaMP = '_465A'
    # ISOS = '_405A'
    # PELLET = 'Pelt'
    # active = 'Left'
    
    
    
    #same as print(data.streams._475A.data)
    #print(data.streams[GCaMP])
    
    #make a time array of our data
    num_samples = len(data.streams[GCaMP].data)
    time = np.linspace(1, num_samples, num_samples) / data.streams[GCaMP].fs
    # Addition by Harry 20-9-22, to account for NPM data that does not start at time 0 secs.
    time = time + data.streams[GCaMP].start_time
    
    #plot the demodulated data traces
    #this is all matplot lib stuff which you will have to learn
    #best way to learn this is to look up examples + stackoverflow
    fig1 = plt.subplots(figsize=(10,6))
    p1, = plt.plot(time,data.streams[GCaMP].data,color='goldenrod',label='GCaMP')
    p2, = plt.plot(time,data.streams[ISOS].data,color='firebrick',label='ISOS')
    plt.title('Demodulated Data Traces',fontsize=16)
    plt.legend(handles=[p1,p2],loc='lower right',bbox_to_anchor=(1.0,1.01))
    plt.autoscale(tight=True)
    #plt.show()
    
    #artefact removal
    
    
    # There is often a large artifact on the onset of LEDs turning on
    # Remove data below a set time t
    # t = 4
    t = Remove
    # Addition by Harry 20-9-22, to account for NPM data that does not start at time 0 secs.
    t = t + data.streams[GCaMP].start_time
    
    inds = np.where(time>t)
    ind = inds[0][0]
    time = time[ind:] # go from ind to final index
    data.streams[GCaMP].data = data.streams[GCaMP].data[ind:]
    data.streams[ISOS].data = data.streams[ISOS].data[ind:]
    
    # Plot again at new time range
    fig2 = plt.figure(figsize=(10, 6))
    ax1 = fig2.add_subplot(111)
    
    # Plotting the traces
    p1, = ax1.plot(time,data.streams[GCaMP].data, linewidth=2, color='green', label='GCaMP')
    p2, = ax1.plot(time,data.streams[ISOS].data, linewidth=2, color='blueviolet', label='ISOS')
    
    ax1.set_ylabel('mV')
    ax1.set_xlabel('Seconds')
    ax1.set_title('Raw Demodulated Responsed with Artifact Removed')
    ax1.legend(handles=[p1,p2],loc='upper right')
    fig2.tight_layout()
    # fig
    
    
    #downsampling data and local averaging
    
    if data_type == 'TDT':
        # Average around every Nth point and downsample Nx
        N = 10 # Average every 10 samples into 1 value
        F415 = []
        F475 = []
        
        for i in range(0, len(data.streams[GCaMP].data), N):
            F475.append(np.mean(data.streams[GCaMP].data[i:i+N-1])) # This is the moving window mean
        data.streams[GCaMP].data = F475
        
        for i in range(0, len(data.streams[ISOS].data), N):
            F415.append(np.mean(data.streams[ISOS].data[i:i+N-1]))
        data.streams[ISOS].data = F415
        
    elif data_type == 'NPM':
        N = 1
        F415 = data.streams[ISOS].data
        F475 = data.streams[GCaMP].data
    
    #decimate time array to match length of demodulated stream
    time = time[::N] # go from beginning to end of array in steps on N
    time = time[:len(data.streams[GCaMP].data)]
    
    # Detrending and dFF
    # Full trace dFF according to Lerner et al. 2015
    # http://dx.doi.org/10.1016/j.cell.2015.07.014
    # dFF using 405 fit as baseline
    
    x = np.array(data.streams[ISOS].data)
    y = np.array(data.streams[GCaMP].data)
    bls = np.polyfit(x, y, 1)
    Y_fit_all = np.multiply(bls[0], x) + bls[1]
    Y_dF_all = y - Y_fit_all
    
    dFF = np.multiply(100, np.divide(Y_dF_all, Y_fit_all))
    std_dFF = np.std(dFF)
    
    
    # organising nose pokes and pellets during recording
    
    
    # First make a continous time series of TTL events (epocs) and plot
    PELLET_ON = data.epocs[PELLET].onset
    PELLET_OFF = data.epocs[PELLET].offset
    # Add the first and last time stamps to make tails on the TTL stream
    PELLET_x = np.append(np.append(time[0], np.reshape(np.kron([PELLET_ON, PELLET_OFF],
                       np.array([[1], [1]])).T, [1,-1])[0]), time[-1])
    sz = len(PELLET_ON)
    d = data.epocs[PELLET].data
    # Add zeros to beginning and end of 0,1 value array to match len of PELLET_x
    PELLET_y = np.append(np.append(0,np.reshape(np.vstack([np.zeros(sz),
        d, d, np.zeros(sz)]).T, [1, -1])[0]),0)
    
    y_scale = (max(dFF)-min(dFF))*0.74 #adjust according to data needs
    y_shift = min(dFF)*0.82 #scale and shift are just for asthetics
    # y_scale = 15 #adjust according to data needs
    # y_shift = -6 #scale and shift are just for asthetics
    
    # First subplot in a series: dFF with lick epocs
    fig3 = plt.figure(figsize=(20,12))
    ax2 = fig3.add_subplot(311)
    
    p1, = ax2.plot(time, dFF, linewidth=2, color='green', label='GCaMP')
    p2, = ax2.plot(PELLET_x, y_scale*PELLET_y+y_shift, linewidth=2, color='dodgerblue', label='Pellet')
    #p3, = ax2.plot(active_x, y_scale*active_y+y_shift, linewidth=2, color='red', label='Poke')
    ax2.set_ylabel(r'$\Delta$F/F')
    ax2.set_xlabel('Seconds')
    ax2.set_title('dopamine response')
    ax2.legend(handles=[p1,p2], loc='upper left')
    fig3.tight_layout()
    #plt.show()
    
    # # Lick Bout Logic
    # Now combine lick epocs that happen in close succession to make a single on/off event (a lick BOUT). Top view logic: if difference between consecutive lick onsets is below a certain time threshold and there was more than one lick in a row, then consider it as one bout, otherwise it is its own bout. Also, make sure a minimum number of licks was reached to call it a bout.
    
    PELLET_EVENT = 'PELLET_EVENT'
    
    PELLET_DICT = {
            "name":PELLET_EVENT,
            "onset":[],
            "offset":[],
            "data":[]
            }
    
    #print(PELLET_DICT)
    #pass StructType our new dictionary to make keys and values
    data.epocs.PELLET_EVENT = StructType(PELLET_DICT)
    
    pellet_on_diff = np.diff(data.epocs[PELLET].onset)
    BOUT_TIME_THRESHOLD = 1
    pellet_diff_ind = np.where(pellet_on_diff >= BOUT_TIME_THRESHOLD)[0]
    #for some reason np.where returns a 2D array, hence the [0]
    
    # Make an onset/ offset array based on threshold indicies
    diff_ind = 0
    for ind in pellet_diff_ind: 
        # BOUT onset is thresholded onset index of lick epoc event
        data.epocs[PELLET_EVENT].onset.append(data.epocs[PELLET].onset[diff_ind])
        # BOUT offset is thresholded offset of lick event before next onset
        data.epocs[PELLET_EVENT].offset.append(data.epocs[PELLET].offset[ind])
        # set the values for data, arbitrary 1
        data.epocs[PELLET_EVENT].data.append(1)
        diff_ind = ind + 1
    
    # special case for last event to handle lick event offset indexing
    # The following 3 lines were added in by Harry - 17-6-22.
    if len(pellet_diff_ind) == 0:
        data.epocs[PELLET_EVENT].onset.append(data.epocs[PELLET].onset[0])
    else:
        data.epocs[PELLET_EVENT].onset.append(data.epocs[PELLET].onset[pellet_diff_ind[-1]+1])
    data.epocs[PELLET_EVENT].offset.append(data.epocs[PELLET].offset[-1])
    data.epocs[PELLET_EVENT].data.append(1)
    
    # Now determine if it was a 'real' lick bout by thresholding by some
    # user-set number of licks in a row
    MIN_PELLET_THRESH = 1 #four licks or more make a bout
    pellet_array = []
    
    # Find number of licks in pellet_array between onset and offset of 
    # our new lick BOUT PELLET_EVENT
    for on, off in zip(data.epocs[PELLET_EVENT].onset,data.epocs[PELLET_EVENT].offset):
        pellet_array.append(
            len(np.where((data.epocs[PELLET].onset >= on) & (data.epocs[PELLET].onset <= off))[0]))
    
    # Remove onsets, offsets, and data of thrown out events
    pellet_array = np.array(pellet_array)
    inds = np.where(pellet_array<MIN_PELLET_THRESH)[0]
    for index in sorted(inds, reverse=True):
        del data.epocs[PELLET_EVENT].onset[index]
        del data.epocs[PELLET_EVENT].offset[index]
        del data.epocs[PELLET_EVENT].data[index]
        
    # Make a continuous time series for lick BOUTS for plotting
    PELLET_EVENT_on = data.epocs[PELLET_EVENT].onset
    PELLET_EVENT_off = data.epocs[PELLET_EVENT].offset
    PELLET_EVENT_x = np.append(time[0], np.append(
        np.reshape(np.kron([PELLET_EVENT_on, PELLET_EVENT_off],np.array([[1], [1]])).T, [1,-1])[0], time[-1]))
    sz = len(PELLET_EVENT_on)
    d = data.epocs[PELLET_EVENT].data
    PELLET_EVENT_y = np.append(np.append(
        0, np.reshape(np.vstack([np.zeros(sz), d, d, np.zeros(sz)]).T, [1 ,-1])[0]), 0)
    
    # START OF COMMENTED OUT SECTION.
    # #--------------------------
    # #active = 'Left'
    # #inactive = 'Rght'
    
    
    # # First make a continous time series of active TTL events (epocs) and plot
    # active_on = data.epocs[active].onset
    # active_off = data.epocs[active].offset
    # # Add the first and last time stamps to make tails on the TTL stream
    # active_x = np.append(np.append(time[0], np.reshape(np.kron([active_on, active_off],
    #                    np.array([[1], [1]])).T, [1,-1])[0]), time[-1])
    # sz = len(active_on)
    # d = data.epocs[active].data
    # # Add zeros to beginning and end of 0,1 value array to match len of active_x
    # active_y = np.append(np.append(0,np.reshape(np.vstack([np.zeros(sz),
    #     d, d, np.zeros(sz)]).T, [1, -1])[0]),0)
    
    # # # Lick Bout Logic
    # # Now combine lick epocs that happen in close succession to make a single on/off event (a lick BOUT). Top view logic: if difference between consecutive lick onsets is below a certain time threshold and there was more than one lick in a row, then consider it as one bout, otherwise it is its own bout. Also, make sure a minimum number of licks was reached to call it a bout.
    
    # active_EVENT = 'active_EVENT'
    
    # active_DICT = {
    #         "name":active_EVENT,
    #         "onset":[],
    #         "offset":[],
    #         "type_str":data.epocs[active].type_str,
    #         "data":[]
    #         }
    
    # print(active_DICT)
    # #pass StructType our new dictionary to make keys and values
    # data.epocs.active_EVENT = StructType(active_DICT)
    
    # active_on_diff = np.diff(data.epocs[active].onset)
    # BOUT_TIME_THRESHOLD = 1
    # active_diff_ind = np.where(active_on_diff >= BOUT_TIME_THRESHOLD)[0]
    # #for some reason np.where returns a 2D array, hence the [0]
    
    # # Make an onset/ offset array based on threshold indicies
    # diff_ind = 0
    # for ind in active_diff_ind: 
    #     # BOUT onset is thresholded onset index of lick epoc event
    #     data.epocs[active_EVENT].onset.append(data.epocs[active].onset[diff_ind])
    #     # BOUT offset is thresholded offset of lick event before next onset
    #     data.epocs[active_EVENT].offset.append(data.epocs[active].offset[ind])
    #     # set the values for data, arbitrary 1
    #     data.epocs[active_EVENT].data.append(1)
    #     diff_ind = ind + 1
    
    # # special case for last event to handle lick event offset indexing
    # data.epocs[active_EVENT].onset.append(data.epocs[active].onset[active_diff_ind[-1]+1])
    # data.epocs[active_EVENT].offset.append(data.epocs[active].offset[-1])
    # data.epocs[active_EVENT].data.append(1)
    
    # # Now determine if it was a 'real' lick bout by thresholding by some
    # # user-set number of licks in a row
    # MIN_active_THRESH = 1 #four licks or more make a bout
    # actives_array = []
    
    # # Find number of licks in pellet_array between onset and offset of 
    # # our new lick BOUT PELLET_EVENT
    # for on, off in zip(data.epocs[active_EVENT].onset,data.epocs[active_EVENT].offset):
    #     actives_array.append(
    #         len(np.where((data.epocs[active].onset >= on) & (data.epocs[active].onset <= off))[0]))
    
    # # Remove onsets, offsets, and data of thrown out events
    # actives_array = np.array(actives_array)
    # inds = np.where(actives_array<MIN_active_THRESH)[0]
    # for index in sorted(inds, reverse=True):
    #     del data.epocs[active_EVENT].onset[index]
    #     del data.epocs[active_EVENT].offset[index]
    #     del data.epocs[active_EVENT].data[index]
        
    # # Make a continuous time series for lick BOUTS for plotting
    # active_EVENT_on = data.epocs[active_EVENT].onset
    # active_EVENT_off = data.epocs[active_EVENT].offset
    # active_EVENT_x = np.append(time[0], np.append(
    #     np.reshape(np.kron([active_EVENT_on, active_EVENT_off],np.array([[1], [1]])).T, [1,-1])[0], time[-1]))
    # sz = len(active_EVENT_on)
    # d = data.epocs[active_EVENT].data
    # active_EVENT_y = np.append(np.append(
    #     0, np.reshape(np.vstack([np.zeros(sz), d, d, np.zeros(sz)]).T, [1 ,-1])[0]), 0)
    # #-------------------------
    # # # Plot dFF with newly defined lick bouts
    # END OF COMMENTED OUT SECTION.
    
    ax3 = fig3.add_subplot(312)
    p1, = ax3.plot(time, dFF, linewidth=2, color='green', label='465 nm')
    p2, = ax3.plot(PELLET_EVENT_x, y_scale*PELLET_EVENT_y+y_shift, linewidth=2, color='magenta', label='Pellet')
    #p3, = ax3.plot(active_EVENT_x, y_scale*active_EVENT_y+y_shift, linewidth=1, color='blue', label='Poke')
    ax3.set_ylabel(r'$\Delta$F/F')
    ax3.set_xlabel('Seconds')
    ax3.set_title('PR session')
    #ax3.legend(handles=[p1, p2, p3], loc='upper left')
    ax3.legend(handles=[p1, p2], loc='upper left')
    fig3.tight_layout()
    fig3
    #plt.show()
    
    #-------------------------------------------------------------------------#
    # Overlay the barcode.
    #-------------------------------------------------------------------------#
    
    # # Make nice area fills instead of epocs for asthetics
    
    ax4 = fig3.add_subplot(313)
    p1, = ax4.plot(time, dFF,linewidth=2, color='green', label='dopamine')
    for on, off in zip(data.epocs[PELLET_EVENT].onset, data.epocs[PELLET_EVENT].offset):
        ax4.axvspan(on, off, alpha=0.25, color='dodgerblue')  
    ax4.set_ylabel(r'$\Delta$F/F')
    ax4.set_xlabel('Seconds')
    ax4.set_title(' ')
    fig3.tight_layout()
    fig3
    
    #plt.show()
    
    if create_barcode["Create?"] == True:
    
        # Create another plot with only the trace + barcodes.
        
        fig_export = plt.figure(figsize=(20,4))
        ax_export = fig_export.add_subplot()
        p1, = ax_export.plot(time, dFF, linewidth=2, color='green', label='dopamine')
        for i in range(len(ethovision_custom)):
            for on, off in zip(data.epocs[ethovision_custom[i]].onset, data.epocs[ethovision_custom[i]].offset):
                ax_export.axvspan(on, off, alpha=1, color=create_barcode['Colours for behaviours'][i], label=ethovision_custom[i])
        
        # Create a legend.
        handles, labels = ax_export.get_legend_handles_labels()
        dict1 = {}
        for i in range(len(handles)):
            dict1[labels[i]] = handles[i]
        ax_export.legend(dict1.values(), dict1.keys(), loc='upper left')
        
        # Choose the time axis range to be the earliest start event and the latest end event.
        
        
        ax_export.set_ylabel(r'$\Delta$F/F')
        ax_export.set_xlabel('Seconds')
        ax_export.set_title(r'$\Delta$F/F with barcode overlayed')
        fig_export.tight_layout()
        
        #plt.show()
        
    #-------------------------------------------------------------------------#
    
    # # Time Filter Around Pellet Epocs
    # Note that we are using dFF of the full time series, not peri-event dFF where f0 is taken from a pre-event basaeline period.
    
    # Convert find the pre-time and post-time values based on the TRANGE.
    PRE_TIME         = -TRANGE[0]
    POST_TIME        = TRANGE[1] + TRANGE[0]
    #PRE_TIME = 5 # five seconds before event onset
    #POST_TIME = 10 # ten seconds after
    fs = data.streams[GCaMP].fs/N #recall we downsampled by N = 10 earlier
    
    ### NOTE THIS IS A DIFFERENT TRANGE DEFINTION TO THE PERI-EVENT CODES. ###
    ### [time before event, time after event] instead of [time before event, window duration] ###
    # time span for peri-event filtering, PRE and POST, in samples
    TRANGE = [-PRE_TIME*np.floor(fs), POST_TIME*np.floor(fs)]
    
    dFF_snips = []
    array_ind = []
    pre_stim = []
    post_stim = []
    
    for on in data.epocs[PELLET_EVENT].onset:
        # If the bout cannot include pre-time seconds before event, make zero
        if on < PRE_TIME:
            #dFF_snips.append(np.zeros(TRANGE[1]-TRANGE[0]))
            # Edit by Harry 23-5-22.
            dFF_snips.append(np.zeros(int(TRANGE[1])-int(TRANGE[0])))
        else: 
            # find first time index after bout onset
            array_ind.append(np.where(time > on)[0][0])
            # find index corresponding to pre and post stim durations
            pre_stim.append(array_ind[-1] + TRANGE[0])
            post_stim.append(array_ind[-1] + TRANGE[1])
            dFF_snips.append(dFF[int(pre_stim[-1]):int(post_stim[-1])])
            
    # Make all snippets the same size based on min snippet length
    min1 = np.min([np.size(x) for x in dFF_snips])
    dFF_snips = [x[1:min1] for x in dFF_snips]
    
    mean_dFF_snips = np.mean(dFF_snips, axis=0)
    std_dFF_snips = np.std(mean_dFF_snips, axis=0)
    
    peri_time = np.linspace(1, len(mean_dFF_snips), len(mean_dFF_snips))/fs - PRE_TIME
    
    
    # # Make a Peri-Event Stimulus Plot and Heat Map
    
    fig4 = plt.figure(figsize=(6,10))
    ax5 = fig4.add_subplot(211)
    
    for snip in dFF_snips:
        p1, = ax5.plot(peri_time, snip, linewidth=.5, color=[.7, .7, .7], label='Individual Trials')
    p2, = ax5.plot(peri_time, mean_dFF_snips, linewidth=2, color='green', label='Mean Response')
    
    # Plotting standard error bands
    p3 = ax5.fill_between(peri_time, mean_dFF_snips+std_dFF_snips, 
                          mean_dFF_snips-std_dFF_snips, facecolor='green', alpha=0.2)
    p4 = ax5.axvline(x=0, linewidth=3, color='slategray', label='Lick Bout Onset')
    
    ax5.axis('tight')
    ax5.set_xlabel('Seconds')
    ax5.set_ylabel(r'$\Delta$F/F')
    ax5.set_title('Peri-Event Pellet retrieval')
    ax5.legend(handles=[p1, p2, p4], bbox_to_anchor=(1.1, 1.05))
    
    ax6 = fig4.add_subplot(212)
    cs = ax6.imshow(dFF_snips, cmap=plt.cm.Greys,
                    interpolation='none', extent=[-PRE_TIME,POST_TIME,len(dFF_snips),0],)
    ax6.set_ylabel('Trial Number')
    ax6.set_yticks(np.arange(.5, len(dFF_snips), 2))
    ax6.set_yticklabels(np.arange(0, len(dFF_snips), 2))
    fig4.colorbar(cs)
    fig4
    
    #plt.show()
    
    
    #save list of timing for pellet and nose pokes
    import os
    import numpy as np
    import pylab
    import itertools, collections
    import matplotlib.pyplot as plt
    import pandas as pd
    
    
    'OUTPUT'
    filename = import_location
    
    #save_to_csv_Pellettiming = False
        # If true then input the file name
    filename_Pellettiming = os.path.join (
        #os.path.dirname (filename), # saves file in same directory
        export_location, # change the path to where you want files saved
        os.path.basename(filename) + "_whole_recording_" + TTL['Custom'] + '_' + setup.replace(' ','_') + ".csv")
    
    if create_barcode['Create?'] == True:
        fig_export.savefig(filename_Pellettiming[:-4]+'.png') # Save the figure in the export location.
    
    if save_to_csv == True: 
    # if save_to_csv_Pellettiming:
        np.savetxt(filename_Pellettiming, PELLET_ON, delimiter=",")
        #print("Printed TTL1 whole recording CSV")
        
    plt.close()
                