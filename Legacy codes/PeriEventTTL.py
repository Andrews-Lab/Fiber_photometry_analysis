# # To run this file manually, uncomment this section, comment the function definition and un-indent the rest of the code.
# import numpy as np
# import_location = 'C:/Users/hazza/Desktop/Fibre photometry GUI/Photometry tanks/A_AgrpCre2837-220222-122913'
# export_location = 'C:/Users/hazza/Desktop/Fibre photometry GUI/Photometry tanks/A_AgrpCre2837-220222-122913'
# TTL             = {'Type': 'Ethovision', 'Name': 'Closed arm', 'Custom':'Closed arm'}
# create_annotated_video = {'Create?':True, 'Import':r"C:\Users\hazza\Desktop\Fibre photometry GUI\Photometry tanks\A_AgrpCre2837-220222-122913\SetupA-220222-093901_A_AgrpCre2837-220222-122913_Cam1.avi"}
# ZTP             = []
# setup           = 'Setup A'
# TRANGE          = [-5, 10]
# BASELINE_PER    = [-5, -1]
# ARTIFACT        = np.inf
# save_to_csv_zScore = False
# save_to_csv_dFF    = False
# save_to_csv_AUC    = False

# import numpy as np
# import_location = 'C:/Users/hazza/Documents/NAc_A3088W_B3089W-211213-101735'
# export_location = 'C:/Users/hazza/Documents/NAc_A3088W_B3089W-211213-101735'
# TTL             = {'Type': 'Video timestamp', 'Camera': 'Cam1', 'Custom':'Cam1_events'}
# create_annotated_video = {'Create?':False, 'Camera':'Cam1'}
# ZTP             = []
# setup           = 'Setup A'
# TRANGE          = [-5, 10]
# BASELINE_PER    = [-5, -1]
# ARTIFACT        = np.inf
# save_to_csv_zScore = True
# save_to_csv_dFF    = True
# save_to_csv_AUC    = False
# save_to_csv_F415   = True
# save_to_csv_F475   = True

# import numpy as np
# import_location = 'C:/Users/hazza/Documents/Alex videos/Alex_Agrp_NEwithintruder-220323-111301/Agrp_A2817_B2837-220323-124611'
# export_location = 'C:/Users/hazza/Documents/Alex videos/Alex_Agrp_NEwithintruder-220323-111301/Agrp_A2817_B2837-220323-124611'
# TTL = {'Type': 'Ethovision', 'Name': 'In zone(Intruder / center-point)', 'Custom': 'Intruder_zone_center_point'}
# ZTP = []
# setup = 'Setup B'
# TRANGE = [-5, 10]
# BASELINE_PER = [-5, -1]
# ARTIFACT = np.inf
# save_to_csv_dFF = True
# save_to_csv_AUC = False

# import numpy as np
# import_location = 'C:/Users/hazza/Desktop/Fibre photometry GUI/Alex Mouse 1'
# export_location = 'C:/Users/hazza/Desktop/Fibre photometry GUI/Alex Mouse 1'
# TTL = {'Type': 'TTLM', 'Name': 'Left_poke2022-02-22T12_42_05', 'Custom': 'left poke'}
# create_annotated_video = {'Create?':False, 'Camera':'Cam1'}
# ZTP = []
# setup = '415_0_green,470_0_green'
# TRANGE = [-10, 20]
# BASELINE_PER = [-10, -2]
# ARTIFACT = np.inf
# save_to_csv_zScore = True
# save_to_csv_dFF    = False
# save_to_csv_AUC    = False
# save_to_csv_F415   = False
# save_to_csv_F475   = False

# import numpy as np
# import_location = 'D:/FED x photometry/FED_reversal-220901-090611/DAsaline_86_87-220902-131431'
# export_location = 'C:/Users/hazza/Desktop'
# TTL = {'Type': 'TTLM', 'Name': 'Left', 'Custom': 'TTLM'}
# create_annotated_video = {'Create?':False, 'Camera':'Cam1'}
# ZTP = []
# setup = 'Setup A'
# TRANGE = [-20, 80]
# BASELINE_PER = [-20, -5]
# ARTIFACT = np.inf
# save_to_csv_zScore = True
# save_to_csv_dFF    = False
# save_to_csv_AUC    = False
# save_to_csv_F415   = False
# save_to_csv_F475   = False

def PeriEventTTL(import_location, export_location, TTL, ZTP, setup, TRANGE, BASELINE_PER, ARTIFACT, save_to_csv_zScore, save_to_csv_dFF, create_annotated_video, save_to_csv_F415, save_to_csv_F475):
    
    """
    # Fiber Photometry Epoc Averaging
    
    This example goes through fiber photometry analysis using techniques such as data smoothing, bleach detrending, and z-score analysis.\
    The epoch averaging was done using TDTfilter.
    
    Author Contributions:\
    TDT, David Root, and the Morales Lab contributed to the writing and/or conceptualization of the code.\
    The signal processing pipeline was inspired by the workflow developed by David Barker et al. (2017) for the Morales Lab.\
    The data used in the example were provided by David Root.
    
    Author Information:\
    David H. Root\
    Assistant Professor\
    Department of Psychology & Neuroscience\
    University of Colorado, Boulder\
    Lab Website: https://www.root-lab.org \
    david.root@colorado.edu
    
    About the authors:\
    The Root lab and Morales lab investigate the neurobiology of reward, aversion, addiction, and depression.
    
    TDT edits all user submissions in coordination with the contributing author(s) prior to publishing.
    """
    
    """ 
    **Front Matter**
    
    Import the read_block function from the tdt package.\
    Also import other python packages we care about.
    """
    
    import numpy as np
    import pandas as pd
    from sklearn.metrics import auc
    import matplotlib.pyplot as plt  # standard Python plotting library
    import scipy.stats as stats
    import matplotlib
    matplotlib.rcParams['font.size'] = 16 # set font size for all plots
    import os
    import sys
    
    from Convert_NPM_to_TDT_data import Convert_NPM_to_TDT_data
    from tdt import read_block, epoc_filter, StructType
    
    """ 
    **Importing the Data**
    """
    
    if ',' in setup:
        data_type = 'NPM'
    else:
        data_type = 'TDT'
    
    # This error comes up when using the artifact removal section of the code.
    # It happens when data.streams[GCaMP].filtered == [].
    # If ARTIFACT == np.inf, this section of the code is skipped anyway.
    np.warnings.filterwarnings('ignore', category=np.VisibleDeprecationWarning)
    
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
    
    if data_type == 'NPM':
        data = Convert_NPM_to_TDT_data(import_location)
        SCORE_EVENT = TTL['Name']
    elif data_type == 'TDT':
        data = read_block(import_location)
    
        """
        **Setup the variables for the data you want to extract**
        
        ACTIVE could be note or FED TTL pulses
        EPOC_ID [] uses all events - or [x] x-te event
        """
        #PELLET = 'Bplt'
        #ACTIVE = 'Blft'
        #EPOC_ID = [] # number of note you want as zero time point
        
        """
        Make some variables up here to so if they change in new recordings you won't have to change everything downstream
        """
        
        #ISOS = '_415A' # 405nm channel setupB. 
        #GCaMP = '_475A' # 465nm channel setupB.
        #TRANGE = [-5, 15] # window size [start time relative to epoc onset, window duration]
        #TRANGE = [-5] # window size [start time relative to epoc onset, window duration]
        #BASELINE_PER = [-5, -4] # baseline period within our window
        #ARTIFACT = np.inf # optionally set an artifact rejection level
        
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
                              "data":     data.epocs.Note.data[indices],
                              "notes":    data.epocs.Note.notes[indices]}
                data.epocs[SCORE_EVENT] = StructType(SCORE_DICT)
        
        elif TTL['Type'] == 'Video timestamp':
            SCORE_EVENT = 'Time'
            if len(ZTP) == 1:
                list_TS    = data.epocs[TTL['Camera']].notes.index
                indices_TS = [i for i in range(len(list_TS)) if list_TS[i]==ZTP[0]]
            SCORE_DICT = {"name":     SCORE_EVENT,
                          "onset":    data.epocs[TTL['Camera']].notes.ts[indices_TS],
                          "offset":   data.epocs[TTL['Camera']].notes.ts[indices_TS] + .01,
                          "type_str": data.epocs[TTL['Camera']].type_str,
                          "data":     data.epocs[TTL['Camera']].notes.index[indices_TS],
                          "notes":    data.epocs[TTL['Camera']].notes.notes[indices_TS]}
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
    
    """
    **Use epoc_filter to extract data around our epoc event**
    
    Using the `t` parameter extracts data only from the time range around our epoc event.\
    Use the `values` parameter to specify allowed values of the `ACTIVE` to extract.\
    For stream events, the chunks of data are stored in cell arrays structured as `data.streams[GCaMP].filtered`
    """
    
    
    data = epoc_filter(data, SCORE_EVENT, t=TRANGE, values=ZTP) 
    
    """
    **Optionally remove artifacts**
    
    If any waveform is above ARTIFACT level, or
    below -ARTIFACT level, remove it from the data set.
    """
    total1 = np.size(data.streams[GCaMP].filtered)
    total2 = np.size(data.streams[ISOS].filtered)
    
    """
    List comprehension checking if any single array in 2D filtered array is > Artifact or < -Artifact
    """
    
    data.streams[GCaMP].filtered = [x for x in data.streams[GCaMP].filtered 
                                    if not np.any(x > ARTIFACT) or np.any(x < -ARTIFACT)]
    data.streams[ISOS].filtered = [x for x in data.streams[ISOS].filtered 
                                    if not np.any(x > ARTIFACT) or np.any(x < -ARTIFACT)]
    
    """
    Get the total number of rejected arrays
    """
    
    bad1 = total1 - np.size(data.streams[GCaMP].filtered)
    bad2 = total2 - np.size(data.streams[ISOS].filtered)
    total_artifacts = bad1 + bad2
    
    """
    Applying a time filter to a uniformly sampled signal means that the length of each segment could vary by one sample. Let's find the minimum length so we can trim the excess off before calculating the mean.
    """
    
    """
    More examples of list comprehensions
    """
    
    min1 = np.min([np.size(x) for x in data.streams[GCaMP].filtered])
    min2 = np.min([np.size(x) for x in data.streams[ISOS].filtered])
    data.streams[GCaMP].filtered = [x[1:min1] for x in data.streams[GCaMP].filtered]
    data.streams[ISOS].filtered = [x[1:min2] for x in data.streams[ISOS].filtered]
    
    """
    Downsample and average 100x via a moving window mean
    """
    
    if data_type == 'TDT':
        N = 100 # Average every 100 samples into 1 value
        F415 = []
        F475 = []
        for lst in data.streams[ISOS].filtered: 
            small_lst = []
            for i in range(0, min2, N):
                small_lst.append(np.mean(lst[i:i+N-1])) # This is the moving window mean
            F415.append(small_lst)
        
        for lst in data.streams[GCaMP].filtered: 
            small_lst = []
            for i in range(0, min1, N):
                small_lst.append(np.mean(lst[i:i+N-1]))
            F475.append(small_lst)
            
    elif data_type == 'NPM':
        N = 1
        F415 = data.streams[ISOS].filtered
        F475 = data.streams[GCaMP].filtered
    
    """
    **Create a mean signal, standard error of signal, and DC offset**
    """
    
    meanF415 = np.mean(F415, axis=0)
    stdF415 = np.std(F415, axis=0) / np.sqrt(len(data.streams[ISOS].filtered))
    dcF415 = np.mean(meanF415)
    meanF475 = np.mean(F475, axis=0)
    stdF475 = np.std(F475, axis=0) / np.sqrt(len(data.streams[GCaMP].filtered))
    dcF475 = np.mean(meanF475)
    
    """
    **Plot epoc averaged response**
    
    Create the time vector for each stream store
    """
    
    ts1 = TRANGE[0] + np.linspace(1, len(meanF475), len(meanF475))/data.streams[GCaMP].fs*N
    ts2 = TRANGE[0] + np.linspace(1, len(meanF415), len(meanF415))/data.streams[ISOS].fs*N
    
    """
    Subtract DC offset to get signals on top of one another
    """
    
    meanF415 = meanF415 - dcF415
    meanF475 = meanF475 - dcF475
    
    """
    Start making a figure with 4 subplots.\
    First plot is the 405 and 465 averaged signals
    """
    
    fig = plt.figure(figsize=(9, 17.5))
    ax0 = fig.add_subplot(511) # work with axes and not current plot (plt.)
    
    """
    Plotting the traces
    """
    
    p1, = ax0.plot(ts1, meanF475, linewidth=2, color='green', label='GCaMP')
    p2, = ax0.plot(ts2, meanF415, linewidth=2, color='blueviolet', label='ISOS')
    
    """
    Plotting standard error bands
    """
    p3 = ax0.fill_between(ts1, meanF475+stdF475, meanF475-stdF475,
                          facecolor='green', alpha=0.2)
    p4 = ax0.fill_between(ts2, meanF415+stdF415, meanF415-stdF415,
                          facecolor='blueviolet', alpha=0.2)
    
    """
    Plotting a vertical line at t=0
    """
    
    p5 = ax0.axvline(x=0, linewidth=3, color='slategray', label=TTL['Custom'])
    
    """
    Finish up the plot
    """
    ax0.set_xlabel('Time (secs)')
    if data_type == 'TDT':
        unit = '(mV)'
    elif data_type == 'NPM':
        unit = '(AU)'
    ax0.set_ylabel('Raw signal '+unit)
    ax0.set_title('Raw ISOS and GCaMP signals for '+TTL['Custom'])
    ax0.legend(handles=[p1, p2, p5], loc='upper right')
    ax0.set_ylim(min(np.min(meanF475-stdF475), np.min(meanF415-stdF415)),
                  max(np.max(meanF475+stdF475), np.max(meanF415+stdF415)))
    ax0.set_xlim(TRANGE[0], TRANGE[1]+TRANGE[0]);
    
    """
    **Fitting 405 channel onto 465 channel to detrend signal bleaching**
    
    Scale and fit data. Algorithm sourced from Tom Davidson's Github: [FP_normalize.m](https://github.com/tjd2002/tjd-shared-code/blob/master/matlab/photometry/FP_normalize.m)
    """
    
    Y_fit_all = []
    Y_dF_all = []
    dFF = []
    for x, y in zip(F415, F475):
        x = np.array(x)
        y = np.array(y)
        bls = np.polyfit(x, y, 1)
        fit_line = np.multiply(bls[0], x) + bls[1]
        Y_fit_all.append(fit_line)
        Y_dF_all.append(y-fit_line)
        dFF.append((y-fit_line)/fit_line)
        
    dFFerror = np.std(dFF, axis=0)/np.sqrt(np.size(dFF, axis=0))
    
    """
    Getting the z-score and standard error
    """
    
    zall = []
    for dF in Y_dF_all: 
        ind = np.where((np.array(ts2)<BASELINE_PER[1]) & (np.array(ts2)>BASELINE_PER[0]))
        zb = np.mean(dF[ind])
        zsd = np.std(dF[ind])
        zall.append((dF - zb)/zsd)
       
    zerror = np.std(zall, axis=0)/np.sqrt(np.size(zall, axis=0))
    
    """
    **Heat Map based on z score of 405 fit subtracted 465**
    """
    
    ax1 = fig.add_subplot(512)
    cs = ax1.imshow(zall, cmap=plt.cm.Greys, interpolation='none', aspect="auto",
        extent=[TRANGE[0], TRANGE[1]+TRANGE[0], 0, len(data.streams[GCaMP].filtered)])
    cbar = fig.colorbar(cs, pad=0.01, fraction=0.02)
    
    ax1.set_title('Z-score heat map for '+TTL['Custom'])
    ax1.set_ylabel('Trials')
    ax1.set_xlabel('Time (secs)')
    
    """
    **Plot the dFF trace for the 465 with std error bands**
    """
    
    ax25 = fig.add_subplot(513)
    p6 = ax25.plot(ts2, np.mean(dFF, axis=0), linewidth=2, color='green', label='GCaMP')
    p7 = ax25.fill_between(ts1, np.mean(dFF, axis=0)+dFFerror
                          ,np.mean(dFF, axis=0)-dFFerror, facecolor='green', alpha=0.2)
    p8 = ax25.axvline(x=0, linewidth=3, color='slategray', label=TTL['Custom'])
    ax25.set_ylabel('dFF')
    ax25.set_xlabel('Time (secs)')
    ax25.set_xlim(TRANGE[0], TRANGE[1]+TRANGE[0])
    ax25.set_title('dFF traces for '+TTL['Custom'])
    
    """
    **Plot the z-score trace for the 465 with std error bands**
    """
    
    ax2 = fig.add_subplot(514)
    p6 = ax2.plot(ts2, np.mean(zall, axis=0), linewidth=2, color='green', label='GCaMP')
    p7 = ax2.fill_between(ts1, np.mean(zall, axis=0)+zerror
                          ,np.mean(zall, axis=0)-zerror, facecolor='green', alpha=0.2)
    p8 = ax2.axvline(x=0, linewidth=3, color='slategray', label=TTL['Custom'])
    ax2.set_ylabel('z-Score')
    ax2.set_xlabel('Time (secs)')
    ax2.set_xlim(TRANGE[0], TRANGE[1]+TRANGE[0])
    ax2.set_title('z-Score traces for '+TTL['Custom'])
    
    """
    **Quantify changes as an area under the curve for expectation, eg (-10-0 sec) vs pellet (0-10 sec)**
    """
    
    cue_ind = np.where((np.array(ts2)<0) & (np.array(ts2)>TRANGE[0]))
    AUC_cue= auc(ts2[cue_ind], np.mean(zall, axis=0)[cue_ind])
    shock_ind = np.where((np.array(ts2)>0) & (np.array(ts2)<(TRANGE[0]+TRANGE[1])))
    AUC_shock= auc(ts2[shock_ind], np.mean(zall, axis=0)[shock_ind])
    AUC = [AUC_cue, AUC_shock]
    
    """
    Run a two-sample T-test
    """
    
    t_stat,p_val = stats.ttest_ind(np.mean(zall, axis=0)[cue_ind],
                                    np.mean(zall, axis=0)[shock_ind], equal_var=False)
    
    """
    **Make a bar plot**
    """
    
    ax3 = fig.add_subplot(515)
    p9 = ax3.bar(np.arange(len(AUC)), AUC, color=[.8, .8, .8], align='center', alpha=0.5)
    
    """
    Statistical annotation
    """
    
    x1, x2 = 0, 1 # columns indices for labels
    y, h, col = max(AUC) + 2, 2, 'k'
    ax3.plot([x1, x1, x2, x2], [y, y+h, y+h, y], lw=1.5, c=col)
    p10 = ax3.text((x1+x2)*.5, y+h, "*", ha='center', va='bottom', color=col)
    
    """
    Finish up the plot
    """
    
    #ax3.set_ylim(0, y+2*h)
    ax3.set_ylabel('Area under curve')
    ax3.set_title('Z-score AUC values before and after '+TTL['Custom'])
    ax3.set_xticks(np.arange(-1, len(AUC)+1))
    ax3.set_xticklabels(['', 'Before', 'After', ''])
    
    fig.tight_layout()
    
    '==============================================='
    import os
    import numpy as np
    import pylab
    import itertools, collections
    import matplotlib.pyplot as plt
    import pandas as pd
    import bisect
    from scipy.stats import linregress
    
    """
    Create the pandas dataframes to export as CSV files.
    """
    
    event_onsets = data.epocs[SCORE_EVENT].onset
    if len(event_onsets) != len(zall):
        print('\nPLEASE NOTE: the last '+str(len(event_onsets)-len(zall)) + ' events '+
              'have been excluded, because their window durations go past the end of '+
              'the recording.\n')
    
    result_arrays = {'zScore':zall, 'dFF':dFF, 'F415':F415, 'F475':F475} # Numpy arrays
    results       = {'zScore':zall, 'dFF':dFF, 'F415':F415, 'F475':F475} # Eventually dataframes with headers
    Zscore_max_threshold = 1
    
    for stat in results.keys():
        
        # Create a header for the results.
        header = {}
        header['Names'] = [str(i) for i in range(1,len(result_arrays[stat])+1)]
        header['Times'] = list(data.epocs[SCORE_EVENT].onset)
        # Make sure the analysed data columns and event names are the same length.
        header['Times'] = header['Times'][:len(header['Names'])]
        all_headers = list(zip(*[header[key] for key in header.keys()]))
        all_headers = pd.MultiIndex.from_tuples(all_headers)
        
        # Create a pandas dataframe for the poke names and Z-score data.
        results[stat] = pd.DataFrame(np.transpose(results[stat]), columns=all_headers)
        
        # Add in extra columns.
        results[stat].insert(0, ('Mean of TTLs',''), results[stat].mean(axis=1))
        results[stat].insert(0, ('Time stamps (secs)',''), ts1)
        results[stat].insert(0, ('','Time of event onset (secs)'), ['']*len(ts1))
        
        # If there is notes information, add many more header rows.
        if 'notes' in data.epocs[SCORE_EVENT].keys():
            
            # Define all the extra rows as entries in a dictionary.
            extra_rows = {}
    
            list_notes = list(data.epocs[SCORE_EVENT].notes)
            # Make sure the analysed data columns and event names are the same length.
            list_notes = list_notes[:len(list_notes)]
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
            # list_slopes = [linregress(ts1,col)[0] for col in result_arrays[stat]]
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
            list_time_Zscores = [ts1[np.argmax(col)] for col in result_arrays[stat]]
            extra_rows['Time of max values'] = ['',''] + list_time_Zscores
            list_thresh_Zscores = [('Yes' if max1>Zscore_max_threshold else 'No') 
                                   for max1 in list_max_Zscores]
            extra_rows['Max value above '+str(Zscore_max_threshold)+'?'] = (
                       ['',''] + list_thresh_Zscores)
            extra_rows['Filename'] = ['',''] + [os.path.basename(import_location)]*len(list_notes)
            extra_rows['Custom name'] = ['',''] + [TTL['Custom']]*len(list_notes)
            
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
        
    row1 = 'Area under curve of Z-Score vs time from '+str(TRANGE[0])+' to 0 seconds (before TTL)'
    row2 = 'Area under curve of Z-Score vs time from 0 to '+str(TRANGE[0]+TRANGE[1])+' seconds (after TTL)'
    results['AUC'] = pd.DataFrame({'Data':AUC},index=[row1,row2])
    
    
    'OUTPUT'
    
    filename = import_location
    
    #save_to_csv_zScore = True
        # If true then input the file name
    filename_zScore = os.path.join (
        #os.path.dirname (filename), # saves file in same directory
        export_location, # change the path to where you want files saved
        os.path.basename(filename) + "_zScore_" + TTL['Custom'] + '_' + setup.replace(' ','_') + ".csv")
        #os.path.basename(filename) + "_zall_activeB" + ".csv")
    
    #save_to_csv_dFF = True
        # If true then input the file name
    filename_dFF = os.path.join (
        # os.path.dirname (filename), # saves file in same directory
        export_location,  # change the path to where you want files saved
        os.path.basename(filename) + "_dFF_" + TTL['Custom'] + '_' + setup.replace(' ','_') + ".csv")
        # os.path.basename(filename) + "_DFF" + str(SHOCK_CODE) + ".csv")
        
    #save_to_csv_dFF = True
        # If true then input the file name
    filename_F415 = os.path.join (
        # os.path.dirname (filename), # saves file in same directory
        export_location,  # change the path to where you want files saved
        os.path.basename(filename) + ISOS + "_" + TTL['Custom'] + '_' + setup.replace(' ','_') + ".csv")
        # os.path.basename(filename) + "_DFF" + str(SHOCK_CODE) + ".csv")
        
    #save_to_csv_dFF = True
        # If true then input the file name
    filename_F475 = os.path.join (
        # os.path.dirname (filename), # saves file in same directory
        export_location,  # change the path to where you want files saved
        os.path.basename(filename) + GCaMP + "_" + TTL['Custom'] + '_' + setup.replace(' ','_') + ".csv")
        # os.path.basename(filename) + "_DFF" + str(SHOCK_CODE) + ".csv")
    
    # save_to_csv_ts1 = True
    #     # If true then input the file name
    # filename_ts1 = os.path.join (
    #     #os.path.dirname (filename), # saves file in same directory
    #     BLOCKPATH, # change the path to where you want files saved
    #     os.path.basename(filename) + "_ts1" + ".csv")    
    '==============================================='
    #if save_to_csv_analysis:
        #import csv
        #with open(filename_analysis, 'w', newline='') as f:
            # w = csv.writer(f)
            # for key, val in analysis.items():
                # w.writerow([key,*val])
            # print("Printed Analysis CSV")
    
    
    if save_to_csv_zScore:
        #np.savetxt(filename_zScore_active, zScore, delimiter=",")
        results['zScore'].to_csv(filename_zScore, index=False)
        fig.savefig(filename_zScore[:-4]+'.png') # Save the figure in the export location.
        #print("Printed zall CSV")
        
    if save_to_csv_dFF:
        #np.savetxt(filename_dFF, dFF, delimiter=",")
        results['dFF'].to_csv(filename_dFF, index=False)
        #print("Printed dFF CSV")
        
    if save_to_csv_F415:
        #np.savetxt(filename_zScore_active, zScore, delimiter=",")
        results['F415'].to_csv(filename_F415, index=False)
        #print("Printed zall CSV")
        
    if save_to_csv_F475:
        #np.savetxt(filename_dFF, dFF, delimiter=",")
        results['F475'].to_csv(filename_F475, index=False)
        #print("Printed dFF CSV")
    
    # if save_to_csv_ts1:
    #     np.savetxt(filename_ts1, ts1, delimiter=",")
    #     print("Printed time CSV")
    
    '==============================================='
     
    #plt.show()
    plt.close()
    
    if create_annotated_video['Create?'] == True:
        
        import cv2 as cv
        import matplotlib
        matplotlib.use('agg')
        plt = matplotlib.pyplot
        import numpy as np
        from tqdm import tqdm
    
        # Import the video by looking for "Cam1" or "Cam2".
        video_file = [file for file in os.listdir(import_location) if 
                      (create_annotated_video['Camera'] in file)]
        if len(video_file) == 0:
            print('Error: check whether')
            print('- The video file is in '+import_location)
            print('- The video filename contains "Cam1" or "Cam2"')
            sys.exit()
        import_destination = os.path.join(import_location,video_file[0])
    
        print('Creating video snippets for '+str(len(zall))+' epochs.')
        
        # Create a folder in the export location with the snipped videos.
        folder_name = 'Video snippets0'
        i = 1
        while folder_name in os.listdir(export_location):
            folder_name = folder_name[:-1] + str(i)
            i += 1
        export_location = os.path.join(export_location, folder_name)
        os.makedirs(export_location)
        
        for i in tqdm(range(len(zall)), ncols=70):
        
            window = 'Choose frames'
            warning = "!!! Failed cap.read()"
            cap = cv.VideoCapture(import_destination)
            
            fps = cap.get(cv.CAP_PROP_FPS)
            start = data.time_ranges[0][i]
            end   = data.time_ranges[1][i]
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
            x1 = ts1
            y1 = zall[i]
            df1 = pd.DataFrame({'x':x1,'y':y1,'Type':len(x1)*['TDT']})
            # Create a list of the time stamps from the video frames and Z-scores.
            x2 = list(np.arange(TRANGE[0]+1/fps, TRANGE[0]+TRANGE[1]+1/fps, 1/fps))
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
            plt.xlim(TRANGE[0],TRANGE[0]+TRANGE[1])
            y_min = min([min(zall[i]) for i in range(len(zall))])
            y_max = max([max(zall[i]) for i in range(len(zall))])
            plt.ylim(y_min,y_max)
            plt.xlabel('Time (secs)')
            plt.ylabel('Z-Score')
            
            plt.tight_layout(h_pad=0)
            plt.subplots_adjust(left=0.07)
            line1, = plt.plot([], [], 'g-', lw=1.5)
            plt.axvline(x = 0, color = 'lightgray', linestyle='dashed')
        
            cap.set(cv.CAP_PROP_POS_FRAMES, start)
            result = cv.VideoWriter(os.path.join(export_location, 'TTL'+str(i)+'.mp4'), 
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
                # cv.imshow(window,frame)
        
                # # If the red X button is pressed, close the window.
                # if cv.getWindowProperty(window,cv.WND_PROP_VISIBLE) < 1:
                #     break
                
                if int(cap.get(cv.CAP_PROP_POS_FRAMES)) == end:
                    break
                    
            # When everything done, release 
            # the video capture and video 
            # write objects
            cap.release()
            result.release()
                
            # Closes all the frames
            cv.destroyAllWindows()
            
            plt.close()
    
