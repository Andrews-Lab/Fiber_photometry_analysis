# # To run this file manually, uncomment this section, comment the function definition and un-indent the rest of the code.
# import numpy as np
# import_location = 'C:/Users/hazza/Desktop/Fibre photometry GUI/Photometry tanks/NAc_3147K_2614K-211118-105517'
# export_location = 'C:/Users/hazza/Desktop/Fibre photometry GUI/Photometry tanks/NAc_3147K_2614K-211118-105517'
# active_poke     = 'Left'
# ZTP             = []
# setup           = 'Setup A'
# TRANGE          = [-20.0, 80.0]
# BASELINE_PER    = [-20.0, -5.0]
# ARTIFACT        = np.inf
# save_to_csv_zScore = True
# save_to_csv_dFF    = False
# save_to_csv_AUC    = False
# save_to_csv_F415   = True
# save_to_csv_F475   = True

# # To run this file manually, uncomment this section, comment the function definition and un-indent the rest of the code.
# import numpy as np
# import_location = r'C:\Users\hazza\Downloads\Alex H tank'
# export_location = r'C:\Users\hazza\Downloads\Alex H tank'
# active_poke     = 'Right'
# ZTP             = []
# setup           = 'Setup B'
# TRANGE          = [-20.0, 60]
# BASELINE_PER    = [-20.0, -5.0]
# ARTIFACT        = np.inf
# save_to_csv_zScore = True
# save_to_csv_dFF    = False
# save_to_csv_AUC    = False
# save_to_csv_F415   = False
# save_to_csv_F475   = False

# # # To run this file manually, uncomment this section, comment the function definition and un-indent the rest of the code.
# import numpy as np
# import_location = 'D:/FED x photometry/FED_reversal-220901-090611/DAsaline_86_87-220902-131431'
# export_location = 'C:/Users/hazza/Desktop'
# active_poke = 'Left'
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

def FED3_one_active_poke(import_location, export_location, poke_to_analyse, active_poke, ZTP, setup, TRANGE, BASELINE_PER, ARTIFACT, save_to_csv_zScore, save_to_csv_dFF, save_to_csv_F415, save_to_csv_F475):
    
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
    from sklearn.metrics import auc
    import matplotlib.pyplot as plt  # standard Python plotting library
    import scipy.stats as stats
    import matplotlib 
    matplotlib.rcParams['font.size'] = 16 # set font size for all plots
    import os
    import sys
    
    from Convert_NPM_to_TDT_data import Convert_NPM_to_TDT_data
    from tdt import read_block, epoc_filter#, download_demo_data
    
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
        data_original = Convert_NPM_to_TDT_data(import_location)
        
        list_events = list(data_original.epocs.keys())
        PELLET = [event for event in list_events if 
                  ('Pellet' in event or 'pellet' in event)][0]
        ACTIVE = [event for event in list_events if 
                  (poke_to_analyse in event or poke_to_analyse.lower() in event)][0]
        
    elif data_type == 'TDT':
        data_original = read_block(import_location)
        
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
        Set the variable names for the pellet drops and active pokes.
        """
        if ISOS == '_405A' and GCaMP == '_465A': # If setup A is used.
            possible_pellets = ['Aplt','APlt']
            if poke_to_analyse == 'Left':
                possible_actives = ['Alft','ALft']
            elif poke_to_analyse == 'Right':
                possible_actives = ['Argt','ARgt']
        
        elif ISOS == '_415A' and GCaMP == '_475A': # If setup B is used.
            possible_pellets = ['Bplt','BPlt']
            if poke_to_analyse == 'Left':
                possible_actives = ['Blft','BLft']
            elif poke_to_analyse == 'Right':
                possible_actives = ['Brgt','BRgt']
        
        # Find the element from data.epocs that matches one of the possible pellet and active poke names.
        PELLET = list(set(possible_pellets).intersection(data_original.epocs.keys()))
        ACTIVE = list(set(possible_actives).intersection(data_original.epocs.keys()))
        
        # If the event names cannot be found, try the generic "Left", "Right" or "Pelt".
        # They could refer to setup A or setup B, so those were not used first.
        if len(PELLET) == 0 or len(ACTIVE) == 0:
            possible_pellets += ['Pelt']
            if poke_to_analyse == 'Left':
                possible_actives += ['Left']
            elif poke_to_analyse == 'Right':
                possible_actives += ['Rght','RGht']
            PELLET = list(set(possible_pellets).intersection(data_original.epocs.keys()))
            ACTIVE = list(set(possible_actives).intersection(data_original.epocs.keys()))
        
        if len(PELLET) != 1 or len(ACTIVE) != 1:
            print('More possible pellet retrieval or active poke names are needed in this code.')
            sys.exit()
        PELLET, ACTIVE = PELLET[0], ACTIVE[0]
    
    """
    **Use epoc_filter to extract data around our epoc event**
    
    Using the `t` parameter extracts data only from the time range around our epoc event.\
    Use the `values` parameter to specify allowed values of the `ACTIVE` to extract.\
    For stream events, the chunks of data are stored in cell arrays structured as `data.streams[GCaMP].filtered`
    """
    
    data = epoc_filter(data_original, ACTIVE, t=TRANGE, values=ZTP) 
    
    """
    **Optionally remove artifacts**
    
    If any waveform is above ARTIFACT level, or
    below -ARTIFACT level, remove it from the data set.
    """
    # A warning appears for the artifacts section only.
    import warnings
    warnings.filterwarnings("ignore", category=np.VisibleDeprecationWarning) 
    
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
    
    fig = plt.figure(figsize=(9, 14))
    ax0 = fig.add_subplot(411) # work with axes and not current plot (plt.)
    
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
    
    p5 = ax0.axvline(x=0, linewidth=3, color='slategray', label='Pellet retrieval')
    
    """
    Finish up the plot
    """
    ax0.set_xlabel('Seconds')
    ax0.set_ylabel('mV')
    ax0.set_title('Pellet retrieval, %i Trials (%i Artifacts Removed)'
                  % (len(data.streams[GCaMP].filtered), total_artifacts))
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
    
    ax1 = fig.add_subplot(412)
    cs = ax1.imshow(zall, cmap=plt.cm.Greys, interpolation='none', aspect="auto",
        extent=[TRANGE[0], TRANGE[1]+TRANGE[0], 0, len(data.streams[GCaMP].filtered)])
    cbar = fig.colorbar(cs, pad=0.01, fraction=0.02)
    
    ax1.set_title('Individual z-Score Traces')
    ax1.set_ylabel('Trials')
    ax1.set_xlabel('Seconds Pellet retrieval')
    
    """
    **Plot the z-score trace for the 465 with std error bands**
    """
    
    ax2 = fig.add_subplot(413)
    p6 = ax2.plot(ts2, np.mean(zall, axis=0), linewidth=2, color='green', label='GCaMP')
    p7 = ax2.fill_between(ts1, np.mean(zall, axis=0)+zerror
                          ,np.mean(zall, axis=0)-zerror, facecolor='green', alpha=0.2)
    p8 = ax2.axvline(x=0, linewidth=3, color='slategray', label='pellet retrieval')
    ax2.set_ylabel('z-Score')
    ax2.set_xlabel('Seconds')
    ax2.set_xlim(TRANGE[0], TRANGE[1]+TRANGE[0])
    ax2.set_title('Pellet retrieval')
    
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
    
    ax3 = fig.add_subplot(414)
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
    
    ax3.set_ylim(0, y+2*h)
    ax3.set_ylabel('AUC')
    ax3.set_title('expectation vs pellet')
    ax3.set_xticks(np.arange(-1, len(AUC)+1))
    ax3.set_xticklabels(['', 'expectation', 'pellet', ''])
    
    fig.tight_layout()
    
    '==============================================='
    import os
    import numpy as np
    import pylab
    import itertools, collections
    import matplotlib.pyplot as plt
    import pandas as pd
    import bisect
    
    
    
    """
    Separate rewarded and non rewarded pokes
    """
    
    # ACTIVE_ON is the time stamps of all the nose pokes.
    # PELLET_ON is the time stamps of the pellet drops.
    ACTIVE_ON = data.epocs[ACTIVE].onset
    # Use the tank "data_original", otherwise pellet drops that happen outside
    # of a given epoch window will be excluded.
    # The tank "data" excludes events that happen outside of each epoch window.
    PELLET_ON = data_original.epocs[PELLET].onset # Pellet retrievals
    
    # # If the number of nose pokes in zall and ACTIVE_ON are different, show a message.
    # if len(ACTIVE_ON) != len(zall):
    #     print('\nDecrease the size of the window duration and run the code again.')
    #     import sys
    #     sys.exit()
    # If the number of nose pokes in zall and ACTIVE_ON are different, show a message.
    if len(ACTIVE_ON) != len(zall):
        print('\nPLEASE NOTE: '+str(len(ACTIVE_ON)-len(zall)) + ' events have '+
              'been excluded, because their window durations go past the end of '+
              'the recording.\n')
        ACTIVE_ON = ACTIVE_ON[:len(zall)]
        
    def is_rewarded(active_poke, current_poke):
        # The order of pellet drops and pokes are needed to determine whether a 
        # poke is active.
        if active_poke == 'Changing':
            return('')
        # Check whether the current poke is active.
        # The values for current_poke can only be 'Left' or 'Right'.
        elif active_poke == current_poke:
            return('Rewarded')
        else:
            return('Non-rewarded')
    
    # List the nose pokes and pellet drops together and order them by time.
    all_events = []
    for i in range(len(ACTIVE_ON)):
        all_events.append({'Time point':ACTIVE_ON[i], 'Event':'Nose poke',   'Poke number':i+1, 'Rewarded':is_rewarded(active_poke, poke_to_analyse)})
    for i in range(len(PELLET_ON)):
        all_events.append({'Time point':PELLET_ON[i], 'Event':'Pellet drop', 'Poke number':'',  'Rewarded':''})
    def sort_by_time(dict1):
        return(dict1['Time point'])
    all_events.sort(key=sort_by_time)
    
    if active_poke == 'Changing':
    
        # If a pellet drop happens after a nose poke, that nose poke is rewarded.
        # Otherwise, the poke is non-rewarded.
        for i in range(len(all_events)):
            if all_events[i]['Event'] == 'Nose poke':
                if i != len(all_events)-1 and all_events[i+1]['Event'] == 'Pellet drop':
                    all_events[i]['Rewarded'] = 'Rewarded'
                else:
                    all_events[i]['Rewarded'] = 'Non-rewarded'
    
    # If you want to visualise this completed table, use pd.DataFrame(all_events).
    # Make a dictionary that shows which poke numbers are rewarded or non-rewarded.
    poke_numbers = [dict1['Poke number'] for dict1 in all_events if dict1['Event'] == 'Nose poke']
    rewarded     = [dict1['Rewarded']    for dict1 in all_events if dict1['Event'] == 'Nose poke']
    reward_poke  = dict(zip(poke_numbers, rewarded))
    
    """
    Create the pandas dataframes to export as CSV files.
    """
    
    results = {'zScore':zall, 'dFF':dFF, 'F415':F415, 'F475':F475}
    
    for stat in results.keys():
    
        # Create a pandas dataframe for the poke names and Z-score data.
        headers = [reward_poke[key]+' poke '+str(key) for key in reward_poke.keys()]
        results[stat] = pd.DataFrame(np.transpose(results[stat]), columns=headers)
        
        # Add the rewarded/non-rewarded labels and poke numbers as separate rows.
        # This prepares the columns for sorting.
        ind_headers = [[reward_poke[poke], poke] for poke in reward_poke.keys()]
        ind_headers = pd.DataFrame(dict(zip(headers, ind_headers)),index=['Rewarded','Poke number'])
        results[stat] = pd.concat([ind_headers, results[stat]])
        
        # Sort the columns by rewarded/non-rewarded labels and then the poke numbers.
        results[stat] = results[stat].sort_values(by=['Rewarded','Poke number'], axis=1, ascending=[False,True])
        results[stat] = results[stat].drop(['Rewarded', 'Poke number'])
        
        # Make the indices the time stamps of each Z-score.
        #results[stat].index = np.arange(0, TRANGE[1], TRANGE[1]/(results[stat].shape[0]))
        results[stat].index = ts1
        results[stat].index.names = ['Time stamps (secs)']
        
        # Find the means of the rewarded and non-rewarded columns.
        mean_rewarded    = results[stat][[col for col in results[stat].columns if col[:8]  == 'Rewarded']].mean(axis=1)
        mean_nonrewarded = results[stat][[col for col in results[stat].columns if col[:12] == 'Non-rewarded']].mean(axis=1)
        results[stat].insert(0, 'Mean of non-rewarded nose pokes', mean_nonrewarded)
        results[stat].insert(0, 'Mean of rewarded nose pokes', mean_rewarded)
        
        # Add the individual headers back.
        ind_headers = ind_headers.sort_values(by=['Rewarded','Poke number'], axis=1, ascending=[False,True])
        ind_headers.insert(0, 'Mean of non-rewarded nose pokes', ['',''])
        ind_headers.insert(0, 'Mean of rewarded nose pokes', ['',''])
        ind_headers.index = ['',' ']
        ind_headers.index.names = ['Time stamps (secs)']
        results[stat] = pd.concat([ind_headers, results[stat]])
        
    row1 = 'Area under curve of Z-Score vs time from '+str(TRANGE[0])+' to 0 seconds (before pellet retrieval)'
    row2 = 'Area under curve of Z-Score vs time from 0 to '+str(TRANGE[0]+TRANGE[1])+' seconds (after pellet retrieval)'
    results['AUC'] = pd.DataFrame({'Data':AUC},index=[row1,row2])
    
    
    'OUTPUT'
    
    filename = import_location
    
    #save_to_csv_zScore = True
        # If true then input the file name
    filename_zScore = os.path.join (
        #os.path.dirname (filename), # saves file in same directory
        export_location, # change the path to where you want files saved
        (os.path.basename(filename) + "_zScore" + "_analyse_" + poke_to_analyse.lower() + 
         "_active_is_" + active_poke.lower() + '_' + setup.replace(' ','_') + ".csv"))
        #os.path.basename(filename) + "_zall_activeB" + ".csv")
        
    #save_to_csv_dFF = True
        # If true then input the file name
    filename_dFF = os.path.join (
        # os.path.dirname (filename), # saves file in same directory
        export_location,  # change the path to where you want files saved
        (os.path.basename(filename) + "_dFF_" + "_analyse_" + poke_to_analyse.lower() + 
         "_active_is_" + active_poke.lower() + '_' + setup.replace(' ','_') + ".csv"))
        # os.path.basename(filename) + "_DFF" + str(SHOCK_CODE) + ".csv")
        
    #save_to_csv_dFF = True
        # If true then input the file name
    filename_F415 = os.path.join (
        # os.path.dirname (filename), # saves file in same directory
        export_location,  # change the path to where you want files saved
        (os.path.basename(filename) + ISOS + "_analyse_" + poke_to_analyse.lower() + 
         "_active_is_" + active_poke.lower() + '_' + setup.replace(' ','_') + ".csv"))
        # os.path.basename(filename) + "_DFF" + str(SHOCK_CODE) + ".csv")
        
    #save_to_csv_dFF = True
        # If true then input the file name
    filename_F475 = os.path.join (
        # os.path.dirname (filename), # saves file in same directory
        export_location,  # change the path to where you want files saved
        (os.path.basename(filename) + GCaMP + "_analyse_" + poke_to_analyse.lower() + 
         "_active_is_" + active_poke.lower() + '_' + setup.replace(' ','_') + ".csv"))
        # os.path.basename(filename) + "_DFF" + str(SHOCK_CODE) + ".csv")
    
    # #save_to_csv_AUC = True
    #     # If true then input the file name
    # filename_AUC = os.path.join (
    #     #os.path.dirname (filename), # saves file in same directory
    #     export_location, # change the path to where you want files saved
    #     os.path.basename(filename) + "_AUC_" + poke_to_analyse.lower() + "_active_poke" + ".csv")
    
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
        results['zScore'].to_csv(filename_zScore)
        #print("Printed zall CSV")
        
    if save_to_csv_dFF:
        #np.savetxt(filename_dFF, dFF, delimiter=",")
        results['dFF'].to_csv(filename_dFF)
        #print("Printed dFF CSV")
        
    if save_to_csv_F415:
        #np.savetxt(filename_zScore_active, zScore, delimiter=",")
        results['F415'].to_csv(filename_F415)
        #print("Printed zall CSV")
        
    if save_to_csv_F475:
        #np.savetxt(filename_dFF, dFF, delimiter=",")
        results['F475'].to_csv(filename_F475)
        #print("Printed dFF CSV")
    
    # if save_to_csv_AUC:
    #     #np.savetxt(filename_AUC_active, AUC, delimiter=",")
    #     results['AUC'].to_csv(filename_AUC, header=False)
    #     print("Printed AUC CSV")
    
    # if save_to_csv_ts1:
    #     np.savetxt(filename_ts1, ts1, delimiter=",")
    #     print("Printed time CSV")
    
    '==============================================='
     
    #plt.show()
    plt.close()
