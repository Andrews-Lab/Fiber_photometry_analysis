# # To run this file manually, uncomment this section, comment the function definition and un-indent the rest of the code.
# import numpy as np
# import_location = 'C:/Users/hazza/Desktop/Photometry/NAc_3147K_2614K-211118-105517'
# export_location = 'C:/Users/hazza/Desktop/Photometry/NAc_3147K_2614K-211118-105517'
# TTL             = {'Test':'2 bottle choice','Type':'Left','Name':'Left'}
# ZTP             = []
# ISOS            = '_405A'
# GCaMP           = '_465A'
# ARTIFACT        = np.inf
# save_to_csv_zScore = True
# save_to_csv_dFF    = False
# save_to_csv_AUC    = False

def Between_TTLs(import_location, export_location, TTL, ZTP, setup, ARTIFACT, save_to_csv_zScore, save_to_csv_dFF):
    
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
    
    #import_location = 'C:/Users/hazza/Desktop/Photometry/NAc_3147K_2614K-211118-105517'
    #download_demo_data()
    data_full = read_block(import_location)
    
    if data_type == 'NPM':
        data_full = Convert_NPM_to_TDT_data(import_location)
    elif data_type == 'TDT':
        data_full = read_block(import_location)
    
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
    **Use epoc_filter to extract data around our epoc event**
    
    Using the `t` parameter extracts data only from the time range around our epoc event.\
    Use the `values` parameter to specify allowed values of the `ACTIVE` to extract.\
    For stream events, the chunks of data are stored in cell arrays structured as `data.streams[GCaMP].filtered`
    """
    
    TRANGE = [0]
    data = epoc_filter(data_full, TTL['Name'], t=TRANGE, values=ZTP)
    
    """
    If there is only one element in TRANGE, append the maximum window duration.
    """
    
    # TRANGE is defined as [time before epoch onset, window duration starting from this time].
    # If only one value is given in TRANGE, the window duration is the time between TRANGE[0] and each epoch offset.
    # But the rest of the code relies on a value for the window duration for the time-axis range.
    # Thus, append the largest window duration to TRANGE.
    
    if len(TRANGE) == 1:
        start_times = data.time_ranges[0]
        end_times   = data.time_ranges[1]
        # If there is an infinity in end_times, make that the end of the recording.
        recording_end = data.info.duration.total_seconds()
        end_times = np.where(end_times==np.inf, recording_end, end_times) 
        # Find the largest window duration by subtracting end_times and start_times.
        TRANGE += [max(np.subtract(end_times, start_times))]
    
    """
    Check that the note comments are included in data.epocs.Note.notes.
    If not, go back to the original Notes.txt file and find the phrases in "" marks.
    Assign these comments to data.epocs.Note.notes.
    """
    
    if 'Note' in data.epocs.keys() and np.all(data.epocs.Note.notes == 'none'):
        
        notes_txt_path = os.path.join(BLOCKPATH, 'Notes.txt')
        with open(notes_txt_path, 'r') as notes_file:
            notes_lines = notes_file.readlines()
            
        def find_comment(note):
            ind = [i for i in range(len(note)) if note[i]=='"']
            return(note[ind[0]+1:ind[1]])
        notes_lines = [find_comment(note) for note in notes_lines if note[:5]=='Note-']
        data.epocs.Note.notes = np.array(notes_lines)
    
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
    
    # min1 = np.min([np.size(x) for x in data.streams[GCaMP].filtered])
    # min2 = np.min([np.size(x) for x in data.streams[ISOS].filtered])
    # data.streams[GCaMP].filtered = [x[1:min1] for x in data.streams[GCaMP].filtered]
    # data.streams[ISOS].filtered = [x[1:min2] for x in data.streams[ISOS].filtered]
    
    """
    Downsample and average 100x via a moving window mean
    """
    
    # N = 100 # Average every 100 samples into 1 value
    # F415 = []
    # F475 = []
    # for lst in data.streams[ISOS].filtered: 
    #     small_lst = []
    #     for i in range(0, min2, N):
    #         small_lst.append(np.mean(lst[i:i+N-1])) # This is the moving window mean
    #     F415.append(small_lst)
    
    
    # for lst in data.streams[GCaMP].filtered: 
    #     small_lst = []
    #     for i in range(0, min1, N):
    #         small_lst.append(np.mean(lst[i:i+N-1]))
    #     F475.append(small_lst)
    
    if data_type == 'TDT':
        N = 100 # Average every 100 samples into 1 value
        F415 = []
        F475 = []
        for lst in data.streams[ISOS].filtered: 
            small_lst = []
            for i in range(0, len(lst), N):
                small_lst.append(np.mean(lst[i:i+N-1])) # This is the moving window mean
            F415.append(small_lst)
        
        for lst in data.streams[GCaMP].filtered: 
            small_lst = []
            for i in range(0, len(lst), N):
                small_lst.append(np.mean(lst[i:i+N-1]))
            F475.append(small_lst)
            
        # Additional section.
        lst = data_full.streams[ISOS].data
        F415_full = []
        for i in range(0, len(lst), N):
            F415_full.append(np.mean(lst[i:i+N-1])) # This is the moving window mean
            
        lst = data_full.streams[GCaMP].data
        F475_full = []
        for i in range(0, len(lst), N):
            F475_full.append(np.mean(lst[i:i+N-1])) # This is the moving window mean
            
    elif data_type == 'NPM':
        N = 1
        F415 =      data.streams[ISOS].filtered
        F475 =      data.streams[GCaMP].filtered
        F415_full = data_full.streams[ISOS].data
        F475_full = data_full.streams[GCaMP].data
    
    """
    **Create a mean signal, standard error of signal, and DC offset**
    """
    
    # meanF415 = np.mean(F415, axis=0)
    # stdF415 = np.std(F415, axis=0) / np.sqrt(len(data.streams[ISOS].filtered))
    # dcF415 = np.mean(meanF415)
    # meanF475 = np.mean(F475, axis=0)
    # stdF475 = np.std(F475, axis=0) / np.sqrt(len(data.streams[GCaMP].filtered))
    # dcF475 = np.mean(meanF475)
    
    # The following functions take a list of lists with unequal lengths.
    # It adds nans to make these lengths equal.
    # Then they find the mean and standard deviations while ignoring nan values.
    
    def mean1(list_of_lists, axis):
        # Find the length of the longest list.
        max_len = max([len(list1) for list1 in list_of_lists])
        # Fill each smaller sublist with nans until they are all the same length.
        list_of_lists = [list1+(max_len-len(list1))*[np.nan] for list1 in list_of_lists]
        # Find the mean across each sublist, ignoring nan values.
        return(np.nanmean(list_of_lists,axis=axis))
    
    def std1(list_of_lists, axis):
        max_len = max([len(list1) for list1 in list_of_lists])
        list_of_lists = [list1+(max_len-len(list1))*[np.nan] for list1 in list_of_lists]
        return(np.nanstd(list_of_lists,axis=axis))
    
    meanF415 = mean1(F415, axis=0)
    stdF415 = std1(F415, axis=0) / np.sqrt(len(data.streams[ISOS].filtered))
    dcF415 = np.mean(meanF415)
    meanF475 = mean1(F475, axis=0)
    stdF475 = std1(F475, axis=0) / np.sqrt(len(data.streams[GCaMP].filtered))
    dcF475 = np.mean(meanF475)
    
    # Additional section.
    list_meanF415 = [np.mean(list1) for list1 in F415]
    list_meanF475 = [np.mean(list1) for list1 in F475]
    list_stdF415  = [np.std(list1)  for list1 in F415]
    
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
        
    # Additional section.
    Y_dF_all_full = []
    x = np.array(F415_full)
    y = np.array(F475_full)
    bls = np.polyfit(x, y, 1)
    fit_line = np.multiply(bls[0], x) + bls[1]
    Y_dF_all_full = y-fit_line
    
    """
    Getting the z-score and standard error
    """
    
    # zall = []
    # for dF in Y_dF_all: 
    #    ind = np.where((np.array(ts2)<BASELINE_PER[1]) & (np.array(ts2)>BASELINE_PER[0]))
    #    zb = np.mean(dF[ind])
    #    zsd = np.std(dF[ind])
    #    zall.append((dF - zb)/zsd)
       
    # zerror = np.std(zall, axis=0)/np.sqrt(np.size(zall, axis=0))
    
    # The following functions take a list of numpy arrays with unequal lengths.
    # It adds nans to the numpy arrays to make these lengths equal.
    # Then they find the mean and standard deviations while ignoring nan values.
    
    def mean2(list_of_lists, axis):
        max_len = max([len(list1) for list1 in list_of_lists])
        list_of_lists = [np.concatenate((list1, np.array((max_len-len(list1))*[np.nan]))) for list1 in list_of_lists]
        return(np.nanmean(list_of_lists,axis=axis))
    
    def std2(list_of_lists, axis):
        max_len = max([len(list1) for list1 in list_of_lists])
        list_of_lists = [np.concatenate((list1, np.array((max_len-len(list1))*[np.nan]))) for list1 in list_of_lists]
        return(np.nanstd(list_of_lists,axis=axis))
    
    # zall = []
    # for dF in Y_dF_all: 
    #    ind = np.where((np.array(ts2)<BASELINE_PER[1]) & (np.array(ts2)>BASELINE_PER[0]))
    #    zb = np.mean(dF[ind])
    #    zsd = np.std(dF[ind])
    #    zall.append((dF - zb)/zsd)
       
    # zerror = std2(zall, axis=0)/np.sqrt(np.size(zall, axis=0))
    
    zall = []
    for dF in Y_dF_all: 
       zb = np.mean(Y_dF_all_full)
       zsd = np.std(Y_dF_all_full)
       zall.append((dF - zb)/zsd)
       
    zerror = std2(zall, axis=0)/np.sqrt(np.size(zall, axis=0))
    
    
    """
    **Heat Map based on z score of 405 fit subtracted 465**
    """
    
    # ax1 = fig.add_subplot(412)
    # cs = ax1.imshow(zall, cmap=plt.cm.Greys, interpolation='none', aspect="auto",
    #     extent=[TRANGE[0], TRANGE[1]+TRANGE[0], 0, len(data.streams[GCaMP].filtered)])
    # cbar = fig.colorbar(cs, pad=0.01, fraction=0.02)
    
    # ax1.set_title('Individual z-Score Traces')
    # ax1.set_ylabel('Trials')
    # ax1.set_xlabel('Seconds Pellet retrieval')
    
    # This function takes a list of numpy arrays with unequal lengths, and adds nans
    # to make each array equal length.
    def add_nans(list_of_lists):
        max_len = max([len(list1) for list1 in list_of_lists])
        list_of_lists = [np.concatenate((list1, np.array((max_len-len(list1))*[np.nan]))) for list1 in list_of_lists]
        return(list_of_lists)
    
    ax1 = fig.add_subplot(412)
    cs = ax1.imshow(add_nans(zall), cmap=plt.cm.Greys, interpolation='none', aspect="auto",
        extent=[TRANGE[0], TRANGE[1]+TRANGE[0], 0, len(data.streams[GCaMP].filtered)])
    cbar = fig.colorbar(cs, pad=0.01, fraction=0.02)
    
    ax1.set_title('Individual z-Score Traces')
    ax1.set_ylabel('Trials')
    ax1.set_xlabel('Seconds Pellet retrieval')
    
    """
    **Plot the z-score trace for the 465 with std error bands**
    """
    
    # ax2 = fig.add_subplot(413)
    # p6 = ax2.plot(ts2, np.mean(zall, axis=0), linewidth=2, color='green', label='GCaMP')
    # p7 = ax2.fill_between(ts1, np.mean(zall, axis=0)+zerror
    #                       ,np.mean(zall, axis=0)-zerror, facecolor='green', alpha=0.2)
    # p8 = ax2.axvline(x=0, linewidth=3, color='slategray', label='pellet retrieval')
    # ax2.set_ylabel('z-Score')
    # ax2.set_xlabel('Seconds')
    # ax2.set_xlim(TRANGE[0], TRANGE[1]+TRANGE[0])
    # ax2.set_title('Pellet retrieval')
    
    ax2 = fig.add_subplot(413)
    p6 = ax2.plot(ts2, mean2(zall, axis=0), linewidth=2, color='green', label='GCaMP')
    p7 = ax2.fill_between(ts1, mean2(zall, axis=0)+zerror
                          ,mean2(zall, axis=0)-zerror, facecolor='green', alpha=0.2)
    p8 = ax2.axvline(x=0, linewidth=3, color='slategray', label='pellet retrieval')
    ax2.set_ylabel('z-Score')
    ax2.set_xlabel('Seconds')
    ax2.set_xlim(TRANGE[0], TRANGE[1]+TRANGE[0])
    ax2.set_title('Pellet retrieval')
    
    '==============================================='
    import os
    import numpy as np
    import pylab
    import itertools, collections
    import matplotlib.pyplot as plt
    import pandas as pd
    import bisect
    
    
    
    
    
    """
    Create the pandas dataframes to export as CSV files.
    """
    
    results = {'zScore':add_nans(zall), 'dFF':add_nans(dFF)}
    
    for stat in ['zScore', 'dFF']:
    
        # Create a pandas dataframe for the poke names and Z-score data.
        headers = [TTL['Type']+' TTL '+str(i) for i in range(1,len(zall)+1)]
        results[stat] = pd.DataFrame(np.transpose(results[stat]), columns=headers)
    
        # Make the indices the time stamps of each Z-score.
        #results[stat].index = np.arange(0, TRANGE[1], TRANGE[1]/(results[stat].shape[0]))
        results[stat].index = ts1
        results[stat].index.names = ['Time stamps (secs)']
        
        # Find the means of the rewarded and non-rewarded columns.
        mean_column = results[stat].mean(axis=1)
        results[stat].insert(0, 'Mean of TTLs', mean_column)
        
        # Find the average of each column.
        mean_headers = [col+' mean' for col in headers]
        results[stat+' means'] = pd.DataFrame(columns=mean_headers)
        for header in headers:
            results[stat+' means'][header+' mean'] = [results[stat][header].mean(axis=0)]
        results[stat+' means'].insert(0, 'Mean of TTLs', [float(results[stat+' means'].mean(axis=1))])
        
    # row1 = 'Area under curve of Z-Score vs time from '+str(TRANGE[0])+' to 0 seconds (before TTL)'
    # row2 = 'Area under curve of Z-Score vs time from 0 to '+str(TRANGE[0]+TRANGE[1])+' seconds (after TTL)'
    # results['AUC'] = pd.DataFrame({'Data':AUC},index=[row1,row2])
    
    'OUTPUT'
    
    filename = import_location
    
    #save_to_csv_zScore = True
        # If true then input the file name
    filename_zScore = os.path.join (
        #os.path.dirname (filename), # saves file in same directory
        export_location, # change the path to where you want files saved
        os.path.basename(filename) + "_zScore_" + TTL['Type'].replace(' ','_') + '_' + setup.replace(' ','_') + ".xlsx")
        #os.path.basename(filename) + "_zall_activeB" + ".csv")
        
    #save_to_csv_dFF = True
        # If true then input the file name
    filename_dFF = os.path.join (
        # os.path.dirname (filename), # saves file in same directory
        export_location,  # change the path to where you want files saved
        os.path.basename(filename) + "_dFF_" + TTL['Type'].replace(' ','_') + '_' + setup.replace(' ','_') + ".xlsx")
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
        with pd.ExcelWriter(filename_zScore) as writer:
            results['zScore'].to_excel(writer, sheet_name='All data')
            results['zScore means'].to_excel(writer, sheet_name='Means', index=False)
        #results['zScore'].to_csv(filename_zScore)
        #print("Printed zall CSV")
        
    if save_to_csv_dFF:
        #np.savetxt(filename_dFF, dFF, delimiter=",")
        with pd.ExcelWriter(filename_dFF) as writer:
            results['dFF'].to_excel(writer, sheet_name='All data')
            results['dFF means'].to_excel(writer, sheet_name='Means', index=False)
        #results['dFF'].to_csv(filename_dFF)
        #print("Printed dFF CSV")
    
    # if save_to_csv_ts1:
    #     np.savetxt(filename_ts1, ts1, delimiter=",")
    #     print("Printed time CSV")
    
    '==============================================='
     
    #plt.show()
    plt.close()
