def FiPhoEpocAveraging_between_events(inputs):

    """
    Fiber Photometry Epoch Averaging Example
    
    This example goes through fiber photometry analysis using techniques such as data smoothing, bleach detrending, and z-score analysis. <br>
    The epoch averaging was done using `epoc_filter`.
    
    Author Contributions:
    TDT, David Root, and the Morales Lab contributed to the writing and/or conceptualization of the code.<br>
    The signal processing pipeline was inspired by the workflow developed by David Barker et al. (2017) for the Morales Lab.<br>
    The data used in the example were provided by David Root.
    
    Author Information:
    David H. Root
    Assistant Professor
    Department of Psychology & Neuroscience
    University of Colorado, Boulder
    Lab Website: https://www.root-lab.org
    david.root@colorado.edu
     
    About the authors:
    The Root lab and Morales lab investigate the neurobiology of reward, aversion, addiction, and depression.
    """
    
    # Import the tdt package and other python packages we care about.
    #import the read_block function from the tdt package
    #also import other python packages we care about
    import numpy as np
    from sklearn.metrics import auc
    import matplotlib.pyplot as plt  # standard Python plotting library
    import scipy.stats as stats
    import tdt
    
    # Jupyter has a bug that requires import of matplotlib outside of cell with 
    # matplotlib inline magic to properly apply rcParams
    import matplotlib 
    matplotlib.rcParams['font.size'] = 16 #set font size for all plots
    
    # This error comes up when using the artifact removal section of the code.
    # It happens when data.streams[GCaMP].filtered == [].
    # If ARTIFACT == np.inf, this section of the code is skipped anyway.
    np.warnings.filterwarnings('ignore', category=np.VisibleDeprecationWarning)
    
    # Setup the variables for the data you want to extract
    # We will extract two different stream stores surrounding the 'PtAB' epoch event. We are interested in a specific event code for the shock onset.
    
    REF_EPOC = 'Analyse_this_event' #event store name. This holds behavioral codes that are 
    # read through ports A & B on the front of the RZ
    SHOCK_CODE = [] #shock onset event code we are interested in
    
    # make some variables up here to so if they change in new recordings you won't
    # have to change everything downstream
    ISOS = inputs['ISOS'] # 405nm channel. Formally STREAM_STORE1 in maltab example
    GCaMP = inputs['GCaMP'] # 465nm channel. Formally STREAM_STORE2 in maltab example
    BASELINE_PER = inputs['Baseline period'] # baseline period within our window
    ARTIFACT = inputs['Artifact RL'] # optionally set an artifact rejection level
    
    if inputs['Baseline type'] == 'Whole recording':
        TRANGE = [0] # window size [start time relative to epoc onset, window duration]
    elif inputs['Baseline type'] == 'Specific':
        TRANGE = [inputs['Baseline period'][0]]
    
    #call read block - new variable 'data' is the full data structure
    data_full = inputs['Tank']
    
    # Use epoc_filter to extract data around our epoc event
    # Using the 't' parameter extracts data only from the time range around our epoc event.<br>
    # Use the 'values' parameter to specify allowed values of the REF_EPOC to extract.<br>
    # For stream events, the chunks of data are stored in cell arrays structured as `data.streams[GCaMP].filtered`
    data = tdt.epoc_filter(data_full, REF_EPOC, t=TRANGE, values=SHOCK_CODE)
    
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
    
    # Optionally remove artifacts. If any waveform is above ARTIFACT level, or
    # below -ARTIFACT level, remove it from the data set.
    total1 = np.size(data.streams[GCaMP].filtered)
    total2 = np.size(data.streams[ISOS].filtered)
    
    # List comprehension checking if any single array in 2D filtered array is > Artifact or < -Artifact
    data.streams[GCaMP].filtered = [x for x in data.streams[GCaMP].filtered 
                                    if not np.any(x > ARTIFACT) or np.any(x < -ARTIFACT)]
    data.streams[ISOS].filtered = [x for x in data.streams[ISOS].filtered 
                                   if not np.any(x > ARTIFACT) or np.any(x < -ARTIFACT)]
    
    # Get the total number of rejected arrays
    bad1 = total1 - np.size(data.streams[GCaMP].filtered)
    bad2 = total2 - np.size(data.streams[ISOS].filtered)
    total_artifacts = bad1 + bad2
    
    # Downsample and average 10x via a moving window mean
    N = inputs['N'] # Average every 10 samples into 1 value
    F405 = []
    F465 = []
    for lst in data.streams[ISOS].filtered: 
        small_lst = []
        for i in range(0, len(lst), N):
            small_lst.append(np.mean(lst[i:i+N-1])) # This is the moving window mean
        F405.append(small_lst)
    
    for lst in data.streams[GCaMP].filtered: 
        small_lst = []
        for i in range(0, len(lst), N):
            small_lst.append(np.mean(lst[i:i+N-1]))
        F465.append(small_lst)
        
    # Additional section.
    lst = data_full.streams[ISOS].data
    F405_full = []
    for i in range(0, len(lst), N):
        F405_full.append(np.mean(lst[i:i+N-1])) # This is the moving window mean
        
    lst = data_full.streams[GCaMP].data
    F465_full = []
    for i in range(0, len(lst), N):
        F465_full.append(np.mean(lst[i:i+N-1])) # This is the moving window mean
    
    #Create a mean signal, standard error of signal, and DC offset
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
    
    meanF405 = mean1(F405, axis=0)
    stdF405 = std1(F405, axis=0) / np.sqrt(len(data.streams[ISOS].filtered))
    dcF405 = np.mean(meanF405)
    meanF465 = mean1(F465, axis=0)
    stdF465 = std1(F465, axis=0) / np.sqrt(len(data.streams[GCaMP].filtered))
    dcF465 = np.mean(meanF465)
    
    # Additional section.
    list_meanF405 = [np.mean(list1) for list1 in F405]
    list_meanF465 = [np.mean(list1) for list1 in F465]
    list_stdF405  = [np.std(list1)  for list1 in F405]
    
    # ## Plot epoc averaged response
    
    # Create the time vector for each stream store
    ts1 = TRANGE[0] + np.linspace(1, len(meanF465), len(meanF465))/data.streams[GCaMP].fs*N
    ts2 = TRANGE[0] + np.linspace(1, len(meanF405), len(meanF405))/data.streams[ISOS].fs*N
    
    # Subtract DC offset to get signals on top of one another
    meanF405 = meanF405 - dcF405
    meanF465 = meanF465 - dcF465
    
    # Start making a figure with 4 subplots
    # First plot is the 405 and 465 averaged signals
    fig = plt.figure(figsize=(9, 14))
    ax0 = fig.add_subplot(411) # work with axes and not current plot (plt.)
    
    # Plotting the traces
    p1, = ax0.plot(ts1, meanF465, linewidth=2, color='green', label='GCaMP')
    p2, = ax0.plot(ts2, meanF405, linewidth=2, color='blueviolet', label='ISOS')
    
    # Plotting standard error bands
    p3 = ax0.fill_between(ts1, meanF465+stdF465, meanF465-stdF465,
                          facecolor='green', alpha=0.2)
    p4 = ax0.fill_between(ts2, meanF405+stdF405, meanF405-stdF405,
                          facecolor='blueviolet', alpha=0.2)
    
    # Plotting a line at t = 0
    p5 = ax0.axvline(x=0, linewidth=3, color='slategray', label='Event onset')
    
    # Finish up the plot
    ax0.set_xlabel('Time (seconds)')
    unit = '(mV)'
    ax0.set_ylabel('Signal '+unit)
    ax0.set_title('Raw traces for '+inputs['Analysis name'])
    ax0.legend(handles=[p1, p2, p5], loc='upper right')
    ax0.set_ylim(min(np.min(meanF465-stdF465), np.min(meanF405-stdF405)),
                 max(np.max(meanF465+stdF465), np.max(meanF405+stdF405)))
    ax0.set_xlim(TRANGE[0], TRANGE[1]+TRANGE[0]);
    
    # Fitting 405 channel onto 465 channel to detrend signal bleaching
    # Scale and fit data. Algorithm sourced from Tom Davidson's Github:
    # https://github.com/tjd2002/tjd-shared-code/blob/master/matlab/photometry/FP_normalize.m
    
    # Getting the z-score and standard error
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
    
    Y_fit_all = []
    Y_dF_all = []
    dFF = []
    for x, y in zip(F405, F465):
        x = np.array(x)
        y = np.array(y)
        bls = np.polyfit(x, y, 1)
        fit_line = np.multiply(bls[0], x) + bls[1]
        Y_fit_all.append(fit_line)
        Y_dF_all.append(y-fit_line)
        dFF.append((y-fit_line)/fit_line)
        
    dFFerror = std2(dFF, axis=0)/np.sqrt(np.size(dFF, axis=0))
    
    # Additional section.
    Y_dF_all_full = []
    x = np.array(F405_full)
    y = np.array(F465_full)
    bls = np.polyfit(x, y, 1)
    fit_line = np.multiply(bls[0], x) + bls[1]
    Y_dF_all_full = y-fit_line
    
    if inputs['Baseline type'] == 'Specific':
        # Getting the z-score and standard error
        zall = []
        for dF in Y_dF_all: 
           ind = np.where((np.array(ts2)<BASELINE_PER[1]) & (np.array(ts2)>BASELINE_PER[0]))
           zb = np.mean(dF[ind])
           zsd = np.std(dF[ind])
           zall.append((dF - zb)/zsd)
           
    elif inputs['Baseline type'] == 'Whole recording':
        # Getting the z-score and standard error
        zall = []
        for dF in Y_dF_all: 
           zb = np.mean(Y_dF_all_full)
           zsd = np.std(Y_dF_all_full)
           zall.append((dF - zb)/zsd)
       
    zerror = std2(zall, axis=0)/np.sqrt(np.size(zall, axis=0))
    
    # # ## Plot the dFF trace for the 465 with std error bands
    
    ax25 = fig.add_subplot(412)
    p6 = ax25.plot(ts2, mean2(dFF, axis=0), linewidth=2, color='green', label='GCaMP')
    p7 = ax25.fill_between(ts1, mean2(dFF, axis=0)+dFFerror
                          ,mean2(dFF, axis=0)-dFFerror, facecolor='green', alpha=0.2)
    p8 = ax25.axvline(x=0, linewidth=3, color='slategray', label='Shock Onset')
    ax25.set_ylabel('dFF')
    ax25.set_xlabel('Time (secs)')
    ax25.set_xlim(TRANGE[0], TRANGE[1]+TRANGE[0])
    ax25.set_title('dFF traces for '+inputs['Analysis name'])
    
    # ## Plot the z-score trace for the 465 with std error bands
    
    ax2 = fig.add_subplot(413)
    p6 = ax2.plot(ts2, mean2(zall, axis=0), linewidth=2, color='green', label='GCaMP')
    p7 = ax2.fill_between(ts1, mean2(zall, axis=0)+zerror
                          ,mean2(zall, axis=0)-zerror, facecolor='green', alpha=0.2)
    p8 = ax2.axvline(x=0, linewidth=3, color='slategray', label='pellet retrieval')
    ax2.set_ylabel('z-Score')
    ax2.set_xlabel('Seconds')
    ax2.set_xlim(TRANGE[0], TRANGE[1]+TRANGE[0])
    ax2.set_title('Pellet retrieval')
    
    plt.close()
    
    # ## Heat Map based on z score of 405 fit subtracted 465
    # This function takes a list of numpy arrays with unequal lengths, and adds nans
    # to make each array equal length.
    def add_nans(list_of_lists):
        max_len = max([len(list1) for list1 in list_of_lists])
        list_of_lists = [np.concatenate((list1, np.array((max_len-len(list1))*[np.nan]))) for list1 in list_of_lists]
        return(list_of_lists)
    
    ax1 = fig.add_subplot(414)
    cs = ax1.imshow(add_nans(zall), cmap=plt.cm.Greys, interpolation='none', aspect="auto",
        extent=[TRANGE[0], TRANGE[1]+TRANGE[0], 0, len(data.streams[GCaMP].filtered)])
    cbar = fig.colorbar(cs, pad=0.01, fraction=0.02)
    
    ax1.set_title('Individual z-Score Traces')
    ax1.set_ylabel('Trials')
    ax1.set_xlabel('Seconds Pellet retrieval')
    
    plt.close() # Suppress figure output again
    
    fig.tight_layout()
    plt.close()
    
    inputs['Tank'] = data
    outputs = {'zScore':zall, 'dFF':dFF, 'ISOS':F405, 'GCaMP':F465, 
               'Timestamps':ts1, 'Figure':fig}
    return(inputs, outputs)

