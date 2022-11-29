def PhotometryPreprocessingv3(StartSecs, DUR, Hz, path, StartofBaseline, BaselineDur):
    
    #!/usr/bin/env python
    # coding: utf-8
    
    # <h2> This notebook achieves several things.    </h2>
    # 
    # 1. It combines the Bonsai output of 470nm and 415nm traces into one side-by-side sheet
    # 2. It crops to a start and end time (primarily to remove start/end artifacts due to plug/unplug)
    # 3. It fits the 470 to the 415 curve, plots these
    # 4. It subtracts fitted 470 from fitted 415 for the deltaF
    # 5. It has the user define a baseline period (start and end in seconds)
    # 6. It calculates the median of this baseline and subtracts this from the entire trace
    # 7. It calculates a % change in dF/F0
    # 8. It calculates a Z-score
    # 9. it combines the behaviour sheet and the camera sheet into one
    # 10. Together this produces 3 output files (ID_m19_SignalZscore; ID_SignalPercentDelta; ID_BehaviourTimeStamped), which are used as inputs for DLC heatmapping, or perievent analysis
    
    
    # Import dependencies
    import pandas as pd
    import glob
    import os
    import numpy as np
    import matplotlib.pyplot as plt
    import array as arr
    import scipy
    from scipy import signal as ss
    from matplotlib.pyplot import figure
    from sklearn.linear_model import HuberRegressor
    
    
    # ## All inputs for the entire sheet can be put in here, and then you should be able to run the whole notebook
    
    
    # Input start time and duration of TOTAL recording in seconds, for cropping
    # StartSecs = 180 ## if using this for peri-event, keep at 0. it will cut the file to this point. 
    # DUR = 3600 ## in seconds. make it at least your session, can be longer
    # Hz = 20 ## Capture rate on NPM system for the 470nm channel
    
    #INPUT YOUR FOLDER HERE with the 5 files (470, 410, cameracsv, DLCtracking, behaviour)
    # path = r'C:\Users\user\Documents\temp\demo' # use your path
    
    # DEFINE your baseline period, start + duration
    # StartofBaseline = 5155  # eventually i will get this to pull keydown information from bonsai
    # BaselineDur = 15 ## input duration in minutes
    BaselineDurFrame = BaselineDur*Hz*60
    
    # # 1) Combine the Bonsai output of 470nm and 415nm traces into one side-by-side sheet, crop to start/end, and plot the traces with a raw delta
    
    
    # Change second / duration input to frame number
    StartFrame = int((StartSecs*Hz)+1)
    EndFrame = DUR*Hz + StartFrame
    print("StartFrame is: ",StartFrame)
    print("EndFrame is: ",EndFrame)
    
    
    
    
    #GETS 470nm and 415nm in separate dataframes
    
    #Makes a new folder to save files in to
    savepath = os.path.join(path,"Pre-processing")
    if not os.path.exists(savepath):
        os.mkdir(savepath)
        
    #finds the 470 and 415 files in the oriignal folder
    df470path = glob.glob(path + "/*470*.csv")
    print("470 file is: ", df470path)
    df415path = glob.glob(path + "/*415*.csv")
    print("415 file is: ", df415path)
    
    ## Extracts animalID from filename of df470path
    str = ''.join(df470path)
    AnimalID = str.split('\\')[-1].split('_')[0]
    
    
    ## Gets info from 470nm sheet and crops to start time
    cols470 = ['FrameCounter', 'Timestamp', 'Region0G']
    csv470 = pd.read_csv(df470path[0], skiprows = range(1,StartFrame), usecols=cols470, )
    # puts that into a dataframe
    df470=pd.DataFrame(csv470.values, columns = ["Frame", "Timestamp", "470nm"])
    #crops to end frame
    df470 = df470[:EndFrame] 
    #df470
    
    ## Gets only the 415nm trace from 415nm sheet
    cols415 = ['Region0G']
    csv415 = pd.read_csv(df415path[0], skiprows = range(1,StartFrame), usecols=cols415)
    # puts that into a dataframe
    df415=pd.DataFrame(csv415.values, columns = ["415nm"])
    #crops to end frame
    df415 = df415[:EndFrame] 
    #df415
    
    # Combines the two dataframes so 470/415 are side-by-side
    combinedone=pd.concat([df470,df415], axis=1)
    #combinedone
    
    # Plot and save the trace you've segmented for delta, 470 and 415. For Delta only, use below function
    
    x = combinedone['Timestamp']
    y1 = combinedone['470nm']
    y2 = combinedone['415nm']
    
    plt.plot(x, y1, color='green', linewidth=0.2, label = "470")
    plt.plot(x, y2, color='black', linewidth=0.2, label = "415")
    plt.legend(['470', '415'], loc='best')
    plt.xlabel('Time(s)')
    plt.ylabel('Activity(AU)')
    
    ## Save the above plot, input what you want to call it
    #filename = input("Save as what? ")
    #current_path = os.getcwd()
    ## Save the above plot
    plt.savefig(savepath+'/'+AnimalID + "_" + "_RawTraces.svg", dpi=600, orientation='landscape')
    
    print('Success! Saved as {}'.format(savepath+'/'+AnimalID + "_RawTraces.svg"))
    
    plt.show()
    
    # Write combinedone df to csv, using animal ID
    #combinedone.to_csv(savepath+'/'+AnimalID + "_RawTraces.csv", encoding='utf-8')
    #print('Success! Saved as {}'.format(savepath+'/'+AnimalID + "_RawTraces.csv"))
    
    
    
    
    
    # # 2) Fit 470 to 415 curve
    
    
    ##### For now, if they are uneven # of frames, you'll need to manually remove the last frame from the one with the highest
    #### Adjust this to take these values from the newly created m19_rawtraces.csv
    raw_signal = df470['470nm']
    raw_reference = df415['415nm']
    
    print("Signal, number of frames: ",raw_signal.shape)
    print("Control, number of frames: ",raw_reference.shape)
    
    
    
    ##Plotting and comparing signal and reference, no manipulations occur here, just plots the same data from the above figure, but on different scales
    fig = plt.figure(figsize=(16, 10))
    ax1 = fig.add_subplot(211)
    ax1.plot(raw_signal,'blue',linewidth=1.5)
    ax2 = fig.add_subplot(212)
    ax2.plot(raw_reference,'purple',linewidth=1.5)
    ax1.set_title("Signal")
    ax2.set_title("Reference")
    
    plt.show()
    
    
    
    ##a linear regression of reference and signal
    ##we use the huberregressor as it is robust to large and infrequent outliers, and often not different to Linear Regression.
    
    model = HuberRegressor(epsilon=1)
    n=len(raw_reference)
    model.fit(raw_reference.values.reshape(n, 1), raw_signal.values.reshape(n,1))
    
    
    
    ##we chart the relationship between reference and signal and show the linear fit. This is because any highly correlated activity is likely noise due to fibre bending, etc.
    
    fig = plt.figure(figsize=(16, 8))
    ax1 = fig.add_subplot(111)
    ax1.plot(raw_reference,raw_signal, 'b.')
    
    plt.xlabel("isosbestic / 415nm")
    plt.ylabel("470nm")
    arr = model.predict(raw_reference.values.reshape(n, 1))
    ax1.plot(raw_reference,arr , 'r--',linewidth=1.5)
    
    
    
    ##the aligned control (arr) is the (control + y_intercept) * gradient of the control regressed to the signal
    
    fig = plt.figure(figsize=(16, 8))
    ax1 = fig.add_subplot(111)
    ax1.plot(raw_signal)
    ##yellow is signal aligned noise
    ax1.plot(arr,'yellow')
    
    
    
    
    ##Now that we have the aligned control, we subtract it from the signal
    res = np.subtract(raw_signal, arr)
    ##and then divide the signal by the control
    norm_data = np.divide(res, arr)
    
    
    
    
    fig = plt.figure(figsize=(16, 8))
    ax1 = fig.add_subplot(111)
    ax1.set_title("Cleaned Signal")
    ax1.plot(norm_data, color='r')
    
    
    
    
    
    # Plot and save the trace you've segmented for delta, 470 and 415. For Delta only, use below function
    
    x = combinedone['Timestamp']
    y1 = raw_signal
    y2 = raw_reference
    y3 = norm_data
    
    
    plt.plot(x, y1, color='purple', linewidth=0.5, label = "Signal")
    plt.plot(x, y2, color='red', linewidth=0.5, label = "Control")
    plt.plot(x, y3, color='green', linewidth=0.5, label = "delta")
    plt.legend(['Signal', 'Control', 'delta'], loc='best')
    plt.xlabel('Time(s)')
    plt.ylabel('Activity(AU)')
    
    ## Save the above plot, input what you want to call it
    #filename = input("Save as what? ")
    #current_path = os.getcwd()
    ## Save the above plot
    plt.savefig(savepath+'/'+AnimalID + "_" + "_NormalisedTrace.svg", dpi=600, orientation='landscape')
    
    print('Success! Saved as {}'.format(savepath+'/'+AnimalID + "_NormalisedTrace.svg"))
    
    plt.show()
    
    ## Saves normalised data into a sheet with the raw data too
    norm_data=pd.DataFrame(norm_data.values, columns = ["norm_data"])
    
    # Combines the dataframes so raw and normalised data are together
    combinedtwo=pd.concat([combinedone,norm_data], axis=1)
    
    
    # Write combinedtwo df to csv, using animal ID
    #combinedtwo.to_csv(savepath+'/'+AnimalID + "_NormalisedTraces.csv", encoding='utf-8')
    #print('Success! Saved as {}'.format(savepath+'/'+AnimalID + "_NormalisedTraces.csv"))
    
    
    # # 3) Define Baseline period, removes it from dF to get dF-f0, to be used for %dF-F0/F0 and z-scores
    # 
    
    
    
    # As in step 1 you have defined the start point, this takes that into consideration
    BaselineStartFrame = StartofBaseline - StartFrame
    #print(BaselineStartFrame)
    BaselineFinalFrame = BaselineDurFrame + BaselineStartFrame
    
    ## Defines Fzero
    dfBase = pd.DataFrame(norm_data)
    dfBase = dfBase[BaselineStartFrame:BaselineFinalFrame] 
    #print(dfBase)
    #Fzero or F0 is the median of the defined baseline period. this is for subtracting from each dF value
    Fzero = dfBase.median()
    print("Baseline Fluorescence [Fzero] is: ", Fzero)
    
    # Calculates dF-Fzero
    dfminusbaseline = norm_data - Fzero
    
    dfminusbaseline=pd.DataFrame(dfminusbaseline.values, columns = ["dF-Fzero"])
    #print("df minus baseline is", dfminusbaseline)
    
    # Combines the two dataframes so 470/415 are side-by-side
    combinedthree=pd.concat([combinedtwo, dfminusbaseline], axis=1)
    #combinedthree
    
    
    # # 4) Uses dF-Fzero to calculate %dF  as (dF-F0/F0) and z-score
    # 
    
    
    
    #Fzero is defined above, as is dfminusebaseline
    pctdFF = np.divide(dfminusbaseline, Fzero)
    pctdFF = pctdFF.rename(columns={'dF-Fzero': 'pctdFF'})
    pctdFF
    
    # Plot and save the trace you've segmented for delta, 470 and 415. For Delta only, use below function
    
    x = combinedthree['Timestamp']
    y1 = pctdFF
    
    
    plt.plot(x, y1, color='purple', linewidth=0.5, label = "%dF")
    plt.legend([AnimalID+' %dF'], loc='best')
    plt.xlabel('Time(s)')
    plt.ylabel('%dF')
    
    
    ## Save the above plot
    plt.savefig(savepath+'/'+AnimalID + "_" + "_pctdF.svg", dpi=600, orientation='landscape')
    
    print('Percent dFF image saved as {}'.format(savepath+'/'+AnimalID + "_pctdF.svg"))
    
    plt.show()
    
    
    
    
    ### calculates z-score
    
    ZdF = scipy.stats.zscore(dfminusbaseline, axis=0, ddof=0, nan_policy='propagate')
    ZdF = pd.DataFrame({"signal":ZdF.reshape(len(ZdF))}, index=np.arange(len(ZdF)))
    #ZdF = ZdF.rename(columns={'dF-Fzero': 'signal'})
    # Plot and save the trace you've segmented for delta, 470 and 415. For Delta only, use below function
    
    x = combinedthree['Timestamp']
    y1 = ZdF
    
    
    plt.plot(x, y1, color='brown', linewidth=0.5, label = "Z-score")
    plt.legend([AnimalID+' Z-score'], loc='best')
    plt.xlabel('Time(s)')
    plt.ylabel('Z-score')
    
    ## Save the above plot, input what you want to call it
    #filename = input("Save as what? ")
    #current_path = os.getcwd()
    ## Save the above plot
    plt.savefig(savepath+'/'+AnimalID + "_" + "_Z-score.svg", dpi=600, orientation='landscape')
    
    print('Z-score image saved as {}'.format(savepath+'/'+AnimalID + "_Z-score.svg"))
    
    plt.show()
    
    
    ## SAVES FINAL OUTPUT CSV to be used with other tools
    # f = final
    
    fTimestamp = combinedthree['Timestamp']
    fFrame = combinedthree['Frame']
    f470 = combinedthree['470nm']
    f415 = combinedthree['415nm']
    
    # Write FinalOutput to csv, using animal ID
    #FullOutput=pd.concat([fFrame, fTimestamp, f470, f415, pctdFF, ZdF], axis=1)
    #FullOutput.to_csv(savepath+'/'+AnimalID + "_FullOutput.csv", encoding='utf-8', index=False)
    #print('Success! Saved as {}'.format(savepath+'/'+AnimalID + "_FullOutput.csv"))
    
    # Write zscore, timestamp, frame to csv for peri-event
    #ZdF = ZdF.rename(columns={'Z-score': 'signal'})
    SignalZscore=pd.concat([fFrame, fTimestamp, ZdF], axis=1)
    SignalZscore.to_csv(savepath+'/'+AnimalID + "_SignalZscore.csv", encoding='utf-8', index=False)
    print('Z-score signal saved as {}'.format(savepath+'/'+AnimalID + "_SignalZscore.csv"))
    
    # Write zscore, timestamp, frame to csv for peri-event
    pctdFF = pctdFF.rename(columns={'pctdFF': 'signal'})
    SignalPercentDelta=pd.concat([fFrame, fTimestamp, pctdFF], axis=1)
    SignalPercentDelta.to_csv(savepath+'/'+AnimalID + "_SignalPercentDelta.csv", encoding='utf-8', index=False)
    print('Percent dFF signal saved as {}'.format(savepath+'/'+AnimalID + "_SignalPercentDelta.csv"))
    
    
    # # 5) This combines your behaviour and cameracsv into one sheet for the peri-event
    
    
    #GETS 470nm and 415nm in separate dataframes
    
    behaviourpath = glob.glob(path + "/*behaviour*.csv")
    print("Behaviour path: ", behaviourpath)
    camerapath = glob.glob(path + "/*csvforvideo*.csv")
    print("Camera path: ", camerapath)
    
    ## Gets info from behaviour sheet, and camera sheet, and collates them
    behaviourcols = ['frame', 'behaviour', 'hits']
    behaviourcsv = pd.read_csv(behaviourpath[0], usecols=behaviourcols)
    
    # puts that into a dataframe
    dfbehaviour=pd.DataFrame(behaviourcsv.values, columns = ["frame", "duration", "hits"])
    
    # Gets info from behaviour sheet, and camera sheet, and collates them
    cameratimestamps = pd.read_csv(camerapath[0], header=None, usecols=[0])
    
    # puts that into a dataframe
    dfcameratimestamps=pd.DataFrame(cameratimestamps.values, columns = ["Timestamp"])
    
    # Combines the two dataframes so behaviour frames and camera timestmaps are side-by-side
    behaviourtimestamped=pd.concat([dfbehaviour,dfcameratimestamps], axis=1)
    
    # Write combinedone df to csv, using animal ID
    behaviourtimestamped.to_csv(savepath+'/'+AnimalID + "_BehaviourTimeStamped.csv", encoding='utf-8', index=False)
    print('Files combined as {}'.format(savepath+'/'+AnimalID + "_BehaviourTimeStamped.csv"))





