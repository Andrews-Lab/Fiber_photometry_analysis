def PhotometryPeriEventv3(pre_window_secs, post_window_secs, path, photom_hz):
    
    #!/usr/bin/env python
    # coding: utf-8
    
    # <h1> Photometry analysis
    
    
    import pandas as pd
    import numpy  as np
    import glob
    import os
    
    
    # <h2> All inputs for the entire sheet can be put in here, and then you should be able to run the whole notebook
    #  </h2>
    # 
    
    
    ## Define your peri-event window
    # pre_window_secs = 5.0 ### How long *before* the behaviour onset do you want to extract? 
    # post_window_secs = 5.0 ### How long *after* the behaviour onset do you want to extract?
    
    ####### The above input MUST be to one decimal point, e.g. 5.0, 3.5, 2.1. It will not work as "5"########
    
    # What is the folder of your initial 5 files? This is the same folder input as the pre-processing book, and it will then look for your "pre-processing" folder which was created
    # path = r'C:\Users\user\Documents\temp\demo' # use your path
    print("Source folder is: ", path)
    
    # What is the capture rate of the photometry system? 
    # If NPM is capturing at 40fps, and you're 1:1 interleaved with 470/415, then put in 20. This is only used to convert your peri-event seconds into # of frames.
    # photom_hz = 20 
    
    
    # <h2> 1) Open and combine pre-processed data </h2>
    #     <b1> Here we will combine photometry and behaviour analysis files and pull out the information we need for analysis </b1>
    
    # <b1> Open your data files</b1>
    
    
    # finds the pre-procesisng folder created in the pre-processing notebook, and uses the files htere
    preprocessingpath = os.path.join(path,"Pre-processing")
    print("Files being used are from: ", preprocessingpath)
    
    
    #Makes a new folder to save files in to
    savepath = os.path.join(path,"Peri-Event-Analysed")
    if not os.path.exists(savepath):
        os.mkdir(savepath)
    
    print("Peri-event files will be saved to: ", savepath)
    
    
    behaviourpath = glob.glob(preprocessingpath + "/*Behaviour*.csv")
    print("Behaviour file is: ", behaviourpath)
    photompath = glob.glob(preprocessingpath + "/*SignalZscore*.csv")
    print("Signal file is: ", photompath)
    ##open the csv of your behaviour file and remove rows that are only nan (added in by excel sometimes)
    behaviour = pd.read_csv(behaviourpath[0]).dropna(axis=0, how='all')
    #print("behaviour is: ", behaviour)
    photom = pd.read_csv(photompath[0]).dropna(axis=0, how='all')
    #print("photom is: ", photom)
    #Extract Animal ID from file name
    concat_str = ''.join(behaviourpath)
    AnimalID = concat_str.split('\\')[-1].split('_')[0]
    print("Animal Name: ", AnimalID)
    
    
    # # Here we create a function that combines a photometry trace and morlog file, since we have timestamp information in both files nothing else is needed <b1/>
    
    
    ## This code combines photometry trace and behaviour file, and theiri respective timestamps. 
    ## Due to these timestamps, we don't actually need the Photometry or Camera frame rates from above.Photometry hz is used for peri-event conversions though.
    photom = photom.rename(columns={'Frame':'photom_frame'})
    behaviour.set_index('Timestamp', inplace=True)
    photom.set_index('Timestamp', inplace=True)
    photom = behaviour.join(photom, how='outer')
    photom = photom.rename(columns={'duration':'behaviour'})
    photom.behaviour = photom.behaviour.fillna('Nothing')
    ##photom variable is now a combined dataframe with synced traces, behaviours and hits
    ##display is an alternative for print that keeps pandas dataframes looking pretty
    #display(photom)
    print("Success! Photometry trace and behaviour file and their timestamps have been combined in to one data frame!")
    
    
    # # 2) Peri-Event extraction 
    # 
    #     Extract the photometry traces X seconds either side of the onset of each behaviour ("hits")
    
    # # Get pre and post window length in seconds
    
    
    ##convert secs to photometry capture frames
    pre_window_frames  = pre_window_secs * photom_hz
    post_window_frames = post_window_secs * photom_hz
    print("Pre (s): ", pre_window_secs)
    print("Post (s): ", post_window_secs)
    print("Pre (frames): ", pre_window_frames)
    print("Post (frames): ", post_window_frames)
    pre_str = str(pre_window_secs)
    post_str = str(post_window_secs)
    ##Create output file name
    outputname = savepath+'/'+AnimalID+'_PeriEvent_'+pre_str+'s-pre_'+post_str+'s-post'+'.xlsx'
    
    
    # # Extract photometry signal for every behaviour, based on pre- and post-frames, and save as multi-sheet excel
    
    
    def df_style(val):
                  return 'font-weight: bold'
    ##create an empty dict to store photometry data for later visualisation
    photom_hit_window = dict()
    ##create a sequence from prewindow photom frames to post window photom frames
    pre_post_secs = np.arange(-pre_window_frames, post_window_frames)/(photom_hz)
    ##convert to ms
    pre_post_secs*=1000
    ##Find all the unique behaviours present in the file
    unique_behaviours = photom['behaviour'].unique()
    ##remove Nothing
    unique_behaviours = unique_behaviours[unique_behaviours!='Nothing']
    ##write each behaviour as an excel sheet
    with pd.ExcelWriter(outputname) as writer:
        for beh in unique_behaviours:
            ##get index of current behaviour
            indexes = np.where(photom.hits==beh)[0]
            time_seconds = (indexes/photom_hz)
            onset_mins = time_seconds/60
            ##get create indexes for each window
            photom_hit_indexes = [np.arange(i-pre_window_frames, i+post_window_frames) for i in indexes]
            ##get photom signal for each window (keeping windows seperate)
            photom_traces = [photom.signal.iloc[i].values for i in photom_hit_indexes]
            ##organise each photom signal into rows with columns labeled as ms pre/post behaviour
            trace_df = pd.DataFrame(photom_traces, columns = ['{} ms'.format(i) for i in pre_post_secs])
            trace_df.insert(0,"onset_seconds", time_seconds)
            trace_df.insert(1,"onset_mins", onset_mins)
            #row title should be Event no. #1, #2, etc
            trace_df.index+=1
            trace_df.index.name=None
            photom_hit_window[beh] = trace_df
            trace_df.index.name = 'Event No.'
            ##write new sheet
            trace_df = trace_df.style.applymap(df_style, subset=["onset_seconds", "onset_mins"])
            trace_df.to_excel(writer, beh)
            ##add to dictionary for later visualisation
    
            
            
    ##this saves as xlsx
        writer.save()
        print("Saved as: ", outputname)
    
    
    ## Check what behaviorus you have in your file. This list is used in the above script
    unique_behaviours
    
    
    # # Plot your behaviours here and save as svg
    
    
    import matplotlib.pyplot as plt
    import numpy as np
    import matplotlib
    import seaborn as sns
    import emoji
    matplotlib.rcParams['figure.dpi']= 500
    sns.set()
    
    def plot_behaviour_trace(photom_hit_window, cur_behaviour, pre_post_secs):
        ##get index values (negative to positive ms)
        x = photom_hit_window[cur_behaviour].columns
        timestampnames = [str(i)+" ms" for i in pre_post_secs]
    
        ##get the mean line of all traces
        mean = photom_hit_window[cur_behaviour][timestampnames].mean()
        ##get the standard error of the mean
        sem = photom_hit_window[cur_behaviour][timestampnames].sem()
        ##plot each individual trace
        [plt.plot(pre_post_secs, i, 'lightblue', label='mean_1', linewidth=1,  zorder=1) for i in photom_hit_window[cur_behaviour][timestampnames].values]
        ##shade the SEM area
        plt.fill_between(pre_post_secs, mean - sem, mean + sem, color='orange', alpha=0.4, zorder=2)
        ##plot the mean trace
        plt.plot(pre_post_secs, mean, 'darkblue', label='mean_1', zorder=3)  
        ##get current axis, i dont really get it but it lets you change chart settings
        ax = plt.gca()
        ##make the chart background white
        ax.set_facecolor('white')
        ##set the little ticks to be black
        ax.spines['bottom'].set_color('0.5')
        ax.spines['top'].set_color('0.5')
        ax.spines['right'].set_color('0.5')
        ax.spines['left'].set_color('0.5')
        ##remove the background grid
        ax.grid(False)
        ##remove unneccessary white space
        plt.tight_layout(pad=0)
        ##get the largest value in the dataset
        max_of_all = photom_hit_window[cur_behaviour][timestampnames].values.max()
        ##draw a line to this point (though im pretty sure this draws an infinite line anyway)
        plt.axline((0,0), (0,max_of_all), linestyle='dotted', color='red')
        ##make sure ticks are on the bottom and left
        plt.tick_params(bottom=True, left=True)
        ##set the plot title to the current behaviour in all caps
        plt.title(cur_behaviour.upper(), fontweight='bold')
        ##manually create a legend using the same lines i used for each type of trace
        lines = [matplotlib.lines.Line2D((0,1),(0,1),linestyle = 'dotted', color='red'), matplotlib.lines.Line2D((0,1),(0,1), color='darkblue'), matplotlib.lines.Line2D((0,1),(0,1), color='orange'),  matplotlib.lines.Line2D((0,1),(0,1), color='lightblue')]
        ##draw the legend and place it in the top right
        ax.legend(lines, ['Behaviour', 'Mean', 'SEM', 'Trace'], loc='upper right')
        ##X axis label
        matplotlib.pyplot.xlabel('Time (ms) from behaviour onset',  fontweight='bold')
        ##Y axis label
        matplotlib.pyplot.ylabel('Z-score',  fontweight='bold') #''% ∆f / f'
        ##Save figure as an SVG
        plt.savefig(savepath+'/'+AnimalID+"_"+cur_behaviour+"_trace.svg", bbox_inches='tight')
        print("Peri-event trace figure saved as: ", savepath+'/'+AnimalID+"_"+cur_behaviour+"_trace.svg")
        ##Display current figure in the notebook
        plt.show()
        
    for beh in unique_behaviours:
        plot_behaviour_trace(photom_hit_window, beh, pre_post_secs)
    
    
    # # Plot a heatmap of each behaviour and save as svg
    
    
    def add_margin(ax,xm=0.1,ym=0.05):
        """adds a margin to a pyplot"""
        ##get min and max position of the X axis
        xlim = ax.get_xlim()
        ##get min and max position of the Y axis
        ylim = ax.get_ylim()
        ##change min and max X by 0.1 (adding a 10% margin)
        ax.set_xlim(xlim[0]-xm,xlim[1]+xm)
        ##change min and max Y by 0.05 (adding a 5% margin)
        ax.set_ylim(ylim[0]+ym,ylim[1]-ym)
        
    def zero_x_labels(pre_window_secs, post_window_secs, heatmap):
        """aligns labels and data, ensuring a label at zero"""
        ##arange from the pre_window to zero, with increment of one second
        neg_ticks = np.arange(-pre_window_secs, 0)
        ##arange from zero to the post_window, with increment of one second
        pos_ticks = np.arange(0, post_window_secs)
        ##append these to each other
        pos_ticks = np.append(pos_ticks, post_window_secs)
        ##convert to ms
        labels = np.append(neg_ticks, pos_ticks)*1000
        ##ticks are integer position of the data and need to line up with the correct labels
        ticks=labels+abs(labels[0])
        ticks/=ticks[-1]
        ticks*=heatmap.shape[1]-1
        return labels, ticks
    
    def create_heatmap(photom_hit_window, behaviour, pre_window_secs, post_window_secs, pre_post_secs):
        """creates a heatmap of data"""
        ##get all the traces for the current behaviour
        timestampnames = [str(i)+" ms" for i in pre_post_secs]
    
        heatmap = photom_hit_window[behaviour][timestampnames].values
        ##align X labels and data 
        labels, ticks = zero_x_labels(pre_window_secs, post_window_secs, heatmap)
        ##get the average trace
        avg_heat = np.mean(heatmap, axis=0).reshape(1, np.shape(heatmap)[1])
        ##if heatmap is shape is under 100, increase each row height by 1 pix until it is >=100
        heatmap_scale = np.ceil(100/heatmap.shape[0])
        heatmap = np.repeat(heatmap,heatmap_scale, axis=0)
        ##create a blank canvas with three plots (3 rows 1 column), the ratio is such that the first two rows will always be the same size
        print(f"ratio: {(10, 10, np.shape(heatmap)[0])}")
        f, axs = plt.subplots(figsize=(7.5, len(heatmap)*0.1), nrows=3, ncols=1, sharex=True, gridspec_kw={'height_ratios': [10, 10, np.shape(heatmap)[0]]})
    
    
        ##set the background of all plots to white
        for ax in axs: ax.set_facecolor('white')
        ##plot the average but make sure the min and max colours are the same as the main plot
        sns.heatmap(avg_heat,linewidths=0, cmap="plasma", rasterized=True, cbar=True, ax=axs[0],cbar_kws={'aspect':1.25}, vmin=np.min(heatmap), vmax=np.max(heatmap))
        ##now plot it again but let the colours scale to min and max "normalising" the plot
        sns.heatmap(avg_heat,linewidths=0, cmap="plasma", rasterized=True, cbar=True, ax=axs[1],cbar_kws={'aspect':1.25})
        ##now plot all the individual traces
        sns.heatmap(heatmap,linewidths=0, cmap="plasma", rasterized=True, cbar=True, ax=axs[2])
        ##set the title to the current behaviour in all caps
        f.suptitle(behaviour.upper(), fontweight='bold', y=1.0)
        ##label the X axis and label the main plot Y axis
        axs[-1].set(xlabel="Time (ms) from behaviour onset", ylabel="Event Number")
        ##label the Y axis of mean
        axs[0].set(ylabel="Mean")
        ##label the Y axis of mean
        axs[1].set(ylabel="Norm")
        ##make sure all three plots have the same indexes and labels
        axs[0].axline((int(ticks[labels==0])+0.5,0), (int(ticks[labels==0])+0.5,1), linestyle=':', color='white')
        axs[1].axline((int(ticks[labels==0])+0.5,0), (int(ticks[labels==0])+0.5,1), linestyle=':', color='white')
        axs[2].axline((int(ticks[labels==0])+0.5,0), (int(ticks[labels==0])+0.5,1), linestyle=':', color='white')
        ##set the yticks of the main plot
        plt.yticks(ticks=np.arange( 0,len(heatmap), 10*heatmap_scale), labels=np.arange( 0,int(len(heatmap)/heatmap_scale), 10))
        plt.xticks(ticks=ticks, labels=labels.astype(np.int64), rotation=0)
        ##ticks are on the bottom and left
        plt.tick_params(bottom=True, left=True)
        ##mean and norm plots have a tick
        axs[0].tick_params(left=True)
        axs[1].tick_params(left=True)
        ##dont label the mean and norm ticks with a number
        empty_string_labels = ['']
        axs[0].set_yticklabels(empty_string_labels)
        axs[1].set_yticklabels(empty_string_labels)
        ##remove unneccesary white space
        f.tight_layout()
        ##set the edges of the plot to be black and have a width of 1 
        plt.rcParams["axes.edgecolor"] = "black"
        plt.rcParams["axes.linewidth"] = 1
        ##remove the background grid
        plt.grid(False)
        ##set the edge colour and line width of all plots
        for ax in axs: ax.patch.set_edgecolor('black')  
        for ax in axs: ax.patch.set_linewidth('1')  
        ##save the figure as an SVG
        plt.savefig(savepath+'/'+AnimalID+f"_{behaviour}_heatmap.svg", bbox_inches='tight')
        print("Peri-event heatmap saved as: ", (savepath+'/'+AnimalID+f"_{behaviour}_heatmap.svg"))
        
        
        ##show it inline in the notebook
        #plt.show()
    for behaviour in unique_behaviours:
        create_heatmap(photom_hit_window, behaviour, pre_window_secs, post_window_secs, pre_post_secs)
