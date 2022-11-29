#import_location = 'C:/Users/hazza/Desktop/Fibre photometry GUI/Alex Mouse 1'

def Convert_NPM_to_TDT_data(import_location):

    import pandas as pd
    import os
    import numpy as np
    from tdt import StructType
    
    # import_location = r"C:\Users\hazza\Desktop\Fibre photometry GUI\Alex Mouse 1"
    import_files = [file for file in os.listdir(import_location) if 
                    (file.endswith(".csv") and file.startswith("~$")==False)]
    channel_numbers = ['415','470','560']
    color_names     = ['Region0G','Region1R','Region2G','Region3R']
    conversion      = {'Region0G':'0_green','Region1R':'1_red','Region2G':'2_green','Region3R':'3_red'}
    excel_files = {}
    
    epocs = {}
    streams = {}
    time_ranges = np.array([[0.],[np.inf]])
    
    for file in import_files:
        
        import_destination    = os.path.join(import_location, file)
        df = pd.read_csv(import_destination)
        file_parts = file[:-4].split('_')
        
        # Analyse the 415, 470 or 560 stream files.
        channel = list(set(channel_numbers).intersection(file_parts))
        color_headings = list(set(color_names).intersection(df.columns))
        if len(color_headings) > 0:
            
            # The channel numbers may appear later in the filename by coincidence, 
            # so only use the first appearance of a channel number.
            channel = channel[0]
            
            # Find the sample rate.
            time_cols = pd.DataFrame()
            time_cols['Normal']  = df['Timestamp'].copy()
            # Shift all the time values down by 1 (so there is a nan at the start).
            time_cols['Shifted'] = time_cols['Normal'].shift()
            # Remove the first row with the nan.
            time_cols = time_cols.iloc[1:]
            time_cols['Difference'] = time_cols['Normal'] - time_cols['Shifted']
            # frequency (Hz) = 1 / period (secs)
            sample_rate = 1/(time_cols['Difference'].mean())
            
            for color in color_names:
            
                name = channel+'_'+conversion[color]
                
                stream = {}
                stream['name'] = name
                stream['data'] = np.array(df[color])
                # stream['time_stamps'] = np.array(df['Timestamp'])
                stream['fs'] = sample_rate
                stream['start_time'] = time_cols['Normal'].iloc[0]
                streams[name] = StructType(stream)
        
        # Analyse the event time stamp files.
        elif 'Timestamp.Timestamp' in df.columns:
            
            name = file[:-4]
    
            epoc = {}
            epoc['name'] = name
            epoc['onset']  = np.array(df['Timestamp.Timestamp'])
            epoc['offset'] = np.append(epoc['onset'][1:], [np.inf])
            epoc['data'] = np.array(range(1,len(epoc['onset'])+1))
            epocs[name] = StructType(epoc)
    
    tank = {'epocs':       StructType(epocs),
            'snips':       StructType(),
            'streams':     StructType(streams),
            'scalars':     StructType(),
            'info':        StructType(),         
            'time_ranges': time_ranges}
    
    return(StructType(tank))
