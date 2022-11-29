from GUI_and_data_processing.Create_GUI import *
from GUI_and_data_processing.Data_processing import *
from Root_Morales_lab_codes.FibPhoEpocAveraging import *
from Analysis_types.Whole_recording import *
from Analysis_types.Peri_events import *
from Root_Morales_lab_codes.FibPhoEpocAveraging_between_events import *
from Analysis_types.FED3 import *
from Analysis_types.Between_events import *
from tqdm import tqdm

def run_TDT_GUI(inputs):
    
    inputs = choose_basic_TDT_options(inputs)
    if inputs['Setup'] == 'Custom':
        inputs = choose_ISOS_and_GCaMP_signals(inputs)
    
    if inputs['Analysis'] == 'Peri-events':
        inputs = choose_type_TDT_event(inputs)
        inputs = find_possible_TDT_event_names(inputs)
        inputs = choose_name_TDT_event(inputs)
        inputs = define_unique_TDT_event(inputs)
        inputs = choose_peri_event_options(inputs)
        
        if inputs['Create snippets'] == True:
            inputs = choose_video_snippet_options(inputs)
        
        inputs, outputs = FiPhoEpocAveraging(inputs)
        outputs = create_export_data_peri_events(inputs, outputs)
        export_analysed_data_peri_events(inputs, outputs)
        
        if inputs['Create snippets'] == True:
            create_annotated_video(inputs, outputs)
            
        if inputs['Image'] == True:
            export_preview_image_peri_events(inputs, outputs)
            
    if inputs['Analysis'] == 'Whole recording':
        inputs = find_lists_of_events(inputs)
        inputs = choose_events_for_whole_recording(inputs)
        inputs = choose_whole_recording_options(inputs)
        inputs = define_all_whole_recording_events(inputs)
        
        inputs, outputs = whole_recording_analysis(inputs)
        outputs = create_export_plots(inputs, outputs)
        export_whole_recording_plots(inputs, outputs)
        
        if inputs['Raw data'] == True:
            export_whole_recording_data(inputs, outputs)
        
    if inputs['Analysis'] == 'FED3':
        inputs = find_options_FED3(inputs)
        inputs = find_event_names(inputs)
        inputs = choose_FED3_options(inputs)
        inputs = choose_peri_event_options(inputs)
        
        if inputs['Create snippets'] == True:
            inputs  = choose_video_snippet_options(inputs)

        inputs = create_unique_TDT_event_FED3(inputs)
        inputs, outputs = FiPhoEpocAveraging(inputs)
        outputs = create_export_data_FED3(inputs, outputs)
        export_analysed_data_FED3(inputs, outputs)
            
        if inputs['Create snippets'] == True:
            create_annotated_video(inputs, outputs)
            
        if inputs['Image'] == True:
            export_preview_image_FED3(inputs, outputs)
            
    if inputs['Analysis'] == 'Between events':
        inputs = choose_between_TTL_test(inputs)
        inputs = choose_type_TDT_event_between_TTL(inputs)
        inputs = find_possible_TDT_event_names(inputs)
        inputs = find_event_names_between_events(inputs)
        inputs = choose_name_TDT_event_between_events(inputs)
        inputs = create_unique_TDT_event_between_events(inputs)
        inputs = choose_between_events_options(inputs)
        
        inputs, outputs = FiPhoEpocAveraging_between_events(inputs)
        outputs = create_export_data_between_events(inputs, outputs)
        export_analysed_data_between_events(inputs, outputs)
            
        if inputs['Image'] == True:
            export_preview_image_between_events(inputs, outputs)
            
    export_settings_excel_file(inputs)

def analyse_settings_file(inputs):
    
    list_inputs = import_settings_excel_file(inputs)
    
    # Run the correct code, based on the information in the settings excel file.
    for inputs in tqdm(list_inputs, ncols=70):
    
        if inputs['Analysis'] == 'Peri-events':
            inputs = find_possible_TDT_event_names(inputs)
            inputs = define_unique_TDT_event(inputs)
            
            inputs, outputs = FiPhoEpocAveraging(inputs)
            outputs = create_export_data_peri_events(inputs, outputs)
            export_analysed_data_peri_events(inputs, outputs)
            
            if inputs['Create snippets'] == True:
                create_annotated_video(inputs, outputs)
                
            if inputs['Image'] == True:
                export_preview_image_peri_events(inputs, outputs)
                
        if inputs['Analysis'] == 'Whole recording':
            inputs = find_lists_of_events(inputs)
            inputs = define_all_whole_recording_events(inputs)
            
            inputs, outputs = whole_recording_analysis(inputs)
            outputs = create_export_plots(inputs, outputs)
            export_whole_recording_plots(inputs, outputs)
            
            if inputs['Raw data'] == True:
                export_whole_recording_data(inputs, outputs)
            
        if inputs['Analysis'] == 'FED3':
            inputs = find_options_FED3(inputs)
            inputs = create_unique_TDT_event_FED3(inputs)
            inputs, outputs = FiPhoEpocAveraging(inputs)
            outputs = create_export_data_FED3(inputs, outputs)
            export_analysed_data_FED3(inputs, outputs)
                
            if inputs['Create snippets'] == True:
                create_annotated_video(inputs, outputs)
                
            if inputs['Image'] == True:
                export_preview_image_FED3(inputs, outputs)
                
        if inputs['Analysis'] == 'Between events':
            inputs = find_possible_TDT_event_names(inputs)
            inputs = create_unique_TDT_event_between_events(inputs)
            
            inputs, outputs = FiPhoEpocAveraging_between_events(inputs)
            outputs = create_export_data_between_events(inputs, outputs)
            export_analysed_data_between_events(inputs, outputs)
                
            if inputs['Image'] == True:
                export_preview_image_between_events(inputs, outputs)
        