from GUI_and_data_processing.Create_GUI import (
    choose_settings_file_or_not, choose_location_settings_file)

from GUI_and_data_processing.Organise_GUI import (
    run_TDT_GUI, analyse_settings_file)

inputs = {}

inputs = choose_settings_file_or_not(inputs)

if inputs["Settings"] == True:
    # Import the options for analysis from a settings excel file or ...
    inputs = choose_location_settings_file(inputs)
    analyse_settings_file(inputs)

else:
    # ... put this data in manually.
    # Run the set of TDT GUIs.
    run_TDT_GUI(inputs)
    