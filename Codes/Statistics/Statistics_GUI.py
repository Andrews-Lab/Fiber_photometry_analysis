import PySimpleGUI as sg

from Statistics.FED3_fiphopha import launch_gui


def statistics_menu():
    """Show the statistics launcher and open the selected analysis tool."""
    sg.theme("DarkTeal2")
    layout = [
        [sg.T("")],
        [sg.Text("Choose a statistical analysis tool.")],
        [sg.T("")],
        [sg.Button("FED3 FiPhoPHA"), sg.Button("Close")],
    ]
    window = sg.Window("Statistics", layout)

    while True:
        event, _values = window.read()
        if event in (sg.WIN_CLOSED, "Close"):
            break
        if event == "FED3 FiPhoPHA":
            window.hide()
            try:
                launch_gui()
            finally:
                window.un_hide()

    window.close()

