from pathlib import Path
import subprocess
import sys

import PySimpleGUI as sg


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
            # Run FiPhoPHA in its own process so PySimpleGUI's existing Tk root
            # cannot interfere with FiPhoPHA's Tk variables or event loop.
            script = Path(__file__).with_name("FED3_fiphopha.py")
            subprocess.Popen([sys.executable, str(script)])
            break

    window.close()
