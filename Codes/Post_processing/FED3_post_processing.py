import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, to_hex, to_rgb
from mpl_toolkits.mplot3d import Axes3D
import tkinter as tk
from tkinter import filedialog, messagebox, colorchooser
import os
import sys
from copy import copy
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch, Rectangle
import re
import textwrap

CHRONO_COLORS = {'Left': '#0072B2', 'Right': '#E69F00',
                 'Pellet': '#009E73', 'Rewarded': '#CC79A7'}


# ------------------------------------------------------------
# READ CHRONOLOGICAL EVENT DATA
# ------------------------------------------------------------
def _chrono_read_frame(df, selected):
    """Read original FED3 export headers, preserving event-to-trace correspondence."""
    def header(label):
        matches = df.index[df.eq(label).any(axis=1)].tolist()
        if len(matches) != 1:
            raise ValueError('Expected one header row: ' + label)
        return matches[0]

    onset_row = header('Time of event onset (secs)')
    note_row = header('Event note')
    start = header('Custom name') + 1
    times = pd.to_numeric(df.iloc[start:, 1], errors='coerce').to_numpy(dtype=float)
    keep = np.isfinite(times)
    times = times[keep]
    if len(times) < 2 or np.any(np.diff(times) <= 0):
        raise ValueError('Peri-event time vector must contain increasing numeric timestamps.')
    events = []
    for col in range(3, df.shape[1]):
        note = str(df.iloc[note_row, col]).strip()
        if note not in selected:
            continue
        onset = pd.to_numeric(df.iloc[onset_row, col], errors='coerce')
        if not np.isfinite(onset):
            raise ValueError('Missing onset for event column %s; sequence cannot be reconstructed.' % col)
        trace = pd.to_numeric(df.iloc[start:, col], errors='coerce').to_numpy(dtype=float)[keep]
        events.append({'onset': float(onset), 'event': note, 'time': times,
                       'trace': trace, 'source_column': col + 1})
    return events


def _chrono_load(path, selected):
    with pd.ExcelFile(path) as book:
        # Overall is authoritative and avoids counting the per-event sheets twice.
        sheets = ['Overall'] if 'Overall' in book.sheet_names else [
            event for event in selected if event in book.sheet_names]
        if not sheets:
            raise ValueError('No Overall or selected event sheets found.')
        missing = [event for event in selected if event not in book.sheet_names]
        if 'Overall' not in book.sheet_names and missing:
            raise ValueError('Missing selected sheets: ' + ', '.join(missing))
        events = []
        for sheet in sheets:
            for event in _chrono_read_frame(pd.read_excel(book, sheet_name=sheet, header=None), selected):
                event['source_sheet'] = sheet
                events.append(event)
    events.sort(key=lambda event: event['onset'])
    if not events:
        raise ValueError('No selected events found.')
    return events


# ------------------------------------------------------------
# CHRONOLOGICAL PLOT EXPORT HELPERS
# ------------------------------------------------------------
def _chrono_png_dpi(width, height):
    """Bound wide PNG memory and dimensions while targeting 300 DPI."""
    return min(300., 60000. / max(width, height), (40000000. / (width * height)) ** .5)


def _chrono_export(records, folder, colors, sequence=True, heatmaps=True, page_events=400,
                   show_plots=False, metadata_headers=('Mouse ID', 'Sex', 'Genotype'), trace_page_events=30, wide_png=True, wide_svg=True, paginated_png=False,
                   metadata_color_maps=None):
    """Export concatenated peri-event traces, heatmaps and an auditable event table."""
    os.makedirs(folder, exist_ok=True)
    def finish(figure, filename, wide=False):
        output_path = os.path.join(folder, filename)
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        try:
            # Keep all line samples in the zoomable file; chunk raster paths for large exports.
            with plt.rc_context({'path.simplify': False, 'agg.path.chunksize': 10000}):
                if wide:
                    stem = os.path.splitext(output_path)[0]
                    if wide_svg:
                        figure.savefig(stem + '.svg', format='svg')
                    if wide_png:
                        width, height = figure.get_size_inches()
                        figure.savefig(stem + '.png', dpi=_chrono_png_dpi(width, height))
                else:
                    figure.savefig(output_path, dpi=300)
            if show_plots:
                # Display a screen-sized preview; exports keep their full dimensions.
                if wide:
                    figure.set_size_inches(18, min(12, figure.get_size_inches()[1]))
                plt.show()
        finally:
            plt.close(figure)

    # ------------------------------------------------------------
    # EXPORT CHRONOLOGICAL EVENT AUDIT TABLE
    # ------------------------------------------------------------
    legend = [Patch(facecolor=color, label=event) for event, color in colors.items()]
    rows = []
    for record in records:
        groups = []
        for event in record['events']:
            if not groups or event['onset'] != groups[-1][0]['onset']:
                groups.append([])
            groups[-1].append(event)
        record['groups'] = groups
        event_row = 0
        for position, group in enumerate(groups, 1):
            for event in group:
                event_row += 1
                rows.append({'Filename': record['filename'], 'Mouse': record['mouse'],
                             'Sex': record['sex'], 'Group': record['group'],
                              'Chronological position': position, 'Event row': event_row,
                             'Window start (secs)': event['time'][0],
                             'Window end (secs)': event['time'][-1], 'Onset (secs)': event['onset'],
                             'Event': event['event'], 'Events at same timestamp': len(group),
                             'Source sheet': event['source_sheet'],
                             'Source column (Excel 1-based)': event['source_column']})
    pd.DataFrame(rows).to_csv(os.path.join(folder, 'Chronological_events.csv'), index=False)
    # ------------------------------------------------------------
    # CALCULATE SHARED CHRONOLOGICAL Z-SCORE SCALE
    # ------------------------------------------------------------
    # A single symmetric scale across all selected recordings; no percentile clipping.
    limit = 0.
    for record in records:
        for event in record['events']:
            finite = event['trace'][np.isfinite(event['trace'])]
            if len(finite):
                limit = max(limit, float(np.max(np.abs(finite))))
    limit = limit or 1.
    # ------------------------------------------------------------
    # CREATE CHRONOLOGICAL EVENT TRACE PLOTS
    # ------------------------------------------------------------
    if sequence:
        modes = []
        if wide_png or wide_svg:
            modes.append(True)
        if paginated_png:
            modes.append(False)
        for wide in modes:
            mice_per_figure = len(records) if wide else 8
            for first_record in range(0, len(records), mice_per_figure):
                batch = records[first_record:first_record + mice_per_figure]
                longest = max(len(r['events']) for r in batch)
                events_per_figure = longest if wide else trace_page_events
                for first in range(0, longest, events_per_figure):
                    last = min(first + events_per_figure, longest)
                    # Metadata occupies aligned columns, rather than a concatenated tick label.
                    width = max(18, 6 + .4 * longest) if wide else 18
                    fig, (meta_ax, ax) = plt.subplots(
                        1, 2, figsize=(width, max(4, len(batch) * 1.1 + 2.5)),
                        gridspec_kw={'width_ratios': [5, width - 6], 'wspace': .5 / width}, sharey=True)
                    meta_ax.set_xlim(0, 3)
                    meta_ax.axis('off')
                    for column, header in enumerate(metadata_headers):
                        meta_ax.text((column + .05) / 3, 1.02,
                                     textwrap.fill(str(header), 16),
                                     transform=meta_ax.transAxes, ha='left', va='bottom',
                                     fontsize=10, fontweight='bold')
                    for row, record in enumerate(batch):
                        if row % 2 == 0:
                            meta_ax.axhspan(row - .5, row + .5, color='#f3f3f3', zorder=0)
                            ax.axhspan(row - .5, row + .5, color='#f3f3f3', zorder=0)
                        for column, record_key, map_key in ((1, 'sex', 'sex'), (2, 'group', 'genotype')):
                            cell_color = (metadata_color_maps or {}).get(map_key, {}).get(record[record_key])
                            if cell_color:
                                # White underlay keeps the chosen tint identical on alternating rows.
                                meta_ax.add_patch(Rectangle((column, row - .5), 1, 1,
                                                           facecolor='white', edgecolor='none', zorder=.1))
                                meta_ax.add_patch(Rectangle((column, row - .5), 1, 1,
                                                           facecolor=cell_color, alpha=.12,
                                                           edgecolor='none', zorder=.2))
                        for column, key in enumerate(('mouse', 'sex', 'group')):
                            meta_ax.text(column + .05, row, textwrap.fill(record[key], 16),
                                         ha='left', va='center', fontsize=9)
                        if row and record['group'] != batch[row - 1]['group']:
                            for axis in (meta_ax, ax):
                                axis.axhline(row - .5, color='#888888', linewidth=.8)
                    for column in (1, 2):
                        meta_ax.axvline(column, color='#dddddd', linewidth=.6)
                    ticks, tick_labels = [], []
                    for row, record in enumerate(batch):
                        ax.axhline(row, color='#bbbbbb', linewidth=.5, zorder=0)
                        previous = None
                        for pos in range(first, min(last, len(record['events']))):
                            event = record['events'][pos]
                            # Keep every original sample and its relative spacing within its window.
                            time = event['time']
                            x = pos + .52 + .96 * (time - time[0]) / (time[-1] - time[0])
                            values = np.where(np.isfinite(event['trace']), event['trace'], np.nan)
                            y = row - .36 * values / limit
                            if previous is not None and np.isfinite(previous[1]) and np.isfinite(y[0]):
                                # Grey joins explicitly mark the splice between separate windows.
                                ax.plot([previous[0], x[0]], [previous[1], y[0]],
                                        color='#999999', linewidth=.6, linestyle=':', zorder=1)
                            ax.plot(x, y, color=colors[event['event']], linewidth=.9, zorder=2)
                            previous = (x[-1], y[-1])
                        ticks.extend([row - .36, row, row + .36])
                        tick_labels.extend(['%g' % limit, '0', '%g' % -limit])
                    ax.set_xlim(first + .5, last + .5)
                    ax.set_ylim(len(batch) - .5, -.5)
                    ax.set_yticks(ticks)
                    ax.set_yticklabels(tick_labels, fontsize=7)
                    ax.yaxis.tick_right()
                    ax.yaxis.set_label_position('right')
                    ax.set_ylabel('Z-score (shared scale)', fontsize=9)
                    tick_step = max(1, int(np.ceil((last - first) / 30)))
                    ax.set_xticks(np.arange(first + 1, last + 1, 1 if wide else tick_step))
                    ax.tick_params(axis='x', labelsize=8)
                    ax.set_xlabel('Event number — concatenated peri-event windows (not session time)')
                    ax.set_title('Chronological peri-event Z-score traces')
                    ax.legend(handles=legend, loc='lower center', bbox_to_anchor=(.5, 1.04), ncol=4)
                    fig.text(.5, .045, 'Grey dotted joins mark separate windows. Equal-onset events are adjacent; their internal order is arbitrary.',
                             ha='center', fontsize=9)
                    fig.subplots_adjust(left=.45 / width, right=1 - .9 / width,
                                        bottom=.18, top=.80)
                    suffix = ''
                    if len(records) > mice_per_figure:
                        suffix += '_Mice_%02d' % (first_record // mice_per_figure + 1)
                    if longest > events_per_figure:
                        suffix += '_Part_%02d' % (first // events_per_figure + 1)
                    name = 'Event_Sequence_Traces_Wide' if wide else 'Event_Sequence_Traces'
                    finish(fig, os.path.join('Sequence_traces', name + suffix + '.png'), wide=wide)
    # ------------------------------------------------------------
    # CREATE CHRONOLOGICAL Z-SCORE HEATMAPS
    # ------------------------------------------------------------
    if heatmaps:
        event_names = list(colors)
        for number, record in enumerate(records, 1):
            events = record['events']
            reference = events[0]['time']
            if any(len(e['time']) != len(reference) or not np.allclose(e['time'], reference)
                   for e in events):
                raise ValueError('Peri-event time vectors differ within ' + record['filename'])
            edges = np.concatenate(([reference[0] - (reference[1] - reference[0]) / 2],
                                    (reference[:-1] + reference[1:]) / 2,
                                    [reference[-1] + (reference[-1] - reference[-2]) / 2]))
            for first in range(0, len(events), page_events):
                part = events[first:first + page_events]
                fig, (strip, ax) = plt.subplots(1, 2, figsize=(11, 8),
                                               gridspec_kw={'width_ratios': [1, 24]}, sharey=True)
                yedges = np.arange(first + .5, first + len(part) + 1.5)
                matrix = np.ma.masked_invalid(np.vstack([e['trace'] for e in part]))
                cmap = copy(plt.get_cmap('RdBu_r'))
                cmap.set_bad('#dddddd')
                mesh = ax.pcolormesh(edges, yedges, matrix, cmap=cmap, vmin=-limit, vmax=limit,
                                     shading='flat', rasterized=True)
                strip.pcolormesh([0, 1], yedges,
                                 np.array([event_names.index(e['event']) for e in part])[:, None],
                                 cmap=ListedColormap(list(colors.values())), vmin=-.5,
                                 vmax=len(colors) - .5, shading='flat')
                strip.set_xticks([])
                strip.set_ylabel('Retained event row (chronological; timestamp ties have no internal order)')
                ax.set_ylim(yedges[-1], yedges[0])
                if reference[0] <= 0 <= reference[-1]:
                    ax.axvline(0, color='black', linewidth=.8, linestyle='--')
                ax.set_xlabel('Time relative to event onset (seconds)')
                fig.colorbar(mesh, ax=ax, label='Z-score (shared scale)')
                fig.suptitle('%s | %s | %s\n%s' % (record['mouse'], record['sex'],
                                                  record['group'], record['filename']), fontsize=10)
                fig.legend(handles=legend, loc='lower center', ncol=4)
                fig.tight_layout(rect=[0, .05, 1, .94])
                safe = re.sub(r'[^\w-]', '_', record['mouse']) or 'Unknown'
                # Include the recording name only when a mouse has multiple selected files.
                same_mouse = [r for r in records if
                              (re.sub(r'[^\w-]', '_', r['mouse']) or 'Unknown') == safe]
                if len(same_mouse) > 1:
                    safe += '_' + re.sub(r'[^\w-]', '_', os.path.splitext(record['filename'])[0])
                suffix = '_Part_%02d' % (first // page_events + 1) if len(events) > page_events else ''
                finish(fig, os.path.join('Individual_heatmaps', safe + '_Chronological_Heatmap' + suffix + '.png'))

# ------------------------------------------------------------
# BUILD CHRONOLOGICAL RECORDS FROM METADATA
# ------------------------------------------------------------
def _chrono_records(metadata, file_map, selected, mouse_col, sex_col, group_col):
    """Load chronology only when selected in the regular plot-options window."""
    records, errors = [], []
    def clean(value):
        return 'Unknown' if pd.isna(value) or not str(value).strip() else str(value).strip()
    for _, row in metadata.iterrows():
        filename = row['Filename']
        if filename not in file_map:
            continue
        try:
            records.append({'filename': filename, 'mouse': clean(row[mouse_col]),
                            'sex': clean(row[sex_col]), 'group': clean(row[group_col]),
                            'events': _chrono_load(file_map[filename], selected)})
        except Exception as error:
            errors.append('%s: %s' % (filename, error))
    missing = set(file_map) - set(metadata['Filename'])
    errors.extend('No metadata for selected file: ' + name for name in sorted(missing))
    if metadata['Filename'].duplicated().any():
        errors.append('Duplicate filenames in metadata; use one row per recording.')
    if errors or not records:
        raise ValueError('\n'.join(errors) or 'No recordings found.')
    return records


def FED3_post_processing():

    # ------------------------------------------------------------
    # SELECT DATA FILES
    # ------------------------------------------------------------
    root = tk.Tk()
    root.withdraw()

    file_paths = filedialog.askopenfilenames(
        title="Select FED3 Photometry Excel files",
        filetypes=[("Excel files", "*.xlsx")]
    )

    if not file_paths:
        root.destroy()
        return

    file_map = {os.path.basename(f): f for f in file_paths}
    if len(file_map) != len(file_paths):
        messagebox.showerror('Duplicate filenames',
                             'Selected files must have unique filenames so metadata can identify each recording.')
        root.destroy()
        return
    save_folder = os.path.dirname(file_paths[0])

    # ------------------------------------------------------------
    # ASK IF METADATA EXISTS
    # ------------------------------------------------------------
    use_existing = messagebox.askyesno(
        "Metadata",
        "Do you have an existing metadata file?\n\n"
        "Do not delete or leave any metadata header titles blank."
    )

    metadata_df = None

    # ------------------------------------------------------------
    # LOAD EXISTING METADATA
    # ------------------------------------------------------------
    if use_existing:

        metadata_file = filedialog.askopenfilename(
            title="Select Metadata File",
            filetypes=[("Excel files", "*.xlsx")]
        )

        if not metadata_file:
            root.destroy()
            return

        metadata_df = pd.read_excel(metadata_file)

    # ------------------------------------------------------------
    # CREATE METADATA GUI
    # ------------------------------------------------------------
    else:

        meta_window = tk.Toplevel(root)
        meta_window.title("Enter Metadata")

        # Make window resizable
        meta_window.geometry("700x500")

        tk.Label(
            meta_window,
            text=("Keep all metadata header titles present and unique. "
                  "Individual metadata values may be left blank."),
            fg="dark red"
        ).pack(fill="x", padx=8, pady=(6, 2))

        canvas = tk.Canvas(meta_window)
        scrollbar = tk.Scrollbar(meta_window, orient="vertical", command=canvas.yview)

        scroll_frame = tk.Frame(canvas)

        scroll_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )

        canvas.create_window((0, 0), window=scroll_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)

        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")

        def _on_mousewheel(event):
            canvas.yview_scroll(int(-1 * (event.delta / 120)), "units")

        canvas.bind("<Enter>", lambda e: canvas.bind_all("<MouseWheel>", _on_mousewheel))
        canvas.bind("<Leave>", lambda e: canvas.unbind_all("<MouseWheel>"))

        headers = ["Filename", "Mouse ID", "Sex", "Genotype"]

        header_entries = {}

        for col, header in enumerate(headers):

            if header == "Filename":
                tk.Label(scroll_frame, text=header, font=("Arial", 10, "bold")).grid(row=0, column=col)
            else:
                entry = tk.Entry(scroll_frame, width=15)
                entry.insert(0, header)
                entry.grid(row=0, column=col)

                header_entries[col] = entry

        rows = []

        for i, filename in enumerate(file_map.keys()):

            tk.Label(scroll_frame, text=filename).grid(row=i+1, column=0)

            mouse_entry = tk.Entry(scroll_frame)
            mouse_entry.grid(row=i+1, column=1)

            sex_entry = tk.Entry(scroll_frame)
            sex_entry.grid(row=i+1, column=2)

            genotype_entry = tk.Entry(scroll_frame)
            genotype_entry.grid(row=i+1, column=3)

            rows.append({
                "filename": filename,
                "mouse": mouse_entry,
                "sex": sex_entry,
                "genotype": genotype_entry
            })

        def collect_metadata():

            metadata_list = []

            mouse_header = header_entries[1].get()
            sex_header = header_entries[2].get()
            group_header = header_entries[3].get()

            for row in rows:
                metadata_list.append({
                    "Filename": row["filename"],
                    mouse_header: row["mouse"].get(),
                    sex_header: row["sex"].get(),
                    group_header: row["genotype"].get()
                })

            nonlocal metadata_df
            metadata_df = pd.DataFrame(metadata_list)

            metadata_path = os.path.join(save_folder, "FED3_FP_Metadata.xlsx")
            metadata_df.to_excel(metadata_path, index=False)

            meta_window.destroy()

        tk.Button(scroll_frame, text="Continue", command=collect_metadata)\
            .grid(row=len(file_map)+2, column=0, columnspan=4)

        root.wait_window(meta_window)

    # ------------------------------------------------------------
    # SORT METADATA
    # ------------------------------------------------------------
    
    if metadata_df is None:
        messagebox.showerror(
            "Metadata Error",
            "Metadata was not created or loaded properly."
        )
        root.destroy()
        return

    meta_columns = [col for col in metadata_df.columns if col != "Filename"]

    mouse_id_col = meta_columns[0]
    sex_col = meta_columns[1]
    group_column = meta_columns[2]

    metadata_df[mouse_id_col] = metadata_df[mouse_id_col].astype(str)

    metadata_df["Mouse ID numeric"] = pd.to_numeric(metadata_df[mouse_id_col], errors="coerce")

    metadata_df = metadata_df.sort_values(
        by=[group_column, sex_col, "Mouse ID numeric", mouse_id_col]
    ).drop(columns=["Mouse ID numeric"]).reset_index(drop=True)

    # ------------------------------------------------------------
    # SELECT EVENT TABS
    # ------------------------------------------------------------
    tab_window = tk.Toplevel(root)
    tab_window.title("Select Event Tabs")

    tab_vars = {}
    available_tabs = ["Left", "Right", "Pellet", "Rewarded"]

    for i, tab in enumerate(available_tabs):
        var = tk.BooleanVar(master=tab_window, value=True)
        tk.Checkbutton(tab_window, text=tab, variable=var).grid(row=i, column=0, sticky="w")
        tab_vars[tab] = var

    selected_tabs = []

    def confirm_tabs():
        nonlocal selected_tabs
        selected_tabs = [tab for tab, var in tab_vars.items() if var.get()]
        tab_window.destroy()

    tk.Button(tab_window, text="Analyze", command=confirm_tabs)\
        .grid(row=len(available_tabs)+1, column=0)

    root.wait_window(tab_window)

    if not selected_tabs:
        messagebox.showerror(
            "Selection Error",
            "No event tabs selected."
        )
        root.destroy()
        return

    # ------------------------------------------------------------
    # OUTPUT AND PLOT OPTIONS
    # ------------------------------------------------------------
    plot_options_window = tk.Toplevel(root)
    plot_options_window.title("FED3 Output and Plot Options")
    plot_options_window.rowconfigure(0, weight=1)
    plot_options_window.columnconfigure(0, weight=1)

    plot_options_canvas = tk.Canvas(
        plot_options_window,
        highlightthickness=0
    )
    plot_options_scrollbar = tk.Scrollbar(
        plot_options_window,
        orient="vertical",
        command=plot_options_canvas.yview
    )
    plot_options_canvas.configure(
        yscrollcommand=plot_options_scrollbar.set
    )
    plot_options_canvas.grid(row=0, column=0, sticky="nsew")
    plot_options_scrollbar.grid(row=0, column=1, sticky="ns")

    plot_options_content = tk.Frame(plot_options_canvas)
    plot_options_canvas_window = plot_options_canvas.create_window(
        (0, 0),
        window=plot_options_content,
        anchor="nw"
    )

    def update_plot_options_scrollregion(_event=None):
        plot_options_canvas.configure(
            scrollregion=plot_options_canvas.bbox("all")
        )

    def resize_plot_options_content(event):
        plot_options_canvas.itemconfigure(
            plot_options_canvas_window,
            width=event.width
        )

    def scroll_plot_options(event):
        if getattr(event, "num", None) == 4:
            scroll_units = -1
        elif getattr(event, "num", None) == 5:
            scroll_units = 1
        else:
            delta = getattr(event, "delta", 0)
            if delta == 0:
                return
            if sys.platform == "darwin":
                scroll_units = -1 if delta > 0 else 1
            else:
                scroll_units = int(-delta / 120)
                if scroll_units == 0:
                    scroll_units = -1 if delta > 0 else 1

        plot_options_canvas.yview_scroll(scroll_units, "units")
        return "break"

    plot_options_content.bind(
        "<Configure>",
        update_plot_options_scrollregion
    )
    plot_options_canvas.bind(
        "<Configure>",
        resize_plot_options_content
    )
    plot_options_window.bind("<MouseWheel>", scroll_plot_options)
    plot_options_window.bind("<Button-4>", scroll_plot_options)
    plot_options_window.bind("<Button-5>", scroll_plot_options)

    tk.Label(
        plot_options_content,
        text=(
            "The FED3_FP_Combined workbook is always created. "
            "Select any optional figures you would also like to generate."
        ),
        fg="dark green",
        justify="left",
        wraplength=520
    ).grid(row=0, column=0, columnspan=2, sticky="w", padx=10, pady=(8, 4))

    checkbox_variables = {}
    checkbox_options = [
        "Create per-mouse mean ± SEM plots",
        "Create grouped overlay and metric plots",
        "Create individual 2D event-progression plots",
        "Create individual 3D event-progression plots",
        "Create group 3D comparison plots (shared axes)",
        "Create chronological event trace plots",
        "Create chronological Z-score heatmaps"
    ]

    for row_number, label in enumerate(checkbox_options):
        variable = tk.BooleanVar(
            master=plot_options_window,
            value=False
        )
        tk.Checkbutton(
            plot_options_content,
            text=label,
            variable=variable
        ).grid(
            row=row_number + 1,
            column=0,
            columnspan=2,
            sticky="w",
            padx=10,
            pady=4
        )
        checkbox_variables[label] = variable

    option_entries = {}
    general_option_defaults = [
        ("Events per group", "1"),
        ("Plot downsampling factor", "1"),
        ("Line width", "1.5")
    ]

    for row_number, (label, default_value) in enumerate(
        general_option_defaults,
        start=8
    ):
        tk.Label(plot_options_content, text=label).grid(
            row=row_number, column=0, sticky="e", padx=8, pady=4
        )
        entry = tk.Entry(plot_options_content, width=10)
        entry.insert(0, default_value)
        entry.grid(row=row_number, column=1, sticky="w", padx=8, pady=4)
        option_entries[label] = entry

    time_range_mode = tk.StringVar(
        master=plot_options_window,
        value="Use full available range"
    )
    tk.Label(plot_options_content, text="3D time range").grid(
        row=11, column=0, sticky="e", padx=8, pady=4
    )
    tk.OptionMenu(
        plot_options_content,
        time_range_mode,
        "Use full available range",
        "Custom range"
    ).grid(row=11, column=1, sticky="w", padx=8, pady=4)

    tk.Label(plot_options_content, text="3D start time (s)").grid(
        row=12, column=0, sticky="e", padx=8, pady=4
    )
    custom_start_entry = tk.Entry(plot_options_content, width=10, state="disabled")
    custom_start_entry.grid(row=12, column=1, sticky="w", padx=8, pady=4)

    tk.Label(plot_options_content, text="3D end time (s)").grid(
        row=13, column=0, sticky="e", padx=8, pady=4
    )
    custom_end_entry = tk.Entry(plot_options_content, width=10, state="disabled")
    custom_end_entry.grid(row=13, column=1, sticky="w", padx=8, pady=4)

    viewing_option_defaults = [
        ("Vertical viewing angle", "25"),
        ("Horizontal viewing angle", "-60")
    ]
    for row_number, (label, default_value) in enumerate(
        viewing_option_defaults,
        start=14
    ):
        tk.Label(plot_options_content, text=label).grid(
            row=row_number, column=0, sticky="e", padx=8, pady=4
        )
        entry = tk.Entry(plot_options_content, width=10)
        entry.insert(0, default_value)
        entry.grid(row=row_number, column=1, sticky="w", padx=8, pady=4)
        option_entries[label] = entry

    def update_custom_time_entries(*_):
        entry_state = (
            "normal"
            if time_range_mode.get() == "Custom range"
            else "disabled"
        )
        custom_start_entry.configure(state=entry_state)
        custom_end_entry.configure(state=entry_state)

    time_range_mode.trace_add("write", update_custom_time_entries)

    progression_options = {}

    def confirm_plot_options():
        try:
            event_group_size = int(option_entries["Events per group"].get())
            downsample_factor = int(option_entries["Plot downsampling factor"].get())
            line_width = float(option_entries["Line width"].get())
            elevation = float(option_entries["Vertical viewing angle"].get())
            azimuth = float(option_entries["Horizontal viewing angle"].get())

            custom_3d_range = time_range_mode.get() == "Custom range"
            plot_time_start = None
            plot_time_end = None
            if custom_3d_range:
                plot_time_start = float(custom_start_entry.get())
                plot_time_end = float(custom_end_entry.get())
                if plot_time_start >= plot_time_end:
                    messagebox.showerror(
                        "Error",
                        "The custom 3D start time must be less than the end time."
                    )
                    return

            if event_group_size < 1 or downsample_factor < 1 or line_width <= 0:
                messagebox.showerror(
                    "Error",
                    "Events per group and downsampling factor must be at least 1, "
                    "and line width must be greater than 0."
                )
                return

            if (checkbox_variables["Create chronological event trace plots"].get() and
                    not any(variable.get() for variable in trace_format_vars.values())):
                messagebox.showerror("Error", "Select at least one chronological trace output format.")
                return
            trace_page_events = int(trace_page_entry.get())
            if trace_page_events < 1:
                messagebox.showerror("Error", "Events per trace page must be at least 1.")
                return
            progression_options.update({
                "per_mouse": checkbox_variables[
                    "Create per-mouse mean ± SEM plots"].get(),
                "grouped_summary": checkbox_variables[
                    "Create grouped overlay and metric plots"].get(),
                "trace_page_events": trace_page_events,
                "trace_wide_png": trace_format_vars["wide_png"].get(),
                "trace_wide_svg": trace_format_vars["wide_svg"].get(),
                "trace_paginated_png": trace_format_vars["paginated_png"].get(),
                "chronological_sequence": checkbox_variables[
                    "Create chronological event trace plots"].get(),
                "chronological_heatmaps": checkbox_variables[
                    "Create chronological Z-score heatmaps"].get(),
                "individual_2d": (
                    checkbox_variables[
                        "Create individual 2D event-progression plots"
                    ].get()
                ),
                "individual_3d": (
                    checkbox_variables[
                        "Create individual 3D event-progression plots"
                    ].get()
                ),
                "group_3d": (
                    checkbox_variables[
                        "Create group 3D comparison plots (shared axes)"
                    ].get()
                ),
                "event_group_size": event_group_size,
                "downsample_factor": downsample_factor,
                "line_width": line_width,
                "elevation": elevation,
                "azimuth": azimuth,
                "time_range_mode": time_range_mode.get(),
                "time_start": plot_time_start,
                "time_end": plot_time_end
            })
            plot_options_window.destroy()

        except ValueError:
            messagebox.showerror("Error", "Please enter valid numeric plot settings")

    # ------------------------------------------------------------
    # CHRONOLOGICAL PLOT FORMAT AND COLOUR OPTIONS
    # ------------------------------------------------------------
    chronological_colors = dict(CHRONO_COLORS)
    event_color_frame = tk.LabelFrame(plot_options_content, text="Chronological event colours")
    event_color_frame.grid(row=16, column=0, columnspan=2, sticky="ew", padx=10, pady=6)
    event_color_buttons = []
    for column, event in enumerate(CHRONO_COLORS):
        button = tk.Button(event_color_frame, text=event, bg=chronological_colors[event], width=10)
        def choose_event_color(name=event, widget=button):
            color = colorchooser.askcolor(chronological_colors[name],
                                         title="Choose colour for " + name,
                                         parent=plot_options_window)[1]
            if color:
                chronological_colors[name] = color
                widget.configure(bg=color)
        button.configure(command=choose_event_color)
        button.grid(row=0, column=column, padx=4, pady=5)
        event_color_buttons.append(button)

    trace_format_vars = {}
    trace_format_buttons = []
    for row, (key, label, default) in enumerate([
            ("wide_png", "Wide PNG — all mice and events", True),
            ("wide_svg", "Zoomable SVG — all mice and events", True),
            ("paginated_png", "Paginated PNGs", False)], start=2):
        variable = tk.BooleanVar(master=plot_options_window, value=default)
        trace_format_vars[key] = variable
        button = tk.Checkbutton(event_color_frame, text=label, variable=variable)
        button.grid(row=row, column=0, columnspan=4, sticky="w", padx=4)
        trace_format_buttons.append(button)

    tk.Label(event_color_frame, text="Events per PNG page").grid(
        row=1, column=0, columnspan=2, sticky="e", padx=4, pady=4)
    trace_page_entry = tk.Entry(event_color_frame, width=8)
    trace_page_entry.insert(0, "30")
    trace_page_entry.grid(row=1, column=2, sticky="w", padx=4, pady=4)

    def update_event_color_controls(*_):
        enabled = (checkbox_variables["Create chronological event trace plots"].get() or
                   checkbox_variables["Create chronological Z-score heatmaps"].get())
        traces_enabled = checkbox_variables["Create chronological event trace plots"].get()
        for button in trace_format_buttons:
            button.configure(state="normal" if traces_enabled else "disabled")
        trace_page_entry.configure(state="normal" if traces_enabled and
                                   trace_format_vars["paginated_png"].get() else "disabled")
        for button in event_color_buttons:
            button.configure(state="normal" if enabled else "disabled")
    for label in ("Create chronological event trace plots", "Create chronological Z-score heatmaps"):
        checkbox_variables[label].trace_add("write", update_event_color_controls)
    trace_format_vars["paginated_png"].trace_add("write", update_event_color_controls)
    update_event_color_controls()

    tk.Button(
        plot_options_content,
        text="Confirm",
        command=confirm_plot_options
    ).grid(row=17, column=0, columnspan=2, pady=10)

    plot_options_window.update_idletasks()
    screen_width = plot_options_window.winfo_screenwidth()
    screen_height = plot_options_window.winfo_screenheight()
    maximum_width = max(360, screen_width - 80)
    maximum_height = max(300, screen_height - 120)
    requested_width = plot_options_content.winfo_reqwidth() + \
        plot_options_scrollbar.winfo_reqwidth() + 4
    requested_height = plot_options_content.winfo_reqheight() + 4
    window_width = min(requested_width, maximum_width)
    window_height = min(requested_height, maximum_height)
    window_x = max(0, (screen_width - window_width) // 2)
    window_y = max(0, (screen_height - window_height) // 2)
    plot_options_window.geometry(
        f"{window_width}x{window_height}+{window_x}+{window_y}"
    )
    plot_options_window.minsize(
        min(360, window_width),
        min(300, window_height)
    )

    root.wait_window(plot_options_window)

    if not progression_options:
        root.destroy()
        return

    # ------------------------------------------------------------
    # ASK FOR ANALYSIS TIME WINDOWS
    # ------------------------------------------------------------
    window_popup = tk.Toplevel(root)
    window_popup.title("Set Time Windows (seconds)")

    tk.Label(window_popup, text="").grid(row=0, column=0, padx=8, pady=4)
    tk.Label(window_popup, text="Start", font=("Arial", 10, "bold")).grid(row=0, column=1, padx=8, pady=4)
    tk.Label(window_popup, text="End", font=("Arial", 10, "bold")).grid(row=0, column=2, padx=8, pady=4)

    tk.Label(window_popup, text="AUC / Mean Z-score").grid(row=1, column=0, sticky="e", padx=8, pady=4)
    auc_start_entry = tk.Entry(window_popup, width=10)
    auc_start_entry.insert(0, "0")
    auc_start_entry.grid(row=1, column=1, padx=8, pady=4)
    auc_end_entry = tk.Entry(window_popup, width=10)
    auc_end_entry.insert(0, "5")
    auc_end_entry.grid(row=1, column=2, padx=8, pady=4)

    tk.Label(window_popup, text="Peak Z-score").grid(row=2, column=0, sticky="e", padx=8, pady=4)
    peak_start_entry = tk.Entry(window_popup, width=10)
    peak_start_entry.insert(0, "0")
    peak_start_entry.grid(row=2, column=1, padx=8, pady=4)
    peak_end_entry = tk.Entry(window_popup, width=10)
    peak_end_entry.insert(0, "5")
    peak_end_entry.grid(row=2, column=2, padx=8, pady=4)

    auc_window = {}
    peak_window = {}

    def confirm_window():
        try:
            auc_start = float(auc_start_entry.get())
            auc_end = float(auc_end_entry.get())
            peak_start = float(peak_start_entry.get())
            peak_end = float(peak_end_entry.get())

            if auc_start >= auc_end or peak_start >= peak_end:
                messagebox.showerror("Error", "Each start time must be less than its end time")
                return

            auc_window["start"] = auc_start
            auc_window["end"] = auc_end
            peak_window["start"] = peak_start
            peak_window["end"] = peak_end

            window_popup.destroy()

        except ValueError:
            messagebox.showerror("Error", "Please enter valid numbers")

    tk.Button(window_popup, text="Confirm", command=confirm_window)\
        .grid(row=3, column=0, columnspan=3, pady=8)

    root.wait_window(window_popup)

    if not auc_window or not peak_window:
        root.destroy()
        return

    # ------------------------------------------------------------
    # PLOTTING HELPERS
    # ------------------------------------------------------------
    def clean_group_value(value):
        if pd.isna(value):
            return "Unknown"

        value = str(value).strip()
        return value if value else "Unknown"

    def safe_filename_value(value):
        value = clean_group_value(value)
        safe = "".join(ch if ch.isalnum() or ch in ["-", "_"] else "_" for ch in value)
        return safe.strip("_") or "Unknown"

    def build_default_color_map(values):
        values = sorted(set(clean_group_value(value) for value in values))
        cmap = plt.get_cmap("tab10")
        return {
            value: to_hex(cmap(index % cmap.N))
            for index, value in enumerate(values)
        }

    genotype_values = [
        value for value in metadata_df[group_column]
        if not pd.isna(value)
    ]
    sex_values = [
        value for value in metadata_df[sex_col]
        if not pd.isna(value)
    ]

    plot_color_maps = {
        "genotype": build_default_color_map(genotype_values),
        "sex": build_default_color_map(sex_values)
    }

    group_colours_used = (
        progression_options["grouped_summary"]
        or progression_options["group_3d"]
        or progression_options["chronological_sequence"]
        or progression_options["chronological_heatmaps"]
    )

    use_custom_colors = False
    if group_colours_used:
        use_custom_colors = messagebox.askyesno(
            "Plot Colours",
            "Would you like to choose custom colours for Sex and Genotype groups?"
        )

    if use_custom_colors:
        color_specs = [
            ("genotype", group_column),
            ("sex", sex_col)
        ]

        for map_name, column_label in color_specs:
            for group_value in sorted(plot_color_maps[map_name].keys()):
                selected_color = colorchooser.askcolor(
                    title=f"Choose colour for {group_value}"
                )[1]

                if selected_color:
                    plot_color_maps[map_name][group_value] = selected_color

    any_plots_requested = any([
        progression_options["per_mouse"],
        progression_options["grouped_summary"],
        progression_options["individual_2d"],
        progression_options["individual_3d"],
        progression_options["group_3d"],
        progression_options["chronological_sequence"],
        progression_options["chronological_heatmaps"]
    ])

    show_plots = False
    if any_plots_requested:
        show_plots = messagebox.askyesno(
            "Plot Display",
            "Display plots in matplotlib windows?\n\n"
            "Yes = show plots\n"
            "No = only save images"
        )

    sex_line_styles = {}
    available_line_styles = ["-", "--", ":", "-."]
    for index, sex_value in enumerate(sorted(plot_color_maps["sex"].keys())):
        sex_line_styles[sex_value] = available_line_styles[index % len(available_line_styles)]

    def get_group_key(item, group_mode):
        mouse, genotype, sex, values = item
        genotype = clean_group_value(genotype)
        sex = clean_group_value(sex)

        if group_mode == "genotype":
            return genotype
        if group_mode == "sex":
            return sex
        if group_mode == "sex_genotype":
            return f"{sex} {genotype}"

        return "All"

    def grouped_items(data, group_mode):
        groups = {}

        for item in data:
            key = get_group_key(item, group_mode)
            groups.setdefault(key, []).append(item)

        return groups

    def subset_items(data, filter_mode, filter_value):
        filtered = []

        for item in data:
            mouse, genotype, sex, values = item
            if filter_mode == "sex" and clean_group_value(sex) == filter_value:
                filtered.append(item)
            elif filter_mode == "genotype" and clean_group_value(genotype) == filter_value:
                filtered.append(item)

        return filtered

    def unique_group_count(data, group_mode):
        return len(set(get_group_key(item, group_mode) for item in data))

    def get_group_style(items, group_mode):
        first_item = items[0]
        genotype = clean_group_value(first_item[1])
        sex = clean_group_value(first_item[2])

        if group_mode == "genotype":
            return plot_color_maps["genotype"].get(genotype), "-"
        if group_mode == "sex":
            return plot_color_maps["sex"].get(sex), "-"
        if group_mode == "sex_genotype":
            return (
                plot_color_maps["genotype"].get(genotype),
                sex_line_styles.get(sex, "-")
            )

        return None, "-"

    def build_group_color_gradient(base_color, number_of_colors):
        if base_color is None:
            base_color = to_hex(plt.get_cmap("tab10")(0))

        base_rgb = np.asarray(to_rgb(base_color), dtype=float)
        dark_rgb = np.clip(base_rgb * 0.45, 0, 1)
        light_rgb = np.clip(base_rgb + (1 - base_rgb) * 0.65, 0, 1)
        gradient = LinearSegmentedColormap.from_list(
            "group_progression",
            [dark_rgb, base_rgb, light_rgb]
        )

        return [
            gradient(index / max(number_of_colors - 1, 1))
            for index in range(number_of_colors)
        ]

    def style_3d_progression_axis(axis, maximum_event_group, z_min, z_max):
        """Match the original pellet plotter's clean panes and time-zero plane."""
        if np.isfinite(z_min) and np.isfinite(z_max):
            if z_min == z_max:
                z_min -= 0.1
                z_max += 0.1

            y_max = max(float(maximum_event_group), 1.0)
            plane_y, plane_z = np.meshgrid(
                np.asarray([0.0, y_max]),
                np.asarray([float(z_min), float(z_max)])
            )
            plane_x = np.zeros_like(plane_y)
            axis.plot_surface(
                plane_x,
                plane_y,
                plane_z,
                color="lightgrey",
                alpha=0.25,
                shade=False
            )

        axis.grid(False)
        axis.xaxis.pane.fill = False
        axis.yaxis.pane.fill = False
        axis.zaxis.pane.fill = False

    def checked_plot_path(destination, filename):
        """Return a save path and give a useful error before Windows/Pillow fails."""
        output_file = os.path.abspath(os.path.join(destination, filename))
        if os.name == "nt" and len(output_file) >= 260:
            raise OSError(
                "The plot output path is too long for this Windows/Python "
                f"environment ({len(output_file)} characters):\n{output_file}\n\n"
                "Choose or move the input data to a shorter parent folder. "
                "The filename was not silently truncated."
            )
        return output_file

    def finish_plot(tab, plot_folder, filename):
        destination = os.path.join(
            save_folder,
            "Plots",
            safe_filename_value(tab),
            safe_filename_value(plot_folder)
        )
        os.makedirs(destination, exist_ok=True)
        plt.savefig(checked_plot_path(destination, filename), dpi=300)

        if show_plots:
            plt.show()
        else:
            plt.close()

    def finish_figure(figure, tab, plot_folder, filename):
        folder_parts = [
            safe_filename_value(part)
            for part in str(plot_folder).replace("\\", "/").split("/")
            if part
        ]
        destination = os.path.join(
            save_folder,
            "Plots",
            safe_filename_value(tab),
            *folder_parts
        )
        os.makedirs(destination, exist_ok=True)
        output_file = checked_plot_path(destination, filename)
        figure.savefig(output_file, dpi=300)

        if show_plots:
            plt.show()
        else:
            plt.close(figure)

    def downsample_plot_matrix(time_values, matrix):
        factor = progression_options["downsample_factor"]
        if factor <= 1 or len(time_values) < factor:
            return time_values, matrix

        usable_rows = (len(time_values) // factor) * factor
        downsampled_time = np.nanmean(
            time_values[:usable_rows].reshape(-1, factor),
            axis=1
        )
        downsampled_matrix = np.nanmean(
            matrix[:usable_rows, :].reshape(-1, factor, matrix.shape[1]),
            axis=1
        )
        return downsampled_time, downsampled_matrix

    def restrict_3d_plot_range(time_values, matrix):
        if progression_options["time_range_mode"] != "Custom range":
            return time_values, matrix

        time_mask = (
            (time_values >= progression_options["time_start"])
            & (time_values <= progression_options["time_end"])
        )
        return time_values[time_mask], matrix[time_mask, :]

    def build_event_blocks(trials):
        group_size = progression_options["event_group_size"]
        blocks = []

        for start in range(0, trials.shape[1], group_size):
            end = min(start + group_size, trials.shape[1])
            block = trials[:, start:end]
            valid_counts = np.sum(~np.isnan(block), axis=1)
            mean_trace = np.nanmean(block, axis=1)
            sem_trace = np.divide(
                np.nanstd(block, axis=1),
                np.sqrt(valid_counts),
                out=np.zeros(block.shape[0], dtype=float),
                where=valid_counts > 0
            )
            blocks.append({
                "start": start + 1,
                "end": end,
                "mean": mean_trace,
                "sem": sem_trace
            })

        return blocks

    max_2d_blocks_per_figure = 20
    individual_2d_split_warning_shown = False

    def plot_individual_event_progression(mouse, trials, tab):
        nonlocal individual_2d_split_warning_shown

        if trials.size == 0 or trials.shape[1] == 0:
            return

        plot_time, plot_trials = downsample_plot_matrix(reference_time, trials)
        blocks = build_event_blocks(plot_trials)
        if not blocks:
            return

        mouse_token = safe_filename_value(mouse)
        line_width = progression_options["line_width"]

        if progression_options["individual_2d"]:
            number_of_pages = int(np.ceil(
                len(blocks) / max_2d_blocks_per_figure
            ))

            if number_of_pages > 1 and not individual_2d_split_warning_shown:
                messagebox.showwarning(
                    "Individual 2D Plots Split Across Files",
                    "At least one subject has too many event groups for a safe "
                    "single 2D image.\n\n"
                    "The individual 2D progression plots will be split "
                    f"automatically into pages of up to "
                    f"{max_2d_blocks_per_figure} event groups. This prevents "
                    "Matplotlib's maximum image-size error.\n\n"
                    "To create fewer pages on a future run, increase "
                    "'Events per group' in the Event-Progression Plot Options.",
                    parent=root
                )
                individual_2d_split_warning_shown = True

            for page_index in range(number_of_pages):
                page_start = page_index * max_2d_blocks_per_figure
                page_end = min(
                    page_start + max_2d_blocks_per_figure,
                    len(blocks)
                )
                page_blocks = blocks[page_start:page_end]

                figure, axes = plt.subplots(
                    len(page_blocks),
                    1,
                    figsize=(10, max(3, len(page_blocks) * 1.8)),
                    sharex=True
                )
                if len(page_blocks) == 1:
                    axes = [axes]

                for axis, block in zip(axes, page_blocks):
                    axis.plot(
                        plot_time,
                        block["mean"],
                        color="black",
                        linewidth=line_width
                    )
                    axis.fill_between(
                        plot_time,
                        block["mean"] - block["sem"],
                        block["mean"] + block["sem"],
                        color="black",
                        alpha=0.2,
                        edgecolor="none",
                        linewidth=0
                    )
                    axis.axvline(
                        0,
                        color="grey",
                        linestyle="--",
                        linewidth=1.0
                    )
                    block_label = (
                        f"E{block['start']}"
                        if block["start"] == block["end"]
                        else f"E{block['start']}-{block['end']}"
                    )
                    axis.set_ylabel(block_label, rotation=0, labelpad=28)
                    axis.spines["top"].set_visible(False)
                    axis.spines["right"].set_visible(False)

                axes[-1].set_xlabel("Peri-event Time (s)")
                page_suffix = (
                    ""
                    if number_of_pages == 1
                    else f" - Part {page_index + 1} of {number_of_pages}"
                )
                figure.suptitle(
                    f"{mouse} {tab} Event Progression (2D){page_suffix}"
                )
                figure.tight_layout()

                if number_of_pages == 1:
                    filename = (
                        f"{mouse_token}_{safe_filename_value(tab)}_2D.png"
                    )
                else:
                    filename = (
                        f"{mouse_token}_{safe_filename_value(tab)}_2D_"
                        f"Part_{page_index + 1:02d}_of_{number_of_pages:02d}.png"
                    )

                finish_figure(
                    figure,
                    tab,
                    os.path.join("Individual_Event_Progression", "2D"),
                    filename
                )

        if progression_options["individual_3d"]:
            three_d_time, three_d_trials = restrict_3d_plot_range(
                plot_time,
                plot_trials
            )
            if len(three_d_time) == 0:
                return

            three_d_blocks = build_event_blocks(three_d_trials)
            if not three_d_blocks:
                return

            figure = plt.figure(figsize=(12, 9))
            axis = figure.add_subplot(111, projection="3d")
            cmap = plt.get_cmap("viridis")

            for block_index, block in enumerate(three_d_blocks):
                color = cmap(block_index / max(len(three_d_blocks) - 1, 1))
                y_values = np.full_like(three_d_time, block_index, dtype=float)
                axis.plot(
                    three_d_time,
                    y_values,
                    block["mean"],
                    color=color,
                    linewidth=line_width
                )

            finite_block_values = [
                block["mean"][np.isfinite(block["mean"])]
                for block in three_d_blocks
                if np.any(np.isfinite(block["mean"]))
            ]
            if finite_block_values:
                finite_z = np.concatenate(finite_block_values)
                individual_z_min = float(np.nanmin(finite_z))
                individual_z_max = float(np.nanmax(finite_z))
                individual_padding = max(
                    (individual_z_max - individual_z_min) * 0.05,
                    0.1
                )
                individual_z_min -= individual_padding
                individual_z_max += individual_padding
                axis.set_zlim(individual_z_min, individual_z_max)
                style_3d_progression_axis(
                    axis,
                    len(three_d_blocks) - 1,
                    individual_z_min,
                    individual_z_max
                )

            axis.set_xlabel("Peri-event Time (s)")
            axis.set_ylabel("Event Group")
            axis.set_zlabel("Z-score")
            axis.set_title(f"{mouse} {tab} Event Progression (3D)")
            axis.view_init(
                elev=progression_options["elevation"],
                azim=progression_options["azimuth"]
            )
            finish_figure(
                figure,
                tab,
                os.path.join("Individual_Event_Progression", "3D"),
                f"{mouse_token}_{safe_filename_value(tab)}_3D.png"
            )

    def plot_trace_overlay(data, tab, group_mode, group_label, filename_suffix, plot_folder, title_suffix=None):
        if len(data) == 0:
            return

        groups = grouped_items(data, group_mode)
        if len(groups) == 0:
            return

        plt.figure()

        for group_name in sorted(groups.keys()):
            matrices = [item[3] for item in groups[group_name]]
            if len(matrices) == 0:
                continue

            combined = np.hstack(matrices)
            mean_trace = np.nanmean(combined, axis=1)
            sem_trace = np.nanstd(combined, axis=1) / np.sqrt(combined.shape[1])
            color, line_style = get_group_style(groups[group_name], group_mode)

            plt.plot(
                reference_time,
                mean_trace,
                label=group_name,
                color=color,
                linestyle=line_style
            )
            plt.fill_between(
                reference_time,
                mean_trace - sem_trace,
                mean_trace + sem_trace,
                color=color,
                alpha=0.3,
                edgecolor="none",
                linewidth=0
            )

        plt.axvline(0, linestyle="--")
        plt.xlim(reference_time.min(), reference_time.max())
        plt.xlabel("Time (s)")
        plt.ylabel("Z-score")

        title_group = title_suffix if title_suffix else f"{group_label} Overlay"
        plt.title(f"{tab} Events ({title_group})")
        plt.legend()

        finish_plot(
            tab,
            plot_folder,
            f"{safe_filename_value(tab)}_Trace{filename_suffix}.png"
        )

    def pad_group_values(items):
        values = [item[3] for item in items]
        if len(values) == 0:
            return None

        max_len = max(len(v) for v in values)
        padded = []

        for v in values:
            arr = np.full(max_len, np.nan)
            arr[:len(v)] = v
            padded.append(arr)

        return np.vstack(padded)

    def transform_event_values(combined, transform):
        if transform == "cumulative":
            return np.nancumsum(combined, axis=1)

        if transform == "cummean":
            cumulative = np.nancumsum(combined, axis=1)
            valid_counts = np.cumsum(~np.isnan(combined), axis=1)
            return np.divide(
                cumulative,
                valid_counts,
                out=np.full_like(cumulative, np.nan, dtype=float),
                where=valid_counts != 0
            )

        return combined

    def plot_event_metric_overlay(
        data,
        tab,
        y_label,
        title_label,
        filename_base,
        group_mode,
        group_label,
        filename_suffix,
        plot_folder,
        transform="raw",
        title_suffix=None
    ):
        if len(data) == 0:
            return

        # "Overlay" is already apparent from the figure and its title. Removing
        # this fixed suffix keeps filenames descriptive without duplicating it.
        compact_filename_base = str(filename_base)
        if compact_filename_base.endswith("_Overlay"):
            compact_filename_base = compact_filename_base[:-len("_Overlay")]

        groups = grouped_items(data, group_mode)
        if len(groups) == 0:
            return

        plt.figure()

        for group_name in sorted(groups.keys()):
            combined = pad_group_values(groups[group_name])
            if combined is None:
                continue

            combined = transform_event_values(combined, transform)
            mean_vals = np.nanmean(combined, axis=0)
            sem_vals = np.nanstd(combined, axis=0) / np.sqrt(combined.shape[0])
            events = np.arange(1, len(mean_vals) + 1)
            color, line_style = get_group_style(groups[group_name], group_mode)

            plt.plot(
                events,
                mean_vals,
                label=group_name,
                color=color,
                linestyle=line_style
            )
            plt.fill_between(
                events,
                mean_vals - sem_vals,
                mean_vals + sem_vals,
                color=color,
                alpha=0.3,
                edgecolor="none",
                linewidth=0
            )

        plt.xlabel("Event Number")
        plt.ylabel(y_label)

        title_group = title_suffix if title_suffix else f"{group_label} Overlay"
        plt.title(f"{tab} {title_label} ({title_group})")
        plt.legend()

        finish_plot(
            tab,
            plot_folder,
            (
                f"{safe_filename_value(tab)}_"
                f"{compact_filename_base}"
                f"{filename_suffix}.png"
            )
        )

    def plot_grouping_set(trace_data, metric_specs, tab, group_mode, group_label, filename_suffix, plot_folder, title_suffix=None):
        plot_trace_overlay(
            trace_data,
            tab,
            group_mode=group_mode,
            group_label=group_label,
            filename_suffix=filename_suffix,
            plot_folder=plot_folder,
            title_suffix=title_suffix
        )

        for spec in metric_specs:
            plot_event_metric_overlay(
                spec["data"],
                tab,
                y_label=spec["y_label"],
                title_label=spec["title_label"],
                filename_base=spec["filename_base"],
                group_mode=group_mode,
                group_label=group_label,
                filename_suffix=filename_suffix,
                plot_folder=plot_folder,
                transform=spec.get("transform", "raw"),
                title_suffix=title_suffix
            )

    def plot_available_groupings(trace_data, metric_specs, tab):
        plot_grouping_set(
            trace_data,
            metric_specs,
            tab,
            group_mode="genotype",
            group_label=group_column,
            filename_suffix="",
            plot_folder=group_column
        )

        if unique_group_count(trace_data, "sex") > 1:
            plot_grouping_set(
                trace_data,
                metric_specs,
                tab,
                group_mode="sex",
                group_label=sex_col,
                filename_suffix=f"_by_{safe_filename_value(sex_col)}",
                plot_folder=sex_col
            )

        if unique_group_count(trace_data, "sex_genotype") > 1:
            plot_grouping_set(
                trace_data,
                metric_specs,
                tab,
                group_mode="sex_genotype",
                group_label=f"{sex_col} x {group_column}",
                filename_suffix=f"_by_{safe_filename_value(sex_col)}_{safe_filename_value(group_column)}",
                plot_folder=f"{sex_col}_x_{group_column}"
            )

        for sex_value in sorted(set(clean_group_value(item[2]) for item in trace_data)):
            subset_trace = subset_items(trace_data, "sex", sex_value)
            if unique_group_count(subset_trace, "genotype") <= 1:
                continue

            subset_specs = []
            for spec in metric_specs:
                subset_spec = spec.copy()
                subset_spec["data"] = subset_items(spec["data"], "sex", sex_value)
                subset_specs.append(subset_spec)

            plot_grouping_set(
                subset_trace,
                subset_specs,
                tab,
                group_mode="genotype",
                group_label=group_column,
                filename_suffix=f"_{safe_filename_value(sex_col)}_{safe_filename_value(sex_value)}_by_{safe_filename_value(group_column)}",
                plot_folder="Subgroup_Comparisons",
                title_suffix=f"{group_column} Overlay, {sex_col}: {sex_value}"
            )

        for genotype_value in sorted(set(clean_group_value(item[1]) for item in trace_data)):
            subset_trace = subset_items(trace_data, "genotype", genotype_value)
            if unique_group_count(subset_trace, "sex") <= 1:
                continue

            subset_specs = []
            for spec in metric_specs:
                subset_spec = spec.copy()
                subset_spec["data"] = subset_items(spec["data"], "genotype", genotype_value)
                subset_specs.append(subset_spec)

            plot_grouping_set(
                subset_trace,
                subset_specs,
                tab,
                group_mode="sex",
                group_label=sex_col,
                filename_suffix=f"_{safe_filename_value(group_column)}_{safe_filename_value(genotype_value)}_by_{safe_filename_value(sex_col)}",
                plot_folder="Subgroup_Comparisons",
                title_suffix=f"{sex_col} Overlay, {group_column}: {genotype_value}"
            )

    def plot_group_3d_comparison(
        trace_data,
        tab,
        group_mode,
        group_label,
        filename_suffix,
        title_suffix=None
    ):
        groups = grouped_items(trace_data, group_mode)
        if len(groups) <= 1:
            return

        group_size = progression_options["event_group_size"]
        line_width = progression_options["line_width"]
        prepared_groups = {}
        all_z_values = []
        common_time = None

        for group_name in sorted(groups.keys()):
            group_items = groups[group_name]
            downsampled_items = []
            max_events = 0

            for item in group_items:
                plot_time, plot_trials = downsample_plot_matrix(
                    reference_time,
                    item[3]
                )
                plot_time, plot_trials = restrict_3d_plot_range(
                    plot_time,
                    plot_trials
                )
                if len(plot_time) == 0:
                    continue
                common_time = plot_time
                downsampled_items.append(plot_trials)
                max_events = max(max_events, plot_trials.shape[1])

            block_traces = []
            for start in range(0, max_events, group_size):
                mouse_block_traces = []

                for plot_trials in downsampled_items:
                    if start >= plot_trials.shape[1]:
                        continue
                    end = min(start + group_size, plot_trials.shape[1])
                    mouse_block_traces.append(
                        np.nanmean(plot_trials[:, start:end], axis=1)
                    )

                if not mouse_block_traces:
                    continue

                group_block_mean = np.nanmean(
                    np.column_stack(mouse_block_traces),
                    axis=1
                )
                block_traces.append(group_block_mean)
                all_z_values.append(group_block_mean)

            prepared_groups[group_name] = block_traces

        if not all_z_values or common_time is None:
            return

        finite_arrays = [
            values[np.isfinite(values)]
            for values in all_z_values
            if np.any(np.isfinite(values))
        ]
        if not finite_arrays:
            return

        finite_z = np.concatenate(finite_arrays)
        if finite_z.size == 0:
            return

        z_min = float(np.nanmin(finite_z))
        z_max = float(np.nanmax(finite_z))
        z_padding = max((z_max - z_min) * 0.05, 0.1)

        group_names = sorted(prepared_groups.keys())
        figure = plt.figure(figsize=(7 * len(group_names), 7))

        for panel_index, group_name in enumerate(group_names, start=1):
            axis = figure.add_subplot(
                1,
                len(group_names),
                panel_index,
                projection="3d"
            )
            base_color, line_style = get_group_style(
                groups[group_name],
                group_mode
            )
            block_colors = build_group_color_gradient(
                base_color,
                len(prepared_groups[group_name])
            )

            group_block_traces = prepared_groups[group_name]
            if len(group_block_traces) > 1:
                surface_color = (
                    base_color
                    if base_color is not None
                    else to_hex(plt.get_cmap("tab10")(0))
                )
                surface_x, surface_y = np.meshgrid(
                    common_time,
                    np.arange(len(group_block_traces), dtype=float)
                )
                surface_z = np.vstack(group_block_traces)
                axis.plot_surface(
                    surface_x,
                    surface_y,
                    surface_z,
                    color=surface_color,
                    alpha=0.08,
                    edgecolor="none",
                    linewidth=0,
                    antialiased=False,
                    shade=False
                )

            # With many event groups, retain the complete surface but draw fewer
            # outlines so the progression reads as a waveform instead of a mesh.
            group_line_width = min(line_width, 0.7)
            line_stride = 2 if len(group_block_traces) > 15 else 1
            plotted_block_indices = list(
                range(0, len(group_block_traces), line_stride)
            )
            if (
                group_block_traces
                and plotted_block_indices[-1] != len(group_block_traces) - 1
            ):
                plotted_block_indices.append(len(group_block_traces) - 1)

            for block_index in plotted_block_indices:
                block_trace = group_block_traces[block_index]
                y_values = np.full_like(
                    common_time,
                    block_index,
                    dtype=float
                )
                axis.plot(
                    common_time,
                    y_values,
                    block_trace,
                    color=block_colors[block_index],
                    linestyle=line_style,
                    linewidth=group_line_width
                )

            axis.set_xlim(common_time.min(), common_time.max())
            axis.set_zlim(z_min - z_padding, z_max + z_padding)
            axis.set_xlabel("Peri-event Time (s)")
            axis.set_ylabel("Event Group")
            axis.set_zlabel("Z-score")
            axis.set_title(str(group_name))
            axis.view_init(
                elev=progression_options["elevation"],
                azim=progression_options["azimuth"]
            )
            style_3d_progression_axis(
                axis,
                len(prepared_groups[group_name]) - 1,
                z_min - z_padding,
                z_max + z_padding
            )

        comparison_title = (
            title_suffix
            if title_suffix
            else f"{tab} Event Progression by {group_label}"
        )
        figure.suptitle(comparison_title)
        finish_figure(
            figure,
            tab,
            "Group_3D_Comparisons",
            f"{safe_filename_value(tab)}_Group3D{filename_suffix}.png"
        )

    def plot_available_group_3d(trace_data, tab):
        if not progression_options["group_3d"]:
            return

        plot_group_3d_comparison(
            trace_data,
            tab,
            group_mode="genotype",
            group_label=group_column,
            filename_suffix=f"_by_{safe_filename_value(group_column)}"
        )

        if unique_group_count(trace_data, "sex") > 1:
            plot_group_3d_comparison(
                trace_data,
                tab,
                group_mode="sex",
                group_label=sex_col,
                filename_suffix=f"_by_{safe_filename_value(sex_col)}"
            )

        if unique_group_count(trace_data, "sex_genotype") > 1:
            plot_group_3d_comparison(
                trace_data,
                tab,
                group_mode="sex_genotype",
                group_label=f"{sex_col} x {group_column}",
                filename_suffix=(
                    f"_by_{safe_filename_value(sex_col)}_"
                    f"{safe_filename_value(group_column)}"
                )
            )

        for sex_value in sorted(set(
            clean_group_value(item[2]) for item in trace_data
        )):
            subset_trace = subset_items(trace_data, "sex", sex_value)
            if unique_group_count(subset_trace, "genotype") <= 1:
                continue
            plot_group_3d_comparison(
                subset_trace,
                tab,
                group_mode="genotype",
                group_label=group_column,
                filename_suffix=(
                    f"_{safe_filename_value(sex_col)}_"
                    f"{safe_filename_value(sex_value)}_by_"
                    f"{safe_filename_value(group_column)}"
                ),
                title_suffix=(
                    f"{tab} Event Progression by {group_column}, "
                    f"{sex_col}: {sex_value}"
                )
            )

        for genotype_value in sorted(set(
            clean_group_value(item[1]) for item in trace_data
        )):
            subset_trace = subset_items(
                trace_data,
                "genotype",
                genotype_value
            )
            if unique_group_count(subset_trace, "sex") <= 1:
                continue
            plot_group_3d_comparison(
                subset_trace,
                tab,
                group_mode="sex",
                group_label=sex_col,
                filename_suffix=(
                    f"_{safe_filename_value(group_column)}_"
                    f"{safe_filename_value(genotype_value)}_by_"
                    f"{safe_filename_value(sex_col)}"
                ),
                title_suffix=(
                    f"{tab} Event Progression by {sex_col}, "
                    f"{group_column}: {genotype_value}"
                )
            )

    # ------------------------------------------------------------
    # STORAGE
    # ------------------------------------------------------------
    combined_raw = {tab: [] for tab in selected_tabs}
    combined_means = {tab: [] for tab in selected_tabs}
    combined_max = {tab: [] for tab in selected_tabs}

    combined_max_time = {tab: [] for tab in selected_tabs}
    combined_time_to_baseline = {tab: [] for tab in selected_tabs}

    combined_auc = {tab: [] for tab in selected_tabs}
    combined_meanz = {tab: [] for tab in selected_tabs}

    reference_time = None

    # ------------------------------------------------------------
    # OPTIONAL CHRONOLOGICAL PLOTS (same display preference as all other plots)
    # ------------------------------------------------------------
    if progression_options["chronological_sequence"] or progression_options["chronological_heatmaps"]:
        try:
            chronological_records = _chrono_records(
                metadata_df, file_map, selected_tabs, mouse_id_col, sex_col, group_column)
            chronological_folder = os.path.join(
                save_folder, "Plots", "Chronological_event_plots")
            _chrono_export(
                chronological_records, chronological_folder,
                {event: color for event, color in chronological_colors.items() if event in selected_tabs},
                sequence=progression_options["chronological_sequence"],
                heatmaps=progression_options["chronological_heatmaps"],
                show_plots=show_plots,
                metadata_headers=(mouse_id_col, sex_col, group_column),
                metadata_color_maps=plot_color_maps,
                trace_page_events=progression_options["trace_page_events"],
                wide_png=progression_options["trace_wide_png"],
                wide_svg=progression_options["trace_wide_svg"],
                paginated_png=progression_options["trace_paginated_png"])
        except Exception as error:
            messagebox.showerror("Chronological plot error", str(error))
            root.destroy()
            return

    # ------------------------------------------------------------
    # EXTRACTION
    # ------------------------------------------------------------
    for tab in selected_tabs:


        for _, row in metadata_df.iterrows():

            file = file_map[row["Filename"]]

            genotype = row[group_column]
            mouse = str(row[mouse_id_col])
            sex = row[sex_col]
            
            try:
                df = pd.read_excel(file, sheet_name=tab, header=None)

                custom_idx = df[df.eq("Custom name").any(axis=1)].index[0]
                event_note_idx = df[df.eq("Event note").any(axis=1)].index[0]
                baseline_idx = df[df.astype(str).apply(
                    lambda r: r.str.contains("Time to baseline", case=False).any(), axis=1
                )].index[0]

            except Exception:
                print(f"Skipping {tab} for {row['Filename']} (invalid structure)")
                continue

            data_start = custom_idx + 1

            time_series = pd.to_numeric(df.iloc[data_start:, 1], errors="coerce")

            event_notes = df.iloc[event_note_idx, :]
            event_columns = event_notes[event_notes == tab].index

            trial_df = df.iloc[data_start:, event_columns]
            trial_df = trial_df.apply(pd.to_numeric, errors="coerce")

            baseline_vals = pd.to_numeric(df.iloc[baseline_idx, event_columns], errors="coerce").values

            time_series = time_series.reset_index(drop=True)
            trial_df = trial_df.reset_index(drop=True)

            valid_rows = ~time_series.isna()

            time_vector = time_series.loc[valid_rows].values
            trial_matrix = trial_df.loc[valid_rows].values

            # ------------------------------------------------------------
            # CALCULATE AUC AND MEAN Z WINDOW
            # ------------------------------------------------------------

            auc_mask = (
                (time_vector >= auc_window["start"]) &
                (time_vector <= auc_window["end"])
            )

            if not np.any(auc_mask):
                messagebox.showerror(
                    "Window Error",
                    f"No data points found in the AUC / Mean Z window "
                    f"({auc_window['start']}–{auc_window['end']} s)"
                )
                root.destroy()
                return

            auc_vals = np.trapz(
                trial_matrix[auc_mask, :],
                x=time_vector[auc_mask],
                axis=0
            )

            meanz_vals = np.nanmean(
                trial_matrix[auc_mask, :],
                axis=0
            )

            # ------------------------------------------------------------
            # CALCULATE PEAK VALUE AND LATENCY WITHIN PEAK WINDOW
            # ------------------------------------------------------------

            peak_mask = (
                (time_vector >= peak_window["start"]) &
                (time_vector <= peak_window["end"])
            )

            if not np.any(peak_mask):
                messagebox.showerror(
                    "Window Error",
                    f"No data points found in the Peak Z-score window "
                    f"({peak_window['start']}–{peak_window['end']} s)"
                )
                root.destroy()
                return

            peak_times = time_vector[peak_mask]
            peak_trials = trial_matrix[peak_mask, :]
            max_vals = np.full(peak_trials.shape[1], np.nan)
            max_time_vals = np.full(peak_trials.shape[1], np.nan)

            for trial_idx in range(peak_trials.shape[1]):
                trial_values = peak_trials[:, trial_idx]
                valid_peak_rows = ~np.isnan(trial_values)

                if not np.any(valid_peak_rows):
                    continue

                valid_values = trial_values[valid_peak_rows]
                valid_times = peak_times[valid_peak_rows]
                peak_idx = np.argmax(valid_values)

                max_vals[trial_idx] = valid_values[peak_idx]
                max_time_vals[trial_idx] = valid_times[peak_idx]

            if reference_time is None:
                reference_time = time_vector
            else:
                if not np.allclose(reference_time, time_vector, atol=1e-10):
                    messagebox.showerror(
                        "Timestamp Error",
                        f"Time vectors differ in file:\n{row['Filename']}"
                    )
                    root.destroy()
                    return

            combined_raw[tab].append((mouse, genotype, sex, trial_matrix))
            combined_means[tab].append((mouse, genotype, sex, np.nanmean(trial_matrix, axis=1)))
            combined_max[tab].append((mouse, genotype, sex, max_vals))
            combined_max_time[tab].append((mouse, genotype, sex, max_time_vals))
            combined_time_to_baseline[tab].append((mouse, genotype, sex, baseline_vals))
            combined_auc[tab].append((mouse, genotype, sex, auc_vals))
            combined_meanz[tab].append((mouse, genotype, sex, meanz_vals))


        # ------------------------------------------------------------
        # STACKED PER-MOUSE MEAN ± SEM PLOT
        # ------------------------------------------------------------
        if len(combined_raw[tab]) == 0:
            print(f"No data found for {tab}, skipping.")
            continue
        
        if progression_options["per_mouse"]:
            n_mice = len(combined_raw[tab])

            fig, axes = plt.subplots(n_mice, 1, figsize=(8, 2*n_mice), sharex=True)

            if n_mice == 1:
                axes = [axes]

            for ax, (mouse, geno, sex, trials) in zip(axes, combined_raw[tab]):

                mean_trace = np.nanmean(trials, axis=1)
                sem_trace = np.nanstd(trials, axis=1) / np.sqrt(trials.shape[1])

                ax.plot(reference_time, mean_trace, color="black", linewidth=2)

                ax.fill_between(
                    reference_time,
                    mean_trace - sem_trace,
                    mean_trace + sem_trace,
                    alpha=0.3
                )

                ax.axvline(0, linestyle="--")
                ax.set_xlim(reference_time.min(), reference_time.max())

                ax.set_ylabel(mouse)

            axes[-1].set_xlabel("Time (s)")
            fig.suptitle(f"{tab} — Per Mouse Mean ± SEM")

            plt.tight_layout()
            finish_plot(
                tab,
                "Per_Mouse",
                f"{safe_filename_value(tab)}_PerMouse.png"
            )

        if (
            progression_options["individual_2d"]
            or progression_options["individual_3d"]
        ):
            for mouse, genotype, sex, trials in combined_raw[tab]:
                plot_individual_event_progression(
                    mouse,
                    trials,
                    tab
                )

        # ------------------------------------------------------------
        # GROUPED OVERLAY PLOTS
        # ------------------------------------------------------------
        metric_specs = [
            {
                "data": combined_max[tab],
                "y_label": f"Peak Z-score ({peak_window['start']}-{peak_window['end']} s)",
                "title_label": f"Peak Value ({peak_window['start']}-{peak_window['end']} s)",
                "filename_base": f"PeakValue_{peak_window['start']}_{peak_window['end']}s_Overlay"
            },
            {
                "data": combined_max[tab],
                "y_label": f"Cumulative Peak Z-score ({peak_window['start']}-{peak_window['end']} s)",
                "title_label": f"Cumulative Peak Value ({peak_window['start']}-{peak_window['end']} s)",
                "filename_base": f"CumPeak_{peak_window['start']}_{peak_window['end']}s_Overlay",
                "transform": "cumulative"
            },
            {
                "data": combined_max[tab],
                "y_label": f"Cumulative Mean Peak Z-score ({peak_window['start']}-{peak_window['end']} s)",
                "title_label": f"Cumulative Mean Peak Value ({peak_window['start']}-{peak_window['end']} s)",
                "filename_base": f"CumMeanPeak_{peak_window['start']}_{peak_window['end']}s_Overlay",
                "transform": "cummean"
            },
            {
                "data": combined_max_time[tab],
                "y_label": f"Time of Peak (s; {peak_window['start']}-{peak_window['end']} s window)",
                "title_label": f"Peak Value Time ({peak_window['start']}-{peak_window['end']} s)",
                "filename_base": f"PeakValTime_{peak_window['start']}_{peak_window['end']}s_Overlay"
            },
            {
                "data": combined_time_to_baseline[tab],
                "y_label": "Time to Baseline (s)",
                "title_label": "Time to Baseline",
                "filename_base": "TimeToBaseline_Overlay"
            },
            {
                "data": combined_time_to_baseline[tab],
                "y_label": "Cumulative Time To Baseline (s)",
                "title_label": "Cumulative Time To Baseline",
                "filename_base": "CumBaseline_Overlay",
                "transform": "cumulative"
            },
            {
                "data": combined_time_to_baseline[tab],
                "y_label": "Cumulative Mean Time To Baseline (s)",
                "title_label": "CumMean Time To Baseline",
                "filename_base": "CumMeanBaseline_Overlay",
                "transform": "cummean"
            },
            {
                "data": combined_auc[tab],
                "y_label": f"AUC ({auc_window['start']}-{auc_window['end']} s)",
                "title_label": f"AUC ({auc_window['start']}-{auc_window['end']} s)",
                "filename_base": f"AUC_{auc_window['start']}_{auc_window['end']}s_Overlay"
            },
            {
                "data": combined_auc[tab],
                "y_label": f"Cumulative AUC ({auc_window['start']}-{auc_window['end']} s)",
                "title_label": f"Cumulative AUC ({auc_window['start']}-{auc_window['end']} s)",
                "filename_base": f"CumAUC_{auc_window['start']}_{auc_window['end']}s_Overlay",
                "transform": "cumulative"
            },
            {
                "data": combined_auc[tab],
                "y_label": f"Cumulative Mean AUC ({auc_window['start']}-{auc_window['end']} s)",
                "title_label": f"CumMean AUC ({auc_window['start']}-{auc_window['end']} s)",
                "filename_base": f"CumMeanAUC_{auc_window['start']}_{auc_window['end']}s_Overlay",
                "transform": "cummean"
            },
            {
                "data": combined_meanz[tab],
                "y_label": f"Mean Z ({auc_window['start']}-{auc_window['end']} s)",
                "title_label": f"Mean Z Window ({auc_window['start']}-{auc_window['end']} s)",
                "filename_base": f"MeanZ_{auc_window['start']}_{auc_window['end']}s_Overlay"
            }
        ]

        if progression_options["grouped_summary"]:
            plot_available_groupings(combined_raw[tab], metric_specs, tab)

        plot_available_group_3d(combined_raw[tab], tab)

    # ------------------------------------------------------------
    # EXPORT COMBINED EXCEL
    # ------------------------------------------------------------
    output_path = os.path.join(save_folder, "FED3_FP_Combined.xlsx")

    if all(len(combined_raw[tab]) == 0 for tab in selected_tabs):
        messagebox.showerror(
            "No Data",
            "No valid data was extracted.\nExcel file will not be created."
        )
        root.destroy()
        return

    with pd.ExcelWriter(output_path) as writer:

        parameter_rows = [
            {
                "Parameter": "AUC / Mean Z-score window",
                "Start Time (s)": auc_window["start"],
                "End Time (s)": auc_window["end"],
                "Value": ""
            },
            {
                "Parameter": "Peak Z-score window",
                "Start Time (s)": peak_window["start"],
                "End Time (s)": peak_window["end"],
                "Value": ""
            },
            {
                "Parameter": "Custom plot colours",
                "Start Time (s)": np.nan,
                "End Time (s)": np.nan,
                "Value": "Yes" if use_custom_colors else "No"
            }
        ]

        progression_parameter_labels = [
            ("Per-mouse mean ± SEM plots", "Yes" if progression_options["per_mouse"] else "No"),
            ("Grouped overlay and metric plots", "Yes" if progression_options["grouped_summary"] else "No"),
            ("Chronological wide PNG", "Yes" if progression_options["trace_wide_png"] else "No"),
            ("Chronological zoomable SVG", "Yes" if progression_options["trace_wide_svg"] else "No"),
            ("Chronological paginated PNGs", "Yes" if progression_options["trace_paginated_png"] else "No"),
            ("Events per chronological trace page", progression_options["trace_page_events"]),
            ("Chronological event trace plots", "Yes" if progression_options["chronological_sequence"] else "No"),
            ("Chronological Z-score heatmaps", "Yes" if progression_options["chronological_heatmaps"] else "No"),
            ("Display plots", "Yes" if show_plots else "No"),
            (
                "Individual 2D event-progression plots",
                "Yes" if progression_options["individual_2d"] else "No"
            ),
            (
                "Individual 3D event-progression plots",
                "Yes" if progression_options["individual_3d"] else "No"
            ),
            (
                "Group 3D comparison plots",
                "Yes" if progression_options["group_3d"] else "No"
            ),
            ("Events per progression group", progression_options["event_group_size"]),
            ("Plot downsampling factor", progression_options["downsample_factor"]),
            ("Progression plot line width", progression_options["line_width"]),
            ("3D time range", progression_options["time_range_mode"]),
            (
                "3D custom start time (s)",
                progression_options["time_start"]
            ),
            (
                "3D custom end time (s)",
                progression_options["time_end"]
            ),
            ("Vertical viewing angle", progression_options["elevation"]),
            ("Horizontal viewing angle", progression_options["azimuth"])
        ]

        for parameter_name, parameter_value in progression_parameter_labels:
            parameter_rows.append({
                "Parameter": parameter_name,
                "Start Time (s)": np.nan,
                "End Time (s)": np.nan,
                "Value": parameter_value
            })

        color_specs = [
            ("genotype", group_column),
            ("sex", sex_col)
        ]
        for map_name, column_label in color_specs:
            for group_value, color in sorted(plot_color_maps[map_name].items()):
                parameter_rows.append({
                    "Parameter": f"Plot colour - {column_label} - {group_value}",
                    "Start Time (s)": np.nan,
                    "End Time (s)": np.nan,
                    "Value": color
                })

        if progression_options["chronological_sequence"] or progression_options["chronological_heatmaps"]:
            for event in selected_tabs:
                parameter_rows.append({"Parameter": "Chronological colour - " + event,
                                       "Start Time (s)": np.nan, "End Time (s)": np.nan,
                                       "Value": chronological_colors[event]})

        parameters = pd.DataFrame(parameter_rows)
        parameters.to_excel(writer, sheet_name="Analysis Parameters", index=False)

        for tab in selected_tabs:

            if len(combined_raw[tab]) == 0:
                print(f"Skipping export for {tab} (no data)")
                continue

            # ------------------------------------------------------------
            # EVENTS
            # ------------------------------------------------------------
            data = {"Time (s)": reference_time}

            mouse_row = [""]
            geno_row = [""]
            sex_row = [""]

            for mouse, geno, sex, trials in combined_raw[tab]:

                for i in range(trials.shape[1]):

                    col = f"{mouse}_event{i+1}"
                    data[col] = trials[:, i]

                    mouse_row.append(mouse)
                    geno_row.append(geno)
                    sex_row.append(sex)

            df = pd.DataFrame(data)

            meta = pd.DataFrame(
                [mouse_row, geno_row, sex_row],
                index=[mouse_id_col, group_column, sex_col],
                columns=df.columns
            )

            final = pd.concat([meta, df])
            final.to_excel(writer, sheet_name=tab, float_format="%.10f")


            # ------------------------------------------------------------
            # EVENT MEANS
            # ------------------------------------------------------------
            data = {"Time (s)": reference_time}

            mouse_row = [""]
            geno_row = [""]
            sex_row = [""]

            for mouse, geno, sex, trace in combined_means[tab]:

                data[mouse] = trace

                mouse_row.append(mouse)
                geno_row.append(geno)
                sex_row.append(sex)

            df = pd.DataFrame(data)

            meta = pd.DataFrame(
                [mouse_row, geno_row, sex_row],
                index=[mouse_id_col, group_column, sex_col],
                columns=df.columns
            )

            final = pd.concat([meta, df])
            final.to_excel(writer, sheet_name=f"{tab} EventMeans", float_format="%.10f")


            # ------------------------------------------------------------
            # PEAK VALUES
            # ------------------------------------------------------------
            max_lists = []
            max_len = 0

            for mouse, geno, sex, vals in combined_max[tab]:
                max_lists.append((mouse, geno, sex, vals))
                max_len = max(max_len, len(vals))

            data = {"Event Number": np.arange(1, max_len + 1)}

            mouse_row = [""]
            geno_row = [""]
            sex_row = [""]

            for mouse, geno, sex, vals in max_lists:

                padded = np.full(max_len, np.nan)
                padded[:len(vals)] = vals

                data[mouse] = padded

                mouse_row.append(mouse)
                geno_row.append(geno)
                sex_row.append(sex)

            df = pd.DataFrame(data)

            meta = pd.DataFrame(
                [mouse_row, geno_row, sex_row],
                index=[mouse_id_col, group_column, sex_col],
                columns=df.columns
            )

            final = pd.concat([meta, df])
            final.to_excel(writer, sheet_name=f"{tab} PeakValues", float_format="%.10f")


            # ------------------------------------------------------------
            # PEAK VALUE TIME
            # ------------------------------------------------------------
            max_time_lists = []
            max_time_len = 0

            for mouse, geno, sex, vals in combined_max_time[tab]:
                max_time_lists.append((mouse, geno, sex, vals))
                max_time_len = max(max_time_len, len(vals))

            data = {"Event Number": np.arange(1, max_time_len + 1)}

            mouse_row = [""]
            geno_row = [""]
            sex_row = [""]

            for mouse, geno, sex, vals in max_time_lists:

                padded = np.full(max_time_len, np.nan)
                padded[:len(vals)] = vals

                data[mouse] = padded

                mouse_row.append(mouse)
                geno_row.append(geno)
                sex_row.append(sex)

            df = pd.DataFrame(data)

            meta = pd.DataFrame(
                [mouse_row, geno_row, sex_row],
                index=[mouse_id_col, group_column, sex_col],
                columns=df.columns
            )

            final = pd.concat([meta, df])
            final.to_excel(writer, sheet_name=f"{tab} PeakValTime", float_format="%.10f")


            # ------------------------------------------------------------
            # CUMULATIVE PEAK VALUES
            # ------------------------------------------------------------
            data = {"Event Number": np.arange(1, max_len + 1)}

            mouse_row = [""]
            geno_row = [""]
            sex_row = [""]

            for mouse, geno, sex, vals in max_lists:

                padded = np.full(max_len, np.nan)
                padded[:len(vals)] = vals

                cum_vals = np.nancumsum(np.nan_to_num(padded))

                data[mouse] = cum_vals

                mouse_row.append(mouse)
                geno_row.append(geno)
                sex_row.append(sex)

            df = pd.DataFrame(data)

            meta = pd.DataFrame(
                [mouse_row, geno_row, sex_row],
                index=[mouse_id_col, group_column, sex_col],
                columns=df.columns
            )

            final = pd.concat([meta, df])
            final.to_excel(writer, sheet_name=f"{tab} CumPeakVal", float_format="%.10f")


            # ------------------------------------------------------------
            # CUMULATIVE MEAN PEAK VALUES
            # ------------------------------------------------------------
            data = {"Event Number": np.arange(1, max_len + 1)}

            mouse_row = [""]
            geno_row = [""]
            sex_row = [""]

            for mouse, geno, sex, vals in max_lists:

                padded = np.full(max_len, np.nan)
                padded[:len(vals)] = vals

                cum_vals = np.nancumsum(np.nan_to_num(padded))

                valid_counts = np.cumsum(~np.isnan(padded))

                cum_mean = np.divide(
                    cum_vals,
                    valid_counts,
                    out=np.full_like(cum_vals, np.nan, dtype=float),
                    where=valid_counts != 0
                )

                data[mouse] = cum_mean

                mouse_row.append(mouse)
                geno_row.append(geno)
                sex_row.append(sex)

            df = pd.DataFrame(data)

            meta = pd.DataFrame(
                [mouse_row, geno_row, sex_row],
                index=[mouse_id_col, group_column, sex_col],
                columns=df.columns
            )

            final = pd.concat([meta, df])
            final.to_excel(writer, sheet_name=f"{tab} CumMeanPeakVal", float_format="%.10f")


            # ------------------------------------------------------------
            # TIME TO BASELINE
            # ------------------------------------------------------------
            baseline_lists = []
            max_len = 0

            for mouse, geno, sex, vals in combined_time_to_baseline[tab]:
                baseline_lists.append((mouse, geno, sex, vals))
                max_len = max(max_len, len(vals))

            data = {"Event Number": np.arange(1, max_len + 1)}

            mouse_row = [""]
            geno_row = [""]
            sex_row = [""]

            for mouse, geno, sex, vals in baseline_lists:

                padded = np.full(max_len, np.nan)
                padded[:len(vals)] = vals

                data[mouse] = padded

                mouse_row.append(mouse)
                geno_row.append(geno)
                sex_row.append(sex)

            df = pd.DataFrame(data)

            meta = pd.DataFrame(
                [mouse_row, geno_row, sex_row],
                index=[mouse_id_col, group_column, sex_col],
                columns=df.columns
            )

            final = pd.concat([meta, df])
            final.to_excel(writer, sheet_name=f"{tab} TimeBaseline", float_format="%.10f")


            # ------------------------------------------------------------
            # CUM TIME TO BASELINE
            # ------------------------------------------------------------
            data = {"Event Number": np.arange(1, max_len + 1)}

            mouse_row = [""]
            geno_row = [""]
            sex_row = [""]

            for mouse, geno, sex, vals in baseline_lists:

                padded = np.full(max_len, np.nan)
                padded[:len(vals)] = vals

                cum_vals = np.nancumsum(np.nan_to_num(padded))

                data[mouse] = cum_vals

                mouse_row.append(mouse)
                geno_row.append(geno)
                sex_row.append(sex)

            df = pd.DataFrame(data)

            meta = pd.DataFrame(
                [mouse_row, geno_row, sex_row],
                index=[mouse_id_col, group_column, sex_col],
                columns=df.columns
            )

            final = pd.concat([meta, df])
            final.to_excel(writer, sheet_name=f"{tab} CumTimeBaseline", float_format="%.10f")


            # ------------------------------------------------------------
            # CUMMEAN TIME TO BASELINE
            # ------------------------------------------------------------
            data = {"Event Number": np.arange(1, max_len + 1)}

            mouse_row = [""]
            geno_row = [""]
            sex_row = [""]

            for mouse, geno, sex, vals in baseline_lists:

                padded = np.full(max_len, np.nan)
                padded[:len(vals)] = vals

                cum_vals = np.nancumsum(np.nan_to_num(padded))

                valid_counts = np.cumsum(~np.isnan(padded))

                cum_mean = np.divide(
                    cum_vals,
                    valid_counts,
                    out=np.full_like(cum_vals, np.nan, dtype=float),
                    where=valid_counts != 0
                )

                data[mouse] = cum_mean

                mouse_row.append(mouse)
                geno_row.append(geno)
                sex_row.append(sex)

            df = pd.DataFrame(data)

            meta = pd.DataFrame(
                [mouse_row, geno_row, sex_row],
                index=[mouse_id_col, group_column, sex_col],
                columns=df.columns
            )

            final = pd.concat([meta, df])
            final.to_excel(writer, sheet_name=f"{tab} CumMeanTimeBaseline", float_format="%.10f")

            # ------------------------------------------------------------
            # AUC VALUES
            # ------------------------------------------------------------

            auc_lists = []
            max_len = 0

            for mouse, geno, sex, vals in combined_auc[tab]:
                auc_lists.append((mouse, geno, sex, vals))
                max_len = max(max_len, len(vals))

            data = {"Event Number": np.arange(1, max_len + 1)}

            mouse_row = [""]
            geno_row = [""]
            sex_row = [""]

            for mouse, geno, sex, vals in auc_lists:

                padded = np.full(max_len, np.nan)
                padded[:len(vals)] = vals

                data[mouse] = padded

                mouse_row.append(mouse)
                geno_row.append(geno)
                sex_row.append(sex)

            df = pd.DataFrame(data)

            meta = pd.DataFrame(
                [mouse_row, geno_row, sex_row],
                index=[mouse_id_col, group_column, sex_col],
                columns=df.columns
            )

            final = pd.concat([meta, df])
            final.to_excel(writer, sheet_name=f"{tab} AUC", float_format="%.10f")

            # ------------------------------------------------------------
            # CUM AUC
            # ------------------------------------------------------------

            data = {"Event Number": np.arange(1, max_len + 1)}

            mouse_row = [""]
            geno_row = [""]
            sex_row = [""]

            for mouse, geno, sex, vals in auc_lists:

                padded = np.full(max_len, np.nan)
                padded[:len(vals)] = vals

                cum_vals = np.nancumsum(np.nan_to_num(padded))

                data[mouse] = cum_vals

                mouse_row.append(mouse)
                geno_row.append(geno)
                sex_row.append(sex)

            df = pd.DataFrame(data)

            meta = pd.DataFrame(
                [mouse_row, geno_row, sex_row],
                index=[mouse_id_col, group_column, sex_col],
                columns=df.columns
            )

            final = pd.concat([meta, df])
            final.to_excel(writer, sheet_name=f"{tab} CumAUC", float_format="%.10f")

            # ------------------------------------------------------------
            # CUMMEAN AUC
            # ------------------------------------------------------------

            data = {"Event Number": np.arange(1, max_len + 1)}

            mouse_row = [""]
            geno_row = [""]
            sex_row = [""]

            for mouse, geno, sex, vals in auc_lists:

                padded = np.full(max_len, np.nan)
                padded[:len(vals)] = vals

                cum_vals = np.nancumsum(np.nan_to_num(padded))

                valid_counts = np.cumsum(~np.isnan(padded))

                cum_mean = np.divide(
                    cum_vals,
                    valid_counts,
                    out=np.full_like(cum_vals, np.nan, dtype=float),
                    where=valid_counts != 0
                )

                data[mouse] = cum_mean

                mouse_row.append(mouse)
                geno_row.append(geno)
                sex_row.append(sex)

            df = pd.DataFrame(data)

            meta = pd.DataFrame(
                [mouse_row, geno_row, sex_row],
                index=[mouse_id_col, group_column, sex_col],
                columns=df.columns
            )

            final = pd.concat([meta, df])
            final.to_excel(writer, sheet_name=f"{tab} CumMeanAUC", float_format="%.10f")

            # ------------------------------------------------------------
            # MEANZ WINDOW
            # ------------------------------------------------------------

            meanz_lists = []
            max_len = 0

            for mouse, geno, sex, vals in combined_meanz[tab]:
                meanz_lists.append((mouse, geno, sex, vals))
                max_len = max(max_len, len(vals))

            data = {"Event Number": np.arange(1, max_len + 1)}

            mouse_row = [""]
            geno_row = [""]
            sex_row = [""]

            for mouse, geno, sex, vals in meanz_lists:

                padded = np.full(max_len, np.nan)
                padded[:len(vals)] = vals

                data[mouse] = padded

                mouse_row.append(mouse)
                geno_row.append(geno)
                sex_row.append(sex)

            df = pd.DataFrame(data)

            meta = pd.DataFrame(
                [mouse_row, geno_row, sex_row],
                index=[mouse_id_col, group_column, sex_col],
                columns=df.columns
            )

            final = pd.concat([meta, df])
            final.to_excel(writer, sheet_name=f"{tab} MeanZ_window", float_format="%.10f")

    print("\nCombined Excel saved:", output_path)
    print("\nAnalysis complete.\n")

    root.destroy()
    return

if __name__ == "__main__":
    FED3_post_processing()

