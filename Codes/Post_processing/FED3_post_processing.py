import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, to_hex, to_rgb
from mpl_toolkits.mplot3d import Axes3D
import tkinter as tk
from tkinter import filedialog, messagebox, colorchooser
import os
import sys

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
    # PLOT AND EVENT-PROGRESSION OPTIONS
    # ------------------------------------------------------------
    plot_options_window = tk.Toplevel(root)
    plot_options_window.title("Event-Progression Plot Options")

    checkbox_variables = {}
    checkbox_options = [
        "Create individual 2D event-progression plots",
        "Create individual 3D event-progression plots",
        "Create group 3D comparison plots (shared axes)"
    ]

    for row_number, label in enumerate(checkbox_options):
        variable = tk.BooleanVar(
            master=plot_options_window,
            value=False
        )
        tk.Checkbutton(
            plot_options_window,
            text=label,
            variable=variable
        ).grid(
            row=row_number,
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
        start=3
    ):
        tk.Label(plot_options_window, text=label).grid(
            row=row_number, column=0, sticky="e", padx=8, pady=4
        )
        entry = tk.Entry(plot_options_window, width=10)
        entry.insert(0, default_value)
        entry.grid(row=row_number, column=1, sticky="w", padx=8, pady=4)
        option_entries[label] = entry

    time_range_mode = tk.StringVar(
        master=plot_options_window,
        value="Use full available range"
    )
    tk.Label(plot_options_window, text="3D time range").grid(
        row=6, column=0, sticky="e", padx=8, pady=4
    )
    tk.OptionMenu(
        plot_options_window,
        time_range_mode,
        "Use full available range",
        "Custom range"
    ).grid(row=6, column=1, sticky="w", padx=8, pady=4)

    tk.Label(plot_options_window, text="3D start time (s)").grid(
        row=7, column=0, sticky="e", padx=8, pady=4
    )
    custom_start_entry = tk.Entry(plot_options_window, width=10, state="disabled")
    custom_start_entry.grid(row=7, column=1, sticky="w", padx=8, pady=4)

    tk.Label(plot_options_window, text="3D end time (s)").grid(
        row=8, column=0, sticky="e", padx=8, pady=4
    )
    custom_end_entry = tk.Entry(plot_options_window, width=10, state="disabled")
    custom_end_entry.grid(row=8, column=1, sticky="w", padx=8, pady=4)

    viewing_option_defaults = [
        ("Vertical viewing angle", "25"),
        ("Horizontal viewing angle", "-60")
    ]
    for row_number, (label, default_value) in enumerate(
        viewing_option_defaults,
        start=9
    ):
        tk.Label(plot_options_window, text=label).grid(
            row=row_number, column=0, sticky="e", padx=8, pady=4
        )
        entry = tk.Entry(plot_options_window, width=10)
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

            progression_options.update({
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

    tk.Button(
        plot_options_window,
        text="Confirm",
        command=confirm_plot_options
    ).grid(row=11, column=0, columnspan=2, pady=10)

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

    def finish_plot(tab, plot_folder, filename):
        destination = os.path.join(
            save_folder,
            "Plots",
            safe_filename_value(tab),
            safe_filename_value(plot_folder)
        )
        os.makedirs(destination, exist_ok=True)
        plt.savefig(
            os.path.join(destination, filename),
            dpi=300
        )

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
        output_file = os.path.join(destination, filename)
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
            f"FED3_FP_{tab}_Overlay{filename_suffix}.png"
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
            f"FED3_FP_{tab}_{filename_base}{filename_suffix}.png"
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
            f"FED3_FP_{safe_filename_value(tab)}_Group3D{filename_suffix}.png"
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
            f"FED3_FP_{tab}_PerMouse.png"
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
