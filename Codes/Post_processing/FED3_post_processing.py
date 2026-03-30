import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tkinter as tk
from tkinter import filedialog, messagebox
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
        "Do you have an existing metadata file?"
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

            global metadata_df
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
        var = tk.BooleanVar(value=True)
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
    # ASK USER IF PLOTS SHOULD DISPLAY
    # ------------------------------------------------------------
    show_plots = messagebox.askyesno(
        "Plot Display",
        "Display plots in matplotlib windows?\n\nYes = show plots\nNo = only save images"
    )

    # ------------------------------------------------------------
    # ASK FOR ANALYSIS TIME WINDOW
    # ------------------------------------------------------------
    window_popup = tk.Toplevel(root)
    window_popup.title("Set AUC / Mean Z Time Window (s)")

    tk.Label(window_popup, text="Start Time (s):").grid(row=0, column=0)
    start_entry = tk.Entry(window_popup)
    start_entry.insert(0, "0")   # default
    start_entry.grid(row=0, column=1)

    tk.Label(window_popup, text="End Time (s):").grid(row=1, column=0)
    end_entry = tk.Entry(window_popup)
    end_entry.insert(0, "5")     # default
    end_entry.grid(row=1, column=1)

    time_window = {}

    def confirm_window():
        try:
            start_val = float(start_entry.get())
            end_val = float(end_entry.get())

            if start_val >= end_val:
                messagebox.showerror("Error", "Start must be less than End")
                return

            time_window["start"] = start_val
            time_window["end"] = end_val

            window_popup.destroy()

        except ValueError:
            messagebox.showerror("Error", "Please enter valid numbers")

    tk.Button(window_popup, text="Confirm", command=confirm_window)\
        .grid(row=2, column=0, columnspan=2)

    root.wait_window(window_popup)

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

        genotype_traces = {}

        for _, row in metadata_df.iterrows():

            file = file_map[row["Filename"]]

            genotype = row[group_column]
            mouse = str(row[mouse_id_col])
            sex = row[sex_col]
            
            try:
                df = pd.read_excel(file, sheet_name=tab, header=None)

                custom_idx = df[df.eq("Custom name").any(axis=1)].index[0]
                event_note_idx = df[df.eq("Event note").any(axis=1)].index[0]
                max_idx = df[df.eq("Max values").any(axis=1)].index[0]

                max_time_idx = df[df.astype(str).apply(
                    lambda r: r.str.contains("Time of max", case=False).any(), axis=1
                )].index[0]

                baseline_idx = df[df.astype(str).apply(
                    lambda r: r.str.contains("Time to baseline", case=False).any(), axis=1
                )].index[0]

            except Exception:
                print(f"Skipping {tab} for {row['Filename']} (invalid structure)")
                continue

            custom_idx = df[df.eq("Custom name").any(axis=1)].index[0]
            event_note_idx = df[df.eq("Event note").any(axis=1)].index[0]
            max_idx = df[df.eq("Max values").any(axis=1)].index[0]

            max_time_idx = df[df.astype(str).apply(lambda r: r.str.contains("Time of max", case=False).any(), axis=1)].index[0]
            baseline_idx = df[df.astype(str).apply(lambda r: r.str.contains("Time to baseline", case=False).any(), axis=1)].index[0]

            data_start = custom_idx + 1

            time_series = pd.to_numeric(df.iloc[data_start:, 1], errors="coerce")

            event_notes = df.iloc[event_note_idx, :]
            event_columns = event_notes[event_notes == tab].index

            trial_df = df.iloc[data_start:, event_columns]
            trial_df = trial_df.apply(pd.to_numeric, errors="coerce")

            max_vals = pd.to_numeric(df.iloc[max_idx, event_columns], errors="coerce").values
            max_time_vals = pd.to_numeric(df.iloc[max_time_idx, event_columns], errors="coerce").values
            baseline_vals = pd.to_numeric(df.iloc[baseline_idx, event_columns], errors="coerce").values

            time_series = time_series.reset_index(drop=True)
            trial_df = trial_df.reset_index(drop=True)

            valid_rows = ~time_series.isna()

            time_vector = time_series.loc[valid_rows].values
            trial_matrix = trial_df.loc[valid_rows].values

            # ------------------------------------------------------------
            # CALCULATE AUC AND MEANZ WINDOW
            # ------------------------------------------------------------

            post_mask = (
                (time_vector >= time_window["start"]) &
                (time_vector <= time_window["end"])
            )

            if not np.any(post_mask):
                messagebox.showerror(
                    "Window Error",
                    f"No data points found in selected window ({time_window['start']}–{time_window['end']} s)"
                )
                root.destroy()
                return

            auc_vals = np.trapz(
                trial_matrix[post_mask, :],
                x=time_vector[post_mask],
                axis=0
            )

            meanz_vals = np.nanmean(
                trial_matrix[post_mask, :],
                axis=0
            )

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

            if genotype not in genotype_traces:
                genotype_traces[genotype] = []

            genotype_traces[genotype].append(trial_matrix)

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

        stack_plot = os.path.join(save_folder, f"FED3_FP_{tab}_PerMouse.png")

        plt.tight_layout()
        plt.savefig(stack_plot, dpi=300)

        if show_plots:
            plt.show()
        else:
            plt.close()

        # ------------------------------------------------------------
        # GENOTYPE OVERLAY PLOT
        # ------------------------------------------------------------
        plt.figure()

        for genotype in sorted(genotype_traces.keys()):

            matrices = genotype_traces[genotype]

            combined = np.hstack(matrices)

            mean_trace = np.nanmean(combined, axis=1)
            sem_trace = np.nanstd(combined, axis=1) / np.sqrt(combined.shape[1])

            plt.plot(reference_time, mean_trace, label=genotype)

            plt.fill_between(
                reference_time,
                mean_trace - sem_trace,
                mean_trace + sem_trace,
                alpha=0.3
            )

        plt.axvline(0, linestyle="--")
        plt.xlim(reference_time.min(), reference_time.max())
        plt.xlabel("Time (s)")
        plt.ylabel("Z-score")
        plt.title(f"{tab} Events ({group_column} Overlay)")
        plt.legend()

        plot_path = os.path.join(save_folder, f"FED3_FP_{tab}_Overlay.png")
        plt.savefig(plot_path, dpi=300)

        if show_plots:
            plt.show()
        else:
            plt.close()

        # ------------------------------------------------------------
        # MAX VALUE OVERLAY PLOT
        # ------------------------------------------------------------
        plt.figure()

        for genotype in sorted(set([g for _, g, _, _ in combined_max[tab]])):

            geno_vals = []

            for mouse, geno, sex, vals in combined_max[tab]:
                if geno == genotype:
                    geno_vals.append(vals)

            if len(geno_vals) == 0:
                continue

            # ---- FIX: pad arrays to equal length ----
            max_len = max(len(v) for v in geno_vals)

            padded = []
            for v in geno_vals:
                arr = np.full(max_len, np.nan)
                arr[:len(v)] = v
                padded.append(arr)

            combined = np.vstack(padded)
            # -----------------------------------------

            mean_vals = np.nanmean(combined, axis=0)
            sem_vals = np.nanstd(combined, axis=0) / np.sqrt(combined.shape[0])

            events = np.arange(1, len(mean_vals)+1)

            plt.plot(events, mean_vals, label=genotype)

            plt.fill_between(
                events,
                mean_vals-sem_vals,
                mean_vals+sem_vals,
                alpha=0.3
            )

        plt.xlabel("Event Number")
        plt.ylabel("Max Z-score")
        plt.title(f"{tab} Max Value ({group_column} Overlay)")
        plt.legend()

        plt.savefig(os.path.join(save_folder, f"FED3_FP_{tab}_MaxValue_Overlay.png"), dpi=300)

        if show_plots:
            plt.show()
        else:
            plt.close()

        # ------------------------------------------------------------
        # CUMULATIVE MAX VALUE OVERLAY PLOT
        # ------------------------------------------------------------
        plt.figure()

        for genotype in sorted(set([g for _, g, _, _ in combined_max[tab]])):

            geno_vals = []

            for mouse, geno, sex, vals in combined_max[tab]:
                if geno == genotype:
                    geno_vals.append(vals)

            if len(geno_vals) == 0:
                continue

            max_len = max(len(v) for v in geno_vals)

            padded = []
            for v in geno_vals:
                arr = np.full(max_len, np.nan)
                arr[:len(v)] = v
                padded.append(arr)

            combined = np.vstack(padded)

            cumulative = np.nancumsum(combined, axis=1)

            mean_vals = np.nanmean(cumulative, axis=0)
            sem_vals = np.nanstd(cumulative, axis=0) / np.sqrt(cumulative.shape[0])

            events = np.arange(1, len(mean_vals)+1)

            plt.plot(events, mean_vals, label=genotype)

            plt.fill_between(events, mean_vals-sem_vals, mean_vals+sem_vals, alpha=0.3)

        plt.xlabel("Event Number")
        plt.ylabel("Cumulative Max Z-score")
        plt.title(f"{tab} Cumulative Max Value ({group_column} Overlay)")
        plt.legend()

        plt.savefig(os.path.join(save_folder, f"FED3_FP_{tab}_CumMax_Overlay.png"), dpi=300)

        if show_plots:
            plt.show()
        else:
            plt.close()

        # ------------------------------------------------------------
        # CUMULATIVE MEAN MAX VALUE OVERLAY PLOT
        # ------------------------------------------------------------
        plt.figure()

        for genotype in sorted(set([g for _, g, _, _ in combined_max[tab]])):

            geno_vals = []

            for mouse, geno, sex, vals in combined_max[tab]:
                if geno == genotype:
                    geno_vals.append(vals)

            if len(geno_vals) == 0:
                continue

            max_len = max(len(v) for v in geno_vals)

            padded = []
            for v in geno_vals:
                arr = np.full(max_len, np.nan)
                arr[:len(v)] = v
                padded.append(arr)

            combined = np.vstack(padded)

            cumulative = np.nancumsum(combined, axis=1)

            valid_counts = np.cumsum(~np.isnan(combined), axis=1)

            cummean = np.divide(
                cumulative,
                valid_counts,
                out=np.full_like(cumulative, np.nan, dtype=float),
                where=valid_counts != 0
            )

            mean_vals = np.nanmean(cummean, axis=0)
            sem_vals = np.nanstd(cummean, axis=0) / np.sqrt(cummean.shape[0])

            events = np.arange(1, len(mean_vals)+1)

            plt.plot(events, mean_vals, label=genotype)
            plt.fill_between(events, mean_vals-sem_vals, mean_vals+sem_vals, alpha=0.3)

        plt.xlabel("Event Number")
        plt.ylabel("Cumulative Mean Max Z-score")
        plt.title(f"{tab} CumMean Max Value ({group_column} Overlay)")
        plt.legend()

        plt.savefig(os.path.join(save_folder, f"FED3_FP_{tab}_CumMeanMax_Overlay.png"), dpi=300)

        if show_plots:
            plt.show()
        else:
            plt.close()


        # ------------------------------------------------------------
        # TIME OF MAX VALUE OVERLAY PLOT
        # ------------------------------------------------------------
        plt.figure()

        for genotype in sorted(set([g for _, g, _, _ in combined_max_time[tab]])):

            geno_vals = []

            for mouse, geno, sex, vals in combined_max_time[tab]:
                if geno == genotype:
                    geno_vals.append(vals)

            if len(geno_vals) == 0:
                continue

            # ---- FIX: pad arrays ----
            max_len = max(len(v) for v in geno_vals)

            padded = []
            for v in geno_vals:
                arr = np.full(max_len, np.nan)
                arr[:len(v)] = v
                padded.append(arr)

            combined = np.vstack(padded)
            # -------------------------

            mean_vals = np.nanmean(combined, axis=0)
            sem_vals = np.nanstd(combined, axis=0) / np.sqrt(combined.shape[0])

            events = np.arange(1, len(mean_vals)+1)

            plt.plot(events, mean_vals, label=genotype)

            plt.fill_between(
                events,
                mean_vals-sem_vals,
                mean_vals+sem_vals,
                alpha=0.3
            )

        plt.xlabel("Event Number")
        plt.ylabel("Time of Max (s)")
        plt.title(f"{tab} Time of Max Value ({group_column} Overlay)")
        plt.legend()

        plt.savefig(os.path.join(save_folder, f"FED3_FP_{tab}_TimeOfMax_Overlay.png"), dpi=300)

        if show_plots:
            plt.show()
        else:
            plt.close()


        # ------------------------------------------------------------
        # TIME TO BASELINE OVERLAY PLOT
        # ------------------------------------------------------------
        plt.figure()

        for genotype in sorted(set([g for _, g, _, _ in combined_time_to_baseline[tab]])):

            geno_vals = []

            for mouse, geno, sex, vals in combined_time_to_baseline[tab]:
                if geno == genotype:
                    geno_vals.append(vals)

            if len(geno_vals) == 0:
                continue

            # ---- FIX: pad arrays ----
            max_len = max(len(v) for v in geno_vals)

            padded = []
            for v in geno_vals:
                arr = np.full(max_len, np.nan)
                arr[:len(v)] = v
                padded.append(arr)

            combined = np.vstack(padded)
            # -------------------------

            mean_vals = np.nanmean(combined, axis=0)
            sem_vals = np.nanstd(combined, axis=0) / np.sqrt(combined.shape[0])

            events = np.arange(1, len(mean_vals)+1)

            plt.plot(events, mean_vals, label=genotype)

            plt.fill_between(
                events,
                mean_vals-sem_vals,
                mean_vals+sem_vals,
                alpha=0.3
            )

        plt.xlabel("Event Number")
        plt.ylabel("Time to Baseline (s)")
        plt.title(f"{tab} Time to Baseline ({group_column} Overlay)")
        plt.legend()

        plt.savefig(os.path.join(save_folder, f"FED3_FP_{tab}_TimeToBaseline_Overlay.png"), dpi=300)

        if show_plots:
            plt.show()
        else:
            plt.close()

        # ------------------------------------------------------------
        # CUMULATIVE TIME TO BASELINE OVERLAY PLOT
        # ------------------------------------------------------------
        plt.figure()

        for genotype in sorted(set([g for _, g, _, _ in combined_time_to_baseline[tab]])):

            geno_vals = []

            for mouse, geno, sex, vals in combined_time_to_baseline[tab]:
                if geno == genotype:
                    geno_vals.append(vals)

            if len(geno_vals) == 0:
                continue

            max_len = max(len(v) for v in geno_vals)

            padded = []
            for v in geno_vals:
                arr = np.full(max_len, np.nan)
                arr[:len(v)] = v
                padded.append(arr)

            combined = np.vstack(padded)

            cumulative = np.nancumsum(combined, axis=1)

            mean_vals = np.nanmean(cumulative, axis=0)
            sem_vals = np.nanstd(cumulative, axis=0) / np.sqrt(cumulative.shape[0])

            events = np.arange(1, len(mean_vals)+1)

            plt.plot(events, mean_vals, label=genotype)

            plt.fill_between(events, mean_vals-sem_vals, mean_vals+sem_vals, alpha=0.3)

        plt.xlabel("Event Number")
        plt.ylabel("Cumulative Time To Baseline (s)")
        plt.title(f"{tab} Cumulative Time To Baseline ({group_column} Overlay)")
        plt.legend()

        plt.savefig(os.path.join(save_folder, f"FED3_FP_{tab}_CumBaseline_Overlay.png"), dpi=300)

        if show_plots:
            plt.show()
        else:
            plt.close()

        # ------------------------------------------------------------
        # CUMULATIVE MEAN TIME TO BASELINE OVERLAY PLOT
        # ------------------------------------------------------------
        plt.figure()

        for genotype in sorted(set([g for _, g, _, _ in combined_time_to_baseline[tab]])):

            geno_vals = []

            for mouse, geno, sex, vals in combined_time_to_baseline[tab]:
                if geno == genotype:
                    geno_vals.append(vals)

            if len(geno_vals) == 0:
                continue

            max_len = max(len(v) for v in geno_vals)

            padded = []
            for v in geno_vals:
                arr = np.full(max_len, np.nan)
                arr[:len(v)] = v
                padded.append(arr)

            combined = np.vstack(padded)

            cumulative = np.nancumsum(combined, axis=1)

            valid_counts = np.cumsum(~np.isnan(combined), axis=1)

            cummean = np.divide(
                cumulative,
                valid_counts,
                out=np.full_like(cumulative, np.nan, dtype=float),
                where=valid_counts != 0
            )

            mean_vals = np.nanmean(cummean, axis=0)
            sem_vals = np.nanstd(cummean, axis=0) / np.sqrt(cummean.shape[0])

            events = np.arange(1, len(mean_vals)+1)

            plt.plot(events, mean_vals, label=genotype)
            plt.fill_between(events, mean_vals-sem_vals, mean_vals+sem_vals, alpha=0.3)

        plt.xlabel("Event Number")
        plt.ylabel("Cumulative Mean Time To Baseline (s)")
        plt.title(f"{tab} CumMean Time To Baseline ({group_column} Overlay)")
        plt.legend()

        plt.savefig(os.path.join(save_folder, f"FED3_FP_{tab}_CumMeanBaseline_Overlay.png"), dpi=300)

        if show_plots:
            plt.show()
        else:
            plt.close()

        # ------------------------------------------------------------
        # AUC OVERLAY PLOT
        # ------------------------------------------------------------
        plt.figure()

        for genotype in sorted(set([g for _, g, _, _ in combined_auc[tab]])):

            geno_vals = []

            for mouse, geno, sex, vals in combined_auc[tab]:
                if geno == genotype:
                    geno_vals.append(vals)

            if len(geno_vals) == 0:
                continue

            max_len = max(len(v) for v in geno_vals)

            padded = []
            for v in geno_vals:
                arr = np.full(max_len, np.nan)
                arr[:len(v)] = v
                padded.append(arr)

            combined = np.vstack(padded)

            mean_vals = np.nanmean(combined, axis=0)
            sem_vals = np.nanstd(combined, axis=0) / np.sqrt(combined.shape[0])

            events = np.arange(1, len(mean_vals)+1)

            plt.plot(events, mean_vals, label=genotype)

            plt.fill_between(events, mean_vals-sem_vals, mean_vals+sem_vals, alpha=0.3)

        plt.xlabel("Event Number")
        plt.ylabel(f"AUC ({time_window['start']}–{time_window['end']} s)")
        plt.title(f"{tab} AUC ({time_window['start']}–{time_window['end']} s)")
        plt.legend()

        plt.savefig(
            os.path.join(
                save_folder,
                f"FED3_FP_{tab}_AUC_{time_window['start']}_{time_window['end']}s_Overlay.png"
            ),
            dpi=300
        )

        if show_plots:
            plt.show()
        else:
            plt.close()

        # ------------------------------------------------------------
        # CUMULATIVE AUC OVERLAY PLOT
        # ------------------------------------------------------------
        plt.figure()

        for genotype in sorted(set([g for _, g, _, _ in combined_auc[tab]])):

            geno_vals = []

            for mouse, geno, sex, vals in combined_auc[tab]:
                if geno == genotype:
                    geno_vals.append(vals)

            if len(geno_vals) == 0:
                continue

            max_len = max(len(v) for v in geno_vals)

            padded = []
            for v in geno_vals:
                arr = np.full(max_len, np.nan)
                arr[:len(v)] = v
                padded.append(arr)

            combined = np.vstack(padded)

            cumulative = np.nancumsum(combined, axis=1)

            mean_vals = np.nanmean(cumulative, axis=0)
            sem_vals = np.nanstd(cumulative, axis=0) / np.sqrt(cumulative.shape[0])

            events = np.arange(1, len(mean_vals)+1)

            plt.plot(events, mean_vals, label=genotype)

            plt.fill_between(events, mean_vals-sem_vals, mean_vals+sem_vals, alpha=0.3)

        plt.xlabel("Event Number")
        plt.ylabel(f"Cumulative AUC ({time_window['start']}–{time_window['end']} s)")
        plt.title(f"{tab} Cumulative AUC ({time_window['start']}–{time_window['end']} s)")
        plt.legend()

        plt.savefig(
            os.path.join(
                save_folder,
                f"FED3_FP_{tab}_CumAUC_{time_window['start']}_{time_window['end']}s_Overlay.png"
            ),
            dpi=300
        )

        if show_plots:
            plt.show()
        else:
            plt.close()
        
        # ------------------------------------------------------------
        # CUMULATIVE MEAN AUC OVERLAY PLOT
        # ------------------------------------------------------------
        plt.figure()

        for genotype in sorted(set([g for _, g, _, _ in combined_auc[tab]])):

            geno_vals = []

            for mouse, geno, sex, vals in combined_auc[tab]:
                if geno == genotype:
                    geno_vals.append(vals)

            if len(geno_vals) == 0:
                continue

            max_len = max(len(v) for v in geno_vals)

            padded = []
            for v in geno_vals:
                arr = np.full(max_len, np.nan)
                arr[:len(v)] = v
                padded.append(arr)

            combined = np.vstack(padded)

            cumulative = np.nancumsum(combined, axis=1)

            valid_counts = np.cumsum(~np.isnan(combined), axis=1)

            cummean = np.divide(
                cumulative,
                valid_counts,
                out=np.full_like(cumulative, np.nan, dtype=float),
                where=valid_counts != 0
            )

            mean_vals = np.nanmean(cummean, axis=0)
            sem_vals = np.nanstd(cummean, axis=0) / np.sqrt(cummean.shape[0])

            events = np.arange(1, len(mean_vals)+1)

            plt.plot(events, mean_vals, label=genotype)
            plt.fill_between(events, mean_vals-sem_vals, mean_vals+sem_vals, alpha=0.3)

        plt.xlabel("Event Number")
        plt.ylabel(f"Cumulative Mean AUC ({time_window['start']}–{time_window['end']} s)")
        plt.title(f"{tab} CumMean AUC ({time_window['start']}–{time_window['end']} s)")
        plt.legend()

        plt.savefig(
            os.path.join(
                save_folder,
                f"FED3_FP_{tab}_CumMeanAUC_{time_window['start']}_{time_window['end']}s_Overlay.png"
            ),
            dpi=300
        )

        if show_plots:
            plt.show()
        else:
            plt.close()

        # ------------------------------------------------------------
        # MEAN Z WINDOW OVERLAY PLOT
        # ------------------------------------------------------------
        plt.figure()

        for genotype in sorted(set([g for _, g, _, _ in combined_meanz[tab]])):

            geno_vals = []

            for mouse, geno, sex, vals in combined_meanz[tab]:
                if geno == genotype:
                    geno_vals.append(vals)

            if len(geno_vals) == 0:
                continue

            max_len = max(len(v) for v in geno_vals)

            padded = []
            for v in geno_vals:
                arr = np.full(max_len, np.nan)
                arr[:len(v)] = v
                padded.append(arr)

            combined = np.vstack(padded)

            mean_vals = np.nanmean(combined, axis=0)
            sem_vals = np.nanstd(combined, axis=0) / np.sqrt(combined.shape[0])

            events = np.arange(1, len(mean_vals)+1)

            plt.plot(events, mean_vals, label=genotype)

            plt.fill_between(events, mean_vals-sem_vals, mean_vals+sem_vals, alpha=0.3)

        plt.xlabel("Event Number")
        plt.ylabel(f"Mean Z ({time_window['start']}–{time_window['end']} s)")
        plt.title(f"{tab} Mean Z Window ({time_window['start']}–{time_window['end']} s)")
        plt.legend()

        plt.savefig(
            os.path.join(
                save_folder,
                f"FED3_FP_{tab}_MeanZ_{time_window['start']}_{time_window['end']}s_Overlay.png"
            ),
            dpi=300
        )

        if show_plots:
            plt.show()
        else:
            plt.close()

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
            # MAX VALUES
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
            final.to_excel(writer, sheet_name=f"{tab} MaxValues", float_format="%.10f")


            # ------------------------------------------------------------
            # MAX VALUE TIME
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
            final.to_excel(writer, sheet_name=f"{tab} MaxValTime", float_format="%.10f")


            # ------------------------------------------------------------
            # CUM MAX VALUES
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
            final.to_excel(writer, sheet_name=f"{tab} CumMaxVal", float_format="%.10f")


            # ------------------------------------------------------------
            # CUMMEAN MAX VALUES
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
            final.to_excel(writer, sheet_name=f"{tab} CumMeanMaxVal", float_format="%.10f")


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
