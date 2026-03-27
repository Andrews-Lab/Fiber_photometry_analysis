import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tkinter as tk
from tkinter import filedialog, messagebox
import os
import sys

def Peri_events_post_processing():

    # ------------------------------------------------------------
    # SELECT DATA FILES
    # ------------------------------------------------------------
    root = tk.Tk()
    root.withdraw()

    file_paths = filedialog.askopenfilenames(
        title="Select Peri-Event Photometry files",
        filetypes=[("Photometry files", "*.xlsx *.csv")]
    )

    if not file_paths:
        root.destroy()
        return

    file_map = {os.path.basename(f): f for f in file_paths}
    save_folder = os.path.dirname(file_paths[0])

    # ------------------------------------------------------------
    # METADATA
    # ------------------------------------------------------------
    use_existing = messagebox.askyesno(
        "Metadata",
        "Do you have an existing metadata file?"
    )

    metadata_df = None

    if use_existing:

        metadata_file = filedialog.askopenfilename(
            title="Select Metadata File",
            filetypes=[("Excel files", "*.xlsx")]
        )

        metadata_df = pd.read_excel(metadata_file)

    else:

        meta_window = tk.Toplevel(root)
        meta_window.title("Enter Metadata")

        headers = ["Filename","Mouse ID","Sex","Genotype"]

        for col,header in enumerate(headers):
            tk.Label(meta_window,text=header,font=("Arial",10,"bold")).grid(row=0,column=col)

        rows=[]

        for i,filename in enumerate(file_map.keys()):

            tk.Label(meta_window,text=filename).grid(row=i+1,column=0)

            mouse_entry=tk.Entry(meta_window)
            mouse_entry.grid(row=i+1,column=1)

            sex_entry=tk.Entry(meta_window)
            sex_entry.grid(row=i+1,column=2)

            genotype_entry=tk.Entry(meta_window)
            genotype_entry.grid(row=i+1,column=3)

            rows.append({
                "filename":filename,
                "mouse":mouse_entry,
                "sex":sex_entry,
                "genotype":genotype_entry
            })

        def collect_metadata():

            metadata_list=[]

            for row in rows:
                metadata_list.append({
                    "Filename":row["filename"],
                    "Mouse ID":row["mouse"].get(),
                    "Sex":row["sex"].get(),
                    "Genotype":row["genotype"].get()
                })

            global metadata_df
            metadata_df=pd.DataFrame(metadata_list)

            metadata_df.to_excel(
                os.path.join(save_folder,"PeriEvent_Metadata.xlsx"),
                index=False
            )

            meta_window.destroy()

        tk.Button(meta_window,text="Continue",command=collect_metadata)\
            .grid(row=len(file_map)+2,column=0,columnspan=4)

        root.wait_window(meta_window)

    # ------------------------------------------------------------
    # SORT METADATA
    # ------------------------------------------------------------
    metadata_df["Mouse ID"]=metadata_df["Mouse ID"].astype(str)
    metadata_df["Mouse ID numeric"]=pd.to_numeric(metadata_df["Mouse ID"],errors="coerce")

    metadata_df=metadata_df.sort_values(
        by=["Genotype","Sex","Mouse ID numeric","Mouse ID"]
    ).drop(columns=["Mouse ID numeric"]).reset_index(drop=True)

    # ------------------------------------------------------------
    # PLOT OPTION
    # ------------------------------------------------------------
    show_plots = messagebox.askyesno(
        "Plot Display",
        "Display plots?\n\nYes = show\nNo = save only"
    )

    # ------------------------------------------------------------
    # STORAGE
    # ------------------------------------------------------------
    combined_raw=[]
    combined_means=[]
    combined_max=[]
    combined_max_time=[]
    combined_time_to_baseline=[]

    reference_time=None
    genotype_traces={}

    # ------------------------------------------------------------
    # EXTRACTION
    # ------------------------------------------------------------
    for _,row in metadata_df.iterrows():

        file=file_map[row["Filename"]]
        genotype=row["Genotype"]
        mouse=str(row["Mouse ID"])
        sex=row["Sex"]

        if file.lower().endswith(".csv"):
            df=pd.read_csv(file,header=None,sep=None,engine="python")
        else:
            df=pd.read_excel(file,header=None)

        custom_idx=df[df.eq("Custom name").any(axis=1)].index[0]
        event_note_idx=df[df.eq("Event note").any(axis=1)].index[0]
        max_idx=df[df.eq("Max values").any(axis=1)].index[0]

        max_time_idx=df[df.astype(str).apply(
            lambda r:r.str.contains("Time of max",case=False).any(),axis=1
        )].index[0]

        baseline_idx=df[df.astype(str).apply(
            lambda r:r.str.contains("Time to baseline",case=False).any(),axis=1
        )].index[0]

        data_start=custom_idx+1

        time_series=pd.to_numeric(df.iloc[data_start:,1],errors="coerce")

        # ------------------------------------------------------------
        # EVENT COLUMN DETECTION
        # ------------------------------------------------------------
        event_notes=df.iloc[event_note_idx,:].astype(str)

        event_columns=event_notes[
            (event_notes.str.strip()!="") &
            (event_notes.str.lower()!="nan")
        ].index

        event_columns=[c for c in event_columns if c>1]

        if len(event_columns)==0:
            continue
        # ------------------------------------------------------------

        trial_df=df.iloc[data_start:,event_columns]
        trial_df=trial_df.apply(pd.to_numeric,errors="coerce")

        max_vals=pd.to_numeric(df.iloc[max_idx,event_columns],errors="coerce").values
        max_time_vals=pd.to_numeric(df.iloc[max_time_idx,event_columns],errors="coerce").values
        baseline_vals=pd.to_numeric(df.iloc[baseline_idx,event_columns],errors="coerce").values

        time_series=time_series.reset_index(drop=True)
        trial_df=trial_df.reset_index(drop=True)

        valid_rows=~time_series.isna()

        time_vector=time_series.loc[valid_rows].values
        trial_matrix=trial_df.loc[valid_rows].values

        if trial_matrix.size==0:
            continue

        if reference_time is None:
            reference_time=time_vector
        else:
            if not np.allclose(reference_time,time_vector,atol=1e-10):
                messagebox.showerror(
                    "Timestamp Error",
                    f"Time vectors differ in file:\n{row['Filename']}"
                )
                root.destroy()
                return

        combined_raw.append((mouse,genotype,sex,trial_matrix))
        combined_means.append((mouse,genotype,sex,np.nanmean(trial_matrix,axis=1)))
        combined_max.append((mouse,genotype,sex,max_vals))
        combined_max_time.append((mouse,genotype,sex,max_time_vals))
        combined_time_to_baseline.append((mouse,genotype,sex,baseline_vals))

        genotype_traces.setdefault(genotype,[]).append(trial_matrix)

    # ------------------------------------------------------------
    # PLOTS
    # ------------------------------------------------------------
    n_mice = len(combined_raw)

    fig, axes = plt.subplots(n_mice, 1, figsize=(8, 2*n_mice), sharex=True)

    if n_mice == 1:
        axes = [axes]

    for ax, (mouse, geno, sex, trials) in zip(axes, combined_raw):

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
        ax.set_xlim(-20, 60)
        ax.set_ylabel(mouse)

    axes[-1].set_xlabel("Time (s)")
    fig.suptitle("Peri-Event — Per Mouse Mean ± SEM")

    plt.tight_layout()
    plt.savefig(os.path.join(save_folder, "PeriEvent_PerMouse.png"), dpi=300)

    if show_plots:
        plt.show()
    else:
        plt.close()


    # ------------------------------------------------------------
    # GENOTYPE OVERLAY PLOT
    # ------------------------------------------------------------
    plt.figure()

    for genotype, matrices in genotype_traces.items():

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
    plt.xlim(-20, 60)
    plt.xlabel("Time (s)")
    plt.ylabel("Z-score")
    plt.title("Peri-Event Response (Genotype Overlay)")
    plt.legend()

    plot_path = os.path.join(save_folder, "PeriEvent_Overlay.png")
    plt.savefig(plot_path, dpi=300)

    if show_plots:
        plt.show()
    else:
        plt.close()

    # ------------------------------------------------------------
    # MAX VALUE OVERLAY PLOT
    # ------------------------------------------------------------
    plt.figure()

    for genotype in set([g for _, g, _, _ in combined_max]):

        geno_vals = []

        for mouse, geno, sex, vals in combined_max:
            if geno == genotype:
                geno_vals.append(vals)

        if len(geno_vals) == 0:
            continue

        # pad arrays to equal length
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

        plt.fill_between(
            events,
            mean_vals - sem_vals,
            mean_vals + sem_vals,
            alpha=0.3
        )

    plt.xlabel("Event Number")
    plt.ylabel("Max Z-score")
    plt.title("Peri-Event Max Value (Genotype Overlay)")
    plt.legend()

    plt.savefig(os.path.join(save_folder, "PeriEvent_MaxValue_Overlay.png"), dpi=300)

    if show_plots:
        plt.show()
    else:
        plt.close()

    # ------------------------------------------------------------
    # TIME OF MAX VALUE OVERLAY PLOT
    # ------------------------------------------------------------
    plt.figure()

    for genotype in set([g for _, g, _, _ in combined_max_time]):

        geno_vals = []

        for mouse, geno, sex, vals in combined_max_time:
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

        plt.fill_between(
            events,
            mean_vals - sem_vals,
            mean_vals + sem_vals,
            alpha=0.3
        )

    plt.xlabel("Event Number")
    plt.ylabel("Time of Max (s)")
    plt.title("Peri-Event Time of Max (Genotype Overlay)")
    plt.legend()

    plt.savefig(os.path.join(save_folder, "PeriEvent_TimeOfMax_Overlay.png"), dpi=300)

    if show_plots:
        plt.show()
    else:
        plt.close()

    # ------------------------------------------------------------
    # TIME TO BASELINE OVERLAY PLOT
    # ------------------------------------------------------------
    plt.figure()

    for genotype in set([g for _, g, _, _ in combined_time_to_baseline]):

        geno_vals = []

        for mouse, geno, sex, vals in combined_time_to_baseline:
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

        plt.fill_between(
            events,
            mean_vals - sem_vals,
            mean_vals + sem_vals,
            alpha=0.3
        )

    plt.xlabel("Event Number")
    plt.ylabel("Time to Baseline (s)")
    plt.title("Peri-Event Time to Baseline (Genotype Overlay)")
    plt.legend()

    plt.savefig(os.path.join(save_folder, "PeriEvent_TimeToBaseline_Overlay.png"), dpi=300)

    if show_plots:
        plt.show()
    else:
        plt.close()

    # ------------------------------------------------------------
    # EXPORT COMBINED EXCEL
    # ------------------------------------------------------------
    output_path = os.path.join(save_folder, "PeriEvent_Combined.xlsx")

    with pd.ExcelWriter(output_path) as writer:

        # EVENTS
        data = {"Time (s)": reference_time}

        mouse_row = [""]
        geno_row = [""]
        sex_row = [""]

        for mouse, geno, sex, trials in combined_raw:

            for i in range(trials.shape[1]):

                col = f"{mouse}_event{i+1}"
                data[col] = trials[:, i]

                mouse_row.append(mouse)
                geno_row.append(geno)
                sex_row.append(sex)

        df = pd.DataFrame(data)

        meta = pd.DataFrame(
            [mouse_row, geno_row, sex_row],
            index=["Mouse ID", "Genotype", "Sex"],
            columns=df.columns
        )

        final = pd.concat([meta, df])
        final.to_excel(writer, sheet_name="Events", float_format="%.10f")

        # ------------------------------------------------------------
        # EVENT MEANS
        # ------------------------------------------------------------
        data = {"Time (s)": reference_time}

        mouse_row = [""]
        geno_row = [""]
        sex_row = [""]

        for mouse, geno, sex, trace in combined_means:

            data[mouse] = trace

            mouse_row.append(mouse)
            geno_row.append(geno)
            sex_row.append(sex)

        df = pd.DataFrame(data)

        meta = pd.DataFrame(
            [mouse_row, geno_row, sex_row],
            index=["Mouse ID", "Genotype", "Sex"],
            columns=df.columns
        )

        final = pd.concat([meta, df])
        final.to_excel(writer, sheet_name="Event Means", float_format="%.10f")

        # ------------------------------------------------------------
        # MAX VALUES
        # ------------------------------------------------------------
        max_lists = []
        max_len = 0

        for mouse, geno, sex, vals in combined_max:
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
            index=["Mouse ID", "Genotype", "Sex"],
            columns=df.columns
        )

        final = pd.concat([meta, df])
        final.to_excel(writer, sheet_name="Max Values", float_format="%.10f")

        # ------------------------------------------------------------
        # MAX VALUE TIME
        # ------------------------------------------------------------
        max_lists = []
        max_len = 0

        for mouse, geno, sex, vals in combined_max_time:
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
            index=["Mouse ID", "Genotype", "Sex"],
            columns=df.columns
        )

        final = pd.concat([meta, df])
        final.to_excel(writer, sheet_name="Max Value Time", float_format="%.10f")

        # ------------------------------------------------------------
        # TIME TO BASELINE
        # ------------------------------------------------------------
        max_lists = []
        max_len = 0

        for mouse, geno, sex, vals in combined_time_to_baseline:
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
            index=["Mouse ID", "Genotype", "Sex"],
            columns=df.columns
        )

        final = pd.concat([meta, df])
        final.to_excel(writer, sheet_name="Time To Baseline", float_format="%.10f")

    print("\nCombined Excel saved:",output_path)
    print("\nAnalysis complete.\n")

    root.destroy()
    return

if __name__ == "__main__":
    peri_events_post_processing()
