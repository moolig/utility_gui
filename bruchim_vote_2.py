import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

# ---------------------- analysis code (unchanged logic) ----------------------

def func(pct, allvalues):
    absolute = pct / 100.0 * np.sum(allvalues)
    return "{:.1f}%\n({:.1f})".format(pct, absolute)

def create_graph(df, not_relevant_columns: list, save_dir: str):
    for column in df.columns:
        if column not in not_relevant_columns:
            value_counts = df[column].value_counts()
            y = np.array(value_counts.values)
            # reverse labels for Hebrew presentation
            mylabels = [idx[::-1] for idx in value_counts.index]

            plt.pie(y, labels=mylabels, autopct=lambda pct: func(pct, y))
            plt.savefig(os.path.join(save_dir, column + '.png'))
            plt.show()
            plt.close()

def create_column_chart(df, not_relevant_columns: list, save_dir: str):
    for column in df.columns:
        if column not in not_relevant_columns:
            names_list = df[column].astype(str).str.split(', ')
            all_names = [name.strip() for sublist in names_list for name in sublist]
            all_names = [idx[::-1] for idx in all_names]  # reverse for Hebrew

            name_counts = pd.Series(all_names).value_counts()
            plt.figure(figsize=(10, 8))
            name_counts.plot(kind='bar')  # no explicit color to keep defaults
            plt.title('Count of Names Selected')
            plt.xlabel('Names')
            plt.ylabel('Count')
            plt.xticks(rotation=90)
            plt.tight_layout()
            plt.savefig(os.path.join(save_dir, column + '.png'))
            plt.show()
            plt.close()

def remove_incoret_pin(df, point_filed, good_val):
    return df[df[point_filed].astype(str).str.contains(good_val) == True]

def get_duplicate_pin(df, pin_filed):
    pin_series = df[pin_filed].value_counts()
    return [pin for pin in pin_series.index if pin_series[pin] > 1]

def get_incorrect_pins(df, pin_filed, point_filed, good_val):
    correct = df[df[point_filed].astype(str).str.contains(good_val) == True]
    return list(set(df[pin_filed]) - set(correct[pin_filed]))

def fusion_same_pin(df, pin_filed):
    aggregation_functions = {col: ('last' if col != pin_filed else 'first') for col in df.columns}
    f_new = df.groupby(df[pin_filed]).aggregate(aggregation_functions)
    # Keep pin as a column (optional—comment out if you prefer index)
    # f_new = f_new.reset_index()
    return f_new

def analysis(csv_input: str, res_dir: str, graph_type='pie'):
    pin_filed = 'סיסמה'
    point_filed = 'ניקוד'  # 'ניקוד'
    good_val = '100 / 100'
    not_relevant_column = [pin_filed, point_filed, 'Timestamp']  # 'חותמת זמן'

    df = pd.read_csv(csv_input, sep=",", encoding='utf-8')
    try:
        incorrect = get_incorrect_pins(df, pin_filed, point_filed, good_val)
        dupes = get_duplicate_pin(df, pin_filed)

        print('incurrect pins:', incorrect)
        print('duplicate pin:', dupes)

        os.makedirs(res_dir, exist_ok=True)
        with open(os.path.join(res_dir, 'incorrect_pins.txt'), 'w', encoding='utf-8') as the_file:
            the_file.write(f'incorrect pins: {incorrect}\n')
            the_file.write(f'duplicate pin: {dupes}\n')

        df = remove_incoret_pin(df, point_filed, good_val)
        df = fusion_same_pin(df, pin_filed)
    except Exception as e:
        print(e)

    if graph_type == 'pie':
        create_graph(df, not_relevant_column, res_dir)
    elif graph_type == 'bar':
        create_column_chart(df, not_relevant_column, res_dir)

    df.to_csv(os.path.join(res_dir, 'csv_output.csv'), index=False)

# ---------------------- Tkinter GUI ----------------------

class VoteAnalyzerApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Vote Analysis (Tkinter)")
        self.geometry("520x240")

        # Variables
        self.input_file_var = tk.StringVar()
        self.output_dir_var = tk.StringVar()
        self.graph_type_var = tk.StringVar(value="pie")

        # Layout
        pad = {'padx': 8, 'pady': 6}

        # Input file row
        frm1 = ttk.Frame(self)
        frm1.pack(fill='x', **pad)
        ttk.Label(frm1, text="Input CSV:").pack(side='left')
        ent1 = ttk.Entry(frm1, textvariable=self.input_file_var)
        ent1.pack(side='left', fill='x', expand=True, padx=6)
        ttk.Button(frm1, text="Browse…", command=self.browse_csv).pack(side='left')

        # Output folder row
        frm2 = ttk.Frame(self)
        frm2.pack(fill='x', **pad)
        ttk.Label(frm2, text="Output Folder:").pack(side='left')
        ent2 = ttk.Entry(frm2, textvariable=self.output_dir_var)
        ent2.pack(side='left', fill='x', expand=True, padx=6)
        ttk.Button(frm2, text="Browse…", command=self.browse_folder).pack(side='left')

        # Graph type row
        frm3 = ttk.Frame(self)
        frm3.pack(fill='x', **pad)
        ttk.Label(frm3, text="Graph Type:").pack(side='left')
        cb = ttk.Combobox(frm3, textvariable=self.graph_type_var, values=["pie", "bar"], state="readonly", width=8)
        cb.pack(side='left', padx=6)

        # Run button
        frm4 = ttk.Frame(self)
        frm4.pack(fill='x', **pad)
        ttk.Button(frm4, text="RUN", command=self.run_analysis).pack(side='left')

        # Status label
        self.status_var = tk.StringVar(value="Ready.")
        ttk.Label(self, textvariable=self.status_var, foreground="gray").pack(anchor='w', padx=10, pady=4)

    def browse_csv(self):
        path = filedialog.askopenfilename(
            title="Select CSV file",
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")]
        )
        if path:
            self.input_file_var.set(path)

    def browse_folder(self):
        path = filedialog.askdirectory(title="Select output folder")
        if path:
            self.output_dir_var.set(path)

    def run_analysis(self):
        csv_path = self.input_file_var.get().strip()
        out_dir = self.output_dir_var.get().strip()
        graph_type = self.graph_type_var.get().strip()

        if not csv_path:
            messagebox.showerror("Missing input", "Please select an input CSV file.")
            return
        if not out_dir:
            messagebox.showerror("Missing output folder", "Please select an output folder.")
            return

        try:
            self.status_var.set("Running…")
            self.update_idletasks()
            analysis(csv_path, out_dir, graph_type=graph_type)
            self.status_var.set("Done. Results saved.")
            messagebox.showinfo("Success", f"Finished!\nSaved outputs to:\n{out_dir}")
        except Exception as e:
            self.status_var.set("Error.")
            messagebox.showerror("Error", str(e))

if __name__ == "__main__":
    app = VoteAnalyzerApp()
    app.mainloop()
