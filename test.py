import os
import time
import threading
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
import pydicom
import numpy as np
import concurrent.futures
import csv
from collections import Counter, defaultdict

class DicomScannerApp:
    def __init__(self, root):
        self.root = root
        self.root.title("DICOM Bulk Metadata Scanner")
        self.root.geometry("600x250")
        self.root.resizable(False, False)

        self.input_folder = tk.StringVar()
        self.is_scanning = False
        self.stop_requested = False
        self.total_files = 0
        self.processed_files = 0
        self.start_time = None

        # GUI Elements
        ttk.Label(root, text="Select DICOM Input Folder:").pack(pady=5)
        frame_input = ttk.Frame(root)
        frame_input.pack(fill='x', padx=10)
        self.input_entry = ttk.Entry(frame_input, textvariable=self.input_folder, width=50)
        self.input_entry.pack(side='left', fill='x', expand=True)
        ttk.Button(frame_input, text="Browse", command=self.browse_input).pack(side='left', padx=5)

        self.progress = ttk.Progressbar(root, orient='horizontal', length=580, mode='determinate')
        self.progress.pack(pady=15)

        self.status_label = ttk.Label(root, text="Idle", anchor='center')
        self.status_label.pack()

        btn_frame = ttk.Frame(root)
        btn_frame.pack(pady=10)
        self.start_button = ttk.Button(btn_frame, text="Start Scan", command=self.start_scan)
        self.start_button.pack(side='left', padx=10)
        self.stop_button = ttk.Button(btn_frame, text="Stop Scan", command=self.stop_scan, state='disabled')
        self.stop_button.pack(side='left', padx=10)

        self.summary_report_path = None

    def browse_input(self):
        folder = filedialog.askdirectory()
        if folder:
            self.input_folder.set(folder)

    def stop_scan(self):
        if self.is_scanning:
            self.stop_requested = True
            self.status_label.config(text="Stopping after current files...")

    def start_scan(self):
        if self.is_scanning:
            messagebox.showwarning("Warning", "Scan is already running!")
            return

        folder = self.input_folder.get()
        if not folder or not os.path.isdir(folder):
            messagebox.showerror("Error", "Please select a valid input folder.")
            return

        # Reset
        self.is_scanning = True
        self.stop_requested = False
        self.processed_files = 0
        self.start_time = time.time()
        self.progress['value'] = 0
        self.status_label.config(text="Scanning...")

        # List all DICOM files
        self.dicom_files = self.get_dicom_files(folder)
        self.total_files = len(self.dicom_files)
        if self.total_files == 0:
            messagebox.showinfo("Info", "No DICOM files found in the folder.")
            self.is_scanning = False
            self.status_label.config(text="Idle")
            return

        self.progress['maximum'] = self.total_files
        self.start_button.config(state='disabled')
        self.stop_button.config(state='normal')

        # Start worker thread to avoid blocking GUI
        threading.Thread(target=self.scan_worker, daemon=True).start()

    def get_dicom_files(self, folder):
        dicom_exts = {'.dcm', ''}  # Some DICOM files may have no extension
        files = []
        for root, _, filenames in os.walk(folder):
            for fname in filenames:
                ext = os.path.splitext(fname)[1].lower()
                if ext in dicom_exts or ext == '':
                    files.append(os.path.join(root, fname))
        return files

    def scan_worker(self):
        # Use concurrent.futures ThreadPoolExecutor or ProcessPoolExecutor
        max_workers = min(16, os.cpu_count() or 4)  # Use up to 16 threads or all CPUs

        # Data structures for aggregation
        bits_allocated_counter = Counter()
        bits_stored_counter = Counter()
        photometric_counter = Counter()
        pixel_representation_counter = Counter()
        modality_counter = Counter()
        rescale_slope_values = []
        rescale_intercept_values = []
        rows_list = []
        cols_list = []
        pixel_min_values = []
        pixel_max_values = []
        pixel_data_types = Counter()
        color_space_counter = Counter()
        errors = 0

        def process_file(filepath):
            try:
                ds = pydicom.dcmread(filepath, stop_before_pixels=False)
                # Extract info
                bits_allocated = getattr(ds, 'BitsAllocated', None)
                bits_stored = getattr(ds, 'BitsStored', None)
                photometric = getattr(ds, 'PhotometricInterpretation', 'N/A')
                pixel_representation = getattr(ds, 'PixelRepresentation', None)
                modality = getattr(ds, 'Modality', 'N/A')
                rescale_slope = getattr(ds, 'RescaleSlope', None)
                rescale_intercept = getattr(ds, 'RescaleIntercept', None)
                rows = getattr(ds, 'Rows', None)
                cols = getattr(ds, 'Columns', None)

                # Pixel data type
                # Try numpy dtype from pixel_array
                try:
                    arr = ds.pixel_array
                    pixel_data_types[str(arr.dtype)] += 1
                    pixel_min_values.append(np.min(arr))
                    pixel_max_values.append(np.max(arr))
                except Exception:
                    # Fallback: look at BitsAllocated
                    pixel_data_types[str(bits_allocated) + '-bits'] += 1

                # Append lists/counters
                if bits_allocated is not None:
                    bits_allocated_counter[bits_allocated] += 1
                if bits_stored is not None:
                    bits_stored_counter[bits_stored] += 1
                if photometric:
                    photometric_counter[photometric] += 1
                if pixel_representation is not None:
                    pixel_representation_counter[pixel_representation] += 1
                if modality:
                    modality_counter[modality] += 1
                if rescale_slope is not None:
                    rescale_slope_values.append(rescale_slope)
                if rescale_intercept is not None:
                    rescale_intercept_values.append(rescale_intercept)
                if rows:
                    rows_list.append(rows)
                if cols:
                    cols_list.append(cols)

                return True
            except Exception as e:
                # Could not read file properly
                return False

        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = []
            for file in self.dicom_files:
                if self.stop_requested:
                    break
                futures.append(executor.submit(process_file, file))

            for future in concurrent.futures.as_completed(futures):
                if self.stop_requested:
                    break
                success = future.result()
                self.processed_files += 1
                self.update_progress()
                if not success:
                    errors += 1

        self.is_scanning = False
        self.start_button.config(state='normal')
        self.stop_button.config(state='disabled')

        # Prepare summary report
        self.summary_report_path = os.path.join(self.input_folder.get(), 'dicom_scan_summary_report.txt')
        with open(self.summary_report_path, 'w') as f:
            f.write(f"DICOM Bulk Scan Summary Report\n")
            f.write(f"Scanned Folder: {self.input_folder.get()}\n")
            f.write(f"Total Files Found: {self.total_files}\n")
            f.write(f"Successfully Scanned: {self.processed_files}\n")
            f.write(f"Failed to Read: {errors}\n\n")

            def format_counter(counter):
                total = sum(counter.values())
                lines = []
                for key, val in sorted(counter.items(), key=lambda x: x[1], reverse=True):
                    pct = (val / total) * 100 if total else 0
                    lines.append(f"  {key}: {val} files ({pct:.2f}%)")
                return "\n".join(lines)

            f.write("Bits Allocated:\n" + format_counter(bits_allocated_counter) + "\n\n")
            f.write("Bits Stored:\n" + format_counter(bits_stored_counter) + "\n\n")
            f.write("Photometric Interpretations:\n" + format_counter(photometric_counter) + "\n\n")
            f.write("Pixel Representation (0=unsigned, 1=signed):\n" + format_counter(pixel_representation_counter) + "\n\n")
            f.write("Modalities:\n" + format_counter(modality_counter) + "\n\n")
            f.write("Pixel Data Types (numpy dtype or bits):\n" + format_counter(pixel_data_types) + "\n\n")

            if rescale_slope_values:
                f.write(f"RescaleSlope: min={min(rescale_slope_values)}, max={max(rescale_slope_values)}\n")
            else:
                f.write("RescaleSlope: None found\n")
            if rescale_intercept_values:
                f.write(f"RescaleIntercept: min={min(rescale_intercept_values)}, max={max(rescale_intercept_values)}\n")
            else:
                f.write("RescaleIntercept: None found\n")

            if rows_list:
                f.write(f"Rows (image height): min={min(rows_list)}, max={max(rows_list)}, average={sum(rows_list)/len(rows_list):.2f}\n")
            else:
                f.write("Rows: None found\n")
            if cols_list:
                f.write(f"Columns (image width): min={min(cols_list)}, max={max(cols_list)}, average={sum(cols_list)/len(cols_list):.2f}\n")
            else:
                f.write("Columns: None found\n")

            if pixel_min_values and pixel_max_values:
                f.write(f"Pixel Value Min (across all images): {min(pixel_min_values)}\n")
                f.write(f"Pixel Value Max (across all images): {max(pixel_max_values)}\n")

        # Update GUI status
        self.status_label.config(text=f"Scan completed. Report saved:\n{self.summary_report_path}")
        messagebox.showinfo("Scan Complete", f"Summary report saved:\n{self.summary_report_path}")

    def update_progress(self):
        self.progress['value'] = self.processed_files
        elapsed = time.time() - self.start_time
        if self.processed_files > 0 and elapsed > 0:
            rate = self.processed_files / elapsed
            remaining = self.total_files - self.processed_files
            eta = remaining / rate if rate > 0 else 0
            self.status_label.config(text=f"Processed {self.processed_files}/{self.total_files} files | ETA: {int(eta)} sec")

def main():
    root = tk.Tk()
    app = DicomScannerApp(root)
    root.mainloop()

if __name__ == "__main__":
    main()
