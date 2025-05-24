import tkinter as tk
from tkinter import filedialog, messagebox, ttk
import os
import json
import sys
import threading
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import pydicom
from PIL import Image
import numpy as np
import pandas as pd
import gc
import time
import uuid

# Global variables
CONFIG_FILE = "C:/Users/anuse/Desktop/moleaiproject/gui_config.json"
PROGRESS_FILE = "C:/Users/anuse/Desktop/moleaiproject/conversion_progress.json"
ERROR_LOG = "C:/Users/anuse/Desktop/moleaiproject/conversion_errors.log"
INPUT_FOLDER = "C:/Users/anuse/Desktop/moleaiproject/dicom"

try:
    import cupy as cp
    GPU_AVAILABLE = True
except ImportError:
    GPU_AVAILABLE = False

def log_error(message):
    with open(ERROR_LOG, "a") as f:
        f.write(f"{message}\n")

def upsample_chroma(pixel_array, expected_width):
    try:
        Y, Cb, Cr = pixel_array[..., 0], pixel_array[..., 1], pixel_array[..., 2]
        current_width = Cb.shape[1]
        if current_width == expected_width:
            return pixel_array
        factor = expected_width // current_width
        if factor * current_width != expected_width:
            raise ValueError(f"Cannot upsample: expected width {expected_width}, got {current_width}")
        if GPU_AVAILABLE:
            Cb_gpu = cp.array(Cb)
            Cr_gpu = cp.array(Cr)
            Cb_upsampled = cp.repeat(Cb_gpu, factor, axis=1)
            Cr_upsampled = cp.repeat(Cr_gpu, factor, axis=1)
            pixel_array = cp.stack([cp.array(Y), Cb_upsampled, Cr_upsampled], axis=-1)
            cp.get_default_memory_pool().free_all_blocks()
            return cp.asnumpy(pixel_array)
        else:
            Cb_upsampled = np.repeat(Cb, factor, axis=1)
            Cr_upsampled = np.repeat(Cr, factor, axis=1)
            pixel_array = np.stack([Y, Cb_upsampled, Cr_upsampled], axis=-1)
            return pixel_array
    except Exception as e:
        raise ValueError(f"Upsampling failed: {str(e)}")
    finally:
        gc.collect()

def ybr_to_rgb(pixel_array):
    try:
        if GPU_AVAILABLE:
            pixel_array_gpu = cp.array(pixel_array)
            Y, Cb, Cr = pixel_array_gpu[..., 0], pixel_array_gpu[..., 1], pixel_array_gpu[..., 2]
            rgb_array = cp.zeros((pixel_array.shape[0], pixel_array.shape[1], 3), dtype=np.uint8)
            rgb_array[..., 0] = cp.clip(Y + 1.402 * (Cr - 128), 0, 255)
            rgb_array[..., 1] = cp.clip(Y - 0.344136 * (Cb - 128) - 0.714136 * (Cr - 128), 0, 255)
            rgb_array[..., 2] = cp.clip(Y + 1.772 * (Cb - 128), 0, 255)
            pixel_array_gpu = None
            cp.get_default_memory_pool().free_all_blocks()
            return cp.asnumpy(rgb_array)
        else:
            Y, Cb, Cr = pixel_array[..., 0], pixel_array[..., 1], pixel_array[..., 2]
            rgb_array = np.zeros((pixel_array.shape[0], pixel_array.shape[1], 3), dtype=np.uint8)
            rgb_array[..., 0] = np.clip(Y + 1.402 * (Cr - 128), 0, 255)
            rgb_array[..., 1] = np.clip(Y - 0.344136 * (Cb - 128) - 0.714136 * (Cr - 128), 0, 255)
            rgb_array[..., 2] = np.clip(Y + 1.772 * (Cb - 128), 0, 255)
            return rgb_array
    except Exception as e:
        raise ValueError(f"YBR to RGB conversion failed: {str(e)}")
    finally:
        gc.collect()

def convert_dicom_to_png(dicom_path, output_path):
    try:
        # Read DICOM file with force=True to handle non-compliant files
        ds = pydicom.dcmread(dicom_path, force=True)
        
        # Ensure pixel data is available
        if not hasattr(ds, 'pixel_array'):
            raise ValueError("No pixel data found in DICOM file")

        # Get pixel array with pydicom's built-in color space conversion
        pixel_array = ds.pixel_array

        # Verify expected shape and photometric interpretation
        if len(pixel_array.shape) != 3 or pixel_array.shape[2] != 3:
            raise ValueError(f"Unexpected pixel array shape: {pixel_array.shape}")
        
        # Check photometric interpretation
        if ds.PhotometricInterpretation not in ["YBR_FULL_422", "RGB"]:
            raise ValueError(f"Unsupported PhotometricInterpretation: {ds.PhotometricInterpretation}")

        # pydicom automatically converts YBR_FULL_422 to RGB when pixel_array is accessed
        # No manual upsampling or YBR to RGB conversion needed
        rgb_array = pixel_array

        # Ensure data type is uint8 (as per report: Bits Allocated = 8, Pixel Representation = unsigned)
        if rgb_array.dtype != np.uint8:
            if rgb_array.dtype in (np.uint16, np.int16):
                # Normalize to uint8 if necessary (unlikely given report)
                rgb_array = np.clip((rgb_array / np.max(rgb_array) * 255), 0, 255).astype(np.uint8)
            else:
                raise ValueError(f"Unexpected pixel data type: {rgb_array.dtype}")

        # Create PIL image in RGB mode
        img = Image.fromarray(rgb_array, mode="RGB")

        # Save as PNG with no compression or optimization to ensure lossless output
        img.save(output_path, "PNG", compress_level=9)

        return True
    except Exception as e:
        error_msg = f"Error converting {dicom_path}: {str(e)}"
        print(error_msg)
        log_error(error_msg)
        return False
    finally:
        # Clean up memory
        ds = None
        pixel_array = None
        rgb_array = None
        img = None
        gc.collect()

class MolePreprocessorGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Mole Detection Preprocessor")
        self.root.geometry("600x400")

        # Threading events
        self.pause_event = threading.Event()
        self.cancel_event = threading.Event()

        # Load previous selections
        self.config = self.load_config()

        # GUI elements
        tk.Label(root, text="Duplicates CSV:").grid(row=0, column=0, padx=5, pady=5, sticky="e")
        self.duplicates_entry = tk.Entry(root, width=50)
        self.duplicates_entry.grid(row=0, column=1, padx=5, pady=5)
        tk.Button(root, text="Browse", command=self.browse_duplicates).grid(row=0, column=2, padx=5, pady=5)
        self.duplicates_entry.insert(0, self.config.get("duplicates_file", ""))

        tk.Label(root, text="Ground Truth CSV:").grid(row=1, column=0, padx=5, pady=5, sticky="e")
        self.groundtruth_entry = tk.Entry(root, width=50)
        self.groundtruth_entry.grid(row=1, column=1, padx=5, pady=5)
        tk.Button(root, text="Browse", command=self.browse_groundtruth).grid(row=1, column=2, padx=5, pady=5)
        self.groundtruth_entry.insert(0, self.config.get("groundtruth_file", ""))

        tk.Label(root, text="Malignant PNG Folder:").grid(row=2, column=0, padx=5, pady=5, sticky="e")
        self.malignant_entry = tk.Entry(root, width=50)
        self.malignant_entry.grid(row=2, column=1, padx=5, pady=5)
        tk.Button(root, text="Browse", command=self.browse_malignant).grid(row=2, column=2, padx=5, pady=5)
        self.malignant_entry.insert(0, self.config.get("malignant_folder", ""))

        self.convert_button = tk.Button(root, text="Convert", command=self.start_conversion_thread, bg="green", fg="white")
        self.convert_button.grid(row=3, column=1, pady=10)

        self.cancel_button = tk.Button(root, text="Cancel", command=self.cancel_conversion, bg="red", fg="white", state="disabled")
        self.cancel_button.grid(row=3, column=2, pady=10)

        self.progress_label = tk.Label(root, text="Progress: 0/0 (0.00%)")
        self.progress_label.grid(row=4, column=0, columnspan=3, pady=5)

        self.progress_bar = ttk.Progressbar(root, length=400, mode="determinate")
        self.progress_bar.grid(row=5, column=0, columnspan=3, pady=5)

        self.conversion_thread = None

    def load_config(self):
        try:
            if os.path.exists(CONFIG_FILE):
                with open(CONFIG_FILE, "r") as f:
                    return json.load(f)
        except Exception as e:
            print(f"Error loading config: {e}")
        return {}

    def save_config(self):
        config = {
            "duplicates_file": self.duplicates_entry.get(),
            "groundtruth_file": self.groundtruth_entry.get(),
            "malignant_folder": self.malignant_entry.get()
        }
        try:
            with open(CONFIG_FILE, "w") as f:
                json.dump(config, f, indent=4)
        except Exception as e:
            print(f"Error saving config: {e}")

    def browse_duplicates(self):
        file = filedialog.askopenfilename(
            initialdir=os.path.dirname(self.duplicates_entry.get()) or INPUT_FOLDER,
            filetypes=[("CSV files", "*.csv")]
        )
        if file:
            self.duplicates_entry.delete(0, tk.END)
            self.duplicates_entry.insert(0, file)

    def browse_groundtruth(self):
        file = filedialog.askopenfilename(
            initialdir=os.path.dirname(self.groundtruth_entry.get()) or INPUT_FOLDER,
            filetypes=[("CSV files", "*.csv")]
        )
        if file:
            self.groundtruth_entry.delete(0, tk.END)
            self.groundtruth_entry.insert(0, file)

    def browse_malignant(self):
        folder = filedialog.askdirectory(initialdir=self.malignant_entry.get() or INPUT_FOLDER)
        if folder:
            self.malignant_entry.delete(0, tk.END)
            self.malignant_entry.insert(0, folder)

    def signal_handler(self):
        self.pause_event.set()
        print("\nConversion paused. Progress saved. Restart the program to resume.")
        sys.exit(0)

    def load_progress(self):
        if os.path.exists(PROGRESS_FILE):
            with open(PROGRESS_FILE, "r") as f:
                return json.load(f)
        return {"converted": [], "total_files": 0}

    def save_progress(self, converted_files, total_files):
        with open(PROGRESS_FILE, "w") as f:
            json.dump({"converted": converted_files, "total_files": total_files}, f)

    def cancel_conversion(self):
        self.cancel_event.set()
        self.convert_button["state"] = "normal"
        self.cancel_button["state"] = "disabled"
        self.progress_label.config(text="Cancelling...")

    def start_conversion_thread(self):
        if self.conversion_thread and self.conversion_thread.is_alive():
            return
        self.convert_button["state"] = "disabled"
        self.cancel_button["state"] = "normal"
        self.progress_label.config(text="Progress: 0/0 (0.00%)")
        self.progress_bar["value"] = 0
        self.pause_event.clear()
        self.cancel_event.clear()
        self.conversion_thread = threading.Thread(target=self.start_conversion)
        self.conversion_thread.start()
        self.root.after(100, self.check_conversion_progress)

    def check_conversion_progress(self):
        if self.conversion_thread and self.conversion_thread.is_alive():
            self.root.after(100, self.check_conversion_progress)
        else:
            self.convert_button["state"] = "normal"
            self.cancel_button["state"] = "disabled"

    def start_conversion(self):
        duplicates_file = self.duplicates_entry.get()
        groundtruth_file = self.groundtruth_entry.get()
        malignant_folder = self.malignant_entry.get()

        if not all([duplicates_file, groundtruth_file, malignant_folder]):
            self.root.after(0, lambda: messagebox.showerror("Error", "Please select all files and folders."))
            self.root.after(0, lambda: self.convert_button.config(state="normal"))
            self.root.after(0, lambda: self.cancel_button.config(state="disabled"))
            return

        if not os.path.exists(duplicates_file):
            self.root.after(0, lambda: messagebox.showerror("Error", "Duplicates file does not exist."))
            self.root.after(0, lambda: self.convert_button.config(state="normal"))
            self.root.after(0, lambda: self.cancel_button.config(state="disabled"))
            return
        if not os.path.exists(groundtruth_file):
            self.root.after(0, lambda: messagebox.showerror("Error", "Ground truth file does not exist."))
            self.root.after(0, lambda: self.convert_button.config(state="normal"))
            self.root.after(0, lambda: self.cancel_button.config(state="disabled"))
            return
        if not os.path.exists(INPUT_FOLDER):
            self.root.after(0, lambda: messagebox.showerror("Error", f"Input DICOM folder {INPUT_FOLDER} does not exist."))
            self.root.after(0, lambda: self.convert_button.config(state="normal"))
            self.root.after(0, lambda: self.cancel_button.config(state="disabled"))
            return

        self.save_config()

        try:
            labels_df = pd.read_csv(groundtruth_file)
            if 'image_name' not in labels_df.columns or 'target' not in labels_df.columns:
                self.root.after(0, lambda: messagebox.showerror("Error", "Ground truth CSV must have 'image_name' and 'target' columns."))
                self.root.after(0, lambda: self.convert_button.config(state="normal"))
                self.root.after(0, lambda: self.cancel_button.config(state="disabled"))
                return
        except Exception as e:
            self.root.after(0, lambda: messagebox.showerror("Error", f"Error reading ground truth: {e}"))
            self.root.after(0, lambda: self.convert_button.config(state="normal"))
            self.root.after(0, lambda: self.cancel_button.config(state="disabled"))
            return

        try:
            duplicates_df = pd.read_csv(duplicates_file)
            if 'image_name_1' not in duplicates_df.columns or 'image_name_2' not in duplicates_df.columns:
                self.root.after(0, lambda: messagebox.showerror("Error", "Duplicates CSV must have 'image_name_1' and 'image_name_2' columns."))
                self.root.after(0, lambda: self.convert_button.config(state="normal"))
                self.root.after(0, lambda: self.cancel_button.config(state="disabled"))
                return
            duplicate_images = set(duplicates_df['image_name_1'].astype(str).str.strip().tolist()) | \
                              set(duplicates_df['image_name_2'].astype(str).str.strip().tolist())
        except Exception as e:
            self.root.after(0, lambda: messagebox.showerror("Error", f"Error reading duplicates: {e}"))
            self.root.after(0, lambda: self.convert_button.config(state="normal"))
            self.root.after(0, lambda: self.cancel_button.config(state="disabled"))
            return

        labels_df['image_name'] = labels_df['image_name'].astype(str).str.strip()
        labels_df = labels_df[~labels_df['image_name'].isin(duplicate_images)]

        # Select only malignant images
        malignant_files = labels_df[labels_df['target'] == 1]['image_name'].tolist()
        num_malignant = len(malignant_files)

        if num_malignant == 0:
            self.root.after(0, lambda: messagebox.showerror("Error", "No malignant images found after excluding duplicates."))
            self.root.after(0, lambda: self.convert_button.config(state="normal"))
            self.root.after(0, lambda: self.cancel_button.config(state="disabled"))
            return

        selected_files = malignant_files
        print(f"Processing {num_malignant} malignant images.")

        dicom_files = [f for f in selected_files if os.path.exists(os.path.join(INPUT_FOLDER, f"{f}.dcm"))]
        if len(dicom_files) != len(selected_files):
            missing_files = [f for f in selected_files if f not in dicom_files]
            missing_msg = f"Only {len(dicom_files)} of {len(selected_files)} selected files found in {INPUT_FOLDER}. Missing: {', '.join(missing_files[:5])}{'...' if len(missing_files) > 5 else ''}"
            print(missing_msg)
            log_error(missing_msg)
            self.root.after(0, lambda: messagebox.showwarning("Warning", missing_msg))
        if not dicom_files:
            self.root.after(0, lambda: messagebox.showerror("Error", "No valid DICOM files found."))
            self.root.after(0, lambda: self.convert_button.config(state="normal"))
            self.root.after(0, lambda: self.cancel_button.config(state="disabled"))
            return

        os.makedirs(malignant_folder, exist_ok=True)

        progress = self.load_progress()
        converted_files = set(progress["converted"])
        dicom_files = [f for f in dicom_files if os.path.join(INPUT_FOLDER, f"{f}.dcm") not in converted_files]
        total_files = len(dicom_files) + len(converted_files)

        if not dicom_files:
            self.root.after(0, lambda: messagebox.showinfo("Info", "No new DICOM files to convert."))
            self.root.after(0, lambda: self.convert_button.config(state="normal"))
            self.root.after(0, lambda: self.cancel_button.config(state="disabled"))
            return

        print(f"Found {len(dicom_files)} new DICOM files to convert.")

        max_workers = min(os.cpu_count() or 6, 4)  # Reduced for stability
        batch_size = 10
        for i in range(0, len(dicom_files), batch_size):
            if self.pause_event.is_set() or self.cancel_event.is_set():
                break
            batch_files = dicom_files[i:i + batch_size]
            with ProcessPoolExecutor(max_workers=max_workers) as executor:
                futures = []
                for dicom_file in batch_files:
                    if self.pause_event.is_set() or self.cancel_event.is_set():
                        break
                    dicom_path = os.path.join(INPUT_FOLDER, f"{dicom_file}.dcm")
                    output_path = os.path.join(malignant_folder, f"{dicom_file}.png")
                    futures.append(executor.submit(convert_dicom_to_png, dicom_path, output_path))

                completed = len(converted_files)
                for future in futures:
                    if self.pause_event.is_set() or self.cancel_event.is_set():
                        break
                    try:
                        if future.result(timeout=300):  # 5-minute timeout per file
                            completed += 1
                            converted_files.add(os.path.join(INPUT_FOLDER, f"{batch_files[completed - len(converted_files) - 1]}.dcm"))
                            self.save_progress(list(converted_files), total_files)
                            self.root.after(0, lambda c=completed, t=total_files: self.progress_label.config(
                                text=f"Progress: {c}/{t} ({(c/t)*100:.2f}%)"))
                            self.root.after(0, lambda c=completed, t=total_files: self.progress_bar.config(
                                value=(c/t)*100))
                            print(f"Converted {completed}/{total_files} files ({(completed/total_files)*100:.2f}%)")
                    except Exception as e:
                        print(f"Batch error: {e}")
                        log_error(f"Batch error: {e}")
                executor._processes = 0  # Force close processes
            gc.collect()
            if GPU_AVAILABLE:
                cp.get_default_memory_pool().free_all_blocks()
            time.sleep(1)  # Brief pause to release resources

        if not (self.pause_event.is_set() or self.cancel_event.is_set()):
            print("Conversion completed.")
            if os.path.exists(PROGRESS_FILE):
                os.remove(PROGRESS_FILE)
            self.root.after(0, lambda: messagebox.showinfo("Success", f"Conversion completed: {num_malignant} malignant images. Check output folder and error log."))
        elif self.cancel_event.is_set():
            print("Conversion cancelled.")
            self.root.after(0, lambda: messagebox.showinfo("Cancelled", "Conversion was cancelled. Partial progress saved."))
        elif self.pause_event.is_set():
            print("Conversion paused.")
            self.root.after(0, lambda: messagebox.showinfo("Paused", "Conversion was paused. Partial progress saved. Restart to resume."))
        self.root.after(0, lambda: self.convert_button.config(state="normal"))
        self.root.after(0, lambda: self.cancel_button.config(state="disabled"))

def main():
    import signal
    root = tk.Tk()
    app = MolePreprocessorGUI(root)
    # Set SIGINT handler in main thread
    signal.signal(signal.SIGINT, lambda sig, frame: app.signal_handler())
    root.mainloop()

if __name__ == "__main__":
    main()