import tkinter as tk
from tkinter import filedialog, messagebox
import cv2
import os
import shutil

# Global config
FRAME_FOLDER = "extracted_frames"
MASK_FOLDER = "annotated_masks"

# --- GUI Application ---
class UltrasoundApp:
    def __init__(self, root):
        self.root = root
        root.title("Ultrasound to STL Generator")

        # Input video path
        self.video_path = tk.StringVar()
        tk.Label(root, text="Ultrasound Video File:").grid(row=0, column=0, sticky="e")
        tk.Entry(root, textvariable=self.video_path, width=50).grid(row=0, column=1)
        tk.Button(root, text="Browse", command=self.browse_video).grid(row=0, column=2)

        # Output STL location
        self.output_path = tk.StringVar()
        tk.Label(root, text="Output STL File Location:").grid(row=1, column=0, sticky="e")
        tk.Entry(root, textvariable=self.output_path, width=50).grid(row=1, column=1)
        tk.Button(root, text="Browse", command=self.browse_output).grid(row=1, column=2)

        # Buttons
        tk.Button(root, text="Extract Frames", command=self.extract_frames).grid(row=2, column=0)
        tk.Button(root, text="Annotate Frames", command=self.annotate_frames).grid(row=2, column=1)
        tk.Button(root, text="Generate STL", command=self.generate_stl).grid(row=2, column=2)

    def browse_video(self):
        path = filedialog.askopenfilename(filetypes=[("Video Files", "*.mp4 *.avi")])
        if path:
            self.video_path.set(path)

    def browse_output(self):
        path = filedialog.asksaveasfilename(defaultextension=".stl", filetypes=[("STL files", "*.stl")])
        if path:
            self.output_path.set(path)

    def extract_frames(self):
        if not self.video_path.get():
            messagebox.showerror("Error", "Please select a video file.")
            return

        cap = cv2.VideoCapture(self.video_path.get())
        os.makedirs(FRAME_FOLDER, exist_ok=True)
        frame_idx = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break
            cv2.imwrite(f"{FRAME_FOLDER}/frame_{frame_idx:04d}.png", frame)
            frame_idx += 1

        cap.release()
        messagebox.showinfo("Done", f"Extracted {frame_idx} frames to {FRAME_FOLDER}")

    def annotate_frames(self):
        messagebox.showinfo("Instructions", f"Manually annotate frames in {FRAME_FOLDER} using your favorite tool (e.g. Paint). Save green annotations in {MASK_FOLDER}")
        os.makedirs(MASK_FOLDER, exist_ok=True)
        shutil.copytree(FRAME_FOLDER, MASK_FOLDER, dirs_exist_ok=True)

    def generate_stl(self):
        if not self.output_path.get():
            messagebox.showerror("Error", "Please specify output STL file location.")
            return

        # 👉 Call your existing STL generation pipeline here:
        # e.g. process annotated_masks/, generate geometry, export STL
        os.system(f"python your_stl_generation_script.py {MASK_FOLDER} {self.output_path.get()}")

        messagebox.showinfo("Done", f"STL file generated at:\n{self.output_path.get()}")

# --- Launch GUI ---
if __name__ == "__main__":
    root = tk.Tk()
    app = UltrasoundApp(root)
    root.mainloop()
