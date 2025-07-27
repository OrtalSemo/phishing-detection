import pyautogui
import os
import pandas as pd
import tkinter as tk
from tkinter import simpledialog, messagebox

output_dir = r"C:\Users\Owner\Pictures\evaluation_dataset\screenshots"
os.makedirs(output_dir, exist_ok=True)
csv_path = r"C:\Users\Owner\Pictures\evaluation_dataset\image_labels.csv"

if os.path.exists(csv_path):
    df = pd.read_csv(csv_path)
else:
    df = pd.DataFrame(columns=["filename", "label", "source_type"])

def get_next_img_num():
    numbers = set()
    for fname in os.listdir(output_dir):
        if fname.startswith('img_') and fname.endswith('.png'):
            try:
                n = int(fname[4:7])
                numbers.add(n)
            except:
                continue
    for fname in df['filename']:
        if isinstance(fname, str) and fname.startswith('img_') and fname.endswith('.png'):
            try:
                n = int(fname[4:7])
                numbers.add(n)
            except:
                continue
    if numbers:
        return max(numbers) + 1
    else:
        return 1

def capture_and_label():
    # הסתרת חלון ה-GUI לפני צילום המסך
    root.withdraw()

    # צילום מסך
    next_num = get_next_img_num()
    filename = f"img_{next_num:03d}.png"
    filepath = os.path.join(output_dir, filename)
    img = pyautogui.screenshot()
    img.save(filepath)
    print(f"Saved screenshot: {filepath}")

    # החזרת חלון ה-GUI לאחר הצילום
    root.deiconify()

    label = simpledialog.askstring(title="Label", prompt="Enter label (phishing/legitimate):", parent=root)
    if label not in ["phishing", "legitimate"]:
        messagebox.showerror("Error", "Invalid label. Try again.", parent=root)
        os.remove(filepath)
        return

    source_type = simpledialog.askstring(title="Source Type", prompt="Enter source type (email/webpage/sms/pdf/app):", parent=root)
    if source_type not in ["email", "webpage", "sms", "pdf", "app"]:
        messagebox.showerror("Error", "Invalid source type. Try again.", parent=root)
        os.remove(filepath)
        return

    global df
    new_row = {"filename": filename, "label": label, "source_type": source_type}
    df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
    df.to_csv(csv_path, index=False)
    print(f"Labeled and saved: {filename}")
    messagebox.showinfo("Saved", f"{filename} saved and labeled.", parent=root)

root = tk.Tk()
root.title("Screenshot Labeler")
root.geometry("300x150")

btn = tk.Button(root, text="Capture & Label Screenshot", command=capture_and_label, height=2, width=30)
btn.pack(pady=40)

root.mainloop()
