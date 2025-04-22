# gui.py
import tkinter as tk
from tkinter import messagebox
from model import predict_spam

# Callback function for button
def check_spam():
    text = input_box.get("1.0", tk.END).strip()
    if not text:
        messagebox.showwarning("Input Error", "Please enter email or SMS content.")
        return
    label, confidence = predict_spam(text)
    result_label.config(text=f"Classification: {label}")
    confidence_label.config(text=f"Confidence: {confidence}%")

# GUI setup
root = tk.Tk()
root.title("Spam Email Detector")
root.geometry("400x300")
root.resizable(False, False)

# Header
header = tk.Label(root, text="Spam Email Detection", font=("Helvetica", 14, "bold"))
header.pack(pady=10)

# Text input field
input_box = tk.Text(root, height=6, width=40, font=("Helvetica", 10))
input_box.pack(pady=5)
input_box.insert("1.0", "")
input_box.insert_placeholder = lambda: input_box.insert("1.0", "Enter the email content here...")
input_box.bind("<FocusIn>", lambda event: input_box.delete("1.0", tk.END) if input_box.get("1.0", tk.END).strip() == "Enter the email content here..." else None)
input_box.bind("<FocusOut>", lambda event: input_box.insert_placeholder() if not input_box.get("1.0", tk.END).strip() else None)
input_box.insert_placeholder()

# Classify button
check_button = tk.Button(root, text="Classify", command=check_spam, font=("Inter", 10))
check_button.pack(pady=10)

# Result labels
result_label = tk.Label(root, text="Classification: ", font=("Inter", 10))
result_label.pack()

confidence_label = tk.Label(root, text="Confidence: ", font=("Inter", 10))
confidence_label.pack()

root.mainloop()