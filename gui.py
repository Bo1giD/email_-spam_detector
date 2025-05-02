# gui.py
import tkinter as tk
from tkinter import messagebox
import re
from typing import TypedDict
from model import predict_spam

class SpamInput(TypedDict):
    sender: str
    subject: str
    body: str

def is_valid_email(email: str) -> bool:
    email_regex = r"^[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+$"
    return bool(re.match(email_regex, email))

def check_spam():
    sender = sender_entry.get().strip()
    subject = subject_entry.get().strip()
    body = body_text.get("1.0", tk.END).strip()

    # Input validation
    if not (sender and subject and body):
        messagebox.showerror("Validation Error", "All fields must be filled out.")
        return
    if not is_valid_email(sender):
        messagebox.showerror("Validation Error", "Please enter a valid sender email address.")
        return

    email_input: SpamInput = {"sender": sender, "subject": subject, "body": body}
    label, confidence = predict_spam(email_input)
    result_label.config(text=f"Classification: {label}")
    confidence_label.config(text=f"Confidence: {confidence}%")

root = tk.Tk()
root.title("Spam Email Detection")
root.geometry("500x400")
root.resizable(False, False)

tk.Label(root, text="Sender Email:", font=("Helvetica", 10)).pack(pady=5)
sender_entry = tk.Entry(root, width=50)
sender_entry.pack()

tk.Label(root, text="Subject:", font=("Helvetica", 10)).pack(pady=5)
subject_entry = tk.Entry(root, width=50)
subject_entry.pack()

tk.Label(root, text="Email Body:", font=("Helvetica", 10)).pack(pady=5)
body_text = tk.Text(root, height=8, width=60)
body_text.pack()

tk.Button(root, text="Classify", command=check_spam, font=("Helvetica", 10)).pack(pady=10)

result_label = tk.Label(root, text="Classification: ", font=("Helvetica", 10, "bold"))
result_label.pack()

confidence_label = tk.Label(root, text="Confidence: ", font=("Helvetica", 10, "bold"))
confidence_label.pack()

root.mainloop()