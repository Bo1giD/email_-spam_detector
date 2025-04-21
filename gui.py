import tkinter as tk
from tkinter import messagebox
from model.classifier import classify_email
from model.types import EmailInput

def launch_gui():
    def on_submit():
        subject = entry_subject.get()
        content = text_content.get("1.0", tk.END).strip()
        sender = entry_sender.get()

        if not subject or not content or not sender:
            messagebox.showerror("Input Error", "All fields must be filled.")
            return

        email: EmailInput = {
            "subject": subject,
            "message_content": content,
            "sender": sender
        }

        result = classify_email(email)
        label_result.config(
            text=f"Result: {result['label']} ({result['confidence_score'] * 100:.1f}%)"
        )

    root = tk.Tk()
    root.title("Spam Email Detector")

    tk.Label(root, text="Sender:").pack()
    entry_sender = tk.Entry(root, width=50)
    entry_sender.pack()

    tk.Label(root, text="Subject:").pack()
    entry_subject = tk.Entry(root, width=50)
    entry_subject.pack()

    tk.Label(root, text="Email Content:").pack()
    text_content = tk.Text(root, height=10, width=60)
    text_content.pack()

    tk.Button(root, text="Check Spam", command=on_submit).pack(pady=10)
    label_result = tk.Label(root, text="", font=("Arial", 12))
    label_result.pack()

    root.mainloop()