from typing import TypedDict, Literal

class EmailInput(TypedDict):
    subject: str
    message_content: str
    sender: str

class ClassificationResult(TypedDict):
    label: Literal["Spam", "Likely Spam", "Not Spam"]
    confidence_score: float