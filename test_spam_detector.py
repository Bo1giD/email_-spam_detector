import unittest
from model import predict_spam

class TestSpamDetection(unittest.TestCase):
    def test_obvious_spam(self):
        email_input = {
            "sender": "win@scam.com",
            "subject": "Free money!",
            "body": "Click here to claim your free $1000"
        }
        label, confidence = predict_spam(email_input)
        self.assertEqual(label, "Likely Spam (Phishing)")
        self.assertGreaterEqual(confidence, 80)

    def test_real_message(self):
        email_input = {
            "sender": "team@company.com",
            "subject": "Meeting Tomorrow",
            "body": "Don't forget about tomorrow's team meeting at 10am."
        }
        label, confidence = predict_spam(email_input)
        self.assertEqual(label, "Not Spam")
        self.assertLessEqual(confidence, 30)

    def test_likely_spam(self):
        email_input = {
            "sender": "promo@deals.com",
            "subject": "Special discount just for you",
            "body": "Save 50% on all items this weekend only!"
        }
        label, confidence = predict_spam(email_input)
        self.assertEqual(label, "Likely Spam")
        self.assertGreater(confidence, 30)
        self.assertLess(confidence, 60)

    def test_empty_fields(self):
        with self.assertRaises(ValueError):
            predict_spam({
                "sender": "",
                "subject": "",
                "body": ""
            })
if __name__ == "__main__":
    unittest.main()
