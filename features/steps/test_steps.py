from behave import given, when, then
from model import predict_spam

@given('I have an email with subject "{subject}"')
def step_given_subject(context, subject):
    context.email = {
        "sender": "test@example.com",
        "subject": subject,
        "body": ""
    }

@given('body "{body}"')
def step_given_body(context, body):
    context.email["body"] = body

@when("I classify the email")
def step_when_classify(context):
    context.label, context.confidence = predict_spam(context.email)

@then('the result should be "{expected}"')
def step_then_result(context, expected):
    assert context.label == expected, f"Expected {expected}, got {context.label}"