Feature: Spam detection

  Scenario: Obvious phishing email is passed

    Given I have an email with subject "Free prize inside"
    And body "Click here to claim your free money"
    When I classify the email
    Then the result should be "Likely Spam (Phishing)"

  Scenario: A normal business email is passed
   
    Given I have an email with subject "Project meeting update"
    And body "Please review the notes before tomorrow"
    When I classify the email
    Then the result should be "Not Spam"