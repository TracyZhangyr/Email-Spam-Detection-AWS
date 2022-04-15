import json
import os
import urllib.parse
import boto3
import logging
from botocore.exceptions import ClientError
from sms_spam_classifier_utilities import one_hot_encode, vectorize_sequences
from email.parser import BytesParser
from email.policy import default


logger = logging.getLogger()
logger.setLevel(logging.DEBUG)


def send_email(sender, recipient, receive_date, subject, body, classification, confidence_score):
    """
    send an email to the user
    """
    SENDER = sender
    RECIPIENT = recipient
    AWS_REGION = "us-east-1"
    SUBJECT = "Spam Email Detection Reply"
    CHARSET = "UTF-8"

    # The email body for recipients with non-HTML email clients.
    BODY_TEXT = "We received your email sent at {EMAIL_RECEIVE_DATE} with the subject {EMAIL_SUBJECT}.\n\n" \
                "Here is a 240 character sample of the email body:\n\n{EMAIL_BODY}\n\n" \
                "The email was categorized as {CLASSIFICATION} with a {CLASSIFICATION_CONFIDENCE_SCORE}% " \
                "confidence.".format(EMAIL_RECEIVE_DATE=receive_date, EMAIL_SUBJECT=subject,
                                     EMAIL_BODY=body[:240], CLASSIFICATION=classification,
                                     CLASSIFICATION_CONFIDENCE_SCORE=confidence_score)

    # Create a new SES resource and specify a region.
    client = boto3.client('ses', region_name=AWS_REGION)

    # Try to send the email.
    try:
        # Provide the contents of the email.
        response = client.send_email(
            Destination={
                'ToAddresses': [
                    RECIPIENT,
                ],
            },
            Message={
                'Body': {
                    'Text': {
                        'Charset': CHARSET,
                        'Data': BODY_TEXT,
                    },
                },
                'Subject': {
                    'Charset': CHARSET,
                    'Data': SUBJECT,
                },
            },
            Source=SENDER,
        )
    # Display an error if something goes wrong.
    except ClientError as e:
        print(e.response['Error']['Message'])
    else:
        print("Email sent! Message ID:"),
        print(response['MessageId'])


def lambda_handler(event, context):
    """
    main handler of events
    """
    # get the object from the event
    s3 = boto3.client('s3')
    bucket = event['Records'][0]['s3']['bucket']['name']
    key = urllib.parse.unquote_plus(event['Records'][0]['s3']['object']['key'], encoding='utf-8')
    response = s3.get_object(Bucket=bucket, Key=key)

    # get email message
    text = response['Body'].read()
    msg = BytesParser(policy=default).parsebytes(text)

    # extract email header items
    recipient = msg['from']
    sender = msg['to']
    receive_date = msg['date']
    subject = msg['subject']

    # extract email body
    simplest = msg.get_body(preferencelist='plain')
    # strip out new line characters "\n" in the email body
    body = ' '.join([line for line in simplest.get_content().splitlines() if line])
    body = body.strip() if body else ''
    print("Received email body: ", body)

    # uses the prediction endpoint (E1) to predict if the email is spam or not
    test_messages = [body]
    vocabulary_length = 9013
    one_hot_test_messages = one_hot_encode(test_messages, vocabulary_length)
    encoded_test_messages = vectorize_sequences(one_hot_test_messages, vocabulary_length)

    # get the env variable for prediction endpoint
    ENDPOINT_NAME = os.environ['ENDPOINT_NAME']
    print("ENDPOINT_NAME: ", ENDPOINT_NAME)

    sagemaker = boto3.client('sagemaker-runtime')

    content_type = "application/json"  # The MIME type of the input data in the request body.
    payload = json.dumps(encoded_test_messages.tolist())  # Payload for inference.

    response = sagemaker.invoke_endpoint(EndpointName=ENDPOINT_NAME, ContentType=content_type, Body=payload)
    result = json.loads(response['Body'].read())
    print("SageMaker Prediction: ", result)

    predicted_label = result['predicted_label'][0][0]
    predicted_probability = result['predicted_probability'][0][0]

    if predicted_label == 1.0:
        classification = "SPAM"
        confidence_score = round((predicted_probability * 100), 5)
    else:
        classification = "HAM"
        confidence_score = round(((1 - predicted_probability) * 100), 5)

    # reply to the sender of the email with a message
    send_email(sender, recipient, receive_date, subject, body, classification, confidence_score)

    return {
        'statusCode': 200,
        'body': json.dumps('Finished LF1.')
    }
