import json
import boto3
import logging

logger = logging.getLogger()
logger.setLevel(logging.DEBUG)


def lambda_handler(event, context):
    """
    main handler of events
    """
    notebook_instance_name = 'email-spam-detection'
    client = boto3.client('sagemaker')

    # start the notebook instance
    response = client.start_notebook_instance(NotebookInstanceName=notebook_instance_name)
    print("Start the notebook instance: ", response)

    return {
        'statusCode': 200,
        'body': json.dumps('Finished retraining the spam classifier.')
    }
