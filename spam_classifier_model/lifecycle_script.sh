set -e

ENVIRONMENT=python3

#Declare all the jupyter notebooks that need to run, within the Sagemaker instance
FILE="/home/ec2-user/SageMaker/smlambdaworkshop/training/sms_spam_classifier_mxnet.ipynb"

#Activate python environment. The lifecycle configuration cannot autodetect the environment
source /home/ec2-user/anaconda3/bin/activate "$ENVIRONMENT"

#Execute the notebook in background
nohup jupyter nbconvert "$FILE" --ExecutePreprocessor.kernel_name=python3 --to notebook --inplace  --ExecutePreprocessor.timeout=600 --execute &

#Deactivate the python environment
source /home/ec2-user/anaconda3/bin/deactivate

IDLE_TIME=600  # 10 minutes

echo "Fetching the autostop script"
wget https://raw.githubusercontent.com/aws-samples/amazon-sagemaker-notebook-instance-lifecycle-config-samples/master/scripts/auto-stop-idle/autostop.py

echo "Starting the SageMaker autostop script in cron"
(crontab -l 2>/dev/null; echo "*/1 * * * * /usr/bin/python $PWD/autostop.py --time $IDLE_TIME --ignore-connections") | crontab -
