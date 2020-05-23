# STARTING INSTANCE AND RUNNING NOTEBOOKS

1. Start the instance from scratch, or start with the provided AMI and skip installation and setup of anaconda, jupyter, (steps 3-7) and other python packages. If starting from scratch add more storage, 8 GBs is not enough, currently using 32 GBs. As for instance type t2.xlarge feels sufficient.

I've created an AMI here https://ca-central-1.console.aws.amazon.com/ec2/v2/home?region=ca-central-1#Images:sort=name

You'll still have to manually choose the instance type, add the key value pair of Owner/capstone-mda, and choose to join the default existing security group, as well as manage your own access key for this instance.

named mds-capstone-mda

2. Connect to the instance with your access key with something like the following:

ssh -i "jamesh.pem" ubuntu@ec2-35-183-182-11.ca-central-1.compute.amazonaws.com

* make sure the above link starts with ubuntu and not root.

3. Run the following:

wget https://repo.anaconda.com/archive/Anaconda3-2020.02-Linux-x86_64.sh

4. Run the following:

bash Anaconda3-2020.02-Linux-x86_64.sh

Hold enter a lot and type yes and enter when prompted, then press enter again, then wait, then type yes and enter again.
* you may need to restart the terminal for the install to take effect.
* if the above doesn't work, use the command ls to get the correct file name

5. Install Jupiter notebook with the following:
conda install -c conda-forge jupyterlab

6. Run the following to set a password for your jupyter server, and remember this password:

jupyter notebook password

* if this throws an error run the following:
jupyter notebook --generate-config

7. Run the following to set up the jupyter server:

cd ~ 
mkdir ssl 
cd ssl 
openssl req -x509 -nodes -days 365 -newkey rsa:2048 -keyout mykey.key -out mycert.pem

* the details of the certification do not matter, just spam enter.

START FROM BELOW HERE IF USING IMAGE

8. Run the following to start the jupyter server:

cd
jupyter notebook --certfile=~/ssl/mycert.pem --keyfile ~/ssl/mykey.key

9. Connecting to the jupyter server in a local browser: 

*Note that this is for OSX machines only, for windows machines consult https://docs.aws.amazon.com/dlami/latest/devguide/dlami-dg.pdf#setup-jupyter on page 23.

Open a new terminal window and navigate to where your .pem access key is and run the following command modified to match your instance and .pem file:

ssh -i "jamesh.pem" -N -f -L 8888:localhost:8888 ubuntu@ec2-35-182-132-222.ca-central-1.compute.amazonaws.com

Open google chrome and go to:

chrome://flags/#allow-insecure-localhost

and enable the first flag, then relaunch the browser as prompted, and go to:

https://localhost:8888/

This should launch the jupyter notebook instance we are all familiar with.



# GETTING DATA FROM S3

IF USING IMAGE, CAN SKIP INSTALL AND CONFIGURATION

Install AWS CLI by running the following, seperately:

suda apt install unzip

curl "https://awscli.amazonaws.com/awscli-exe-linux-x86_64.zip" -o "awscliv2.zip"

unzip awscliv2.zip

sudo ./aws/install

Now you need to configure AWS to connect to the bucket that I created at https://s3.console.aws.amazon.com/s3/buckets/mds-capstone-mda/?region=ca-central-1 by running the following:

aws configure

And inputting my access key credentials:

Access key ID: AKIATB63UHM3JGA72B46

Secret access key: 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

And the rest of the fields can be left empty

Create and/or navigate to the folder that you want the files to be downloaded to and run the following:

aws s3 sync s3://mds-capstone-mda .

Including the period ^^, and that should be enough to clone the entire bucket to the folder.

## Dump of commands used to install packages for baseline model:

conda install pytorch torchvision -c pytorch
pip install torchsummary

