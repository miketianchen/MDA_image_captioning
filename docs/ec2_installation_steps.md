# Creating AWS EC2 instance/AMI

Please follow the steps in this file to create your AMI/instance.

1. Launch an EC2 instance with the `Deep Learning Base AMI (Ubuntu 18.04)` AMI

2. Run the following command in your local terminal to connect with the EC2 instance

```
ssh -i [key.pem] -L 8888:localhost:8888 ubuntu@[public domain name].ca-central-1.compute.amazonaws.com
```

3. Prepare the EC2 instance:

```
# Update ubuntu
sudo apt-get update
sudo apt-get upgrade

# Install pip
sudo apt install python3-pip

# Install miniconda instead of Anaconda to save space and time
wget https://repo.anaconda.com/miniconda/Miniconda3-py37_4.8.3-Linux-x86_64.sh
bash Miniconda3-py37_4.8.3-Linux-x86_64.sh
export PATH="/home/ubuntu/miniconda3/bin:$PATH"

# Set up AWS CLI
aws configure

# Set up aws configure, please replace with your own information
AWS Access Key ID [None]: AKIATB63UHM3KTWNDPFG
AWS Secret Access Key [None]: 7TarprC5CWlTHdOcNJ1LwmV80d/spZV5ShbMhRCO
Default region name [None]: ca-central-1
Default output format [None]: json

# OPTIONAL: install jupyter lab 
# We used jupyter lab to develop our model and perform analysis
# all our works can be found under notebooks folder of this repo
conda install -c conda-forge jupyterlab
# Set notebook password to 123
jupyter notebook password

# OPTIONAL: update git
sudo add-apt-repository ppa:git-core/ppa 
sudo apt update
sudo apt install git

# OPTIONAL: handle trash
sudo apt install trash-cli
trash-empty

####################################################
# Install extra packages 
####################################################

####################################################
# Option 1

# Clone the repo 
git clone https://github.com/UBC-MDS/591_capstone_2020-mda-mds.git
cd 591_capstone_2020-mda-mds

pip install -r requirements.txt

# OPTIONAL: packages to run notebooks
pip install -r optional.txt
jupyter nbextension enable --py --sys-prefix widgetsnbextension

####################################################

####################################################
# Option 2
pip install boto3==1.14.9
pip install pandas==1.0.5
pip install nltk==3.5
pip install sklearn==0.0
pip install docopt==0.6.2
pip install django==3.0.3
pip install tensorflow_hub==0.8.0
pip install tensorflow_text==2.2.1
conda install pytorch=1.5.1 torchvision=0.6.1 cudatoolkit=10.2 -c pytorch

# OPTIONAL: packages to run notebooks
pip install matplotlib==3.2.2
pip install altair==3.2.0
pip install torchsummary==1.5.1
pip install ipywidgets==7.5.1
pip install gensim==3.8.0
pip install wordcloud==1.7.0
conda install -c conda-forge pyldavis=2.1.2
jupyter nbextension enable --py --sys-prefix widgetsnbextension

####################################################

# Download nltk_data
python
>>> import nltk
>>> nltk.download('punkt')
```
4. After configuring your instance, you can choose to save this instance as AMI so that you do not need to configure another instance from scratching again. Please follow the instruction [here](https://docs.aws.amazon.com/toolkit-for-visual-studio/latest/user-guide/tkv-create-ami-from-instance.html).

5. Before launching jupyter lab or visualization tool from instance, you need to configure chrome on your machine.
  - Open google chrome and go to: chrome://flags/#allow-insecure-localhost
  - Enable the first flag, then relaunch the browser as prompted


# Running visualization tool on the EC2 instance

1. From the root of the repo, open `scr/visualization/mda_mds/mda_mds/settings.py`.

2. Add `[public domain name].ca-central-1.compute.amazonaws.com` to `ALLOWED_HOSTS`

For example:

```
ALLOWED_HOSTS = ['ec2-3-96-51-16.ca-central-1.compute.amazonaws.com', 'localhost', '127.0.0.1']
```

3. Save and exit, navigate to `scr/visualization/mda_mds/` and call

```
python manage.py runserver [public domain name].ca-central-1.compute.amazonaws.com:[port]
```

For example:

```
python manage.py runserver ec2-3-951-16.ca-central-1.compute.amazonaws.com:8443
```

You can define the port number when you launch the EC2 instance when setting the `Security Groups` by adding `Custom TCP Rule` and setting the `Port Range` to the port number.

If you launched an instance in the `sg-4a03c42a` group, then the port number is `8443`.

4. Open `http://[public domain name].ca-central-1.compute.amazonaws.com:[port]` in Chrome.

For example: `http://ec2-3-96-51-16.ca-central-1.compute.amazonaws.com:8443`
