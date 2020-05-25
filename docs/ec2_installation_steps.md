
The `mds-capstone-mda-gpu` AMI is prepared with the following commands:

Launch an EC2 instance with the `Deep Learning Base AMI (Ubuntu 16.04)` AMI

Run the following command in your local terminal to connect with the EC2 instance

```
ssh -i [key.pem] -L 8888:localhost:8888 ubuntu@[public domain name].ca-central-1.compute.amazonaws.com
```

Prepare the EC2 instance:

```
# update ubuntu
sudo apt-get update
sudo apt-get upgrade

# install pip
sudo apt install python3-pip

# install miniconda instead of Anaconda to save space and time
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh
export PATH="/home/ubuntu/miniconda3/bin:$PATH"


# install
conda install -c conda-forge jupyterlab
# set notebook password to 123
jupyter notebook password

# set up AWS CLI
curl "https://awscli.amazonaws.com/awscli-exe-linux-x86_64.zip" -o "awscliv2.zip"
unzip awscliv2.zip
sudo ./aws/install
aws configure

# set up aws configure
AWS Access Key ID [None]: AKIATB63UHM3KTWNDPFG
AWS Secret Access Key [None]: 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
Default region name [None]: ca-central-1
Default output format [None]: json

# download data
aws s3 sync s3://mds-capstone-mda s3

# update git
sudo add-apt-repository ppa:git-core/ppa 
sudo apt update
sudo apt install git

# handle trash
sudo apt install trash-cli
trash-empty

# install extra packages
pip install matplotlib
pip install nltk
pip install sklearn
pip install wordcloud
pip install altair
pip install gensim
conda install -c conda-forge pyldavis
conda install pytorch torchvision cudatoolkit -c pytorch
pip install torchsummary
pip install ipywidgets
jupyter nbextension enable --py --sys-prefix widgetsnbextension
```