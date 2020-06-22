dora\_report
================
Dora Qian
22/06/2020

# Data Sience Techniques

## Evaluation Metrics

Evaluating the quality of machine-generated captions is challenging as
there exist many possible ways to describe an image and there is no one
correct way.

To examine our model performance, we have used 2 types of evaluation
metrics: N-gram based and semantic-based metrics. In total, 9 different
evaluation metrics are used in the evaluation stage.

N-gram based metrics includes Bleu 1-4 \[REF\], Rouge\_L \[REF\], Meteor
\[REF\] and CIDEr \[REF\]. These metrics are commonly used in the
natural language processing community and related research papers. Bleu
score counts the occurrence of n-grams of generated captions in the
reference captions and is precision-based. Similar to the Bleu score,
Rouge\_L is recall-based and calculated as an F-measure using the
longest common subsequences. Meteor is generated by using alignments
between reference and generated captions. CIDEr is the newest one which
is proven to have a more human consensus as it incorporates TF-IDF
weights in the calculation. We used the defence version of CIDEr in our
evaluation script. The main problem with the n-gram based metrics is
that they are sensitive to word overlapping. However, for two captions
to have the same meaning, word overlapping is not necessary. Moreover,
MDA is more interested in the semantic meaning of the caption, therefore
we have includes 2 more metrics.

Semantic-based metrics include Universal Sentence Encoder similarity
(i.e. USC\_Simialrity)\[REF\] and SPICE\[REF\]. USC\_Similarty first
encodes any caption into a matrix using their pre-trained multi-language
model and then computes the inner product of any two captions. While
SPICE parses a caption into a semantic scene graph that lists all the
objects, attributes and relations in the sentences. By using the
dependency graph, the captions with similar semantic meanings will have
a much more reasonable score compared with using n-gram based metrics.

By incorporating both n-gram and semantic-based metrics, we have a more
comprehensive view of model performances.

# Data Product

The final data product is a complete image captioning pipeline ,
consisting of 3 independent modules: a database, a deep learning model
and a visualization tool. When designing our product pipeline, we have
separated the visualization tool from the other two because it can be
run without GPU. The flowchart describing the whole workflow can be
found [here](image%20link). For the main pipeline, we use Make file to
create the whole workflow. The process starts with loading raw data and
preprocessing them, to model generating and evaluating. All the steps
can be executed by using `make all` command in the terminal. We also
allow users to call any specific part of the workflow. For example,
`make data` to prepare the data for training and testing. The
visualization tool workflow is implemented using Django. it can interact
with our database and model in 3 different ways.

## Database

The first module is a database. AWS S3 bucket is chosen to be used as
our database mainly because it can integrate well with AWS GPU instance
that we used for training the model. Other advantage includes great
scalability and ease of use.

In order to use this database, we will provide a private link on google
drive for users to download raw data and upload them to their own S3
bucket as the starting files. After running the whole pipeline, the
users should have the database structure as shown below. They will have
8 folders containing raw, preprocessed image and JSON files, as well as
model results and scores.

## Deep learning model

The second module is a deep learning model. The final model we used in
data product is the baseline model with VGG 16 as pre-trained CNN and
Glove embedding as the pre-trained word embedding. The model is written
in Pytorch and AWS GPU instance is required to train the model. In this
module, We allow users to train the model, generate caption and evaluate
the results. After running the pipeline, all the trained model, model
results and scores will be saved back to the S3 database.

## Visualization tool - Mike’s part