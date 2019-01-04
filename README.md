# IRES Research: Architectural Support for Machine Learning, Summer 2018, Pázmány Péter Catholic University, Budapest, Hungary

I spent most of the summer of 2018 looking into how to improve performance in a [Matching Network](https://arxiv.org/abs/1606.04080) through the use of Locality Sensitive Hasing in order to decrease the memory used and time spent performing an operation.

## Matching Networks
A matching network is a model that uses a support set of embeddings of samples from various classes to compare against the embedding of a query to determine which class the query belongs to.  This involves comparing the query against most, if not all, of the samples in the support set.

### Embeddings
There are various ways to make embeddings for raw data, but this was done by generating a neural network that was missing the last Fully Connected classifiation layer.  This vector was taken as an embedding and was then used to determine a distance, and the distance to each sample was fed into a softmax function against an expected ratio of what should be highest, and was used as the loss function to train this function to create the embeddings.

### Comparisons
These embeddings need to be compared somehow to determine which sample is closest to the query. Based on the feature vector generated, a Cosine Distance measurement was made to compare the two against one another.  The higher the value, the closer these two were, helping us determine which is the nearest neighbor.

## Locality Sensitive Hashing
The embeddings mentioned ideally have some sense of locality in the space, so we need to make sure that we can mantain that aspect.  So a normal hash is not necessarily acceptable.  The LSH uses a set of planes which transform the embedding into a sequence of ones and zeros via a dot product calculation of the embedding against the planes. If the result is non-negative, the result for that plane is a 1, and if it is negative, it is a 0.  For two hashes, we use an XNOR calculation.  Based on the number of bits that are the same, we can determine how close they are. The higher the bits, the closer the embeddings are to one another.

## Network Generation
All the programs used to generate the networks are in `[data-source-name]_matching_network_[distance_metric].py` and can be configured to use batch normalization dropout, and how many classes and supports to use per run.  You can also specify the end location of the model.

## Testing Inference Accuracy
To test a networks accuracy you can use the program `[data-source-name]_lsh_testing_suite.py`.  Specify the model to use by using the `-l` flag.  You can specify and output file location by using the `-f` flag.  Use the `-t` flag to save tensorflow statistics.

## Results Summary
Looking in the data folder, you can find results for how the accuracy increases as the number of planes increases.  Usually this was done on models with 10000 iterations, 3 convolutions, batch normalization and dropout with a keep rate of 0.8.

In the metadata section you can find Tensorboard logs where it can be found that under certain number of planes, LSH can be more efficient on a GPU in terms of both memory and time.

## Preliminary Extra Explorations

### LSH Specialized Planes
This is an attempt to achieve the same LSH accuracy through less planes. One method was to use a one-vs-rest method of each class against another, so that each class would have a defined sequence that is only as long as the number of classes.  Any of these attempts are under the name `one-rest`.

### Using LSH to Train Matching Network Embedding Functions
The networks that can be used to generate embedding functions that are trained with LSH to improve performance can be found with the suffix `...matching_network_lsh_*.py`.
