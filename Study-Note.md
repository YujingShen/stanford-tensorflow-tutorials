# Stanford CS20si

## Slide 1

#### Multi-devices

* ```with tf.device(...):```

* Creates a session with log\_device\_placement set to True.

```python
sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))
```

#### Graph
* better use unique graph
* some possible operations

```python
g = tf.Graph()
with g.as_default():
g = tf.get_default_graph()
```

## Slide 2

#### constant 
* constantare stored in graph definition, not recommond to use

#### initialize
* global_variables_initializer
* variables_initializer
* local_variables_initializer

#### self add / subtract
* assign\_add() assign\_sub()
* each session maintains its own copy of variables


#### control dependence
* control which part should be run firs: ```tf.Graph.control_dependencies```

#### lazy loading
* variables has value after session created

# Slide 4
Example: word2vec.py

#### NCE
* ```tf.nn.nce_loss```
* negative sampling is a simplified version of noise contrastive estimation
* train => nce\_loss; eval => sigmoid\_cross\_entropy\_with\_logits
* labels must be sorted in order of decreasing frequency to achieve good results.  For more details, see tf.nn.log\_uniform\_candidate\_sampler.

#### Embedding
* look up: ```tf.nn.embedding_lookup(...)```

# Slide 5
	
#### Checkpoint
```python
saver = tf.train.Saver()
saver.save(sess, 'checkpoint_dir/model_name', ...)

ckpt = tf.train.get_checkpoint_state('checkpoint_dir')
if ckpt and ckpt.model_checkpoint_path:
     saver.restore(sess, ckpt.model_checkpoint_path)
```
* only save variables, not graph

#### Data Reader VS Feed dict
* Feed dict problem: slow when client and workers are in different machines
* data reader allows us directly load data directly into the worker process

* tf.TextLineReader: Outputs the lines of a file delimited by newlines
	* E.g. text files, CSV files
* tf.FixedLengthRecordReader: Outputs the entire file when all files have same fixed lengths
	* E.g. each MNIST file has 28 x 28 pixels, CIFAR-10 32 x 32 x 3
* tf.WholeFileReader: Outputs the entire file content
* tf.TFRecordReader: Reads samples from TensorFlow’s own binary format (TFRecord)


## Slide 6*

#### Feature Inversion
* x = argmin loss-of-feature + regulation-term

#### Texture Synthesis
1. pretrain CNN
2. run input texture forward through CNN, record activations
3. at each layer compute Gram matrix
4. init generated image from random noise
5. pass generated image through CNN, compute Gram on each layer
6. compute loss
7. backprop to get gradient on image
8. apply gradient on image
9. GO TO 5

#### Style Transfer
* style transfer = feature inversion + texture synthesis
	* match CNN feature of the content image (feature inv)
	* match the Gram matrices to the style image (texture syn)

	1. pretrain CNN
	2. compute feature for content image
	3. compute gram matrices for style image
	4. randomly initialize new image
	5. forward new image through CNN
	6. compute style loss (L2 between Gram Matrix) and content loss (L2 between features)
	7. backprop and apply to image
	8. GO TO 5

#### Improvements

* multiple style => weighted average of Gram Matrices
* preserve color => perform style only on luminance channel (Y in YUV colorspace)
* style + DeepDream => jointly minimize feature reconstruction + style reconstruction and maximize DeepDream feature loss
* on video => run independently result in low consistency
	* init: init t+1 frame with a warped version of stylized result at frame t
	* short-term temporary consistency: forward optical flow should be opposite of backward optical flow
	* long-term temporary consistency: same region should look the same
	* multipass processing: multiple forward and backward passes over teh video with few iterations per pass
* CNNMRF => use patch matching on CNN feature space 
* Fast Style => train a feedforward network(FFNetwork) => learn seperate scale and shift parameters per style