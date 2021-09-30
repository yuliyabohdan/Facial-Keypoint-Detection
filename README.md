# Facial-Keypoint-Detection
Facial Keypoint Detection with PyTorch

##The project structure:
config.py
dataset.py
model.py
utils.py
face_keypoints.ipynb - the main file to train the model.

### Model
A pre-trained ResNet50 was used as a baseline. The increase the quality the are several additional opportunities:
  1. training 100+ epoches
  2. augumentation
  3. initiation the last layer with zero(so 'bad grad' will not come from untrained last layer to already trained first layers)
  4. update_bn_stats (https://github.com/facebookresearch/fvcore/blob/master/fvcore/nn/precise_bn.py#L88)

### Results
test_keypoints.csv with the keypoints for the test data.
