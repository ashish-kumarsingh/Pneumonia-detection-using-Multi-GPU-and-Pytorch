Kaggle Dataset: https://www.kaggle.com/paultimothymooney/chest-xray-pneumonia

Process to run files:

1. Make sure Pytorch and Cuda is installed on your machine. 
2. Load images in data folder which has to be predicted 
3. Add your images command in predict_dense or predict_vgg python file
	print(predict('data/normal4.jpeg'))

4. run predict_dense file (python predict_dense.py) for densenet121 model
  OR run predict_vgg file (python predict_vgg.py) for vgg16 model

Whole project can be found on github:
https://github.com/ashish-kumarsingh/Pneumonia-detection-using-Multi-GPU-and-Pytorch