# Pneumonia-detection-using-Multi-GPU-and-Pytorch

Pneumonia has been one of the fatal
diseases & has the potential to result in severe
consequences within a short period of time, due to
the flow of fluid in lungs, which leads to drowning.
If not acted upon by drugs at the right time,
pneumonia may result in death of individuals.
Therefore, the early diagnosis is a key factor along
the progress of the disease. This paper focuses on
the biological progress of pneumonia and its
detection by x-ray imaging, overviews the studies
conducted on enhancing the level of diagnosis, and
presents the methodology and results of parallel
computing with deep learning based on various
parameters in order to detect the disease. In this
study we propose our deep learning architecture
for the classification task, which is trained with
x-ray images with and without pneumonia.We have
implemented two models, VGG16 and
DenseNet121. We achieved better accuracy with
DenseNet121. Our findings yield an accuracy of
87.10 % using a single GPU without data
parallelism and an accuracy of 89.40 % using 4
GPUs with data parallelism. Also, the time
required to train the model using 4 GPUs is a lot
faster than training on a single GPU.
