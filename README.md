Yaksha
======

Make your image archives searchable by tagging with [ImageNet](http://image-net.org/challenges/LSVRC/2012/browse-synsets) classes. This uses a __deep learning__ model trained on ILSVRC2012 using TensorFlow.

##Setup
Install tensorflow, pyexif and the trained model.

    pip install tensorflow
    pip install pyexif
    mkdir imagenet
    wget "http://download.tensorflow.org/models/image/imagenet/inception-2015-12-05.tgz"
    tar -xzvf inception-2015-12-05.tgz -C imagenet
    
##Configure and run
Set the image root directory in `config.py`

    IMAGES_ROOT = "/home/alse/Pictures"
    
Run Yaksha

    python yaksha.py


