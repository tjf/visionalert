# visionalert

Monitor your security cameras for the presence of various objects (people, animals, etc) using Tensorflow

This is very much a work in progress but it works ok at the moment.  For the time being, the only notification options are push messages via a [Gotify](https://gotify.net) server with the images hosted on S3 or other compatible object storage service.  (It does work with [min.io](https://min.io), it's what I actually use myself.)  

**Note:** In order to use this, you'll need the Tensorflow Lite model and respective label map file.  They can be downloaded [from here](https://storage.googleapis.com/download.tensorflow.org/models/tflite/coco_ssd_mobilenet_v1_1.0_quant_2018_06_29.zip).  Configure the location to the .tflite file and labelmap in the configuration file.  The code is also compatible with a Google Coral accelerator if you happen to have one.  Be sure you're using an appropriate model for it, however as the one at the above link will not work for it.

