# visionalert

Visionalert is a python application that monitors RTSP streams for objects and sends a push notification to your Android device when something is detected.  Upon detecting a configured object, a notification is scheduled to be sent in 5 seconds containing the image with the highest confidence score detected.

## Requirements
 * Tensorflow Lite SSD MobileNet model and label file [from here](https://storage.googleapis.com/download.tensorflow.org/models/tflite/coco_ssd_mobilenet_v1_1.0_quant_2018_06_29.zip)
 * [Gotify](https://gotify.net) instance and Android application for push notifications
 * [MinIO](https://min.io) for image hosting (This should work with Amazon S3 as well)
 
## Installation

The easiest way to get this running is by using Docker.  

Download the TensorFlow model from the link above, unzip it into a directory and create a configuration file named `config.yml` like the following: 

1) Download the Tensorflow model and unzip it into your current directory.  This also supports the [Google Coral EdgeTPU Accelerator](https://coral.ai) but you need to use a compatible SSD model with it.  Note: Only configure the model, use the labelmap from the above link.  This is due to a different format for the labelmap that comes with the SSD MobileNet for the EdgeTPU.  
    ``` 
    $ wget -q https://storage.googleapis.com/download.tensorflow.org/models/tflite/coco_ssd_mobilenet_v1_1.0_quant_2018_06_29.zip
    
    $ unzip coco_ssd_mobilenet_v1_1.0_quant_2018_06_29.zip 
    Archive:  coco_ssd_mobilenet_v1_1.0_quant_2018_06_29.zip
      inflating: detect.tflite           
      inflating: labelmap.txt            
    ```

2) Create a configuration file named `config.yml` in your current directory.  There is a sample configuration file [located here](https://github.com/tjf/visionalert/blob/master/config-sample.yml)

3) Start the visionalert service by executing the following command.  Note, this will launch the container into the background.
    ``` 
    docker run -d -v ${PWD}:/conf --name visionalert tylerfrederick/visionalert:latest
    ```
4) You can monitor the behavior of the application or view the logs by running `docker logs visionalert`

## Roadmap

* Object area boundaries. Suppress alerts if a detected object is too large or small to help prevent false alerts.
* Other notification strategies.  Considering MQTT so I can use it to trigger Node-Red flows.

## Contributing

Pull requests are welcome!