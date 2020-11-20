# RQNet
A lightweight DNN network based on cuDNN
## Status
Still under development, Anyone who is interested can contact me via email r@uamgno.cn or wechat *umango_ross*

## Dev Enviroment
Windows 10 + Visual Studio 2019 + cuda11.1 + GTX 1060 

## Usage

### Command Line : 

     RQNet train|test|detect|demo|wconv [options]

### Options

  To train a network:

       RQNet train -d <path/to/data/defintions> -n <path/to/network/defintion> [-w <path/to/weights>]

 weights file is .pb file. If weights file is not given, then a random set of weighs are initialized.


  To test a network:

       RQNet test  -d <path/to/data/defintions> -n <path/to/network/defintion> -w <path/to/weights>


  To detect objects in image:

       RQNet detect -n <path/to/network/defintion> -w <path/to/weights> -i <path/to/image>


  To detect objects in video:

       RQNet demo -n <path/to/network/defintion> -w <path/to/weights> [-i <path/to/vedio>]
 If input file is not given, then use a camera.


  To convert .weights file to .pb files:

       RQNet wconv -c <path/to/darknet/network/config> -i <path/to/darknet/weights> [-o <path/to/output>]
       
  To convert RQNet model to openvino model, if "-d" option is not given, FP32 is used.
  
       RQNet openvino -n <path/to/network/defintion> -w <path/to/weights> [-o <dir/to/output>] [-d FP16|FP32] [-t model_name]


### ATTENSION 

 This program is running only with CUDA support!
 
