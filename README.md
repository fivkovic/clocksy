
  
<p align="center"><img width=30% src="https://i.ibb.co/ZL1CFbP/clocksy-logo.png"></p>
<p align="center">
⏰ A machine-learning-based analog clock recognition and time detection app developed during the Soft Computing university course.</p>

## Requirements
* [Python 3.7](https://www.python.org/downloads/)

## Environment setup

1. Create a new virtual environment
	```
	python3 -m venv myenv
	```
2.  Use `requirements.txt` to install all required dependencies.
	```
	pip install -r requirements.txt
	```

## Running the application

### Use
#### External stream

1. Download [IP Webcam](https://play.google.com/store/apps/details?id=com.pas.webcam&hl=en) on your phone
2. Connect your PC and phone to the same local network
3. Start application and select *Start Server* option. The application will start capturing video and show you your IP address.
5. Use this IP address to run the `controller.py` script

```
python controller.py -s stream -su http://192.168.1.102:8080/video --width 720 --height 720
```
#### Webcam

```
python controller.py -s webcam --width 640 --height 480
```

### Training
Run the scripts with default parameters.
#### Clock tracking
```
clock_tracking\core.py -m train
```
#### Time reading
```
time_reading\core.py -m train
```
### Testing
Run the scripts with specifying the image count and noise threshold.
#### Clock tracking
```
clock_tracking\core.py -m test -ic 2000 -nt 0.5
```
#### Time reading
```
time_reading\core.py -m test -ic 2000 -nt 0.3
```

## About the Team

| [<img src="https://avatars1.githubusercontent.com/u/17569172?s=88&v=4" width="100px;"/>](https://github.com/fivkovic)<br/> [<sub>Filip Ivković</sub>](https://github.com/fivkovic) |
  [<img src="https://i.ibb.co/VmhxPnd/c38df0e9296d992a3d7e67a0eb7bb86f.png" width="100px;"/>](https://github.com/kettkitt)<br/> [<sub>Katarina Tukelić</sub>](https://github.com/kettkitt) |
 | --- | --- |