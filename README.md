# 2018-SURF-Trajectory-Estimation

2018 SURF of EEE department XJTLU: Trajectory Estimation of Mobile Users/Devices based on Wi-Fi Fingerprinting and Deep Neural Networks



p.s. The algorithm code is preserved untill workshop paper accepted.



## Background

This 2018 SURF(Summer Undergraduate Research Fellowships) project is based on the last year's SURF of [Indoor Localization  Based on Wi-Fi Fingerprinting](http://kyeongsoo.github.io/research/projects/surf-201739/index.html) and will go deeper in two directions, one is **advanced DNN-based indoor localization** (including CNN-based approaches); the other is **RNN-based (LSTM) trajectory estimation**.



## People

###   Supervisor

- [Dr Kyeong Soo Kim](http://kyeongsoo.github.io/) ([Department of Electrical and Electronic Engineering](http://www.xjtlu.edu.cn/en/departments/academic-departments/electrical-and-electronic-engineering/), [XJTLU](http://www.xjtlu.edu.cn/en/) )		
	 [Dr Sanghyuk Lee](http://www.xjtlu.edu.cn/en/departments/academic-departments/electrical-and-electronic-engineering/staff/sanghyuk-lee) ([Department of Electrical and Electronic Engineering](http://www.xjtlu.edu.cn/en/departments/academic-departments/electrical-and-electronic-engineering/), [XJTLU](http://www.xjtlu.edu.cn/en/) )			

###   Research Assistant

- Xintao Huan (E-mail: Xintao.Huan_at_xjtlu.edu.cn; PhD Candidate,  ([Department of Electrical and Electronic Engineering](http://www.xjtlu.edu.cn/en/departments/academic-departments/electrical-and-electronic-engineering/), [XJTLU](http://www.xjtlu.edu.cn/en/) )

## Meetings:

- Please refer to the project homepage, [Trajectory Estimation of Mobile Users/Devices based on Wi-Fi Fingerprinting and Deep Neural Networks](http://kyeongsoo.github.io/research/projects/surf-201830/index.html), with presentation materials.



## Research part (15/6/2018 - 7/9/2018)

### Android mobile phone and coordination system

![**Android mobile phone and coordination system**](https://github.com/ZzhKlaus/2018-SURF-Trajectory-Estimation/raw/master/img/axisdevice.png)

We used two android mobile phones for collecting Wi-Fi fingerprinting as well as IMU data, for IMU, these data were collected: Geomagnetic field intensity in (x,y,z) coordination, pose direction of phones (pitch, roll, yaw), accelaration data in (x,y,z) coordination.

- **x - pitch**
- **y - roll**
- **z - yaw (*Azimuth*)**

### Transfer the geomagnetic filed intensity from device coordination system to global coordination system

We have tried several ways :

1. **Rotation Matrix**
2. **quaternion**

The performance was not as expected and varied in different devices, we made a tradeodd by keeping the devides at same pose when measuring.



### To keep data accurate and stable

While WiFi and IMU data were collected together, for one measurement of each point, the collection times of IMU data would be same as the number of APs of that point detected. (Usually more than 50 times), and **Kalman Filter** was implemented in android APP.



### Data collection 

![](https://github.com/ZzhKlaus/2018-SURF-Trajectory-Estimation/raw/master/img/Location_Of_Data.jpg)

(Drawn by Zhe (Tim) Tang )

We collected two areas in 4th and 5th floors of IBSS, XJTLU. The total area is around 300 $m^2$  . 

### Study on geomagnetic field by different devices on one path 

![](https://github.com/ZzhKlaus/2018-SURF-Trajectory-Estimation/raw/master/img/4Y_3.png)

We found that although the detecting data at one point varies a lot for different devices, the characteristic of changing patterns along one path are very similar.



### Study on geomagnetic field by different poses of one device

![](https://github.com/ZzhKlaus/2018-SURF-Trajectory-Estimation/raw/master/img/ori.PNG)

Although the deteacting data varies when changing the pose of device at one point,  the changing patterns along one path are similar, two. 



### The geomagnetic field map for two floors

![](https://github.com/ZzhKlaus/2018-SURF-Trajectory-Estimation/raw/master/img/floor_4.png)

 **4th floor**

![](https://github.com/ZzhKlaus/2018-SURF-Trajectory-Estimation/raw/master/img/floor_5.png)

 **5th floor** 

### Training data: Random waypoint model

For training data, considering the special characteristic that despite very stable spatially, there are many points with same geomagnetic field intensity, which limits the usage as fingerprint like Wi-Fi signal (where the mac address is unique). We generated a long path of 10,000 steps based on random waypoint model, for both training and evaluation (0.75:0.25).

![](https://github.com/ZzhKlaus/2018-SURF-Trajectory-Estimation/raw/master/img/tra.png)

**random waypoint model**

### Training the LSTM network

![](https://github.com/ZzhKlaus/2018-SURF-Trajectory-Estimation/raw/master/img/model_ts=30_hn=128_bs=5_ep=100.png)

**model structure**

![](https://github.com/ZzhKlaus/2018-SURF-Trajectory-Estimation/raw/master/img/diff_hidden_nodes.png)

Training accuracy with different hidden nodes

![](https://github.com/ZzhKlaus/2018-SURF-Trajectory-Estimation/raw/master/img/diff_batch_size.png)

Training accuracy with different batch size



![](https://github.com/ZzhKlaus/2018-SURF-Trajectory-Estimation/raw/master/img/diff_time_steps.png)

Training accuracy with different time steps

![](https://github.com/ZzhKlaus/2018-SURF-Trajectory-Estimation/raw/master/img/error_LSTM_all.png)



**Evaluation error based on 25,000 points**



## Outcomes	

### Github repository 

- [Meng (Lemon) Wei](https://github.com/weimengmeng1999/CNN_based_indoor_localization)
- [Zhenghang (Klaus) Zhong](https://github.com/ZzhKlaus/2018-SURF-Trajectory-Estimation)

### Publications

- Zhenghang Zhong, Zhe Tang, Xiangxing Li, Tiancheng Yuan, Yang Yang, Wei Meng, Yuanyuan Zhang, Renzhi Sheng, Naomi Grant, Chongfeng Ling, Xintao Huan, Kyeong Soo Kim and Sanghyuk Lee, "XJTLUIndoorLoc: A new fingerprinting database for indoor localization and trajectory estimation based on Wi-Fi RSS and geomagnetic field," *submitted to* [*GCA'18*](https://is-candar.org/GCA18/), Sep. 7, 2018.

## References

### Trajectory estimation and PDR

1. Ho Jun Jang, Jae Min Shin, and Lynn Choi, "Geomagnetic field based indoor localization using recurrent neural networks," *Proc. GLOBECOM 2017*, pp. 1-6, Dec. 2017. ([DOI](https://doi.org/10.1109/GLOCOM.2017.8254556))

2. [DeepML: Deep LSTM for Indoor Localization with Smartphone Magnetic and Light Sensors](http://www.eng.auburn.edu/~szm0001/papers/ICC18_DeepML.pdf)

   

### Rucurrent neural networks (RNNs) and long short-term memory (LSTM) networks 

1. [A Critical Review of Recurrent Neural Networks for Sequence Learning](https://arxiv.org/abs/1506.00019)

2. [An advanced method for pedestrian dead reckoning using BLSTM-RNNs](https://ieeexplore.ieee.org/document/7346954/)

3. [From Recurrent Neural Network to Long Short Term Memory Architecture](https://static.googleusercontent.com/media/research.google.com/zh-CN//pubs/archive/43905.pdf)

4. [LSTM AND EXTENDED DEAD RECKONING AUTOMOBILE ROUTE PREDICTION USING SMARTPHONE SENSORS](https://www.ideals.illinois.edu/handle/2142/97357)

5. [Keras & machinelearningmastery](https://machinelearningmastery.com/)

   

### Fingerprint database combined with IMU data

1. [A multisource and multivariate dataset for indoor localization methods based on WLAN and geo-magnetic field fingerprinting](https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=7743678)

2. [MagPIE: A Dataset for Indoor Positioning with Magnetic Anomalies](https://ieeexplore.ieee.org/document/8115961/)

3. [UJIIndoorLoc-Mag: A New Database for Magnetic Field-Based Localization Problems](https://ieeexplore.ieee.org/document/7346763/)

   
