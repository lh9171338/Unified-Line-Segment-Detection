[<img height="23" src="https://github.com/lh9171338/Outline/blob/master/icon.jpg"/>](https://github.com/lh9171338/Outline) Unified Line Segment Detection
===
This repository contains the official PyTorch implementation of the paper: [ULSD: Unified Line Segment Detection across Pinhole, Fisheye, and Spherical Cameras](https://arxiv.org/abs/2011.03174).

## Introduction
[ULSD](https://arxiv.org/abs/2011.03174) is a unified line segment detection method for both distorted and undistorted images from pinhole, fisheye or spherical cameras. With a novel line segment representation based on the Bezier curve, our method can detect arbitrarily distorted line segments. Experimental results on the pinhole, fisheye, and spherical image datasets validate the superiority of the proposed ULSD to the SOTA methods both in accuracy and efficiency.

## Network Architecture
<p align="center"><img width="600" src="https://github.com/lh9171338/Unified-Line-Segment-Detection/blob/main/figure/Network.png"/></p>
    
## Results
### Pinhole Image Datasets

#### Quantitative Comparisons
<html>
<table align="center">
    <tr>
        <td rowspan="2" align="center">Method</td> 
        <td colspan="7" align="center"><a href="https://github.com/huangkuns/wireframe">Wireframe Dataset</a></td>
        <td colspan="7" align="center"><a href="http://www.elderlab.yorku.ca/resources/york-urban-line-segment-database-information/">YorkUrban Dataset</a></td>
        <td rowspan="2" align="center">FPS</td>     
    </tr>
    <tr>
        <td align="center">sAP<sup>5</sup></td>
        <td align="center">sAP<sup>10</sup></td>
        <td align="center">sAP<sup>15</sup></td>
        <td align="center">msAP</td>     
        <td align="center">mAP<sup>J</sup></td>    
        <td align="center">AP<sup>H</sup></td>    
        <td align="center">F<sup>H</sup></td>  
        <td align="center">sAP<sup>5</sup></td>
        <td align="center">sAP<sup>10</sup></td>
        <td align="center">sAP<sup>15</sup></td>
        <td align="center">msAP</td>     
        <td align="center">mAP<sup>J</sup></td>    
        <td align="center">AP<sup>H</sup></td>    
        <td align="center">F<sup>H</sup></td>      
    </tr>  
    <tr>
        <td align="center">LSD</td>
		<td align="center">8.3</td>
		<td align="center">10.8</td>
		<td align="center">12.7</td>
		<td align="center">10.6</td>
		<td align="center">17.2</td>
		<td align="center">54.3</td>
		<td align="center">61.5</td>
		<td align="center">8.5</td>
		<td align="center">10.6</td>
		<td align="center">12.2</td>
		<td align="center">10.4</td>
		<td align="center">15.4</td>
		<td align="center">49.7</td>
		<td align="center">60.0</td>
		<td align="center"><b>50.9</b></td>       
    </tr>    
    <tr>
        <td align="center"><a href="https://github.com/huangkuns/wireframe">DWP</a></td>
		<td align="center">5.8</td>
		<td align="center">7.6</td>
		<td align="center">8.8</td>
		<td align="center">7.4</td>
		<td align="center">38.6</td>
		<td align="center">65.9</td>
		<td align="center">72.2</td>
		<td align="center">2.3</td>
		<td align="center">3.2</td>
		<td align="center">4.1</td>
		<td align="center">3.2</td>
		<td align="center">23.4</td>
		<td align="center">51.6</td>
		<td align="center">62.3</td>
		<td align="center">2.3</td>     
    </tr>    
    <tr>
        <td align="center"><a href="https://github.com/cherubicXN/afm_cvpr2019">AFM</a></td>
		<td align="center">21.2</td>
		<td align="center">26.8</td>
		<td align="center">30.2</td>
		<td align="center">26.1</td>
		<td align="center">24.3</td>
		<td align="center">70.1</td>
		<td align="center">77.0</td>
		<td align="center">8.0</td>
		<td align="center">10.3</td>
		<td align="center">12.1</td>
		<td align="center">10.1</td>
		<td align="center">12.5</td>
		<td align="center">48.5</td>
		<td align="center">63.2</td>
		<td align="center">14.3</td>        
    </tr> 
    <tr>
        <td align="center"><a href="https://github.com/zhou13/lcnn">L-CNN</a></td>
		<td align="center">60.7</td>
		<td align="center">64.1</td>
		<td align="center">65.6</td>
		<td align="center">63.5</td>
		<td align="center">59.3</td>
		<td align="center">80.3</td>
		<td align="center">76.9</td>
		<td align="center">25.3</td>
		<td align="center">27.2</td>
		<td align="center">28.5</td>
		<td align="center">27.0</td>
		<td align="center">30.3</td>
		<td align="center">57.8</td>
		<td align="center">61.6</td>
		<td align="center">13.7</td>       
    </tr>    
    <tr>
        <td align="center"><a href="https://github.com/cherubicXN/hawp">HAWP</a></td>
		<td align="center">64.5</td>
		<td align="center">67.7</td>
		<td align="center">69.2</td>
		<td align="center">67.1</td>
		<td align="center">60.2</td>
		<td align="center"><b>83.2</b></td>
		<td align="center">80.2</td>
		<td align="center">27.3</td>
		<td align="center">29.5</td>
		<td align="center">30.8</td>
		<td align="center">29.2</td>
		<td align="center">31.7</td>
		<td align="center"><b>58.8</b></td>
		<td align="center"><b>64.8</b></td>
		<td align="center">30.9</td>        
    </tr>   
	<tr>
		<td align="center">ULSD<sup>1</sup>(ours)</td>
		<td align="center"><b>65.3</b></td>
		<td align="center">69.0</td>
		<td align="center">70.6</td>
		<td align="center">68.3</td>
		<td align="center"><b>61.6</b></td>
		<td align="center">82.3</td>
		<td align="center"><b>80.6</b></td>
		<td align="center">26.6</td>
		<td align="center">29.2</td>
		<td align="center">30.9</td>
		<td align="center">28.9</td>
		<td align="center">31.3</td>
		<td align="center">56.6</td>
		<td align="center">63.4</td>
		<td align="center">38.3</td>
	</tr>
	<tr>
		<td align="center">ULSD<sup>2</sup>(ours)</td>
		<td align="center"><b>65.3</b></td>
		<td align="center"><b>69.2</b></td>
		<td align="center"><b>70.9</b></td>
		<td align="center"><b>68.5</b></td>
		<td align="center">61.5</td>
		<td align="center">82.5</td>
		<td align="center">80.4</td>
		<td align="center">27.3</td>
		<td align="center">30.2</td>
		<td align="center">32.0</td>
		<td align="center">29.8</td>
		<td align="center"><b>32.3</b></td>
		<td align="center">56.6</td>
		<td align="center">63.6</td>
		<td align="center">37.2</td>
	</tr>
	<tr>
		<td align="center">ULSD<sup>3</sup>(ours)</td>
		<td align="center">65.0</td>
		<td align="center">68.9</td>
		<td align="center">70.5</td>
		<td align="center">68.1</td>
		<td align="center"><b>61.6</b></td>
		<td align="center">82.2</td>
		<td align="center">80.1</td>
		<td align="center">26.1</td>
		<td align="center">28.6</td>
		<td align="center">30.4</td>
		<td align="center">28.4</td>
		<td align="center">31.0</td>
		<td align="center">56.1</td>
		<td align="center">63.3</td>
		<td align="center">37.6</td>
	</tr>
	<tr>
                <td align="center">ULSD<sup>4</sup>(ours)</td>
		<td align="center"><b>65.3</b></td>
		<td align="center"><b>69.2</b></td>
		<td align="center"><b>70.9</b></td>
		<td align="center"><b>68.5</b></td>
		<td align="center">61.4</td>
		<td align="center">82.2</td>
		<td align="center">80.4</td>
		<td align="center"><b>27.7</b></td>
		<td align="center"><b>30.4</b></td>
		<td align="center"><b>32.0</b></td>
		<td align="center"><b>30.0</b></td>
		<td align="center">31.5</td>
		<td align="center">56.9</td>
		<td align="center">63.8</td>
		<td align="center">37.2</td>
	</tr> 
</table>
</html>

#### Qualitative Comparisons

<p align="center">
    <img src="https://github.com/lh9171338/Unified-Line-Segment-Detection/blob/main/figure/pinhole_result.png"/>
</p> 

#### More Results of ULSD

<p align="center">
    <img src="https://github.com/lh9171338/Unified-Line-Segment-Detection/blob/main/figure/pinholes.png"/>
</p> 

### Fisheye Image Datasets

#### Quantitative Comparisons

<html>
<table align="center">
    <tr>
        <td rowspan="2" align="center">Method</td> 
        <td colspan="5" align="center">F-Wireframe Dataset</td>
        <td colspan="5" align="center">F-YorkUrban Dataset</td>
        <td rowspan="2" align="center">FPS</td>     
    </tr>
    <tr>
        <td align="center">sAP<sup>5</sup></td>
        <td align="center">sAP<sup>10</sup></td>
        <td align="center">sAP<sup>15</sup></td>
        <td align="center">msAP</td>     
        <td align="center">mAP<sup>J</sup></td>    
        <td align="center">sAP<sup>5</sup></td>
        <td align="center">sAP<sup>10</sup></td>
        <td align="center">sAP<sup>15</sup></td>
        <td align="center">msAP</td>     
        <td align="center">mAP<sup>J</sup></td>      
    </tr>  
    <tr>
        <td align="center">SHT</td>
		<td align="center">0.4</td>
		<td align="center">0.9</td>
		<td align="center">1.3</td>
		<td align="center">0.8</td>
		<td align="center">2.2</td>
		<td align="center">0.4</td>
		<td align="center">0.8</td>
		<td align="center">1.1</td>
		<td align="center">0.8</td>
		<td align="center">2.3</td>
		<td align="center">0.3</td>     
    </tr>    
    <tr>
        <td align="center"><a href="https://github.com/zhou13/lcnn">L-CNN</a></td>
		<td align="center">40.0</td>
		<td align="center">43.4</td>
		<td align="center">45.2</td>
		<td align="center">42.9</td>
		<td align="center">44.2</td>
		<td align="center">18.2</td>
		<td align="center">19.9</td>
		<td align="center">20.8</td>
		<td align="center">19.6</td>
		<td align="center">26.5</td>
		<td align="center">14.3</td>      
    </tr>    
    <tr>
        <td align="center"><a href="https://github.com/cherubicXN/hawp">HAWP</a></td>
		<td align="center">42.5</td>
		<td align="center">46.3</td>
		<td align="center">48.0</td>
		<td align="center">45.6</td>
		<td align="center">43.8</td>
		<td align="center">19.5</td>
		<td align="center">21.5</td>
		<td align="center">22.5</td>
		<td align="center">21.2</td>
		<td align="center">26.4</td>
		<td align="center">31.5</td>        
    </tr>   
    <tr>
        <td align="center">ULSD<sup>2</sup>(ours)</td>
		<td align="center">59.4</td>
		<td align="center">64.3</td>
		<td align="center">66.3</td>
		<td align="center">63.3</td>
		<td align="center"><b>56.4</b></td>
		<td align="center">27.7</td>
		<td align="center">30.7</td>
		<td align="center">32.4</td>
		<td align="center">30.3</td>
		<td align="center">33.6</td>
		<td align="center"><b>37.2</b></td>        
    </tr>   
    <tr>
        <td align="center">ULSD<sup>3</sup>(ours)</td>
		<td align="center"><b>59.7</b></td>
		<td align="center"><b>64.7</b></td>
		<td align="center"><b>66.7</b></td>
		<td align="center"><b>63.7</b></td>
		<td align="center">56.0</td>
		<td align="center">27.1</td>
		<td align="center">30.2</td>
		<td align="center">32.0</td>
		<td align="center">29.8</td>
		<td align="center">32.9</td>
		<td align="center">37.1</td>       
    </tr>  
    <tr>
        <td align="center">ULSD<sup>4</sup>(ours)</td>
		<td align="center">59.4</td>
		<td align="center">64.3</td>
		<td align="center">66.3</td>
		<td align="center">63.3</td>
		<td align="center">56.1</td>
		<td align="center"><b>28.8</b></td>
		<td align="center"><b>32.0</b></td>
		<td align="center"><b>33.8</b></td>
		<td align="center"><b>31.5</b></td>
		<td align="center"><b>33.9</b></td>
		<td align="center">36.9</td>      
    </tr>
</table>
</html>

#### Qualitative Comparisons

<p align="center">
    <img src="https://github.com/lh9171338/Unified-Line-Segment-Detection/blob/main/figure/fisheye_result.png"/>
</p> 

#### More Results of ULSD

<p align="center">
    <img src="https://github.com/lh9171338/Unified-Line-Segment-Detection/blob/main/figure/fisheyes.png"/>
</p> 

### Spherical Image Dataset

#### Quantitative Comparisons

<html>
<table align="center">
    <tr>
        <td rowspan="2" align="center">Method</td> 
        <td colspan="5" align="center"><a href="https://drive.google.com/drive/folders/1ooaYwvNuFd-iEEcmOQHpLunJEmo7b4NM">SUN360 Dataset</a></td>
        <td rowspan="2" align="center">FPS</td>     
    </tr>
    <tr>
        <td align="center">sAP<sup>5</sup></td>
        <td align="center">sAP<sup>10</sup></td>
        <td align="center">sAP<sup>15</sup></td>
        <td align="center">msAP</td>     
        <td align="center">mAP<sup>J</sup></td>      
    </tr>  
    <tr>
        <td align="center">SHT</td>
		<td align="center">0.8</td>
		<td align="center">1.6</td>
		<td align="center">2.3</td>
		<td align="center">1.5</td>
		<td align="center">4.0</td>
		<td align="center">0.2</td>        
    </tr>    
    <tr>
        <td align="center"><a href="https://github.com/zhou13/lcnn">L-CNN</a></td>
		<td align="center">39.8</td>
		<td align="center">42.4</td>
		<td align="center">43.6</td>
		<td align="center">41.9</td>
		<td align="center">41.2</td>
		<td align="center">13.4</td>  
    </tr>    
    <tr>
        <td align="center"><a href="https://github.com/cherubicXN/hawp">HAWP</a></td>
		<td align="center">41.6</td>
		<td align="center">44.7</td>
		<td align="center">45.8</td>
		<td align="center">44.0</td>
		<td align="center">39.2</td> 
		<td align="center"><b>25.4</b></td>      
    </tr>   
    <tr>
        <td align="center">ULSD<sup>2</sup>(ours)</td>
		<td align="center">63.1</td>
		<td align="center">69.4</td>
		<td align="center">71.5</td>
		<td align="center">68.0</td>
		<td align="center">57.8</td>
		<td align="center">23.9</td>        
    </tr>   
    <tr>
        <td align="center">ULSD<sup>3</sup>(ours)</td>
		<td align="center">61.9</td>
		<td align="center">69.1</td>
		<td align="center">71.1</td>
		<td align="center">67.3</td>
		<td align="center">57.2</td>
		<td align="center">23.7</td>      
    </tr>  
    <tr>
        <td align="center">ULSD<sup>4</sup>(ours)</td>
		<td align="center"><b>63.8</b></td>
		<td align="center"><b>70.1</b></td>
		<td align="center"><b>71.8</b></td>
		<td align="center"><b>68.6</b></td>
		<td align="center"><b>57.9</b></td>
		<td align="center">23.8</td>        
    </tr>
</table>
</html>

#### Qualitative Comparisons

<p align="center">
    <img src="https://github.com/lh9171338/Unified-Line-Segment-Detection/blob/main/figure/spherical_result.png"/>
</p> 

#### More Results of ULSD

<p align="center">
    <img src="https://github.com/lh9171338/Unified-Line-Segment-Detection/blob/main/figure/sphericals.png"/>
</p> 

## Video

<p align="center">
    <a href="https://youtu.be/9h79zK2H8OI">
        <img src="https://github.com/lh9171338/Unified-Line-Segment-Detection/blob/main/figure/video.png"/>
    </a>
</p> 

## Requirements

* python3
* pytorch==1.6.0
* CUDA==10.1
* opencv, numpy, scipy, matplotlib, argparse, yacs, tqdm, json, multiprocessing, sklearn, tensorboardX

## Step-by-step installation
```
conda create --name ulsd python=3.7
conda activate ulsd

cd <ulsd-path>
git clone https://github.com/lh9171338/Unified-Line-Segment-Detection.git
cd Unified-Line-Segment-Detection

pip install -r requirements.txt
conda install pytorch==1.6.0 torchvision==0.7.0 cudatoolkit=10.1 -c pytorch
```

## Quickstart with the pretrained model
* There are 3 pretrained models (**pinhole.pkl**, **fisheye.pkl**, and **spherical.pkl**) in [Google drive](https://drive.google.com/drive/folders/1QyNjfLKoKqX8smi3e922Z8PEeBZid_St)
. Please download them and put in the **model/** folder.  
* There are some testing images in **dataset/** folder. 

```
python test.py --config_file pinhole.yaml --dataset_name pinhole --save_image
```
* The result is saved in **output/** folder.

## Training & Testing

### Data Preparation

* Download the json-format dataset<!-- from [Google Drive]()-->.
* Convert the dataset from json-format to npz-format.
```
cd dataset/
python json2npz.py --config_file fisheye.yaml --dataset_name fwireframe --order 2
```
* Generate the ground truth for evaluation.
```
cd dataset/
python json2npz_gt.py --config_file fisheye.yaml --dataset_name fwireframe
```

### Train

```
python train.py --config_file fisheye.yaml --dataset_name fwireframe --order 2 [--gpu 0]
```

### Test

```
python test.py --config_file fisheye.yaml --dataset_name fwireframe --order 2 --model_name best.pkl [--gpu 0] [--save_image] [--evaluate]
```

### Evaluation

* Evaluate mAP<sup>J</sup>, sAP, and FPS
```
python test.py --config_file pinhole.yaml --dataset_name wireframe --evaluate
```
* Evaluate AP<sup>H</sup>
```
cd metric/
python eval_APH.py --config_file pinhole.yaml --dataset_name wireframe
```

## Citation
```
@misc{li2020ulsd,
      title={ULSD: Unified Line Segment Detection across Pinhole, Fisheye, and Spherical Cameras}, 
      author={Hao Li and Huai Yu and Wen Yang and Lei Yu and Sebastian Scherer},
      year={2020},
      eprint={2011.03174},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
```

