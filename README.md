[<img height="23" src="https://github.com/lh9171338/Outline/blob/master/icon.jpg"/>](https://github.com/lh9171338/Outline) Unified Line Segment Detection
===

## Introduction

## Network Architecture
<img width="700" src="https://github.com/lh9171338/Unified-Line-Segment-Detection/blob/main/figure/Network.png"/>

## Quantitative Results
### Pinhole Image Datasets
<html>
<table align="center">
    <tr>
        <td rowspan="2" align="center">Method</td> 
        <td colspan="5" align="center"><a href="https://github.com/huangkuns/wireframe">Wireframe Dataset</a></td>
        <td colspan="5" align="center"><a href="http://www.elderlab.yorku.ca/resources/york-urban-line-segment-database-information/">YorkUrban Dataset</a></td>
        <td rowspan="2" align="center">FPS</td>     
    </tr>
    <tr>
        <td align="center">sAP<sup>10</sup></td>
        <td align="center">msAP</td>     
        <td align="center">mAP<sup>J</sup></td>    
        <td align="center">AP<sup>H</sup></td>    
        <td align="center">F<sup>H</sup></td>  
        <td align="center">sAP<sup>10</sup></td>
        <td align="center">msAP</td>     
        <td align="center">mAP<sup>J</sup></td>    
        <td align="center">AP<sup>H</sup></td>    
        <td align="center">F<sup>H</sup></td>      
    </tr>  
    <tr>
        <td align="center">LSD</td>
        <td align="center">9.5</td>
        <td align="center">9.3</td>    
        <td align="center">17.2</td>    
        <td align="center">54.3</td>    
        <td align="center">61.5</td>    
        <td align="center">9.4</td>
        <td align="center">9.4</td>    
        <td align="center">15.4</td>    
        <td align="center">49.7</td>    
        <td align="center">60.0</td>          
        <td align="center"><b>50.9</b></td>          
    </tr>    
    <tr>
        <td align="center"><a href="https://github.com/huangkuns/wireframe">DWP</a></td>
        <td align="center">6.8</td> 
        <td align="center">6.6</td>    
        <td align="center">38.6</td>    
        <td align="center">65.9</td>    
        <td align="center">72.2</td>    
        <td align="center">2.7</td> 
        <td align="center">2.7</td>    
        <td align="center">23.4</td>    
        <td align="center">51.6</td>    
        <td align="center">62.3</td>   
        <td align="center">2.3</td>          
    </tr>    
    <tr>
        <td align="center"><a href="https://github.com/cherubicXN/afm_cvpr2019">AFM</a></td>
        <td align="center">24.3</td>
        <td align="center">23.4</td>    
        <td align="center">24.3</td>    
        <td align="center">70.1</td>    
        <td align="center">77.0</td>    
        <td align="center">9.1</td>
        <td align="center">8.9</td>    
        <td align="center">12.5</td>    
        <td align="center">48.5</td>    
        <td align="center">63.2</td>    
        <td align="center">14.3</td>          
    </tr> 
    <tr>
        <td align="center"><a href="https://github.com/zhou13/lcnn">L-CNN</a></td>
        <td align="center">62.9</td>
        <td align="center">62.1</td>    
        <td align="center">59.3</td>    
        <td align="center">80.3</td>    
        <td align="center">76.9</td>    
        <td align="center">26.4</td>
        <td align="center">26.1</td>    
        <td align="center">30.4</td>    
        <td align="center">57.8</td>    
        <td align="center">61.6</td>    
        <td align="center">13.7</td>          
    </tr>    
    <tr>
        <td align="center"><a href="https://github.com/cherubicXN/hawp">HAWP</a></td>
        <td align="center"><b>66.5</b></td>
        <td align="center"><b>65.7</b></td>    
        <td align="center">60.2</td>    
        <td align="center"><b>83.2</b></td>    
        <td align="center"><b>80.2</b></td>    
        <td align="center"><b>28.5</b></td>
        <td align="center"><b>28.1</b></td>    
        <td align="center"><b>31.7</b></td>    
        <td align="center"><b>58.8</b></td>    
        <td align="center"><b>64.8</b></td>   
        <td align="center">30.9</td>          
    </tr>   
    <tr>
        <td align="center">ULSD<sup>1</sup>(ours)</td>
        <td align="center">66.4</td>
        <td align="center">65.6</td>    
        <td align="center"><b>61.4</b></td>    
        <td align="center">81.7</td>    
        <td align="center">79.4</td>    
        <td align="center">27.4</td>
        <td align="center">27.0</td>    
        <td align="center">31.0</td>    
        <td align="center">56.5</td>    
        <td align="center">63.3</td>          
        <td align="center">40.6</td>          
    </tr>   
</table>
</html>

PR curves of sAP<sup>10</sup> and AP<sup>H</sup> on the Wireframe dataset (the left two plots) and YorkUrban dataset (the right two plots).

<img width="230" src="https://github.com/lh9171338/Unified-Line-Segment-Detection/blob/main/figure/wireframe-sAP10.png"/><img width="230" src="https://github.com/lh9171338/Unified-Line-Segment-Detection/blob/main/figure/wireframe-APH.png"/><img width="230" src="https://github.com/lh9171338/Unified-Line-Segment-Detection/blob/main/figure/york-sAP10.png"/><img width="230" src="https://github.com/lh9171338/Unified-Line-Segment-Detection/blob/main/figure/york-APH.png"/>

### Fisheye Image Datasets

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
        <td align="center">LSD</td>
        <td align="center">3.1</td>
        <td align="center">4.3</td>    
        <td align="center">5.3</td>    
        <td align="center">4.3</td>    
        <td align="center">11.1</td>    
        <td align="center">3.8</td>
        <td align="center">5.2</td>    
        <td align="center">6.3</td>    
        <td align="center">5.1</td>    
        <td align="center">10.7</td>          
        <td align="center"><b>47.9</b></td>          
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
        <td align="center">26.4</td>    
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
        <td align="center"><b>56.5</b></td>
        <td align="center"><b>61.2</b></td>    
        <td align="center"><b>63.0</b></td>    
        <td align="center"><b>60.2</b></td>    
        <td align="center"><b>56.3</b></td>    
        <td align="center"><b>26.9</b></td>
        <td align="center"><b>30.2</b></td>    
        <td align="center">31.7</td>    
        <td align="center"><b>29.6</b></td>    
        <td align="center">32.6</td>          
        <td align="center">36.8</td>          
    </tr>   
    <tr>
        <td align="center">ULSD<sup>3</sup>(ours)</td>
        <td align="center">55.5</td>
        <td align="center">60.3</td>    
        <td align="center">62.1</td>    
        <td align="center">59.3</td>    
        <td align="center">56.1</td>    
        <td align="center">25.4</td>
        <td align="center">28.6</td>    
        <td align="center">30.1</td>    
        <td align="center">28.0</td>    
        <td align="center">31.5</td>   
        <td align="center">36.5</td>          
    </tr>  
    <tr>
        <td align="center">ULSD<sup>4</sup>(ours)</td>
        <td align="center">55.2</td>
        <td align="center">59.9</td>    
        <td align="center">61.8</td>    
        <td align="center">59.0</td>    
        <td align="center">56.3</td>    
        <td align="center"><b>26.9</b></td>
        <td align="center">30.1</td>    
        <td align="center"><b>31.8</b></td>    
        <td align="center"><b>29.6</b></td>    
        <td align="center"><b>33.1</b></td>   
        <td align="center">36.3</td>          
    </tr>
</table>
</html>

PR curves of sAP<sup>10</sup> on the F-Wireframe dataset (the left plot) and F-YorkUrban dataset (the right plot).

<img width="400" src="https://github.com/lh9171338/Unified-Line-Segment-Detection/blob/main/figure/dwireframe-sAP10.png"/><img width="400" src="https://github.com/lh9171338/Unified-Line-Segment-Detection/blob/main/figure/dwireframe-sAP10.png"/>

### Spherical Image Dataset

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
        <td align="center">0.9</td>
        <td align="center">1.7</td>    
        <td align="center">2.5</td>    
        <td align="center">1.7</td>    
        <td align="center">3.4</td>           
        <td align="center">0.05</td>          
    </tr>    
    <tr>
        <td align="center"><a href="https://github.com/zhou13/lcnn">L-CNN</a></td>
        <td align="center">39.8</td>
        <td align="center">42.5</td>    
        <td align="center">43.6</td>    
        <td align="center">42.0</td>    
        <td align="center">34.8</td>    
        <td align="center">12.6</td>    
    </tr>    
    <tr>
        <td align="center"><a href="https://github.com/cherubicXN/hawp">HAWP</a></td>
        <td align="center">41.7</td>
        <td align="center">44.7</td>    
        <td align="center">45.8</td>    
        <td align="center">44.1</td>    
        <td align="center">33.1</td>    
        <td align="center"><b>25.4</b></td>          
    </tr>   
    <tr>
        <td align="center">ULSD<sup>2</sup>(ours)</td>
        <td align="center"><b>58.5</b></td>
        <td align="center"><b>64.4</b></td>    
        <td align="center"><b>66.8</b></td>    
        <td align="center"><b>63.2</b></td>    
        <td align="center">46.6</td>          
        <td align="center">24.2</td>          
    </tr>   
    <tr>
        <td align="center">ULSD<sup>3</sup>(ours)</td>
        <td align="center">57.2</td>
        <td align="center">64.0</td>    
        <td align="center">66.1</td>    
        <td align="center">62.4</td>    
        <td align="center"><b>46.8</b></td>     
        <td align="center">23.8</td>          
    </tr>  
    <tr>
        <td align="center">ULSD<sup>4</sup>(ours)</td>
        <td align="center">56.7</td>
        <td align="center">63.1</td>    
        <td align="center">65.6</td>    
        <td align="center">61.8</td>    
        <td align="center">46.7</td>     
        <td align="center">23.3</td>          
    </tr>
</table>
</html>

PR curves of sAP<sup>10</sup> on the SUN360 dataset.

<img width="400" src="https://github.com/lh9171338/Unified-Line-Segment-Detection/blob/main/figure/sun360-sAP10.png"/>

## Qualitative Results
<img src="https://github.com/lh9171338/Unified-Line-Segment-Detection/blob/main/figure/Qualitative_results.png"/>
