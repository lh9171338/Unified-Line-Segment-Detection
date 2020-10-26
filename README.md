[<img height="23" src="https://github.com/lh9171338/Outline/blob/master/icon.jpg"/>](https://github.com/lh9171338/Outline) Unified Line Segment Detection
===

## Network Architecture
<img width="600" src="https://github.com/lh9171338/Unified-Line-Segment-Detection/blob/main/figure/Network.png"/>

## Quantitative Results
### Pinhole Image Dataset
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



This repository is the spherical Hough transform algorithm implemented in Windows, and you can also access the [Linux version](https://github.com/lh9171338/Spherical-Hough-Transform/tree/Linux).

<img width="300" src="https://github.com/lh9171338/Unified-Line-Segment-Detection/blob/main/figure/wireframe-sAP10.png"/><img width="300" src="https://github.com/lh9171338/Unified-Line-Segment-Detection/blob/main/figure/dwireframe-sAP10.png"/><img width="300" src="https://github.com/lh9171338/Unified-Line-Segment-Detection/blob/main/figure/sun360-sAP10.png"/>

## Qualitative Results
<img width="700" src="https://github.com/lh9171338/Unified-Line-Segment-Detection/blob/main/figure/Qualitative_results.png"/>
