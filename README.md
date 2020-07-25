# Aelous
### Problem Statement : 
Climate change has the potential to impact every human, every industry and every living organism on the planet. It sounds extreme because it is. Exhaustive research has confirmed changing weather patterns, rapidly rising sea levels, and extreme weather events proliferating around the world.
### How can Technology help ? 
Today's turbines have sensors and precision controllers, which constantly tweak the blade position to optimize the use of the wind energy and provide information to wind farm operators. Intelligent controllers expose more of the blade to capture the most wind. If gusts grow dangerously strong, the blades can also be rotated to minimize their exposure and the risk of damage.
### How does Aelous serves the cause ?
Improvements in weather forecasting are also increasing the output from wind farms. Accurate wind forecasts can increase the power dispatch by 10 percent by having a better grip on the wind's intermittent nature.
### The Idea
Our vision is to moderate this inexhaustible wellspring of energy and thus, save the earth and construct a sustainable future.
We predict the accessibility of wind vitality assets and encourages the improvement in wind energy production. 
<p  align="center"><img  src = "https://github.com/Apurva-tech/Aelouss/blob/master/templates/intro.gif"></p>
<p  align="center">
<mark>By this application we will predict a few qualities which are fundamental parts of bridling wind energy to its extreme</mark>
<p  align="center">We predict certain aspects like no of Turbines Generate Enegry,Wind Speed,Air Density,Different radius of the Turbines. To predict the optimum energy extractable from that particular location. </p><br>
</p>

### The architecture 

![Video transcription/translation app](https://github.com/Apurva-tech/Aelouss/blob/master/templates/Untitled%20Diagram-4.png)

1. The user navigates to the site and we access their location.
2. We used a LSTM model built using Tensorflow API.
3. Box Map API detects the location and extracts data.
4. Model predicts Wind Energy, Wind Speed and Air Density.

### To visit the website  https://aelous-map.eu-gb.mybluemix.net/Land.html


### The steps required to run it on local system

<mark>Follow the steps</mark>
  <p>Then open your terminal and create new folder<br>
  
  ```mkdir (Name of the folder)```<br>
  
  after making the directory get into the directory by using the following command<br>
  
  ```cd (Name of the folder)```
  </p>
  

<p>
  <mark>First create a virtual environment where it doesn't merge with the existing environment</mark>
<p>
  
  ```virtualenv --python=python3 app```

</p><br>

<p>Enter into the the Virtual Environment and then</p><br>

<mark>Clone the github repository into the system</mark><br>
  <p>
    
   ```git clone https://github.com/Speedster-95/Aelouss.git```
    
</p><br>

<mark>use the following command to go into the directory </mark>
<p>
  
  ```cd Aelouss```
  
 </p><br>
 
<mark> First install all the prerequisite by the following command</mark>
<p>
  
  ```pip install -r requirements.txt```

</p><br>

<mark> Once everything is done. Now we can run it on the local by using this command</mark>

<p>
  
  ```python server.py```
  
</p><br>  


The contributors : <br>
[Bhanu Prakash](https://www.linkedin.com/in/chittampalli-bhanu-prakash-72a7071b1/)<br>
[Apurva Sharma](http://linkedin.com/in/apurva-sharma-46a091190)<br>
[Shifali Agrahari](http://127.0.0.1:5000/www.linkedin.com/in/shifali-agrahari-5b1495196)<br>
[Anamika Gupta](http://127.0.0.1:5000/linkedin.com/in/anamika-gupta-a69a27183)<br>
