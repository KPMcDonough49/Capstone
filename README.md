## Infection Detection with Convolutional Neural Network
#### Author: Kevin McDonough 
![Logo](Images/Digital_healthcare_logo.png)

### Overview
This project analyzes over 1200 images of infected wounds, non-infected wounds and healthy skin. Using these images, as well as image generation techniques such as data augmentation, a convolutional neural network was constructed in order to predict the likelihood of a wound being infected. The model was then deployed on a Steamlit app, allowing a user to upload an image at their convenience. 

### Business Problem / Use Case 
The current cost of treating wounds in the U.S. is ~$25 billion per year. Infections represent a large proportion of this, and if an infection is not detected early it can lead to significant economic and human costs. Due to potential barriers to seeing a doctor, such as expensive bills and inability to travel, patients with wounds who are in need of care occasionally put off doctor's appointments. Deploying a image classification app to screen patients would help solve these issues by allowing patients to upload images from the convenience of their home at no cost. The app would then let them know if they need to see a doctor. An additional benefit is that, by screening out patients with a low probability of infection, the app would also save doctors time by allowing them to focus on patients that are truly at risk. Below, I've highlighted some of the advantages of using a CNN for infection detection: 

![Infection](Images/Use_case.png)

### Data Understanding 
The dataset consists of over 1200 images. There are two classes: Infection and No Infection. The two classes are roughly equally represented in the overall dataset -- there are around 610 images in each class. The Infection class is made up of wounds with varying degrees of infection. Around 50% of the No Infection class contains images of cuts, scrapes and scabs and the other 50% is images of healthy skin with no wound. This data was collected by scraping google images. Additionally, several images are from the [Medetec Wound Database](http://www.medetec.co.uk/files/medetec-image-databases.html). Below is an example of an image used for each class. 

**Example of Infection:**

![Infection](Images/Infection_example.png)

**Example of No Infection:**

![Infection](Images/No_infection_example.png)


### Methods
* Data collection using Jupyter, Python and Google Images
* Preprocessing techniques such as data augmentation, rescaling, and performing a train-test split  
* Model building using Keras and Google Colab 
* Model evaluation using seaborn, sklearn and Matplotlib
* Selecting final model and creating predictions for validation data
* Deployment using Heroku and Streamlit. 

The image below gives an overview of the steps taken in the project: 

![Methods](Images/Methods.png)


### Model Creation and Evaluation  

After preprocessing our data, I created an initial CNN model. The model trained on 75% of the images and used the remaining 25% for validation. The initial model only achieved an accuracy of .53 on the validation set. 

#### Best Model 

I realized that more data was necessary, so I used data augmentation to triple the training set to ~2500 images by performing rotations and shifts. Due to computational complexity, I used google Colab for the next model. I also decreased batch size and increased the number of epochs to 50. The chart below shows the training and validation accuracy for the resulting model: 

![Confusion Matrix](Images/best_model_accuracy.png)

Below is the confusion matrix for the epoch that had the best weights 

![Confusion Matrix](Images/confusion_matrix.png)

The validation accuracy has increased from .53 to ~.905. Additionally, the false negative rate (instances in which an infection is classified as not infected) dropped to ~4%. 


Because our Random Forest model and XGBoost model had the highest accuracy scores, we decided to select these to use moving forward. Next, we wanted to see which models did the best job of maximizing functional precision. In this scenario, we do not want our model to predict wells as "functional" when they are not. In this scenario, people would lack access to water because the well isn't working and we aren't doing anything to remedy the situation due to the fact that our model predicted that there was nothing wrong with the well. 

##### Random Forest Confusion Matrix
![RF Confusion](images/random_forest_confusion.png)

##### XGBoost Confusion Matrix
![XGB Confusion](images/XGBoost_confusion.png)

Looking at the output of the confusion matrices for our Random Forest Model and XGBoost model, you can see that the Random Forest model incorrectly predicts "non functional" and "functional needs repair" wells as "functional" at a lower rate than the XGBoost model (the two boxes at the bottom of the left column). Further corraborating this, we looked at the classification reports for each model and noticed that the Random Forest Model had a higher functional precision than the XGBoost model (.81 versus .78). Due to this, we decided to use the Random Forest model. 

After selecting our classification model, we wanted to optimize the model for the metric we deemed most important. As we discussed above, we want to minimize instances in which the model predicts that wells are functonal when they are not. To achieve this, we would like to maximize micro-precision: true "functional" predictions / (true "functional" predictions + false positive "functional" predictions). In order to achieve this, we ran another grid search using different hyperparameters in order to optimize micro-precision. 

![Final Confusion](images/final_model_confusion.png)

The image above shows us the confusion matrix for our final model after optimizing for micro precision. Comparing this to our base models, you can see that the values in the two lower boxes on the left column are lower than they were previously. Additionally, on the classification report, the functional precision improved from .81 to .82 without causing a decrease in accuracy (still .79).

We also charted the most important features of our model to get a better sense of what features led to a well being non functional: 

![Feature Importances](images/feature_importance.png)

Our last step was to take our final model, fit it on our testing data and use it to create predictions. We did this and then mapped the "non functional" predictions on folium so we could figure out which locations to begin fixing wells. 

![Predictions Map](images/predicted_non_functional.png)

In the map above we circled areas in which we noticed high counts of predicted non functional wells. 

### Conclusions 

**1. Start fixing wells that are non-functional and located where the counts are the highest .** We looked at where every non-functional well is located on a map. We could start where the clusters were the highest. 

**2. Recommend replacing non-functional wells with the top installers.** One of the top predictors of the status group of the well was the installer count, which showed the number of installs by each installer company. Consider using a top installer for replacing non-functional wells. 

**3. Consider replacing older wells.** Most functional wells are under 20 years old. Consider looking at the age of the well to determine if it is non-functional.

### Next Steps

Further analyses could provide even more insight into how we can predict the operating condition of wells in Tanzania: 

**More features of the wells** Other features that could indicate a functioning well include usage rate (how often it gets used in a given day/week and month), and proximity to nearest subvillage.

**Better idea of populations of specific regions.** There was a lot of missing data that we had to impute for the population feature. If we could gather actual population estimates through a thorough census, we could then determine which wells to focus on first. Those wells that are non-functioning in a larger populated area would be highest priority.

**Better idea of when the well was contructed.** This was another column of  a lot of missing values. If we could gather more information on the contruction year of the well based on geological surveys, we can determine how much longer a well will last before it becomes non-functional.

**Implement new technology for functionality tracking.** We could implement sensors on all the wells to determine if they are being used. 

### For More Information:
Please review our full analysis in our [Final Notebook](./final_notebook.ipynb), our [Images](./images), our [Presentation](./final_presentation_phase_3.pdf)   
and our [Dashboard Code](./working_notebooks/dashboard_code.py). 

For any additional questions, please contact Ryan Reilly and Kevin McDonough

Ryan: 
Email: ryan.m.reilly@gmail.com
Github: https://github.com/ryanreilly
Linkedin: https://www.linkedin.com/in/ryanreilly1/

Kevin: 
Email: kpmcdonough@gmail.com
Github: https://github.com/KPMcDonough49
Linkedin: https://www.linkedin.com/in/kevin-mcdonough-01466a178/
