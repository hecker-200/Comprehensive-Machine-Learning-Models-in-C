## Machine Learning in C


## Overview


This project implements several machine learning models in C, including Data Scaling, Linear Regression, Logistic Regression, and K Nearest Neighbour. It provides a command-line interface for users to choose a model and perform tasks such as training and testing using their own data files.


## Installation


1.)GNUPLOT INSTALLATION
* For Ubuntu 
```
sudo apt-get update
sudo apt install gnuplot
```


*  For MacOs
```
brew install gnuplot
```


* For Windows
 You can download the installer from the official Gnuplot website: https://sourceforge.net/projects/gnuplot/files/gnuplot/

to use gnuplot in your code replace the location of gnuplot in model_functions.c line 130 with the loaction where gnuplot in installed locally.

         






2.) CODE INSTALLATION                 


1. Clone the repository:

   https://github.com/Luv-valecha/ICS_Major_Project.git



2. Navigate to the project directory:
```
   cd ICS_Major_Project
```


3. Compile the project
```
   gcc -o main main.c -lm
```








## Usage


1. Run the compiled executable:
   ```
   ./main
   ```


2. Follow the prompts to enter the training file name and test file name.


3. Select the desired model from the main menu:
   - Data Scaling
   - Linear Regression
   - Logistic Regression
   - K Nearest Neighbour
   - Exit


4. If you choose Data Scaling:
   - Choose scaler type:
     - Min Max Scaler
     - Standard Scaler


5. The selected model will perform the task using the provided data files.


## File Structure


- **main.c**: Entry point of the program. It handles user input and menu navigation.
- **machine_learning.h**: Header file containing function declarations for machine learning models.
- **models.c**: Implementation file containing functions for the machine learning models.
- **model_functions.c**.implementation file containing common functions across all models






## contributions


- **Dhruv Sharma(b23ci1016)**
- **Luv Valecha(b23cm1022)**
- **Dheeraj Kumar(b23cs1016)**
- **Adithya Subhash(b23mt1004)**

