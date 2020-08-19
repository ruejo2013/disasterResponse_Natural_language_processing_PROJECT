

Table of Contents
1.	Installations
2.	Project Motivation
3.	File Descriptions
4.	Running the program
5.	Results
6.	Licensing, Authors, and Acknowledgements
Installation
To run the code, python3 or above will be require. Other libraries needed to run these codes may be found in the requirements.txt file. 
Other dependencies are:
i.	Pyhton 3 and above 
ii.	ML Libraries: Sklearn, Numpy, Pandas
iii.	Natural Language Processing: NLTK (import and download)
iv.	SQLite Database: SQLalchemy
v.	Web app libraries: Flask, Plotly.
Project Motivation
This project is part of Data Science Nanodegree Program by Udacity. It is in collaboration with Figure Eight, which provided the data and project specification. The Project aim, is to use Natural Language Processing (NLP) Machine Learning model to predict multi-class labels of disaster data. It categorizes predictions into 36 different categories. 
The project is in three parts 
1.	ETL Pipeline: Extracts, transforms and load the data into an sqlite database 
2.	Machine Learning model. Trains the model, makes prediction, defines the best parameters (using gridsearchcv) and export the model to pickle file.
3.	Web app. Web app to display the analysis of the request for the different categories from the messages
When a disaster happens, say hurricane, victims send messages, tweet, or emails requesting for help. This project collects these types of messages and classifier them, to the type of help requested. 
The project helps in: 
1.	Classifying messages to the type of needs.
2.	Disaster response agencies can use the model to assign task to responsible agencies as the request are made. 
3.	Making predictions on requested needs during a disaster, and so helping in proactive stock piling of help materials.
File Descriptions
There are three folders in the project folder (all in this github repo).
1.	Data folder: which contains:  
i.	Message.csv data file.
ii.	Categories.csv data file.
iii.	process_data.py
2.	Model Folder:
i.	train_classifier.py
3.	App folder:
i.	go.html
ii.	master.html
iii.	run.py
Running the program
Run the program in these sequence:
i.	Run the ETL pipeline that, load, extract, clean and store the data in an sqlite database run on the command line 
Python3 data/process_data.py data/messages.csv data/categories.csv data/DisasterResponse.db
ii.	To run the ML classifier. It trains and export model to a pickle file 
Python3 models/train_classifier.py data/DisasterResponse.db models/classifier.pkl 
iii.	Run the web app:
Cd into the app directory 
Python3 run.py 
Results
The model performed at about 93% Precision, and about 92% Recall. So it is pretty effective in predicting the target class from the message. 
Licensing, Authors, Acknowledgements
Credit must be given to the Udacity Data Science Nano Degree team, especially to the instructors, mentors, and code reviewers, for the excellent course content. Credit also goes to Figure Eight for the Data used in the project, the project description and solutions it tends to provide. 
