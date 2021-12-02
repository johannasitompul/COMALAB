# COMALAB
A Covid-19 Machine Learning Application that predicts the risk of Covid-19 infection from chest x-ray images.

# Installation Guide
This assumes that the user already has Python, along with its basic libraries installed.

# List of Package dependencies
This is a list of the main packages installed for this application.
Any additional packages are sub-dependencies of these main packages.

- django 3.2.9 					`pip install django`
- tensorflow 2.7.0 				`pip install tensorflow`
- django Crispy Forms 1.13.0 			`pip install django-crispy-forms`
- grad-cam 1.3.5 				`pip install grad-cam`
- matplotlib 3.4.3				`pip install matplotlib`
- pydicom					`pip instlal pydicom`

Run the corresponding `pip` commands in your command prompt to fetch and install the latest versions of these packages.

# Starting the Application
The application will be hosted on a server configured for Django.
This assumes that the user already has a server available, or is otherwise running on a local PC.

- Open the command prompt and change the directory to the location of the app's files as such: `C:\your_path_here\comalab`
- Run the command `py manage.py runserver` to start the server. 
- The system will check for any errors/missing packages, if there are no errors, the process is done.
- In the command window, you should see `Starting development server at <server_ip>`, by connecting to the specified IP of the browser, you can access the application.
