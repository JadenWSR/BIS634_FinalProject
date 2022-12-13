## How to run this app on your computer from command line
 At the command prompt or terminal, navigate to your projects directory
- Mac: $ export FLASK_ENV=development; flask run
- Windows: set FLASK_ENV=development; flask run

Site will be available at: http://localhost:5000

Note:

- Please make sure you have installed the required versions of all python packages needed in [requirements.txt](https://github.com/JadenWSR/SteamProject/blob/main/requirements.txt) before you run the app. For both Mac and Windows: `pip install -r requirements.txt`
- Please make sure your Python version is above **3.10**. Otherwise, the match-case function in database.py will throw an error and would lead to the whole app failing to work.