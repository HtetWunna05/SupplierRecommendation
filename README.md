Download the Code: Click the green "<> Code" button on GitHub and select "Download ZIP" (or use Git Clone).
Install Python Tools: Open the folder in VS Code and run this in the terminal:
py -m pip install pymongo python-dotenv
Create their own "Key" (.env):
Create a new file named .env.
Paste their own MongoDB link inside: MONGODB_URI=mongodb://localhost:27017/supplier_db
Setup the Database: Run py database.py first to connect to their local MongoDB.
Import Data: Use MongoDB Compass to import the cleaned_master_data.csv file into their database, just like you did.
