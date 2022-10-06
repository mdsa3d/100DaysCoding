# **Machine Learning API**

Build a machine learning API using Python, FastAPI and Scikit-Learn AND DEPLOY it using Heroku

### **Directory**:

<table border=1>
    <tr>
        <th><b>File</b></th>
        <th><b>Description</b></th>
    </tr>
    <tr>
        <td><code>rfmodel.pkl</code></td>
        <td> a scikit learn model</td>
    </tr>
    <tr>
        <td><code>sample.json</code></td>
        <td>this is how we should be sending the body of request to ML API</td>
    </tr>
    <tr>
        <td><code>mlapi.py</code></td>
        <td></td>
</table> 


### **Steps**:
1. Create Environment `python -m venv fastml` and activate the virtual environment.
2. Install dependencies:
    - `uvicorn` :-> help to start the server
    - `gunicorn` :-> require when deploying to heroku
    - `fastapi`
    - `pydantic` :-> structure or pass through the requests and have them in appropriate file format.
    - `sklearn`
    - `pandas`

    Install usign follwoing command: `pip install uvicorn gunicorn fastapi pydantic sklearn pandas`
3. Check the environment using `pip list`
4. Create requirements txt `pip3 freeze > requirements.txt`.
5. Create a sample app
6. To launch the app, go to terminal and type `uvicorn mlapi:app --reload`
7. Test the app usign Postman, untill satisfied
8. Bring in machine learning model

Deploying on Heroku
1. Create a `Procfile`
```
web: guniccorn -w 2 -k uvicorn.workers.UvicornWorker mlapi:app
```
Params:
- `-w` : workers and the value here is set to 2
- `-k` : how it is going to be deployed
2. Generate `requirements.txt` file.
3. Create `.gitignore` file:
```
__pycache__
fastml
sample.json
```
4. create `runtime.txt`, and this file contains teh python version on which the app will run.
5. Ready to deploy (open terminal)
    - Initialise folder as git repo : `git init`
    - Add all the files in teh folder: `git add .`
    - Commit the changes : `git commit -m "initial commit"`
    - Login to heroku : `heroku login`
    - create api : `heroku create`
    - push the changes : `git push heroku master` , this will upload the files and build the source.
    - You will get the heroku api address, which you can use to connect to the API


