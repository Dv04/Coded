from markupsafe import escape
from flask import Flask, url_for, render_template, request

app = Flask(__name__)


@app.route("/")
def index():
    return "Index Page"


@app.route("/hello/")
@app.route("/hello/<name>")
def hello(name=None):
    return render_template("hello.html", name=name)


@app.route("/about")
def about():
    return "The about page"


@app.route("/login", methods=["POST", "GET"])
def login():
    error = None

    if request.method == "POST":
        entered_username = request.form["username"]
        entered_password = request.form["password"]

        # check if the user details are valid
        if valid_login(entered_username, entered_password):
            return log_the_user_in(entered_username)
        else:
            error = "Invalid username/password"

    # the code below is executed if the request method was GET or the credentials were invalid
    return render_template("login.html", error=error)


# Function to validate the user entered details and log them in
def valid_login(username, password):
    if username == "admin" and password == "pass":
        return True
    else:
        return False


# Function to log the user in if they enter the right credentials
def log_the_user_in(username):
    return "Welcome " + username + "!"


if __name__ == "__main__":
    app.run()
