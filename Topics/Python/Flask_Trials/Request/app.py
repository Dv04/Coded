from flask import Flask, url_for, render_template, request, session, redirect, flash
import secrets

app = Flask(__name__, static_folder="static")
app.app_context().push()
app.secret_key = secrets.token_hex()


@app.route("/")
def index():
    # if "username" in session:
    #     return f'Logged in as {session["username"]}'
    # else:
    return render_template("index.html")


# @app.route("/login", methods=["POST", "GET"])
# def login():
#     error = None

#     if request.method == "POST":
#         entered_username = request.form["username"]
#         entered_password = request.form["password"]

#         # check if the user details are valid
#         if valid_login(entered_username, entered_password):
#             session["username"] = request.form["username"]
#             flash('You were successfully logged in')
#             return redirect(url_for("index"))
#         else:
#             error = "Invalid username/password"

#     # the code below is executed if the request method was GET or the credentials were invalid
#     return render_template("login.html", error=error)


@app.route("/login", methods=["GET", "POST"])
def login():
    error = None
    if request.method == "POST":
        if request.form["username"] != "admin" or request.form["password"] != "secret":
            error = "Invalid credentials"
        else:
            flash("You were successfully logged in")
            return redirect(url_for("index"))
    return render_template("login.html", error=error)


@app.route("/logout")
def logout():
    # remove the username from the session if it's there
    session.pop("username", None)
    return redirect(url_for("index"))


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
    app.run(host="127.0.0.1", port=5000, debug=True)
