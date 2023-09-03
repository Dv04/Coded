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


@app.route("/projects/")
def projects():
    return "The project page"


@app.route("/about")
def about():
    return "The about page"


@app.route("/login")
def login():
    return "login"


@app.route("/user/<username>")
def profile(username):
    return f"{username}'s profile"


@app.route("/post/<int:post_id>")
def show_post(post_id):
    # show the post with the given id, the id is an integer
    return f"Post {post_id}"


@app.route("/path/<path:subpath>")
def show_subpath(subpath):
    # show the subpath after /path/
    return f"Subpath {escape(subpath)}"


if __name__ == "__main__":
    with app.test_request_context():
        print(url_for("index"))
        print(url_for("login"))
        print(url_for("login", next="/"))
        print(url_for("profile", username="John Doe"))

    app.run()
