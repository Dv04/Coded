import os
from flask import (
    Flask,
    flash,
    request,
    redirect,
    url_for,
    render_template,
    send_from_directory,
)
from werkzeug.utils import secure_filename

app = Flask(__name__, static_folder="static")
app.app_context().push()
app.secret_key = "secret-tunnel"
app.add_url_rule("/uploads/<name>", endpoint="download_file", build_only=True)

UPLOAD_FOLDER = "upload"
ALLOWED_EXTENSIONS = {"txt", "pdf", "png", "jpg", "jpeg", "gif"}

app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
app.config["MAX_CONTENT_LENGTH"] = 160 * 1000 * 1000


def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route("/", methods=["GET", "POST"])
def upload_file():
    if request.method == "POST":
        # check if the post request has the file part
        if "file" not in request.files:
            flash("No file part")
            return redirect(request.url)
        file = request.files["file"]
        # If the user does not select a file, the browser submits an
        # empty file without a filename.
        if file.filename == "":
            flash("No selected file")
            return redirect(request.url)
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file.save(os.path.join(app.config["UPLOAD_FOLDER"], filename))
            return redirect(url_for("download_file", name=filename))
    return render_template("upload.html")


@app.route("/uploads/<name>")
def download_file(name):
    return send_from_directory(app.config["UPLOAD_FOLDER"], name)


from flask import abort, redirect, url_for


@app.route("/trial")
def index():
    return redirect(url_for("login"))


@app.route("/login")
def login():
    abort(400)


@app.errorhandler(400)
def page_not_found(error):
    return render_template("upload.html"), 404


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=True)
