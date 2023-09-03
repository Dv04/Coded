import os

from flask import Flask


def create_app(test_config=None):
    # Set the instance path
    instance_path = os.path.join(os.path.dirname(__file__), "instance")

    # create and configure the app
    app = Flask(__name__, instance_path=instance_path, instance_relative_config=True)
    app.config.from_mapping(
        SECRET_KEY="dev",
        DATABASE=os.path.join(app.instance_path, "flaskr.sqlite"),
    )

    if test_config is None:
        # load the instance config, if it exists, when not testing
        app.config.from_pyfile("config.py", silent=True)
    else:
        # load the test config if passed in
        app.config.from_mapping(test_config)

    # ensure the instance folder exists
    try:
        os.makedirs(app.instance_path)
    except OSError:
        pass

    # a simple page that says hello
    @app.route("/hello/")
    def hello():
        return "Hello, World!"

    from . import db, auth, blog

    db.init_app(app)
    app.register_blueprint(auth.bp)
    app.register_blueprint(blog.bp)
    app.add_url_rule('/', endpoint='index')
    return app


# Call create_app() to run the application
if __name__ == "__main__":
    app = create_app()
    app.run()
