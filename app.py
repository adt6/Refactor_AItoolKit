from flask import Flask, request, render_template
from src.Logger import MyLogger

app = Flask(__name__)
logger = MyLogger()


@app.route('/')
def show_main_page():
    logger.info("Requesting main page")
    return render_template("login.html")


@app.route('/results', methods=['POST'])
def login():
    result = request.form
    logger.info("Posting data {}".format(result))
    return render_template("results.html", result=result)


if __name__ == '__main__':
    app.run()
