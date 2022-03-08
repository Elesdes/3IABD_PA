import os
from flask import Flask, flash
from flask import request
from flask import redirect, url_for
from werkzeug.utils import secure_filename

app = Flask(__name__, static_url_path='/App', static_folder='App')
app.secret_key = "super secret key"
UPLOAD_FOLDER = './static/Temp_A_Suppr/'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


# -------------------------------------------------------------------------
@app.route("/submit", methods=['POST', 'GET'])
def request_from_paris():
    if request.method == 'GET':
        return redirect(url_for('static', filename='index.html', code=302))
    if request.method == 'POST':
        # check if the post request has the file part
        if 'file' not in request.files:
            print("I'm here2")
            flash('No file part')
            return redirect(url_for('static', filename='index.html', code=302))
        file = request.files['file']
        # If the user does not select a file, the browser submits an
        # empty file without a filename.
        if file.filename == '':
            flash('No selected file')
            return redirect(request.url)
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            return redirect(url_for('static', filename="Temp_A_Suppr/" + filename))

    return redirect(url_for('static', filename='index.html', code=302))


@app.route("/")
def root():
    return redirect(url_for('static', filename='index.html', code=302))


# ----------------------------------------------------------------------------------------------------------------------
if __name__ == '__main__':
    app.run(host='127.0.0.1', port=5001)

# ----------------------------------------------------------------------------------------------------------------------
