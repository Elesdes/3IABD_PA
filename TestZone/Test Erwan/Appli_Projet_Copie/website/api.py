import os
import numpy as np
import assert_img
from flask import Flask, flash, render_template
from flask import request
from flask import redirect, url_for
from werkzeug.utils import secure_filename
from PIL import Image


app = Flask(__name__)
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
            # Save to destroy
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            f = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            newsize = (32, 32)
            img = Image.open(f)
            img = img.convert('L')
            img = img.resize(newsize)
            img = np.array(img)
            img = np.ravel(img)
            os.remove(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            if assert_img.assert_img(img) == 0:
                # return redirect(url_for('templates', filename="eiffel.html"))
                return render_template('eiffel.html')
            if assert_img.assert_img(img) == 1:
                return render_template('triumphal.html')
            if assert_img.assert_img(img) == 2:
                return render_template('louvre.html')
            if assert_img.assert_img(img) == 3:
                return render_template('pantheon.html')
            if assert_img.assert_img(img) == -1:
                return render_template('not_asserted.html')
    return redirect(url_for('static', filename='index.html', code=302))


# -------------------------------------------------------------------------
@app.route("/choose", methods=['POST', 'GET'])
def request_from_choose():
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
            # Save to destroy
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            f = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            newsize = (32, 32)
            img = Image.open(f)
            img = img.convert('L')
            img = img.resize(newsize)
            img = np.array(img)
            img = np.ravel(img)
            os.remove(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            choice_of_algo = int(request.form['choice'])

            if choice_of_algo == 0:
                guess = assert_img.linear_was_chosen(img)
                if guess == 0:
                    return render_template('eiffel.html')
                elif guess == 1:
                    return render_template('triumphal.html')
                elif guess == 2:
                    return render_template('louvre.html')
                elif guess == 3:
                    return render_template('pantheon.html')
                else:
                    render_template('not_asserted.html')
            elif choice_of_algo == 1:
                guess = assert_img.MLP_was_chosen(img)
                if guess == 0:
                    return render_template('eiffel.html')
                elif guess == 1:
                    return render_template('triumphal.html')
                elif guess == 2:
                    return render_template('louvre.html')
                elif guess == 3:
                    return render_template('pantheon.html')
                else:
                    render_template('not_asserted.html')
            elif choice_of_algo == 2:
                guess = assert_img.RBF_was_chosen(img)
                if guess == 0:
                    return render_template('eiffel.html')
                elif guess == 1:
                    return render_template('triumphal.html')
                elif guess == 2:
                    return render_template('louvre.html')
                elif guess == 3:
                    return render_template('pantheon.html')
                else:
                    render_template('not_asserted.html')
            elif choice_of_algo == 3:
                guess = assert_img.SVM_was_chosen(img)
                if guess == 0:
                    return render_template('eiffel.html')
                elif guess == 1:
                    return render_template('triumphal.html')
                elif guess == 2:
                    return render_template('louvre.html')
                elif guess == 3:
                    return render_template('pantheon.html')
                else:
                    return render_template('not_asserted.html')
            else:
                flash('No choice selected')
                return redirect(request.url)
    return redirect(url_for('static', filename='index.html', code=302))


# -------------------------------------------------------------------------
@app.route("/")
def root():
    return redirect(url_for('static', filename='index.html', code=302))


# ----------------------------------------------------------------------------------------------------------------------
if __name__ == '__main__':
    app.run(host='127.0.0.1', port=5001)

# ----------------------------------------------------------------------------------------------------------------------
