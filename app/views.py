#!/usr/bin/env python3

import os
from flask import Flask, render_template, request, make_response, jsonify, redirect, url_for, send_from_directory
from keras.preprocessing.image import img_to_array, load_img
import cherrypy
from paste.translogger import TransLogger
from werkzeug.utils import secure_filename
from app import app
from vgg16 import Vgg16
from IPython.display import display
from PIL import Image


ANIMALS_FOLDER = '/Users/carterlandis/Documents/GitHub/checkyourself-backend/app/static/uploadedFiles/classif'
ALLOWED_EXTENSIONS = set(['txt', 'pdf', 'png', 'jpg', 'jpeg', 'gif'])
app.config['ANIMALS_FOLDER'] = ANIMALS_FOLDER


model = Vgg16()
model.model.load_weights('my_model.h5')


# SNAPWAT FRONTEND
@app.route("/", methods=['GET','POST'])
def hello():
    return render_template('index.html')


# ALLOWERD FILES
def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


# CURRENT TESTING ROUTE
@app.route('/test', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        # check if the post request has the file part
        if 'file' not in request.files:
            flash('No file part')
            return redirect(request.url)
        file = request.files['file']
        # if user does not select file, browser also
        # submit a empty part without filename
        if file.filename == '':
            flash('No selected file')
            return redirect(request.url)
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file.save(os.path.join(app.config['ANIMALS_FOLDER'], filename))
            return redirect('/api/v1/classify_image')

            # return redirect('/api/v1/classify_image/{}'.format(filename))
            # return redirect(url_for('upload_file',
            #                         filename=filename))
    return '''
    <!doctype html>
    <title>Upload new File</title>
    <h1>Upload new File</h1>
    <form method=post enctype=multipart/form-data>
      <p><input type=file name=file>
         <input type=submit value=Upload>
    </form>
    '''

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['ANIMALS_FOLDER'],
                               filename)


# @app.route('/api/v1/classify_image', methods=['GET','POST'])
# def classify_image():
#     if 'image' not in request.files:
#         resp = "error: bad request"
#     else:
#         cherrypy.log("Image request")
#         image_request = request.files['image']
#         img = read_image_from_ioreader(image_request)
#         resp = model.predict(img, True)
#     return make_response(jsonify({'message': resp}))
#     # return make_response(jsonify({'message': resp}), STATUS_OK)


@app.route('/api/v1/classify_image', methods=['GET','POST'])
def classify_image():
    abs_path = os.path.join(os.path.dirname(__file__), 'static', 'uploadedFiles', 'classify', 'catdog.jpg')

    prob = model.model.predict( img_to_array(load_img(abs_path, target_size=[224, 224])).reshape(1, 3, 224, 224))
    resp_json = []
    print prob[0][1]
    if prob[0][1] < .5:
        resp_json = "cat"
    else:
        resp_json = "dog"

    resp_actual = {}
    resp_actual["class"] = resp_json
    resp_actual["propability"] = float(prob[0][1])


    if 'image' not in request.files:
        resp = "error: bad request"
    else:
        cherrypy.log("Image request")
        image_request = request.files[image]
        img = read_image_from_ioreader(image_request)
        resp = model.predict(img, True)
    return make_response(jsonify({'result': resp_actual}))
    # return make_response(jsonify({'message': resp}), STATUS_OK)


@app.route('/image')
def showImage():
    full_filename = os.path.join(app.config['ANIMALS_FOLDER'], 'cat.png')
    return render_template("image.html", image=full_filename)


# RUN SERVER
def run_server():
    # Enable WSGI access logging via Paste
    app_logged = TransLogger(app)

    # Mount the WSGI callable object (app) on the root directory
    cherrypy.tree.graft(app_logged, '/')

    # Set the configuration of the web server
    cherrypy.config.update({
        'engine.autoreload_on': True,
        'log.screen': True,
        'server.socket_port': 5000,
        'server.socket_host': '0.0.0.0'
    })

    # Start the CherryPy WSGI web server
    cherrypy.engine.start()
    cherrypy.engine.block()

if __name__ == "__main__":
    run_server()

