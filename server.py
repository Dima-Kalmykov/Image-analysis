import os
from multiprocessing import Value
from typing import Any

from flask import request, flash, redirect
from werkzeug.utils import secure_filename

from classifier.classifier import ImageClassifier
from flask import Flask, render_template
from PIL import Image
import base64
import io

from utils.file_paths import FilePaths

app = Flask(__name__)

app.config['UPLOAD_FOLDER'] = 'C:\\temp\\userTiff'
app.secret_key = 'super secret key'
app.config['SESSION_TYPE'] = 'filesystem'
counter = Value('i', 0)


def get_response(file_number: str) -> Any:
    """
    Return page with result image.
    :param file_number: id of file
    :return: page with result image
    """
    path_ro_result = FilePaths.result_path_with_color + str(file_number) + ".tif"
    image = Image.open(path_ro_result)
    data = io.BytesIO()
    image = image.convert('RGB')
    image.save(data, "JPEG")
    encoded_image_data = base64.b64encode(data.getvalue())

    return render_template("index.html", img_data=encoded_image_data.decode('utf-8'))


@app.route('/upload', methods=['POST'])
def upload_file() -> Any:
    """
    Upload tif file to server.
    :return: Page with result image
    """
    with counter.get_lock():
        counter.value += 1
        file_number = counter.value

    if 'file' not in request.files:
        flash('No file part')
        return redirect(request.url)

    file = request.files['file']
    filename = secure_filename(file.filename)
    filename = filename[:-4] + str(file_number) + ".tif"
    print(filename)
    path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(path)
    ImageClassifier.classify(path, file_number)
    return get_response(file_number)


app.run(debug=True, port=8080)
