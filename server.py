import os
from multiprocessing import Value
from typing import Any

from PIL import Image
from flask import Flask, send_from_directory
from flask import request
from flask_cors import CORS

from Backend.classifier.classifier_file import ImageClassifier
from Backend.utils.utils import Utils

app = Flask(__name__)
CORS(app)
upload_folder = 'C:\\temp\\userTiff'
result_folder = 'C:/temp/serverResult/Colored'
app.config['UPLOAD_FOLDER'] = upload_folder
app.secret_key = 'super secret key'
app.config['SESSION_TYPE'] = 'filesystem'
counter = Value('i', 0)


def get_color(key: str) -> str:
    """
    Get color from request by key.
    :param key: color key
    :return: string representation of color
    """
    return f'#{request.args.get(key)}'


@app.route('/progress/<id>', methods=['GET'])
def progress(id: int) -> str:
    """
    Get progress.
    :param id: file id
    :return: string representation of progress
    """
    return str(Utils.get_progress(id))


@app.route('/classify/<id>', methods=['POST'])
def classify(id) -> Any:
    """
    Upload '.tif' file to server.
    :param id: file id
    :return: Page with result image
    """
    water = get_color('water')
    infrastructure = get_color('infrastructure')
    vegetation = get_color('vegetation')
    fields = get_color('fields')
    trees = get_color('trees')
    ground = get_color('ground')

    colors = [water, infrastructure, vegetation, fields, trees, ground]
    kernel = request.args.get('kernel')
    path = f"{upload_folder}/{id}.tif"

    classifier = ImageClassifier()
    classifier.classify(path, id, colors, kernel)

    return send_from_directory(result_folder, f'classified{id}.jpg')


@app.route('/upload', methods=['POST'])
def upload():
    """
    Upload file to server.
    :return: jpg image
    """
    with counter.get_lock():
        counter.value += 1
        file_number = counter.value

    file = request.files['file']

    filename = str(file_number) + ".tif"
    path = os.path.join(app.config['UPLOAD_FOLDER'], filename)

    file.save(path)
    out_path = path[:-3] + "jpg"
    image = Image.open(path)
    image.thumbnail(image.size)
    rgb_image = image.convert('RGB')
    rgb_image.save(out_path, "JPEG", quiality=100)
    out_path = out_path[out_path.rfind('\\') + 1:]

    print(f"File with id = {file_number} was uploaded successfully!")

    return send_from_directory(upload_folder, out_path)


if __name__ == '__main__':
    # app.run(debug=True, host='0.0.0.0', port=8080)
    app.run(debug=True, port=8080)
