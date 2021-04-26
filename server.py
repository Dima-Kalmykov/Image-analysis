from PIL import Image
import io
from flask import Flask, send_file
from file_paths import FilePaths
from flask import Flask, render_template
from PIL import Image
import base64
import io

app = Flask(__name__)


# http://192.168.8.36:8080/result?v=1


@app.route('/')
def tuna2():
    # path = FilePaths.result_path_with_color
    im = Image.open("C:/temp/naip/res.jpg")
    data = io.BytesIO()
    im = im.convert('RGB')
    im.save(data, "JPEG")
    encoded_img_data = base64.b64encode(data.getvalue())

    return render_template("index.html", img_data=encoded_img_data.decode('utf-8'))


# return render_template("index.html", img_data=encoded_img_data.decode('utf-8'))


# if __name__ == "__main__":
# app.run(debug=True, port=8080, host='0.0.0.0')
app.run(debug=True, port=8080)
