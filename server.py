import os
from flask import Flask, request, render_template, send_file
from flask_cors import CORS
from werkzeug.utils import secure_filename
from datetime import datetime

from model_utils import predict

UPLOAD_FOLDER = './build/imgs'
ALLOWED_EXTENSIONS = set(['png', 'jpg', 'jpeg', 'webp'])
template_dir = os.path.abspath('./build')
static_folder='./build/static'

app = Flask(__name__, template_folder=template_dir, static_folder=static_folder)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

CORS(app)

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/plantimg.png')
def return_files_tut():
	try:
		return send_file('./public/plantimg.png')
	except Exception as e:
		return str(e)

@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        # check if the post request has the file part
        if 'file' not in request.files:
            data = {
                "message" : "No file was provided!",
            }
            return data, 400
        
        file = request.files['file']
        # if user does not select file, browser also
        # submit a empty part without filename

        if file.filename == '':
            data = {
                "message" : "No file was provided!",
            }
            return data, 400
        
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            NOW = datetime.now()
            newfilename = NOW.strftime("%d_%m_%Y_%H_%M_%S") + filename
            image_path = os.path.join(app.config['UPLOAD_FOLDER'],  newfilename);
            file.save(image_path)


            pred_text = predict(image_path=image_path);

            print(pred_text)

            data = {
                "message" : "Uploaded Successfully!",
                "pred_text" :pred_text
            }
            return data, 200
    else:
        return render_template('index.html')

    

app.run(debug=True, port=80)