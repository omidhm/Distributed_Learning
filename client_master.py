from flask import Flask, request, make_response, jsonify
from werkzeug.utils import secure_filename
import codecs



#Flask part
app = Flask(__name__)
@app.route('/file', methods = ['GET', 'POST'])
def textFile():
    if request.method == 'POST':  
        f = request.files['document'] 
        filename = secure_filename(f.filename)
        file_path = filename
        f.save(file_path) 
        return "received"

if __name__ == '__main__':
    app.run(host = '0.0.0.0', port = 5001, threaded = True) 
    
