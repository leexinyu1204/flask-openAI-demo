import os

from langchain import FAISS
from authentication import predictor, publish_iteration_name, project_id
from chatgpt import get_response_azure, get_response_bert
from flask import Flask, render_template, request, redirect, url_for, jsonify, send_from_directory
from werkzeug.utils import secure_filename
from authentication import embeddings

db = FAISS.load_local("./faiss_index", embeddings)
# db = create_index_database("./data")
app = Flask(__name__)

basedir = os.path.abspath(os.path.dirname(__file__))
upload_dir = os.path.join(basedir, 'uploads')


def del_file(path):
    ls = os.listdir(path)
    for i in ls:
        c_path = os.path.join(path, i)
        if os.path.isdir(c_path):
            del_file(c_path)
        else:
            os.remove(c_path)


del_file(upload_dir)
app.config['UPLOAD_FOLDER'] = upload_dir


@app.route("/")
def home():
    return render_template("index.html")


#
# Define route for home page
@app.route("/get", methods=["GET", "POST"])
def get_response_text():
    if request.method == 'GET':
        userText = request.args.get('msg')

        # if respond_img  != '':
        #     imgSRC = f"/static/images/{respond_img }"
        answer, _ = get_response_azure(db, userText)
        print(answer)
        response_data = {'status': 'success', 'text': answer, 'img': ""}
        return jsonify(response_data)

    # return answer


@app.route("/send", methods=["POST"])
def upload_file():
    file = request.files['avatar']
    userText = request.form.get('msg')

    imgSRC = ''
    try:
        if file:

            filename = secure_filename(file.filename)
            img_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)

            if not os.path.isfile(img_path):
                print("file not exist")
                file.save(img_path)

            imgSRC = f"/uploads/{filename}"

        response_data = {'status': 'success', 'text': userText, 'img': imgSRC}
    except Exception as e:
        print(e)
        response_data = {'status': 'fail', 'msg': 'upload failure, please try to upload again'}

    return jsonify(response_data)


@app.route('/uploads/<path:filename>')
def download_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)


@app.route("/response", methods=["POST", "GET"])
def get_response_image():
    imgSRC = ''
    userText = request.args.get('msg')
    filename = request.args.get('avatar')
    print(filename)
    if len(filename) != 0:
        img_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)

        # if not os.path.isfile(img_path):
        #     file.save(img_path)
        print("GETTING RESPONSE ......")
        while not os.path.isfile(img_path):
            print("uploading")

        with open(img_path, "rb") as image_contents:
            results = predictor.classify_image(
                project_id, publish_iteration_name, image_contents.read())

        description = results.predictions[0].tag_name

        userText += " and the image description: " + description

    print("user input is: ", userText)
    answer, respond_img = get_response_azure(db, userText)
    if respond_img != '':
        imgSRC = f"/uploads/{respond_img}"
    # answer = answer.encode('utf8')
    #
    # print("answer is: ", answer)
    response_data = {'status': 'success', 'text': answer, 'img': imgSRC}
    return jsonify(response_data)


if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0')
