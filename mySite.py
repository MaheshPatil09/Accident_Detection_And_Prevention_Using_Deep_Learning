# import the necessary packages
from flask import Flask, render_template, redirect, url_for, request,session,Response
from werkzeug import secure_filename
import os
import cv2
from detect_drowsiness import detectDrowsiness
from supportFile import *

login_status = 0

app = Flask(__name__)

app.secret_key = '1234'
app.config["CACHE_TYPE"] = "null"
app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 0

@app.route('/home', methods=['GET', 'POST'])
def home():
	return redirect(url_for('input'))

@app.route('/', methods=['GET', 'POST'])
def input():
	error = None
	global login_status
	if request.method == 'POST':
		if request.form['username'] != 'admin' or request.form['password'] != 'admin':
			error = 'Invalid Credentials. Please try again.'
		else:
			login_status = 1
			return redirect(url_for('video'))

	return render_template('input.html', error=error)

@app.route('/image', methods=['GET', 'POST'])
def image():
	if login_status != 1:
		return redirect(url_for('input'))
	if request.method == 'POST':
		if request.form['sub']=='Upload':
			savepath = r'upload/'
			photo = request.files['photo']
			photo.save(os.path.join(savepath,(secure_filename(photo.filename))))
			image = cv2.imread(os.path.join(savepath,secure_filename(photo.filename)))
			cv2.imwrite(os.path.join("static/images/","test_image.jpg"),image)
			return render_template('image.html')
		elif request.form['sub'] == 'Test':
			#num = request.form['num']
			result = predict()
			return render_template('image.html',result=result)
	return render_template('image.html')


@app.route('/video', methods=['GET', 'POST'])
def video():
	if login_status != 1:
		return redirect(url_for('input'))
	return render_template('video.html')

@app.route('/video_stream')
def video_stream():  
    return Response(detectDrowsiness(),mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/video1', methods=['GET', 'POST'])
def video1():
	if login_status != 1:
		return redirect(url_for('input'))
	return render_template('video1.html')

@app.route('/video1_stream')
def video1_stream():  
    return Response(get_frame(),mimetype='multipart/x-mixed-replace; boundary=frame')

# No caching at all for API endpoints.
@app.after_request
def add_header(response):
	# response.cache_control.no_store = True
	response.headers['Cache-Control'] = 'no-store, no-cache, must-revalidate, post-check=0, pre-check=0, max-age=0'
	response.headers['Pragma'] = 'no-cache'
	response.headers['Expires'] = '-1'
	return response


if __name__ == '__main__':
	app.run(host='0.0.0.0', debug=True, threaded=True)