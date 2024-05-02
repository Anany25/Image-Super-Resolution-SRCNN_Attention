from flask import Flask, render_template, request, redirect, url_for
from torchvision import transforms
from PIL import Image
import torch
import numpy as np
import io
import cv2
from model import DeeperSRCNNwithAttention,SRCNN

app = Flask(__name__)

y_dim=1020//2
x_dim=678//2
# Load your PyTorch model
# Example: model = YourPyTorchModel()
model=DeeperSRCNNwithAttention()
model.load_state_dict(torch.load('model_deep_att.pth'))

#src_model=SRCNN()
#src_model.load_state_dict(torch.load('model_y.pth'))
# Define transformations for input image

def process_image(image_bytes):
	image = Image.open(io.BytesIO(image_bytes))
	image = np.array(image)
	downscaled_img=cv2.resize(image,(y_dim//2,x_dim//2),interpolation=cv2.INTER_CUBIC)
	upscaled_img=cv2.resize(downscaled_img,(y_dim,x_dim),interpolation=cv2.INTER_CUBIC)
	upscaled_img_YCbCr=cv2.cvtColor(upscaled_img, cv2.COLOR_BGR2YCrCb)
	Y,Cb,Cr=cv2.split(upscaled_img_YCbCr)
	test_image_tensor=torch.from_numpy(Y).float().unsqueeze(0)
	test_output_tensor = model(test_image_tensor.unsqueeze(0))
	output_array = test_output_tensor.detach().squeeze().cpu().numpy()
	output_array=output_array.astype('uint8')
	Cb_resized = cv2.resize(Cb, (510, 339))
	Cr_resized = cv2.resize(Cr, (510, 339))
	enhanced_image_ycbcr = cv2.merge([output_array,  Cr_resized,Cb_resized,])
	enhanced_image_rgb_srcnn = cv2.cvtColor(enhanced_image_ycbcr, cv2.COLOR_YCrCb2BGR)
	#output_image=cv2.cvtColor(enhanced_image_rgb_srcnn,cv2.COLOR_BGR2RGB)
	output_image=enhanced_image_rgb_srcnn
	cv2.imwrite('static/output_image.png',output_image)
	return output_image

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        if 'file' not in request.files:
            return redirect(request.url)
        file = request.files['file']
        if file.filename == '':
            return redirect(request.url)
        if file:
            image_bytes = file.read()
            output_image = process_image(image_bytes)
            # Save the output image temporarily or send it directly to the UI
            # Example: output_image.save('output_image.png')
            # Return the filename or image data to be displayed in the UI
            return render_template('index.html', output_image='output_image.png')
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)