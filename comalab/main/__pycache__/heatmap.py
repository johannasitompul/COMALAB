# import the necessary packages
import matplotlib.pyplot as plt
from tensorflow.keras.models import Model
import tensorflow as tf
import numpy as np
import cv2
from keras.preprocessing import image
class GradCAM:
    def __init__(self, model, classIdx, layerName=None):
        self.model = model
        self.classIdx = classIdx
        self.layerName = layerName
        if self.layerName is None:
            self.layerName = self.find_target_layer()
    def find_target_layer(self):
        for layer in reversed(self.model.layers):
# check to see if the layer has a 4D output
            if len(layer.output_shape) == 4:
                return layer.name
        raise ValueError("Could not find 4D layer. Cannot apply GradCAM.")
    def compute_heatmap(self, image, eps=1e-8):
        gradModel = Model(
            inputs=[self.model.inputs],
            outputs=[self.model.get_layer(self.layerName).output,
                self.model.output])
# record operations for automatic differentiation
        with tf.GradientTape() as tape:
            inputs = tf.cast(image, tf.float32)
            (convOutputs, predictions) = gradModel(inputs)
            loss = predictions[:, self.classIdx]
# use automatic differentiation to compute the gradients
        grads = tape.gradient(loss, convOutputs)
# compute the guided gradients
        castConvOutputs = tf.cast(convOutputs > 0, "float32")
        castGrads = tf.cast(grads > 0, "float32")
        guidedGrads = castConvOutputs * castGrads * grads
        convOutputs = convOutputs[0]
        guidedGrads = guidedGrads[0]
        weights = tf.reduce_mean(guidedGrads, axis=(0, 1))
        cam = tf.reduce_sum(tf.multiply(weights, convOutputs), axis=-1)
# resize the heatmap to oringnal X-Ray image size
        (w, h) = (image.shape[2], image.shape[1])
        heatmap = cv2.resize(cam.numpy(), (w, h))
# normalize the heatmap
        numer = heatmap - np.min(heatmap)
        denom = (heatmap.max() - heatmap.min()) + eps
        heatmap = numer / denom
        heatmap = (heatmap * 255).astype("uint8")
# return the resulting heatmap to the calling function
        return heatmap
        
model_graph = Graph()
with model_graph.as_default():
    tf_session = tf.compat.v1.Session()
    with tf_session.as_default():
        model=load_model('./model/vgg_model.h5')

def createheatmap(picture):
	requestpage = request.POST.get('predict_requestpage', '/') #Store requestor page for reference when redirecting later
    random_img = image.load_img(picture, target_size=(224,224))
    random_img_array = image.img_to_array(random_img)
    random_img_array_rescaled = random_img_array/255.0
    dataXG = np.expand_dims(random_img_array_rescaled, axis = 0)
    original = cv2.imread(picture)
    orig = cv2.cvtColor(original , cv2.COLOR_BGR2RGB)
    prediction = loaded_model.predict(dataXG)
    i = np.argmax(prediction[0])
    heatmap = cam.compute_heatmap(dataXG)#show the calculated heatmap
    cam = GradCAM(model=loaded_model, classIdx=i, layerName='mixed10')
    # Old fashioned way to overlay a transparent heatmap onto original image, the same as above
    heatmapY = cv2.resize(heatmap, (orig.shape[1], orig.shape[0]))
    heatmapY = cv2.applyColorMap(heatmapY, cv2.COLORMAP_HOT)  # COLORMAP_JET, COLORMAP_VIRIDIS, COLORMAP_HOT
    imageY = cv2.addWeighted(heatmapY, 0.5, original, 1.0, 0)

    # draw the orignal x-ray, the heatmap, and the overlay together
    output = np.hstack([orig, heatmapY, imageY])
    fig, ax = plt.subplots(figsize=(20, 18))
    ax.imshow(np.random.rand(1, 99), interpolation='nearest')
    plt.imshow(output)
    plt.savefig("media/heatmap/"+ picture)
    
def predict(request):
	requestpage = request.POST.get('predict_requestpage', '/') #Store requestor page for reference when redirecting later
	"""
	Currently WIP
	all_objects = ImagePool.objects.all() 			#Get all objects
	for object in all_objects:
		if image.risk == 0.0: 						#this means "if a prediction has not been run yet"
			image = Image.open(object.image.path) 	#Open the image
			image = data_preprocessing(image) 		#Send the image variable into the preprocessing function
			x = model.predict(image)				#Send preprocessed image for prediction, store return value
			probability_covid = 1 - x[0][0] * 100 	#Percentage chance of having covid, apply number rounding as needed
			object.risk = percent_value 			#Store probability value into risk column of the object database table
			if percent value > 90:
				object.risk_class = "highrisk" 		#Will be attached to <td> elements as a secondary class for the alert highlighting feature
	"""
	all_objects= ImagePool.objects.all()
	
	for object in all_objects:
		if object.risk == 0.0:
			
			img = tf.keras.preprocessing.image.load_img(object.image.path, target_size=(224, 224))
			img_array = tf.keras.preprocessing.image.img_to_array(img)
			norm_array = img_array/255
			processed_img = norm_array.reshape(1,224,224,3)
			with model_graph.as_default():
        			with tf_session.as_default():
            				z=model.predict(processed_img)
			covid_prob = (1 - z[0][0]) * 100 
			percent_value = round(covid_prob,2)
			object.risk = percent_value
			
			if percent_value > 90:
				object.risk_class = 'highrisk'
			
			object.save()