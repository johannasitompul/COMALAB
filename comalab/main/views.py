#Django libraries
from django.http.response import HttpResponseRedirect
from django.shortcuts import render, redirect
from django.http import HttpResponse
from django.contrib.auth.decorators import login_required
from .models import ImagePool
from .forms import CreateNewPool

#File I/O libraries
import os
import csv

#Machine Learning libraries
from tensorflow.keras.models import Model
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.preprocessing import *
import tensorflow as tf

#Image processing libraries
import cv2
from PIL import Image
import numpy as np
import pydicom as dicom

#Heatmap Libraries
from .gradcam import GradCAM as gc
import matplotlib.pyplot as plt
from tensorflow import Graph


# Create your views here
def index(response, id):
	ls = ImagePool.objects.get(id=id)
	return render(response, "main/test.html", {"ls":ls})

def home(request):
	return render(request, 'main/home.html', {"name":"test"})

def upload(request):
	requestpage = request.POST.get('upload_requestpage', '/')
	if request.method == "POST":
		form = CreateNewPool(request.POST, request.FILES)
		images = request.FILES.getlist('uploaded_image')
		for img in images:
			filename, extension = os.path.splitext(img.name)
			if extension == ".dcm":
				#Convert dcm to png, file will be written and stored
				new_filename = convert_dcm_to_png(img, filename)
				#Manually create database entry and assign converted image path to entry
				instance = ImagePool()
				instance.image = "images/" + new_filename
				instance.filename= new_filename
				instance.save()
			elif extension in (".jpg", ".png", ".jpeg"):
				instance = ImagePool(image=img, filename=img.name)
				instance.save()
			else:
				print("Invalid file format, abort upload")
				pass
	else:
		form = CreateNewPool()
	return redirect(requestpage)

def convert_dcm_to_png(img, filename):
	ds = dicom.dcmread(img.file)			#Open image as dicom image
	pixel_array_np = ds.pixel_array 		#Parse dicom image into a numpy pixel array
	new_filename = filename + ".png"
	image_path = "media/images/" + new_filename
	cv2.imwrite(image_path, pixel_array_np) #Write numpy array to new file with .png extension
	return new_filename						#Return imagepath of the new .png file

#Load_table function, this is run everytime predict.html is loaded/refreshed
@login_required(login_url='/home')
def load_table(request):
	filenames = ImagePool.objects.all()
	return render(request, 'main/predict.html', {'fn_dict':filenames})

##Delete function
def del_images(request):
	requestpage = request.POST.get('delete_requestpage', '/')			#Store requestor page for reference when redirecting later
	if request.method == "POST":
		selected_ids = request.POST.getlist('image_id')					#Selected checkboxes will send their value {{image.id}} in the form
		image_objects = ImagePool.objects.filter(id__in=selected_ids) 	#Search db for selected_ids
		for image in image_objects:
			#Constructing filepath to heatmap folder
			image_folder = os.path.dirname(os.path.normpath(image.image.path))

			#Delete physical files and finally DB entry
			if image.heatmap_link != "":
				heatmap_file_path = image_folder + "/heatmap/" + "heatmap_" + image.filename
				os.remove(heatmap_file_path)
			os.remove(image.image.path)
			image.delete()
	return redirect(requestpage)										#Return to previous page(predict/) when done


##Predict function
def predict(request):
	requestpage = request.POST.get('predict_requestpage', '/') #Store requestor page for reference when redirecting later
	all_objects= ImagePool.objects.all()
	model=load_model('./model/vgg_model.h5')
	for object in all_objects:
		if object.heatmap_link == "":
			img = tf.keras.preprocessing.image.load_img(object.image.path, target_size=(224, 224))
			img_array = tf.keras.preprocessing.image.img_to_array(img)
			norm_array = img_array/255
			processed_img = norm_array.reshape(1,224,224,3)
			z=model.predict(processed_img)
			covid_prob = (1 - z[0][0]) * 100 
			percent_value = round(covid_prob,2)
			object.risk = percent_value
            #heatmap
			generate_heatmap(object, z, model, processed_img)

	return redirect(requestpage) #Return to previous page(predict/) when done

def generate_heatmap(object, probability, model, input_image):
	original = cv2.imread(object.image.path)
	orig = cv2.cvtColor(original , cv2.COLOR_BGR2RGB)		
	i = np.argmax(probability[0])
	cam = gc(model=model, classIdx=i, layerName='block5_conv3')
	heatmap = cam.compute_heatmap(input_image)#show the calculated heatmap
	heatmapY = cv2.resize(heatmap, (orig.shape[1], orig.shape[0]))
	heatmapY = cv2.applyColorMap(heatmapY, cv2.COLORMAP_HOT)  
	imageY = cv2.addWeighted(heatmapY, 0.5, original, 1.0, 0)

	# draw the orignal x-ray, the heatmap, and the overlay together
	output = np.hstack([orig, heatmapY, imageY])
	fig, ax = plt.subplots(figsize=(20, 18))
	ax.imshow(np.random.rand(1, 99), interpolation='nearest')
	plt.imshow(output)

	heatmap_href = "media/images/heatmap/" + "heatmap_" + object.filename
	plt.savefig(heatmap_href)
	object.heatmap_link = heatmap_href
	object.save()

def view_guide(request):
	return render(request, 'main/guide.html')

def export(request):
	response = HttpResponse(content_type='text/csv')
	writer = csv.writer(response)
	writer.writerow(['filename','risk'])
	
	for object in ImagePool.objects.all().values_list('filename','risk', 'heatmap_link'):
		print(object)
		#If heatmap link is not empty, aka predictions were ran
		if object[2] != "":
			writer.writerow(object[0:2])
	
	response['Content-Disposition']= 'attachment;filename="predictions.csv" '

	return response