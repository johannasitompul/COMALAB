from django import forms
from django.contrib.auth.forms import UserCreationForm
from django.contrib.auth.models import User

class CreateNewPool(forms.Form):
    uploaded_image = forms.ImageField()



