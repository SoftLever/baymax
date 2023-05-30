from django.urls import path
from inference.views import wavUpload

urlpatterns = [
    path('', wavUpload),
]
