from rest_framework.parsers import FileUploadParser
from rest_framework.response import Response
from rest_framework.views import APIView
from rest_framework import status

from .serializers import FileSerializer

from PIL import Image
import numpy as np
# from PIL import Image
import cv2


class FileUploadView(APIView):
    parser_class = (FileUploadParser,)

    def post(self, request, *args, **kwargs):
        
        file_serializer = FileSerializer(data=request.data)
        file = request.data.get('file')
        decoded_image = cv2.imdecode(np.frombuffer(file.read(), np.uint8), -1)
        if file_serializer.is_valid():
            file_serializer.save()
            return Response(file_serializer.data, status=status.HTTP_201_CREATED)
        else:
            return Response(file_serializer.errors, status=status.HTTP_400_BAD_REQUEST)

# curl -F "file=@/home/danjiii/Рабочий стол/photo_2020-02-05_14-25-31.jpg" http://127.0.0.1:8000/upload/
