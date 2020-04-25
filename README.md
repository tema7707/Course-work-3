# Style
App for virtual fitting of clothing fitting.

## Main functionality:
This app is designed to show you how you will look in your chosen clothes.

[Demonstration](https://drive.google.com/file/d/15Cb37ovY1rqtocVPbV_EHjqrzC_HdXBi/view?usp=sharing)

| Clothes\Persons | ![Person1](./images/1/original.jpg) | ![Person2](./images/2/original.jpg) | ![Person3](./images/3/original.jpg) | ![Person4](./images/4/original.jpg) |
|---|---|---|---|---|
| ![Cloth1](./images/clothes/0.jpg) | ![Person1_0](./images/1/vans.jpg) | ![Person2_0](./images/2/vans.jpg) | ![Person3_0](./images/3/vans.jpg) | ![Person4_0](./images/4/vans.jpg) |
| ![Cloth1](./images/clothes/1.jpg) | ![Person1_1](./images/1/puma.jpg) | ![Person2_1](./images/2/puma.jpg) | ![Person3_1](./images/3/puma.jpg) | ![Person4_1](./images/4/puma.jpg) |
| ![Cloth1](./images/clothes/2.jpg) | ![Person1_2](./images/1/levice.jpg) | ![Person2_2](./images/2/levice.jpg) | ![Person3_2](./images/3/levice.jpg) | ![Person4_2](./images/4/levice.jpg) |

How does this happen:
1. You choose the type of clothing (t-shirt, pants, sweater, etc.)
2. You choose a specific clothing model
3. Upload a photo of a person
4. Get a generated photo of this person in the selected clothing


## Implementation
The app consists of two main parts an Android app and a REST API server.

### Android
Through the Android app, the user selects things to try on and uploads their photo. Then it sends data to the server and receives the image that has already been generated.

**Technologies used:** Android Studio, Java, and OkHttp.

### Server
The server processes the image. Communication with the client side is performed via the REST API. Several models are implemented here: segmentation of clothing, segmentation of the human head and body, definition of key points, and a model for image generation (GAN).

**Technologies used:** Flask, Python, PyTorch, Detectron2, DensePose, and Mask R-CNN.

### Segmentation
For clothing segmentation, we used a combination of Mask R-CNN and the GrabCut algorithm. Using a neural network, we get a clothing class, an approximate mask, and a bounding box. GrabCut makes the mask more accurate.

|Mask R-CNN|Aplying GrabCut|Result|
|---|---|---|
| <img src="./images/clothes_segmentation/mask.jpg" width="250"> | <img src="./images/clothes_segmentation/apply.jpg" width="250"> | <img src="./images/clothes_segmentation/grabcut.jpg" width="250"> |
