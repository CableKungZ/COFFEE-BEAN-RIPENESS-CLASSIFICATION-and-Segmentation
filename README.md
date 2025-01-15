# Coffee Bean Ripeness Classification and Segmentation

This project aims to create an AI model deployed on the OKDO NVIDIA Nano Module to classify and segment coffee bean ripeness. The model predicts the positions of coffee beans in images and classifies them based on their colors.

## Dataset Sources

1. **Original Coffee Bean Images**: Sourced from [Zenodo](https://zenodo.org/records/14271151?fbclid=IwY2xjawHctJ1leHRuA2FlbQIxMAABHYNqZB-VkHU147VEcePLeGKpx5ARMJ4TjrNGzFbSavZUIFpdE5sNe2hYHA_aem_6_5OX2mR0CLd84wQ9VhT1g).
2. **Real Images**: Collected from [Phaiwan Farm](https://www.facebook.com/profile.php?id=100064945350451), Nakhon Si Thammarat, Thailand.

## Dataset Preparation

1. **Background Removal**:
   - Used [remove.bg](https://www.remove.bg/) to remove the background from images.
   - Saved and cropped the images to ensure uniformity.

2. **Synthetic Image Generation**:
   - Leveraged OpenCV (CV2) to randomly generate coffee bean placements on a whiteboard background of size 1280x1280 pixels.

3. **Segmentation and Annotation**:
   - Uploaded the dataset to [Roboflow](https://roboflow.com/).
   - Used the **Smart Polygon** annotation tool in Roboflow for precise segmentation.

## Model Training

- **Training Environment**: Google Colab
  - [Colab Notebook](https://colab.research.google.com/drive/1NS4xAejqud0e59tvqqcQzUdFUYZ8HSvp#scrollTo=BSd93ZJzZZKt)
- After training, download the `best.pt` file and place it in the `Model` directory.

## Roboflow Project

Access the dataset and project details on Roboflow:
[Coffee Cherry 3 400 Datasets](https://universe.roboflow.com/jassadakornsu/coffee-cherry-3-400-datasets)

## Deployment

The trained model is deployed on the OKDO NVIDIA Nano Module for real-time segmentation and classification of coffee beans. The AI identifies beans within the image and segments them based on ripeness color.

## Acknowledgements

- Dataset and annotations were enhanced using [remove.bg](https://www.remove.bg/) and Roboflow tools.
- Coffee bean images are a combination of publicly available datasets and locally collected images from Phaiwan Farm in Thailand.

---
For any inquiries or contributions, feel free to open an issue or submit a pull request.
