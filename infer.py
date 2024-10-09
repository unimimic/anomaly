from anomalib.deploy.inferencers import OpenVINOInferencer
from pathlib import Path
import cv2

output_path = Path('export')
image_path = 'datasets/hazelnut_toy/good/01.jpg'
image_path = 'datasets/hazelnut_toy/crack/01.jpg'

openvino_model_path = output_path / "weights" / "openvino" / "model.bin"
metadata = output_path / "weights" / "openvino" / "metadata.json"
print(openvino_model_path.exists(), metadata.exists())

inferencer = OpenVINOInferencer(
    path=openvino_model_path,  # Path to the OpenVINO IR model.
    metadata=metadata,  # Path to the metadata file.
    device="CPU",  # We would like to run it on an Intel CPU.
)

predictions = inferencer.predict(image=image_path)
print(predictions.pred_score, predictions.pred_label)

# Display the original image
cv2.imshow('Original Image', predictions.image)
cv2.waitKey(0)

# Display the anomaly map
cv2.imshow('Anomaly Map', predictions.anomaly_map)
cv2.waitKey(0)

# Display the heat map
cv2.imshow('Heat Map', predictions.heat_map)
cv2.waitKey(0)

# Display the prediction mask
cv2.imshow('Prediction Mask', predictions.pred_mask)
cv2.waitKey(0)

# Display the segmentations
cv2.imshow('Segmentations', predictions.segmentations)
cv2.waitKey(0)

# Close all OpenCV windows
cv2.destroyAllWindows()

# inferencer = TorchInferencer("export_folder/weights/torch/model.pt")
# image = torch.rand((3, 900, 900))
# prediction = inferencer.predict(image)
# prediction.pred_label
# print(prediction.pred_label)