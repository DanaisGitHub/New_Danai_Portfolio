---
title: AI Robot Perception
slug: ai-robot-perception
description: Using MaskRCNN 50, combine Object Recognition & Image Segmentation & PCA from 1 AI model 
labels: [Python, PyTorch, AI/ML, Computer Vision, Robot Perception]
github: https://github.com/DanaisGitHub/Robot_Perception
---

# 🎉 Mask R-CNN with PCA for Video Processing 📹

### Final Result


[Youtube Video 😀](https://youtu.be/tYTFxwDruac)

## 🚀 Load the Pre-trained Model

```python
model = torchvision.models.detection.maskrcnn_resnet50_fpn(pretrained=True)
model.eval().cpu()

```

Load the Mask R-CNN model pre-trained on COCO dataset and set it to evaluation mode. 🧠

## 🌈 Define COCO Object Names and Colours

```python
coco_names = ['unlabeled', 'person', 'bicycle', 'car', ...]
colors = [[random.randint(0, 255) for _ in range(3)] for _ in coco_names]

```

Define the object names and assign random colours for visualization. 🎨

## 🖌️ Apply Mask to Image

```python
def apply_mask(image, mask, color, alpha=0.5):
    for c in range(3):
        image[:, :, c] = np.where(mask == 1,
                                  image[:, :, c] *
                                  (1 - alpha) + alpha * color[c],
                                  image[:, :, c])
    return image

```

Apply the given mask to the image with transparency. 🖼️

## 📍 Calculate Mask Centre

```python
def mask_center(mask):
    m = mask.nonzero()
    x, y = np.mean(m[1]), np.mean(m[0])
    return int(x), int(y)

```

Calculate the centre of the binary mask using the centroid formula. 📏

## 📐 Compute Principal Axes with PCA

```python
def principal_axes(mask):
    points = np.column_stack(np.where(mask.transpose() > 0))
    pca = PCA(n_components=2)
    pca.fit(points)
    center = pca.mean_
    direction1 = pca.components_[0] * np.sqrt(pca.explained_variance_[0])
    direction2 = pca.components_[1] * np.sqrt(pca.explained_variance_[1])
    return center, direction1, direction2

```

Compute the principal axes of the mask using PCA. 🧮

## 🎥 Video Processing

### 📂 Specify Paths

```python
input_video_path = 'WIN_20240414_22_50_44_Pro.mp4'
output_video_path = '000000000000.mp4'

```

Specify the input and output video paths. 📁

### 🎬 Capture Video

```python
cap = cv2.VideoCapture(input_video_path)
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output_video_path, fourcc, 10.0, (frame_width, frame_height))

```

Capture video from the file and set up the video writer. 🎥

### 🔄 Process Each Frame

```python
try:
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        image_tensor = torchvision.transforms.functional.to_tensor(image).cpu()

        with torch.no_grad():
            output = model([image_tensor])[0]

        for box, label, mask, score in zip(output['boxes'], output['labels'], output['masks'], output['scores']):
            if score > 0.5:
                mask_array = mask[0].cpu().numpy() > 0.5
                center = mask_center(mask_array)
                mean, direction1, direction2 = principal_axes(mask_array)

                bbox_width = box[2] - box[0]
                bbox_height = box[3] - box[1]
                axis_length = min(bbox_width, bbox_height) / 1000

                color = random.choice(colors)
                frame = apply_mask(frame, mask_array, color)

                cv2.circle(frame, (int(center[0]), int(center[1])), 5, (255, 0, 0), -1)

                start_point1 = (int(mean[0] - direction1[0] * axis_length), int(mean[1] - direction1[1] * axis_length))
                end_point1 = (int(mean[0] + direction1[0] * axis_length), int(mean[1] + direction1[1] * axis_length))
                start_point2 = (int(mean[0] - direction2[0] * axis_length), int(mean[1] - direction2[1] * axis_length))
                end_point2 = (int(mean[0] + direction2[0] * axis_length), int(mean[1] + direction2[1] * axis_length))

                cv2.line(frame, start_point1, end_point1, (0, 255, 0), 2)
                cv2.line(frame, start_point2, end_point2, (0, 0, 255), 2)

                cv2.rectangle(frame, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), color, 2)
                label_name = coco_names[label]
                cv2.putText(frame, f'{label_name}: {score:.2f}', (int(box[0]), int(box[1]-5)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        out.write(frame)
finally:
    cap.release()
    out.release()

```

Process each frame, apply masks, draw bounding boxes, and save the output video. 🎞️

## **📏 Principal Axes Formula**

The principal axes are computed using PCA:

$$
center = pca.mean
\space \newline
direction1 = pca.components_{[0]} \times \sqrt{pca.explained_variance_{[0]}}
\newline
direction2 = pca.components_{[1]} \times \sqrt{pca.explained_variance_{[1]}}
$$