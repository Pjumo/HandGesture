import gather_dataset_cam as cam

for i in range(20):
    path = f'./video_webcam/doubleclick_{i+1}.mp4'
    webcam = cam.WebCamData(path)
    print(i+1)
    webcam.split_video()
