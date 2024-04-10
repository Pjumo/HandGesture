import gather_dataset_cam as cam

path = './video_webcam/click_10.mp4'
webcam = cam.WebCamData(path)
webcam.show_skeleton()
