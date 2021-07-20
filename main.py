# Author: Paritosh Parmar (https://github.com/ParitoshParmar)
# Code used in the following, also if you find it useful, please consider citing the following:
#
# @inproceedings{parmar2019and,
#   title={What and How Well You Performed? A Multitask Learning Approach to Action Quality Assessment},
#   author={Parmar, Paritosh and Tran Morris, Brendan},
#   booktitle={Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition},
#   pages={304--313},
#   year={2019}
# }

import torch
from torch.utils.data import DataLoader
import random
from models.C3D_altered import C3D_altered
from models.my_fc6 import my_fc6
from models.score_regressor import score_regressor
from opts import *
import numpy as np
import streamlit as st
import os
import cv2 as cv
import tempfile
from torchvision import transforms
import boto3
# import urllib

torch.manual_seed(randomseed)
torch.cuda.manual_seed_all(randomseed)
random.seed(randomseed)
np.random.seed(randomseed)
torch.backends.cudnn.deterministic = True

current_path = os.path.abspath(os.getcwd())
m1_path = os.path.join(current_path, m1_path)
m2_path = os.path.join(current_path, m2_path)
m3_path = os.path.join(current_path, m3_path)

transform = transforms.Compose([transforms.ToTensor(),
                                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

def center_crop(img, dim):
    """Returns center cropped image

    Args:Image Scaling
    img: image to be center cropped
    dim: dimensions (width, height) to be cropped from center
    """
    width, height = img.shape[1], img.shape[0]
    #process crop width and height for max available dimension
    crop_width = dim[0] if dim[0]<img.shape[1] else img.shape[1]
    crop_height = dim[1] if dim[1]<img.shape[0] else img.shape[0]
    mid_x, mid_y = int(width/2), int(height/2)
    cw2, ch2 = int(crop_width/2), int(crop_height/2)
    crop_img = img[mid_y-ch2:mid_y+ch2, mid_x-cw2:mid_x+cw2]
    return crop_img


def preprocess_one_video(video_file):
    tfile = tempfile.NamedTemporaryFile(delete=False)
    tfile.write(video_file.read())

    vf = cv.VideoCapture(tfile.name)

    # https: // discuss.streamlit.io / t / how - to - access - uploaded - video - in -streamlit - by - open - cv / 5831 / 8
    frames = None
    while vf.isOpened():
        ret, frame = vf.read()
        if not ret:
            break
        frame = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
        frame = cv.resize(frame, input_resize, interpolation=cv.INTER_LINEAR) #frame resized: (128, 171, 3)
        frame = center_crop(frame, (H, H))
        frame = transform(frame).unsqueeze(0)
        if frames is not None:
            frame = np.vstack((frames, frame))
            frames = frame
        else:
            frames = frame

    vf.release()
    cv.destroyAllWindows()
    rem = len(frames) % 16
    rem = 16 - rem

    if rem != 0:
        padding = np.zeros((rem, C, H, H))
        frames = np.vstack((frames, padding))

    frames = np.expand_dims(frames, axis=0)
    # frames shape: (137, 3, 112, 112)
    frames = DataLoader(frames, batch_size=test_batch_size, shuffle=False)
    return frames


def inference_with_one_video_frames(frames):
    model_CNN = C3D_altered()
    model_CNN.load_state_dict(torch.load(m1_path, map_location={'cuda:0': 'cpu'}))

    # loading our fc6 layer
    model_my_fc6 = my_fc6()
    model_my_fc6.load_state_dict(torch.load(m2_path, map_location={'cuda:0': 'cpu'}))

    # loading our score regressor
    model_score_regressor = score_regressor()
    model_score_regressor.load_state_dict(torch.load(m3_path, map_location={'cuda:0': 'cpu'}))
    with torch.no_grad():
        pred_scores = [];

        model_CNN.eval()
        model_my_fc6.eval()
        model_score_regressor.eval()

        for video in frames:
            video = video.transpose_(1, 2)
            video = video.double()
            clip_feats = torch.Tensor([])
            for i in np.arange(0, len(video), 16):
                clip = video[:, :, i:i + 16, :, :]
                model_CNN = model_CNN.double()
                clip_feats_temp = model_CNN(clip)

                # clip_feats_temp shape: torch.Size([1, 8192])

                clip_feats_temp.unsqueeze_(0)

                # clip_feats_temp unsqueeze shape: torch.Size([1, 1, 8192])

                clip_feats_temp.transpose_(0, 1)

                # clip_feats_temp transposes shape: torch.Size([1, 1, 8192])

                clip_feats = torch.cat((clip_feats.double(), clip_feats_temp), 1)

                # clip_feats shape: torch.Size([1, 1, 8192])

            clip_feats_avg = clip_feats.mean(1)


            model_my_fc6 = model_my_fc6.double()
            sample_feats_fc6 = model_my_fc6(clip_feats_avg)
            model_score_regressor = model_score_regressor.double()
            temp_final_score = model_score_regressor(sample_feats_fc6)
            pred_scores.extend([element[0] for element in temp_final_score.data.cpu().numpy()])

            return pred_scores

def load_weights():
    cnn_loaded = os.path.isfile(m1_path)
    fc6_loaded = os.path.isfile(m2_path)
    if cnn_loaded and fc6_loaded:
        return

    s3 = boto3.client(
            's3',
            aws_access_key_id = st.secrets["access_id"],
            aws_secret_access_key = st.secrets["access_key"]
            )
    if not cnn_loaded:
        s3.download_file(BUCKET_NAME, BUCKET_WEIGHT_CNN, m1_path)
    if not fc6_loaded:
        s3.download_file(BUCKET_NAME, BUCKET_WEIGHT_FC6, m2_path)

    # urllib.request.urlretrieve(
    #         "https://aqa-diving.s3.us-west-2.amazonaws.com/{}".format(BUCKET_WEIGHT_CNN), m1_path)
    # urllib.request.urlretrieve(
    #         "https://aqa-diving.s3.us-west-2.amazonaws.com/{}".format(BUCKET_WEIGHT_FC6), m2_path)

if __name__ == '__main__':
    st.title("Olympics diving !JUDGE")
    st.subheader("Upload Olympics diving video and check its predicted score.")
    st.markdown("---")
    video_file = st.file_uploader("Upload a video here", type=["mp4", "mov"])

    # transforms.CenterCrop(H),

    # Whenever there is a file uploaded
    if video_file is not None:
        # Load the model (if needed) when user actually wants to predict
        with st.spinner('Loading to welcome you...'):
            load_weights()

        # Display a message while perdicting
        val = 0
        res_img = st.empty()
        res_msg = st.empty()

        res_img.image(
            "https://media.tenor.com/images/eab0c68ee47331c4b86d679633e6d7bc/tenor.gif",
            width = 100)
        res_msg.markdown("### _Making Prediction now..._")

        # Makig prediction
        frames = preprocess_one_video(video_file)
        preds = inference_with_one_video_frames(frames)
        val = int(preds[0] * 17)

        # Clear waiting messages and show results
        print(f"Predicted score after multiplication: {val}")
        res_img.empty()
        res_msg.success("Predicted score: {}".format(val))
