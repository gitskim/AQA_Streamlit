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
from models.C3D_model import C3D
import streamlit_analytics
from opts import *
import numpy as np
import streamlit as st
import os
import cv2 as cv
import tempfile
from torchvision import transforms
import boto3
import urllib
from htbuilder import HtmlElement, div, ul, li, br, hr, a, p, img, styles, classes, fonts
from htbuilder.units import percent, px

torch.manual_seed(randomseed)
torch.cuda.manual_seed_all(randomseed)
random.seed(randomseed)
np.random.seed(randomseed)
torch.backends.cudnn.deterministic = True

current_path = os.path.abspath(os.getcwd())
m1_path = os.path.join(current_path, m1_path)
m2_path = os.path.join(current_path, m2_path)
m3_path = os.path.join(current_path, m3_path)
c3d_path = os.path.join(current_path, c3d_path)

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


def action_classifier(frames):
    # C3D raw
    model_C3D = C3D()
    model_C3D.load_state_dict(torch.load(c3d_path, map_location={'cuda:0': 'cpu'}))
    model_C3D.eval()

    with torch.no_grad():
        X = torch.zeros((1, 3, 16, 112, 112))
        frames2keep = np.linspace(0, frames.shape[2] - 1, 16, dtype=int)
        ctr = 0
        for i in frames2keep:
            X[:, :, ctr, :, :] = frames[:, :, i, :, :]
            ctr += 1
        print('X shape: ', X.shape)

        # modifying
        model_C3D.eval()

        # perform prediction
        X = X*255
        X = torch.flip(X, [1])
        prediction = model_C3D(X)
        prediction = prediction.data.cpu().numpy()

        # print top predictions
        top_inds = prediction[0].argsort()[::-1][:5]  # reverse sort and take five largest items
        print('\nTop 5:')
        print('Top inds: ', top_inds)
    return top_inds[0]


def preprocess_one_video(video_file):
    if video_file != "sample":
        tfile = tempfile.NamedTemporaryFile(delete=False)
        tfile.write(video_file.read())

        vf = cv.VideoCapture(tfile.name)
    else:
        vf = cv.VideoCapture("054.avi")

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
            frames = np.vstack((frames, frame))
        else:
            frames = frame

    print('frames shape: ', frames.shape)

    vf.release()
    cv.destroyAllWindows()
    rem = len(frames) % 16
    rem = 16 - rem

    if rem != 0:
        padding = np.zeros((rem, C, H, H))
        frames = np.vstack((frames, padding))

    # frames shape: (137, 3, 112, 112)
    frames = torch.from_numpy(frames).unsqueeze(0)

    print(f"video shape: {frames.shape}") # video shape: torch.Size([1, 144, 3, 112, 112])
    frames = frames.transpose_(1, 2)
    frames = frames.double()
    return frames


def inference_with_one_video_frames(frames):
    action_class = action_classifier(frames)
    if action_class != 463:
        return None

    model_CNN = C3D_altered()
    model_CNN.load_state_dict(torch.load(m1_path, map_location={'cuda:0': 'cpu'}))

    # loading our fc6 layer
    model_my_fc6 = my_fc6()
    model_my_fc6.load_state_dict(torch.load(m2_path, map_location={'cuda:0': 'cpu'}))

    # loading our score regressor
    model_score_regressor = score_regressor()
    model_score_regressor.load_state_dict(torch.load(m3_path, map_location={'cuda:0': 'cpu'}))
    with torch.no_grad():
        pred_scores = []

        model_CNN.eval()
        model_my_fc6.eval()
        model_score_regressor.eval()

        clip_feats = torch.Tensor([])
        print(f"frames shape: {frames.shape}")
        for i in np.arange(0, frames.shape[2], 16):
            clip = frames[:, :, i:i + 16, :, :]
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
    c3d_loaded = os.path.isfile(c3d_path)
    if cnn_loaded and fc6_loaded and c3d_loaded:
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
    if not c3d_loaded:
        urllib.request.urlretrieve(
            "http://imagelab.ing.unimore.it/files/c3d_pytorch/c3d.pickle",
            c3d_path
            )

    # urllib.request.urlretrieve(
    #         "https://aqa-diving.s3.us-west-2.amazonaws.com/{}".format(BUCKET_WEIGHT_CNN), m1_path)
    # urllib.request.urlretrieve(
    #         "https://aqa-diving.s3.us-west-2.amazonaws.com/{}".format(BUCKET_WEIGHT_FC6), m2_path)


def image(src_as_string, **style):
    return img(src=src_as_string, style=styles(**style))


def link(link, text, **style):
    return a(_href=link, _target="_blank", style=styles(**style))(text)


def layout(*args):

    style = """
    <style>
      footer {visibility: hidden;}
    </style>
    """

    style_div = styles(
        position="fixed",
        left=0,
        bottom=0,
        margin=px(0, 0, 0, 0),
        width=percent(100),
        color="pink",
        text_align="center",
        height="auto",
        opacity=1
    )

    body = p()
    foot = div(
        style=style_div
    )(
        body
    )

    st.markdown(style, unsafe_allow_html=True)

    for arg in args:
        if isinstance(arg, str):
            body(arg)

        elif isinstance(arg, HtmlElement):
            body(arg)

    st.markdown(str(foot), unsafe_allow_html=True)


def footer():
    myargs = [
        "Made with ❤️ by ",
        link("https://paritoshparmar.github.io/", "@Paritosh Parmar, "),
        link("https://www.linkedin.com/in/yanqing-dai-2001948a/", "@Yanqing Dai, "),
        link("https://www.linkedin.com/in/suhyundroid/", "@Suhyun Kim"),
        br(),
    ]
    layout(*myargs)


def make_prediction(video_file):
    if video_file is not None or video_file == "sample":
        # Display a message while perdicting
        val = 0
        res_img = st.empty()
        res_msg = st.empty()

        # Making prediction
        frames = preprocess_one_video(video_file)
        if frames.shape[2] > 400:
            res_msg.error("The uploaded video is too long.")
        else:
            preds = inference_with_one_video_frames(frames)
            if preds is None:
                res_img.empty()
                res_msg.error("The uploaded video does not seem to be a diving video.")
            else:
                val = int(preds[0] * 17)

                # Clear waiting messages and show results
                print(f"Predicted score after multiplication: {val}")
                res_img.empty()
                res_msg.success("Predicted score: {}".format(val))


if __name__ == '__main__':
    with st.spinner('Loading to welcome you...'):
        load_weights()
    with streamlit_analytics.track(unsafe_password=st.secrets["analytics"]):
        st.title("AI Olympics Judge")
        st.subheader("Upload Olympics diving video and check its AI predicted score")
        footer()

        video_file = st.file_uploader("Upload a video here", type=["mp4", "mov", "avi"])
        if video_file is None:
            st.subheader("Don't have Olympics diving videos? Try the sample video below.")
            diving_img = st.empty()
            if st.button("Sample Video"):
                diving_img.empty()
                diving_img.image(
                    "https://raw.githubusercontent.com/gitskim/AQA_Streamlit/main/054.gif",
                    width = 300)
                col2 = st.empty()
                col2.markdown("Actual Score: 84.15")
                col2_msg = st.empty()
                col2_msg.error("Please wait. Making predictions now...")
                make_prediction("sample")
                col2_msg.empty()

        else:
            # Display a message while perdicting
            val = 0
            res_img = st.empty()
            res_msg = st.empty()
            col1, col2, col3 = st.beta_columns([1,1,1])
            with col2:
                res_img.image(
                    "https://media.tenor.com/images/eab0c68ee47331c4b86d679633e6d7bc/tenor.gif",
                    width = 100)
                res_msg.markdown("### _Making Prediction now..._")

            # Making prediction
            frames = preprocess_one_video(video_file)
            if frames.shape[2] > 400:
                res_msg.error("The uploaded video is too long.")
            else:
                preds = inference_with_one_video_frames(frames)
                if preds is None:
                    res_img.empty()
                    res_msg.error("The uploaded video does not seem to be a diving video.")
                else:
                    val = int(preds[0] * 17)

                    # Clear waiting messages and show results
                    print(f"Predicted score after multiplication: {val}")
                    res_img.empty()
                    res_msg.success("Predicted score: {}".format(val))



