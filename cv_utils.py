import cv2
import numpy
from PIL import Image, ImageDraw, ImageFont
import os
def put_chinese(img_OpenCV, text='你好，世界', font_size=28, display=None):
    # 图像从OpenCV格式转换成PIL格式
    img_PIL = Image.fromarray(cv2.cvtColor(img_OpenCV, cv2.COLOR_BGR2RGB))

    # 查找指令locate *.ttc
    font = ImageFont.truetype('/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc', font_size)
    fillColor = (255,0,0)
    position = (10,10)
    string = text

    if not isinstance(string, str):
        string = string.decode('utf8')

    draw = ImageDraw.Draw(img_PIL)
    draw.text(position, string, font=font, fill=fillColor)

    img_OpenCV = cv2.cvtColor(numpy.asarray(img_PIL),cv2.COLOR_RGB2BGR)
    if display:
        cv2.imshow("print chinese to image",img_OpenCV)
        cv2.waitKey(2000)
    return img_OpenCV

def add_text(img, text, pos=(30,30), fontScale=0.8):
    font                   = cv2.FONT_HERSHEY_SIMPLEX
    bottomLeftCornerOfText = pos # (30,30)
    fontScale              = fontScale
    fontColor              = (2,200,2)
    lineType               = 4

    img = cv2.putText(img,text,
        bottomLeftCornerOfText,
        font,
        fontScale,
        fontColor,
        lineType)

    return img

def video_out(cap, img, save_name='result.mp4'):
    # out.write(img)
    H,W = img.shape[:2]
    output_fps = cap.get(cv2.CAP_PROP_FPS)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(save_name,fourcc, output_fps, (W,H))
    return out

def img_resize(image, max_length=640):
    H, W = image.shape[:2]
    if max(W, H) > max_length: #shrink
        interpolation = cv2.INTER_AREA
    else:
        interpolation = cv2.INTER_LINEAR


    if W>H:
        W_resize = max_length
        H_resize = int(H * max_length / W)
    else:
        H_resize = max_length
        W_resize = int(W * max_length / H)
    image = cv2.resize(image, (W_resize, H_resize), interpolation=interpolation)
    return image, W_resize, H_resize


if __name__ == '__main__':
    img_path = os.path.join(os.environ.get('CVTOOLBOX'), 'data/test.png')
    img = cv2.imread(img_path)
    put_chinese(img, '你好吗', display=1, font_size=25)
