import os
import numpy as np
import torch
from PIL import Image
import matplotlib.pyplot as plt
import argparse

from torchvision import transforms
from fastreid.engine.utils import GradCAM, show_cam_on_image
import cv2
from skimage import io, transform


def vis(model):

    image_dir = "demo"
    image_name_idx = 9  # test9.jpg idx=9
    image_name = "test{}.jpg".format(image_name_idx)
    model = model
    state_dict = torch.load("./logs/veri/1003_ablation_erase95_dhg_gn/model_best.pth", map_location=torch.device('cpu'))
    model.load_state_dict(state_dict['model'], strict=False)
    # xx = torch.randn(1,3,256,256).to('cuda:0')
    # with torch.no_grad():
    #     torch.onnx.export(
    #         model,
    #         xx,
    #         'baseline_gn.onnx',
    #         opset_version=11,
    #         input_names=["input"],
    #         output_names=["output"]
    #     )
    # print('success')
    # ii = 0
    # assert ii > 1
    # target_layers = [model.backbone.layer4[-1]]
    target_layers = [model.transformer_gcn_dhg.identity]

    # model = models.resnet34(pretrained=True)
    # target_layers = [model.layer4]

    # load image
    # img_path = "demo/test9.jpg"
    img_path = os.path.join(image_dir, image_name)
    assert os.path.exists(img_path), "file: '{}' dose not exist.".format(img_path)
    # img = Image.open(img_path).convert('RGB')
    # img = np.array(img, dtype=np.uint8)
    original_image = cv2.imread(img_path, 1)
    # img = np.float32(rgb_img)
    original_image = original_image[:, :, ::-1]
    # Apply pre-processing to image.
    image = cv2.resize(original_image, tuple([256, 256][::-1]), interpolation=cv2.INTER_CUBIC)
    plt.imshow(image)
    plt.savefig(os.path.join('figure', 'figure{}.png'.format(image_name_idx)), bbox_inches='tight', dpi=800,
                pad_inches=0.0)
    # plt.show()
    plt.close()
    # Make shape with a new batch dimension which is adapted for
    # network input
    # [N, C, H, W]
    input_tensor = torch.as_tensor(image.astype("float32").transpose(2, 0, 1))[None]  # 1 3 256 256
    cam = GradCAM(model=model, target_layers=target_layers, use_cuda=True)
    target_category = image_name_idx
    # target_category = 9  # tabby, tabby cat
    # target_category = 254  # pug, pug-dog

    grayscale_cam = cam(input_tensor=input_tensor, target_category=target_category)

    grayscale_cam = grayscale_cam[0, :]

    visualization = show_cam_on_image(image.astype(dtype=np.float32) / 255.,
                                      grayscale_cam,
                                      use_rgb=True)
    # 隐藏坐标轴
    ax = plt.gca()
    ax.axes.xaxis.set_visible(False)
    ax.axes.yaxis.set_visible(False)
    plt.imshow(visualization)
    plt.savefig(os.path.join('figure', 'figure{}_1.png'.format(image_name_idx)), bbox_inches='tight', dpi=800,
                pad_inches=0.0)
    # plt.show()
    plt.close()
