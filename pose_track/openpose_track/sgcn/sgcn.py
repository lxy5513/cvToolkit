import numpy as np
import sys, os
import argparse
import torch
main_path = os.path.split(os.path.abspath(__file__))[0]
sys.path.insert(0, main_path+"/graph/")
from gcn_utils.io import IO
from gcn_utils.gcn_model import Model
from gcn_utils.processor_siamese_gcn import SGCN_Processor

class Pose_Matcher(SGCN_Processor):
    def __init__(self, argv=None):
        self.load_arg(argv)
        self.init_environment()
        self.load_model()
        self.load_weights()
        self.gpu()
        return

    @staticmethod
    def get_parser(add_help=False):
        parent_parser = IO.get_parser(add_help=False)
        parser = argparse.ArgumentParser(
            add_help=False,
            parents=[parent_parser],
            description='Graph Convolution Network for Pose Matching')

        parser.set_defaults(config=main_path+'/graph/config/inference.yaml')
        return parser


    def inference(self, data_1, data_2):
        self.model.eval()

        with torch.no_grad():
            data_1 = torch.from_numpy(data_1)
            data_1 = data_1.unsqueeze(0)
            data_1 = data_1.float().to(self.dev)

            data_2 = torch.from_numpy(data_2)
            data_2 = data_2.unsqueeze(0)
            data_2 = data_2.float().to(self.dev)

            feature_1, feature_2 = self.model.forward(data_1, data_2)

        # euclidian distance
        diff = feature_1 - feature_2
        dist_sq = torch.sum(pow(diff, 2), 1)
        dist = torch.sqrt(dist_sq)

        margin = 0.2
        distance = dist.data.cpu().numpy()[0]
        print("_____ Pose Matching: [dist: {:04.2f}]". format(distance))
        if dist >= margin:
            return False, distance  # Do not match
        else:
            return True, distance # Match

pose_matcher = Pose_Matcher()
def pose_matching(graph_A_data, graph_B_data):
    flag_match, dist = pose_matcher.inference(graph_A_data, graph_B_data)
    return flag_match, dist


if __name__ == "__main__":
    B = [[[[  9.],[ 19.],[ 26.],[ 44.],[ 49.],[ 52.],[ 37.],[ 26.],[ 24.],[ 49.],[ 62.],[ 72.],[ 37.],[ 38.],[ 37.]]],[[[150.],[123.],[ 95.],[ 92.],[119.],[150.],[ 61.],[ 68.],[ 49.],[ 50.],[ 68.],[ 70.],[ 44.],[ 37.],[ 31.]]]]
    A = [[[[ 28.],[ 56.],[ 73.],[ 57.],[ 74.],[ 78.],[ 55.],[ 84.],[ 88.],[ 59.],[ 50.],[ 43.],[ 74.],[ 74.],[ 74.]]],[[[143.],[129.],[ 91.],[ 88.],[120.],[156.],[ 67.],[ 66.],[ 47.],[ 45.],[ 68.],[ 70.],[ 37.],[ 27.],[ 18.]]]]

    A=np.array(A)
    B=np.array(B)
    pose_matching(A,B)

    from tqdm import tqdm 
    for i in tqdm(range(1000)):
        pose_matching(A,B)
