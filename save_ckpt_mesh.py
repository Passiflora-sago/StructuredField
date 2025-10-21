#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import torch
from scene import Scene,TetScene
import os,sys
from tqdm import tqdm
from os import makedirs
from gaussian_renderer import render, render_tet
import torchvision
from utils.general_utils import safe_state
from argparse import ArgumentParser
from arguments import ModelParams, PipelineParams, get_combined_args, OptimizationParams
from gaussian_renderer import GaussianModel,TetrahedraModel
import numpy as np


def render_set(model_path, name, iteration, views, gaussians, pipeline, background):
    render_path = os.path.join(name, "ours_{}".format(iteration), "renders")
    renderw_path = os.path.join(name, "ours_{}".format(iteration), "renders_w")
    gts_path = os.path.join(name, "ours_{}".format(iteration), "gt")
    makedirs(name, exist_ok=True)
    makedirs(render_path, exist_ok=True)
    makedirs(renderw_path, exist_ok=True)
    makedirs(gts_path, exist_ok=True)

    for idx, view in enumerate(tqdm(views, desc="Rendering progress")):
        rendering = render_tet(view, gaussians, pipeline, background)["render"]
        rendering2 = render_tet(view, gaussians, pipeline, torch.tensor([1,1,1], dtype=torch.float32, device="cuda"))["render"]
        gt = view.original_image[0:3, :, :]
        torchvision.utils.save_image(rendering, os.path.join(render_path, '{0:05d}'.format(idx) + ".png"))
        torchvision.utils.save_image(rendering2, os.path.join(renderw_path, '{0:05d}'.format(idx) + ".png"))
        torchvision.utils.save_image(gt, os.path.join(gts_path, '{0:05d}'.format(idx) + ".png"))




def render_sets(dataset : ModelParams, pipeline : PipelineParams, out_path = "/data/kevin/output/43a524d6-3/chkpnt30000.pth"):
    with torch.no_grad():
        tets = TetrahedraModel(sh_degree=3)
        # scene = TetScene(dataset, tets)
        model_path = os.path.join(out_path, "chkpnt2000.pth")
        msh_path = os.path.join(out_path, "chkpnt2000.msh")
        model = torch.load(model_path,map_location="cuda:0")
        tets.restore(model[0],None)
        save_as_meshio(tets,filename=msh_path)
        # render_set(dataset.model_path, output_dir, scene.loaded_iter, scene.getTestCameras(), tets, pipeline, background)
     
   
def save_point_cloud(tets):
    import open3d as o3d
    import numpy as np
    points = tets._xyz.detach().cpu().numpy()
       
    point_cloud = o3d.geometry.PointCloud()
    point_cloud.points = o3d.utility.Vector3dVector(points)


    o3d.io.write_point_cloud("output.ply", point_cloud)

    print("Point cloud saved as output.ply")

def save_as_meshio(tets, filename='/data/kevin/gaussian-splatting/nerf_synthetic/lego/points3d.msh'):

    tetrahedrons = tets._cells.detach().cpu().numpy()
    vertices = tets._xyz.detach().cpu().numpy()
    tetrahedrons = tetrahedrons   # 由 1-based 索引调整为 0-based
    import meshio

    mesh = meshio.Mesh(
        points=vertices,
        cells=[("tetra", tetrahedrons)]
    )
    

    mesh.write(filename, file_format="gmsh") 
    print(f"saved  {filename}")


if __name__ == "__main__":
    # Set up command line argument parser
    
    
    
    parser = ArgumentParser(description="Testing script parameters")
    
    parser.add_argument('--input_model', type=str, default="/home/juyonggroup/kevin2000/Desktop/gaussian-splatting/output/materials/", help='Path to the input model file')
    parser.add_argument('--output_dir', type=str, default="./output/render/mesh_output", help='Directory to save the output')

    model = ModelParams(parser)
    op = OptimizationParams(parser)
    pipeline = PipelineParams(parser)
    args = parser.parse_args(sys.argv[1:])
    


    render_sets(model.extract(args), pipeline.extract(args),out_path=args.input_model)