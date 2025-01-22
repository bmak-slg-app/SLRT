from motion_gen import GuassianBlur
import numpy as np
import torch
import utils
import smplx
import pickle
import time
import os
from cmd_parser import parse_config
from sign_connector_train import MLP
from human_body_prior.tools.model_loader import load_model
from human_body_prior.models.vposer_model import VPoser
from pydantic import BaseModel
from collections.abc import Callable

class GenerateResult(BaseModel):
    gloss: str
    gloss_frame_mapping: list[list[int]]

dtype = torch.float32

class Spoken2SignService:
    def __init__(self, config_file: str = "cfg_files/fit_smplsignx_tvb.yaml"):
        args = parse_config(['--config', config_file])
        use_cuda = args.get('use_cuda', True)
        if use_cuda and torch.cuda.is_available():
            device = torch.device('cuda')
        else:
            device = torch.device('cpu')
        self.device = device

        mapping = utils.smpl_to_openpose(model_type='smplx', use_hands=True, use_face=True,
                                            use_face_contour=False, openpose_format='coco25')
        joint_mapper = utils.JointMapper(mapping)

        model_params = dict(model_path=args.get('model_folder'),
                            joint_mapper=joint_mapper,
                            create_global_orient=True,
                            create_body_pose=not args.get('use_vposer'),
                            create_betas=True,
                            create_left_hand_pose=True,
                            create_right_hand_pose=True,
                            create_expression=True,
                            create_jaw_pose=True,
                            create_leye_pose=True,
                            create_reye_pose=True,
                            create_transl=False,
                            dtype=dtype,
                            **args)

        model_type = args.get('model_type', 'smplx')
        self.model_type = model_type
        print('[INFO] Model type:', model_type)
        print('[INFO] Model folder: ',args.get('model_folder'))

        #-------------------------------------------------------Sign Connector---------------------------------------
        joint_idx = np.array([3, 4, 6, 7, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36,
                                37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52,
                                53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66], dtype=np.int32)
        model = smplx.create(**model_params)
        model = model.to(device=device)
        self.model = model
        sign_connector = MLP(input_dim=len(joint_idx)*3*2+len(joint_idx))
        sign_connector.load_state_dict(torch.load('../../data/connector_tvb_ep258.pth', map_location='cuda:0'), strict=True)
        sign_connector.to(device)
        sign_connector.eval()
        self.sign_connector = sign_connector

        batch_size = args.get('batch_size', 1)
        use_vposer = args.get('use_vposer', True)
        self.use_vposer = use_vposer
        vposer, pose_embedding = [None, ] * 2
        vposer_ckpt = args.get('vposer_ckpt', '')
        if use_vposer:
            pose_embedding = torch.zeros([batch_size, 32],
                                            dtype=dtype, device=device,
                                            requires_grad=True)
            vposer_ckpt = os.path.expandvars(vposer_ckpt)
            # https://github.com/vchoutas/smplify-x/issues/144
            vposer, _ = load_model(vposer_ckpt, model_code=VPoser, remove_words_in_model_weights='vp_model.', disable_grad=True)
            vposer = vposer.to(device=device)
            vposer.eval()
            self.vposer = vposer
        self.pose_embedding = pose_embedding

        #-----------------------------------------------------Prepare Dict---------------------------------------------
        with open('../../data/tvb_all.pkl', 'rb') as f:
            render_results_all = pickle.load(f)
        self.render_results_all = render_results_all
        gloss2items_path = '../../data/gloss2items.pkl'
        with open(gloss2items_path, 'rb') as f:
            gloss2items = pickle.load(f)
        self.gloss2items = gloss2items

    def generate(self, task_id: int, gloss: str) -> GenerateResult:
        video_id = f"custom-input-{task_id}"
        glosses_translated = gloss.split()
        clips = []
        for gloss in glosses_translated:
            if gloss not in self.gloss2items:
                # print(gloss, 'not in dict')  #maybe due to smplified-to-traditional conversion. rare cases.
                continue
            clips.append(self.gloss2items[gloss][0][0])

        est_params_all = []
        inter_flag = []
        gloss_frame_mapping = []
        for id_idx in range(len(clips)):
            render_id = clips[id_idx]['video_file']
            render_results = self.render_results_all[render_id]
            start_frame = len(est_params_all)

            for pkl_idx in range(clips[id_idx]['start'], clips[id_idx]['end']):
                data = render_results[pkl_idx]
                est_params = {}
                for key, val in data[0]['result'].items():
                    if key in ['body_pose', 'left_hand_pose', 'right_hand_pose']:
                        est_params[key] = data[1][key + '_rot']
                    else:
                        est_params[key] = val
                est_params_all.append(est_params)
                inter_flag.append(False)
            end_frame = len(est_params_all)
            gloss_frame_mapping.append([start_frame, end_frame])

            if id_idx != len(clips)-1:
                clip_pre, clip_nex = clips[id_idx], clips[id_idx+1]
                data_0, data_1 = render_results[clip_pre['end']-1], self.render_results_all[clip_nex['video_file']][clip_nex['start']]

                est_params_pre = {}
                est_params_nex = {}
                for key, val in data_1[0]['result'].items():
                    est_params_pre[key] = val
                    est_params_nex[key] = val

                for k in range(2):
                    est_params = {}
                    if self.use_vposer:
                        with torch.no_grad():
                            if k == 0:
                                self.pose_embedding[:] = torch.tensor(
                                    est_params_pre['body_pose'], device=self.device, dtype=dtype)
                            else:
                                self.pose_embedding[:] = torch.tensor(
                                    est_params_nex['body_pose'], device=self.device, dtype=dtype)

                    for key, val in data[0]['result'].items():
                        if key == 'body_pose' and self.use_vposer:
                            # https://github.com/vchoutas/smplify-x/issues/144
                            body_pose = (self.vposer.decode(self.pose_embedding).get( 'pose_body')).reshape(1, -1) if self.use_vposer else None
                            if self.model_type == 'smpl':
                                wrist_pose = torch.zeros([body_pose.shape[0], 6],
                                                        dtype=body_pose.dtype,
                                                        device=body_pose.device)
                                body_pose = torch.cat([body_pose, wrist_pose], dim=1)
                            est_params['body_pose'] = body_pose
                        # elif key == 'betas':
                        #     est_params[key] = torch.zeros([1, 10], dtype=dtype, device=device)
                        # elif key == 'global_orient':
                        #     est_params[key] = torch.zeros([1, 3], dtype=dtype, device=device)
                        else:
                            if k == 0:
                                est_params[key] = torch.tensor(est_params_pre[key], dtype=dtype, device=self.device)
                            else:
                                est_params[key] = torch.tensor(est_params_nex[key], dtype=dtype, device=self.device)
                    model_output = self.model(**est_params)
                    joints_location = model_output.joints
                    joints_idx = torch.tensor([3, 4, 6, 7, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36,
                                            37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52,
                                            53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66],
                                            dtype=torch.int32).to(device=self.device)
                    joints_location = torch.index_select(joints_location, 1, joints_idx)
                    if k == 0:
                        joints_location_pre = joints_location
                    else:
                        joints_location_nex = joints_location
                joints_dis = torch.sqrt(((joints_location_pre-joints_location_nex)**2).sum(dim=-1))
                joints_location_pre = joints_location_pre.reshape([1,-1])
                joints_location_nex = joints_location_nex.reshape([1,-1])
                # print(joints_location_pre.shape, joints_location_nex.shape, joints_dis.shape)

                len_inter = self.sign_connector(torch.cat((joints_location_pre, joints_location_nex, joints_dis), 1))
                len_inter = max(round(len_inter.item()),1)
                # print(len_inter)

                weights = np.zeros(len_inter)
                interval = 1.0/(len_inter+1)
                for i in range(len_inter):
                    weights[i] = 1.0-(i+1)*interval
                for idx_w, weight in enumerate(weights):
                    est_params = {}
                    for key, val in data_0[0]['result'].items():
                        if key in ['body_pose', 'left_hand_pose', 'right_hand_pose']:
                            est_params[key] = weight*data_0[1][key + '_rot'] +(1-weight)*data_1[1][key + '_rot']
                        else:
                            est_params[key] = weight*data_0[0]['result'][key] + (1-weight)*data_1[0]['result'][key]
                    est_params_all.append(est_params)
                    inter_flag.append(True)

        for key, val in data[0]['result'].items():
            if key == 'camera_rotation':
                date_temp = np.zeros([len(est_params_all), 1, 9])
                for i in range(len(est_params_all)):
                    date_temp[i] = est_params_all[i][key].reshape(1, 9)
                GuassianBlur_ = GuassianBlur(1)
                out_smooth = GuassianBlur_.guassian_blur(date_temp, flag=0)
                for i in range(len(est_params_all)):
                    est_params_all[i][key] = out_smooth[i].reshape(1, 3, 3)
            elif key == 'betas':
                for i in range(len(est_params_all)):
                    est_params_all[i][key] = np.asarray([[0.421,-1.658,0.361,0.314,0.226,0.065,0.175,-0.150,-0.097,-0.191]])
            elif key == 'global_orient':
                for i in range(len(est_params_all)):
                    est_params_all[i][key] = np.asarray([[0,0,0]])
            else:
                date_temp = np.zeros([len(est_params_all), 1, est_params_all[0][key].shape[1]])
                for i in range(len(est_params_all)):
                    date_temp[i] = est_params_all[i][key]
                GuassianBlur_ = GuassianBlur(1)
                out_smooth = GuassianBlur_.guassian_blur(date_temp, flag=0)
                for i in range(len(est_params_all)):
                    est_params_all[i][key] = out_smooth[i]

        save_dir = os.path.join('./motions', video_id)
        os.makedirs(save_dir, exist_ok=True)

        for i in range(len(est_params_all)):
            est_params_all[i]['body_pose'][:, 0:15] = 0.
            est_params_all[i]['body_pose'][:, 18:24] = 0.
            est_params_all[i]['body_pose'][:, 27:33] = 0.
            fname = os.path.join(save_dir, str(i).zfill(3)+'.pkl')
            if inter_flag[i]:
                fname = os.path.join(save_dir, str(i).zfill(3)+'_inter.pkl')
            with open(fname, 'wb') as f:
                pickle.dump(est_params_all[i], f)

        return GenerateResult(
            gloss=gloss,
            gloss_frame_mapping=gloss_frame_mapping,
        )

frame_rate = 12

import bpy
import ffmpeg
import math

def format_timestamp(time: float):
    milliseconds = math.floor((time - math.floor(time)) * 1000)
    seconds = math.floor(time)
    minutes = seconds // 60
    seconds %= 60
    hours = minutes // 60
    minutes %= 60
    formatted_time = f"{hours:02d}:{minutes:02d}:{seconds:02d},{milliseconds:03d}"
    return formatted_time

class RenderAvatarService:
    def __init__(
            self,
            blender_addon_path: str = '../pretrained_models/smplx_blender_addon_300_20220623.zip',
            blender_mainfile: str = '../pretrained_models/smplx_tommy.blend',
            smplx_model_object: str = 'SMPLX-female',
            motions_dir: str = './motions',
            images_dir: str = './images',
            videos_dir: str = './videos',
            ):
        bpy.ops.preferences.addon_install(filepath = blender_addon_path, overwrite = True)
        bpy.ops.preferences.addon_enable(module = 'smplx_blender_addon')
        bpy.ops.wm.save_userpref()
        bpy.ops.wm.open_mainfile(filepath=blender_mainfile)

        path = os.path.abspath('../pretrained_models/smplx_blender_addon/data')
        bpy.ops.file.find_missing_files(directory=path)

        bpy.data.scenes['Scene'].render.resolution_y = 512
        bpy.data.scenes['Scene'].render.resolution_x = 512
        bpy.data.scenes['Scene'].render.fps = 60
        # bpy.data.objects["Camera"].location[0] = -0.02
        bpy.data.objects["Camera"].location[1] = -0.85 #-0.725
        bpy.data.objects["Camera"].location[2] = 0.155
        bpy.context.scene.render.image_settings.file_format = 'PNG'
        bpy.context.scene.render.image_settings.color_mode = 'RGBA'

        bpy.context.view_layer.objects.active = bpy.data.objects[smplx_model_object]

        self.motions_dir = motions_dir
        self.videos_dir = videos_dir
        self.images_dir = images_dir
        self.smplx_model_object = smplx_model_object

    def generate_subtitles(self, task_id: str, gloss: str, gloss_frame_mapping: list[int], karaoke: bool = True):
        video_id = f"custom-input-{task_id}"
        subtitle_fname = os.path.join(self.videos_dir, f"{video_id}.srt")
        video_dir = os.path.dirname(subtitle_fname)
        if not os.path.exists(video_dir): # video id may contains slash
            os.makedirs(video_dir)
        gloss_lst = gloss.split()
        with open(subtitle_fname, "w") as subtitle_file:
            for i, (gloss_item, frame_map) in enumerate(zip(gloss_lst, gloss_frame_mapping)):
                subtitle_file.write(f"{i + 1}\n")
                start_frame, end_frame = frame_map
                if karaoke:
                    if i + 1 < len(gloss_frame_mapping):
                        # subtitle should be continuous in karaoke mode
                        end_frame = gloss_frame_mapping[i + 1][0]
                start_time = start_frame / frame_rate
                end_time = end_frame / frame_rate
                subtitle_file.write(f"{format_timestamp(start_time)} --> {format_timestamp(end_time)}\n")
                if karaoke:
                    subtitle_file.write(f"<font color=\"#00ffff\">{' '.join(gloss_lst[:i+1])}</font>")
                    subtitle_file.write(f"{' ' if i + 1 < len(gloss_lst) else ''}{' '.join(gloss_lst[i+1:])}")
                    subtitle_file.write("\n\n")
                else:
                    subtitle_file.write(f"{gloss_item}\n\n")
        print(f"[INFO] subtitle for {video_id} generated at {subtitle_fname}")

    def render_video(self, task_id: str):
        video_id = f"custom-input-{task_id}"
        motion_path = os.path.join(self.motions_dir, video_id)
        motion_lst = os.listdir(motion_path)
        motion_lst.sort()

        current_frame = 0
        for i in range(len(motion_lst)):
            fname = os.path.join(motion_path, motion_lst[i])
            bpy.context.view_layer.objects.active = bpy.data.objects[self.smplx_model_object]
            bpy.ops.object.smplx_load_pose(filepath=fname)
            bpy.context.view_layer.objects.active = bpy.data.objects[self.smplx_model_object]
            bpy.ops.object.posemode_toggle()
            bpy.ops.object.mode_set(mode='POSE')
            bpy.ops.pose.select_all(action='SELECT')
            bpy.ops.anim.keyframe_insert()
            current_frame += 5

        vid_fname = os.path.join(self.videos_dir, f"{video_id}.mp4")
        video_dir = os.path.dirname(vid_fname)
        if not os.path.exists(video_dir):
            os.makedirs(video_dir)
        bpy.data.scenes['Scene'].render.filepath = vid_fname
        bpy.data.scenes["Scene"].frame_end = current_frame
        bpy.ops.render.render(animation=True)

    def render_images(self, task_id: str, progress_callback: Callable[[float], None]=None):
        video_id = f"custom-input-{task_id}"
        motion_path = os.path.join(self.motions_dir, video_id)
        motion_lst = os.listdir(motion_path)
        motion_lst.sort()

        img_dir = os.path.join(self.images_dir, video_id)
        os.makedirs(img_dir, exist_ok=True)

        for i in range(len(motion_lst)):
            fname = os.path.join(motion_path, motion_lst[i])
            bpy.context.view_layer.objects.active = bpy.data.objects[self.smplx_model_object]
            bpy.ops.object.smplx_load_pose(filepath=fname)
            bpy.ops.render.render()
            bpy.data.images["Render Result"].save_render(os.path.join(img_dir, f"{i:03}.png"))
            if progress_callback is not None:
                progress_callback((i + 1) / len(motion_lst))

        vid_fname = os.path.join(self.videos_dir, f"{video_id}.mp4")
        subtitle_fname = os.path.join(self.videos_dir, f"{video_id}.srt")
        video_dir = os.path.dirname(vid_fname)
        if not os.path.exists(video_dir):
            os.makedirs(video_dir)
        ffmpeg.input(os.path.join(img_dir, "*.png"), pattern_type='glob', framerate=frame_rate).output(vid_fname, vf=f"subtitles={subtitle_fname}:force_style='FontSize=24'", pix_fmt="yuv420p").run()
