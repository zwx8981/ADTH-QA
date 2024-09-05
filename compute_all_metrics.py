
import numpy as np
import torch
from PIL import Image
from tqdm import tqdm
import warnings
warnings.filterwarnings("ignore")

#load videos
import skvideo.io as skio

video_s = 'adthqa/source_videos/French_slice.mp4'
video_g = 'adthqa/dfanerf/French_French.mp4'
source_video = skio.vread(video_s)
generated_video = skio.vread(video_g)

##visual quality
import pyiqa
from pooling import pooling

nr_metric = 'liqe' # or unique, musiq

if nr_metric == 'liqe':
    model = pyiqa.create_metric('liqe_mix', as_loss=False)
elif nr_metric == 'unique':
    model = pyiqa.create_metric('unique', as_loss=False)
elif nr_metric == 'musiq':
    model = pyiqa.create_metric('musiq', as_loss=False)
else:
    raise 'Not recommended metrics!'

scores = []
for i in tqdm(range(len(generated_video))):
    frame = generated_video[i, ...]
    frame = Image.fromarray(frame)

    pred = model(frame)
    pred = pred.item()
    # score += pmos
    scores.append(pred)

pool_method = 'percentile'
visual_score = pooling(scores, pool_method)

print('Visual quality score ({}) of {} is {} (Higher better)'.format(nr_metric, video_g, visual_score))

#lip synchronization
import face_alignment

fa = face_alignment.FaceAlignment(face_alignment.LandmarksType._2D, flip_input=False) #face_alignment

pred_types = {'face': slice(0, 17),
              'eyebrow1': slice(17, 22),
              'eyebrow2': slice(22, 27),
              'nose': slice(27, 31),
              'nostril': slice(31, 36),
              'eye1': slice(36, 42),
              'eye2': slice(42, 48),
              'lips': slice(48, 60),
              'teeth': slice(60, 68)
              }

def compute_lmd(src_video, gen_video):
    area = np.shape(src_video)[1]
    face_lmd = 0
    lips_lmd = 0
    eyes_lmd = 0
    omit_frame = 0

    if len(src_video) > len(gen_video):
        src_video = src_video[:len(gen_video), ...]
    elif len(gen_video) > len(src_video):
        gen_video = gen_video[:len(src_video), ...]

    for i in range(len(src_video)):
        frame1 = src_video[i, ...]
        frame2 = gen_video[i, ...]

        preds1 = fa.get_landmarks(frame1)
        preds2 = fa.get_landmarks(frame2)
        # assert len(preds1) == 1
        # assert len(preds2) == 1
        if (len(preds1) == 0) | (len(preds2) == 0):
            omit_frame += 1
            continue
        face1 = preds1[0]
        face2 = preds2[0]

        face1_lmk = face1[pred_types['face'], :]
        face2_lmk = face2[pred_types['face'], :]
        lips1_lmk = face1[pred_types['lips'], :]
        lips2_lmk = face2[pred_types['lips'], :]
        leye1_lmk = face1[pred_types['eye1'], :]
        leye2_lmk = face2[pred_types['eye1'], :]
        reye1_lmk = face1[pred_types['eye2'], :]
        reye2_lmk = face2[pred_types['eye2'], :]

        face_lmd = face_lmd + np.mean(np.sqrt(np.sum((face1_lmk.astype(np.float64) - face2_lmk.astype(np.float64))**2, axis=1)))
        lips_lmd = lips_lmd + np.mean(np.sqrt(np.sum((lips1_lmk.astype(np.float64) - lips2_lmk.astype(np.float64))**2, axis=1)))
        leye_lmd = np.mean(np.sqrt(np.sum((leye1_lmk.astype(np.float64) - leye2_lmk.astype(np.float64))**2, axis=1)))
        reye_lmd = np.mean(np.sqrt(np.sum((reye1_lmk.astype(np.float64) - reye2_lmk.astype(np.float64))**2, axis=1)))
        eyes_lmd = eyes_lmd + (leye_lmd + reye_lmd)/2

    src_len = len(src_video) - omit_frame
    lmd = {'face': face_lmd/src_len/area, 'lips': lips_lmd/src_len/area, 'eyes': eyes_lmd/src_len/area}

    return lmd

lmd = compute_lmd(source_video, generated_video)
lmd_lips = lmd['lips']

print('Lips LMD of {} is {} (lower better)'.format(video_g, lmd_lips))


#sparse_sync
import argparse
import subprocess
import sys
from pathlib import Path

import torchaudio
import torchvision
from omegaconf import OmegaConf

sys.path.insert(0, '.')  # nopep8
from dataset.dataset_utils import get_video_and_audio
from dataset.transforms import make_class_grid, quantize_offset
from sparse_model.modules.attn_recorder import Recorder
from sparse_model.modules.feature_selector import CrossAttention
from sparse_utils.utils import check_if_file_exists_else_download, which_ffmpeg

from scripts.train_utils import get_model, get_transforms, prepare_inputs

def reencode_video(path, vfps=10, afps=22050, input_size=256):
    assert which_ffmpeg() != '', 'Is ffmpeg installed? Check if the conda environment is activated.'
    new_path = Path.cwd() / 'vis' / f'{Path(path).stem}_{vfps}fps_{input_size}side_{afps}hz.mp4'
    new_path.parent.mkdir(exist_ok=True)
    new_path = str(new_path)
    cmd = f'{which_ffmpeg()}'
    # no info/error printing
    cmd += ' -hide_banner -loglevel panic'
    cmd += f' -y -i {path}'
    # 1) change fps, 2) resize: min(H,W)=MIN_SIDE (vertical vids are supported), 3) change audio framerate
    cmd += f" -vf fps={vfps},scale=iw*{input_size}/'min(iw,ih)':ih*{input_size}/'min(iw,ih)',crop='trunc(iw/2)'*2:'trunc(ih/2)'*2"
    cmd += f" -ar {afps}"
    cmd += f' {new_path}'
    subprocess.call(cmd.split())
    cmd = f'{which_ffmpeg()}'
    cmd += ' -hide_banner -loglevel panic'
    cmd += f' -y -i {new_path}'
    cmd += f' -acodec pcm_s16le -ac 1'
    cmd += f' {new_path.replace(".mp4", ".wav")}'
    subprocess.call(cmd.split())
    return new_path

def decode_single_video_prediction(off_logits, grid, item):
    label = item['targets']['offset_label'].item()
    print('Ground Truth offset (sec):', f'{label:.2f} ({quantize_offset(grid, label)[-1].item()})')
    print()
    print('Prediction Results:')
    off_probs = torch.softmax(off_logits, dim=-1)
    k = min(off_probs.shape[-1], 5)
    topk_logits, topk_preds = torch.topk(off_logits, k)
    # remove batch dimension
    assert len(topk_logits) == 1, 'batch is larger than 1'
    topk_logits = topk_logits[0]
    topk_preds = topk_preds[0]
    off_logits = off_logits[0]
    off_probs = off_probs[0]
    for target_hat in topk_preds:
        print(f'p={off_probs[target_hat]:.4f} ({off_logits[target_hat]:.4f}), "{grid[target_hat]:.2f}" ({target_hat})')

    target_hat = topk_preds[0]
    result = {}
    result['p'] = off_probs[target_hat].cpu().item()
    result['off_logit'] = off_logits[target_hat].cpu().item()
    result['grid_target'] = grid[target_hat].cpu().item()
    result['target_hat'] = target_hat.cpu().item()

    return result

def reconstruct_video_from_input(aud, vid, meta, orig_vid_path, v_start_i_sec, offset_sec, vfps, afps):
    # assumptions
    n_fft = 512
    hop_length = 128
    torchvision_means = [0.485, 0.456, 0.406]
    torchvision_stds = [0.229, 0.224, 0.225]

    # inverse audio transforms
    assert aud.shape[0] == 1, f'batchsize > 1: imgs.shape {aud.shape}'
    means = meta['spec_means'].view(1, 1, -1, 1)
    stds = meta['spec_stds'].view(1, 1, -1, 1)
    spec = aud.cpu() * stds + means
    spec = spec.squeeze(0).squeeze(0)  # was: (B=1, C=1, F, Ta)
    # spec = torch.exp(spec)
    # AudioSpectrogram
    aud_rec = torchaudio.transforms.GriffinLim(n_fft=n_fft, hop_length=hop_length)(torch.exp(spec))
    aud_rec = aud_rec[None]

    # inverse visual transforms
    means = torch.tensor(torchvision_means).view(1, 1, 3, 1, 1)
    stds = torch.tensor(torchvision_stds).view(1, 1, 3, 1, 1)
    vid_rec = ((vid.cpu() * stds + means) * 255).short()
    vid_rec = vid_rec[0].permute(0, 2, 3, 1)

    # make a path to the reconstructed video:
    vis_folder = Path.cwd() / 'vis'
    vis_folder.mkdir(exist_ok=True)
    save_vid_path = vis_folder / f'rec_{Path(orig_vid_path).stem}_off{offset_sec}_t{v_start_i_sec}.mp4'
    save_vid_path = str(save_vid_path)
    print(f'Reconstructed video: {save_vid_path} (vid_crop starts at {v_start_i_sec}, offset {offset_sec})')

    # save the reconstructed input
    torchvision.io.write_video(save_vid_path, vid_rec, vfps, audio_array=aud_rec, audio_fps=afps, audio_codec='aac')

def run_sparse_sync(args):
    cfg_path = f'./logs/sync_models/{args.exp_name}/cfg-{args.exp_name}.yaml'
    ckpt_path = f'./logs/sync_models/{args.exp_name}/{args.exp_name}.pt'

    # if the model does not exist try to download it from the server
    check_if_file_exists_else_download(cfg_path)
    check_if_file_exists_else_download(ckpt_path)

    # load config
    cfg = OmegaConf.load(cfg_path)

    # checking if the provided video has the correct frame rates
    print(f'Using video: {args.vid_path}')
    v, a, vid_meta = torchvision.io.read_video(args.vid_path, pts_unit='sec')
    T, H, W, C = v.shape
    if vid_meta['video_fps'] != args.vfps or vid_meta['audio_fps'] != args.afps or min(H, W) != args.input_size:
        print(f'Reencoding. vfps: {vid_meta["video_fps"]} -> {args.vfps};', end=' ')
        print(f'afps: {vid_meta["audio_fps"]} -> {args.afps};', end=' ')
        print(f'{(H, W)} -> min(H, W)={args.input_size}')
        args.vid_path = reencode_video(args.vid_path, args.vfps, args.afps, args.input_size)
    else:
        print(f'No need to reencode. vfps: {vid_meta["video_fps"]}; afps: {vid_meta["audio_fps"]}; min(H, W)={args.input_size}')

    device = torch.device(args.device)

    # load the model
    _, model = get_model(cfg, device)
    ckpt = torch.load(ckpt_path, map_location=torch.device('cpu'))
    model.load_state_dict(ckpt['model'])

    # Recorder wraps the model to access attention values
    # type2submodule = {'rgb': 'v_selector', 'audio': 'a_selector'}
    submodule_name = 'v_selector'  # else 'v_selector'
    model = Recorder(model, watch_module=CrossAttention, submodule_name=submodule_name)

    model.eval()

    # load visual and audio streams
    # (Tv, 3, H, W) in [0, 225], (Ta, C) in [-1, 1]
    rgb, audio, meta = get_video_and_audio(args.vid_path, get_meta=True)

    # TODO: check if the offset and start are zeros and print this
    # making an item (dict) to apply transformations
    item = {
        'video': rgb, 'audio': audio, 'meta': meta, 'path': args.vid_path, 'split': 'test',
        'targets': {
            # setting the start of the visual crop and the offset size.
            # For instance, if the model is trained on 5sec clips, the provided video is 9sec, and `v_start_i_sec=1.3`
            # the transform will crop out a 5sec-clip from 1.3 to 6.3 seconds and shift the start of the audio
            # track by `args.offset_sec` seconds. It means that if `offset_sec` > 0, the audio will
            # start `offset_sec` earlier than the rgb track.
            # It is a good idea to use something in [-`max_off_sec`, `max_off_sec`] (see `grid`)
            'v_start_i_sec': args.v_start_i_sec,
            'offset_sec': args.offset_sec,
            # dummy values -- don't mind them
            'vggsound_target': 0,
            'vggsound_label': 'PLACEHOLDER',
        },
    }

    # making the offset class grid similar to the one used in transforms
    max_off_sec = cfg.data.max_off_sec
    grid = make_class_grid(-max_off_sec, max_off_sec, cfg.model.params.transformer.params.num_offset_cls)
    # TODO: maybe?
    # assert min(grid) <= args.offset_sec <= max(grid)

    # applying the transform
    transforms = get_transforms(cfg)['test']
    item = transforms(item)

    # prepare inputs for inference
    batch = torch.utils.data.default_collate([item])
    aud, vid, targets = prepare_inputs(batch, device)

    # sanity check: we will take the input to the `model` and recontruct make a video from it.
    # Use this check to make sure the input makes sense (audio should be ok but shifted as you specified)
    reconstruct_video_from_input(aud, vid, batch['meta'], args.vid_path, args.v_start_i_sec, args.offset_sec,
                                 args.vfps, args.afps)

    # forward pass
    _, off_logits, attention = model(vid, aud, targets)

    # simply prints the results of the prediction
    result = decode_single_video_prediction(off_logits, grid, item)

    return result

parser = argparse.ArgumentParser()
parser.add_argument('--exp_name', default='22-07-13T22-25-49', help='In a format: xx-xx-xxTxx-xx-xx')
parser.add_argument('--vid_path', help='A path to .mp4 video')
parser.add_argument('--offset_sec', type=float, default=0.0)
parser.add_argument('--v_start_i_sec', type=float, default=0.0)
parser.add_argument('--vfps', type=int, default=25)
parser.add_argument('--afps', type=int, default=16000)
parser.add_argument('--input_size', type=int, default=224)
parser.add_argument('--device', default='cuda:0')
args = parser.parse_args()

args.vid_path = video_g

result = run_sparse_sync(args)
result = abs(result['grid_target'])

print('Sparsesync offset of {} is {} (lower better)'.format(video_g, result))

# Head movement
from model.visil import ViSiL
from utils import load_video

model = ViSiL(pretrained=True).to('cuda')
model.eval()

target_video = torch.from_numpy(load_video(video_s))
target_features = model.extract_features(target_video.to('cuda'))

query_video = torch.from_numpy(load_video(video_g))
query_features = model.extract_features(query_video.to('cuda'))

similarity = model.calculate_video_similarity(query_features, target_features)

print('Headmovement naturalness measured by ViSiL of {} is {} (Higher better)'.format(video_g, similarity.item()))