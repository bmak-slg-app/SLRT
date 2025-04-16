import os
import shutil
import torch

from collections import defaultdict

from modelling.model import build_model
from dataset.Dataloader import collate_fn_

from utils.misc import (
    get_logger, load_config, make_logger,
    move_to_device, neq_load_customized
)

class FeatureExtractionService:
    def __init__(self, config_path: str):
        cfg = load_config(config_path)
        cfg['device'] = torch.device('cuda')
        cfg['training']['num_workers'] = 1

        output_subdir = "extract_feature"
        outputdir = os.path.join(cfg['training']['model_dir'], output_subdir)
        os.makedirs(outputdir, exist_ok=True)
        shutil.copy(config_path, outputdir)

        logger = make_logger(model_dir=outputdir, log_file=f"prediction.log.0")
        model = build_model(cfg)

        ckpt_path = os.path.join(cfg['training']['model_dir'], 'ckpts/best.ckpt')
        if os.path.isfile(ckpt_path):
            load_model_path = os.path.join(ckpt_path)
            state_dict = torch.load(load_model_path, map_location='cuda')
            model.load_state_dict(state_dict['model_state'])
            logger.info('Load model ckpt from '+load_model_path)
        else:
            logger.info(f'{ckpt_path} does not exist, model from scratch')

        self.cfg = cfg
        self.model = model

    def extract_feature(self, frames_path: str) -> torch.Tensor:
        cfg = self.cfg
        model = self.model

        img_path_list = [x for x in os.listdir(frames_path) if x.endswith(".png")]
        custom_data = [{
            'name': f"custom-data@{frames_path}",
            'num_frames': len(img_path_list),
        }]

        custom_batch = collate_fn_(custom_data, cfg["data"], cfg["task"], is_train=False, gloss_tokenizer=model.gloss_tokenizer)

        suffix2data = {'head_rgb_input':[],'head_keypoint_input':[]}
        model.eval()
        with torch.no_grad():
            batch = move_to_device(custom_batch, cfg['device'])
            forward_output = model(is_train=False, **custom_batch)
            entry = {}
            for ii in range(len(custom_batch['name'])):
                for key in ['name','gloss','text','num_frames']:
                    entry[key] = custom_batch[key][ii]
                for suffix, output_data in suffix2data.items():
                    if suffix in forward_output:
                        output_data.append({**entry,
                            'sign':forward_output[suffix][ii].detach().cpu()})
                    if suffix == 'inputs_embeds':
                        valid_len = torch.sum(forward_output['transformer_inputs']['attention_mask'][ii])
                        output_data.append({**entry,
                            'sign':forward_output['transformer_inputs'][suffix][ii,:valid_len].detach().cpu()})

        return suffix2data['head_rgb_input'][0]['sign']

class PredictionService:
    def __init__(self, config_path: str):
        cfg = load_config(config_path)
        cfg['device'] = torch.device('cuda')
        cfg['training']['num_workers'] = 1

        model_dir = cfg['training']['model_dir']
        ckpt_name = 'best.ckpt'
        os.makedirs(model_dir, exist_ok=True)

        logger = make_logger(model_dir=model_dir, log_file='prediction.log')
        model = build_model(cfg)

        load_model_path = os.path.join(model_dir,'ckpts',ckpt_name)
        if os.path.isfile(load_model_path):
            state_dict = torch.load(load_model_path, map_location='cuda')
            neq_load_customized(model, state_dict['model_state'], verbose=True)
            logger.info('Load model ckpt from '+load_model_path)
        else:
            logger.info(f'{load_model_path} does not exist')

        self.cfg = cfg
        self.model = model

    def predict(self, frames_path: str, features: torch.Tensor) -> str:
        cfg = self.cfg
        model = self.model

        img_path_list = [x for x in os.listdir(frames_path) if x.endswith(".png")]
        custom_data = [{
            'name': f"custom-data@{frames_path}",
            'num_frames': len(img_path_list),
            'head_rgb_input': features
        }]

        custom_batch = collate_fn_(custom_data, cfg['data'], cfg['task'], False, gloss_tokenizer=model.gloss_tokenizer, text_tokenizer=model.text_tokenizer)

        model.eval()
        total_val_loss = defaultdict(int)
        results = defaultdict(dict)
        with torch.no_grad():
            batch = move_to_device(custom_batch, cfg['device'])
            forward_output = model(is_train=False, **batch)
            #rgb/keypoint/fuse/ensemble_last_logits
            for k, gls_logits in forward_output.items():
                if not 'gloss_logits' in k or gls_logits==None:
                    continue
                logits_name = k.replace('gloss_logits','')
                if logits_name in ['rgb_','keypoint_','fuse_','ensemble_last_','ensemble_early_','']:
                    if logits_name=='ensemble_early_':
                        input_lengths = forward_output['aux_lengths']['rgb'][-1]
                    else:
                        input_lengths = forward_output['input_lengths']
                    ctc_decode_output = model.predict_gloss_from_logits(
                        gloss_logits=gls_logits,
                        beam_size=cfg['testing']['cfg']['recognition']['beam_size'],
                        input_lengths=input_lengths
                    )
                    batch_pred_gls = model.gloss_tokenizer.convert_ids_to_tokens(ctc_decode_output)
                    for name, gls_hyp, gls_ref in zip(batch['name'], batch_pred_gls, batch['gloss']):
                        results[name][f'{logits_name}gls_hyp'] = \
                            ' '.join(gls_hyp).upper() if model.gloss_tokenizer.lower_case \
                                else ' '.join(gls_hyp)
                        results[name]['gls_ref'] = gls_ref.upper() if model.gloss_tokenizer.lower_case \
                                else gls_ref
                        # print(logits_name)
                        # print(results[name][f'{logits_name}gls_hyp'])
                        # print(results[name]['gls_ref'])

                else:
                    print(logits_name)
                    raise ValueError
            #multi-head
            if 'aux_logits' in forward_output:
                for stream, logits_list in forward_output['aux_logits'].items(): #['rgb', 'keypoint]
                    lengths_list = forward_output['aux_lengths'][stream] #might be empty
                    for i, (logits, lengths) in enumerate(zip(logits_list, lengths_list)):
                        ctc_decode_output = model.predict_gloss_from_logits(
                            gloss_logits=logits,
                            beam_size=cfg['testing']['cfg']['recognition']['beam_size'],
                            input_lengths=lengths)
                        batch_pred_gls = model.gloss_tokenizer.convert_ids_to_tokens(ctc_decode_output)
                        for name, gls_hyp, gls_ref in zip(batch['name'], batch_pred_gls, batch['gloss']):
                            results[name][f'{stream}_aux_{i}_gls_hyp'] = \
                                ' '.join(gls_hyp).upper() if model.gloss_tokenizer.lower_case \
                                    else ' '.join(gls_hyp)
            generate_output = model.generate_txt(
                transformer_inputs=forward_output['transformer_inputs'],
                generate_cfg=cfg['testing']['cfg']['translation'])
            #decoded_sequences
            for name, txt_hyp, txt_ref in zip(batch['name'], generate_output['decoded_sequences'], batch['text']):
                results[name]['txt_hyp'], results[name]['txt_ref'] = txt_hyp, txt_ref
        print(results)
        return results[f"custom-data@{frames_path}"]['txt_hyp']
