from utils.misc import (
    load_config,
    neq_load_customized,
    move_to_device,
    make_logger,
)
from dataset.Dataloader import collate_fn_
from dataset.Dataset import SignLanguageDataset
from modelling.model import build_model
from copy import deepcopy
from opencc import OpenCC
import os
import torch


class T2GService:
    def __init__(self,
                 config: str = "./configs/T2G_tvb.yaml",
                 ckpt_name: str = "best.ckpt"):
        print(f"[INFO] loading T2G configuration from {config}")
        cfg = load_config(config)
        model_dir = cfg['training']['model_dir']
        print(f"[INFO] T2G model dir: {model_dir}")
        os.makedirs(model_dir, exist_ok=True)
        self.logger = make_logger(model_dir=model_dir)

        cfg['device'] = torch.device(
            'cuda' if torch.cuda.is_available() else 'cpu')
        self.cfg = cfg

        model = build_model(cfg)
        self.model = model

        for datasetname in cfg['datanames']:
            print('[INFO] Evaluate '+datasetname)
            load_model_path = os.path.join(
                model_dir, 'ckpts', datasetname+'_'+ckpt_name)
            if os.path.isfile(load_model_path):
                state_dict = torch.load(
                    load_model_path, map_location='cuda' if torch.cuda.is_available() else 'cpu')
                neq_load_customized(
                    model, state_dict['model_state'], verbose=True)
                print('[INFO] Load model ckpt from '+load_model_path)
            else:
                print(f'[INFO] {load_model_path} does not exist')
        self.datasetname = datasetname
        cfg_ = deepcopy(cfg)
        cfg_['datanames'] = [datasetname]
        cfg_['data'] = {k: v for k, v in cfg['data'].items(
        ) if not k in cfg['datanames'] or k == datasetname}
        generate_cfg = cfg_['testing']['cfg']
        self.generate_cfg = generate_cfg

        model.eval()
        self.model = model

        cc = OpenCC('s2t')
        self.cc = cc

        dataset = SignLanguageDataset(cfg["data"][datasetname], "dev")
        self.dataset = dataset

    def translate(self, text: str):
        custom_dataset = [({'name': 'custom-input',
                            'gloss': '',
                            'text': text},
                           self.datasetname)]
        collated = collate_fn_(custom_dataset, data_cfg=self.cfg["data"][self.datasetname], task=self.cfg["task"], is_train=False, dataset=self.dataset,
                               text_tokenizer=self.model.text_tokenizer, gloss_tokenizer=self.model.gloss_tokenizer)
        with torch.no_grad():
            batch_0 = move_to_device(
                collated, self.cfg['device'] if torch.cuda.is_available() else 'cpu')
            forward_output = self.model(is_train=False, **batch_0)
            generate_output = self.model.generate_txt(
                transformer_inputs=forward_output['transformer_inputs'],
                generate_cfg=self.generate_cfg['translation'])
            for hyp in generate_output['decoded_sequences']:
                clean_hyp = [g for g in hyp.split(
                    ' ') if g not in self.model.gloss_tokenizer.special_tokens]
                clean_hyp = ' '.join(clean_hyp)
                gls_hyp = self.cc.convert(clean_hyp).upper(
                ) if self.model.gloss_tokenizer.lower_case else hyp
                # temp hack to remove weird prefix output
                gls_hyp = gls_hyp.lstrip("說 ")
                return gls_hyp
