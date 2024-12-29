# Work Log for building app for SLG

## Oct 9

I cloned the repository from https://github.com/FangyunWei/SLRT pointed from the paper. I cloned to `~/workspace/SLRT`. Data and pretrained model is stored at `~/workspace/data` and `~/workspace/pretrained_models`.

I used the following docker command for spinning up a container as instructed. The current directory is `SLRT/Spoken2Sign`.

```bash
docker run --gpus all -v $PWD/../../data:/data -v $PWD:/workspace -v $PWD/../../pretrained_models:/pretrained_models --ipc=host -it --rm rzuo/pose:sing_ISLR_smplx /bin/bash
```

I am on my attempt to run

```bash
python text2gloss/prediction.py --config=text2gloss/configs/T2G.yaml
```

First error is opencc is not installed.

```
Traceback (most recent call last):
  File "/workspace/text2gloss/prediction.py", line 41, in <module>
    from opencc import OpenCC
ModuleNotFoundError: No module named 'opencc'
```

I reinstalled it with

```bash
pip install opencc-python-reimplemented
```

I find the configuration filename is `T2G_csl.yaml` and `T2G_phoenix.yaml` instead of `T2G.yaml`. For now I'm assuming I will be using the `phoenix` dataset.

```bash
python text2gloss/prediction.py --config=text2gloss/configs/T2G_phoenix.yaml
```

I found a weird error the script is trying to get the model from HuggingHub. 

```
Traceback (most recent call last):
  File "/workspace/text2gloss/prediction.py", line 271, in <module>
    model = build_model(cfg)
  File "/workspace/text2gloss/modelling/model.py", line 234, in build_model
    model = SignLanguageModel(cfg)
  File "/workspace/text2gloss/modelling/model.py", line 50, in __init__
    self.translation_network = TranslationNetwork(
  File "/workspace/text2gloss/modelling/translation.py", line 15, in __init__
    self.text_tokenizer = TextTokenizer(tokenizer_cfg=cfg['TextTokenizer'])
  File "/workspace/text2gloss/modelling/Tokenizer.py", line 53, in __init__
    self.tokenizer = MBartTokenizer.from_pretrained(
  File "/usr/local/conda/lib/python3.9/site-packages/transformers/tokenization_utils_base.py", line 1651, in from_pretrained
    fast_tokenizer_file = get_fast_tokenizer_file(
  File "/usr/local/conda/lib/python3.9/site-packages/transformers/tokenization_utils_base.py", line 3462, in get_fast_tokenizer_file
    all_files = get_list_of_files(
  File "/usr/local/conda/lib/python3.9/site-packages/transformers/file_utils.py", line 1729, in get_list_of_files
    model_info = HfApi(endpoint=HUGGINGFACE_CO_RESOLVE_ENDPOINT).model_info(
  File "/usr/local/conda/lib/python3.9/site-packages/huggingface_hub/utils/_validators.py", line 92, in _inner_fn
    validate_repo_id(arg_value)
  File "/usr/local/conda/lib/python3.9/site-packages/huggingface_hub/utils/_validators.py", line 136, in validate_repo_id
    raise HFValidationError(
huggingface_hub.utils._validators.HFValidationError: Repo id must be in the form 'repo_name' or 'namespace/repo_name': '../../pretrained_models/mBart_de_t2g'. Use `repo_type` argument if needed.
```

It turns out to be the pretrained model is not in the correct path. The error message is not useful at all finding this out. I redownloaded the model and unzip it to `/pretrained_models/mBart_de_t2g`. Then, this error throws when I try to run prediction again.

```
2024-10-09 14:26:27,283 Initialize translation network from ../../pretrained_models/mBart_de_t2g
Traceback (most recent call last):
  File "/workspace/text2gloss/prediction.py", line 271, in <module>
    model = build_model(cfg)
  File "/workspace/text2gloss/modelling/model.py", line 234, in build_model
    model = SignLanguageModel(cfg)
  File "/workspace/text2gloss/modelling/model.py", line 50, in __init__
    self.translation_network = TranslationNetwork(
  File "/workspace/text2gloss/modelling/translation.py", line 33, in __init__
    self.model = MBartForConditionalGeneration.from_pretrained(
  File "/usr/local/conda/lib/python3.9/site-packages/transformers/modeling_utils.py", line 1424, in from_pretrained
    model, missing_keys, unexpected_keys, mismatched_keys, error_msgs = cls._load_state_dict_into_model(
  File "/usr/local/conda/lib/python3.9/site-packages/transformers/modeling_utils.py", line 1576, in _load_state_dict_into_model
    raise RuntimeError(f"Error(s) in loading state_dict for {model.__class__.__name__}:\n\t{error_msg}")
RuntimeError: Error(s) in loading state_dict for MBartForConditionalGeneration:
	size mismatch for final_logits_bias: copying a param with shape torch.Size([1, 1124]) from checkpoint, the shape in current model is torch.Size([1, 1120]).
	size mismatch for model.shared.weight: copying a param with shape torch.Size([1124, 1024]) from checkpoint, the shape in current model is torch.Size([1120, 1024]).
	size mismatch for model.encoder.embed_tokens.weight: copying a param with shape torch.Size([1124, 1024]) from checkpoint, the shape in current model is torch.Size([1120, 1024]).
	size mismatch for model.decoder.embed_tokens.weight: copying a param with shape torch.Size([1124, 1024]) from checkpoint, the shape in current model is torch.Size([1120, 1024]).
	size mismatch for lm_head.weight: copying a param with shape torch.Size([1124, 1024]) from checkpoint, the shape in current model is torch.Size([1120, 1024]).
```

The offender is from the `transformer` library. The shape from the model checkpoint does not match the expected model. Relevant code at `text2gloss/modelling/translation.py`:

```py
from transformers import MBartForConditionalGeneration, MBartTokenizer, MBartConfig

...

        if 'pretrained_model_name_or_path' in cfg:
            overwrite_cfg = cfg.get('overwrite_cfg', {})
            if self.task == 'T2G':
                overwrite_cfg['vocab_size'] = len(self.gloss_tokenizer)
            self.logger.info('Initialize translation network from {}'.format(cfg['pretrained_model_name_or_path']))
            self.model = MBartForConditionalGeneration.from_pretrained(
                cfg['pretrained_model_name_or_path'],
                **overwrite_cfg 
            )
```

After some source code digging and guessing, I suspect the size of the embedding is incorrect. From the source code, we can see the line `overwrite_cfg['vocab_size'] = len(self.gloss_tokenizer)` sets the vocab size.

The next task will be investigating where the `vocab_size` comes from and how to correct the size.


## Oct 19

Indeed, `len(self.gloss_tokenizer)` is `1120` instead of `1124`.

```py
print(len(self.gloss_tokenizer))
# Output: 1120
```

`gloss_tokenizer` comes from line 21 of `translation.py` as our task is text to gloss.

```py
        elif self.task == 'T2G':
            self.gloss_tokenizer = GlossTokenizer_G2G(tokenizer_cfg=cfg['GlossTokenizer'])
```

In `GlossTokenizer` section of the config, there is no relevant vocab size information.
```yaml
    GlossTokenizer:
      gloss2id_file: ../../pretrained_models/mBart_de_t2g/gloss2ids.pkl
      src_lang: de_DGS
```


Trace, from `Spoken2Sign/text2gloss/modelling/Tokenizer.py`

```py
class GlossTokenizer_G2G(BaseGlossTokenizer):
```

```py
class BaseGlossTokenizer(BaseTokenizer):
    def __init__(self, tokenizer_cfg):
        super().__init__(tokenizer_cfg)
        with open(tokenizer_cfg['gloss2id_file'],'rb') as f:
            self.gloss2id = pickle.load(f) #
        self.gloss2id = defaultdict(lambda: self.gloss2id['<unk>'], self.gloss2id)
        #check 
        ids = [id_ for gls, id_ in self.gloss2id.items()]
        assert len(ids)==len(set(ids))
        self.id2gloss = {}
        for gls, id_ in self.gloss2id.items():
            self.id2gloss[id_] = gls        
        self.lower_case = tokenizer_cfg.get('lower_case',True)
        
    def convert_tokens_to_ids(self, tokens):
        if type(tokens)==list:
            return [self.convert_tokens_to_ids(t) for t in tokens]
        else:
            return self.gloss2id[tokens]

    def convert_ids_to_tokens(self, ids):
        if type(ids)==list:
            return [self.convert_ids_to_tokens(i) for i in ids]
        else:
            return self.id2gloss[ids]
    
    def __len__(self):
        return len(self.id2gloss)
```

Length is from `self.gloss2id` which points to `gloss2id_file` of config.

```yaml
    GlossTokenizer:
      gloss2id_file: ../../pretrained_models/mBart_de_t2g/gloss2ids.pkl
      src_lang: de_DGS
```

Length is indeed 1120.

```py
import pickle

with open('/pretrained_models/mBart_de_t2g/gloss2ids.pkl','rb') as f:
    gloss2id = pickle.load(f)

print(len(gloss2id))
```

## Oct 23

I checked the CSL dataset do not have the same model loading problem.

```bash
python text2gloss/prediction.py --config=text2gloss/configs/T2G_csl.yaml
```

As expected, I don't have the dataset downloaded. So, it returned error.

```
2024-10-23 08:05:48,493 Initialize translation network from ../../pretrained_models/mBart_zh_t2g
2024-10-23 08:05:51,398 Evaluate csl
2024-10-23 08:05:51,398 results/T2G_csl/ckpts/csl_best.ckpt does not exist
2024-10-23 08:05:51,398 Evaluate on dev set
Traceback (most recent call last):
  File "/workspace/text2gloss/dataset/Dataset.py", line 100, in load_annotations
    with open(self.annotation_file, 'rb') as f:
FileNotFoundError: [Errno 2] No such file or directory: '../../data/csl-daily/csl.dev'
```

## Oct 27

Next, I tried the TVB dataset given to Kris.

```
Traceback (most recent call last):
  File "/workspace/text2gloss/prediction.py", line 292, in <module>
    dataloader, sampler = build_dataloader(cfg_, split, model.text_tokenizer, model.gloss_tokenizer, mode='test')
  File "/workspace/text2gloss/dataset/Dataloader.py", line 190, in build_dataloader
    dataset_collect[datasetname] = build_dataset(cfg['data'][datasetname], split)
  File "/workspace/text2gloss/dataset/Dataset.py", line 199, in build_dataset
    dataset = SignLanguageDataset(dataset_cfg, split)
  File "/workspace/text2gloss/dataset/Dataset.py", line 47, in __init__
    self.load_annotations()
  File "/workspace/text2gloss/dataset/Dataset.py", line 103,in load_annotations
    with gzip.open(self.annotation_file, 'rb') as f:
  File "/usr/local/conda/lib/python3.9/gzip.py", line 58, in open
    binary_file = GzipFile(filename, gz_mode, compresslevel)
  File "/usr/local/conda/lib/python3.9/gzip.py", line 173, in __init__
    fileobj = self.myfileobj = builtins.open(filename, mode or 'rb')
FileNotFoundError: [Errno 2] No such file or directory: '../../data/tvb/v5.7_dev_sim.pkl'
```

I suspect `v5.7_dev_sim.pkl` is same as `tvb_dev.pkl`. I updated `T2G_tvb.yaml`.


Before:
```yaml
task: T2G
data:  
  input_data: videos
  input_streams:
  - rgb
  zip_file: ../../data/tvb/tvb
  dev: ../../data/tvb/v5.7_dev_sim.pkl
  test: ../../data/tvb/v5.7_dev_sim.pkl
  train: ../../data/tvb/v5.7_dev_sim.pkl
```

After:
```yaml
task: T2G
data:  
  input_data: videos
  input_streams:
  - rgb
  zip_file: ../../data/tvb/tvb
  dev: ../../data/tvb_dev.pkl
  test: ../../data/tvb_dev.pkl
  train: ../../data/tvb_dev.pkl
```

Wrong guess.

```
Traceback (most recent call last):
  File "/workspace/text2gloss/prediction.py", line 292, in <module>
    dataloader, sampler = build_dataloader(cfg_, split, model.text_tokenizer, model.gloss_tokenizer, mode='test')
  File "/workspace/text2gloss/dataset/Dataloader.py", line 190, in build_dataloader
    dataset_collect[datasetname] = build_dataset(cfg['data'][datasetname], split)
  File "/workspace/text2gloss/dataset/Dataset.py", line 199, in build_dataset
    dataset = SignLanguageDataset(dataset_cfg, split)
  File "/workspace/text2gloss/dataset/Dataset.py", line 47, in __init__
    self.load_annotations()
  File "/workspace/text2gloss/dataset/Dataset.py", line 107, in load_annotations
    a['sign_features'] = a.pop('sign',None)
AttributeError: 'str' object has no attribute 'pop'
```

I get back to investigating how to prepare the data. The README linked [TwoStreamNetwork](https://github.com/FangyunWei/SLRT/tree/main/TwoStreamNetwork)
in the data preparation. The download script seems relevant.

https://github.com/FangyunWei/SLRT/blob/main/TwoStreamNetwork/download.sh


Unfortunately, I failed to unzip the file.


## Oct 29

Let just focus on Text2Gloss part only. I started reading how data is loaded.

This missing file is from `Spoken2Sign/text2gloss/dataset/Dataset.py`.

It is trying to open a gzipped pickle/ raw pickle file.

```py
    def load_annotations(self):
        self.annotation_file = self.dataset_cfg[self.split]
        try:
            with open(self.annotation_file, 'rb') as f:
                self.annotation = pickle.load(f)
        except:
            with gzip.open(self.annotation_file, 'rb') as f:
                self.annotation = pickle.load(f)
```

`dataset_cfg[self.split]` just means that three `dev`, `test`, `train` keys in config. Therefrore, it can be inferred
the three keys are for annotation data.


I get back to `prediction.py`.

Looking at the main function. Line 272:

```py
do_translation, do_recognition = cfg['task'] not in ['S2G','T2G'], cfg['task'] not in ['G2T','T2G'] #(and recognition loss>0 if S2T)
```

False for both params.


```py
    for datasetname in cfg['datanames']:
        logger.info('Evaluate '+datasetname)
        load_model_path = os.path.join(model_dir,'ckpts',datasetname+'_'+args.ckpt_name)
        if os.path.isfile(load_model_path):
            state_dict = torch.load(load_model_path, map_location='cuda')
            neq_load_customized(model, state_dict['model_state'], verbose=True)
            epoch, global_step = state_dict.get('epoch',0), state_dict.get('global_step',0)
            logger.info('Load model ckpt from '+load_model_path)
        else:
            logger.info(f'{load_model_path} does not exist')
            epoch, global_step = 0, 0
        cfg_ = deepcopy(cfg)
        cfg_['datanames'] = [datasetname]
        cfg_['data'] = {k:v for k,v in cfg['data'].items() if not k in cfg['datanames'] or k==datasetname}
        for split in ['dev', 'test']:
            logger.info('Evaluate on {} set'.format(split))
            dataloader, sampler = build_dataloader(cfg_, split, model.text_tokenizer, model.gloss_tokenizer, mode='test')
            evaluation(model=model, val_dataloader=dataloader, cfg=cfg_, 
                    epoch=epoch, global_step=global_step, 
                    generate_cfg=cfg_['testing']['cfg'],
                    save_dir=os.path.join(model_dir,args.save_subdir,split),
                    do_translation=do_translation, do_recognition=do_recognition, external_logits=args.external_logits, save_logits=True)
```

New finding! The test tokenizer is put inside the dataloader.

The evaluation function just put data batches inside the model for do_translation = False and do_recognition = False

```py
    with torch.no_grad():
        for step, batch in enumerate(val_dataloader):
            #forward -- loss
            batch = move_to_device(batch, cfg['device'])
            # print(batch['gloss'])
            datasetname = batch['datasetname']
            st_time = time.time()
            forward_output = model(is_train=False, **batch)
            time_cost = time.time() - st_time
            tot_time += time_cost
            for k,v in forward_output.items():
                if '_loss' in k:
                    total_val_loss[k] += v.item()
```

That's not useful. Let's see how the model is built. From `training.py`

```py
model = build_model(cfg)
```

From `text2gloss/modelling/model.py`

```py
def build_model(cfg):
    model = SignLanguageModel(cfg)
    return model.to(cfg['device'])
```

```py
class SignLanguageModel(torch.nn.Module):
    def __init__(self, cfg) -> None:
        super().__init__()
        self.logger = get_logger()
        self.task, self.device = cfg['task'], cfg['device']
        model_cfg = cfg['model']
        self.frozen_modules = []
...

        if self.task in ['G2T', 'T2G']:
            input_type = model_cfg['TranslationNetwork'].get('input_type', 'gloss')
            self.translation_network = TranslationNetwork(
                input_type=input_type, cfg=model_cfg['TranslationNetwork'],
                task=self.task)
            self.text_tokenizer = self.translation_network.text_tokenizer
            self.gloss_tokenizer = self.translation_network.gloss_tokenizer #G2T
```

From `text2gloss/modelling/translation.py`. Important assignment statements in constructor.

```py
self.text_tokenizer = TextTokenizer(tokenizer_cfg=cfg['TextTokenizer'])
self.gloss_tokenizer = GlossTokenizer_G2G(tokenizer_cfg=cfg['GlossTokenizer'])
self.text_embedding = torch.load(cfg['TextTokenizer']['text2embed_file'])
self.model = MBartForConditionalGeneration.from_pretrained(
    cfg['pretrained_model_name_or_path'],
    **overwrite_cfg 
)
```

Let's look at forward.

```py
        elif self.input_type == 'text':
            input_ids = kwargs.pop('input_ids')
            # print(input_ids)
            kwargs['inputs_embeds'] = self.text_embedding[input_ids].to(input_ids.device) * self.input_embed_scale
        # else:
            # raise ValueError
        # print(kwargs['attention_mask'])
        output_dict = self.model(**kwargs, return_dict=True)
        #print(output_dict.keys()) loss, logits, past_key_values, encoder_last_hidden_state
        log_prob = torch.nn.functional.log_softmax(output_dict['logits'], dim=-1)  # B, T, L
        batch_loss_sum = self.translation_loss_fun(log_probs=log_prob, targets=kwargs['labels'])
        output_dict['translation_loss'] = batch_loss_sum/log_prob.shape[0]

        output_dict['transformer_inputs'] = kwargs #for later use (decoding)
        return output_dict
```

Input embedding is passed to MBart. Where does `input_ids` come from??

Back to `text2gloss/modelling/model.py`

```py
    def forward(self, is_train, translation_inputs={}, recognition_inputs={}, **kwargs):

        elif self.task in ['G2T', 'T2G']:
            model_outputs = self.translation_network(**translation_inputs)
            model_outputs['total_loss'] = model_outputs['translation_loss']
```

translation_inputs in forward.

Now lets read `text2gloss/dataset/Dataloader.py`

```py
def collate_fn_(inputs, data_cfg, task, is_train, dataset,
    text_tokenizer=None, gloss_tokenizer=None):
      outputs = {
        'name':[i['name'] for i,n in inputs],
        'gloss':[i.get('gloss','') for i,n in inputs],
        'text':[i.get('text','') for i,n in inputs],
        'num_frames':[i.get('num_frames',None) for i,n in inputs],
        'datasetname': [n for i,n in inputs]}
      if task in ['S2T','G2T','S2T_glsfree','S2T_Ensemble','T2G']:
        tokenized_text = text_tokenizer(input_str=outputs['text'], text_input=(task=='T2G'))
        outputs['translation_inputs'] = {**tokenized_text}

        elif task == 'T2G':
            gls_tok_results = gloss_tokenizer(label_gls_seq=outputs['gloss'], need_input=False, need_label=True)
            # gls_tok_results = gloss_tokenizer(input_str=outputs['gloss'], text_input=False)
            outputs['translation_inputs']['labels'] = gls_tok_results['labels']
            outputs['translation_inputs']['decoder_input_ids'] = gls_tok_results['decoder_input_ids']
            # print(outputs['translation_inputs']['labels'], outputs['translation_inputs']['decoder_input_ids'])
```

decoder_input_ids???? not input_ids???

Maybe the text_tokenizer?

`text2gloss/modelling/Tokenizer.py`. Note that text_input is True.

```py
from transformers import MBartTokenizer

class BaseTokenizer(object):
class TextTokenizer(BaseTokenizer):

    def __call__(self, input_str, text_input=False):
        outputs = {}
        if text_input:
            # for text-to-gloss
            raw_outputs = self.tokenizer(input_str, return_attention_mask=True, return_length=True, padding='longest')
            # print(raw_outputs)
            pruned_input_ids = self.prune(raw_outputs['input_ids'])
            outputs['input_ids'] = pruned_input_ids
            outputs['attention_mask'] = torch.tensor(raw_outputs['attention_mask']).long()
        return outputs 
    
```

https://huggingface.co/docs/transformers/en/model_doc/mbart#transformers.MBartTokenizer
https://huggingface.co/docs/transformers/en/main_classes/tokenizer#transformers.PreTrainedTokenizer.__call__

Cool! So we just need to get the dataset downloaded and start working.

## Oct 30

Too optimistic?

```
ls PHOENIX-2014-T-release-v3/PHOENIX-2014-T/annotations/manual/
PHOENIX-2014-T.dev.corpus.csv  PHOENIX-2014-T.test.corpus.csv  PHOENIX-2014-T.train-complex-annotation.corpus.csv  PHOENIX-2014-T.train.corpus.csv
```

`head PHOENIX-2014-T-release-v3/PHOENIX-2014-T/annotations/manual/PHOENIX-2014-T.dev.corpus.csv`

```csv
name|video|start|end|speaker|orth|translation
11August_2010_Wednesday_tagesschau-2|11August_2010_Wednesday_tagesschau-2/1/*.png|-1|-1|Signer08|DRUCK TIEF KOMMEN|tiefer luftdruck bestimmt in den nächsten tagen unser wetter
11August_2010_Wednesday_tagesschau-3|11August_2010_Wednesday_tagesschau-3/1/*.png|-1|-1|Signer08|ES-BEDEUTET VIEL WOLKE UND KOENNEN REGEN GEWITTER KOENNEN|das bedeutet viele wolken und immer wieder zum teil kräftige schauer und gewitter
11August_2010_Wednesday_tagesschau-8|11August_2010_Wednesday_tagesschau-8/1/*.png|-1|-1|Signer08|WIND MAESSIG SCHWACH REGION WENN GEWITTER WIND KOENNEN|meist weht nur ein schwacher wind aus unterschiedlichen richtungen der bei schauern und gewittern stark böig sein kann
25October_2010_Monday_tagesschau-22|25October_2010_Monday_tagesschau-22/1/*.png|-1|-1|Signer01|MITTWOCH REGEN KOENNEN NORDWEST WAHRSCHEINLICH NORD STARK WIND|am mittwoch hier und da nieselregen in der nordwesthälfte an den küsten kräftiger wind
05May_2011_Thursday_tagesschau-25|05May_2011_Thursday_tagesschau-25/1/*.png|-1|-1|Signer08|JETZT WETTER WIE-AUSSEHEN MORGEN FREITAG SECHSTE MAI ZEIGEN-BILDSCHIRM|und nun die wettervorhersage für morgen freitag den sechsten mai
15December_2010_Wednesday_tagesschau-40|15December_2010_Wednesday_tagesschau-40/1/*.png|-1|-1|Signer05|DANN STARK SCHNEE SCHNEIEN KOMMEN|am tag breiten sich die teilweise kräftigen schneefälle weiter aus
10March_2011_Thursday_heute-51|10March_2011_Thursday_heute-51/1/*.png|-1|-1|Signer01|SUEDWEST KOMMEN WARM MORGEN SCHON FRANKREICH REGION FUENFZEHN BIS ZWANZIG GRAD WOCHENENDE KOMMEN WARM|es wird wärmer aus dem südwesten morgen haben wir schon über frankreich die fünfzehn bis zwanzig grad und am wochenende kommt diese wärme langsam auch zu uns
14August_2009_Friday_tagesschau-72|14August_2009_Friday_tagesschau-72/1/*.png|-1|-1|Signer05|MORGEN DAENEMARK IX ZWANZIG MAXIMAL DREISSIG GRAD|morgen werte von zwanzig grad an der dänischen grenze bis dreißig grad am oberrhein
26January_2010_Tuesday_heute-107|26January_2010_Tuesday_heute-107/1/*.png|-1|-1|Signer03|SCHOEN ABEND WUENSCHEN|jetzt wünsche ich ihnen noch einen schönen abend
```

## Dec 9

Following up previous email, I prepared the TVB dataset into a pickle.

However, this left me CUDA errors as following:

```
Traceback (most recent call last):
  File "/workspace/text2gloss/prediction.py", line 293, in <module>
    evaluation(model=model, val_dataloader=dataloader, cfg=cfg_, 
  File "/workspace/text2gloss/prediction.py", line 97, in evaluation
    forward_output = model(is_train=False, **batch)
  File "/usr/local/conda/lib/python3.9/site-packages/torch/nn/modules/module.py", line 1051, in _call_impl
    return forward_call(*input, **kwargs)
  File "/workspace/text2gloss/modelling/model.py", line 143, in forward
    model_outputs = self.translation_network(**translation_inputs)
  File "/usr/local/conda/lib/python3.9/site-packages/torch/nn/modules/module.py", line 1051, in _call_impl
    return forward_call(*input, **kwargs)
  File "/workspace/text2gloss/modelling/translation.py", line 204, in forward
    kwargs['inputs_embeds'] = self.text_embedding[input_ids].to(input_ids.device) * self.input_embed_scale
RuntimeError: CUDA error: no kernel image is available for execution on the device
CUDA kernel errors might be asynchronously reported at some other API call,so the stacktrace below might be incorrect.
For debugging consider passing CUDA_LAUNCH_BLOCKING=1.
```

I tried updated PyTorch cuda version to 11.3 (latest available for that pytorch version) in the docker image

```
conda install cudatoolkit=11.3 -c pytorch -c conda-forge
```

```
Solving environment: failed with initial frozen solve. Retrying with flexible solve.

# >>>>>>>>>>>>>>>>>>>>>> ERROR REPORT <<<<<<<<<<<<<<<<<<<<<<

    Traceback (most recent call last):
      File "/usr/local/conda/lib/python3.9/site-packages/conda/common/logic.py", line 125, in _convert
        return self.names[name]
    KeyError: 'conda-forge/linux-64::lintel-1.0-py38h0717a4f_1003'
    
    During handling of the above exception, another exception occurred:
    
    Traceback (most recent call last):
      File "/usr/local/conda/lib/python3.9/site-packages/conda/exceptions.py", line 1129, in __call__
        return func(*args, **kwargs)
      File "/usr/local/conda/lib/python3.9/site-packages/conda/cli/main.py", line 86, in main_subshell
        exit_code = do_call(args, p)
      File "/usr/local/conda/lib/python3.9/site-packages/conda/cli/conda_argparse.py", line 93, in do_call
        return getattr(module, func_name)(args, parser)
      File "/usr/local/conda/lib/python3.9/site-packages/conda/notices/core.py", line 72, in wrapper
        return_value = func(*args, **kwargs)
      File "/usr/local/conda/lib/python3.9/site-packages/conda/cli/main_install.py", line 22, in execute
        install(args, parser, 'install')
      File "/usr/local/conda/lib/python3.9/site-packages/conda/cli/install.py", line 261, in install
        unlink_link_transaction = solver.solve_for_transaction(
      File "/usr/local/conda/lib/python3.9/site-packages/conda/core/solve.py", line 156, in solve_for_transaction
        unlink_precs, link_precs = self.solve_for_diff(update_modifier, deps_modifier,
      File "/usr/local/conda/lib/python3.9/site-packages/conda/core/solve.py", line 199, in solve_for_diff
        final_precs = self.solve_final_state(update_modifier, deps_modifier, prune, ignore_pinned,
      File "/usr/local/conda/lib/python3.9/site-packages/conda/core/solve.py", line 317, in solve_final_state
        ssc = self._add_specs(ssc)
      File "/usr/local/conda/lib/python3.9/site-packages/conda/core/solve.py", line 609, in _add_specs
        conflict_specs = ssc.r.get_conflicting_specs(tuple(concatv(
      File "/usr/local/conda/lib/python3.9/site-packages/conda/resolve.py", line 1110, in get_conflicting_specs
        C = r2.gen_clauses()
      File "/usr/local/conda/lib/python3.9/site-packages/conda/common/io.py", line 86, in decorated
        return f(*args, **kwds)
      File "/usr/local/conda/lib/python3.9/site-packages/conda/resolve.py", line 912, in gen_clauses
        nkey = C.Not(self.to_sat_name(prec))
      File "/usr/local/conda/lib/python3.9/site-packages/conda/common/logic.py", line 144, in Not
        return self._eval(self._clauses.Not, (x,), (), polarity, name)
      File "/usr/local/conda/lib/python3.9/site-packages/conda/common/logic.py", line 130, in _eval
        args = self._convert(args)
      File "/usr/local/conda/lib/python3.9/site-packages/conda/common/logic.py", line 120, in _convert
        return type(x)(map(self._convert, x))
      File "/usr/local/conda/lib/python3.9/site-packages/conda/common/logic.py", line 127, in _convert
        raise ValueError("Unregistered SAT variable name: {}".format(name))
    ValueError: Unregistered SAT variable name: conda-forge/linux-64::lintel-1.0-py38h0717a4f_1003

`$ /usr/local/bin/conda install cudatoolkit=11.3 -c pytorch -c conda-forge`

  environment variables:
                 CIO_TEST=<not set>
               CONDA_ROOT=/usr/local/conda
           CURL_CA_BUNDLE=<not set>
          LD_LIBRARY_PATH=/usr/local/nvidia/lib:/usr/local/nvidia/lib64:/usr/local/cuda/lib64:/u
                          sr/local/cuda/extras/CUPTI/lib64:/usr/lib:/usr/local/lib:/usr/local/nc
                          cl-rdma-sharp-plugins/lib
             LIBRARY_PATH=/usr/local/cuda/lib64/stubs
                     PATH=/usr/local/nvidia/bin:/usr/local/cuda/bin:/usr/local/sbin:/usr/local/b
                          in:/usr/sbin:/usr/bin:/sbin:/bin:/usr/local/conda/bin
                   PYTHON=python3
  PYTHONDONTWRITEBYTECODE=1
         PYTHONIOENCODING=UTF-8
         PYTHONUNBUFFERED=1
       REQUESTS_CA_BUNDLE=<not set>
            SSL_CERT_FILE=<not set>

     active environment : None
       user config file : /root/.condarc
 populated config files : /usr/local/conda/.condarc
          conda version : 22.9.0
    conda-build version : not installed
         python version : 3.9.13.final.0
       virtual packages : __cuda=12.2=0
                          __linux=6.1.0=0
                          __glibc=2.27=0
                          __unix=0=0
                          __archspec=1=x86_64
       base environment : /usr/local/conda  (writable)
      conda av data dir : /usr/local/conda/etc/conda
  conda av metadata url : None
           channel URLs : https://conda.anaconda.org/pytorch/linux-64
                          https://conda.anaconda.org/pytorch/noarch
                          https://conda.anaconda.org/conda-forge/linux-64
                          https://conda.anaconda.org/conda-forge/noarch
                          https://repo.anaconda.com/pkgs/main/linux-64
                          https://repo.anaconda.com/pkgs/main/noarch
                          https://repo.anaconda.com/pkgs/r/linux-64
                          https://repo.anaconda.com/pkgs/r/noarch
          package cache : /usr/local/conda/pkgs
                          /root/.conda/pkgs
       envs directories : /root/.conda/envs
                          /usr/local/conda/envs
               platform : linux-64
             user-agent : conda/22.9.0 requests/2.28.1 CPython/3.9.13 Linux/6.1.0-25-amd64 ubuntu/18.04.6 glibc/2.27
                UID:GID : 0:0
             netrc file : None
           offline mode : False


An unexpected error has occurred. Conda has prepared the above report.

Upload did not complete.
```

I tried to use my local environment instead.

```
Traceback (most recent call last):
  File "/home/system/workspace/SLRT/Spoken2Sign/text2gloss/prediction.py", line 6, in <module>
    from modelling.model import build_model
  File "/home/system/workspace/SLRT/Spoken2Sign/text2gloss/modelling/model.py", line 2, in <module>
    from modelling.recognition import RecognitionNetwork
  File "/home/system/workspace/SLRT/Spoken2Sign/text2gloss/modelling/recognition.py", line 6, in <module>
    from modelling.resnet3d import ResNet3dSlowOnly_backbone
  File "/home/system/workspace/SLRT/Spoken2Sign/text2gloss/modelling/resnet3d.py", line 6, in <module>
    from mmcv.cnn import ConvModule, build_activation_layer, constant_init, kaiming_init
ImportError: cannot import name 'constant_init' from 'mmcv.cnn' (/home/system/miniconda3/envs/SLG/lib/python3.10/site-packages/mmcv/cnn/__init__.py)
```

I downgraded mmcv to version in docker image

```
conda install mmcv-full==1.7.0
```

I suspect PyTorch 2.x does not work.

```
Traceback (most recent call last):                                                                                                                                                                                                    
  File "/home/system/workspace/SLRT/Spoken2Sign/text2gloss/prediction.py", line 6, in <module>                                                                                                                                        
    from modelling.model import build_model                                                                                                                                                                                           
  File "/home/system/workspace/SLRT/Spoken2Sign/text2gloss/modelling/model.py", line 2, in <module>                                                                                                                                   
    from modelling.recognition import RecognitionNetwork                                                                                                                                                                              
  File "/home/system/workspace/SLRT/Spoken2Sign/text2gloss/modelling/recognition.py", line 5, in <module>                                                                                                                             
    from modelling.ResNet2d import ResNet2d_backbone                                                                                                                                                                                  
  File "/home/system/workspace/SLRT/Spoken2Sign/text2gloss/modelling/ResNet2d.py", line 2, in <module>                                                                                                                                
    import torch, torchvision                                                                                                                                                                                                         
  File "/home/system/miniconda3/envs/SLG/lib/python3.10/site-packages/torchvision/__init__.py", line 6, in <module>                                                                                                                   
    from torchvision import _meta_registrations, datasets, io, models, ops, transforms, utils                                                                                                                                         
  File "/home/system/miniconda3/envs/SLG/lib/python3.10/site-packages/torchvision/_meta_registrations.py", line 4, in <module>                                                                                                        
    import torch._custom_ops                                                                                                                                                                                                          
  File "/home/system/miniconda3/envs/SLG/lib/python3.10/site-packages/torch/_custom_ops.py", line 3, in <module>                                                                                                                      
    from torch._custom_op.impl import (
  File "/home/system/miniconda3/envs/SLG/lib/python3.10/site-packages/torch/_custom_op/impl.py", line 13, in <module>
    from torch._library.abstract_impl import AbstractImplCtx
  File "/home/system/miniconda3/envs/SLG/lib/python3.10/site-packages/torch/_library/__init__.py", line 1, in <module>
    import torch._library.abstract_impl
  File "/home/system/miniconda3/envs/SLG/lib/python3.10/site-packages/torch/_library/abstract_impl.py", line 117, in <module>
    class AbstractImplCtx:
  File "/home/system/miniconda3/envs/SLG/lib/python3.10/site-packages/torch/_library/abstract_impl.py", line 126, in AbstractImplCtx
    def create_unbacked_symint(self, *, min=2, max=None) -> torch.SymInt:
AttributeError: module 'torch' has no attribute 'SymInt'
```

Then, I downgraded PyTorch

```
conda install pytorch==1.13.1 torchvision==0.14.1 torchaudio==0.13.1 pytorch-cuda=11.7 -c pytorch -c nvidia
```

Next issue

```
Traceback (most recent call last):
  File "/home/system/workspace/SLRT/Spoken2Sign/text2gloss/prediction.py", line 6, in <module>
    from modelling.model import build_model
  File "/home/system/workspace/SLRT/Spoken2Sign/text2gloss/modelling/model.py", line 2, in <module>
    from modelling.recognition import RecognitionNetwork
  File "/home/system/workspace/SLRT/Spoken2Sign/text2gloss/modelling/recognition.py", line 16, in <module>
    from ctcdecode import CTCBeamDecoder
ModuleNotFoundError: No module named 'ctcdecode'
```

Not working

```
conda install pyctcdecode
```

This install the package as pyctcdecode which python cannot discover

```
pip install ctcdecode
```

```
❯ pip install ctcdecode
Collecting ctcdecode
  Using cached ctcdecode-1.0.2.tar.gz (125 kB)
  Preparing metadata (setup.py) ... done
Building wheels for collected packages: ctcdecode
  Building wheel for ctcdecode (setup.py) ... error
  error: subprocess-exited-with-error
  
  × python setup.py bdist_wheel did not run successfully.
  │ exit code: 1
  ╰─> [165 lines of output]
      /tmp/pip-install-fxgtjml5/ctcdecode_a306757a5af04bd1be52dabac4a498b6/setup.py:34: UserWarning: File `third_party/kenlm/setup.py` does not appear to be present. Did you forget `git submodule update`?
        warnings.warn('File `{}` does not appear to be present. Did you forget `git submodule update`?'.format(file))
      /tmp/pip-install-fxgtjml5/ctcdecode_a306757a5af04bd1be52dabac4a498b6/setup.py:34: UserWarning: File `third_party/ThreadPool/ThreadPool.h` does not appear to be present. Did you forget `git submodule update`?
        warnings.warn('File `{}` does not appear to be present. Did you forget `git submodule update`?'.format(file))
      /home/system/miniconda3/envs/SLG/lib/python3.10/site-packages/setuptools/_distutils/extension.py:134: UserWarning: Unknown Extension options: 'package', 'with_cuda'
        warnings.warn(msg)
      running bdist_wheel
      running build
      running build_py
      creating build
      creating build/lib.linux-x86_64-cpython-310
      creating build/lib.linux-x86_64-cpython-310/ctcdecode
      copying ctcdecode/__init__.py -> build/lib.linux-x86_64-cpython-310/ctcdecode
      running build_ext
      building 'ctcdecode._ext.ctc_decode' extension
      creating /tmp/pip-install-fxgtjml5/ctcdecode_a306757a5af04bd1be52dabac4a498b6/build/temp.linux-x86_64-cpython-310
      creating /tmp/pip-install-fxgtjml5/ctcdecode_a306757a5af04bd1be52dabac4a498b6/build/temp.linux-x86_64-cpython-310/ctcdecode
      creating /tmp/pip-install-fxgtjml5/ctcdecode_a306757a5af04bd1be52dabac4a498b6/build/temp.linux-x86_64-cpython-310/ctcdecode/src
      creating /tmp/pip-install-fxgtjml5/ctcdecode_a306757a5af04bd1be52dabac4a498b6/build/temp.linux-x86_64-cpython-310/third_party
      creating /tmp/pip-install-fxgtjml5/ctcdecode_a306757a5af04bd1be52dabac4a498b6/build/temp.linux-x86_64-cpython-310/third_party/kenlm
      creating /tmp/pip-install-fxgtjml5/ctcdecode_a306757a5af04bd1be52dabac4a498b6/build/temp.linux-x86_64-cpython-310/third_party/kenlm/lm
      creating /tmp/pip-install-fxgtjml5/ctcdecode_a306757a5af04bd1be52dabac4a498b6/build/temp.linux-x86_64-cpython-310/third_party/kenlm/util
      creating /tmp/pip-install-fxgtjml5/ctcdecode_a306757a5af04bd1be52dabac4a498b6/build/temp.linux-x86_64-cpython-310/third_party/kenlm/util/double-conversion
      creating /tmp/pip-install-fxgtjml5/ctcdecode_a306757a5af04bd1be52dabac4a498b6/build/temp.linux-x86_64-cpython-310/third_party/openfst-1.6.7
      creating /tmp/pip-install-fxgtjml5/ctcdecode_a306757a5af04bd1be52dabac4a498b6/build/temp.linux-x86_64-cpython-310/third_party/openfst-1.6.7/src
      creating /tmp/pip-install-fxgtjml5/ctcdecode_a306757a5af04bd1be52dabac4a498b6/build/temp.linux-x86_64-cpython-310/third_party/openfst-1.6.7/src/lib
      Emitting ninja build file /tmp/pip-install-fxgtjml5/ctcdecode_a306757a5af04bd1be52dabac4a498b6/build/temp.linux-x86_64-cpython-310/build.ninja...
      Compiling objects...
      Allowing ninja to set a default number of workers... (overridable by setting the environment variable MAX_JOBS=N)
      [1/54] c++ -MMD -MF /tmp/pip-install-fxgtjml5/ctcdecode_a306757a5af04bd1be52dabac4a498b6/build/temp.linux-x86_64-cpython-310/ctcdecode/src/ctc_beam_search_decoder.o.d -pthread -B /home/system/miniconda3/envs/SLG/compiler_compat -Wno-unused-result -Wsign-compare -DNDEBUG -fwrapv -O2 -Wall -fPIC -O2 -isystem /home/system/miniconda3/envs/SLG/include -fPIC -O2 -isystem /home/system/miniconda3/envs/SLG/include -fPIC -I/tmp/pip-install-fxgtjml5/ctcdecode_a306757a5af04bd1be52dabac4a498b6/third_party/kenlm -I/tmp/pip-install-fxgtjml5/ctcdecode_a306757a5af04bd1be52dabac4a498b6/third_party/openfst-1.6.7/src/include -I/tmp/pip-install-fxgtjml5/ctcdecode_a306757a5af04bd1be52dabac4a498b6/third_party/ThreadPool -I/tmp/pip-install-fxgtjml5/ctcdecode_a306757a5af04bd1be52dabac4a498b6/third_party/boost_1_67_0 -I/tmp/pip-install-fxgtjml5/ctcdecode_a306757a5af04bd1be52dabac4a498b6/third_party/utf8 -I/home/system/miniconda3/envs/SLG/lib/python3.10/site-packages/torch/include -I/home/system/miniconda3/envs/SLG/lib/python3.10/site-packages/torch/include/torch/csrc/api/include -I/home/system/miniconda3/envs/SLG/lib/python3.10/site-packages/torch/include/TH -I/home/system/miniconda3/envs/SLG/lib/python3.10/site-packages/torch/include/THC -I/home/system/miniconda3/envs/SLG/lib/python3.10/site-packages/torch/include -I/home/system/miniconda3/envs/SLG/lib/python3.10/site-packages/torch/include/torch/csrc/api/include -I/home/system/miniconda3/envs/SLG/lib/python3.10/site-packages/torch/include/TH -I/home/system/miniconda3/envs/SLG/lib/python3.10/site-packages/torch/include/THC -I/home/system/miniconda3/envs/SLG/include/python3.10 -c -c /tmp/pip-install-fxgtjml5/ctcdecode_a306757a5af04bd1be52dabac4a498b6/ctcdecode/src/ctc_beam_search_decoder.cpp -o /tmp/pip-install-fxgtjml5/ctcdecode_a306757a5af04bd1be52dabac4a498b6/build/temp.linux-x86_64-cpython-310/ctcdecode/src/ctc_beam_search_decoder.o -O3 -DKENLM_MAX_ORDER=6 -std=c++14 -fPIC -DINCLUDE_KENLM -DKENLM_MAX_ORDER=6 -DTORCH_API_INCLUDE_EXTENSION_H '-DPYBIND11_COMPILER_TYPE="_gcc"' '-DPYBIND11_STDLIB="_libstdcpp"' '-DPYBIND11_BUILD_ABI="_cxxabi1014"' -DTORCH_EXTENSION_NAME=ctc_decode -D_GLIBCXX_USE_CXX11_ABI=1
      FAILED: /tmp/pip-install-fxgtjml5/ctcdecode_a306757a5af04bd1be52dabac4a498b6/build/temp.linux-x86_64-cpython-310/ctcdecode/src/ctc_beam_search_decoder.o
      c++ -MMD -MF /tmp/pip-install-fxgtjml5/ctcdecode_a306757a5af04bd1be52dabac4a498b6/build/temp.linux-x86_64-cpython-310/ctcdecode/src/ctc_beam_search_decoder.o.d -pthread -B /home/system/miniconda3/envs/SLG/compiler_compat -Wno-unused-result -Wsign-compare -DNDEBUG -fwrapv -O2 -Wall -fPIC -O2 -isystem /home/system/miniconda3/envs/SLG/include -fPIC -O2 -isystem /home/system/miniconda3/envs/SLG/include -fPIC -I/tmp/pip-install-fxgtjml5/ctcdecode_a306757a5af04bd1be52dabac4a498b6/third_party/kenlm -I/tmp/pip-install-fxgtjml5/ctcdecode_a306757a5af04bd1be52dabac4a498b6/third_party/openfst-1.6.7/src/include -I/tmp/pip-install-fxgtjml5/ctcdecode_a306757a5af04bd1be52dabac4a498b6/third_party/ThreadPool -I/tmp/pip-install-fxgtjml5/ctcdecode_a306757a5af04bd1be52dabac4a498b6/third_party/boost_1_67_0 -I/tmp/pip-install-fxgtjml5/ctcdecode_a306757a5af04bd1be52dabac4a498b6/third_party/utf8 -I/home/system/miniconda3/envs/SLG/lib/python3.10/site-packages/torch/include -I/home/system/miniconda3/envs/SLG/lib/python3.10/site-packages/torch/include/torch/csrc/api/include -I/home/system/miniconda3/envs/SLG/lib/python3.10/site-packages/torch/include/TH -I/home/system/miniconda3/envs/SLG/lib/python3.10/site-packages/torch/include/THC -I/home/system/miniconda3/envs/SLG/lib/python3.10/site-packages/torch/include -I/home/system/miniconda3/envs/SLG/lib/python3.10/site-packages/torch/include/torch/csrc/api/include -I/home/system/miniconda3/envs/SLG/lib/python3.10/site-packages/torch/include/TH -I/home/system/miniconda3/envs/SLG/lib/python3.10/site-packages/torch/include/THC -I/home/system/miniconda3/envs/SLG/include/python3.10 -c -c /tmp/pip-install-fxgtjml5/ctcdecode_a306757a5af04bd1be52dabac4a498b6/ctcdecode/src/ctc_beam_search_decoder.cpp -o /tmp/pip-install-fxgtjml5/ctcdecode_a306757a5af04bd1be52dabac4a498b6/build/temp.linux-x86_64-cpython-310/ctcdecode/src/ctc_beam_search_decoder.o -O3 -DKENLM_MAX_ORDER=6 -std=c++14 -fPIC -DINCLUDE_KENLM -DKENLM_MAX_ORDER=6 -DTORCH_API_INCLUDE_EXTENSION_H '-DPYBIND11_COMPILER_TYPE="_gcc"' '-DPYBIND11_STDLIB="_libstdcpp"' '-DPYBIND11_BUILD_ABI="_cxxabi1014"' -DTORCH_EXTENSION_NAME=ctc_decode -D_GLIBCXX_USE_CXX11_ABI=1
      /tmp/pip-install-fxgtjml5/ctcdecode_a306757a5af04bd1be52dabac4a498b6/ctcdecode/src/ctc_beam_search_decoder.cpp:1:10: fatal error: ctc_beam_search_decoder.h: No such file or directory
          1 | #include "ctc_beam_search_decoder.h"
            |          ^~~~~~~~~~~~~~~~~~~~~~~~~~~
      compilation terminated.
      [2/54] c++ -MMD -MF /tmp/pip-install-fxgtjml5/ctcdecode_a306757a5af04bd1be52dabac4a498b6/build/temp.linux-x86_64-cpython-310/ctcdecode/src/decoder_utils.o.d -pthread -B /home/system/miniconda3/envs/SLG/compiler_compat -Wno-unused-result -Wsign-compare -DNDEBUG -fwrapv -O2 -Wall -fPIC -O2 -isystem /home/system/miniconda3/envs/SLG/include -fPIC -O2 -isystem /home/system/miniconda3/envs/SLG/include -fPIC -I/tmp/pip-install-fxgtjml5/ctcdecode_a306757a5af04bd1be52dabac4a498b6/third_party/kenlm -I/tmp/pip-install-fxgtjml5/ctcdecode_a306757a5af04bd1be52dabac4a498b6/third_party/openfst-1.6.7/src/include -I/tmp/pip-install-fxgtjml5/ctcdecode_a306757a5af04bd1be52dabac4a498b6/third_party/ThreadPool -I/tmp/pip-install-fxgtjml5/ctcdecode_a306757a5af04bd1be52dabac4a498b6/third_party/boost_1_67_0 -I/tmp/pip-install-fxgtjml5/ctcdecode_a306757a5af04bd1be52dabac4a498b6/third_party/utf8 -I/home/system/miniconda3/envs/SLG/lib/python3.10/site-packages/torch/include -I/home/system/miniconda3/envs/SLG/lib/python3.10/site-packages/torch/include/torch/csrc/api/include -I/home/system/miniconda3/envs/SLG/lib/python3.10/site-packages/torch/include/TH -I/home/system/miniconda3/envs/SLG/lib/python3.10/site-packages/torch/include/THC -I/home/system/miniconda3/envs/SLG/lib/python3.10/site-packages/torch/include -I/home/system/miniconda3/envs/SLG/lib/python3.10/site-packages/torch/include/torch/csrc/api/include -I/home/system/miniconda3/envs/SLG/lib/python3.10/site-packages/torch/include/TH -I/home/system/miniconda3/envs/SLG/lib/python3.10/site-packages/torch/include/THC -I/home/system/miniconda3/envs/SLG/include/python3.10 -c -c /tmp/pip-install-fxgtjml5/ctcdecode_a306757a5af04bd1be52dabac4a498b6/ctcdecode/src/decoder_utils.cpp -o /tmp/pip-install-fxgtjml5/ctcdecode_a306757a5af04bd1be52dabac4a498b6/build/temp.linux-x86_64-cpython-310/ctcdecode/src/decoder_utils.o -O3 -DKENLM_MAX_ORDER=6 -std=c++14 -fPIC -DINCLUDE_KENLM -DKENLM_MAX_ORDER=6 -DTORCH_API_INCLUDE_EXTENSION_H '-DPYBIND11_COMPILER_TYPE="_gcc"' '-DPYBIND11_STDLIB="_libstdcpp"' '-DPYBIND11_BUILD_ABI="_cxxabi1014"' -DTORCH_EXTENSION_NAME=ctc_decode -D_GLIBCXX_USE_CXX11_ABI=1
      FAILED: /tmp/pip-install-fxgtjml5/ctcdecode_a306757a5af04bd1be52dabac4a498b6/build/temp.linux-x86_64-cpython-310/ctcdecode/src/decoder_utils.o
      c++ -MMD -MF /tmp/pip-install-fxgtjml5/ctcdecode_a306757a5af04bd1be52dabac4a498b6/build/temp.linux-x86_64-cpython-310/ctcdecode/src/decoder_utils.o.d -pthread -B /home/system/miniconda3/envs/SLG/compiler_compat -Wno-unused-result -Wsign-compare -DNDEBUG -fwrapv -O2 -Wall -fPIC -O2 -isystem /home/system/miniconda3/envs/SLG/include -fPIC -O2 -isystem /home/system/miniconda3/envs/SLG/include -fPIC -I/tmp/pip-install-fxgtjml5/ctcdecode_a306757a5af04bd1be52dabac4a498b6/third_party/kenlm -I/tmp/pip-install-fxgtjml5/ctcdecode_a306757a5af04bd1be52dabac4a498b6/third_party/openfst-1.6.7/src/include -I/tmp/pip-install-fxgtjml5/ctcdecode_a306757a5af04bd1be52dabac4a498b6/third_party/ThreadPool -I/tmp/pip-install-fxgtjml5/ctcdecode_a306757a5af04bd1be52dabac4a498b6/third_party/boost_1_67_0 -I/tmp/pip-install-fxgtjml5/ctcdecode_a306757a5af04bd1be52dabac4a498b6/third_party/utf8 -I/home/system/miniconda3/envs/SLG/lib/python3.10/site-packages/torch/include -I/home/system/miniconda3/envs/SLG/lib/python3.10/site-packages/torch/include/torch/csrc/api/include -I/home/system/miniconda3/envs/SLG/lib/python3.10/site-packages/torch/include/TH -I/home/system/miniconda3/envs/SLG/lib/python3.10/site-packages/torch/include/THC -I/home/system/miniconda3/envs/SLG/lib/python3.10/site-packages/torch/include -I/home/system/miniconda3/envs/SLG/lib/python3.10/site-packages/torch/include/torch/csrc/api/include -I/home/system/miniconda3/envs/SLG/lib/python3.10/site-packages/torch/include/TH -I/home/system/miniconda3/envs/SLG/lib/python3.10/site-packages/torch/include/THC -I/home/system/miniconda3/envs/SLG/include/python3.10 -c -c /tmp/pip-install-fxgtjml5/ctcdecode_a306757a5af04bd1be52dabac4a498b6/ctcdecode/src/decoder_utils.cpp -o /tmp/pip-install-fxgtjml5/ctcdecode_a306757a5af04bd1be52dabac4a498b6/build/temp.linux-x86_64-cpython-310/ctcdecode/src/decoder_utils.o -O3 -DKENLM_MAX_ORDER=6 -std=c++14 -fPIC -DINCLUDE_KENLM -DKENLM_MAX_ORDER=6 -DTORCH_API_INCLUDE_EXTENSION_H '-DPYBIND11_COMPILER_TYPE="_gcc"' '-DPYBIND11_STDLIB="_libstdcpp"' '-DPYBIND11_BUILD_ABI="_cxxabi1014"' -DTORCH_EXTENSION_NAME=ctc_decode -D_GLIBCXX_USE_CXX11_ABI=1
      /tmp/pip-install-fxgtjml5/ctcdecode_a306757a5af04bd1be52dabac4a498b6/ctcdecode/src/decoder_utils.cpp:1:10: fatal error: decoder_utils.h: No such file or directory
          1 | #include "decoder_utils.h"
            |          ^~~~~~~~~~~~~~~~~
      compilation terminated.
      [3/54] c++ -MMD -MF /tmp/pip-install-fxgtjml5/ctcdecode_a306757a5af04bd1be52dabac4a498b6/build/temp.linux-x86_64-cpython-310/third_party/kenlm/lm/config.o.d -pthread -B /home/system/miniconda3/envs/SLG/compiler_compat -Wno-unused-result -Wsign-compare -DNDEBUG -fwrapv -O2 -Wall -fPIC -O2 -isystem /home/system/miniconda3/envs/SLG/include -fPIC -O2 -isystem /home/system/miniconda3/envs/SLG/include -fPIC -I/tmp/pip-install-fxgtjml5/ctcdecode_a306757a5af04bd1be52dabac4a498b6/third_party/kenlm -I/tmp/pip-install-fxgtjml5/ctcdecode_a306757a5af04bd1be52dabac4a498b6/third_party/openfst-1.6.7/src/include -I/tmp/pip-install-fxgtjml5/ctcdecode_a306757a5af04bd1be52dabac4a498b6/third_party/ThreadPool -I/tmp/pip-install-fxgtjml5/ctcdecode_a306757a5af04bd1be52dabac4a498b6/third_party/boost_1_67_0 -I/tmp/pip-install-fxgtjml5/ctcdecode_a306757a5af04bd1be52dabac4a498b6/third_party/utf8 -I/home/system/miniconda3/envs/SLG/lib/python3.10/site-packages/torch/include -I/home/system/miniconda3/envs/SLG/lib/python3.10/site-packages/torch/include/torch/csrc/api/include -I/home/system/miniconda3/envs/SLG/lib/python3.10/site-packages/torch/include/TH -I/home/system/miniconda3/envs/SLG/lib/python3.10/site-packages/torch/include/THC -I/home/system/miniconda3/envs/SLG/lib/python3.10/site-packages/torch/include -I/home/system/miniconda3/envs/SLG/lib/python3.10/site-packages/torch/include/torch/csrc/api/include -I/home/system/miniconda3/envs/SLG/lib/python3.10/site-packages/torch/include/TH -I/home/system/miniconda3/envs/SLG/lib/python3.10/site-packages/torch/include/THC -I/home/system/miniconda3/envs/SLG/include/python3.10 -c -c /tmp/pip-install-fxgtjml5/ctcdecode_a306757a5af04bd1be52dabac4a498b6/third_party/kenlm/lm/config.cc -o /tmp/pip-install-fxgtjml5/ctcdecode_a306757a5af04bd1be52dabac4a498b6/build/temp.linux-x86_64-cpython-310/third_party/kenlm/lm/config.o -O3 -DKENLM_MAX_ORDER=6 -std=c++14 -fPIC -DINCLUDE_KENLM -DKENLM_MAX_ORDER=6 -DTORCH_API_INCLUDE_EXTENSION_H '-DPYBIND11_COMPILER_TYPE="_gcc"' '-DPYBIND11_STDLIB="_libstdcpp"' '-DPYBIND11_BUILD_ABI="_cxxabi1014"' -DTORCH_EXTENSION_NAME=ctc_decode -D_GLIBCXX_USE_CXX11_ABI=1
      FAILED: /tmp/pip-install-fxgtjml5/ctcdecode_a306757a5af04bd1be52dabac4a498b6/build/temp.linux-x86_64-cpython-310/third_party/kenlm/lm/config.o
      c++ -MMD -MF /tmp/pip-install-fxgtjml5/ctcdecode_a306757a5af04bd1be52dabac4a498b6/build/temp.linux-x86_64-cpython-310/third_party/kenlm/lm/config.o.d -pthread -B /home/system/miniconda3/envs/SLG/compiler_compat -Wno-unused-result -Wsign-compare -DNDEBUG -fwrapv -O2 -Wall -fPIC -O2 -isystem /home/system/miniconda3/envs/SLG/include -fPIC -O2 -isystem /home/system/miniconda3/envs/SLG/include -fPIC -I/tmp/pip-install-fxgtjml5/ctcdecode_a306757a5af04bd1be52dabac4a498b6/third_party/kenlm -I/tmp/pip-install-fxgtjml5/ctcdecode_a306757a5af04bd1be52dabac4a498b6/third_party/openfst-1.6.7/src/include -I/tmp/pip-install-fxgtjml5/ctcdecode_a306757a5af04bd1be52dabac4a498b6/third_party/ThreadPool -I/tmp/pip-install-fxgtjml5/ctcdecode_a306757a5af04bd1be52dabac4a498b6/third_party/boost_1_67_0 -I/tmp/pip-install-fxgtjml5/ctcdecode_a306757a5af04bd1be52dabac4a498b6/third_party/utf8 -I/home/system/miniconda3/envs/SLG/lib/python3.10/site-packages/torch/include -I/home/system/miniconda3/envs/SLG/lib/python3.10/site-packages/torch/include/torch/csrc/api/include -I/home/system/miniconda3/envs/SLG/lib/python3.10/site-packages/torch/include/TH -I/home/system/miniconda3/envs/SLG/lib/python3.10/site-packages/torch/include/THC -I/home/system/miniconda3/envs/SLG/lib/python3.10/site-packages/torch/include -I/home/system/miniconda3/envs/SLG/lib/python3.10/site-packages/torch/include/torch/csrc/api/include -I/home/system/miniconda3/envs/SLG/lib/python3.10/site-packages/torch/include/TH -I/home/system/miniconda3/envs/SLG/lib/python3.10/site-packages/torch/include/THC -I/home/system/miniconda3/envs/SLG/include/python3.10 -c -c /tmp/pip-install-fxgtjml5/ctcdecode_a306757a5af04bd1be52dabac4a498b6/third_party/kenlm/lm/config.cc -o /tmp/pip-install-fxgtjml5/ctcdecode_a306757a5af04bd1be52dabac4a498b6/build/temp.linux-x86_64-cpython-310/third_party/kenlm/lm/config.o -O3 -DKENLM_MAX_ORDER=6 -std=c++14 -fPIC -DINCLUDE_KENLM -DKENLM_MAX_ORDER=6 -DTORCH_API_INCLUDE_EXTENSION_H '-DPYBIND11_COMPILER_TYPE="_gcc"' '-DPYBIND11_STDLIB="_libstdcpp"' '-DPYBIND11_BUILD_ABI="_cxxabi1014"' -DTORCH_EXTENSION_NAME=ctc_decode -D_GLIBCXX_USE_CXX11_ABI=1
      /tmp/pip-install-fxgtjml5/ctcdecode_a306757a5af04bd1be52dabac4a498b6/third_party/kenlm/lm/config.cc:1:10: fatal error: lm/config.hh: No such file or directory
          1 | #include "lm/config.hh"
            |          ^~~~~~~~~~~~~~
      compilation terminated.
      [4/54] c++ -MMD -MF /tmp/pip-install-fxgtjml5/ctcdecode_a306757a5af04bd1be52dabac4a498b6/build/temp.linux-x86_64-cpython-310/third_party/kenlm/lm/binary_format.o.d -pthread -B /home/system/miniconda3/envs/SLG/compiler_compat -Wno-unused-result -Wsign-compare -DNDEBUG -fwrapv -O2 -Wall -fPIC -O2 -isystem /home/system/miniconda3/envs/SLG/include -fPIC -O2 -isystem /home/system/miniconda3/envs/SLG/include -fPIC -I/tmp/pip-install-fxgtjml5/ctcdecode_a306757a5af04bd1be52dabac4a498b6/third_party/kenlm -I/tmp/pip-install-fxgtjml5/ctcdecode_a306757a5af04bd1be52dabac4a498b6/third_party/openfst-1.6.7/src/include -I/tmp/pip-install-fxgtjml5/ctcdecode_a306757a5af04bd1be52dabac4a498b6/third_party/ThreadPool -I/tmp/pip-install-fxgtjml5/ctcdecode_a306757a5af04bd1be52dabac4a498b6/third_party/boost_1_67_0 -I/tmp/pip-install-fxgtjml5/ctcdecode_a306757a5af04bd1be52dabac4a498b6/third_party/utf8 -I/home/system/miniconda3/envs/SLG/lib/python3.10/site-packages/torch/include -I/home/system/miniconda3/envs/SLG/lib/python3.10/site-packages/torch/include/torch/csrc/api/include -I/home/system/miniconda3/envs/SLG/lib/python3.10/site-packages/torch/include/TH -I/home/system/miniconda3/envs/SLG/lib/python3.10/site-packages/torch/include/THC -I/home/system/miniconda3/envs/SLG/lib/python3.10/site-packages/torch/include -I/home/system/miniconda3/envs/SLG/lib/python3.10/site-packages/torch/include/torch/csrc/api/include -I/home/system/miniconda3/envs/SLG/lib/python3.10/site-packages/torch/include/TH -I/home/system/miniconda3/envs/SLG/lib/python3.10/site-packages/torch/include/THC -I/home/system/miniconda3/envs/SLG/include/python3.10 -c -c /tmp/pip-install-fxgtjml5/ctcdecode_a306757a5af04bd1be52dabac4a498b6/third_party/kenlm/lm/binary_format.cc -o /tmp/pip-install-fxgtjml5/ctcdecode_a306757a5af04bd1be52dabac4a498b6/build/temp.linux-x86_64-cpython-310/third_party/kenlm/lm/binary_format.o -O3 -DKENLM_MAX_ORDER=6 -std=c++14 -fPIC -DINCLUDE_KENLM -DKENLM_MAX_ORDER=6 -DTORCH_API_INCLUDE_EXTENSION_H '-DPYBIND11_COMPILER_TYPE="_gcc"' '-DPYBIND11_STDLIB="_libstdcpp"' '-DPYBIND11_BUILD_ABI="_cxxabi1014"' -DTORCH_EXTENSION_NAME=ctc_decode -D_GLIBCXX_USE_CXX11_ABI=1
      FAILED: /tmp/pip-install-fxgtjml5/ctcdecode_a306757a5af04bd1be52dabac4a498b6/build/temp.linux-x86_64-cpython-310/third_party/kenlm/lm/binary_format.o
      c++ -MMD -MF /tmp/pip-install-fxgtjml5/ctcdecode_a306757a5af04bd1be52dabac4a498b6/build/temp.linux-x86_64-cpython-310/third_party/kenlm/lm/binary_format.o.d -pthread -B /home/system/miniconda3/envs/SLG/compiler_compat -Wno-unused-result -Wsign-compare -DNDEBUG -fwrapv -O2 -Wall -fPIC -O2 -isystem /home/system/miniconda3/envs/SLG/include -fPIC -O2 -isystem /home/system/miniconda3/envs/SLG/include -fPIC -I/tmp/pip-install-fxgtjml5/ctcdecode_a306757a5af04bd1be52dabac4a498b6/third_party/kenlm -I/tmp/pip-install-fxgtjml5/ctcdecode_a306757a5af04bd1be52dabac4a498b6/third_party/openfst-1.6.7/src/include -I/tmp/pip-install-fxgtjml5/ctcdecode_a306757a5af04bd1be52dabac4a498b6/third_party/ThreadPool -I/tmp/pip-install-fxgtjml5/ctcdecode_a306757a5af04bd1be52dabac4a498b6/third_party/boost_1_67_0 -I/tmp/pip-install-fxgtjml5/ctcdecode_a306757a5af04bd1be52dabac4a498b6/third_party/utf8 -I/home/system/miniconda3/envs/SLG/lib/python3.10/site-packages/torch/include -I/home/system/miniconda3/envs/SLG/lib/python3.10/site-packages/torch/include/torch/csrc/api/include -I/home/system/miniconda3/envs/SLG/lib/python3.10/site-packages/torch/include/TH -I/home/system/miniconda3/envs/SLG/lib/python3.10/site-packages/torch/include/THC -I/home/system/miniconda3/envs/SLG/lib/python3.10/site-packages/torch/include -I/home/system/miniconda3/envs/SLG/lib/python3.10/site-packages/torch/include/torch/csrc/api/include -I/home/system/miniconda3/envs/SLG/lib/python3.10/site-packages/torch/include/TH -I/home/system/miniconda3/envs/SLG/lib/python3.10/site-packages/torch/include/THC -I/home/system/miniconda3/envs/SLG/include/python3.10 -c -c /tmp/pip-install-fxgtjml5/ctcdecode_a306757a5af04bd1be52dabac4a498b6/third_party/kenlm/lm/binary_format.cc -o /tmp/pip-install-fxgtjml5/ctcdecode_a306757a5af04bd1be52dabac4a498b6/build/temp.linux-x86_64-cpython-310/third_party/kenlm/lm/binary_format.o -O3 -DKENLM_MAX_ORDER=6 -std=c++14 -fPIC -DINCLUDE_KENLM -DKENLM_MAX_ORDER=6 -DTORCH_API_INCLUDE_EXTENSION_H '-DPYBIND11_COMPILER_TYPE="_gcc"' '-DPYBIND11_STDLIB="_libstdcpp"' '-DPYBIND11_BUILD_ABI="_cxxabi1014"' -DTORCH_EXTENSION_NAME=ctc_decode -D_GLIBCXX_USE_CXX11_ABI=1
      /tmp/pip-install-fxgtjml5/ctcdecode_a306757a5af04bd1be52dabac4a498b6/third_party/kenlm/lm/binary_format.cc:1:10: fatal error: lm/binary_format.hh: No such file or directory
          1 | #include "lm/binary_format.hh"
            |          ^~~~~~~~~~~~~~~~~~~~~
      compilation terminated.
      [5/54] c++ -MMD -MF /tmp/pip-install-fxgtjml5/ctcdecode_a306757a5af04bd1be52dabac4a498b6/build/temp.linux-x86_64-cpython-310/third_party/kenlm/lm/lm_exception.o.d -pthread -B /home/system/miniconda3/envs/SLG/compiler_compat -Wno-unused-result -Wsign-compare -DNDEBUG -fwrapv -O2 -Wall -fPIC -O2 -isystem /home/system/miniconda3/envs/SLG/include -fPIC -O2 -isystem /home/system/miniconda3/envs/SLG/include -fPIC -I/tmp/pip-install-fxgtjml5/ctcdecode_a306757a5af04bd1be52dabac4a498b6/third_party/kenlm -I/tmp/pip-install-fxgtjml5/ctcdecode_a306757a5af04bd1be52dabac4a498b6/third_party/openfst-1.6.7/src/include -I/tmp/pip-install-fxgtjml5/ctcdecode_a306757a5af04bd1be52dabac4a498b6/third_party/ThreadPool -I/tmp/pip-install-fxgtjml5/ctcdecode_a306757a5af04bd1be52dabac4a498b6/third_party/boost_1_67_0 -I/tmp/pip-install-fxgtjml5/ctcdecode_a306757a5af04bd1be52dabac4a498b6/third_party/utf8 -I/home/system/miniconda3/envs/SLG/lib/python3.10/site-packages/torch/include -I/home/system/miniconda3/envs/SLG/lib/python3.10/site-packages/torch/include/torch/csrc/api/include -I/home/system/miniconda3/envs/SLG/lib/python3.10/site-packages/torch/include/TH -I/home/system/miniconda3/envs/SLG/lib/python3.10/site-packages/torch/include/THC -I/home/system/miniconda3/envs/SLG/lib/python3.10/site-packages/torch/include -I/home/system/miniconda3/envs/SLG/lib/python3.10/site-packages/torch/include/torch/csrc/api/include -I/home/system/miniconda3/envs/SLG/lib/python3.10/site-packages/torch/include/TH -I/home/system/miniconda3/envs/SLG/lib/python3.10/site-packages/torch/include/THC -I/home/system/miniconda3/envs/SLG/include/python3.10 -c -c /tmp/pip-install-fxgtjml5/ctcdecode_a306757a5af04bd1be52dabac4a498b6/third_party/kenlm/lm/lm_exception.cc -o /tmp/pip-install-fxgtjml5/ctcdecode_a306757a5af04bd1be52dabac4a498b6/build/temp.linux-x86_64-cpython-310/third_party/kenlm/lm/lm_exception.o -O3 -DKENLM_MAX_ORDER=6 -std=c++14 -fPIC -DINCLUDE_KENLM -DKENLM_MAX_ORDER=6 -DTORCH_API_INCLUDE_EXTENSION_H '-DPYBIND11_COMPILER_TYPE="_gcc"' '-DPYBIND11_STDLIB="_libstdcpp"' '-DPYBIND11_BUILD_ABI="_cxxabi1014"' -DTORCH_EXTENSION_NAME=ctc_decode -D_GLIBCXX_USE_CXX11_ABI=1
      FAILED: /tmp/pip-install-fxgtjml5/ctcdecode_a306757a5af04bd1be52dabac4a498b6/build/temp.linux-x86_64-cpython-310/third_party/kenlm/lm/lm_exception.o
      c++ -MMD -MF /tmp/pip-install-fxgtjml5/ctcdecode_a306757a5af04bd1be52dabac4a498b6/build/temp.linux-x86_64-cpython-310/third_party/kenlm/lm/lm_exception.o.d -pthread -B /home/system/miniconda3/envs/SLG/compiler_compat -Wno-unused-result -Wsign-compare -DNDEBUG -fwrapv -O2 -Wall -fPIC -O2 -isystem /home/system/miniconda3/envs/SLG/include -fPIC -O2 -isystem /home/system/miniconda3/envs/SLG/include -fPIC -I/tmp/pip-install-fxgtjml5/ctcdecode_a306757a5af04bd1be52dabac4a498b6/third_party/kenlm -I/tmp/pip-install-fxgtjml5/ctcdecode_a306757a5af04bd1be52dabac4a498b6/third_party/openfst-1.6.7/src/include -I/tmp/pip-install-fxgtjml5/ctcdecode_a306757a5af04bd1be52dabac4a498b6/third_party/ThreadPool -I/tmp/pip-install-fxgtjml5/ctcdecode_a306757a5af04bd1be52dabac4a498b6/third_party/boost_1_67_0 -I/tmp/pip-install-fxgtjml5/ctcdecode_a306757a5af04bd1be52dabac4a498b6/third_party/utf8 -I/home/system/miniconda3/envs/SLG/lib/python3.10/site-packages/torch/include -I/home/system/miniconda3/envs/SLG/lib/python3.10/site-packages/torch/include/torch/csrc/api/include -I/home/system/miniconda3/envs/SLG/lib/python3.10/site-packages/torch/include/TH -I/home/system/miniconda3/envs/SLG/lib/python3.10/site-packages/torch/include/THC -I/home/system/miniconda3/envs/SLG/lib/python3.10/site-packages/torch/include -I/home/system/miniconda3/envs/SLG/lib/python3.10/site-packages/torch/include/torch/csrc/api/include -I/home/system/miniconda3/envs/SLG/lib/python3.10/site-packages/torch/include/TH -I/home/system/miniconda3/envs/SLG/lib/python3.10/site-packages/torch/include/THC -I/home/system/miniconda3/envs/SLG/include/python3.10 -c -c /tmp/pip-install-fxgtjml5/ctcdecode_a306757a5af04bd1be52dabac4a498b6/third_party/kenlm/lm/lm_exception.cc -o /tmp/pip-install-fxgtjml5/ctcdecode_a306757a5af04bd1be52dabac4a498b6/build/temp.linux-x86_64-cpython-310/third_party/kenlm/lm/lm_exception.o -O3 -DKENLM_MAX_ORDER=6 -std=c++14 -fPIC -DINCLUDE_KENLM -DKENLM_MAX_ORDER=6 -DTORCH_API_INCLUDE_EXTENSION_H '-DPYBIND11_COMPILER_TYPE="_gcc"' '-DPYBIND11_STDLIB="_libstdcpp"' '-DPYBIND11_BUILD_ABI="_cxxabi1014"' -DTORCH_EXTENSION_NAME=ctc_decode -D_GLIBCXX_USE_CXX11_ABI=1
      /tmp/pip-install-fxgtjml5/ctcdecode_a306757a5af04bd1be52dabac4a498b6/third_party/kenlm/lm/lm_exception.cc:1:10: fatal error: lm/lm_exception.hh: No such file or directory
          1 | #include "lm/lm_exception.hh"
            |          ^~~~~~~~~~~~~~~~~~~~
      compilation terminated.
      [6/54] c++ -MMD -MF /tmp/pip-install-fxgtjml5/ctcdecode_a306757a5af04bd1be52dabac4a498b6/build/temp.linux-x86_64-cpython-310/third_party/kenlm/lm/model.o.d -pthread -B /home/system/miniconda3/envs/SLG/compiler_compat -Wno-unused-result -Wsign-compare -DNDEBUG -fwrapv -O2 -Wall -fPIC -O2 -isystem /home/system/miniconda3/envs/SLG/include -fPIC -O2 -isystem /home/system/miniconda3/envs/SLG/include -fPIC -I/tmp/pip-install-fxgtjml5/ctcdecode_a306757a5af04bd1be52dabac4a498b6/third_party/kenlm -I/tmp/pip-install-fxgtjml5/ctcdecode_a306757a5af04bd1be52dabac4a498b6/third_party/openfst-1.6.7/src/include -I/tmp/pip-install-fxgtjml5/ctcdecode_a306757a5af04bd1be52dabac4a498b6/third_party/ThreadPool -I/tmp/pip-install-fxgtjml5/ctcdecode_a306757a5af04bd1be52dabac4a498b6/third_party/boost_1_67_0 -I/tmp/pip-install-fxgtjml5/ctcdecode_a306757a5af04bd1be52dabac4a498b6/third_party/utf8 -I/home/system/miniconda3/envs/SLG/lib/python3.10/site-packages/torch/include -I/home/system/miniconda3/envs/SLG/lib/python3.10/site-packages/torch/include/torch/csrc/api/include -I/home/system/miniconda3/envs/SLG/lib/python3.10/site-packages/torch/include/TH -I/home/system/miniconda3/envs/SLG/lib/python3.10/site-packages/torch/include/THC -I/home/system/miniconda3/envs/SLG/lib/python3.10/site-packages/torch/include -I/home/system/miniconda3/envs/SLG/lib/python3.10/site-packages/torch/include/torch/csrc/api/include -I/home/system/miniconda3/envs/SLG/lib/python3.10/site-packages/torch/include/TH -I/home/system/miniconda3/envs/SLG/lib/python3.10/site-packages/torch/include/THC -I/home/system/miniconda3/envs/SLG/include/python3.10 -c -c /tmp/pip-install-fxgtjml5/ctcdecode_a306757a5af04bd1be52dabac4a498b6/third_party/kenlm/lm/model.cc -o /tmp/pip-install-fxgtjml5/ctcdecode_a306757a5af04bd1be52dabac4a498b6/build/temp.linux-x86_64-cpython-310/third_party/kenlm/lm/model.o -O3 -DKENLM_MAX_ORDER=6 -std=c++14 -fPIC -DINCLUDE_KENLM -DKENLM_MAX_ORDER=6 -DTORCH_API_INCLUDE_EXTENSION_H '-DPYBIND11_COMPILER_TYPE="_gcc"' '-DPYBIND11_STDLIB="_libstdcpp"' '-DPYBIND11_BUILD_ABI="_cxxabi1014"' -DTORCH_EXTENSION_NAME=ctc_decode -D_GLIBCXX_USE_CXX11_ABI=1
      FAILED: /tmp/pip-install-fxgtjml5/ctcdecode_a306757a5af04bd1be52dabac4a498b6/build/temp.linux-x86_64-cpython-310/third_party/kenlm/lm/model.o
      c++ -MMD -MF /tmp/pip-install-fxgtjml5/ctcdecode_a306757a5af04bd1be52dabac4a498b6/build/temp.linux-x86_64-cpython-310/third_party/kenlm/lm/model.o.d -pthread -B /home/system/miniconda3/envs/SLG/compiler_compat -Wno-unused-result -Wsign-compare -DNDEBUG -fwrapv -O2 -Wall -fPIC -O2 -isystem /home/system/miniconda3/envs/SLG/include -fPIC -O2 -isystem /home/system/miniconda3/envs/SLG/include -fPIC -I/tmp/pip-install-fxgtjml5/ctcdecode_a306757a5af04bd1be52dabac4a498b6/third_party/kenlm -I/tmp/pip-install-fxgtjml5/ctcdecode_a306757a5af04bd1be52dabac4a498b6/third_party/openfst-1.6.7/src/include -I/tmp/pip-install-fxgtjml5/ctcdecode_a306757a5af04bd1be52dabac4a498b6/third_party/ThreadPool -I/tmp/pip-install-fxgtjml5/ctcdecode_a306757a5af04bd1be52dabac4a498b6/third_party/boost_1_67_0 -I/tmp/pip-install-fxgtjml5/ctcdecode_a306757a5af04bd1be52dabac4a498b6/third_party/utf8 -I/home/system/miniconda3/envs/SLG/lib/python3.10/site-packages/torch/include -I/home/system/miniconda3/envs/SLG/lib/python3.10/site-packages/torch/include/torch/csrc/api/include -I/home/system/miniconda3/envs/SLG/lib/python3.10/site-packages/torch/include/TH -I/home/system/miniconda3/envs/SLG/lib/python3.10/site-packages/torch/include/THC -I/home/system/miniconda3/envs/SLG/lib/python3.10/site-packages/torch/include -I/home/system/miniconda3/envs/SLG/lib/python3.10/site-packages/torch/include/torch/csrc/api/include -I/home/system/miniconda3/envs/SLG/lib/python3.10/site-packages/torch/include/TH -I/home/system/miniconda3/envs/SLG/lib/python3.10/site-packages/torch/include/THC -I/home/system/miniconda3/envs/SLG/include/python3.10 -c -c /tmp/pip-install-fxgtjml5/ctcdecode_a306757a5af04bd1be52dabac4a498b6/third_party/kenlm/lm/model.cc -o /tmp/pip-install-fxgtjml5/ctcdecode_a306757a5af04bd1be52dabac4a498b6/build/temp.linux-x86_64-cpython-310/third_party/kenlm/lm/model.o -O3 -DKENLM_MAX_ORDER=6 -std=c++14 -fPIC -DINCLUDE_KENLM -DKENLM_MAX_ORDER=6 -DTORCH_API_INCLUDE_EXTENSION_H '-DPYBIND11_COMPILER_TYPE="_gcc"' '-DPYBIND11_STDLIB="_libstdcpp"' '-DPYBIND11_BUILD_ABI="_cxxabi1014"' -DTORCH_EXTENSION_NAME=ctc_decode -D_GLIBCXX_USE_CXX11_ABI=1
      /tmp/pip-install-fxgtjml5/ctcdecode_a306757a5af04bd1be52dabac4a498b6/third_party/kenlm/lm/model.cc:1:10: fatal error: lm/model.hh: No such file or directory
          1 | #include "lm/model.hh"
            |          ^~~~~~~~~~~~~
      compilation terminated.
      [7/54] c++ -MMD -MF /tmp/pip-install-fxgtjml5/ctcdecode_a306757a5af04bd1be52dabac4a498b6/build/temp.linux-x86_64-cpython-310/ctcdecode/src/path_trie.o.d -pthread -B /home/system/miniconda3/envs/SLG/compiler_compat -Wno-unused-result -Wsign-compare -DNDEBUG -fwrapv -O2 -Wall -fPIC -O2 -isystem /home/system/miniconda3/envs/SLG/include -fPIC -O2 -isystem /home/system/miniconda3/envs/SLG/include -fPIC -I/tmp/pip-install-fxgtjml5/ctcdecode_a306757a5af04bd1be52dabac4a498b6/third_party/kenlm -I/tmp/pip-install-fxgtjml5/ctcdecode_a306757a5af04bd1be52dabac4a498b6/third_party/openfst-1.6.7/src/include -I/tmp/pip-install-fxgtjml5/ctcdecode_a306757a5af04bd1be52dabac4a498b6/third_party/ThreadPool -I/tmp/pip-install-fxgtjml5/ctcdecode_a306757a5af04bd1be52dabac4a498b6/third_party/boost_1_67_0 -I/tmp/pip-install-fxgtjml5/ctcdecode_a306757a5af04bd1be52dabac4a498b6/third_party/utf8 -I/home/system/miniconda3/envs/SLG/lib/python3.10/site-packages/torch/include -I/home/system/miniconda3/envs/SLG/lib/python3.10/site-packages/torch/include/torch/csrc/api/include -I/home/system/miniconda3/envs/SLG/lib/python3.10/site-packages/torch/include/TH -I/home/system/miniconda3/envs/SLG/lib/python3.10/site-packages/torch/include/THC -I/home/system/miniconda3/envs/SLG/lib/python3.10/site-packages/torch/include -I/home/system/miniconda3/envs/SLG/lib/python3.10/site-packages/torch/include/torch/csrc/api/include -I/home/system/miniconda3/envs/SLG/lib/python3.10/site-packages/torch/include/TH -I/home/system/miniconda3/envs/SLG/lib/python3.10/site-packages/torch/include/THC -I/home/system/miniconda3/envs/SLG/include/python3.10 -c -c /tmp/pip-install-fxgtjml5/ctcdecode_a306757a5af04bd1be52dabac4a498b6/ctcdecode/src/path_trie.cpp -o /tmp/pip-install-fxgtjml5/ctcdecode_a306757a5af04bd1be52dabac4a498b6/build/temp.linux-x86_64-cpython-310/ctcdecode/src/path_trie.o -O3 -DKENLM_MAX_ORDER=6 -std=c++14 -fPIC -DINCLUDE_KENLM -DKENLM_MAX_ORDER=6 -DTORCH_API_INCLUDE_EXTENSION_H '-DPYBIND11_COMPILER_TYPE="_gcc"' '-DPYBIND11_STDLIB="_libstdcpp"' '-DPYBIND11_BUILD_ABI="_cxxabi1014"' -DTORCH_EXTENSION_NAME=ctc_decode -D_GLIBCXX_USE_CXX11_ABI=1
      FAILED: /tmp/pip-install-fxgtjml5/ctcdecode_a306757a5af04bd1be52dabac4a498b6/build/temp.linux-x86_64-cpython-310/ctcdecode/src/path_trie.o
      c++ -MMD -MF /tmp/pip-install-fxgtjml5/ctcdecode_a306757a5af04bd1be52dabac4a498b6/build/temp.linux-x86_64-cpython-310/ctcdecode/src/path_trie.o.d -pthread -B /home/system/miniconda3/envs/SLG/compiler_compat -Wno-unused-result -Wsign-compare -DNDEBUG -fwrapv -O2 -Wall -fPIC -O2 -isystem /home/system/miniconda3/envs/SLG/include -fPIC -O2 -isystem /home/system/miniconda3/envs/SLG/include -fPIC -I/tmp/pip-install-fxgtjml5/ctcdecode_a306757a5af04bd1be52dabac4a498b6/third_party/kenlm -I/tmp/pip-install-fxgtjml5/ctcdecode_a306757a5af04bd1be52dabac4a498b6/third_party/openfst-1.6.7/src/include -I/tmp/pip-install-fxgtjml5/ctcdecode_a306757a5af04bd1be52dabac4a498b6/third_party/ThreadPool -I/tmp/pip-install-fxgtjml5/ctcdecode_a306757a5af04bd1be52dabac4a498b6/third_party/boost_1_67_0 -I/tmp/pip-install-fxgtjml5/ctcdecode_a306757a5af04bd1be52dabac4a498b6/third_party/utf8 -I/home/system/miniconda3/envs/SLG/lib/python3.10/site-packages/torch/include -I/home/system/miniconda3/envs/SLG/lib/python3.10/site-packages/torch/include/torch/csrc/api/include -I/home/system/miniconda3/envs/SLG/lib/python3.10/site-packages/torch/include/TH -I/home/system/miniconda3/envs/SLG/lib/python3.10/site-packages/torch/include/THC -I/home/system/miniconda3/envs/SLG/lib/python3.10/site-packages/torch/include -I/home/system/miniconda3/envs/SLG/lib/python3.10/site-packages/torch/include/torch/csrc/api/include -I/home/system/miniconda3/envs/SLG/lib/python3.10/site-packages/torch/include/TH -I/home/system/miniconda3/envs/SLG/lib/python3.10/site-packages/torch/include/THC -I/home/system/miniconda3/envs/SLG/include/python3.10 -c -c /tmp/pip-install-fxgtjml5/ctcdecode_a306757a5af04bd1be52dabac4a498b6/ctcdecode/src/path_trie.cpp -o /tmp/pip-install-fxgtjml5/ctcdecode_a306757a5af04bd1be52dabac4a498b6/build/temp.linux-x86_64-cpython-310/ctcdecode/src/path_trie.o -O3 -DKENLM_MAX_ORDER=6 -std=c++14 -fPIC -DINCLUDE_KENLM -DKENLM_MAX_ORDER=6 -DTORCH_API_INCLUDE_EXTENSION_H '-DPYBIND11_COMPILER_TYPE="_gcc"' '-DPYBIND11_STDLIB="_libstdcpp"' '-DPYBIND11_BUILD_ABI="_cxxabi1014"' -DTORCH_EXTENSION_NAME=ctc_decode -D_GLIBCXX_USE_CXX11_ABI=1
      /tmp/pip-install-fxgtjml5/ctcdecode_a306757a5af04bd1be52dabac4a498b6/ctcdecode/src/path_trie.cpp:1:10: fatal error: path_trie.h: No such file or directory
          1 | #include "path_trie.h"
            |          ^~~~~~~~~~~~~
      compilation terminated.
      [8/54] c++ -MMD -MF /tmp/pip-install-fxgtjml5/ctcdecode_a306757a5af04bd1be52dabac4a498b6/build/temp.linux-x86_64-cpython-310/third_party/kenlm/lm/bhiksha.o.d -pthread -B /home/system/miniconda3/envs/SLG/compiler_compat -Wno-unused-result -Wsign-compare -DNDEBUG -fwrapv -O2 -Wall -fPIC -O2 -isystem /home/system/miniconda3/envs/SLG/include -fPIC -O2 -isystem /home/system/miniconda3/envs/SLG/include -fPIC -I/tmp/pip-install-fxgtjml5/ctcdecode_a306757a5af04bd1be52dabac4a498b6/third_party/kenlm -I/tmp/pip-install-fxgtjml5/ctcdecode_a306757a5af04bd1be52dabac4a498b6/third_party/openfst-1.6.7/src/include -I/tmp/pip-install-fxgtjml5/ctcdecode_a306757a5af04bd1be52dabac4a498b6/third_party/ThreadPool -I/tmp/pip-install-fxgtjml5/ctcdecode_a306757a5af04bd1be52dabac4a498b6/third_party/boost_1_67_0 -I/tmp/pip-install-fxgtjml5/ctcdecode_a306757a5af04bd1be52dabac4a498b6/third_party/utf8 -I/home/system/miniconda3/envs/SLG/lib/python3.10/site-packages/torch/include -I/home/system/miniconda3/envs/SLG/lib/python3.10/site-packages/torch/include/torch/csrc/api/include -I/home/system/miniconda3/envs/SLG/lib/python3.10/site-packages/torch/include/TH -I/home/system/miniconda3/envs/SLG/lib/python3.10/site-packages/torch/include/THC -I/home/system/miniconda3/envs/SLG/lib/python3.10/site-packages/torch/include -I/home/system/miniconda3/envs/SLG/lib/python3.10/site-packages/torch/include/torch/csrc/api/include -I/home/system/miniconda3/envs/SLG/lib/python3.10/site-packages/torch/include/TH -I/home/system/miniconda3/envs/SLG/lib/python3.10/site-packages/torch/include/THC -I/home/system/miniconda3/envs/SLG/include/python3.10 -c -c /tmp/pip-install-fxgtjml5/ctcdecode_a306757a5af04bd1be52dabac4a498b6/third_party/kenlm/lm/bhiksha.cc -o /tmp/pip-install-fxgtjml5/ctcdecode_a306757a5af04bd1be52dabac4a498b6/build/temp.linux-x86_64-cpython-310/third_party/kenlm/lm/bhiksha.o -O3 -DKENLM_MAX_ORDER=6 -std=c++14 -fPIC -DINCLUDE_KENLM -DKENLM_MAX_ORDER=6 -DTORCH_API_INCLUDE_EXTENSION_H '-DPYBIND11_COMPILER_TYPE="_gcc"' '-DPYBIND11_STDLIB="_libstdcpp"' '-DPYBIND11_BUILD_ABI="_cxxabi1014"' -DTORCH_EXTENSION_NAME=ctc_decode -D_GLIBCXX_USE_CXX11_ABI=1
      FAILED: /tmp/pip-install-fxgtjml5/ctcdecode_a306757a5af04bd1be52dabac4a498b6/build/temp.linux-x86_64-cpython-310/third_party/kenlm/lm/bhiksha.o
      c++ -MMD -MF /tmp/pip-install-fxgtjml5/ctcdecode_a306757a5af04bd1be52dabac4a498b6/build/temp.linux-x86_64-cpython-310/third_party/kenlm/lm/bhiksha.o.d -pthread -B /home/system/miniconda3/envs/SLG/compiler_compat -Wno-unused-result -Wsign-compare -DNDEBUG -fwrapv -O2 -Wall -fPIC -O2 -isystem /home/system/miniconda3/envs/SLG/include -fPIC -O2 -isystem /home/system/miniconda3/envs/SLG/include -fPIC -I/tmp/pip-install-fxgtjml5/ctcdecode_a306757a5af04bd1be52dabac4a498b6/third_party/kenlm -I/tmp/pip-install-fxgtjml5/ctcdecode_a306757a5af04bd1be52dabac4a498b6/third_party/openfst-1.6.7/src/include -I/tmp/pip-install-fxgtjml5/ctcdecode_a306757a5af04bd1be52dabac4a498b6/third_party/ThreadPool -I/tmp/pip-install-fxgtjml5/ctcdecode_a306757a5af04bd1be52dabac4a498b6/third_party/boost_1_67_0 -I/tmp/pip-install-fxgtjml5/ctcdecode_a306757a5af04bd1be52dabac4a498b6/third_party/utf8 -I/home/system/miniconda3/envs/SLG/lib/python3.10/site-packages/torch/include -I/home/system/miniconda3/envs/SLG/lib/python3.10/site-packages/torch/include/torch/csrc/api/include -I/home/system/miniconda3/envs/SLG/lib/python3.10/site-packages/torch/include/TH -I/home/system/miniconda3/envs/SLG/lib/python3.10/site-packages/torch/include/THC -I/home/system/miniconda3/envs/SLG/lib/python3.10/site-packages/torch/include -I/home/system/miniconda3/envs/SLG/lib/python3.10/site-packages/torch/include/torch/csrc/api/include -I/home/system/miniconda3/envs/SLG/lib/python3.10/site-packages/torch/include/TH -I/home/system/miniconda3/envs/SLG/lib/python3.10/site-packages/torch/include/THC -I/home/system/miniconda3/envs/SLG/include/python3.10 -c -c /tmp/pip-install-fxgtjml5/ctcdecode_a306757a5af04bd1be52dabac4a498b6/third_party/kenlm/lm/bhiksha.cc -o /tmp/pip-install-fxgtjml5/ctcdecode_a306757a5af04bd1be52dabac4a498b6/build/temp.linux-x86_64-cpython-310/third_party/kenlm/lm/bhiksha.o -O3 -DKENLM_MAX_ORDER=6 -std=c++14 -fPIC -DINCLUDE_KENLM -DKENLM_MAX_ORDER=6 -DTORCH_API_INCLUDE_EXTENSION_H '-DPYBIND11_COMPILER_TYPE="_gcc"' '-DPYBIND11_STDLIB="_libstdcpp"' '-DPYBIND11_BUILD_ABI="_cxxabi1014"' -DTORCH_EXTENSION_NAME=ctc_decode -D_GLIBCXX_USE_CXX11_ABI=1
      /tmp/pip-install-fxgtjml5/ctcdecode_a306757a5af04bd1be52dabac4a498b6/third_party/kenlm/lm/bhiksha.cc:1:10: fatal error: lm/bhiksha.hh: No such file or directory
          1 | #include "lm/bhiksha.hh"
            |          ^~~~~~~~~~~~~~~
      compilation terminated.
      [9/54] c++ -MMD -MF /tmp/pip-install-fxgtjml5/ctcdecode_a306757a5af04bd1be52dabac4a498b6/build/temp.linux-x86_64-cpython-310/ctcdecode/src/scorer.o.d -pthread -B /home/system/miniconda3/envs/SLG/compiler_compat -Wno-unused-result -Wsign-compare -DNDEBUG -fwrapv -O2 -Wall -fPIC -O2 -isystem /home/system/miniconda3/envs/SLG/include -fPIC -O2 -isystem /home/system/miniconda3/envs/SLG/include -fPIC -I/tmp/pip-install-fxgtjml5/ctcdecode_a306757a5af04bd1be52dabac4a498b6/third_party/kenlm -I/tmp/pip-install-fxgtjml5/ctcdecode_a306757a5af04bd1be52dabac4a498b6/third_party/openfst-1.6.7/src/include -I/tmp/pip-install-fxgtjml5/ctcdecode_a306757a5af04bd1be52dabac4a498b6/third_party/ThreadPool -I/tmp/pip-install-fxgtjml5/ctcdecode_a306757a5af04bd1be52dabac4a498b6/third_party/boost_1_67_0 -I/tmp/pip-install-fxgtjml5/ctcdecode_a306757a5af04bd1be52dabac4a498b6/third_party/utf8 -I/home/system/miniconda3/envs/SLG/lib/python3.10/site-packages/torch/include -I/home/system/miniconda3/envs/SLG/lib/python3.10/site-packages/torch/include/torch/csrc/api/include -I/home/system/miniconda3/envs/SLG/lib/python3.10/site-packages/torch/include/TH -I/home/system/miniconda3/envs/SLG/lib/python3.10/site-packages/torch/include/THC -I/home/system/miniconda3/envs/SLG/lib/python3.10/site-packages/torch/include -I/home/system/miniconda3/envs/SLG/lib/python3.10/site-packages/torch/include/torch/csrc/api/include -I/home/system/miniconda3/envs/SLG/lib/python3.10/site-packages/torch/include/TH -I/home/system/miniconda3/envs/SLG/lib/python3.10/site-packages/torch/include/THC -I/home/system/miniconda3/envs/SLG/include/python3.10 -c -c /tmp/pip-install-fxgtjml5/ctcdecode_a306757a5af04bd1be52dabac4a498b6/ctcdecode/src/scorer.cpp -o /tmp/pip-install-fxgtjml5/ctcdecode_a306757a5af04bd1be52dabac4a498b6/build/temp.linux-x86_64-cpython-310/ctcdecode/src/scorer.o -O3 -DKENLM_MAX_ORDER=6 -std=c++14 -fPIC -DINCLUDE_KENLM -DKENLM_MAX_ORDER=6 -DTORCH_API_INCLUDE_EXTENSION_H '-DPYBIND11_COMPILER_TYPE="_gcc"' '-DPYBIND11_STDLIB="_libstdcpp"' '-DPYBIND11_BUILD_ABI="_cxxabi1014"' -DTORCH_EXTENSION_NAME=ctc_decode -D_GLIBCXX_USE_CXX11_ABI=1
      FAILED: /tmp/pip-install-fxgtjml5/ctcdecode_a306757a5af04bd1be52dabac4a498b6/build/temp.linux-x86_64-cpython-310/ctcdecode/src/scorer.o
      c++ -MMD -MF /tmp/pip-install-fxgtjml5/ctcdecode_a306757a5af04bd1be52dabac4a498b6/build/temp.linux-x86_64-cpython-310/ctcdecode/src/scorer.o.d -pthread -B /home/system/miniconda3/envs/SLG/compiler_compat -Wno-unused-result -Wsign-compare -DNDEBUG -fwrapv -O2 -Wall -fPIC -O2 -isystem /home/system/miniconda3/envs/SLG/include -fPIC -O2 -isystem /home/system/miniconda3/envs/SLG/include -fPIC -I/tmp/pip-install-fxgtjml5/ctcdecode_a306757a5af04bd1be52dabac4a498b6/third_party/kenlm -I/tmp/pip-install-fxgtjml5/ctcdecode_a306757a5af04bd1be52dabac4a498b6/third_party/openfst-1.6.7/src/include -I/tmp/pip-install-fxgtjml5/ctcdecode_a306757a5af04bd1be52dabac4a498b6/third_party/ThreadPool -I/tmp/pip-install-fxgtjml5/ctcdecode_a306757a5af04bd1be52dabac4a498b6/third_party/boost_1_67_0 -I/tmp/pip-install-fxgtjml5/ctcdecode_a306757a5af04bd1be52dabac4a498b6/third_party/utf8 -I/home/system/miniconda3/envs/SLG/lib/python3.10/site-packages/torch/include -I/home/system/miniconda3/envs/SLG/lib/python3.10/site-packages/torch/include/torch/csrc/api/include -I/home/system/miniconda3/envs/SLG/lib/python3.10/site-packages/torch/include/TH -I/home/system/miniconda3/envs/SLG/lib/python3.10/site-packages/torch/include/THC -I/home/system/miniconda3/envs/SLG/lib/python3.10/site-packages/torch/include -I/home/system/miniconda3/envs/SLG/lib/python3.10/site-packages/torch/include/torch/csrc/api/include -I/home/system/miniconda3/envs/SLG/lib/python3.10/site-packages/torch/include/TH -I/home/system/miniconda3/envs/SLG/lib/python3.10/site-packages/torch/include/THC -I/home/system/miniconda3/envs/SLG/include/python3.10 -c -c /tmp/pip-install-fxgtjml5/ctcdecode_a306757a5af04bd1be52dabac4a498b6/ctcdecode/src/scorer.cpp -o /tmp/pip-install-fxgtjml5/ctcdecode_a306757a5af04bd1be52dabac4a498b6/build/temp.linux-x86_64-cpython-310/ctcdecode/src/scorer.o -O3 -DKENLM_MAX_ORDER=6 -std=c++14 -fPIC -DINCLUDE_KENLM -DKENLM_MAX_ORDER=6 -DTORCH_API_INCLUDE_EXTENSION_H '-DPYBIND11_COMPILER_TYPE="_gcc"' '-DPYBIND11_STDLIB="_libstdcpp"' '-DPYBIND11_BUILD_ABI="_cxxabi1014"' -DTORCH_EXTENSION_NAME=ctc_decode -D_GLIBCXX_USE_CXX11_ABI=1
      /tmp/pip-install-fxgtjml5/ctcdecode_a306757a5af04bd1be52dabac4a498b6/ctcdecode/src/scorer.cpp:1:10: fatal error: scorer.h: No such file or directory
          1 | #include "scorer.h"
            |          ^~~~~~~~~~
      compilation terminated.
      [10/54] c++ -MMD -MF /tmp/pip-install-fxgtjml5/ctcdecode_a306757a5af04bd1be52dabac4a498b6/build/temp.linux-x86_64-cpython-310/ctcdecode/src/binding.o.d -pthread -B /home/system/miniconda3/envs/SLG/compiler_compat -Wno-unused-result -Wsign-compare -DNDEBUG -fwrapv -O2 -Wall -fPIC -O2 -isystem /home/system/miniconda3/envs/SLG/include -fPIC -O2 -isystem /home/system/miniconda3/envs/SLG/include -fPIC -I/tmp/pip-install-fxgtjml5/ctcdecode_a306757a5af04bd1be52dabac4a498b6/third_party/kenlm -I/tmp/pip-install-fxgtjml5/ctcdecode_a306757a5af04bd1be52dabac4a498b6/third_party/openfst-1.6.7/src/include -I/tmp/pip-install-fxgtjml5/ctcdecode_a306757a5af04bd1be52dabac4a498b6/third_party/ThreadPool -I/tmp/pip-install-fxgtjml5/ctcdecode_a306757a5af04bd1be52dabac4a498b6/third_party/boost_1_67_0 -I/tmp/pip-install-fxgtjml5/ctcdecode_a306757a5af04bd1be52dabac4a498b6/third_party/utf8 -I/home/system/miniconda3/envs/SLG/lib/python3.10/site-packages/torch/include -I/home/system/miniconda3/envs/SLG/lib/python3.10/site-packages/torch/include/torch/csrc/api/include -I/home/system/miniconda3/envs/SLG/lib/python3.10/site-packages/torch/include/TH -I/home/system/miniconda3/envs/SLG/lib/python3.10/site-packages/torch/include/THC -I/home/system/miniconda3/envs/SLG/lib/python3.10/site-packages/torch/include -I/home/system/miniconda3/envs/SLG/lib/python3.10/site-packages/torch/include/torch/csrc/api/include -I/home/system/miniconda3/envs/SLG/lib/python3.10/site-packages/torch/include/TH -I/home/system/miniconda3/envs/SLG/lib/python3.10/site-packages/torch/include/THC -I/home/system/miniconda3/envs/SLG/include/python3.10 -c -c /tmp/pip-install-fxgtjml5/ctcdecode_a306757a5af04bd1be52dabac4a498b6/ctcdecode/src/binding.cpp -o /tmp/pip-install-fxgtjml5/ctcdecode_a306757a5af04bd1be52dabac4a498b6/build/temp.linux-x86_64-cpython-310/ctcdecode/src/binding.o -O3 -DKENLM_MAX_ORDER=6 -std=c++14 -fPIC -DINCLUDE_KENLM -DKENLM_MAX_ORDER=6 -DTORCH_API_INCLUDE_EXTENSION_H '-DPYBIND11_COMPILER_TYPE="_gcc"' '-DPYBIND11_STDLIB="_libstdcpp"' '-DPYBIND11_BUILD_ABI="_cxxabi1014"' -DTORCH_EXTENSION_NAME=ctc_decode -D_GLIBCXX_USE_CXX11_ABI=1
      FAILED: /tmp/pip-install-fxgtjml5/ctcdecode_a306757a5af04bd1be52dabac4a498b6/build/temp.linux-x86_64-cpython-310/ctcdecode/src/binding.o
      c++ -MMD -MF /tmp/pip-install-fxgtjml5/ctcdecode_a306757a5af04bd1be52dabac4a498b6/build/temp.linux-x86_64-cpython-310/ctcdecode/src/binding.o.d -pthread -B /home/system/miniconda3/envs/SLG/compiler_compat -Wno-unused-result -Wsign-compare -DNDEBUG -fwrapv -O2 -Wall -fPIC -O2 -isystem /home/system/miniconda3/envs/SLG/include -fPIC -O2 -isystem /home/system/miniconda3/envs/SLG/include -fPIC -I/tmp/pip-install-fxgtjml5/ctcdecode_a306757a5af04bd1be52dabac4a498b6/third_party/kenlm -I/tmp/pip-install-fxgtjml5/ctcdecode_a306757a5af04bd1be52dabac4a498b6/third_party/openfst-1.6.7/src/include -I/tmp/pip-install-fxgtjml5/ctcdecode_a306757a5af04bd1be52dabac4a498b6/third_party/ThreadPool -I/tmp/pip-install-fxgtjml5/ctcdecode_a306757a5af04bd1be52dabac4a498b6/third_party/boost_1_67_0 -I/tmp/pip-install-fxgtjml5/ctcdecode_a306757a5af04bd1be52dabac4a498b6/third_party/utf8 -I/home/system/miniconda3/envs/SLG/lib/python3.10/site-packages/torch/include -I/home/system/miniconda3/envs/SLG/lib/python3.10/site-packages/torch/include/torch/csrc/api/include -I/home/system/miniconda3/envs/SLG/lib/python3.10/site-packages/torch/include/TH -I/home/system/miniconda3/envs/SLG/lib/python3.10/site-packages/torch/include/THC -I/home/system/miniconda3/envs/SLG/lib/python3.10/site-packages/torch/include -I/home/system/miniconda3/envs/SLG/lib/python3.10/site-packages/torch/include/torch/csrc/api/include -I/home/system/miniconda3/envs/SLG/lib/python3.10/site-packages/torch/include/TH -I/home/system/miniconda3/envs/SLG/lib/python3.10/site-packages/torch/include/THC -I/home/system/miniconda3/envs/SLG/include/python3.10 -c -c /tmp/pip-install-fxgtjml5/ctcdecode_a306757a5af04bd1be52dabac4a498b6/ctcdecode/src/binding.cpp -o /tmp/pip-install-fxgtjml5/ctcdecode_a306757a5af04bd1be52dabac4a498b6/build/temp.linux-x86_64-cpython-310/ctcdecode/src/binding.o -O3 -DKENLM_MAX_ORDER=6 -std=c++14 -fPIC -DINCLUDE_KENLM -DKENLM_MAX_ORDER=6 -DTORCH_API_INCLUDE_EXTENSION_H '-DPYBIND11_COMPILER_TYPE="_gcc"' '-DPYBIND11_STDLIB="_libstdcpp"' '-DPYBIND11_BUILD_ABI="_cxxabi1014"' -DTORCH_EXTENSION_NAME=ctc_decode -D_GLIBCXX_USE_CXX11_ABI=1
      /tmp/pip-install-fxgtjml5/ctcdecode_a306757a5af04bd1be52dabac4a498b6/ctcdecode/src/binding.cpp:7:10: fatal error: scorer.h: No such file or directory
          7 | #include "scorer.h"
            |          ^~~~~~~~~~
      compilation terminated.
      ninja: build stopped: subcommand failed.
      Traceback (most recent call last):
        File "/home/system/miniconda3/envs/SLG/lib/python3.10/site-packages/torch/utils/cpp_extension.py", line 1884, in _run_ninja_build
          subprocess.run(
        File "/home/system/miniconda3/envs/SLG/lib/python3.10/subprocess.py", line 526, in run
          raise CalledProcessError(retcode, process.args,
      subprocess.CalledProcessError: Command '['ninja', '-v']' returned non-zero exit status 1.
      
      The above exception was the direct cause of the following exception:
      
      Traceback (most recent call last):
        File "<string>", line 2, in <module>
        File "<pip-setuptools-caller>", line 34, in <module>
        File "/tmp/pip-install-fxgtjml5/ctcdecode_a306757a5af04bd1be52dabac4a498b6/setup.py", line 113, in <module>
          setup(
        File "/home/system/miniconda3/envs/SLG/lib/python3.10/site-packages/setuptools/__init__.py", line 104, in setup
          return distutils.core.setup(**attrs)
        File "/home/system/miniconda3/envs/SLG/lib/python3.10/site-packages/setuptools/_distutils/core.py", line 184, in setup
          return run_commands(dist)
        File "/home/system/miniconda3/envs/SLG/lib/python3.10/site-packages/setuptools/_distutils/core.py", line 200, in run_commands
          dist.run_commands()
        File "/home/system/miniconda3/envs/SLG/lib/python3.10/site-packages/setuptools/_distutils/dist.py", line 969, in run_commands
          self.run_command(cmd)
        File "/home/system/miniconda3/envs/SLG/lib/python3.10/site-packages/setuptools/dist.py", line 967, in run_command
          super().run_command(command)
        File "/home/system/miniconda3/envs/SLG/lib/python3.10/site-packages/setuptools/_distutils/dist.py", line 988, in run_command
          cmd_obj.run()
        File "/home/system/miniconda3/envs/SLG/lib/python3.10/site-packages/wheel/bdist_wheel.py", line 368, in run
          self.run_command("build")
        File "/home/system/miniconda3/envs/SLG/lib/python3.10/site-packages/setuptools/_distutils/cmd.py", line 316, in run_command
          self.distribution.run_command(command)
        File "/home/system/miniconda3/envs/SLG/lib/python3.10/site-packages/setuptools/dist.py", line 967, in run_command
          super().run_command(command)
        File "/home/system/miniconda3/envs/SLG/lib/python3.10/site-packages/setuptools/_distutils/dist.py", line 988, in run_command
          cmd_obj.run()
        File "/home/system/miniconda3/envs/SLG/lib/python3.10/site-packages/setuptools/_distutils/command/build.py", line 132, in run
          self.run_command(cmd_name)
        File "/home/system/miniconda3/envs/SLG/lib/python3.10/site-packages/setuptools/_distutils/cmd.py", line 316, in run_command
          self.distribution.run_command(command)
        File "/home/system/miniconda3/envs/SLG/lib/python3.10/site-packages/setuptools/dist.py", line 967, in run_command
          super().run_command(command)
        File "/home/system/miniconda3/envs/SLG/lib/python3.10/site-packages/setuptools/_distutils/dist.py", line 988, in run_command
          cmd_obj.run()
        File "/home/system/miniconda3/envs/SLG/lib/python3.10/site-packages/setuptools/command/build_ext.py", line 91, in run
          _build_ext.run(self)
        File "/home/system/miniconda3/envs/SLG/lib/python3.10/site-packages/setuptools/_distutils/command/build_ext.py", line 359, in run
          self.build_extensions()
        File "/home/system/miniconda3/envs/SLG/lib/python3.10/site-packages/torch/utils/cpp_extension.py", line 827, in build_extensions
          build_ext.build_extensions(self)
        File "/home/system/miniconda3/envs/SLG/lib/python3.10/site-packages/setuptools/_distutils/command/build_ext.py", line 479, in build_extensions
          self._build_extensions_serial()
        File "/home/system/miniconda3/envs/SLG/lib/python3.10/site-packages/setuptools/_distutils/command/build_ext.py", line 505, in _build_extensions_serial
          self.build_extension(ext)
        File "/home/system/miniconda3/envs/SLG/lib/python3.10/site-packages/setuptools/command/build_ext.py", line 252, in build_extension
          _build_ext.build_extension(self, ext)
        File "/home/system/.local/lib/python3.10/site-packages/Cython/Distutils/build_ext.py", line 135, in build_extension
          super(build_ext, self).build_extension(ext)
        File "/home/system/miniconda3/envs/SLG/lib/python3.10/site-packages/setuptools/_distutils/command/build_ext.py", line 560, in build_extension
          objects = self.compiler.compile(
        File "/home/system/miniconda3/envs/SLG/lib/python3.10/site-packages/torch/utils/cpp_extension.py", line 642, in unix_wrap_ninja_compile
          _write_ninja_file_and_compile_objects(
        File "/home/system/miniconda3/envs/SLG/lib/python3.10/site-packages/torch/utils/cpp_extension.py", line 1557, in _write_ninja_file_and_compile_objects
          _run_ninja_build(
        File "/home/system/miniconda3/envs/SLG/lib/python3.10/site-packages/torch/utils/cpp_extension.py", line 1900, in _run_ninja_build
          raise RuntimeError(message) from e
      RuntimeError: Error compiling objects for extension
      [end of output]
  
  note: This error originates from a subprocess, and is likely not a problem with pip.
  ERROR: Failed building wheel for ctcdecode
  Running setup.py clean for ctcdecode
Failed to build ctcdecode
ERROR: Could not build wheels for ctcdecode, which is required to install pyproject.toml-based projects
```

I tried install from source with this command, suggested by https://github.com/parlance/ctcdecode/issues/101

```
pip install git+https://github.com/parlance/ctcdecode.git
```

```
  warnings.warn(
Traceback (most recent call last):
  File "/home/system/workspace/SLRT/Spoken2Sign/text2gloss/prediction.py", line 271, in <module>
    model = build_model(cfg)
  File "/home/system/workspace/SLRT/Spoken2Sign/text2gloss/modelling/model.py", line 234, in build_model
    model = SignLanguageModel(cfg)
  File "/home/system/workspace/SLRT/Spoken2Sign/text2gloss/modelling/model.py", line 50, in __init__
    self.translation_network = TranslationNetwork(
  File "/home/system/workspace/SLRT/Spoken2Sign/text2gloss/modelling/translation.py", line 15, in __init__
    self.text_tokenizer = TextTokenizer(tokenizer_cfg=cfg['TextTokenizer'])
  File "/home/system/workspace/SLRT/Spoken2Sign/text2gloss/modelling/Tokenizer.py", line 53, in __init__
    self.tokenizer = MBartTokenizer.from_pretrained(
  File "/home/system/miniconda3/envs/SLG/lib/python3.10/site-packages/transformers/utils/import_utils.py", line 1543, in __getattribute__
    requires_backends(cls, cls._backends)
  File "/home/system/miniconda3/envs/SLG/lib/python3.10/site-packages/transformers/utils/import_utils.py", line 1531, in requires_backends
    raise ImportError("".join(failed))
ImportError: 
MBartTokenizer requires the SentencePiece library but it was not found in your environment. Checkout the instructions on the
installation page of its repo: https://github.com/google/sentencepiece#installation and follow the ones
that match your environment. Please note that you may need to restart your runtime after installation.
```

```
Traceback (most recent call last):
  File "/home/system/workspace/SLRT/Spoken2Sign/text2gloss/prediction.py", line 293, in <module>
    evaluation(model=model, val_dataloader=dataloader, cfg=cfg_, 
  File "/home/system/workspace/SLRT/Spoken2Sign/text2gloss/prediction.py", line 97, in evaluation
    forward_output = model(is_train=False, **batch)
  File "/home/system/miniconda3/envs/SLG/lib/python3.10/site-packages/torch/nn/modules/module.py", line 1194, in _call_impl
    return forward_call(*input, **kwargs)
  File "/home/system/workspace/SLRT/Spoken2Sign/text2gloss/modelling/model.py", line 143, in forward
    model_outputs = self.translation_network(**translation_inputs)
  File "/home/system/miniconda3/envs/SLG/lib/python3.10/site-packages/torch/nn/modules/module.py", line 1194, in _call_impl
    return forward_call(*input, **kwargs)
  File "/home/system/workspace/SLRT/Spoken2Sign/text2gloss/modelling/translation.py", line 204, in forward
    kwargs['inputs_embeds'] = self.text_embedding[input_ids].to(input_ids.device) * self.input_embed_scale
RuntimeError: indices should be either on cpu or on the same device as the indexed tensor (cpu)
```

# Dec 11

Temp fix (not sure whether it will be run on GPU):

translation.py: L204

```py
            kwargs['inputs_embeds'] = self.text_embedding.to(input_ids.device)[input_ids].to(input_ids.device) * self.input_embed_scale
```


```
[Validation] 322/322 [==============================] 444.5ms/step
2024-12-11 10:20:16,554 #samples: 322, average time cost per video: 0.02420728177017307s
2024-12-11 10:20:16,555 translation_loss Average:119.69
2024-12-11 10:20:16,555 total_loss Average:119.69
Traceback (most recent call last):
  File "/home/system/workspace/SLRT/Spoken2Sign/text2gloss/prediction.py", line 293, in <module>
    evaluation(model=model, val_dataloader=dataloader, cfg=cfg_, 
  File "/home/system/workspace/SLRT/Spoken2Sign/text2gloss/prediction.py", line 186, in evaluation
    wer_results = wer_list(hypotheses=gls_hyp, references=gls_ref)
  File "/home/system/workspace/SLRT/Spoken2Sign/text2gloss/utils/metrics.py", line 113, in wer_list
    res = wer_single(r=r, h=h)
  File "/home/system/workspace/SLRT/Spoken2Sign/text2gloss/utils/metrics.py", line 179, in wer_single
    edit_distance_matrix = edit_distance(r=r, h=h)
  File "/home/system/workspace/SLRT/Spoken2Sign/text2gloss/utils/metrics.py", line 217, in edit_distance
    d[0][j] = j * WER_COST_INS
OverflowError: Python integer 258 out of bounds for uint8
```

# Dec 18

```
Traceback (most recent call last):
  File "/home/system/workspace/SLRT/Spoken2Sign/motion_gen.py", line 196, in <module>
    body_pose = vposer.decode(
  File "/home/system/workspace/SLRT/Spoken2Sign/vposer/vposer_smpl.py", line 114, in decode
    if output_type == 'aa': return VPoser.matrot2aa(Xout)
  File "/home/system/workspace/SLRT/Spoken2Sign/vposer/vposer_smpl.py", line 153, in matrot2aa
    pose = tgm.rotation_matrix_to_angle_axis(homogen_matrot).view(batch_size, 1, -1, 3).contiguous()
  File "/home/system/miniconda3/envs/SLG/lib/python3.10/site-packages/torchgeometry/core/conversions.py", line 233, in rotation_matrix_to_angle_axis
    quaternion = rotation_matrix_to_quaternion(rotation_matrix)
  File "/home/system/miniconda3/envs/SLG/lib/python3.10/site-packages/torchgeometry/core/conversions.py", line 302, in rotation_matrix_to_quaternion
    mask_c1 = mask_d2 * (1 - mask_d0_d1)
  File "/home/system/miniconda3/envs/SLG/lib/python3.10/site-packages/torch/_tensor.py", line 39, in wrapped
    return f(*args, **kwargs)
  File "/home/system/miniconda3/envs/SLG/lib/python3.10/site-packages/torch/_tensor.py", line 834, in __rsub__
    return _C._VariableFunctions.rsub(self, other)
RuntimeError: Subtraction, the `-` operator, with a bool tensor is not supported. If you are trying to invert a mask, use the `~` or `logical_not()` operator instead.
```

https://stackoverflow.com/questions/65637222/runtimeerror-subtraction-the-operator-with-a-bool-tensor-is-not-supported
