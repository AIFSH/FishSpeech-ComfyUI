import os,sys
now_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(now_dir)

from huggingface_hub import snapshot_download

ckpt_dir = os.path.join(now_dir,"checkpoints","fish-speech-1.4")

import time
import torch
import torchaudio
from tools.vqgan.inference import load_model as load_vqgan_model
from tools.llama.generate import load_model,generate_long

class TextNode:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {"text": ("STRING", {"multiline": True, "dynamicPrompts": True})}}
    RETURN_TYPES = ("TEXT",)
    FUNCTION = "encode"

    CATEGORY = "AIFSH_FishSpeech"

    def encode(self,text):
        return (text, )

class FishSpeechNode:
    def __init__(self):
        if not os.path.exists(os.path.join(ckpt_dir,"model.pth")):
            snapshot_download(repo_id="fishaudio/fish-speech-1.4",
                            local_dir=ckpt_dir)
        else:
            print("use cached model weights!")
        self.vqgan_model = None
        self.llama_model = None

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required":{
                "text":("TEXT",),
                "num_samples":("INT",{
                    "default":2
                }),
                "max_new_tokens":("INT",{
                    "default":0
                }),
                "top_p":("FLOAT",{
                    "min":0,
                    "max":1,
                    "default":0.7
                }),
                "repetition_penalty":("FLOAT",{
                    "default":1.2
                }),
                "temperature":("FLOAT",{
                    "min":0,
                    "max":1,
                    "default":0.7
                }),
                "chunk_length":("INT",{
                    "default":100
                }),
                "seed":("INT",{
                    "default":42
                })
            },
            "optional":{
                "prompt_text":("TEXT",),
                "prompt_audio":("AUDIO",),
            }
        }
    
    RETURN_TYPES = ("AUDIO",)
    #RETURN_NAMES = ("image_output_name",)

    FUNCTION = "gen_audio"

    #OUTPUT_NODE = False

    CATEGORY = "AIFSH_FishSpeech"

    def gen_audio(self,text,num_samples,max_new_tokens,top_p,repetition_penalty,temperature,
                  chunk_length,seed,prompt_text=None,prompt_audio=None):
        device = "cuda"
        if self.vqgan_model is None:
            self.vqgan_model = load_vqgan_model(config_name="firefly_gan_vq",
                                    checkpoint_path=os.path.join(ckpt_dir,"firefly-gan-vq-fsq-8x1024-21hz-generator.pth"))
        prompt_sr = self.vqgan_model.spec_transform.sample_rate
        ## 1. 从语音生成 prompt
        if prompt_audio is not None:
            speech = prompt_audio["waveform"].squeeze(0)
            if speech.shape[0] > 1:
                speech = speech.mean(dim=0,keepdim=True)
            source_sr = prompt_audio["sample_rate"]
            print(speech.shape)
            print(source_sr)
            if source_sr != prompt_sr:
                speech = torchaudio.transforms.Resample(orig_freq=source_sr, new_freq=prompt_sr)(speech)
            
            audios = speech[None].to(device)
            print(audios.shape)
            print(f"Loaded audio with {audios.shape[2] / prompt_sr:.2f} seconds")
            
            # VQ Encoder
            audio_lengths = torch.tensor([audios.shape[2]], device=device, dtype=torch.long)
            indices = self.vqgan_model.encode(audios, audio_lengths)[0][0]

            print(f"Generated indices of shape {indices.shape}")
            prompt_tokens = [indices]
            prompt_text = [prompt_text]
        else:
            prompt_tokens = None
            prompt_text = None

        # 2.从文本生成语义 token
        precision = torch.half
        if self.llama_model is None:
            print("Loading model ...")
            t0 = time.time()
            self.llama_model, self.decode_one_token = load_model(
                ckpt_dir, device, precision, compile=False
            )
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            print(f"Time to load model: {time.time() - t0:.02f} seconds")
        # prompt_tokens = [indices]

        torch.manual_seed(seed)

        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
        
        generator = generate_long(
            model=self.llama_model,
            device=device,
            decode_one_token=self.decode_one_token,
            text=text,
            num_samples=num_samples,
            max_new_tokens=max_new_tokens,
            top_p=top_p,
            repetition_penalty=repetition_penalty,
            temperature=temperature,
            compile=True,
            iterative_prompt=True,
            chunk_length=chunk_length,
            prompt_text=prompt_text,
            prompt_tokens=prompt_tokens,
        )

        idx = 0
        codes = []

        for response in generator:
            if response.action == "sample":
                codes.append(response.codes)
                print(f"Sampled text: {response.text}")
            elif response.action == "next":
                if codes:
                    new_indices = torch.cat(codes, dim=1)
                    # np.save(f"codes_{idx}.npy", torch.cat(codes, dim=1).cpu().numpy())
                    # logger.info(f"Saved codes to codes_{idx}.npy")
                    break
                print(f"Next sample")
                codes = []
                idx += 1
            else:
                print(f"Error: {response}")
        
        # 3. 从语义 token 生成人声
        feature_lengths = torch.tensor([new_indices.shape[1]], device=device)
        fake_audios, _ = self.vqgan_model.decode(
            indices=new_indices[None], feature_lengths=feature_lengths
        )
        audio_time = fake_audios.shape[-1] / self.vqgan_model.spec_transform.sample_rate

        print(
            f"Generated audio of shape {fake_audios.shape}, equivalent to {audio_time:.2f} seconds from {new_indices.shape[1]} features, features/second: {new_indices.shape[1] / audio_time:.2f}"
        )
        self.vqgan_model = None
        self.llama_model = None
        # Save audio
        res = {
            "waveform": fake_audios,
            "sample_rate": prompt_sr
        }
        return (res, )
            

NODE_CLASS_MAPPINGS = {
    "TextNode": TextNode,
    "FishSpeechNode": FishSpeechNode
}