import time
from datetime import datetime
from contextlib import nullcontext

from tqdm import tqdm
from itertools import chain
from copy import deepcopy

import numpy as np
import s3tokenizer
import torch

from transformers import AutoTokenizer, AutoModelForCausalLM, DynamicCache
from soulxpodcast.config import Config, SamplingParams, AutoPretrainedConfig
from soulxpodcast.engine.llm_engine import (
    HFLLMEngine, VLLMEngine
)
from soulxpodcast.models.modules.flow import CausalMaskedDiffWithXvec
from soulxpodcast.models.modules.hifigan import HiFTGenerator

class SoulXPodcast(torch.nn.Module):
    def __init__(self, config: Config = None):
        super().__init__()
        self.config = Config() if config is None else config

        # auto-select device: prefer CUDA, then MPS (Apple), else CPU
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
        elif getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
            self.device = torch.device("mps")
        else:
            self.device = torch.device("cpu")

        # audio tokenizer: try moving to device if supported; keep backwards-compatibility
        self.audio_tokenizer = s3tokenizer.load_model("speech_tokenizer_v2_25hz")
        try:
            # some tokenizers support .to(device)
            self.audio_tokenizer = self.audio_tokenizer.to(self.device)
        except Exception:
            # ignore if not supported
            pass
        self.audio_tokenizer = self.audio_tokenizer.eval()

        if self.config.llm_engine == "hf":
            self.llm = HFLLMEngine(**self.config.__dict__)
        elif self.config.llm_engine == "vllm":
            self.llm = VLLMEngine(**self.config.__dict__)
        else:
            raise NotImplementedError

        # whether to show progress bars; can be toggled externally
        self.use_tqdm = True

        self.flow = CausalMaskedDiffWithXvec()
        if self.config.hf_config.fp16_flow:
            timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S,%f')[:-3]
            tqdm.write(f"[{timestamp}] - [INFO] - Casting flow to fp16")
            # only change dtype here; device move happens after loading weights
            self.flow = self.flow.half()
        # load to CPU first to be robust, then move to chosen device
        self.flow.load_state_dict(torch.load(f"{self.config.model}/flow.pt", map_location="cpu", weights_only=True), strict=True)
        self.flow = self.flow.to(self.device).eval()

        self.hift = HiFTGenerator()
        hift_state_dict = {k.replace('generator.', ''): v for k, v in torch.load(f"{self.config.model}/hift.pt", map_location="cpu", weights_only=True).items()}
        self.hift.load_state_dict(hift_state_dict, strict=True)
        self.hift = self.hift.to(self.device).eval()

    
    @torch.inference_mode()
    def forward_longform(
        self, prompt_mels_for_llm,
        prompt_mels_lens_for_llm: torch.Tensor,
        prompt_text_tokens_for_llm: list[list[int]],
        text_tokens_for_llm: list[list[int]],
        prompt_mels_for_flow_ori, 
        spk_emb_for_flow: torch.Tensor,
        sampling_params: SamplingParams | list[SamplingParams],
        spk_ids: list[list[int]],
        use_dialect_prompt: bool = False,
        dialect_prompt_text_tokens_for_llm: list[list[int]] = None,
        dialect_prefix: list[list[int]] = None,
        **kwargs,  # for compatibility
    ):

        prompt_size, turn_size = len(prompt_mels_for_llm), len(text_tokens_for_llm)

        # Audio tokenization
        # ensure tensors are on the right device before passing to tokenizer.quantize
        device = self.device
        try:
            prompt_speech_tokens_ori, prompt_speech_tokens_lens_ori = self.audio_tokenizer.quantize(
                prompt_mels_for_llm.to(device), prompt_mels_lens_for_llm.to(device)
            )
        except Exception:
            # fallback: some tokenizers expect CPU tensors or handle device internally
            prompt_speech_tokens_ori, prompt_speech_tokens_lens_ori = self.audio_tokenizer.quantize(
                prompt_mels_for_llm, prompt_mels_lens_for_llm
            )

        # align speech token with speech feat as to reduce
        #    the noise ratio during the generation process.
        prompt_speech_tokens = []
        prompt_mels_for_flow, prompt_mels_lens_for_flow = [], []

        for prompt_index in range(prompt_size):
            prompt_speech_token_len = prompt_speech_tokens_lens_ori[prompt_index].item()
            prompt_speech_token = prompt_speech_tokens_ori[prompt_index, :prompt_speech_token_len]
            prompt_mel = prompt_mels_for_flow_ori[prompt_index]
            prompt_mel_len = prompt_mel.shape[0]
            if prompt_speech_token_len * 2 > prompt_mel_len:
                prompt_speech_token = prompt_speech_token[:int(prompt_mel_len/2)]
                prompt_mel_len = torch.tensor([prompt_mel_len], device=device)
            else:
                prompt_mel = prompt_mel.detach().clone()[:prompt_speech_token_len * 2].to(device)
                prompt_mel_len = torch.tensor([prompt_speech_token_len * 2], device=device)
            prompt_speech_tokens.append(prompt_speech_token)
            prompt_mels_for_flow.append(prompt_mel)
            prompt_mels_lens_for_flow.append(prompt_mel_len)

        # Prepare LLM inputs
        prompt_inputs = []
        history_inputs = []
        
        for i in range(prompt_size):
            speech_tokens_i = [token+self.config.hf_config.speech_token_offset for token in prompt_speech_tokens[i].tolist()]
            speech_tokens_i += [self.config.hf_config.eos_token_id]
            if use_dialect_prompt and len(dialect_prompt_text_tokens_for_llm[i])>0:
                dialect_prompt_input = prompt_text_tokens_for_llm[i] + speech_tokens_i + dialect_prompt_text_tokens_for_llm[i]
                if i>0:
                    dialect_prompt_input = dialect_prefix[0] + dialect_prompt_input
                prompt_input = self.llm.generate(dialect_prompt_input, sampling_params, past_key_values=None)['token_ids']
                prompt_inputs.append(dialect_prefix[i+1]+dialect_prompt_text_tokens_for_llm[i] + prompt_input)
                history_inputs.append(dialect_prefix[i+1]+dialect_prompt_text_tokens_for_llm[i] + prompt_input)
            else:
                prompt_inputs.append(prompt_text_tokens_for_llm[i] + speech_tokens_i )
                history_inputs.append(prompt_text_tokens_for_llm[i] + speech_tokens_i )

        generated_wavs, results_dict = [], {}
        
        # LLM generation
        inputs = list(chain.from_iterable(prompt_inputs))
        cache_config = AutoPretrainedConfig().from_dataclass(self.llm.config.hf_config)
        past_key_values = DynamicCache(config=cache_config)
        valid_turn_size = prompt_size

        # choose amp/autocast context: enable only for CUDA (safe and minimal change)
        if device.type == "cuda":
            amp_ctx = torch.cuda.amp.autocast
            amp_kwargs = {"dtype": torch.float16 if self.config.hf_config.fp16_flow else torch.float32}
        else:
            amp_ctx = nullcontext
            amp_kwargs = {}

        # build iterator with optional progress bar to preserve original loop semantics
        if self.use_tqdm:
            loop_iter = tqdm(range(turn_size), desc="Generating turns", unit="turn", leave=True)
        else:
            loop_iter = range(turn_size)

        for i in loop_iter:

            # # set ratio: reach the reset cache ratio;
            if valid_turn_size > self.config.max_turn_size or len(inputs)>self.config.turn_tokens_threshold:
                assert self.config.max_turn_size >= self.config.prompt_context + self.config.history_context, "Invalid Long history size setting, "
                prompt_text_bound = max(self.config.prompt_context, len(history_inputs)-self.config.history_text_context-self.config.history_context)
                inputs = list(chain.from_iterable(
                    history_inputs[:self.config.prompt_context]+ \
                    history_inputs[prompt_text_bound:-self.config.history_context]+ \
                    prompt_inputs[-self.config.history_context:]
                ))
                valid_turn_size = self.config.prompt_context + len(history_inputs) - prompt_text_bound
                past_key_values = DynamicCache(config=cache_config)
            valid_turn_size += 1
            
            inputs.extend(text_tokens_for_llm[i])
            start_time = time.time()
            llm_outputs = self.llm.generate(inputs, sampling_params, past_key_values=past_key_values)

            inputs.extend(llm_outputs['token_ids'])
            prompt_inputs.append(text_tokens_for_llm[i]+llm_outputs['token_ids'])
            history_inputs.append(text_tokens_for_llm[i][:-1]) # remove the <|audio_start|>
            
            # Prepare Flow inputs
            turn_spk = spk_ids[i]
            generated_speech_tokens = [token - self.config.hf_config.speech_token_offset for token in  llm_outputs['token_ids'][:-1]]  # ignore last eos
            prompt_speech_token = prompt_speech_tokens[turn_spk].tolist()
            flow_input = torch.tensor([prompt_speech_token + generated_speech_tokens], device=device)
            flow_inputs_len = torch.tensor([len(prompt_speech_token) + len(generated_speech_tokens)], device=device)

            # Flow generation and HiFi-GAN generation            
            start_idx = spk_ids[i]
            prompt_mels = prompt_mels_for_flow[start_idx][None].to(device)
            prompt_mels_lens = prompt_mels_lens_for_flow[start_idx][None].to(device)
            spk_emb = spk_emb_for_flow[start_idx:start_idx+1].to(device)

            # Flow generation
            with amp_ctx(**amp_kwargs):
                generated_mels, generated_mels_lens = self.flow(
                    flow_input.to(device), flow_inputs_len.to(device),
                    prompt_mels, prompt_mels_lens, spk_emb,
                    streaming=False, finalize=True
                )

            # HiFi-GAN generation
            mel = generated_mels[:, :, prompt_mels_lens[0].item():generated_mels_lens[0].item()]
            wav, _ = self.hift(speech_feat=mel)
            generated_wavs.append(wav.detach().cpu())

            # optionally update description with intermediate info (keeps backward-compatible)
            if self.use_tqdm and isinstance(loop_iter, tqdm):
                # show simple timing info in postfix
                elapsed = time.time() - start_time
                loop_iter.set_postfix({'last_turn_s': f"{elapsed:.2f}"})

        # Save the generated wav;
        results_dict['generated_wavs'] = generated_wavs
        return results_dict
