#!/usr/bin/env python
"""
Refactored voice conversion inference module.
"""

import os
import sys
import gc
import copy
import warnings
import threading
from time import time as ttime
from urllib.parse import urlparse

import torch
import torch.nn as nn
import numpy as np
import soundfile as sf
from scipy import signal
import librosa
from tqdm import tqdm
import faiss
import wget
import logging

from transformers import HubertModel

from infer_rvc_python.lib.log_config import logger
from infer_rvc_python.lib.infer_pack.models import (
    SynthesizerTrnMs256NSFsid,
    SynthesizerTrnMs256NSFsid_nono,
    SynthesizerTrnMs768NSFsid,
    SynthesizerTrnMs768NSFsid_nono,
)
from infer_rvc_python.lib.audio import load_audio
from infer_rvc_python.root_pipe import VC, change_rms, bh, ah

# Configure warnings and logging levels
warnings.filterwarnings("ignore")
logging.getLogger("fairseq").setLevel(logging.ERROR)
logging.getLogger("faiss.loader").setLevel(logging.ERROR)
logging.getLogger("transformers").setLevel(logging.ERROR)
logging.getLogger("torch").setLevel(logging.ERROR)

# Ensure current directory is in the path
now_dir: str = os.getcwd()
sys.path.append(now_dir)


class HubertModelWithFinalProj(HubertModel):
    """Extends HubertModel by adding a final projection layer."""

    def __init__(self, config):
        super().__init__(config)
        self.final_proj = nn.Linear(config.hidden_size, config.classifier_proj_size)


def load_embedding(embedder_model: str) -> HubertModelWithFinalProj:
    """
    Load a Hubert embedder model from local or remote resources.
    """
    custom_embedder = None  # Placeholder for a custom embedder path if provided
    embedder_root = now_dir
    embedding_list = {
        "contentvec": os.path.join(embedder_root, "contentvec"),
        "chinese-hubert-base": os.path.join(embedder_root, "chinese_hubert_base"),
        "japanese-hubert-base": os.path.join(embedder_root, "japanese_hubert_base"),
        "korean-hubert-base": os.path.join(embedder_root, "korean_hubert_base"),
    }
    online_embedders = {
        "contentvec": "https://huggingface.co/IAHispano/Applio/resolve/main/Resources/embedders/contentvec/pytorch_model.bin",
        "chinese-hubert-base": "https://huggingface.co/IAHispano/Applio/resolve/main/Resources/embedders/chinese_hubert_base/pytorch_model.bin",
        "japanese-hubert-base": "https://huggingface.co/IAHispano/Applio/resolve/main/Resources/embedders/japanese_hubert_base/pytorch_model.bin",
        "korean-hubert-base": "https://huggingface.co/IAHispano/Applio/resolve/main/Resources/embedders/korean_hubert_base/pytorch_model.bin",
    }
    config_files = {
        "contentvec": "https://huggingface.co/IAHispano/Applio/resolve/main/Resources/embedders/contentvec/config.json",
        "chinese-hubert-base": "https://huggingface.co/IAHispano/Applio/resolve/main/Resources/embedders/chinese_hubert_base/config.json",
        "japanese-hubert-base": "https://huggingface.co/IAHispano/Applio/resolve/main/Resources/embedders/japanese_hubert_base/config.json",
        "korean-hubert-base": "https://huggingface.co/IAHispano/Applio/resolve/main/Resources/embedders/korean_hubert_base/config.json",
    }

    if embedder_model == "custom":
        if custom_embedder and os.path.exists(custom_embedder):
            model_path = custom_embedder
        else:
            print(f"Custom embedder not found: {custom_embedder}, using contentvec")
            model_path = embedding_list["contentvec"]
    else:
        model_path = embedding_list[embedder_model]
        bin_file = os.path.join(model_path, "pytorch_model.bin")
        json_file = os.path.join(model_path, "config.json")
        os.makedirs(model_path, exist_ok=True)
        if not os.path.exists(bin_file):
            url = online_embedders[embedder_model]
            print(f"Downloading {url} to {model_path}...")
            wget.download(url, out=bin_file)
        if not os.path.exists(json_file):
            url = config_files[embedder_model]
            print(f"Downloading {url} to {model_path}...")
            wget.download(url, out=json_file)

    model = HubertModelWithFinalProj.from_pretrained(model_path)
    return model


class Config:
    """Holds inference configuration and device settings."""

    def __init__(self, only_cpu: bool = False) -> None:
        self.device: str = "cuda:0"
        self.is_half: bool = True
        self.n_cpu: int = 0
        self.gpu_name: str | None = None
        self.gpu_mem: int | None = None
        (
            self.x_pad,
            self.x_query,
            self.x_center,
            self.x_max,
        ) = self.device_config(only_cpu)

    def device_config(self, only_cpu: bool) -> tuple[int, int, int, int]:
        """
        Determine device configuration based on GPU availability and memory.
        """
        if torch.cuda.is_available() and not only_cpu:
            i_device = int(self.device.split(":")[-1])
            self.gpu_name = torch.cuda.get_device_name(i_device)
            # Adjust half precision setting based on GPU model
            if (("16" in self.gpu_name and "V100" not in self.gpu_name.upper())
                    or "P40" in self.gpu_name.upper()
                    or "1060" in self.gpu_name
                    or "1070" in self.gpu_name
                    or "1080" in self.gpu_name):
                logger.info("16/10 Series GPUs and P40 excel in single-precision tasks.")
                self.is_half = False
            else:
                self.gpu_name = None
            self.gpu_mem = int(
                torch.cuda.get_device_properties(i_device).total_memory / (1024**3) + 0.4
            )
        elif torch.backends.mps.is_available() and not only_cpu:
            logger.info("Supported N-card not found, using MPS for inference")
            self.device = "mps"
        else:
            logger.info("No supported N-card found, using CPU for inference")
            self.device = "cpu"
            self.is_half = False

        if self.n_cpu == 0:
            self.n_cpu = os.cpu_count() or 1

        # Set configuration values based on available precision and GPU memory
        if self.is_half:
            x_pad, x_query, x_center, x_max = 3, 10, 60, 65
        else:
            x_pad, x_query, x_center, x_max = 1, 6, 38, 41

        if self.gpu_mem is not None and self.gpu_mem <= 4:
            x_pad, x_query, x_center, x_max = 1, 5, 30, 32

        logger.info(f"Config: Device is {self.device}, half precision is {self.is_half}")
        return x_pad, x_query, x_center, x_max


# Global constants (if needed)
BASE_DOWNLOAD_LINK = "https://huggingface.co/r3gm/sonitranslate_voice_models/resolve/main/"
BASE_MODELS = ["rmvpe.pt"]
BASE_DIR = "."


def load_file_from_url(
    url: str,
    model_dir: str,
    file_name: str | None = None,
    overwrite: bool = False,
    progress: bool = True,
) -> str:
    """
    Download a file from `url` into `model_dir`, unless it already exists.
    Returns the absolute path to the downloaded file.
    """
    os.makedirs(model_dir, exist_ok=True)
    if not file_name:
        file_name = os.path.basename(urlparse(url).path)
    cached_file = os.path.abspath(os.path.join(model_dir, file_name))

    if os.path.exists(cached_file) and (overwrite or os.path.getsize(cached_file) == 0):
        os.remove(cached_file)

    if not os.path.exists(cached_file):
        logger.info(f'Downloading: "{url}" to {cached_file}')
        from torch.hub import download_url_to_file
        download_url_to_file(url, cached_file, progress=progress)
    else:
        logger.debug(cached_file)

    return cached_file


def friendly_name(file: str) -> tuple[str, str]:
    """
    Return a tuple (model_name, extension) for the given file.
    """
    if file.startswith("http"):
        file = urlparse(file).path
    file = os.path.basename(file)
    model_name, extension = os.path.splitext(file)
    return model_name, extension


def download_manager(
    url: str,
    path: str,
    extension: str = "",
    overwrite: bool = False,
    progress: bool = True,
) -> str:
    """
    Manage the downloading of a file by name and extension.
    """
    url = url.strip()
    name, ext = friendly_name(url)
    name += ext if not extension else f".{extension}"

    if url.startswith("http"):
        filename = load_file_from_url(
            url=url,
            model_dir=path,
            file_name=name,
            overwrite=overwrite,
            progress=progress,
        )
    else:
        filename = path
    return filename


def load_hu_bert(config: Config, hubert_path: str | None = None) -> torch.nn.Module:
    """
    Load and return the Hubert embedder model.
    """
    hubert_embedder = "contentvec"
    model = load_embedding(hubert_embedder)
    model = model.to(config.device).float()
    model.eval()
    return model


def load_trained_model(model_path: str, config: Config) -> tuple:
    """
    Load a trained VC model from file.
    Returns a tuple with speaker count, target sample rate, network,
    VC pipe, checkpoint data, and version.
    """
    if not model_path:
        raise ValueError("No model found")

    logger.info("Loading %s" % model_path)
    cpt = torch.load(model_path, map_location="cpu")
    tgt_sr = cpt["config"][-1]
    cpt["config"][-3] = cpt["weight"]["emb_g.weight"].shape[0]  # number of speakers
    if_f0 = cpt.get("f0", 1)
    version = cpt.get("version", "v1")
    if version == "v1":
        if if_f0 == 1:
            net_g = SynthesizerTrnMs256NSFsid(*cpt["config"], is_half=config.is_half)
        else:
            net_g = SynthesizerTrnMs256NSFsid_nono(*cpt["config"])
    elif version == "v2":
        if if_f0 == 1:
            net_g = SynthesizerTrnMs768NSFsid(*cpt["config"], is_half=config.is_half)
        else:
            net_g = SynthesizerTrnMs768NSFsid_nono(*cpt["config"])
    else:
        raise ValueError(f"Unknown version: {version}")

    # Remove encoder to save memory
    del net_g.enc_q

    net_g.load_state_dict(cpt["weight"], strict=False)
    net_g.eval().to(config.device)
    net_g = net_g.half() if config.is_half else net_g.float()

    vc_pipe = VC(tgt_sr, config)
    n_spk = cpt["config"][-3]

    return n_spk, tgt_sr, net_g, vc_pipe, cpt, version


class BaseLoader:
    """
    Loader class that manages configuration, inference,
    multi-threaded processing, and model caching.
    """

    def __init__(
        self, only_cpu: bool = False, hubert_path: str | None = None, rmvpe_path: str | None = None
    ) -> None:
        self.model_config: dict = {}
        self.config: Config | None = None
        self.cache_model: dict = {}
        self.only_cpu: bool = only_cpu
        self.hubert_path: str | None = hubert_path
        self.rmvpe_path: str | None = rmvpe_path
        self.output_list: list = []
        self.hu_bert_model: torch.nn.Module | None = None
        self.model_pitch_estimator = None
        self.model_vc: dict = {}

    def apply_conf(
        self,
        tag: str = "base_model",
        file_model: str = "",
        pitch_algo: str = "pm",
        pitch_lvl: int = 0,
        file_index: str = "",
        index_influence: float = 0.66,
        respiration_median_filtering: int = 3,
        envelope_ratio: float = 0.25,
        consonant_breath_protection: float = 0.33,
        resample_sr: int = 0,
        file_pitch_algo: str = "",
    ) -> str:
        """
        Apply a configuration for a given model tag.
        """
        if not file_model:
            raise ValueError("Model not found")

        if file_index is None:
            file_index = ""

        if file_pitch_algo is None:
            file_pitch_algo = ""

        if not self.config:
            self.config = Config(self.only_cpu)
            self.hu_bert_model = None

        self.model_config[tag] = {
            "file_model": file_model,
            "pitch_algo": pitch_algo,
            "pitch_lvl": pitch_lvl,
            "file_index": file_index,
            "index_influence": index_influence,
            "respiration_median_filtering": respiration_median_filtering,
            "envelope_ratio": envelope_ratio,
            "consonant_breath_protection": consonant_breath_protection,
            "resample_sr": resample_sr,
            "file_pitch_algo": file_pitch_algo,
        }
        return f"CONFIGURATION APPLIED FOR {tag}: {file_model}"

    def infer(
        self,
        task_id: str,
        params: dict,
        # Loaded model parameters
        n_spk,
        tgt_sr,
        net_g,
        pipe,
        cpt,
        version,
        if_f0,
        # Index parameters
        index_rate,
        index,
        big_npy,
        # f0 file data
        inp_f0,
        # Audio input (file path or tuple (<array>, sample_rate))
        input_audio_path,
        overwrite: bool,
        type_output: str | None,
    ) -> None:
        """
        Run inference on a given audio input.
        """
        f0_method = params["pitch_algo"]
        f0_up_key = int(params["pitch_lvl"])
        filter_radius = params["respiration_median_filtering"]
        resample_sr = params["resample_sr"]
        rms_mix_rate = params["envelope_ratio"]
        protect = params["consonant_breath_protection"]
        base_sr = 16000

        # Load and possibly resample audio
        if isinstance(input_audio_path, tuple):
            if f0_method == "harvest":
                raise ValueError("Harvest not supported from array")
            audio, source_sr = input_audio_path
            if source_sr != base_sr:
                audio = librosa.resample(audio.astype(np.float32), source_sr, base_sr)
            audio = audio.astype(np.float32).flatten()
        elif not os.path.exists(input_audio_path):
            raise ValueError(
                f"The audio file was not found or is not valid: {input_audio_path}"
            )
        else:
            audio = load_audio(input_audio_path, base_sr)

        # Normalize audio
        audio_max = np.abs(audio).max() / 0.95
        if audio_max > 1:
            audio /= audio_max

        times = [0, 0, 0]
        # Filter and pad audio
        audio = signal.filtfilt(bh, ah, audio)
        audio_pad = np.pad(audio, (pipe.window // 2, pipe.window // 2), mode="reflect")
        opt_ts = []
        if audio_pad.shape[0] > pipe.t_max:
            audio_sum = sum(audio_pad[i: i - pipe.window] for i in range(pipe.window))
            for t in range(pipe.t_center, audio.shape[0], pipe.t_center):
                window_slice = audio_sum[t - pipe.t_query : t + pipe.t_query]
                offset = np.argmin(np.abs(window_slice))
                opt_ts.append(t - pipe.t_query + offset)

        s = 0
        audio_opt = []
        t_last = None
        t1 = ttime()

        sid = torch.tensor(0, device=pipe.device).unsqueeze(0).long()

        # Pad audio for processing
        audio_pad = np.pad(audio, (pipe.t_pad, pipe.t_pad), mode="reflect")
        p_len = audio_pad.shape[0] // pipe.window

        # Estimate pitch if required
        pitch, pitchf = None, None
        if if_f0 == 1:
            pitch, pitchf = pipe.get_f0(
                input_audio_path,
                audio_pad,
                p_len,
                f0_up_key,
                f0_method,
                filter_radius,
                inp_f0,
            )
            pitch, pitchf = pitch[:p_len], pitchf[:p_len]
            if pipe.device == "mps":
                pitchf = pitchf.astype(np.float32)
            pitch = torch.tensor(pitch, device=pipe.device).unsqueeze(0).long()
            pitchf = torch.tensor(pitchf, device=pipe.device).unsqueeze(0).float()

        t2 = ttime()
        times[1] += t2 - t1

        # Process slices based on optimal time indices
        for t_val in opt_ts:
            t_val = (t_val // pipe.window) * pipe.window
            if if_f0 == 1:
                pitch_slice = pitch[:, s // pipe.window : (t_val + pipe.t_pad2) // pipe.window]
                pitchf_slice = pitchf[:, s // pipe.window : (t_val + pipe.t_pad2) // pipe.window]
            else:
                pitch_slice, pitchf_slice = None, None

            audio_slice = audio_pad[s : t_val + pipe.t_pad2 + pipe.window]
            result = pipe.vc(
                self.hu_bert_model,
                net_g,
                sid,
                audio_slice,
                pitch_slice,
                pitchf_slice,
                times,
                index,
                big_npy,
                index_rate,
                version,
                protect,
            )
            audio_opt.append(result[pipe.t_pad_tgt : -pipe.t_pad_tgt])
            s = t_val
            t_last = t_val

        # Process final slice
        if t_last is not None:
            pitch_end_slice = pitch[:, t_last // pipe.window :] if if_f0 == 1 else None
            pitchf_end_slice = pitchf[:, t_last // pipe.window :] if if_f0 == 1 else None
        else:
            pitch_end_slice, pitchf_end_slice = pitch, pitchf

        result = pipe.vc(
            self.hu_bert_model,
            net_g,
            sid,
            audio_pad[t_last:],
            pitch_end_slice,
            pitchf_end_slice,
            times,
            index,
            big_npy,
            index_rate,
            version,
            protect,
        )
        audio_opt.append(result[pipe.t_pad_tgt : -pipe.t_pad_tgt])

        audio_opt = np.concatenate(audio_opt)
        if rms_mix_rate != 1:
            audio_opt = change_rms(audio, 16000, audio_opt, tgt_sr, rms_mix_rate)
        if resample_sr >= 16000 and tgt_sr != resample_sr:
            audio_opt = librosa.resample(audio_opt, tgt_sr, resample_sr)

        # Normalize output audio to int16
        audio_max = np.abs(audio_opt).max() / 0.99
        max_int16 = 32768 if audio_max <= 1 else 32768 / audio_max
        audio_opt = (audio_opt * max_int16).astype(np.int16)

        del pitch, pitchf, sid
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        final_sr = resample_sr if (resample_sr >= 16000 and tgt_sr != resample_sr) else tgt_sr

        # Determine output file path if needed
        if type_output == "array":
            return audio_opt, final_sr

        if overwrite:
            output_audio_path = input_audio_path  # Overwrite input file
        else:
            basename = os.path.basename(input_audio_path)
            dirname = os.path.dirname(input_audio_path)
            name_part, ext = os.path.splitext(basename)
            new_basename = f"{name_part}_edited{ext}"
            output_audio_path = os.path.join(dirname, new_basename)

        if type_output:
            output_audio_path = os.path.splitext(output_audio_path)[0] + f".{type_output}"

        try:
            sf.write(file=output_audio_path, samplerate=final_sr, data=audio_opt)
        except Exception as e:
            logger.error(e)
            logger.error("Error saving file, trying with WAV format")
            output_audio_path = os.path.splitext(output_audio_path)[0] + ".wav"
            sf.write(file=output_audio_path, samplerate=final_sr, data=audio_opt)

        logger.info(output_audio_path)
        self.model_config[task_id].setdefault("result", []).append(output_audio_path)
        self.output_list.append(output_audio_path)

    def run_threads(self, threads: list[threading.Thread]) -> None:
        """Start and join a list of threads, then free memory."""
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def unload_models(self) -> None:
        """Unload and clear cached models and resources."""
        self.hu_bert_model = None
        self.model_pitch_estimator = None
        self.model_vc.clear()
        self.cache_model.clear()
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def __call__(
        self,
        audio_files: list[str | tuple] = [],
        tag_list: list[str] = [],
        overwrite: bool = False,
        parallel_workers: int = 1,
        type_output: str | None = None,  # e.g., "mp3", "wav", "ogg", "flac"
    ) -> list:
        """
        Run inference on a list of audio files with specified model tags.
        """
        logger.info(f"Parallel workers: {parallel_workers}")
        self.output_list = []

        if not self.model_config:
            raise ValueError("No model has been configured for inference")

        if isinstance(audio_files, str):
            audio_files = [audio_files]
        if isinstance(tag_list, str):
            tag_list = [tag_list]

        if not audio_files:
            raise ValueError("No audio found to convert")
        if not tag_list:
            tag_list = [list(self.model_config.keys())[-1]] * len(audio_files)

        # Ensure tag list length matches audio files
        if len(audio_files) > len(tag_list):
            tag_list.extend([tag_list[0]] * (len(audio_files) - len(tag_list)))
        elif len(audio_files) < len(tag_list):
            tag_list = tag_list[: len(audio_files)]

        tag_file_pairs = list(zip(tag_list, audio_files))
        sorted_tag_file = sorted(tag_file_pairs, key=lambda x: x[0])

        # Initialize base model if not loaded
        if not self.hu_bert_model:
            self.hu_bert_model = load_hu_bert(self.config, self.hubert_path)

        cache_params = None
        threads = []
        progress_bar = tqdm(total=len(tag_list), desc="Progress")
        for id_tag, input_audio_path in sorted_tag_file:
            if id_tag not in self.model_config:
                logger.info(f"No configured model for {id_tag} with {input_audio_path}")
                continue

            if len(threads) >= parallel_workers or (cache_params is not None and cache_params != id_tag):
                self.run_threads(threads)
                progress_bar.update(len(threads))
                threads = []

            if cache_params != id_tag:
                self.model_config[id_tag].setdefault("result", [])
                # Unload previous model if switching tags
                for _ in range(11):
                    gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

                params = self.model_config[id_tag]
                model_path = params["file_model"]
                file_index = params["file_index"]
                index_rate = params["index_influence"]
                f0_file = params["file_pitch_algo"]

                (n_spk, tgt_sr, net_g, pipe, cpt, version) = load_trained_model(model_path, self.config)
                if_f0 = cpt.get("f0", 1)

                # Load index if available
                if os.path.exists(file_index) and index_rate != 0:
                    try:
                        index = faiss.read_index(file_index)
                        big_npy = index.reconstruct_n(0, index.ntotal)
                    except Exception as error:
                        logger.error(f"Index: {error}")
                        index_rate, index, big_npy = 0, None, None
                else:
                    logger.warning("File index not found")
                    index_rate, index, big_npy = 0, None, None

                # Load f0 file if available
                inp_f0 = None
                if os.path.exists(f0_file):
                    try:
                        with open(f0_file, "r") as f:
                            lines = f.read().strip("\n").split("\n")
                        inp_f0 = np.array([[float(i) for i in line.split(",")] for line in lines], dtype="float32")
                    except Exception as error:
                        logger.error(f"f0 file: {error}")

                if "rmvpe" in params["pitch_algo"]:
                    if not self.model_pitch_estimator:
                        from infer_rvc_python.lib.rmvpe import RMVPE

                        logger.info("Loading vocal pitch estimator model")
                        rm_local_path = self.rmvpe_path if self.rmvpe_path and os.path.exists(self.rmvpe_path) else "rmvpe.pt"
                        self.model_pitch_estimator = RMVPE(rm_local_path, is_half=self.config.is_half, device=self.config.device)
                    pipe.model_rmvpe = self.model_pitch_estimator

                cache_params = id_tag

            thread = threading.Thread(
                target=self.infer,
                args=(
                    id_tag,
                    params,
                    n_spk,
                    tgt_sr,
                    net_g,
                    pipe,
                    cpt,
                    version,
                    if_f0,
                    index_rate,
                    index,
                    big_npy,
                    inp_f0,
                    input_audio_path,
                    overwrite,
                    type_output,
                ),
            )
            threads.append(thread)

        if threads:
            self.run_threads(threads)
            progress_bar.update(len(threads))
        progress_bar.close()

        final_result = []
        for tag in set(tag_list):
            if tag in self.model_config and "result" in self.model_config[tag]:
                final_result.extend(self.model_config[tag]["result"])
        return final_result

    def generate_from_cache(
        self, audio_data: str | tuple = None, tag: str = None, reload: bool = False
    ):
        """
        Generate output using cached models; `audio_data` may be a file path or a tuple (<array>, sample_rate).
        """
        if not self.model_config:
            raise ValueError("No model has been configured for inference")
        if not audio_data:
            raise ValueError("An audio file or tuple (<numpy data>, sample rate) is needed")
        if tag not in self.model_config:
            raise ValueError(f"No configured model for {tag}")

        now_data = self.model_config[tag]
        now_data["tag"] = tag

        if self.cache_model != now_data or reload:
            # Unload previous VC model and clear cache
            self.model_vc.clear()
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            model_path = now_data["file_model"]
            file_index = now_data["file_index"]
            index_rate = now_data["index_influence"]
            f0_file = now_data["file_pitch_algo"]

            (
                self.model_vc["n_spk"],
                self.model_vc["tgt_sr"],
                self.model_vc["net_g"],
                self.model_vc["pipe"],
                self.model_vc["cpt"],
                self.model_vc["version"],
            ) = load_trained_model(model_path, self.config)
            self.model_vc["if_f0"] = self.model_vc["cpt"].get("f0", 1)

            if os.path.exists(file_index) and index_rate != 0:
                try:
                    index = faiss.read_index(file_index)
                    big_npy = index.reconstruct_n(0, index.ntotal)
                except Exception as error:
                    logger.error(f"Index: {error}")
                    index_rate, index, big_npy = 0, None, None
            else:
                logger.warning("File index not found")
                index_rate, index, big_npy = 0, None, None

            self.model_vc["index_rate"] = index_rate
            self.model_vc["index"] = index
            self.model_vc["big_npy"] = big_npy

            inp_f0 = None
            if os.path.exists(f0_file):
                try:
                    with open(f0_file, "r") as f:
                        lines = f.read().strip("\n").split("\n")
                    inp_f0 = np.array([[float(i) for i in line.split(",")] for line in lines], dtype="float32")
                except Exception as error:
                    logger.error(f"f0 file: {error}")
            self.model_vc["inp_f0"] = inp_f0

            if "rmvpe" in now_data["pitch_algo"]:
                if not self.model_pitch_estimator:
                    from infer_rvc_python.lib.rmvpe import RMVPE

                    logger.info("Loading vocal pitch estimator model")
                    rm_local_path = self.rmvpe_path if self.rmvpe_path and os.path.exists(self.rmvpe_path) else "rmvpe.pt"
                    self.model_pitch_estimator = RMVPE(rm_local_path, is_half=self.config.is_half, device=self.config.device)
                self.model_vc["pipe"].model_rmvpe = self.model_pitch_estimator

            self.cache_model = copy.deepcopy(now_data)

        return self.infer(
            tag,
            now_data,
            self.model_vc["n_spk"],
            self.model_vc["tgt_sr"],
            self.model_vc["net_g"],
            self.model_vc["pipe"],
            self.model_vc["cpt"],
            self.model_vc["version"],
            self.model_vc["if_f0"],
            self.model_vc["index_rate"],
            self.model_vc["index"],
            self.model_vc["big_npy"],
            self.model_vc["inp_f0"],
            audio_data,
            False,
            "array",
        )
