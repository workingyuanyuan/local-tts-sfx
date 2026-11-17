"""Adapter for OpenMOSS-Team/MOSS-TTS-Local-Transformer-v1.5 (follows the official example)."""
import gc
import logging
from pathlib import Path
from typing import Any, Optional

import torch

from local_tts.adapters.base import Adapter, SegmentAudio
from local_tts.config import VoiceEntry

log = logging.getLogger(__name__)

# Generated speech reused as a reference is capped to this many frames (~15 s).
MAX_SEGMENT_REFERENCE_FRAMES = 188

LANGUAGES = {
    "zh": "Chinese", "yue": "Cantonese", "en": "English", "ar": "Arabic", "cs": "Czech", "da": "Danish",
    "nl": "Dutch", "fi": "Finnish", "fr": "French", "de": "German", "el": "Greek", "he": "Hebrew",
    "hi": "Hindi", "hu": "Hungarian", "it": "Italian", "ja": "Japanese", "ko": "Korean", "mk": "Macedonian",
    "ms": "Malay", "fa": "Persian", "pl": "Polish", "pt": "Portuguese", "ro": "Romanian", "ru": "Russian",
    "es": "Spanish", "sw": "Swahili", "sv": "Swedish", "tl": "Tagalog", "th": "Thai", "tr": "Turkish",
    "vi": "Vietnamese",
}


def item_codes(generation_ids: torch.Tensor, start_length: int, audio_pad: int) -> torch.Tensor:
    """The audio codes one batch row generated: rows after the audio-start row and the prompt, audio
    channels only, without the all-padding rows of a row that finished before the rest of the batch."""
    codes = generation_ids[1 + int(start_length):, 1:]
    return codes[~codes.eq(audio_pad).all(dim=1)]


class MossLocalAdapter(Adapter):
    languages = LANGUAGES
    capabilities = {"clone": True, "pause_marker": True, "pinyin": True, "continuation": False}
    frames_per_second = 12.5
    supported_options = {"language", "seed"}
    supports_batch = True

    def __init__(self, entry):
        super().__init__(entry)
        self.model = None
        self._model = None  # set only while loading
        self.processor = None
        self._ref_cache: dict[tuple[str, float], torch.Tensor] = {}

    @property
    def loaded(self) -> bool:
        return self.model is not None

    def load(self) -> None:
        from huggingface_hub import snapshot_download
        from transformers import AutoModel, AutoProcessor

        if not torch.cuda.is_available():
            raise RuntimeError("MOSS-TTS needs an NVIDIA GPU with CUDA, but torch sees none "
                               "(check `nvidia-smi` and the driver; CUDA 12.8 needs driver 570 or newer).")

        # Official SDPA setup: cuDNN SDPA off, the other backends on.
        torch.backends.cuda.enable_cudnn_sdp(False)
        torch.backends.cuda.enable_flash_sdp(True)
        torch.backends.cuda.enable_mem_efficient_sdp(True)
        torch.backends.cuda.enable_math_sdp(True)

        o = self.entry.options
        # The remote processor rejects `revision` (and forwards it to the codec), so pin both
        # repos by loading from their resolved snapshot paths.
        model_path = snapshot_download(o["repo"], revision=o["revision"])
        codec_path = snapshot_download(o["codec_repo"], revision=o["codec_revision"])
        # Store each object on self as soon as it exists so unload() can free a partial load.
        self.processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True, codec_path=codec_path,
                                                       codec_weight_dtype=o["codec_weight_dtype"])
        self.processor.audio_tokenizer = self.processor.audio_tokenizer.to("cuda")
        self._model = AutoModel.from_pretrained(model_path, trust_remote_code=True, attn_implementation="sdpa",
                                                dtype=torch.bfloat16)
        self._model.to("cuda").eval()
        log.info("codec dtypes: %s", self.processor.audio_tokenizer.get_codec_dtype_summary())
        self.model, self._model = self._model, None

    def unload(self) -> None:
        self.model = None
        self._model = None
        self.processor = None
        self._ref_cache.clear()
        gc.collect()
        torch.cuda.empty_cache()

    def release_cache(self) -> None:
        # The caching allocator keeps a long job's activations reserved (seen: ~2 GiB above the
        # loaded weights); hand them back so other GPU work on this machine can use them.
        if self.model is not None:
            torch.cuda.empty_cache()

    def voice_reference(self, voice: VoiceEntry) -> torch.Tensor:
        path = Path(voice.reference_audio)
        key = (str(path), path.stat().st_mtime)
        if key not in self._ref_cache:
            with torch.no_grad():
                self._ref_cache[key] = self.processor.encode_audios_from_path([str(path)])[0]
        return self._ref_cache[key]

    def generate(self, text: str, *, language: Optional[str], reference: Any,
                 max_frames: int, seed: Optional[int]) -> SegmentAudio:
        p = self.processor
        msg = p.build_user_message(text=text, reference=None if reference is None else [reference],
                                   language=LANGUAGES[language] if language else None)
        if seed is not None:
            torch.manual_seed(seed)
        with torch.no_grad():
            batch = p([msg], mode="generation")
            outputs = self.model.generate(
                input_ids=batch["input_ids"].to("cuda"),
                attention_mask=batch["attention_mask"].to("cuda"),
                max_new_tokens=max_frames,
                do_sample=True,
                **self.entry.options["sampling"],
            )
            start_length, generation_ids = outputs[0]
            # Rows after the audio-start row are the generated frames; the loop only produces
            # `max_frames` of them when it never sampled the end token.
            frames = int(generation_ids.shape[0]) - 1 - int(start_length)
            message = p.decode(outputs)[0]
        sr = int(p.model_config.sampling_rate)
        wavs = [] if message is None else [w for w in message.audio_codes_list if w.numel()]
        audio = torch.cat(wavs, dim=-1).float().cpu().numpy() if wavs else torch.zeros(2, 0).numpy()

        codes = generation_ids[:, 1:]
        codes = codes[~codes.eq(int(p.model_config.audio_pad_token_id)).all(dim=1)]
        return SegmentAudio(audio=audio, sample_rate=sr, frames=frames, truncated=frames >= max_frames,
                            reference=codes[:MAX_SEGMENT_REFERENCE_FRAMES].contiguous() if len(codes) else None)

    def generate_batch(self, texts: list[str], *, language: Optional[str], reference: Any,
                       max_frames: list[int], seed: Optional[int]) -> list[SegmentAudio]:
        # Measured in docs/batch-bench-report.md. The processor left-pads with an attention mask and
        # the decoder derives positions from it; each row stops on its own end token, but the call
        # runs until every row has ended or the largest budget is used up.
        p = self.processor
        lang = LANGUAGES[language] if language else None
        conversations = [[p.build_user_message(text=t, reference=None if reference is None else [reference],
                                               language=lang)] for t in texts]
        if seed is not None:
            torch.manual_seed(seed)
        pad = int(p.model_config.audio_pad_token_id)
        with torch.no_grad():
            batch = p(conversations, mode="generation")
            outputs = self.model.generate(
                input_ids=batch["input_ids"].to("cuda"),
                attention_mask=batch["attention_mask"].to("cuda"),
                max_new_tokens=max(max_frames),
                do_sample=True,
                **self.entry.options["sampling"],
            )
            messages = p.decode(outputs)
        sr = int(p.model_config.sampling_rate)
        results = []
        for (start_length, generation_ids), message, budget in zip(outputs, messages, max_frames):
            codes = item_codes(generation_ids, start_length, pad)
            frames = int(codes.shape[0])
            wavs = [] if message is None else [w for w in message.audio_codes_list if w.numel()]
            audio = torch.cat(wavs, dim=-1).float().cpu().numpy() if wavs else torch.zeros(2, 0).numpy()
            # A row that ran past its own budget (possible while a longer row keeps the batch going)
            # counts as truncated, as it would have alone.
            results.append(SegmentAudio(audio=audio, sample_rate=sr, frames=frames, truncated=frames >= budget,
                                        reference=codes[:MAX_SEGMENT_REFERENCE_FRAMES].contiguous()
                                        if frames else None))
        if len(results) != len(texts):
            raise RuntimeError(f"batch of {len(texts)} texts returned {len(results)} results")
        return results

