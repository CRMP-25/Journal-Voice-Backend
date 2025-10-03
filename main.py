# optimized_stt_server/main.py - Clean version without noisereduce
import asyncio, json, uvicorn, base64, io, time, logging
from typing import List, Optional, Dict, Tuple
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from pydantic import BaseModel
import numpy as np
import soundfile as sf
from faster_whisper import WhisperModel
from scipy import signal
from fuzzywuzzy import fuzz, process
from concurrent.futures import ThreadPoolExecutor
from math import gcd
from starlette.websockets import WebSocketState


# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Optimized Journal STT")

# PRODUCTION SETTINGS - Choose based on your hardware
# Option 1: CPU (Windows safe)
# MODEL_SIZE = "large-v3"
# DEVICE = "cpu"
# COMPUTE_TYPE = "int8"
# NUM_THREADS = 4

# Option 2: GPU (if available) - MUCH faster and more accurate
MODEL_SIZE = "large-v3"
DEVICE = "cuda"
COMPUTE_TYPE = "float16"
NUM_THREADS = 4

# logger.info(f"Loading Whisper model: {MODEL_SIZE} on {DEVICE}")
# model = WhisperModel(
#     MODEL_SIZE, 
#     device=DEVICE, 
#     compute_type=COMPUTE_TYPE,
#     cpu_threads=NUM_THREADS if DEVICE == "cpu" else None
# )

logger.info(f"Loading Whisper model: {MODEL_SIZE} on {DEVICE}")
_model_kwargs = dict(device=DEVICE, compute_type=COMPUTE_TYPE)
if DEVICE == "cpu":
    _model_kwargs["cpu_threads"] = NUM_THREADS  # only on CPU
model = WhisperModel(MODEL_SIZE, **_model_kwargs)


# Thread pool for CPU-intensive audio processing
thread_pool = ThreadPoolExecutor(max_workers=2)

class OptimizedAudioProcessor:
    """Minimal audio preprocessing - preserves speech clarity"""
    
    @staticmethod
    def gentle_enhance_audio(audio: np.ndarray, sr: int) -> np.ndarray:
        """Gentle enhancement without distortion"""
        try:
            if np.max(np.abs(audio)) == 0:
                return audio
            
            # Normalize to 85% to prevent clipping
            audio = audio / np.max(np.abs(audio)) * 0.85
            
            # Gentle high-pass filter (remove rumble only)
            if sr > 160:
                nyquist = sr / 2
                high_freq = min(80, nyquist * 0.08)  # Very low cutoff
                if high_freq < nyquist * 0.8:
                    sos = signal.butter(1, high_freq, btype='high', fs=sr, output='sos')
                    audio = signal.sosfilt(sos, audio)
            
            # Simple noise gate (gentler than noisereduce)
            try:
                if len(audio) > sr * 0.3:
                    # Estimate noise from first 100ms
                    noise_sample = audio[:int(sr * 0.1)]
                    noise_level = np.std(noise_sample)
                    # Gentle gate threshold
                    threshold = noise_level * 1.5
                    # Soft gate (reduces, doesn't eliminate)
                    audio = np.where(np.abs(audio) < threshold, audio * 0.7, audio) 
            except Exception as e:
                logger.debug(f"Noise gate skipped: {e}")
            
            # Very gentle compression for consistency
            compressed = audio       
            
            # Final normalization
            if np.max(np.abs(compressed)) > 0:
                compressed = compressed / np.max(np.abs(compressed)) * 0.9
                
            return compressed.astype(np.float32)
            
        except Exception as e:
            logger.warning(f"Audio enhancement failed: {e}")
            # Return simple normalized audio
            return (audio / np.max(np.abs(audio)) * 0.85).astype(np.float32)
        
   


    @staticmethod
    def detect_speech_segments(audio: np.ndarray, sr: int) -> List[Tuple[int, int]]:
        """Detect speech segments via energy detection"""
        try:
            if len(audio) == 0:
                return []

            # Frame parameters
            frame_len = max(1, int(sr * 0.025))  # 25ms
            hop = max(1, int(sr * 0.010))  # 10ms

            # Calculate frame-wise RMS energy
            n_frames = 1 + max(0, (len(audio) - frame_len) // hop)
            rms = np.empty(n_frames, dtype=np.float32)
            for i in range(n_frames):
                s = i * hop
                e = s + frame_len
                frame = audio[s:e]
                rms[i] = np.sqrt(np.mean(frame**2) + 1e-12)

            # Adaptive threshold
            threshold = max(0.01, float(np.median(rms) * 1.2))
            voiced = rms > threshold

            # Build segments
            segments = []
            i = 0
            while i < n_frames:
                if voiced[i]:
                    j = i + 1
                    while j < n_frames and voiced[j]:
                        j += 1
                    s = i * hop
                    e = min(len(audio), (j * hop) + frame_len)
                    segments.append((s, e))
                    i = j
                else:
                    i += 1

            if not segments:
                return [(0, len(audio))]

            # Merge gaps < 250ms
            merged = []
            gap_limit = int(sr * 0.25)
            cur_s, cur_e = segments[0]
            for s, e in segments[1:]:
                if s - cur_e <= gap_limit:
                    cur_e = e
                else:
                    merged.append((cur_s, cur_e))
                    cur_s, cur_e = s, e
            merged.append((cur_s, cur_e))

            # Add padding ±300ms
            pad = int(sr * 0.30)
            padded = [(max(0, s - pad), min(len(audio), e + pad)) for s, e in merged]
            return padded

        except Exception as e:
            logger.warning(f"Speech detection failed: {e}")
            return [(0, len(audio))]


class SmartMatcher:
    """Enhanced field-specific matching"""
    
    # Phonetic corrections for common mishearings
    PHONETIC_MAP = {
        "mood": {
            "run": "none", "done": "none", "noon": "none", "known": "none",
            "and cheers": "anxious", "and she is": "anxious", "anchors": "anxious", "anches": "anxious"
        },
        # NEW: helps food allergy/intolerance lists
        "food_allergies": {
            # “none” heard as short/weird words
            "run": "none", "nun": "none", "non": "none", "known": "none", "noon": "none",
            "no allergies": "none", "nothing": "none", "nine": "none", "9": "none"
        },
        "food_intolerances": {
            "run": "none", "nun": "none", "non": "none", "known": "none", "noon": "none",
            "nothing": "none", "nine": "none", "9": "none",
            # “gluten” mishears
            "glue ten": "gluten", "blue ten": "gluten", "glue 10": "gluten", "blue 10": "gluten",
            "gluton": "gluten", "glutan": "gluten", "glutin": "gluten",
            # “lactose” mishears
            "lack toast": "lactose", "lak toes": "lactose", "lacto's": "lactose", "lactos": "lactose"
        },
        "diet_preferences": {
            "vegeterian": "vegetarian", "vegit": "vegetarian", "veggie": "vegetarian",
            "low sodiam": "low sodium", "low so dium": "low sodium", "low-sodium": "low sodium",
            "low salt": "low sodium", "less salt": "low sodium", "salt restricted": "low sodium"
        },

        # NEW: list-type fields also map "run/done/noon/known" → "none"
        "medical_conditions": {"run": "none", "done": "none", "noon": "none", "known": "none"},
       
        "medicationsList":    {"run": "none", "done": "none", "noon": "none", "known": "none"},
    }

    
    FIELD_VOCABULARIES = {
        "mood": {
            "positive": ["happy", "good", "great", "excellent", "cheerful", "joyful", "content", "fine", "okay"],
            "negative": ["sad", "bad", "depressed", "upset", "frustrated", "angry", "stressed", "anxious", "tired"],
            "neutral": ["neutral", "normal", "calm", "none"],
            "energy": ["energetic", "excited", "active", "lazy", "exhausted"]
        },
        "food": {
            "quality": {
                "low": ["low", "poor", "little", "minimal"],
                "medium": ["medium", "average", "okay", "moderate", "normal"],
                "high": ["high", "good", "excellent", "abundant"]
            },
            "taste": ["tasty", "delicious", "yummy"],
            "combinations": {
                "medium_tasty": ["medium and tasty", "medium tasty"],
                "high_tasty": ["high and tasty", "high tasty"]
            }
        },
        "water": {
            "amounts": {
                "0": ["zero", "none", "nothing"],
                "0.5": ["half", "0.5", "point five"],
                "1": ["one", "1", "single"],
                "1.5": ["one and half", "one point five", "1.5"],
                "2": ["two", "2", "couple"],
                "2.5": ["two and half", "two point five"],
                "3": ["three", "3"],
                "4": ["four", "4"],
                "5": ["five", "5"]
            }
        },
        "medications": {
            "yes": ["yes", "yeah", "yep", "sure", "took them", "taken"],
            "no": ["no", "nope", "didn't", "forgot", "missed"]
        },
        "activities": {
            "yes": ["yes", "yeah", "did some", "participated"],
            "no": ["no", "nope", "didn't", "nothing"]
        },

        "diet_preferences": {
        "vegetarian": ["vegetarian", "veg", "veg diet"],
        "non-vegetarian": ["non vegetarian", "non-vegetarian", "nonveg", "non veg"],
        "vegan": ["vegan"],
        "pescatarian": ["pescatarian", "fish only"],
        "keto": ["keto", "ketogenic"],
        "paleo": ["paleo"],
        "mediterranean": ["mediterranean"],
        "low sodium": ["low sodium", "low salt", "less salt", "salt restricted"],
        "low sugar": ["low sugar", "low sucrose", "diabetic diet"],
        "low carb": ["low carb"],
        "low fat": ["low fat"],
        "gluten free": ["gluten free"],
        "dairy free": ["dairy free", "no dairy"]
    },

    }
    
    @classmethod
    def apply_phonetic_corrections(cls, text: str, field: str) -> str:
        """Fix common mishearings before matching"""
        text_lower = text.lower().strip()
        
        if field in cls.PHONETIC_MAP:
            for wrong, right in cls.PHONETIC_MAP[field].items():
                if wrong in text_lower:
                    logger.info(f"Phonetic correction: '{wrong}' -> '{right}' in '{text}'")
                    text_lower = text_lower.replace(wrong, right)
        
        return text_lower
    
    @classmethod
    def enhanced_match(cls, text: str, field: str) -> Dict:
        """Enhanced matching with phonetic corrections"""
        # Apply phonetic corrections first
        corrected_text = cls.apply_phonetic_corrections(text, field)
        
        if field not in cls.FIELD_VOCABULARIES:
            return {"text": text, "confidence": 30, "matched": False}
        
        vocab = cls.FIELD_VOCABULARIES[field]
        
        # Strategy 1: Exact substring matching
        for category, terms in vocab.items():
            if isinstance(terms, dict):
                for subcategory, subterms in terms.items():
                    for term in subterms:
                        if term in corrected_text:
                            return {
                                "text": text,
                                "confidence": 95,
                                "matched": True,
                                "category": subcategory,
                                "strategy": "exact_match"
                            }
            elif isinstance(terms, list):
                for term in terms:
                    if term in corrected_text:
                        return {
                            "text": text,
                            "confidence": 95,
                            "matched": True,
                            "category": category,
                            "strategy": "exact_match"
                        }
        
        # Strategy 2: Word-level matching
        text_words = set(corrected_text.split())
        best_match = None
        best_score = 0
        
        for category, terms in vocab.items():
            if isinstance(terms, dict):
                for subcategory, subterms in terms.items():
                    for term in subterms:
                        term_words = set(term.split())
                        intersection = len(text_words.intersection(term_words))
                        if intersection > 0:
                            score = (intersection / len(term_words)) * 90
                            if score > best_score:
                                best_score = score
                                best_match = {
                                    "text": text,
                                    "confidence": score,
                                    "matched": True,
                                    "category": subcategory,
                                    "strategy": "word_match"
                                }
            elif isinstance(terms, list):
                for term in terms:
                    term_words = set(term.split())
                    intersection = len(text_words.intersection(term_words))
                    if intersection > 0:
                        score = (intersection / len(term_words)) * 90
                        if score > best_score:
                            best_score = score
                            best_match = {
                                "text": text,
                                "confidence": score,
                                "matched": True,
                                "category": category,
                                "strategy": "word_match"
                            }
        
        if best_match and best_score > 60:
            return best_match
        
        # Strategy 3: Fuzzy matching
        all_terms = []
        term_to_category = {}
        
        for category, terms in vocab.items():
            if isinstance(terms, dict):
                for subcategory, subterms in terms.items():
                    for term in subterms:
                        all_terms.append(term)
                        term_to_category[term] = subcategory
            elif isinstance(terms, list):
                for term in terms:
                    all_terms.append(term)
                    term_to_category[term] = category
        
        try:
            best_fuzzy = process.extractOne(corrected_text, all_terms, scorer=fuzz.partial_ratio)
            if best_fuzzy and best_fuzzy[1] > 65:  # Increased threshold
                matched_term = best_fuzzy[0]
                category = term_to_category[matched_term]
                return {
                    "text": text,
                    "confidence": best_fuzzy[1] * 0.8,
                    "matched": True,
                    "category": category,
                    "strategy": "fuzzy_match"
                }
        except Exception as e:
            logger.warning(f"Fuzzy matching failed: {e}")
        
        return {"text": text, "confidence": 25, "matched": False, "strategy": "no_match"}


def decode_wav_pcm16(audio_b64: str, target_sr: int = 16000) -> Tuple[np.ndarray, int]:
    """Fast WAV decoding"""
    try:
        raw = base64.b64decode(audio_b64)
        bio = io.BytesIO(raw)
        audio, sr = sf.read(bio, dtype="float32")

        if audio.ndim > 1:
            audio = np.mean(audio, axis=1)

        # Efficient resampling if needed
        if sr != target_sr and len(audio) > 0:
            g = gcd(int(sr), int(target_sr))
            up = target_sr // g
            down = sr // g
            audio = signal.resample_poly(audio, up, down).astype(np.float32)
            sr = target_sr

        return audio.astype(np.float32), sr

    except Exception as e:
        logger.error(f"Audio decoding failed: {e}")
        raise


def optimized_transcribe(
    samples: np.ndarray,
    sr: int,
    language: str,
    hints: List[str],
    field_context: Optional[str] = None,
    n_best: int = 3
) -> List[Dict]:
    """Optimized transcription with better accuracy"""
    
    processor = OptimizedAudioProcessor()
    
    try:
        if len(samples) == 0:
            return [{"text": "", "confidence": 0, "source": "empty_audio"}]
        
        # Gentle enhancement
        samples = processor.gentle_enhance_audio(samples, sr)
        
        # Detect speech
        speech_segments = processor.detect_speech_segments(samples, sr)
        
        if not speech_segments:
            return [{"text": "", "confidence": 0, "source": "no_speech"}]
        
        # Use up to 8 segments (captures short multi-word answers)
        usable = [(s, e) for (s, e) in speech_segments if (e - s) >= int(sr * 0.10)]
        usable = usable[:8]
        if not usable:
            return [{"text": "", "confidence": 0, "source": "too_short"}]

        segment_audio = np.concatenate([samples[s:e] for (s, e) in usable])
        
        logger.info(f"Processing {len(segment_audio)/sr:.2f}s of speech")
        
        # Build field-specific prompt
        initial_prompt = "Transcribe this short health journal answer accurately. "
        
        if field_context == "mood":
            initial_prompt += "Common moods: excited, happy, calm, neutral, sad, stressed, anxious, tired, energetic, peaceful, none."
        elif field_context == "food":
            initial_prompt += "Answer format: low, medium, high, medium and tasty, or high and tasty."
        elif field_context == "water":
            initial_prompt += "Water intake in liters: zero, half, one, one and half, two, three, four, or five liters."
        elif field_context == "medications":
            initial_prompt += "Answer: yes or no."
        elif field_context in ("medical_conditions", "food_allergies", "food_intolerances", "medicationsList"):
            initial_prompt += "Answer is a short list of items. If none, answer 'none'."
        elif field_context == "diet_preferences":
            initial_prompt += (
                "Answer is a short diet label, such as: "
                "vegetarian, non-vegetarian, vegan, pescatarian, keto, paleo, mediterranean, "
                "low sodium, low sugar, low carb, low fat, gluten free, dairy free. "
                "Return only the label."
            )


        
        if hints:
            hint_str = ", ".join(f"'{h}'" for h in hints[:10])
            initial_prompt += f" Vocabulary: {hint_str}."
        
        results = []
        
        # Primary transcription with optimized settings
        try:
            segments, info = model.transcribe(
                segment_audio,
                language=language,
                vad_filter=False,
                # Deterministic first pass:
                temperature=0.0,
                beam_size=5,                     # 5 is a good balance
                best_of=None,                    # only used for sampling; disable here
                patience=1.0,                     # faster, deterministic
                length_penalty=1.0,
                log_prob_threshold=-0.8,
                no_speech_threshold=0.6,         # slightly stricter; reduces false positives
                compression_ratio_threshold=2.4,
                condition_on_previous_text=True, # let whisper keep intra-utterance context
                word_timestamps=False,
                initial_prompt=initial_prompt,
                repetition_penalty=1.2,
                no_repeat_ngram_size=3
            )

            
            text_parts = []
            confidence_scores = []
            
            for segment in segments:
                if segment.text.strip():
                    text_parts.append(segment.text.strip())
                    log_prob = getattr(segment, 'avg_logprob', -1.0)
                    confidence = max(0, min(100, (log_prob + 2) * 50))
                    confidence_scores.append(confidence)
            
            if text_parts:
                full_text = " ".join(text_parts)
                avg_confidence = np.mean(confidence_scores) if confidence_scores else 60
                
                result = {
                    "text": full_text,
                    "confidence": float(avg_confidence),
                    "source": "whisper_primary"
                }
                
                # Apply smart matching
                if field_context:
                    smart_match = SmartMatcher.enhanced_match(full_text, field_context)
                    result.update(smart_match)
                    if smart_match.get("matched", False):
                        result["confidence"] = min(100, result["confidence"] + 15)
                
                results.append(result)
                logger.info(f"Primary: '{full_text}' (conf: {avg_confidence:.1f}, matched: {result.get('matched', False)})")
        
        except Exception as e:
            logger.warning(f"Primary transcription failed: {e}")
        
        # Alternative hypotheses with temperature sampling
        if len(results) < n_best:
            for temp in [0.2, 0.4]:
                if len(results) >= n_best:
                    break
                    
                try:
                    segments_temp, _ = model.transcribe(
                        segment_audio,
                        language=language,
                        temperature=temp,
                        beam_size=5,
                        vad_filter=False,
                        condition_on_previous_text=False,
                        initial_prompt=initial_prompt,
                        word_timestamps=False
                    )
                    
                    temp_parts = [s.text.strip() for s in segments_temp if s.text.strip()]
                    if temp_parts:
                        temp_text = " ".join(temp_parts)
                        
                        # Avoid duplicates
                        if not any(temp_text.lower() == r["text"].lower() for r in results):
                            temp_result = {
                                "text": temp_text,
                                "confidence": 55,
                                "source": f"whisper_temp_{temp}"
                            }
                            
                            if field_context:
                                smart_match = SmartMatcher.enhanced_match(temp_text, field_context)
                                temp_result.update(smart_match)
                                if smart_match.get("matched", False):
                                    temp_result["confidence"] = min(100, temp_result["confidence"] + 10)
                            
                            results.append(temp_result)
                
                except Exception as e:
                    logger.warning(f"Temp {temp} failed: {e}")
        
        # Sort by confidence + matching
        results.sort(key=lambda x: (x.get("confidence", 0) + (25 if x.get("matched", False) else 0)), reverse=True)
        
        return results[:n_best] if results else [{"text": "", "confidence": 0, "source": "all_failed"}]
    
    except Exception as e:
        logger.error(f"Transcription error: {e}")
        return [{"text": "", "confidence": 0, "source": "error", "error": str(e)}]


class STTInit(BaseModel):
    language: Optional[str] = "en"
    hints: Optional[List[str]] = []
    n_best: int = 3
    field_context: Optional[str] = None
    session_context: Optional[List[str]] = []   # NEW


async def ws_safe_send(ws: WebSocket, payload: dict):
    if ws.application_state == WebSocketState.CONNECTED:
        await ws.send_text(json.dumps(payload))
    

@app.websocket("/ws/stt")
async def stt_websocket(ws: WebSocket):
    """WebSocket STT endpoint"""
    sr_current = None  # <- track the sample rate of the stream

    await ws.accept()
    logger.info(f"Connection from {ws.client}")
    
    language = "en"
    hints: List[str] = []
    n_best = 3
    field_context = None
    audio_chunks = []
    
    try:
        while True:
            try:
                msg = await asyncio.wait_for(ws.receive_text(), timeout=60.0)
                obj = json.loads(msg)
            except asyncio.TimeoutError:
                # tell client and stop listening
                await ws_safe_send(ws, {"type": "error", "message": "timeout"})
                break
            except json.JSONDecodeError:
                await ws_safe_send(ws, {"type": "error", "message": "bad_json"})
                continue


            if obj.get("type") == "init":
                try:
                    cfg = STTInit(**obj.get("payload", {}))
                    language = (cfg.language or "en").split("-")[0]
                    hints = cfg.hints or []
                    n_best = max(1, min(3, cfg.n_best or 3))
                    field_context = cfg.field_context
                    
                    logger.info(f"Init: lang={language}, field={field_context}")
                    await ws_safe_send(ws, {
                        "type": "ack",
                        "model": MODEL_SIZE,
                        "field_context": field_context
                    })

                except Exception as e:
                    await ws_safe_send(ws, {"type": "error", "message": str(e)})

            elif obj.get("type") == "chunk":
                try:
                    payload = obj.get("data")
                    fmt = (obj.get("format") or "wav").lower()

                    if not payload:
                        await ws_safe_send(ws, {"type": "error", "message": "empty_chunk"})
                        continue

                    # 1) Decode the incoming base64 WAV (and resample to 16k inside)
                    chunk_audio, sr_local = decode_wav_pcm16(payload, target_sr=16000)

                    # 2) Initialize sr_current or resample to it for consistency
                    if sr_current is None:
                        sr_current = sr_local
                    if sr_local != sr_current and len(chunk_audio) > 0:
                        # resample to sr_current
                        from math import gcd as _gcd
                        g = _gcd(int(sr_local), int(sr_current))
                        up = sr_current // g
                        down = sr_local // g
                        chunk_audio = signal.resample_poly(chunk_audio, up, down).astype(np.float32)
                        sr_local = sr_current

                    # 3) Append to buffer
                    audio_chunks.append((chunk_audio, sr_local))

                    # 4) Acknowledge chunk
                    await ws_safe_send(ws, {"type": "chunk_ack", "samples": len(chunk_audio)})

                    # 5) (Optional) Emit a quick partial after ~600ms of audio buffered
                    total_len = sum(len(c[0]) for c in audio_chunks)
                    if sr_current and total_len > int(sr_current * 0.6):
                        # take only the last ~6s to keep latency low
                        combined_tail = np.concatenate([c[0] for c in audio_chunks])[-int(sr_current * 6):]
                        partial = optimized_transcribe(
                            combined_tail, sr_current, language, hints, field_context, n_best=1
                        )[0]
                        await ws_safe_send(ws, {"type": "partial", "text": partial.get("text", "")})

                except Exception as e:
                    logger.error(f"Chunk handling error: {e}")
                    await ws_safe_send(ws, {"type": "error", "message": str(e)})




            elif obj.get("type") == "end":
                if not audio_chunks:
                    await ws_safe_send(ws, {
                        "type": "final",
                        "alternatives": [],
                        "metadata": {"message": "No audio"}
                    })

                    break

                try:
                    sr_final = sr_current or (audio_chunks[0][1] if audio_chunks else 16000)
                    combined = np.concatenate([chunk[0] for chunk in audio_chunks])

                    loop = asyncio.get_event_loop()
                    alternatives = await loop.run_in_executor(
                        thread_pool,
                        optimized_transcribe,
                        combined, sr_final, language, hints, field_context, n_best
                    )

                    
                    await ws_safe_send(ws, {
                        "type": "final",
                        "alternatives": [alt.get("text", "") for alt in alternatives],
                        "metadata": {
                            "confidences": [alt.get("confidence", 0) for alt in alternatives],
                            "matched": [alt.get("matched", False) for alt in alternatives],
                            "sources": [alt.get("source", "") for alt in alternatives],
                            "field_context": field_context
                        }
                    })

                    
                    logger.info("Results sent")
                
                except Exception as e:
                    logger.error(f"Processing error: {e}")
                    await ws_safe_send(ws, {"type": "error", "message": str(e)})
                
                break
                
    except WebSocketDisconnect:
        logger.info("Disconnected")
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
    finally:
        try:
            if ws.application_state != WebSocketState.DISCONNECTED:
                await ws.close(code=1000)
        except RuntimeError:
            # If the client/server already initiated close, ignore the second close.
            pass


@app.get("/health")
async def health():
    return {"status": "healthy", "model": MODEL_SIZE, "device": DEVICE}

@app.get("/")
async def root():
    return {
        "service": "Journal STT Server",
        "model": MODEL_SIZE,
        "device": DEVICE
    }

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8081, log_level="info")