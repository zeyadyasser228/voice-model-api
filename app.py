# ============================================================
#  Nesyan — AI Voice Recognition & Speaker Identification
#  ECAPA-TDNN + FAISS Pipeline — Railway Deployment
# ============================================================

import os
import uuid
import tempfile

import torch
import torchaudio
import numpy as np
import faiss

from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware

# ── Load ECAPA-TDNN Pretrained Model ────────────────────────────────────────
try:
    from speechbrain.inference.classifiers import EncoderClassifier
except ImportError:
    from speechbrain.pretrained import EncoderClassifier

device = 'cuda' if torch.cuda.is_available() else 'cpu'

classifier = EncoderClassifier.from_hparams(
    source='speechbrain/spkrec-ecapa-voxceleb',
    savedir='/tmp/ecapa_model',
    run_opts={'device': device}
)

# ── Constants ────────────────────────────────────────────────────────────────
EMBEDDING_DIM       = 192    # ECAPA-TDNN output size (Table 5.6)
SIMILARITY_THRESHOLD = 0.75  # Cosine similarity threshold (Table 5.9)
MIN_AUDIO_SECONDS   = 2.5    # Minimum speech duration (Section 5.13)
SAMPLE_RATE         = 16000  # Required sample rate for ECAPA-TDNN

# ── In-memory Speaker Registry + FAISS Index ────────────────────────────────
# NOTE: Data resets on server restart.
# For production, replace with PostgreSQL + persistent FAISS index.
speaker_registry = []
faiss_index = faiss.IndexFlatIP(EMBEDDING_DIM)


# ── Helper Functions ─────────────────────────────────────────────────────────

def load_and_validate_audio(audio_path: str):
    """
    Load audio and enforce minimum 2.5s duration (Validation Rule 2, Section 5.13).
    Returns (waveform_tensor, sample_rate) or raises ValueError.
    """
    waveform, sr = torchaudio.load(audio_path)

    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)

    if sr != SAMPLE_RATE:
        resampler = torchaudio.transforms.Resample(sr, SAMPLE_RATE)
        waveform = resampler(waveform)
        sr = SAMPLE_RATE

    duration = waveform.shape[1] / sr
    if duration < MIN_AUDIO_SECONDS:
        raise ValueError(
            f'Audio too short: {duration:.1f}s (minimum: {MIN_AUDIO_SECONDS}s). '
            f'ECAPA-TDNN requires sufficient phonetic diversity.'
        )

    return waveform, sr


def extract_embedding(audio_path: str) -> np.ndarray:
    """
    Extract 192-dim L2-normalised voice embedding using ECAPA-TDNN (Section 5.9 Stage 3).
    """
    waveform, _ = load_and_validate_audio(audio_path)
    waveform = waveform.to(device)

    with torch.no_grad():
        embedding = classifier.encode_batch(waveform)
        embedding = embedding.squeeze().cpu().numpy()

    if embedding.ndim > 1:
        embedding = embedding.squeeze()

    norm = np.linalg.norm(embedding)
    if norm > 0:
        embedding = embedding / norm

    return embedding.astype(np.float32)


def enroll_speaker(audio_path: str, name: str, relation: str, patient_id: str) -> dict:
    """
    Register a new speaker into the patient's Circle of Trust.
    Stores 192-dim embedding in FAISS index + speaker_registry.
    """
    embedding = extract_embedding(audio_path)
    speaker_id = str(uuid.uuid4())

    faiss_index.add(embedding.reshape(1, -1))
    speaker_registry.append({
        'speaker_id': speaker_id,
        'name': name,
        'relation': relation,
        'patient_id': patient_id,
        'faiss_index': faiss_index.ntotal - 1
    })

    return {
        'status': 'enrolled',
        'speaker_id': speaker_id,
        'name': name,
        'relation': relation,
        'patient_id': patient_id,
        'embedding_dim': len(embedding),
        'total_enrolled': faiss_index.ntotal
    }


def identify_speaker(audio_path: str, patient_id: str) -> dict:
    """
    Identify a speaker via FAISS cosine similarity search (Section 5.9 Stage 4).
    Applies Validation Rules 1–3 (Section 5.13).
    """
    if faiss_index.ntotal == 0:
        return {
            'status': 'unrecognised',
            'reason': 'No speakers enrolled in Circle of Trust'
        }

    embedding = extract_embedding(audio_path)

    patient_speakers = [s for s in speaker_registry if s['patient_id'] == patient_id]
    if not patient_speakers:
        return {
            'status': 'unrecognised',
            'reason': f'No speakers enrolled for patient {patient_id}'
        }

    similarities, indices = faiss_index.search(embedding.reshape(1, -1), k=faiss_index.ntotal)
    similarities = similarities[0]
    indices = indices[0]

    best_score = -1
    best_speaker = None

    for speaker in patient_speakers:
        idx = speaker['faiss_index']
        pos = np.where(indices == idx)[0]
        if len(pos) > 0:
            score = float(similarities[pos[0]])
            if score > best_score:
                best_score = score
                best_speaker = speaker

    if best_score >= SIMILARITY_THRESHOLD:
        return {
            'status': 'identified',
            'name': best_speaker['name'],
            'relation': best_speaker['relation'],
            'confidence': round(best_score, 4),
            'speaker_id': best_speaker['speaker_id'],
            'patient_id': patient_id,
            'threshold_used': SIMILARITY_THRESHOLD,
            'alert_text': f"This is {best_speaker['name']} — Your {best_speaker['relation']}"
        }
    else:
        return {
            'status': 'unrecognised',
            'reason': f'Best cosine similarity {best_score:.4f} < threshold {SIMILARITY_THRESHOLD}',
            'best_score': round(best_score, 4),
            'alert_text': 'Unrecognised Speaker'
        }


# ── FastAPI App ──────────────────────────────────────────────────────────────

app = FastAPI(
    title='Nesyan Voice Recognition API',
    description='AI Voice Recognition & Speaker Identification — ECAPA-TDNN + FAISS',
    version='1.0.0'
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=['*'],
    allow_methods=['*'],
    allow_headers=['*'],
)


@app.get('/')
async def root():
    """Health check — returns system status and enrolled speaker count."""
    return {
        'system': 'Nesyan Voice Recognition API',
        'status': 'running',
        'enrolled_speakers': faiss_index.ntotal,
        'embedding_dim': EMBEDDING_DIM,
        'similarity_threshold': SIMILARITY_THRESHOLD
    }


@app.post('/enroll')
async def enroll_endpoint(
    audio: UploadFile = File(..., description='Voice sample (.wav, min 3s)'),
    name: str = Form(..., description='Speaker name (e.g. Mona)'),
    relation: str = Form(..., description='Relation to patient (e.g. Daughter)'),
    patient_id: str = Form(default='patient_001', description='Patient UUID')
):
    """
    Enroll a new speaker into the patient's Circle of Trust.
    Extracts 192-dim ECAPA-TDNN embedding and stores in FAISS index.
    """
    with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp:
        tmp.write(await audio.read())
        tmp_path = tmp.name

    try:
        result = enroll_speaker(tmp_path, name=name, relation=relation, patient_id=patient_id)
        return result
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        os.unlink(tmp_path)  # Raw audio never persisted (Privacy-by-Design, Section 5.12)


@app.post('/identify')
async def identify_endpoint(
    audio: UploadFile = File(..., description='Live audio capture (.wav, min 2.5s)'),
    patient_id: str = Form(default='patient_001', description='Patient UUID')
):
    """
    Identify a speaker from live audio.
    Returns JSON payload for delivery to patient device (Section 5.9 Stage 4).
    """
    with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp:
        tmp.write(await audio.read())
        tmp_path = tmp.name

    try:
        result = identify_speaker(tmp_path, patient_id=patient_id)
        return result
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        os.unlink(tmp_path)  # Raw audio purged from disk (Table 5.8)


@app.get('/speakers/{patient_id}')
async def list_speakers(patient_id: str):
    """
    List all enrolled speakers for a patient (Circle of Trust).
    """
    speakers = [
        {
            'name': s['name'],
            'relation': s['relation'],
            'speaker_id': s['speaker_id']
        }
        for s in speaker_registry if s['patient_id'] == patient_id
    ]
    return {
        'patient_id': patient_id,
        'total_enrolled': len(speakers),
        'circle_of_trust': speakers
    }
