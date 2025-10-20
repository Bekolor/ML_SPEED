"""
SPEED SPI Aviation Safety Root Cause Analyzer - Enhanced Version
================================================================

A production-ready pipeline for analyzing aviation safety incidents using LM Studio.

Features:
- Parallel processing with configurable workers
- Comprehensive checkpoint/resume capability
- Robust JSON parsing with fallbacks
- Metrics and analytics tracking
- Validation at each pipeline stage
- Confidence scoring
- Dry-run mode for testing
- Enhanced error handling and logging
"""

import os
import re
import json
import time
import math
import logging
import argparse
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional, Tuple, Generator
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from collections import defaultdict

import requests
import pandas as pd
from tqdm import tqdm


# ============================================================================
# Configuration
# ============================================================================

@dataclass
class PipelineConfig:
    """Complete pipeline configuration."""
    
    # Input/Output
    input_file: str = "SPEED_SPI.xlsx"
    sheet_name: Optional[str] = None
    output_dir: str = "outputs"
    
    # LM Studio
    endpoint: str = "http://localhost:1234/v1/chat/completions"
    model: str = "lmstudio-local"
    reasoning_effort: str = "none"  # one of: none, low, medium, high
    
    # Performance
    requests_per_minute: int = 60
    max_workers: int = 4  # Parallel processing threads
    request_timeout_seconds: int = 60
    max_retries: int = 3
    
    # Token budgets (-1 means unlimited if supported by server)
    categories_context_budget: int = 6000
    categories_response_tokens: int = -1
    phase1_max_tokens: int = -1
    phase3_max_tokens: int = 256
    
    # Processing options
    dry_run: bool = False
    skip_phase1: bool = False
    skip_phase2: bool = False
    skip_phase3: bool = False
    force_reprocess: bool = False
    
    # Validation
    min_categories: int = 10
    max_categories: int = 20
    
    # Logging
    log_level: str = "INFO"
    log_api_calls: bool = True


@dataclass
class PipelineMetrics:
    """Track pipeline execution metrics."""
    
    phase1_total: int = 0
    phase1_success: int = 0
    phase1_errors: int = 0
    phase1_skipped: int = 0
    phase1_duration: float = 0.0
    
    phase2_attempts: int = 0
    phase2_success: bool = False
    phase2_duration: float = 0.0
    
    phase3_total: int = 0
    phase3_success: int = 0
    phase3_errors: int = 0
    phase3_duration: float = 0.0
    
    total_api_calls: int = 0
    total_api_errors: int = 0
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
    
    def summary(self) -> str:
        """Generate human-readable summary."""
        lines = [
            "="*60,
            "Pipeline Execution Summary",
            "="*60,
            f"\nPhase 1 (Event Analysis):",
            f"  Total events: {self.phase1_total}",
            f"  Successful: {self.phase1_success}",
            f"  Errors: {self.phase1_errors}",
            f"  Skipped (cached): {self.phase1_skipped}",
            f"  Duration: {self.phase1_duration:.2f}s",
            f"\nPhase 2 (Category Generation):",
            f"  Success: {'✓' if self.phase2_success else '✗'}",
            f"  Attempts: {self.phase2_attempts}",
            f"  Duration: {self.phase2_duration:.2f}s",
            f"\nPhase 3 (Classification):",
            f"  Total events: {self.phase3_total}",
            f"  Successful: {self.phase3_success}",
            f"  Errors: {self.phase3_errors}",
            f"  Duration: {self.phase3_duration:.2f}s",
            f"\nAPI Statistics:",
            f"  Total calls: {self.total_api_calls}",
            f"  Errors: {self.total_api_errors}",
            f"  Success rate: {(1 - self.total_api_errors/max(1, self.total_api_calls))*100:.1f}%",
            "="*60,
        ]
        return "\n".join(lines)


# ============================================================================
# Logging Setup
# ============================================================================

def setup_logging(config: PipelineConfig) -> logging.Logger:
    """Configure logging with file and console handlers."""
    log_dir = Path(config.output_dir) / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    log_file = log_dir / f"pipeline_{timestamp}.log"
    
    # Create logger
    logger = logging.getLogger("SPEED_SPI")
    logger.setLevel(getattr(logging, config.log_level.upper()))
    
    # File handler
    fh = logging.FileHandler(log_file, encoding="utf-8")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    ))
    
    # Console handler
    ch = logging.StreamHandler()
    ch.setLevel(getattr(logging, config.log_level.upper()))
    ch.setFormatter(logging.Formatter('%(levelname)s - %(message)s'))
    
    logger.addHandler(fh)
    logger.addHandler(ch)
    
    return logger


# ============================================================================
# Config validation helpers
# ============================================================================

def validate_token_limits(config: PipelineConfig, logger: logging.Logger) -> None:
    """Warn about unrealistic token limits. Allows -1 as 'unlimited'."""
    max_reasonable = 16384

    if config.phase1_max_tokens != -1 and config.phase1_max_tokens > max_reasonable:
        logger.warning(
            f"phase1_max_tokens ({config.phase1_max_tokens}) is very high. "
            f"Most models support up to ~{max_reasonable} tokens."
        )

    if config.categories_response_tokens != -1 and config.categories_response_tokens > max_reasonable:
        logger.warning(
            f"categories_response_tokens ({config.categories_response_tokens}) is very high. "
            f"Consider using 800–3000 depending on the model."
        )

    if config.phase3_max_tokens != -1 and config.phase3_max_tokens > max_reasonable:
        logger.warning(
            f"phase3_max_tokens ({config.phase3_max_tokens}) is very high for short classifications."
        )


# ============================================================================
# Utilities
# ============================================================================

def ensure_dir(path: Path) -> None:
    """Create directory if it doesn't exist."""
    path.mkdir(parents=True, exist_ok=True)


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    """Read JSONL file with error handling."""
    if not path.exists():
        return []
    
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as e:
                logging.warning(f"Malformed JSON at line {i}: {e}")
                rows.append({"raw": line, "error": str(e)})
    return rows


def append_jsonl(path: Path, obj: Dict[str, Any]) -> None:
    """Append object to JSONL file."""
    ensure_dir(path.parent)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(obj, ensure_ascii=False) + "\n")


def write_json(path: Path, obj: Any) -> None:
    """Write object to JSON file."""
    ensure_dir(path.parent)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def read_json(path: Path) -> Any:
    """Read JSON file."""
    if not path.exists():
        return None
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def estimate_tokens(text: str) -> int:
    """
    Improved token estimation for Turkish text.
    
    Uses combination of character count and word count for better accuracy.
    Turkish typically requires ~3.5 chars per token.
    """
    if not text:
        return 0
    
    chars = len(text)
    words = len(text.split())
    
    # Weighted average of char-based and word-based estimates
    char_estimate = chars / 3.5
    word_estimate = words * 1.3
    
    return max(1, int((char_estimate + word_estimate) / 2))


def now_iso() -> str:
    """Return current timestamp in ISO format."""
    return time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime())


def coalesce(*vals) -> str:
    """Return first non-empty value."""
    for v in vals:
        if v is None:
            continue
        if isinstance(v, float) and math.isnan(v):
            continue
        s = str(v).strip()
        if s and s.lower() not in ("nan", "none", "null"):
            return s
    return ""


class RateLimiter:
    """Token bucket rate limiter."""
    
    def __init__(self, rpm: int):
        self.min_interval = 60.0 / max(1, rpm)
        self._last_time = 0.0
    
    def wait(self):
        """Wait if necessary to maintain rate limit."""
        elapsed = time.time() - self._last_time
        to_sleep = self.min_interval - elapsed
        if to_sleep > 0:
            time.sleep(to_sleep)
        self._last_time = time.time()


# ============================================================================
# Enhanced JSON Parsing
# ============================================================================

def safe_parse_json(text: str, logger: Optional[logging.Logger] = None) -> Optional[Any]:
    """
    Robust JSON parsing with multiple fallback strategies.
    
    Strategies:
    1. Direct parse
    2. Strip markdown code blocks
    3. Extract largest JSON object
    4. Extract JSON array
    5. Extract with regex patterns
    """
    
    if not text or not text.strip():
        return None
    
    original_text = text
    
    # Strategy 1: Direct parse
    try:
        return json.loads(text)
    except Exception:
        pass
    
    # Strategy 2: Strip markdown code blocks
    text = re.sub(r'```json\s*', '', text, flags=re.IGNORECASE)
    text = re.sub(r'```\s*', '', text)
    text = text.strip()
    
    try:
        return json.loads(text)
    except Exception:
        pass
    
    # Strategy 3: Extract largest JSON object
    try:
        start = text.find("{")
        end = text.rfind("}")
        if start != -1 and end != -1 and end > start:
            candidate = text[start:end + 1]
            return json.loads(candidate)
    except Exception:
        pass
    
    # Strategy 4: Extract JSON array
    try:
        start = text.find("[")
        end = text.rfind("]")
        if start != -1 and end != -1 and end > start:
            candidate = text[start:end + 1]
            return json.loads(candidate)
    except Exception:
        pass
    
    # Strategy 5: Look for JSON-like patterns
    patterns = [
        r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}',  # Nested objects
        r'\[[^\[\]]*(?:\[[^\[\]]*\][^\[\]]*)*\]',  # Nested arrays
    ]
    
    for pattern in patterns:
        matches = re.findall(pattern, text, re.DOTALL)
        for match in matches:
            try:
                return json.loads(match)
            except Exception:
                continue
    
    if logger:
        logger.warning(f"Failed to parse JSON from response: {original_text[:200]}...")
    
    return None


# ============================================================================
# LM Studio Client
# ============================================================================

class LMStudioClient:
    """Enhanced LM Studio API client with retry logic and logging."""
    
    def __init__(self, config: PipelineConfig, metrics: PipelineMetrics, 
                 logger: logging.Logger):
        self.config = config
        self.metrics = metrics
        self.logger = logger
        self.rate_limiter = RateLimiter(config.requests_per_minute)
        # Capability flags (auto-adjust based on server errors)
        self._supports_response_format = True
        self._supports_json_schema = True
        self._supports_json_object = False  # Some servers don't accept this; enable if detected
    
    def chat(self,
             messages: List[Dict[str, str]],
             temperature: float = 0.7,
             max_tokens: int = 512,
             stop: Optional[List[str]] = None,
             log_name: Optional[str] = None,
             json_mode: bool = False,
             reasoning_effort: Optional[str] = None,
             json_schema: Optional[Dict[str, Any]] = None,
             extra_params: Optional[Dict[str, Any]] = None) -> str:
        """
        Send chat completion request with retry logic.
        
        Args:
            messages: Conversation messages
            temperature: Sampling temperature
            max_tokens: Maximum tokens in response
            stop: Stop sequences
            log_name: Optional name for logging this call
            
        Returns:
            Response text
            
        Raises:
            RuntimeError: If all retries fail
        """
        
        if self.config.dry_run:
            self.logger.info(f"[DRY RUN] Would call API with {len(messages)} messages")
            return '{"mock": "response"}'
        
        payload: Dict[str, Any] = {
            "model": self.config.model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "stream": False,
        }

        if stop:
            payload["stop"] = stop

        # Encourage JSON-only outputs when requested
        if json_mode and self._supports_response_format:
            if json_schema and self._supports_json_schema:
                payload["response_format"] = {
                    "type": "json_schema",
                    "json_schema": json_schema,
                }
            elif self._supports_json_object:
                payload["response_format"] = {"type": "json_object"}
            else:
                # Safe default that most servers accept
                payload["response_format"] = {"type": "text"}

        # Try to disable/adjust reasoning if supported by server
        if reasoning_effort is not None:
            payload["reasoning"] = {"effort": reasoning_effort}

        # Allow custom params passthrough
        if extra_params:
            try:
                payload.update(extra_params)
            except Exception:
                pass
        
        last_error = None
        
        for attempt in range(1, self.config.max_retries + 1):
            try:
                self.rate_limiter.wait()
                
                self.logger.debug(f"API call attempt {attempt}/{self.config.max_retries}")
                
                response = requests.post(
                    self.config.endpoint,
                    headers={"Content-Type": "application/json"},
                    json=payload,
                    timeout=self.config.request_timeout_seconds,
                )
                
                self.metrics.total_api_calls += 1
                
                # Log request/response if enabled
                if self.config.log_api_calls and log_name:
                    self._log_api_call(log_name, payload, response)
                
                if response.status_code != 200:
                    # Build informative error and adjust capabilities if needed
                    error_detail = ""
                    try:
                        err_json = response.json()
                        if isinstance(err_json, dict):
                            # LM Studio often returns {"error": "message"} or {"error": {"message": ...}}
                            e = err_json.get("error")
                            if isinstance(e, dict):
                                error_detail = e.get("message") or str(e)
                            elif isinstance(e, str):
                                error_detail = e
                    except Exception:
                        error_detail = response.text[:500]

                    error_msg = (
                        f"HTTP {response.status_code}: {error_detail} | "
                        f"Endpoint: {self.config.endpoint} | Model: {self.config.model}"
                    )
                    self.logger.warning(f"API error: {error_msg}")

                    # Downgrade response_format support on hint
                    if "response_format" in (error_detail or ""):
                        if "json_schema" in (error_detail or ""):
                            self._supports_json_schema = False
                            self.logger.debug("Disabling json_schema; server rejected it.")
                        self._supports_response_format = False
                        self.logger.debug("Disabling response_format; retrying without it.")

                    last_error = RuntimeError(error_msg)

                    # Exponential backoff
                    sleep_time = min(2 ** attempt, 10)
                    self.logger.debug(f"Retrying in {sleep_time}s...")
                    time.sleep(sleep_time)
                    continue
                
                # Parse response JSON safely and support multiple formats
                try:
                    data = response.json()
                except Exception:
                    last_error = RuntimeError(f"Invalid JSON response: {response.text[:500]}")
                    self.logger.warning(str(last_error))
                    time.sleep(min(2 ** attempt, 10))
                    continue

                # Surface explicit API error payloads
                api_error = data.get("error")
                if api_error:
                    self.logger.warning(f"API returned error: {api_error}")
                    last_error = RuntimeError(str(api_error))
                    time.sleep(min(2 ** attempt, 10))
                    continue

                # Prefer chat-style content, but fall back to completion-style text
                choices = data.get("choices") or []
                content = ""
                if choices:
                    first = choices[0] or {}
                    if isinstance(first, dict):
                        msg = first.get("message")
                        if isinstance(msg, dict):
                            content = msg.get("content", "") or ""
                            # Some reasoning-capable servers place output under 'reasoning' and leave content empty
                            if not content:
                                reasoning_text = msg.get("reasoning")
                                if isinstance(reasoning_text, str) and reasoning_text.strip():
                                    content = reasoning_text
                        if not content:
                            content = first.get("text", "") or ""

                if not content:
                    self.logger.warning("Empty response from API")
                    last_error = RuntimeError("Empty response")
                    time.sleep(min(2 ** attempt, 10))
                    continue
                
                return content
                
            except requests.Timeout:
                last_error = RuntimeError(f"Request timeout after {self.config.request_timeout_seconds}s")
                self.logger.warning(str(last_error))
                time.sleep(min(2 ** attempt, 10))
                
            except requests.RequestException as e:
                last_error = e
                self.logger.warning(f"Request failed: {e}")
                time.sleep(min(2 ** attempt, 10))
                
            except Exception as e:
                last_error = e
                self.logger.error(f"Unexpected error: {e}", exc_info=True)
                time.sleep(min(2 ** attempt, 10))
        
        # All retries exhausted
        self.metrics.total_api_errors += 1
        raise RuntimeError(
            f"API call failed after {self.config.max_retries} attempts. "
            f"Last error: {last_error}"
        )
    
    def _log_api_call(self, log_name: str, payload: Dict, response: requests.Response):
        """Log API request and response to file."""
        log_dir = Path(self.config.output_dir) / "logs" / "api_calls"
        ensure_dir(log_dir)
        
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        
        # Log request
        req_file = log_dir / f"{log_name}_{timestamp}_request.json"
        write_json(req_file, {
            "timestamp": now_iso(),
            "payload": payload
        })
        
        # Log response
        resp_file = log_dir / f"{log_name}_{timestamp}_response.json"
        try:
            resp_data = response.json()
        except Exception:
            resp_data = {"raw": response.text}
        
        write_json(resp_file, {
            "timestamp": now_iso(),
            "status_code": response.status_code,
            "response": resp_data
        })


# ============================================================================
# Prompts (Turkish)
# ============================================================================

PHASE1_SYSTEM = """Sen deneyimli bir havacılık emniyeti uzmanısın. Aşağıdaki olay bilgilerini analiz ederek, olayı açıklayan temel (kök) faktörleri, tekrar eden temaları ve benzer emniyet faktörlerini çıkar.

Cevabı kesinlikle geçerli JSON olarak döndür. Anahtarlar:
- underlying_factors: [string] - Olayın temelindeki faktörler
- common_themes: [string] - Ortak temalar ve tekrarlayan nedenler
- similar_safety_factors: [string] - Benzer emniyet faktörleri
- confidence: float - Analizin güvenilirlik skoru (0.0-1.0)
- notes: string - Opsiyonel notlar

ÖNEMLİ: Sadece geçerli JSON döndür, başka hiçbir açıklama ekleme."""

PHASE1_USER_TEMPLATE = """Olay bilgileri:
- Occurrence No: {occ}
- Description: {desc}
- Probable Cause: {cause}
- Flight Safety Events: {events}
- Flight Safety Factors: {factors}

Soru(lar):
1) Bu olayın temelinde hangi faktörler yatıyor?
2) Hangi ortak temalar ve tekrarlayan nedenler gözlemleniyor?
3) Benzer emniyet faktörleri nelerdir?

Sadece geçerli JSON formatında yanıt ver. Markdown, açıklama veya ek metin ekleme."""

# JSON Schemas for structured output (LM Studio/OpenAI Compatible)
PHASE1_JSON_SCHEMA = {
    "name": "phase1_root_cause_analysis",
    "schema": {
        "$schema": "http://json-schema.org/draft-07/schema#",
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "underlying_factors": {
                "type": "array",
                "items": {"type": "string"},
                "minItems": 1
            },
            "common_themes": {
                "type": "array",
                "items": {"type": "string"}
            },
            "similar_safety_factors": {
                "type": "array",
                "items": {"type": "string"}
            },
            "confidence": {"type": "number", "minimum": 0.0, "maximum": 1.0},
            "notes": {"type": "string"}
        },
        "required": [
            "underlying_factors",
            "common_themes",
            "similar_safety_factors",
            "confidence"
        ]
    }
}

PHASE2_SYSTEM = """Sen kıdemli bir havacılık emniyeti uzmanısın. Aşağıdaki çok sayıdaki olay analizinden yola çıkarak, veriden doğal olarak ortaya çıkan {min_cat} ile {max_cat} arasında mantıksal, birbirinden ayrışık ve havacılık emniyeti açısından anlamlı kök neden kategorisi tasarla.

Her kategori için şu alanları üret ve sadece JSON döndür:
- code: string (RC01, RC02, ...)
- name: string (kısa ve öz, Türkçe)
- description: string (1-2 cümle, Türkçe)

ÖNEMLİ: Sadece JSON array döndür, başka hiçbir metin ekleme."""

PHASE2_USER_TEMPLATE_SINGLE = """AŞAMA 1 analiz çıktılarının tamamı aşağıda verilmiştir.
Her analizi incele ve {min_cat}-{max_cat} arası kök neden kategorisi üret.

Analiz listesi:
{analyses}

Sadece aşağıdaki formatta JSON array döndür:
[{{"code":"RC01","name":"...","description":"..."}}, ...]

Markdown veya açıklama ekleme."""

PHASE2_JSON_SCHEMA = {
    "name": "phase2_root_cause_categories",
    "schema": {
        "$schema": "http://json-schema.org/draft-07/schema#",
        "type": "array",
        "items": {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "code": {"type": "string"},
                "name": {"type": "string"},
                "description": {"type": "string"}
            },
            "required": ["code", "name", "description"]
        },
        "minItems": 1
    }
}

def build_phase2_json_schema(min_cat: int, max_cat: int) -> Dict[str, Any]:
    """Return a JSON schema enforcing category count between bounds."""
    schema = {
        "name": "phase2_root_cause_categories",
        "schema": {
            "$schema": "http://json-schema.org/draft-07/schema#",
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "code": {"type": "string"},
                    "name": {"type": "string"},
                    "description": {"type": "string"}
                },
                "required": ["code", "name", "description"]
            },
            "minItems": max(1, int(min_cat)),
            "maxItems": int(max_cat),
        }
    }
    return schema

PHASE2_USER_TEMPLATE_CHUNK = """Aşağıdaki parça analizlerden yola çıkarak {min_cat}-{max_cat} arası aday kök neden kategorisi öner.

Analiz parçası:
{analyses}

Sadece JSON array döndür:
[{{"code":"RC01","name":"...","description":"..."}}]"""

PHASE2_USER_TEMPLATE_MERGE = """Aşağıdaki farklı parçalardan türetilmiş aday kategori listelerini birleştir.
Aynı/benzer kategorileri birleştir, {min_cat}-{max_cat} arası net ve birbirinden ayrışık bir nihai liste üret.

Aday listeleri:
{candidates}

Sadece JSON array döndür:
[{{"code":"RC01","name":"...","description":"..."}}]"""

PHASE3_SYSTEM = """Sen bir havacılık emniyeti uzmanısın. Verilen olay metnini, sağlanan kök neden kategorileri arasından en uygun kategori(ler)e ata.

Sadece kategori kodlarını tire ile ayırarak döndür (örnek: RC01 - RC05 - RC12).
Başka hiçbir metin, açıklama veya markdown ekleme.
En fazla 3 kategori seç."""

PHASE3_USER_TEMPLATE = """Kategoriler:
{categories}

Olay bilgileri:
- Occurrence No: {occ}
- Description: {desc}
- Probable Cause: {cause}
- Flight Safety Events: {events}
- Flight Safety Factors: {factors}

Çıktı formatı: RC01 - RC05 (sadece kodlar, tire ile ayrılmış)"""


# ============================================================================
# Phase 1: Event Analysis
# ============================================================================

def analyze_single_event(
    row: Dict[str, Any],
    client: LMStudioClient,
    config: PipelineConfig,
    logger: logging.Logger
) -> Dict[str, Any]:
    """Analyze a single event (for parallel processing)."""
    
    occ = str(row.get("Occurrence No", "")) or str(row.get("Occurrence", ""))
    desc = coalesce(row.get("Description"))
    cause = coalesce(row.get("Probable Cause"))
    events = coalesce(row.get("Flight Safety Events"))
    factors = coalesce(row.get("Flight Safety Factors"))
    
    messages = [
        {"role": "system", "content": PHASE1_SYSTEM},
        {"role": "user", "content": PHASE1_USER_TEMPLATE.format(
            occ=occ, desc=desc, cause=cause, events=events, factors=factors
        )},
    ]
    
    try:
        content = client.chat(
            messages,
            temperature=0.7,
            max_tokens=config.phase1_max_tokens,
            log_name=f"phase1_{occ}",
            json_mode=True,
            reasoning_effort=config.reasoning_effort,
            json_schema=PHASE1_JSON_SCHEMA
        )
        
        parsed = safe_parse_json(content, logger)
        
        if parsed is None:
            logger.warning(f"Failed to parse JSON for occurrence {occ}")
            parsed = {"raw": content, "parse_error": True}
        
        # Validate expected fields
        if isinstance(parsed, dict) and "underlying_factors" not in parsed:
            logger.warning(f"Missing expected fields for occurrence {occ}")
        
        return {
            "occurrence_no": occ,
            "analysis": parsed,
            "timestamp": now_iso(),
            "success": True
        }
        
    except Exception as e:
        logger.error(f"Error analyzing occurrence {occ}: {e}")
        return {
            "occurrence_no": occ,
            "error": str(e),
            "timestamp": now_iso(),
            "success": False
        }


def phase1_analyze(
    df: pd.DataFrame,
    client: LMStudioClient,
    config: PipelineConfig,
    metrics: PipelineMetrics,
    logger: logging.Logger
) -> List[Dict[str, Any]]:
    """
    Phase 1: Analyze individual events.
    
    Uses parallel processing and checkpoint/resume capability.
    """
    
    logger.info("Starting Phase 1: Event Analysis")
    start_time = time.time()
    
    results_path = Path(config.output_dir) / "phase1_results.jsonl"
    
    # Load existing results
    existing = read_jsonl(results_path)
    processed_ids = {
        str(r.get("occurrence_no"))
        for r in existing
        if "occurrence_no" in r and r.get("success", False)
    }
    
    outputs = list(existing) if not config.force_reprocess else []
    
    # Filter rows to process
    rows = df.to_dict(orient="records")
    to_process = []
    
    for row in rows:
        occ = str(row.get("Occurrence No", "")) or str(row.get("Occurrence", ""))
        if not occ:
            continue
        
        if occ in processed_ids and not config.force_reprocess:
            metrics.phase1_skipped += 1
            continue
        
        to_process.append(row)
    
    metrics.phase1_total = len(to_process)
    
    if not to_process:
        logger.info("No events to process (all cached)")
        metrics.phase1_duration = time.time() - start_time
        return outputs
    
    logger.info(f"Processing {len(to_process)} events ({metrics.phase1_skipped} cached)")
    
    # Parallel processing
    with ThreadPoolExecutor(max_workers=config.max_workers) as executor:
        futures = {
            executor.submit(analyze_single_event, row, client, config, logger): row
            for row in to_process
        }
        
        with tqdm(total=len(futures), desc="Phase 1", unit="event") as pbar:
            for future in as_completed(futures):
                try:
                    result = future.result()
                    append_jsonl(results_path, result)
                    outputs.append(result)
                    
                    if result.get("success"):
                        metrics.phase1_success += 1
                    else:
                        metrics.phase1_errors += 1
                    
                except Exception as e:
                    logger.error(f"Future failed: {e}")
                    metrics.phase1_errors += 1
                
                pbar.update(1)
    
    metrics.phase1_duration = time.time() - start_time
    logger.info(f"Phase 1 completed in {metrics.phase1_duration:.2f}s")
    
    return outputs


# ============================================================================
# Phase 2: Category Generation
# ============================================================================

def build_analyses_text(phase1_results: List[Dict[str, Any]], max_chars: int = 2000) -> str:
    """Build text representation of analyses for prompting."""
    parts = []
    
    for r in phase1_results:
        if not r.get("success"):
            continue
        
        occ = r.get("occurrence_no", "")
        analysis = r.get("analysis")
        
        if analysis is None:
            continue
        
        try:
            text = json.dumps(analysis, ensure_ascii=False)
        except Exception:
            text = str(analysis)
        
        # Limit each analysis to prevent explosion
        if len(text) > max_chars:
            text = text[:max_chars] + "..."
        
        parts.append(f'{{"occ":"{occ}","analysis":{text}}}')
    
    return "\n".join(parts)


def validate_categories(
    categories: List[Dict[str, str]],
    config: PipelineConfig,
    logger: logging.Logger
) -> bool:
    """
    Validate category list meets requirements.
    
    Checks:
    - List type
    - Count within bounds
    - Required fields present
    - Unique codes
    """
    
    if not isinstance(categories, list):
        logger.error("Categories must be a list")
        return False
    
    if not (config.min_categories <= len(categories) <= config.max_categories):
        logger.error(
            f"Category count {len(categories)} not in range "
            f"[{config.min_categories}, {config.max_categories}]"
        )
        return False
    
    codes = []
    for i, cat in enumerate(categories):
        if not isinstance(cat, dict):
            logger.error(f"Category {i} is not a dict")
            return False
        
        if not all(k in cat for k in ["code", "name", "description"]):
            logger.error(f"Category {i} missing required fields")
            return False
        
        code = cat.get("code", "").strip()
        if not code:
            logger.error(f"Category {i} has empty code")
            return False
        
        codes.append(code)
    
    if len(codes) != len(set(codes)):
        logger.error("Duplicate category codes found")
        return False
    
    return True


def dedupe_and_normalize_categories(
    categories: List[Dict[str, str]],
    logger: logging.Logger,
    max_count: Optional[int] = None,
) -> List[Dict[str, str]]:
    """Deduplicate and normalize category codes."""
    
    seen = set()
    out = []
    
    for cat in categories:
        name = cat.get("name", "").strip()
        desc = cat.get("description", "").strip()
        
        if not name:
            continue
        
        # Use name+desc as key for deduplication
        key = (name.lower(), desc.lower()[:100])
        
        if key in seen:
            logger.debug(f"Skipping duplicate category: {name}")
            continue
        
        seen.add(key)
        out.append({
            "code": cat.get("code", ""),
            "name": name,
            "description": desc
        })
    
    # If too many, trim to the requested maximum while preserving order
    if isinstance(max_count, int) and max_count > 0 and len(out) > max_count:
        out = out[:max_count]

    # Reindex codes to RC01, RC02, ...
    for i, cat in enumerate(out, start=1):
        cat["code"] = f"RC{i:02d}"
    
    return out


def phase2_generate_categories_single(
    client: LMStudioClient,
    analyses_text: str,
    config: PipelineConfig,
    logger: logging.Logger
) -> List[Dict[str, str]]:
    """Generate categories in a single call."""
    
    messages = [
        {"role": "system", "content": PHASE2_SYSTEM.format(
            min_cat=config.min_categories,
            max_cat=config.max_categories
        )},
        {"role": "user", "content": PHASE2_USER_TEMPLATE_SINGLE.format(
            analyses=analyses_text,
            min_cat=config.min_categories,
            max_cat=config.max_categories
        )},
    ]
    
    content = client.chat(
        messages,
        temperature=0.7,
        max_tokens=config.categories_response_tokens,
        log_name="phase2_full",
        json_mode=True,
        reasoning_effort=config.reasoning_effort,
        json_schema=build_phase2_json_schema(config.min_categories, config.max_categories)
    )
    
    categories = safe_parse_json(content, logger)
    
    if not isinstance(categories, list):
        raise RuntimeError(f"Expected list, got {type(categories)}")
    
    return categories


def phase2_generate_categories_chunked(
    client: LMStudioClient,
    analyses_text: str,
    config: PipelineConfig,
    logger: logging.Logger
) -> List[Dict[str, str]]:
    """Generate categories using chunking + merge strategy."""
    
    # Split into chunks
    parts = analyses_text.split("\n")
    chunks: List[str] = []
    current_chunk = []
    current_tokens = 0
    
    for part in parts:
        part_tokens = estimate_tokens(part) + 10  # Buffer
        
        if current_tokens + part_tokens > config.categories_context_budget and current_chunk:
            chunks.append("\n".join(current_chunk))
            current_chunk = [part]
            current_tokens = part_tokens
        else:
            current_chunk.append(part)
            current_tokens += part_tokens
    
    if current_chunk:
        chunks.append("\n".join(current_chunk))
    
    logger.info(f"Split analyses into {len(chunks)} chunks")
    
    # Generate candidates per chunk
    candidate_lists: List[List[Dict[str, str]]] = []
    
    for i, chunk in enumerate(tqdm(chunks, desc="Generating category candidates")):
        messages = [
            {"role": "system", "content": PHASE2_SYSTEM.format(
                min_cat=config.min_categories,
                max_cat=config.max_categories
            )},
            {"role": "user", "content": PHASE2_USER_TEMPLATE_CHUNK.format(
                analyses=chunk,
                min_cat=config.min_categories,
                max_cat=config.max_categories
            )},
        ]
        
        try:
            content = client.chat(
                messages,
                temperature=0.7,
                max_tokens=config.categories_response_tokens,
                log_name=f"phase2_chunk_{i+1}",
                json_mode=True,
                reasoning_effort=config.reasoning_effort,
                json_schema=build_phase2_json_schema(config.min_categories, config.max_categories)
            )
            
            parsed = safe_parse_json(content, logger)
            
            if isinstance(parsed, list):
                candidate_lists.append(parsed)
            else:
                logger.warning(f"Chunk {i+1} did not return a list")
                
        except Exception as e:
            logger.error(f"Error processing chunk {i+1}: {e}")
    
    if not candidate_lists:
        raise RuntimeError("No valid candidate lists generated")
    
    # Merge candidates
    candidates_text = json.dumps(candidate_lists, ensure_ascii=False)
    
    messages = [
        {"role": "system", "content": PHASE2_SYSTEM.format(
            min_cat=config.min_categories,
            max_cat=config.max_categories
        )},
        {"role": "user", "content": PHASE2_USER_TEMPLATE_MERGE.format(
            candidates=candidates_text,
            min_cat=config.min_categories,
            max_cat=config.max_categories
        )},
    ]
    
    content = client.chat(
        messages,
        temperature=0.7,
        max_tokens=config.categories_response_tokens,
        log_name="phase2_merge",
        json_mode=True,
        reasoning_effort=config.reasoning_effort,
        json_schema=build_phase2_json_schema(config.min_categories, config.max_categories)
    )
    
    categories = safe_parse_json(content, logger)
    
    def _flatten_candidates() -> List[Dict[str, str]]:
        flat: List[Dict[str, str]] = []
        for cl in candidate_lists:
            flat.extend(cl)
        return flat

    if not isinstance(categories, list):
        logger.warning("Merge failed, flattening candidates")
        categories = _flatten_candidates()
    else:
        # If count outside desired range, fallback to combined candidate pool
        if not (config.min_categories <= len(categories) <= config.max_categories):
            logger.warning(
                f"Merge produced {len(categories)} categories; "
                f"fallback to flatten+dedupe within target range"
            )
            categories = _flatten_candidates()

    return categories


def phase2_categories(
    client: LMStudioClient,
    phase1_results: List[Dict[str, Any]],
    config: PipelineConfig,
    metrics: PipelineMetrics,
    logger: logging.Logger
) -> List[Dict[str, str]]:
    """
    Phase 2: Generate root cause categories.
    
    Uses single-call or chunked strategy based on data size.
    Includes validation and checkpoint capability.
    """
    
    logger.info("Starting Phase 2: Category Generation")
    start_time = time.time()
    
    # Check for cached categories
    categories_path = Path(config.output_dir) / "root_cause_categories.json"
    
    if categories_path.exists() and not config.force_reprocess:
        logger.info("Loading cached categories")
        categories = read_json(categories_path)
        
        if validate_categories(categories, config, logger):
            metrics.phase2_success = True
            metrics.phase2_duration = time.time() - start_time
            return categories
        else:
            logger.warning("Cached categories failed validation, regenerating")
    
    # Build analyses text
    analyses_text = build_analyses_text(phase1_results)
    total_tokens = estimate_tokens(analyses_text)
    
    logger.info(f"Total analyses: ~{total_tokens} tokens")
    
    # Choose strategy
    if total_tokens <= config.categories_context_budget:
        logger.info("Using single-call strategy")
        categories = phase2_generate_categories_single(
            client, analyses_text, config, logger
        )
    else:
        logger.info("Using chunked strategy")
        categories = phase2_generate_categories_chunked(
            client, analyses_text, config, logger
        )
    
    metrics.phase2_attempts += 1
    
    # Dedupe and normalize
    categories = dedupe_and_normalize_categories(categories, logger, max_count=config.max_categories)

    # If still below minimum, try augmenting with a full-context generation
    if len(categories) < config.min_categories:
        logger.warning(
            f"Only {len(categories)} categories after merge; attempting augmentation with single-call generation"
        )
        try:
            extra = phase2_generate_categories_single(client, analyses_text, config, logger)
            categories = dedupe_and_normalize_categories(categories + extra, logger, max_count=config.max_categories)
        except Exception as e:
            logger.warning(f"Augmentation attempt failed: {e}")
    
    # Validate
    if not validate_categories(categories, config, logger):
        raise RuntimeError("Generated categories failed validation")
    
    # Save
    write_json(categories_path, categories)
    
    # Also save as Excel
    categories_xlsx = Path(config.output_dir) / "root_cause_categories.xlsx"
    df_cats = pd.DataFrame(categories)
    df_cats.to_excel(categories_xlsx, index=False)
    
    metrics.phase2_success = True
    metrics.phase2_duration = time.time() - start_time
    
    logger.info(f"Phase 2 completed: {len(categories)} categories generated")
    
    return categories


# ============================================================================
# Phase 3: Classification
# ============================================================================

def normalize_rc_code(code: str) -> str:
    """Normalize RC code to standard format (RC01, RC02, ...)."""
    
    code = code.strip().upper()
    
    # Match RCxx or just digits
    match = re.match(r'^(?:RC)?(\d{1,2})$', code)
    if match:
        num = int(match.group(1))
        return f"RC{num:02d}"
    
    # Already in correct format
    if re.match(r'^RC\d{2}$', code):
        return code
    
    return code


def extract_codes(
    text: str,
    valid_codes: set,
    logger: logging.Logger
) -> List[str]:
    """
    Extract category codes from LLM response.
    
    Looks for patterns like:
    - RC01
    - RC01 - RC05
    - RC 01
    """
    
    # Find all RCxx patterns
    patterns = re.findall(r'RC\s*\d{1,2}', text, re.IGNORECASE)
    
    codes = []
    seen = set()
    
    for pattern in patterns:
        code = normalize_rc_code(pattern.replace(" ", ""))
        
        if code in valid_codes and code not in seen:
            codes.append(code)
            seen.add(code)
    
    return codes


def classify_single_event(
    row: Dict[str, Any],
    categories: List[Dict[str, str]],
    client: LMStudioClient,
    config: PipelineConfig,
    logger: logging.Logger
) -> Tuple[str, str]:
    """Classify a single event (for parallel processing)."""
    
    occ = str(row.get("Occurrence No", "")) or str(row.get("Occurrence", ""))
    desc = coalesce(row.get("Description"))
    cause = coalesce(row.get("Probable Cause"))
    events = coalesce(row.get("Flight Safety Events"))
    factors = coalesce(row.get("Flight Safety Factors"))
    
    categories_json = json.dumps(categories, ensure_ascii=False)
    valid_codes = {c["code"] for c in categories}
    
    messages = [
        {"role": "system", "content": PHASE3_SYSTEM},
        {"role": "user", "content": PHASE3_USER_TEMPLATE.format(
            categories=categories_json,
            occ=occ,
            desc=desc,
            cause=cause,
            events=events,
            factors=factors
        )},
    ]
    
    try:
        content = client.chat(
            messages,
            temperature=0.3,
            max_tokens=config.phase3_max_tokens,
            log_name=f"phase3_{occ}",
            reasoning_effort=config.reasoning_effort
        )
        
        # Extract codes
        selected_codes = extract_codes(content, valid_codes, logger)
        
        if not selected_codes:
            logger.warning(f"No valid codes extracted for {occ}: {content[:100]}")
            root_cause = "UNCLASSIFIED"
        else:
            # Limit to 3 codes
            selected_codes = selected_codes[:3]
            root_cause = " - ".join(selected_codes)
        
        return occ, root_cause
        
    except Exception as e:
        logger.error(f"Error classifying {occ}: {e}")
        return occ, f"ERROR: {str(e)[:100]}"


def phase3_classify(
    df: pd.DataFrame,
    categories: List[Dict[str, str]],
    client: LMStudioClient,
    config: PipelineConfig,
    metrics: PipelineMetrics,
    logger: logging.Logger
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Phase 3: Classify events into root cause categories.
    
    Uses parallel processing and generates mapping table.
    """
    
    logger.info("Starting Phase 3: Classification")
    start_time = time.time()
    
    df = df.copy()
    rows = df.to_dict(orient="records")
    
    metrics.phase3_total = len(rows)
    
    # Parallel classification
    mapping_rows = []
    
    with ThreadPoolExecutor(max_workers=config.max_workers) as executor:
        futures = {
            executor.submit(
                classify_single_event, row, categories, client, config, logger
            ): i
            for i, row in enumerate(rows)
        }
        
        with tqdm(total=len(futures), desc="Phase 3", unit="event") as pbar:
            for future in as_completed(futures):
                try:
                    occ, root_cause = future.result()
                    idx = futures[future]
                    
                    df.at[idx, "Root Cause"] = root_cause
                    
                    mapping_rows.append({
                        "Occurrence No": occ,
                        "Root Cause": root_cause
                    })
                    
                    if not root_cause.startswith("ERROR"):
                        metrics.phase3_success += 1
                    else:
                        metrics.phase3_errors += 1
                    
                except Exception as e:
                    logger.error(f"Future failed: {e}")
                    metrics.phase3_errors += 1
                
                pbar.update(1)
    
    # Create mapping DataFrame
    mapping_df = pd.DataFrame(mapping_rows)
    
    # Save mapping table
    mapping_path = Path(config.output_dir) / "occurrence_root_cause_mapping.xlsx"
    mapping_df.to_excel(mapping_path, index=False)
    
    metrics.phase3_duration = time.time() - start_time
    logger.info(f"Phase 3 completed in {metrics.phase3_duration:.2f}s")
    
    return df, mapping_df


# ============================================================================
# Main Pipeline
# ============================================================================

def load_and_validate_data(
    config: PipelineConfig,
    logger: logging.Logger
) -> pd.DataFrame:
    """Load and validate input Excel file."""
    
    logger.info(f"Loading data from {config.input_file}")
    
    if not os.path.exists(config.input_file):
        raise FileNotFoundError(f"Input file not found: {config.input_file}")
    
    # Load Excel
    try:
        if config.sheet_name:
            df = pd.read_excel(config.input_file, sheet_name=config.sheet_name)
        else:
            df = pd.read_excel(config.input_file)
    except Exception as e:
        raise RuntimeError(f"Failed to load Excel file: {e}")
    
    logger.info(f"Loaded {len(df)} rows")
    
    # Validate columns
    required_cols = [
        "Occurrence No", "Description", "Probable Cause",
        "Flight Safety Events", "Flight Safety Factors"
    ]
    
    # Try alternate column names
    alternate_cols = {
        "Occurrence No": ["Occurrence", "Occ No", "ID"],
    }
    
    missing = []
    for col in required_cols:
        if col not in df.columns:
            # Try alternates
            found = False
            for alt in alternate_cols.get(col, []):
                if alt in df.columns:
                    logger.info(f"Using '{alt}' as '{col}'")
                    df[col] = df[alt]
                    found = True
                    break
            
            if not found:
                missing.append(col)
    
    if missing:
        logger.warning(f"Missing columns: {missing}")
        logger.warning("Available columns: " + ", ".join(df.columns))
        raise ValueError(f"Required columns not found: {missing}")
    
    # Check for empty occurrence numbers
    empty_occ = df["Occurrence No"].isna().sum()
    if empty_occ > 0:
        logger.warning(f"{empty_occ} rows have empty Occurrence No")
        df = df[df["Occurrence No"].notna()]
        logger.info(f"Filtered to {len(df)} rows")
    
    return df


def run_pipeline(config: PipelineConfig):
    """
    Main pipeline execution.
    
    Phases:
    1. Analyze individual events
    2. Generate root cause categories
    3. Classify events into categories
    """
    
    # Setup
    ensure_dir(Path(config.output_dir))
    logger = setup_logging(config)
    metrics = PipelineMetrics()
    
    logger.info("="*60)
    logger.info("SPEED SPI Aviation Safety Root Cause Analyzer")
    logger.info("="*60)
    logger.info(f"Input: {config.input_file}")
    logger.info(f"Output: {config.output_dir}")
    logger.info(f"Model: {config.model}")
    logger.info(f"Endpoint: {config.endpoint}")
    logger.info(f"Max Workers: {config.max_workers}")
    logger.info(f"Reasoning Effort: {config.reasoning_effort}")
    logger.info(f"Phase1 max_tokens: {config.phase1_max_tokens}")
    logger.info(f"Categories max_tokens: {config.categories_response_tokens}")
    logger.info(f"Phase3 max_tokens: {config.phase3_max_tokens}")
    logger.info(f"Dry Run: {config.dry_run}")
    logger.info("="*60)

    try:
        # Load data
        df = load_and_validate_data(config, logger)
        
        # Initialize client
        client = LMStudioClient(config, metrics, logger)

        # Validate configured token limits and warn if needed
        validate_token_limits(config, logger)
        
        # Phase 1: Analyze events
        if not config.skip_phase1:
            phase1_results = phase1_analyze(df, client, config, metrics, logger)
        else:
            logger.info("Skipping Phase 1 (loading cached results)")
            phase1_path = Path(config.output_dir) / "phase1_results.jsonl"
            phase1_results = read_jsonl(phase1_path)
        
        # Phase 2: Generate categories
        if not config.skip_phase2:
            categories = phase2_categories(
                client, phase1_results, config, metrics, logger
            )
        else:
            logger.info("Skipping Phase 2 (loading cached categories)")
            categories_path = Path(config.output_dir) / "root_cause_categories.json"
            categories = read_json(categories_path)
            
            if not categories:
                raise RuntimeError("No cached categories found")
        
        # Phase 3: Classify events
        if not config.skip_phase3:
            analyzed_df, mapping_df = phase3_classify(
                df, categories, client, config, metrics, logger
            )
            
            # Save results
            output_path = Path(config.output_dir) / "SPEED_SPI_analyzed.xlsx"
            analyzed_df.to_excel(output_path, index=False)
            logger.info(f"Saved analyzed data to {output_path}")
        else:
            logger.info("Skipping Phase 3")
        
        # Save metrics
        metrics_path = Path(config.output_dir) / "metrics.json"
        write_json(metrics_path, metrics.to_dict())
        
        # Print summary
        print("\n" + metrics.summary())
        
        logger.info("Pipeline completed successfully!")
        
    except Exception as e:
        logger.error(f"Pipeline failed: {e}", exc_info=True)
        raise


# ============================================================================
# CLI
# ============================================================================

def parse_args() -> PipelineConfig:
    """Parse command line arguments."""
    
    parser = argparse.ArgumentParser(
        description="SPEED SPI Aviation Safety Root Cause Analyzer",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic usage
  python speed_spi_analyzer_v2.py --input data.xlsx
  
  # Custom endpoint and parallelization
  python speed_spi_analyzer_v2.py --input data.xlsx \\
      --endpoint http://localhost:5000/v1/chat/completions \\
      --max-workers 8
  
  # Dry run to test without API calls
  python speed_spi_analyzer_v2.py --input data.xlsx --dry-run
  
  # Skip phases (use cached results)
  python speed_spi_analyzer_v2.py --input data.xlsx \\
      --skip-phase1 --skip-phase2
  
  # Force reprocess everything
  python speed_spi_analyzer_v2.py --input data.xlsx --force-reprocess
        """
    )
    
    # Input/Output
    parser.add_argument(
        "--input",
        default="SPEED_SPI.xlsx",
        help="Input Excel file path (default: SPEED_SPI.xlsx)"
    )
    parser.add_argument(
        "--sheet",
        dest="sheet_name",
        default=None,
        help="Excel sheet name (default: first sheet)"
    )
    parser.add_argument(
        "--output-dir",
        dest="output_dir",
        default="outputs",
        help="Output directory (default: outputs)"
    )
    
    # LM Studio
    parser.add_argument(
        "--endpoint",
        default="http://localhost:1234/v1/chat/completions",
        help="LM Studio endpoint URL"
    )
    parser.add_argument(
        "--model",
        default="lmstudio-local",
        help="Model name"
    )
    
    # Performance
    parser.add_argument(
        "--max-workers",
        type=int,
        default=4,
        help="Maximum parallel workers (default: 4)"
    )
    parser.add_argument(
        "--rpm",
        dest="requests_per_minute",
        type=int,
        default=60,
        help="Requests per minute rate limit (default: 60)"
    )
    parser.add_argument(
        "--timeout",
        dest="request_timeout_seconds",
        type=int,
        default=60,
        help="Request timeout in seconds (default: 60)"
    )
    parser.add_argument(
        "--retries",
        dest="max_retries",
        type=int,
        default=3,
        help="Maximum retry attempts (default: 3)"
    )
    
    # Processing options
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Test pipeline without making API calls"
    )
    parser.add_argument(
        "--force-reprocess",
        action="store_true",
        help="Ignore cached results and reprocess everything"
    )
    parser.add_argument(
        "--skip-phase1",
        action="store_true",
        help="Skip Phase 1 (use cached event analyses)"
    )
    parser.add_argument(
        "--skip-phase2",
        action="store_true",
        help="Skip Phase 2 (use cached categories)"
    )
    parser.add_argument(
        "--skip-phase3",
        action="store_true",
        help="Skip Phase 3 (skip classification)"
    )
    
    # Categories
    parser.add_argument(
        "--min-categories",
        type=int,
        default=10,
        help="Minimum number of categories (default: 10)"
    )
    parser.add_argument(
        "--max-categories",
        type=int,
        default=20,
        help="Maximum number of categories (default: 20)"
    )

    # Logging
    parser.add_argument(
        "--log-level",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        default="INFO",
        help="Logging level (default: INFO)"
    )
    parser.add_argument(
        "--no-log-api",
        dest="log_api_calls",
        action="store_false",
        help="Disable API call logging"
    )

    # Reasoning effort control (for reasoning-capable models)
    parser.add_argument(
        "--reasoning-effort",
        choices=["none", "low", "medium", "high"],
        default="medium",
        help="Reasoning effort hint to the server (default: medium)"
    )

    # Token budgets (advanced)
    parser.add_argument(
        "--phase1-max-tokens",
        dest="phase1_max_tokens",
        type=int,
        default=-1,
        help="Max tokens for Phase 1 responses (-1 for unlimited; default: -1)"
    )
    parser.add_argument(
        "--categories-tokens",
        dest="categories_response_tokens",
        type=int,
        default=-1,
        help="Max tokens for category generation responses (-1 for unlimited; default: -1)"
    )
    parser.add_argument(
        "--phase3-max-tokens",
        dest="phase3_max_tokens",
        type=int,
        default=256,
        help="Max tokens for Phase 3 classification responses (default: 256)"
    )
    
    args = parser.parse_args()
    
    # Convert to PipelineConfig
    config = PipelineConfig(
        input_file=args.input,
        sheet_name=args.sheet_name,
        output_dir=args.output_dir,
        endpoint=args.endpoint,
        model=args.model,
        requests_per_minute=args.requests_per_minute,
        max_workers=args.max_workers,
        request_timeout_seconds=args.request_timeout_seconds,
        max_retries=args.max_retries,
        dry_run=args.dry_run,
        skip_phase1=args.skip_phase1,
        skip_phase2=args.skip_phase2,
        skip_phase3=args.skip_phase3,
        force_reprocess=args.force_reprocess,
        min_categories=args.min_categories,
        max_categories=args.max_categories,
        log_level=args.log_level,
        log_api_calls=args.log_api_calls,
        reasoning_effort=args.reasoning_effort,
        # Override token budgets when provided
        categories_response_tokens=args.categories_response_tokens,
        phase1_max_tokens=args.phase1_max_tokens,
        phase3_max_tokens=args.phase3_max_tokens,
    )
    
    return config


if __name__ == "__main__":
    config = parse_args()
    run_pipeline(config)
