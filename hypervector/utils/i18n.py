"""Internationalization and localization support for HyperVector Lab."""

import json
import os
from typing import Dict, Optional, Any
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


class I18nManager:
    """Internationalization manager for multi-language support."""
    
    def __init__(self, default_language: str = 'en', locale_dir: Optional[str] = None):
        """Initialize i18n manager.
        
        Args:
            default_language: Default language code (ISO 639-1)
            locale_dir: Directory containing translation files
        """
        self.default_language = default_language
        self.current_language = default_language
        
        # Set up locale directory
        if locale_dir:
            self.locale_dir = Path(locale_dir)
        else:
            self.locale_dir = Path(__file__).parent.parent / 'locales'
        
        # Create locales directory if it doesn't exist
        self.locale_dir.mkdir(exist_ok=True)
        
        # Translation cache
        self.translations: Dict[str, Dict[str, str]] = {}
        
        # Supported languages
        self.supported_languages = {
            'en': 'English',
            'es': 'Español', 
            'fr': 'Français',
            'de': 'Deutsch',
            'ja': '日本語',
            'zh': '中文',
            'ko': '한국어',
            'pt': 'Português',
            'ru': 'Русский',
            'it': 'Italiano'
        }
        
        # Load default translations
        self._load_translations()
        
        logger.info(f"I18n initialized with language: {self.current_language}")
    
    def set_language(self, language_code: str) -> bool:
        """Set current language.
        
        Args:
            language_code: ISO 639-1 language code
            
        Returns:
            True if language was set successfully, False otherwise
        """
        if language_code not in self.supported_languages:
            logger.warning(f"Unsupported language: {language_code}")
            return False
        
        self.current_language = language_code
        self._load_translations()
        
        logger.info(f"Language changed to: {language_code}")
        return True
    
    def get_supported_languages(self) -> Dict[str, str]:
        """Get list of supported languages.
        
        Returns:
            Dictionary of language codes to language names
        """
        return self.supported_languages.copy()
    
    def translate(self, key: str, **kwargs) -> str:
        """Translate a text key to current language.
        
        Args:
            key: Translation key
            **kwargs: Variables for string formatting
            
        Returns:
            Translated text
        """
        # Get translation for current language
        if self.current_language in self.translations:
            text = self.translations[self.current_language].get(key)
            if text:
                try:
                    return text.format(**kwargs)
                except KeyError as e:
                    logger.warning(f"Missing variable in translation: {e}")
                    return text
        
        # Fallback to default language
        if self.default_language in self.translations:
            text = self.translations[self.default_language].get(key)
            if text:
                try:
                    return text.format(**kwargs)
                except KeyError:
                    return text
        
        # Ultimate fallback: return the key itself
        logger.warning(f"Missing translation for key: {key}")
        return key
    
    def _load_translations(self):
        """Load translation files for current language."""
        for lang_code in [self.current_language, self.default_language]:
            if lang_code not in self.translations:
                translation_file = self.locale_dir / f"{lang_code}.json"
                
                if translation_file.exists():
                    try:
                        with open(translation_file, 'r', encoding='utf-8') as f:
                            self.translations[lang_code] = json.load(f)
                        logger.debug(f"Loaded translations for {lang_code}")
                    except Exception as e:
                        logger.error(f"Failed to load translations for {lang_code}: {e}")
                else:
                    # Create default translation file
                    self._create_default_translations(lang_code)
    
    def _create_default_translations(self, language_code: str):
        """Create default translation file for a language."""
        default_translations = self._get_default_translations(language_code)
        
        translation_file = self.locale_dir / f"{language_code}.json"
        try:
            with open(translation_file, 'w', encoding='utf-8') as f:
                json.dump(default_translations, f, indent=2, ensure_ascii=False)
            
            self.translations[language_code] = default_translations
            logger.info(f"Created default translations for {language_code}")
            
        except Exception as e:
            logger.error(f"Failed to create translations for {language_code}: {e}")
            # Use in-memory defaults
            self.translations[language_code] = default_translations
    
    def _get_default_translations(self, language_code: str) -> Dict[str, str]:
        """Get default translations for a language."""
        translations = {
            'en': {
                # System messages
                'system.initialized': 'HDC System initialized successfully',
                'system.error': 'System error: {error}',
                'system.warning': 'Warning: {message}',
                
                # Operations
                'operation.encode_text': 'Encoding text to hypervector',
                'operation.encode_image': 'Encoding image to hypervector', 
                'operation.encode_eeg': 'Encoding EEG signal to hypervector',
                'operation.bind': 'Binding hypervectors',
                'operation.bundle': 'Bundling hypervectors',
                'operation.similarity': 'Computing similarity',
                
                # Validation messages
                'validation.invalid_input': 'Invalid input: {details}',
                'validation.dimension_error': 'Invalid dimension: {dim}',
                'validation.device_error': 'Device error: {device}',
                
                # Performance messages
                'performance.slow_operation': 'Slow operation detected: {operation} took {time}ms',
                'performance.memory_warning': 'High memory usage: {memory}MB',
                'performance.optimization_suggestion': 'Optimization suggestion: {suggestion}',
                
                # Security messages
                'security.input_sanitized': 'Input has been sanitized',
                'security.dangerous_pattern': 'Dangerous pattern detected: {pattern}',
                'security.rate_limit_exceeded': 'Rate limit exceeded',
                
                # Research messages
                'research.experiment_started': 'Research experiment started: {experiment}',
                'research.results_available': 'Research results available',
                'research.benchmark_complete': 'Benchmark completed: {benchmark}'
            },
            'es': {
                # Spanish translations
                'system.initialized': 'Sistema HDC inicializado correctamente',
                'system.error': 'Error del sistema: {error}',
                'system.warning': 'Advertencia: {message}',
                
                'operation.encode_text': 'Codificando texto a hipervector',
                'operation.encode_image': 'Codificando imagen a hipervector',
                'operation.encode_eeg': 'Codificando señal EEG a hipervector',
                'operation.bind': 'Vinculando hipervectores',
                'operation.bundle': 'Agrupando hipervectores',
                'operation.similarity': 'Calculando similitud',
                
                'validation.invalid_input': 'Entrada inválida: {details}',
                'validation.dimension_error': 'Dimensión inválida: {dim}',
                'validation.device_error': 'Error de dispositivo: {device}',
                
                'performance.slow_operation': 'Operación lenta detectada: {operation} tardó {time}ms',
                'performance.memory_warning': 'Uso alto de memoria: {memory}MB',
                'performance.optimization_suggestion': 'Sugerencia de optimización: {suggestion}',
                
                'security.input_sanitized': 'La entrada ha sido sanitizada',
                'security.dangerous_pattern': 'Patrón peligroso detectado: {pattern}',
                'security.rate_limit_exceeded': 'Límite de velocidad excedido',
                
                'research.experiment_started': 'Experimento de investigación iniciado: {experiment}',
                'research.results_available': 'Resultados de investigación disponibles',
                'research.benchmark_complete': 'Benchmark completado: {benchmark}'
            },
            'fr': {
                # French translations
                'system.initialized': 'Système HDC initialisé avec succès',
                'system.error': 'Erreur système: {error}',
                'system.warning': 'Avertissement: {message}',
                
                'operation.encode_text': 'Encodage du texte en hypervecteur',
                'operation.encode_image': 'Encodage de l\'image en hypervecteur',
                'operation.encode_eeg': 'Encodage du signal EEG en hypervecteur',
                'operation.bind': 'Liaison des hypervecteurs',
                'operation.bundle': 'Regroupement des hypervecteurs',
                'operation.similarity': 'Calcul de similarité',
                
                'validation.invalid_input': 'Entrée invalide: {details}',
                'validation.dimension_error': 'Dimension invalide: {dim}',
                'validation.device_error': 'Erreur de périphérique: {device}',
                
                'performance.slow_operation': 'Opération lente détectée: {operation} a pris {time}ms',
                'performance.memory_warning': 'Utilisation mémoire élevée: {memory}MB',
                'performance.optimization_suggestion': 'Suggestion d\'optimisation: {suggestion}',
                
                'security.input_sanitized': 'L\'entrée a été nettoyée',
                'security.dangerous_pattern': 'Motif dangereux détecté: {pattern}',
                'security.rate_limit_exceeded': 'Limite de débit dépassée',
                
                'research.experiment_started': 'Expérience de recherche commencée: {experiment}',
                'research.results_available': 'Résultats de recherche disponibles',
                'research.benchmark_complete': 'Benchmark terminé: {benchmark}'
            },
            'de': {
                # German translations
                'system.initialized': 'HDC-System erfolgreich initialisiert',
                'system.error': 'Systemfehler: {error}',
                'system.warning': 'Warnung: {message}',
                
                'operation.encode_text': 'Text wird zu Hypervektor kodiert',
                'operation.encode_image': 'Bild wird zu Hypervektor kodiert',
                'operation.encode_eeg': 'EEG-Signal wird zu Hypervektor kodiert',
                'operation.bind': 'Hypervektoren werden gebunden',
                'operation.bundle': 'Hypervektoren werden gebündelt',
                'operation.similarity': 'Ähnlichkeit wird berechnet',
                
                'validation.invalid_input': 'Ungültige Eingabe: {details}',
                'validation.dimension_error': 'Ungültige Dimension: {dim}',
                'validation.device_error': 'Gerätfehler: {device}',
                
                'performance.slow_operation': 'Langsame Operation erkannt: {operation} dauerte {time}ms',
                'performance.memory_warning': 'Hoher Speicherverbrauch: {memory}MB',
                'performance.optimization_suggestion': 'Optimierungsvorschlag: {suggestion}',
                
                'security.input_sanitized': 'Eingabe wurde bereinigt',
                'security.dangerous_pattern': 'Gefährliches Muster erkannt: {pattern}',
                'security.rate_limit_exceeded': 'Ratenlimit überschritten',
                
                'research.experiment_started': 'Forschungsexperiment gestartet: {experiment}',
                'research.results_available': 'Forschungsergebnisse verfügbar',
                'research.benchmark_complete': 'Benchmark abgeschlossen: {benchmark}'
            },
            'ja': {
                # Japanese translations
                'system.initialized': 'HDCシステムが正常に初期化されました',
                'system.error': 'システムエラー: {error}',
                'system.warning': '警告: {message}',
                
                'operation.encode_text': 'テキストをハイパーベクターにエンコード中',
                'operation.encode_image': '画像をハイパーベクターにエンコード中',
                'operation.encode_eeg': 'EEG信号をハイパーベクターにエンコード中',
                'operation.bind': 'ハイパーベクターを結合中',
                'operation.bundle': 'ハイパーベクターを束ね中',
                'operation.similarity': '類似度を計算中',
                
                'validation.invalid_input': '無効な入力: {details}',
                'validation.dimension_error': '無効な次元: {dim}',
                'validation.device_error': 'デバイスエラー: {device}',
                
                'performance.slow_operation': '遅い操作が検出されました: {operation}は{time}msかかりました',
                'performance.memory_warning': 'メモリ使用量が高い: {memory}MB',
                'performance.optimization_suggestion': '最適化の提案: {suggestion}',
                
                'security.input_sanitized': '入力がサニタイズされました',
                'security.dangerous_pattern': '危険なパターンが検出されました: {pattern}',
                'security.rate_limit_exceeded': 'レート制限を超過しました',
                
                'research.experiment_started': '研究実験が開始されました: {experiment}',
                'research.results_available': '研究結果が利用可能です',
                'research.benchmark_complete': 'ベンチマークが完了しました: {benchmark}'
            },
            'zh': {
                # Chinese translations
                'system.initialized': 'HDC系统初始化成功',
                'system.error': '系统错误: {error}',
                'system.warning': '警告: {message}',
                
                'operation.encode_text': '正在将文本编码为超向量',
                'operation.encode_image': '正在将图像编码为超向量',
                'operation.encode_eeg': '正在将EEG信号编码为超向量',
                'operation.bind': '正在绑定超向量',
                'operation.bundle': '正在捆绑超向量',
                'operation.similarity': '正在计算相似性',
                
                'validation.invalid_input': '无效输入: {details}',
                'validation.dimension_error': '无效维度: {dim}',
                'validation.device_error': '设备错误: {device}',
                
                'performance.slow_operation': '检测到慢操作: {operation}花费了{time}毫秒',
                'performance.memory_warning': '内存使用量高: {memory}MB',
                'performance.optimization_suggestion': '优化建议: {suggestion}',
                
                'security.input_sanitized': '输入已被清理',
                'security.dangerous_pattern': '检测到危险模式: {pattern}',
                'security.rate_limit_exceeded': '超过速率限制',
                
                'research.experiment_started': '研究实验已开始: {experiment}',
                'research.results_available': '研究结果可用',
                'research.benchmark_complete': '基准测试完成: {benchmark}'
            }
        }
        
        return translations.get(language_code, translations['en'])
    
    def add_translation(self, key: str, text: str, language_code: Optional[str] = None):
        """Add or update a translation.
        
        Args:
            key: Translation key
            text: Translated text
            language_code: Language code (uses current language if None)
        """
        if language_code is None:
            language_code = self.current_language
        
        if language_code not in self.translations:
            self.translations[language_code] = {}
        
        self.translations[language_code][key] = text
        
        # Save to file
        self._save_translations(language_code)
    
    def _save_translations(self, language_code: str):
        """Save translations to file."""
        translation_file = self.locale_dir / f"{language_code}.json"
        
        try:
            with open(translation_file, 'w', encoding='utf-8') as f:
                json.dump(self.translations[language_code], f, indent=2, ensure_ascii=False)
        except Exception as e:
            logger.error(f"Failed to save translations for {language_code}: {e}")


# Global i18n manager instance
_i18n_manager = I18nManager()

def set_language(language_code: str) -> bool:
    """Set the global language."""
    return _i18n_manager.set_language(language_code)

def get_supported_languages() -> Dict[str, str]:
    """Get supported languages."""
    return _i18n_manager.get_supported_languages()

def translate(key: str, **kwargs) -> str:
    """Translate a key to the current language."""
    return _i18n_manager.translate(key, **kwargs)

def t(key: str, **kwargs) -> str:
    """Short alias for translate function."""
    return translate(key, **kwargs)

def add_translation(key: str, text: str, language_code: Optional[str] = None):
    """Add a translation."""
    _i18n_manager.add_translation(key, text, language_code)

def get_current_language() -> str:
    """Get current language code."""
    return _i18n_manager.current_language


# Localized logger
class LocalizedLogger:
    """Logger that outputs localized messages."""
    
    def __init__(self, name: str):
        self.logger = logging.getLogger(name)
    
    def info(self, key: str, **kwargs):
        """Log localized info message."""
        message = translate(key, **kwargs)
        self.logger.info(message)
    
    def warning(self, key: str, **kwargs):
        """Log localized warning message."""
        message = translate(key, **kwargs)
        self.logger.warning(message)
    
    def error(self, key: str, **kwargs):
        """Log localized error message.""" 
        message = translate(key, **kwargs)
        self.logger.error(message)
    
    def debug(self, key: str, **kwargs):
        """Log localized debug message."""
        message = translate(key, **kwargs)
        self.logger.debug(message)

def get_localized_logger(name: str) -> LocalizedLogger:
    """Get a localized logger."""
    return LocalizedLogger(name)