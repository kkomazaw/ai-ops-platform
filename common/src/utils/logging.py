import logging
import json
from datetime import datetime
from typing import Dict, Any, Optional
import sys
import os
from pathlib import Path
from logging.handlers import RotatingFileHandler, TimedRotatingFileHandler
import threading
from pythonjsonlogger import jsonlogger
import traceback
from functools import wraps
import time

class CustomJsonFormatter(jsonlogger.JsonFormatter):
    """カスタムJSONフォーマッター"""
    def add_fields(self, log_record: Dict[str, Any], record: logging.LogRecord, message_dict: Dict[str, Any]) -> None:
        """
        ログレコードにカスタムフィールドを追加

        Args:
            log_record (Dict[str, Any]): ログレコード
            record (logging.LogRecord): LogRecordインスタンス
            message_dict (Dict[str, Any]): メッセージ辞書
        """
        super().add_fields(log_record, record, message_dict)

        # 基本フィールドの追加
        log_record['timestamp'] = datetime.utcnow().isoformat()
        log_record['level'] = record.levelname
        log_record['logger'] = record.name

        # スレッド情報の追加
        log_record['thread_id'] = str(threading.current_thread().ident)
        log_record['thread_name'] = threading.current_thread().name

        # 環境情報の追加
        log_record['environment'] = os.getenv('ENVIRONMENT', 'development')
        log_record['service'] = os.getenv('SERVICE_NAME', 'unknown')

        # エラー情報の追加
        if record.exc_info:
            log_record['exception'] = {
                'type': record.exc_info[0].__name__,
                'message': str(record.exc_info[1]),
                'stacktrace': traceback.format_exception(*record.exc_info)
            }

class LoggerFactory:
    """ロガーファクトリー"""
    _instance = None
    _lock = threading.Lock()

    def __new__(cls):
        """シングルトンパターンの実装"""
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        """ロガーファクトリーの初期化"""
        self.default_config = {
            'version': 1,
            'disable_existing_loggers': False,
            'formatters': {
                'json': {
                    '()': CustomJsonFormatter,
                    'format': '%(timestamp)s %(level)s %(name)s %(message)s'
                },
                'standard': {
                    'format': '%(asctime)s [%(levelname)s] %(name)s: %(message)s'
                }
            },
            'handlers': {
                'console': {
                    'class': 'logging.StreamHandler',
                    'formatter': 'standard',
                    'stream': sys.stdout
                },
                'file': {
                    'class': 'logging.handlers.RotatingFileHandler',
                    'formatter': 'json',
                    'filename': 'logs/application.log',
                    'maxBytes': 10485760,  # 10MB
                    'backupCount': 5
                }
            },
            'loggers': {
                '': {  # rootロガー
                    'handlers': ['console', 'file'],
                    'level': 'INFO',
                    'propagate': True
                }
            }
        }

    def get_logger(self, 
                  name: str, 
                  config: Optional[Dict] = None,
                  log_file: Optional[str] = None) -> logging.Logger:
        """
        ロガーの取得または作成

        Args:
            name (str): ロガー名
            config (Optional[Dict]): カスタム設定
            log_file (Optional[str]): ログファイルパス

        Returns:
            logging.Logger: 設定済みロガー
        """
        # 設定の準備
        logger_config = config if config else self.default_config.copy()
        
        # ログディレクトリの作成
        log_dir = Path('logs')
        log_dir.mkdir(exist_ok=True)

        # ログファイルパスの設定
        if log_file:
            logger_config['handlers']['file']['filename'] = str(log_dir / log_file)

        # ロガーの取得と設定
        logger = logging.getLogger(name)
        
        if not logger.handlers:  # ハンドラが未設定の場合のみ設定
            # コンソールハンドラの設定
            console_handler = logging.StreamHandler(sys.stdout)
            console_handler.setFormatter(
                logging.Formatter(
                    logger_config['formatters']['standard']['format']
                )
            )
            logger.addHandler(console_handler)

            # ファイルハンドラの設定
            file_handler = RotatingFileHandler(
                logger_config['handlers']['file']['filename'],
                maxBytes=logger_config['handlers']['file']['maxBytes'],
                backupCount=logger_config['handlers']['file']['backupCount']
            )
            file_handler.setFormatter(CustomJsonFormatter())
            logger.addHandler(file_handler)

            # ログレベルの設定
            logger.setLevel(logger_config['loggers']['']['level'])

        return logger

def log_execution_time(logger: Optional[logging.Logger] = None):
    """
    関数の実行時間を記録するデコレータ

    Args:
        logger (Optional[logging.Logger]): 使用するロガー
    """
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            _logger = logger or logging.getLogger(func.__module__)
            
            start_time = time.time()
            try:
                result = func(*args, **kwargs)
                execution_time = time.time() - start_time
                _logger.info(
                    f"Function '{func.__name__}' executed in {execution_time:.3f} seconds"
                )
                return result
            except Exception as e:
                execution_time = time.time() - start_time
                _logger.error(
                    f"Function '{func.__name__}' failed after {execution_time:.3f} seconds",
                    exc_info=True
                )
                raise
        return wrapper
    return decorator

def log_method_calls(logger: Optional[logging.Logger] = None):
    """
    メソッド呼び出しをログに記録するデコレータ

    Args:
        logger (Optional[logging.Logger]): 使用するロガー
    """
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            _logger = logger or logging.getLogger(func.__module__)
            
            class_name = args[0].__class__.__name__ if args else ''
            method_name = func.__name__
            _logger.debug(
                f"Calling {class_name}.{method_name} with "
                f"args: {args[1:]} kwargs: {kwargs}"
            )
            
            try:
                result = func(*args, **kwargs)
                _logger.debug(
                    f"Finished {class_name}.{method_name}"
                )
                return result
            except Exception as e:
                _logger.error(
                    f"Error in {class_name}.{method_name}: {str(e)}",
                    exc_info=True
                )
                raise
        return wrapper
    return decorator

class ContextLogger:
    """コンテキスト情報付きロガー"""
    def __init__(self, logger: logging.Logger, context: Dict[str, Any]):
        """
        初期化

        Args:
            logger (logging.Logger): 基本ロガー
            context (Dict[str, Any]): コンテキスト情報
        """
        self.logger = logger
        self.context = context

    def _log_with_context(self, level: str, message: str, *args, **kwargs):
        """コンテキスト付きでログを記録"""
        extra = kwargs.get('extra', {})
        extra.update(self.context)
        kwargs['extra'] = extra
        
        log_func = getattr(self.logger, level)
        log_func(message, *args, **kwargs)

    def info(self, message: str, *args, **kwargs):
        """INFO レベルのログを記録"""
        self._log_with_context('info', message, *args, **kwargs)

    def error(self, message: str, *args, **kwargs):
        """ERROR レベルのログを記録"""
        self._log_with_context('error', message, *args, **kwargs)

    def warning(self, message: str, *args, **kwargs):
        """WARNING レベルのログを記録"""
        self._log_with_context('warning', message, *args, **kwargs)

    def debug(self, message: str, *args, **kwargs):
        """DEBUG レベルのログを記録"""
        self._log_with_context('debug', message, *args, **kwargs)

def setup_logger(
    name: str,
    log_file: Optional[str] = None,
    level: int = logging.INFO,
    rotation: str = 'size',
    **kwargs
) -> logging.Logger:
    """
    ロガーのセットアップユーティリティ

    Args:
        name (str): ロガー名
        log_file (Optional[str]): ログファイルパス
        level (int): ログレベル
        rotation (str): ローテーション方式 ('size' or 'time')
        **kwargs: 追加設定

    Returns:
        logging.Logger: 設定済みロガー
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)

    # フォーマッターの作成
    json_formatter = CustomJsonFormatter()
    console_formatter = logging.Formatter(
        '%(asctime)s [%(levelname)s] %(name)s: %(message)s'
    )

    # コンソールハンドラの設定
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(console_formatter)
    logger.addHandler(console_handler)

    # ファイルハンドラの設定
    if log_file:
        log_dir = Path('logs')
        log_dir.mkdir(exist_ok=True)
        file_path = log_dir / log_file

        if rotation == 'size':
            file_handler = RotatingFileHandler(
                file_path,
                maxBytes=kwargs.get('maxBytes', 10485760),  # 10MB
                backupCount=kwargs.get('backupCount', 5)
            )
        else:  # time-based rotation
            file_handler = TimedRotatingFileHandler(
                file_path,
                when=kwargs.get('when', 'midnight'),
                interval=kwargs.get('interval', 1),
                backupCount=kwargs.get('backupCount', 30)
            )

        file_handler.setFormatter(json_formatter)
        logger.addHandler(file_handler)

    return logger

# 使用例
if __name__ == "__main__":
    # ロガーファクトリーの使用例
    factory = LoggerFactory()
    logger = factory.get_logger("example")
    logger.info("This is an example log message")

    # 実行時間ロギングの例
    @log_execution_time()
    def slow_function():
        time.sleep(1)
        return "Done"

    slow_function()

    # メソッド呼び出しロギングの例
    class ExampleClass:
        @log_method_calls()
        def example_method(self, arg1, arg2):
            return f"Processing {arg1} and {arg2}"

    obj = ExampleClass()
    obj.example_method("value1", "value2")

    # コンテキストロガーの使用例
    context = {
        "request_id": "123",
        "user_id": "456"
    }
    context_logger = ContextLogger(logger, context)
    context_logger.info("Processing request")

    # セットアップユーティリティの使用例
    custom_logger = setup_logger(
        "custom_logger",
        log_file="custom.log",
        level=logging.DEBUG,
        rotation='time',
        when='D',
        interval=1,
        backupCount=7
    )
    custom_logger.info("Custom logger initialized")

