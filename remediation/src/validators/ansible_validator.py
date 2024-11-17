import yaml
from typing import List, Dict, Optional, Tuple
import logging
import subprocess
import os
import json
from dataclasses import dataclass
import re
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class ValidationResult:
    """検証結果"""
    is_valid: bool
    errors: List[str]
    warnings: List[str]
    suggestions: List[str]

@dataclass
class PlaybookCheck:
    """プレイブックのチェック項目"""
    name: str
    description: str
    severity: str  # error/warning/info
    check_function: callable

class AnsibleValidator:
    def __init__(self, ansible_lint_config: Optional[str] = None):
        """
        Ansible Validator

        Args:
            ansible_lint_config (Optional[str]): ansible-lintの設定ファイルパス
        """
        self.ansible_lint_config = ansible_lint_config
        self.checks = self._initialize_checks()
        self.best_practices = self._load_best_practices()

    def _initialize_checks(self) -> List[PlaybookCheck]:
        """チェック項目の初期化"""
        return [
            PlaybookCheck(
                name="syntax_check",
                description="Syntax validation of YAML and Ansible structures",
                severity="error",
                check_function=self._check_syntax
            ),
            PlaybookCheck(
                name="idempotency_check",
                description="Check for idempotent tasks",
                severity="warning",
                check_function=self._check_idempotency
            ),
            PlaybookCheck(
                name="security_check",
                description="Security best practices validation",
                severity="error",
                check_function=self._check_security
            ),
            PlaybookCheck(
                name="naming_convention",
                description="Naming convention validation",
                severity="warning",
                check_function=self._check_naming_convention
            ),
            PlaybookCheck(
                name="module_check",
                description="Deprecated module usage check",
                severity="warning",
                check_function=self._check_modules
            )
        ]

    def _load_best_practices(self) -> Dict:
        """ベストプラクティスの読み込み"""
        return {
            "naming_conventions": {
                "playbook": r"^[a-z][a-z0-9_-]+\.ya?ml$",
                "role": r"^[a-z][a-z0-9_-]+$",
                "variable": r"^[a-z][a-z0-9_]+$"
            },
            "deprecated_modules": [
                "command",  # shell or ansible.builtin.commandを推奨
                "raw",     # scriptを推奨
                "git",     # ansible.builtin.gitを推奨
            ],
            "secure_modules": {
                "ansible.builtin.template": ["validate", "mode"],
                "ansible.builtin.copy": ["validate", "mode"],
                "ansible.builtin.file": ["mode"]
            }
        }

    def validate_playbook(self, playbook_path: str) -> ValidationResult:
        """
        プレイブックの検証を実行

        Args:
            playbook_path (str): プレイブックのパス

        Returns:
            ValidationResult: 検証結果
        """
        try:
            errors = []
            warnings = []
            suggestions = []

            # 各チェックを実行
            for check in self.checks:
                check_result = check.check_function(playbook_path)
                if check_result:
                    if check.severity == "error":
                        errors.extend(check_result)
                    elif check.severity == "warning":
                        warnings.extend(check_result)
                    else:
                        suggestions.extend(check_result)

            # ansible-lintの実行
            lint_result = self._run_ansible_lint(playbook_path)
            if lint_result:
                warnings.extend(lint_result)

            return ValidationResult(
                is_valid=len(errors) == 0,
                errors=errors,
                warnings=warnings,
                suggestions=suggestions
            )

        except Exception as e:
            logger.error(f"Error validating playbook: {e}")
            return ValidationResult(
                is_valid=False,
                errors=[str(e)],
                warnings=[],
                suggestions=[]
            )

    def _check_syntax(self, playbook_path: str) -> List[str]:
        """構文チェック"""
        errors = []
        
        try:
            # YAMLの構文チェック
            with open(playbook_path, 'r') as f:
                playbook_content = yaml.safe_load(f)

            # 基本構造のチェック
            if not isinstance(playbook_content, list):
                errors.append("Playbook must be a list of plays")
                return errors

            for play in playbook_content:
                if not isinstance(play, dict):
                    errors.append("Each play must be a dictionary")
                    continue

                # 必須フィールドのチェック
                required_fields = ['hosts']
                for field in required_fields:
                    if field not in play:
                        errors.append(f"Missing required field '{field}' in play")

                # タスクのチェック
                if 'tasks' in play:
                    for task in play['tasks']:
                        if not isinstance(task, dict):
                            errors.append("Each task must be a dictionary")
                            continue
                        if 'name' not in task:
                            errors.append("Each task should have a name")

            # ansible-playbook --syntax-check の実行
            result = subprocess.run(
                ['ansible-playbook', '--syntax-check', playbook_path],
                capture_output=True,
                text=True
            )
            
            if result.returncode != 0:
                errors.append(f"Ansible syntax check failed: {result.stderr}")

        except yaml.YAMLError as e:
            errors.append(f"YAML syntax error: {str(e)}")
        except Exception as e:
            errors.append(f"Error checking syntax: {str(e)}")

        return errors

    def _check_idempotency(self, playbook_path: str) -> List[str]:
        """べき等性チェック"""
        warnings = []
        
        try:
            with open(playbook_path, 'r') as f:
                playbook_content = yaml.safe_load(f)

            non_idempotent_modules = ['command', 'shell', 'raw']
            
            for play in playbook_content:
                if 'tasks' in play:
                    for task in play['tasks']:
                        # コマンド系モジュールのチェック
                        for module in non_idempotent_modules:
                            if module in task:
                                if 'creates' not in task[module] and 'removes' not in task[module]:
                                    warnings.append(
                                        f"Task '{task.get('name', 'unnamed')}' uses {module} "
                                        "without creates/removes - may not be idempotent"
                                    )

                        # shell/commandの代わりにAnsibleモジュールの使用を推奨
                        if 'shell' in task or 'command' in task:
                            warnings.append(
                                f"Consider using Ansible modules instead of shell/command "
                                f"for task '{task.get('name', 'unnamed')}'"
                            )

        except Exception as e:
            warnings.append(f"Error checking idempotency: {str(e)}")

        return warnings

    def _check_security(self, playbook_path: str) -> List[str]:
        """セキュリティチェック"""
        errors = []
        
        try:
            with open(playbook_path, 'r') as f:
                playbook_content = yaml.safe_load(f)

            for play in playbook_content:
                if 'tasks' in play:
                    for task in play['tasks']:
                        # パーミッションのチェック
                        if 'file' in task or 'copy' in task or 'template' in task:
                            module_args = task.get('file', task.get('copy', task.get('template', {})))
                            if 'mode' not in module_args:
                                errors.append(
                                    f"File permissions not specified for task "
                                    f"'{task.get('name', 'unnamed')}'"
                                )

                        # 機密情報の平文使用チェック
                        if 'password' in str(task).lower() or 'secret' in str(task).lower():
                            if 'no_log' not in task or not task['no_log']:
                                errors.append(
                                    f"Task '{task.get('name', 'unnamed')}' may contain sensitive "
                                    "information but no_log is not set"
                                )

                        # sudoの使用チェック
                        if 'become' in task and task.get('become_method', 'sudo') == 'sudo':
                            if 'become_user' not in task:
                                errors.append(
                                    f"Sudo used without specific user in task "
                                    f"'{task.get('name', 'unnamed')}'"
                                )

        except Exception as e:
            errors.append(f"Error checking security: {str(e)}")

        return errors

    def _check_naming_convention(self, playbook_path: str) -> List[str]:
        """命名規則チェック"""
        warnings = []
        
        try:
            # ファイル名のチェック
            playbook_name = os.path.basename(playbook_path)
            if not re.match(self.best_practices['naming_conventions']['playbook'], playbook_name):
                warnings.append(
                    f"Playbook filename '{playbook_name}' does not follow naming convention"
                )

            with open(playbook_path, 'r') as f:
                playbook_content = yaml.safe_load(f)

            for play in playbook_content:
                # 変数名のチェック
                if 'vars' in play:
                    for var_name in play['vars'].keys():
                        if not re.match(self.best_practices['naming_conventions']['variable'], var_name):
                            warnings.append(
                                f"Variable name '{var_name}' does not follow naming convention"
                            )

                # ロール名のチェック
                if 'roles' in play:
                    for role in play['roles']:
                        if isinstance(role, str) and not re.match(
                            self.best_practices['naming_conventions']['role'],
                            role
                        ):
                            warnings.append(
                                f"Role name '{role}' does not follow naming convention"
                            )

        except Exception as e:
            warnings.append(f"Error checking naming conventions: {str(e)}")

        return warnings

    def _check_modules(self, playbook_path: str) -> List[str]:
        """モジュールのチェック"""
        warnings = []
        
        try:
            with open(playbook_path, 'r') as f:
                playbook_content = yaml.safe_load(f)

            for play in playbook_content:
                if 'tasks' in play:
                    for task in play['tasks']:
                        # 非推奨モジュールのチェック
                        for module in self.best_practices['deprecated_modules']:
                            if module in task:
                                warnings.append(
                                    f"Task '{task.get('name', 'unnamed')}' uses deprecated "
                                    f"module '{module}'"
                                )

                        # セキュアなモジュールの設定チェック
                        for module, required_params in self.best_practices['secure_modules'].items():
                            if module in task:
                                for param in required_params:
                                    if param not in task[module]:
                                        warnings.append(
                                            f"Task '{task.get('name', 'unnamed')}' uses {module} "
                                            f"without required parameter '{param}'"
                                        )

        except Exception as e:
            warnings.append(f"Error checking modules: {str(e)}")

        return warnings

    def _run_ansible_lint(self, playbook_path: str) -> List[str]:
        """ansible-lintの実行"""
        warnings = []
        
        try:
            cmd = ['ansible-lint', playbook_path]
            if self.ansible_lint_config:
                cmd.extend(['-c', self.ansible_lint_config])

            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True
            )

            if result.returncode != 0:
                warnings.extend(result.stdout.splitlines())

        except Exception as e:
            warnings.append(f"Error running ansible-lint: {str(e)}")

        return warnings

    def generate_report(self, 
                       validation_result: ValidationResult,
                       output_path: str) -> None:
        """
        検証結果レポートの生成

        Args:
            validation_result (ValidationResult): 検証結果
            output_path (str): 出力パス
        """
        try:
            report = {
                'summary': {
                    'is_valid': validation_result.is_valid,
                    'error_count': len(validation_result.errors),
                    'warning_count': len(validation_result.warnings),
                    'suggestion_count': len(validation_result.suggestions)
                },
                'errors': validation_result.errors,
                'warnings': validation_result.warnings,
                'suggestions': validation_result.suggestions,
                'timestamp': str(datetime.now())
            }

            # JSONレポートの生成
            with open(output_path, 'w') as f:
                json.dump(report, f, indent=2)

            logger.info(f"Validation report generated: {output_path}")

        except Exception as e:
            logger.error(f"Error generating report: {e}")
            raise
