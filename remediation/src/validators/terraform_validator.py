import hcl2
import json
from typing import List, Dict, Optional, Set
from dataclasses import dataclass
import logging
import subprocess
import os
from pathlib import Path
from datetime import datetime

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class ValidationResult:
    """検証結果"""
    is_valid: bool
    errors: List[str]
    warnings: List[str]
    suggestions: List[str]
    resource_count: int
    security_score: int

class TerraformValidator:
    def __init__(self, 
                 tflint_config: Optional[str] = None,
                 checkov_config: Optional[str] = None):
        """
        Terraform Validator

        Args:
            tflint_config (Optional[str]): tflintの設定ファイルパス
            checkov_config (Optional[str]): checkovの設定ファイルパス
        """
        self.tflint_config = tflint_config
        self.checkov_config = checkov_config
        self.best_practices = self._load_best_practices()
        self.security_rules = self._load_security_rules()

    def _load_best_practices(self) -> Dict:
        """ベストプラクティスの読み込み"""
        return {
            "naming_convention": {
                "resource": r"^[a-z][a-z0-9_]*$",
                "variable": r"^[a-z][a-z0-9_]*$",
                "output": r"^[a-z][a-z0-9_]*$"
            },
            "required_tags": [
                "Environment",
                "Owner",
                "Project"
            ],
            "deprecated_resources": [
                "aws_db_security_group",
                "aws_elasticache_security_group"
            ],
            "max_resource_count": 50,
            "required_providers": [
                "aws",
                "azurerm",
                "google"
            ]
        }

    def _load_security_rules(self) -> Dict:
        """セキュリティルールの読み込み"""
        return {
            "aws_security_group": {
                "no_public_ingress": {
                    "description": "No public ingress (0.0.0.0/0)",
                    "severity": "HIGH"
                }
            },
            "aws_s3_bucket": {
                "encryption": {
                    "description": "Enable server-side encryption",
                    "severity": "HIGH"
                },
                "versioning": {
                    "description": "Enable versioning",
                    "severity": "MEDIUM"
                }
            },
            "aws_rds_instance": {
                "encryption": {
                    "description": "Enable storage encryption",
                    "severity": "HIGH"
                },
                "backup": {
                    "description": "Enable automated backups",
                    "severity": "MEDIUM"
                }
            }
        }

    def validate_terraform(self, terraform_dir: str) -> ValidationResult:
        """
        Terraformコードの検証を実行

        Args:
            terraform_dir (str): Terraformコードのディレクトリ

        Returns:
            ValidationResult: 検証結果
        """
        try:
            errors = []
            warnings = []
            suggestions = []
            resource_count = 0
            security_score = 100

            # HCL構文チェック
            hcl_errors = self._validate_hcl_syntax(terraform_dir)
            errors.extend(hcl_errors)

            # terraform fmt チェック
            fmt_errors = self._check_formatting(terraform_dir)
            warnings.extend(fmt_errors)

            # リソースの検証
            resource_errors, count = self._validate_resources(terraform_dir)
            errors.extend(resource_errors)
            resource_count = count

            # セキュリティチェック
            security_issues, score = self._check_security(terraform_dir)
            errors.extend(security_issues)
            security_score = score

            # ベストプラクティスチェック
            practice_warnings = self._check_best_practices(terraform_dir)
            warnings.extend(practice_warnings)

            # tflintの実行
            tflint_warnings = self._run_tflint(terraform_dir)
            warnings.extend(tflint_warnings)

            # checkovの実行
            checkov_issues = self._run_checkov(terraform_dir)
            warnings.extend(checkov_issues)

            return ValidationResult(
                is_valid=len(errors) == 0,
                errors=errors,
                warnings=warnings,
                suggestions=suggestions,
                resource_count=resource_count,
                security_score=security_score
            )

        except Exception as e:
            logger.error(f"Error validating Terraform code: {e}")
            return ValidationResult(
                is_valid=False,
                errors=[str(e)],
                warnings=[],
                suggestions=[],
                resource_count=0,
                security_score=0
            )

    def _validate_hcl_syntax(self, terraform_dir: str) -> List[str]:
        """HCL構文の検証"""
        errors = []
        
        try:
            for tf_file in Path(terraform_dir).glob("*.tf"):
                with open(tf_file, 'r') as f:
                    try:
                        hcl2.load(f)
                    except Exception as e:
                        errors.append(f"Syntax error in {tf_file}: {str(e)}")

            # terraform validate の実行
            result = subprocess.run(
                ['terraform', 'validate'],
                cwd=terraform_dir,
                capture_output=True,
                text=True
            )
            
            if result.returncode != 0:
                errors.append(f"Terraform validation failed: {result.stderr}")

        except Exception as e:
            errors.append(f"Error validating HCL syntax: {str(e)}")

        return errors

    def _check_formatting(self, terraform_dir: str) -> List[str]:
        """フォーマットチェック"""
        warnings = []
        
        try:
            result = subprocess.run(
                ['terraform', 'fmt', '-check', '-diff'],
                cwd=terraform_dir,
                capture_output=True,
                text=True
            )
            
            if result.returncode != 0:
                warnings.append(
                    "Terraform files are not properly formatted. "
                    "Run 'terraform fmt' to fix formatting."
                )
                warnings.extend(result.stdout.splitlines())

        except Exception as e:
            warnings.append(f"Error checking formatting: {str(e)}")

        return warnings

    def _validate_resources(self, terraform_dir: str) -> Tuple[List[str], int]:
        """リソースの検証"""
        errors = []
        resource_count = 0
        
        try:
            for tf_file in Path(terraform_dir).glob("*.tf"):
                with open(tf_file, 'r') as f:
                    try:
                        tf_data = hcl2.load(f)
                        
                        # リソースのチェック
                        if 'resource' in tf_data:
                            for resource_type, resources in tf_data['resource'].items():
                                resource_count += len(resources)
                                
                                # 非推奨リソースのチェック
                                if resource_type in self.best_practices['deprecated_resources']:
                                    errors.append(
                                        f"Deprecated resource type used: {resource_type}"
                                    )
                                
                                # 命名規則のチェック
                                for resource_name in resources.keys():
                                    if not re.match(
                                        self.best_practices['naming_convention']['resource'],
                                        resource_name
                                    ):
                                        errors.append(
                                            f"Resource name '{resource_name}' does not follow "
                                            "naming convention"
                                        )

                    except Exception as e:
                        errors.append(f"Error parsing {tf_file}: {str(e)}")

        except Exception as e:
            errors.append(f"Error validating resources: {str(e)}")

        return errors, resource_count

    def _check_security(self, terraform_dir: str) -> Tuple[List[str], int]:
        """セキュリティチェック"""
        issues = []
        security_score = 100
        penalty_per_issue = 10
        
        try:
            for tf_file in Path(terraform_dir).glob("*.tf"):
                with open(tf_file, 'r') as f:
                    tf_data = hcl2.load(f)
                    
                    if 'resource' in tf_data:
                        for resource_type, resources in tf_data['resource'].items():
                            if resource_type in self.security_rules:
                                rules = self.security_rules[resource_type]
                                
                                for resource_name, config in resources.items():
                                    for rule_name, rule in rules.items():
                                        if self._check_security_rule(
                                            resource_type,
                                            rule_name,
                                            config
                                        ):
                                            issue = (
                                                f"Security issue in {resource_type}.{resource_name}: "
                                                f"{rule['description']}"
                                            )
                                            issues.append(issue)
                                            
                                            if rule['severity'] == 'HIGH':
                                                security_score -= penalty_per_issue
                                            elif rule['severity'] == 'MEDIUM':
                                                security_score -= penalty_per_issue / 2

        except Exception as e:
            issues.append(f"Error checking security: {str(e)}")
            security_score = 0

        return issues, max(0, security_score)

    def _check_security_rule(self,
                           resource_type: str,
                           rule_name: str,
                           config: Dict) -> bool:
        """個別のセキュリティルールチェック"""
        if resource_type == 'aws_security_group' and rule_name == 'no_public_ingress':
            if 'ingress' in config:
                for rule in config['ingress']:
                    if '0.0.0.0/0' in str(rule.get('cidr_blocks', [])):
                        return True

        elif resource_type == 'aws_s3_bucket':
            if rule_name == 'encryption':
                if 'server_side_encryption_configuration' not in config:
                    return True
            elif rule_name == 'versioning':
                if 'versioning' not in config or not config['versioning'].get('enabled', False):
                    return True

        elif resource_type == 'aws_rds_instance':
            if rule_name == 'encryption':
                if not config.get('storage_encrypted', False):
                    return True
            elif rule_name == 'backup':
                if not config.get('backup_retention_period', 0):
                    return True

        return False

    def _check_best_practices(self, terraform_dir: str) -> List[str]:
        """ベストプラクティスのチェック"""
        warnings = []
        
        try:
            for tf_file in Path(terraform_dir).glob("*.tf"):
                with open(tf_file, 'r') as f:
                    tf_data = hcl2.load(f)
                    
                    # プロバイダーのチェック
                    if 'terraform' in tf_data:
                        providers = tf_data['terraform'].get('required_providers', {})
                        for required in self.best_practices['required_providers']:
                            if required not in providers:
                                warnings.append(
                                    f"Required provider '{required}' is not specified"
                                )

                    # タグのチェック
                    if 'resource' in tf_data:
                        for resources in tf_data['resource'].values():
                            for resource in resources.values():
                                if 'tags' in resource:
                                    missing_tags = [
                                        tag for tag in self.best_practices['required_tags']
                                        if tag not in resource['tags']
                                    ]
                                    if missing_tags:
                                        warnings.append(
                                            f"Missing required tags: {', '.join(missing_tags)}"
                                        )

        except Exception as e:
            warnings.append(f"Error checking best practices: {str(e)}")

        return warnings

    def _run_tflint(self, terraform_dir: str) -> List[str]:
        """tflintの実行"""
        warnings = []
        
        try:
            cmd = ['tflint']
            if self.tflint_config:
                cmd.extend(['--config', self.tflint_config])
            
            result = subprocess.run(
                cmd,
                cwd=terraform_dir,
                capture_output=True,
                text=True
            )
            
            if result.returncode != 0:
                warnings.extend(result.stdout.splitlines())

        except Exception as e:
            warnings.append(f"Error running tflint: {str(e)}")

        return warnings

    def _run_checkov(self, terraform_dir: str) -> List[str]:
        """checkovの実行"""
        warnings = []
        
        try:
            cmd = ['checkov', '--directory', terraform_dir]
            if self.checkov_config:
                cmd.extend(['--config-file', self.checkov_config])
            
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True
            )
            
            if result.returncode != 0:
                warnings.extend(result.stdout.splitlines())

        except Exception as e:
            warnings.append(f"Error running checkov: {str(e)}")

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
                    'resource_count': validation_result.resource_count,
                    'security_score': validation_result.security_score
                },
                'errors': validation_result.errors,
                'warnings': validation_result.warnings,
                'suggestions': validation_result.suggestions,
                'timestamp': str(datetime.now())
            }

            with open(output_path, 'w') as f:
                json.dump(report, f, indent=2)

            logger.info(f"Validation report generated: {output_path}")

        except Exception as e:
            logger.error(f"Error generating report: {e}")
            raise
