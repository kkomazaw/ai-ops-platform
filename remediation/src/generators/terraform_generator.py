from typing import List, Dict, Optional, Union
from dataclasses import dataclass
import logging
import json
import os
from jinja2 import Environment, FileSystemLoader
import hcl2
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class TerraformResource:
    """Terraformリソースの定義"""
    type: str
    name: str
    config: Dict
    count: Optional[int] = None
    depends_on: Optional[List[str]] = None
    tags: Optional[Dict] = None

@dataclass
class TerraformModule:
    """Terraformモジュールの定義"""
    name: str
    source: str
    variables: Dict
    providers: Optional[Dict] = None
    depends_on: Optional[List[str]] = None

class TerraformGenerator:
    def __init__(self, template_dir: Optional[str] = None):
        """
        Terraform コード生成クラス

        Args:
            template_dir (Optional[str]): テンプレートディレクトリのパス
        """
        self.template_dir = template_dir or "templates"
        self.jinja_env = Environment(
            loader=FileSystemLoader(self.template_dir),
            trim_blocks=True,
            lstrip_blocks=True
        )
        self.resource_templates = self._load_resource_templates()

    def _load_resource_templates(self) -> Dict:
        """リソーステンプレートの読み込み"""
        return {
            "aws_instance": {
                "type": "aws_instance",
                "config": {
                    "ami": "{{ ami_id }}",
                    "instance_type": "{{ instance_type }}",
                    "subnet_id": "{{ subnet_id }}",
                    "vpc_security_group_ids": ["{{ security_group_id }}"],
                    "tags": {
                        "Name": "{{ name }}",
                        "Environment": "{{ environment }}"
                    }
                }
            },
            "aws_autoscaling_group": {
                "type": "aws_autoscaling_group",
                "config": {
                    "name": "{{ name }}",
                    "max_size": "{{ max_size }}",
                    "min_size": "{{ min_size }}",
                    "desired_capacity": "{{ desired_capacity }}",
                    "vpc_zone_identifier": ["{{ subnet_ids }}"],
                    "launch_template": {
                        "id": "{{ launch_template_id }}",
                        "version": "$Latest"
                    }
                }
            },
            "aws_rds_instance": {
                "type": "aws_db_instance",
                "config": {
                    "identifier": "{{ identifier }}",
                    "engine": "{{ engine }}",
                    "engine_version": "{{ engine_version }}",
                    "instance_class": "{{ instance_class }}",
                    "allocated_storage": "{{ allocated_storage }}",
                    "storage_type": "{{ storage_type }}",
                    "multi_az": "{{ multi_az }}"
                }
            }
        }

    def generate_terraform_code(self,
                              infrastructure_data: Dict,
                              output_dir: str) -> str:
        """
        インフラストラクチャ定義からTerraformコードを生成

        Args:
            infrastructure_data (Dict): インフラストラクチャ定義
            output_dir (str): 出力ディレクトリ

        Returns:
            str: 生成されたコードの保存パス
        """
        try:
            # 出力ディレクトリの作成
            os.makedirs(output_dir, exist_ok=True)

            # 各種設定ファイルの生成
            self._generate_provider_config(infrastructure_data, output_dir)
            self._generate_variables(infrastructure_data, output_dir)
            self._generate_main_tf(infrastructure_data, output_dir)
            self._generate_outputs(infrastructure_data, output_dir)

            # バックエンド設定の生成（オプション）
            if 'backend' in infrastructure_data:
                self._generate_backend_config(infrastructure_data['backend'], output_dir)

            logger.info(f"Terraform code generated in {output_dir}")
            return output_dir

        except Exception as e:
            logger.error(f"Error generating Terraform code: {e}")
            raise

    def _generate_provider_config(self,
                                infrastructure_data: Dict,
                                output_dir: str) -> None:
        """プロバイダー設定の生成"""
        provider_config = {
            "terraform": {
                "required_providers": {
                    "aws": {
                        "source": "hashicorp/aws",
                        "version": "~> 4.0"
                    }
                }
            },
            "provider": {
                "aws": {
                    "region": infrastructure_data.get('region', 'us-west-2'),
                    "profile": infrastructure_data.get('profile', 'default')
                }
            }
        }

        with open(os.path.join(output_dir, 'provider.tf'), 'w') as f:
            f.write(self._format_hcl(provider_config))

    def _generate_variables(self,
                          infrastructure_data: Dict,
                          output_dir: str) -> None:
        """変数定義の生成"""
        variables = {}
        
        # 共通変数
        variables["environment"] = {
            "description": "Environment name",
            "type": "string",
            "default": infrastructure_data.get('environment', 'development')
        }

        # リソース固有の変数
        for resource in infrastructure_data.get('resources', []):
            for key, value in resource.get('variables', {}).items():
                variables[key] = {
                    "description": value.get('description', ''),
                    "type": value.get('type', 'string'),
                    "default": value.get('default')
                }

        with open(os.path.join(output_dir, 'variables.tf'), 'w') as f:
            f.write(self._format_hcl({"variable": variables}))

    def _generate_main_tf(self,
                         infrastructure_data: Dict,
                         output_dir: str) -> None:
        """メインのTerraformコード生成"""
        resources = {}
        
        for resource_data in infrastructure_data.get('resources', []):
            resource_type = resource_data['type']
            resource_name = resource_data['name']
            
            if resource_type in self.resource_templates:
                template = self.resource_templates[resource_type]
                config = self._render_resource_config(
                    template['config'],
                    resource_data
                )
                
                resource_key = f"{template['type']}.{resource_name}"
                resources[resource_key] = config

        # モジュールの追加
        modules = {}
        for module_data in infrastructure_data.get('modules', []):
            module_name = module_data['name']
            modules[module_name] = {
                "source": module_data['source'],
                "version": module_data.get('version'),
                **module_data.get('variables', {})
            }

        main_config = {
            "resource": resources,
            "module": modules
        }

        with open(os.path.join(output_dir, 'main.tf'), 'w') as f:
            f.write(self._format_hcl(main_config))

    def _generate_outputs(self,
                         infrastructure_data: Dict,
                         output_dir: str) -> None:
        """出力定義の生成"""
        outputs = {}
        
        for resource in infrastructure_data.get('resources', []):
            resource_type = resource['type']
            resource_name = resource['name']
            
            if 'outputs' in resource:
                for output_name, output_config in resource['outputs'].items():
                    outputs[output_name] = {
                        "value": f"${{{resource_type}.{resource_name}.{output_config['value']}}}",
                        "description": output_config.get('description', ''),
                        "sensitive": output_config.get('sensitive', False)
                    }

        with open(os.path.join(output_dir, 'outputs.tf'), 'w') as f:
            f.write(self._format_hcl({"output": outputs}))

    def _generate_backend_config(self,
                               backend_config: Dict,
                               output_dir: str) -> None:
        """バックエンド設定の生成"""
        config = {
            "terraform": {
                "backend": {
                    backend_config['type']: backend_config['config']
                }
            }
        }

        with open(os.path.join(output_dir, 'backend.tf'), 'w') as f:
            f.write(self._format_hcl(config))

    def _render_resource_config(self,
                              template_config: Dict,
                              resource_data: Dict) -> Dict:
        """リソース設定のレンダリング"""
        config = template_config.copy()
        variables = resource_data.get('variables', {})
        
        # テンプレート変数の置換
        for key, value in config.items():
            if isinstance(value, str) and '{{' in value:
                var_name = value.strip('{{ }}').strip()
                if var_name in variables:
                    config[key] = variables[var_name]

        # 追加の設定をマージ
        if 'additional_config' in resource_data:
            config.update(resource_data['additional_config'])

        return config

    def _format_hcl(self, data: Dict) -> str:
        """HCL形式でデータをフォーマット"""
        return json.dumps(data, indent=2).replace('"', '')

    def validate_terraform_code(self, output_dir: str) -> bool:
        """Terraformコードの検証"""
        try:
            current_dir = os.getcwd()
            os.chdir(output_dir)
            
            # terraform init の実行
            init_result = os.system("terraform init -backend=false")
            if init_result != 0:
                logger.error("Terraform init failed")
                return False

            # terraform validate の実行
            validate_result = os.system("terraform validate")
            return validate_result == 0

        except Exception as e:
            logger.error(f"Error validating Terraform code: {e}")
            return False

        finally:
            os.chdir(current_dir)

    def generate_documentation(self,
                             output_dir: str,
                             doc_format: str = 'markdown') -> str:
        """ドキュメントの生成"""
        try:
            docs = {}
            
            # 変数の説明を抽出
            variables_file = os.path.join(output_dir, 'variables.tf')
            if os.path.exists(variables_file):
                with open(variables_file, 'r') as f:
                    variables = hcl2.load(f)
                    docs['Variables'] = {
                        var_name: var_config.get('description', '')
                        for var in variables
                        if 'variable' in var
                        for var_name, var_config in var['variable'].items()
                    }

            # 出力の説明を抽出
            outputs_file = os.path.join(output_dir, 'outputs.tf')
            if os.path.exists(outputs_file):
                with open(outputs_file, 'r') as f:
                    outputs = hcl2.load(f)
                    docs['Outputs'] = {
                        out_name: out_config.get('description', '')
                        for out in outputs
                        if 'output' in out
                        for out_name, out_config in out['output'].items()
                    }

            # リソースの一覧を抽出
            main_file = os.path.join(output_dir, 'main.tf')
            if os.path.exists(main_file):
                with open(main_file, 'r') as f:
                    main = hcl2.load(f)
                    docs['Resources'] = [
                        resource_type
                        for res in main
                        if 'resource' in res
                        for resource_type in res['resource'].keys()
                    ]

            # ドキュメントの生成
            if doc_format == 'markdown':
                doc_path = os.path.join(output_dir, 'README.md')
                with open(doc_path, 'w') as f:
                    f.write("# Terraform Infrastructure Documentation\n\n")
                    
                    # 変数のドキュメント
                    f.write("## Variables\n\n")
                    for var_name, description in docs.get('Variables', {}).items():
                        f.write(f"- **{var_name}**: {description}\n")
                    
                    # リソースのドキュメント
                    f.write("\n## Resources\n\n")
                    for resource in docs.get('Resources', []):
                        f.write(f"- {resource}\n")
                    
                    # 出力のドキュメント
                    f.write("\n## Outputs\n\n")
                    for out_name, description in docs.get('Outputs', {}).items():
                        f.write(f"- **{out_name}**: {description}\n")

            return doc_path

        except Exception as e:
            logger.error(f"Error generating documentation: {e}")
            raise
