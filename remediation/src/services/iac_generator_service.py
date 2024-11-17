import yaml
import json
from typing import Dict, List, Union, Optional
from enum import Enum
import logging
import os
from jinja2 import Template
from dataclasses import dataclass
import hcl2

# ログ設定
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class IaCType(Enum):
    ANSIBLE = "ansible"
    TERRAFORM = "terraform"

@dataclass
class IaCTemplate:
    type: IaCType
    template: str
    variables: Dict
    dependencies: List[str]

class IaCGenerator:
    def __init__(self):
        """IaC生成システムの初期化"""
        self.ansible_templates = self._load_ansible_templates()
        self.terraform_templates = self._load_terraform_templates()
        self.template_engine = TemplateEngine()
        self.validator = IaCValidator()

    def generate_iac(self, solution: Dict) -> Dict[str, str]:
        """
        解決策からIaCコードを生成

        Parameters:
        solution (Dict): 解決策の詳細情報

        Returns:
        Dict[str, str]: 生成されたIaCコード
        """
        try:
            # 解決策の分析
            iac_type = self._determine_iac_type(solution)
            
            if iac_type == IaCType.ANSIBLE:
                return self._generate_ansible(solution)
            else:
                return self._generate_terraform(solution)
                
        except Exception as e:
            logger.error(f"IaC生成エラー: {e}")
            raise

    def _determine_iac_type(self, solution: Dict) -> IaCType:
        """解決策の内容に基づいて適切なIaCタイプを決定"""
        # インフラ構成変更を含む場合はTerraform
        if any(step.get('type') == 'infrastructure' for step in solution['steps']):
            return IaCType.TERRAFORM
        # それ以外の場合はAnsible
        return IaCType.ANSIBLE

    def _generate_ansible(self, solution: Dict) -> Dict[str, str]:
        """Ansibleプレイブックの生成"""
        playbooks = {}
        
        # メインプレイブック
        main_playbook = {
            'name': f"Execute solution: {solution['title']}",
            'hosts': "{{ target_hosts }}",
            'become': True,
            'tasks': []
        }

        # 前提条件チェック
        pre_check_tasks = self._generate_pre_check_tasks(solution)
        if pre_check_tasks:
            playbooks['pre_checks.yml'] = yaml.dump(pre_check_tasks)

        # メインタスク
        for step in solution['steps']:
            task = self._convert_step_to_ansible_task(step)
            if task:
                main_playbook['tasks'].extend(task)

        # ロールバックタスク
        rollback_tasks = self._generate_rollback_tasks(solution)
        if rollback_tasks:
            playbooks['rollback.yml'] = yaml.dump(rollback_tasks)

        playbooks['main.yml'] = yaml.dump(main_playbook)
        
        # 検証タスク
        validation_tasks = self._generate_validation_tasks(solution)
        if validation_tasks:
            playbooks['validate.yml'] = yaml.dump(validation_tasks)

        return playbooks

    def _generate_terraform(self, solution: Dict) -> Dict[str, str]:
        """Terraformコードの生成"""
        terraform_code = {}
        
        # プロバイダー設定
        providers = self._generate_terraform_providers(solution)
        terraform_code['providers.tf'] = providers

        # メインのリソース定義
        main_resources = self._generate_terraform_resources(solution)
        terraform_code['main.tf'] = main_resources

        # 変数定義
        variables = self._generate_terraform_variables(solution)
        terraform_code['variables.tf'] = variables

        # 出力定義
        outputs = self._generate_terraform_outputs(solution)
        terraform_code['outputs.tf'] = outputs

        return terraform_code

    def _convert_step_to_ansible_task(self, step: Dict) -> List[Dict]:
        """解決ステップをAnsibleタスクに変換"""
        if step['type'] == 'database':
            return self._generate_database_tasks(step)
        elif step['type'] == 'service':
            return self._generate_service_tasks(step)
        elif step['type'] == 'configuration':
            return self._generate_configuration_tasks(step)
        elif step['type'] == 'monitoring':
            return self._generate_monitoring_tasks(step)
        return []

    def _generate_database_tasks(self, step: Dict) -> List[Dict]:
        """データベース関連タスクの生成"""
        tasks = []
        
        if step['action'] == 'optimize':
            tasks.append({
                'name': 'Optimize database parameters',
                'mysql_variables':
                    'variable_name': "{{ item.name }}",
                    'value': "{{ item.value }}"
                'with_items': step['parameters']
            })
            
        elif step['action'] == 'backup':
            tasks.append({
                'name': 'Create database backup',
                'mysql_db':
                    'name': 'all',
                    'state': 'dump',
                    'target': '/backup/db_{{ ansible_date_time.iso8601 }}.sql'
            })
            
        return tasks

    def _generate_service_tasks(self, step: Dict) -> List[Dict]:
        """サービス管理タスクの生成"""
        tasks = []
        
        if step['action'] == 'restart':
            tasks.append({
                'name': f"Restart {step['service_name']}",
                'systemd':
                    'name': step['service_name'],
                    'state': 'restarted'
            })
            
        elif step['action'] == 'scale':
            tasks.append({
                'name': f"Scale {step['service_name']}",
                'command': f"kubectl scale deployment {step['service_name']} --replicas={step['replicas']}"
            })
            
        return tasks

    def _generate_terraform_resources(self, solution: Dict) -> str:
        """Terraformリソース定義の生成"""
        resources = []
        
        for step in solution['steps']:
            if step['type'] == 'infrastructure':
                resource = self._convert_step_to_terraform_resource(step)
                if resource:
                    resources.append(resource)

        return '\n'.join(resources)

    def _convert_step_to_terraform_resource(self, step: Dict) -> str:
        """解決ステップをTerraformリソースに変換"""
        if step['action'] == 'scale_instance':
            return f"""
resource "aws_autoscaling_group" "{step['name']}" {{
  name                = "{step['name']}"
  max_size           = {step['max_size']}
  min_size           = {step['min_size']}
  desired_capacity   = {step['desired_capacity']}
  vpc_zone_identifier = var.subnet_ids
  
  launch_template {{
    id = var.launch_template_id
    version = "$Latest"
  }}
  
  tag {{
    key                 = "Environment"
    value               = var.environment
    propagate_at_launch = true
  }}
}}
"""
        elif step['action'] == 'modify_database':
            return f"""
resource "aws_db_instance" "{step['name']}" {{
  identifier           = "{step['name']}"
  instance_class      = "{step['instance_class']}"
  allocated_storage   = {step['storage']}
  
  backup_retention_period = {step['backup_retention']}
  multi_az             = {str(step['multi_az']).lower()}
  
  parameter_group_name = aws_db_parameter_group.{step['name']}_params.name
}}

resource "aws_db_parameter_group" "{step['name']}_params" {{
  family = "mysql8.0"
  
  parameter {{
    name  = "max_connections"
    value = "{step['max_connections']}"
  }}
}}
"""

class TemplateEngine:
    def __init__(self):
        """テンプレートエンジンの初期化"""
        self.ansible_templates = {}
        self.terraform_templates = {}
        self._load_templates()

    def _load_templates(self):
        """テンプレートの読み込み"""
        # Ansibleテンプレート
        self.ansible_templates = {
            'database_optimization': Template('''
- name: Optimize database configuration
  mysql_variables:
    variable_name: "{{ item.name }}"
    value: "{{ item.value }}"
  with_items:
    {{ parameters | to_yaml }}
'''),
            'service_scaling': Template('''
- name: Scale service
  kubernetes.core.k8s_scale:
    api_version: apps/v1
    kind: Deployment
    name: "{{ service_name }}"
    replicas: "{{ replicas }}"
    namespace: "{{ namespace }}"
''')
        }

        # Terraformテンプレート
        self.terraform_templates = {
            'auto_scaling': Template('''
resource "aws_autoscaling_group" "{{ name }}" {
  name                = "{{ name }}"
  max_size           = {{ max_size }}
  min_size           = {{ min_size }}
  desired_capacity   = {{ desired_capacity }}
  vpc_zone_identifier = var.subnet_ids
  
  launch_template {
    id = var.launch_template_id
    version = "$Latest"
  }
}
'''),
            'rds_modification': Template('''
resource "aws_db_instance" "{{ name }}" {
  identifier           = "{{ name }}"
  instance_class      = "{{ instance_class }}"
  allocated_storage   = {{ storage }}
  
  parameter_group_name = aws_db_parameter_group.{{ name }}_params.name
}
''')
        }

class IaCValidator:
    def validate_ansible(self, playbook: str) -> bool:
        """Ansibleプレイブックの検証"""
        try:
            # YAMLとしての妥当性チェック
            yaml.safe_load(playbook)
            
            # Ansibleの構文チェック
            result = os.system(f"ansible-playbook --syntax-check {playbook}")
            return result == 0
            
        except Exception as e:
            logger.error(f"Ansible検証エラー: {e}")
            return False

    def validate_terraform(self, terraform_code: str) -> bool:
        """Terraformコードの検証"""
        try:
            # HCLとしての妥当性チェック
            hcl2.loads(terraform_code)
            
            # Terraformの構文チェック
            result = os.system(f"terraform validate")
            return result == 0
            
        except Exception as e:
            logger.error(f"Terraform検証エラー: {e}")
            return False

def main():
    # 使用例
    solution = {
        'title': 'データベース最適化',
        'steps': [
            {
                'type': 'database',
                'action': 'optimize',
                'parameters': [
                    {'name': 'max_connections', 'value': '1000'},
                    {'name': 'innodb_buffer_pool_size', 'value': '4G'}
                ]
            },
            {
                'type': 'service',
                'action': 'restart',
                'service_name': 'mysql'
            }
        ],
        'validation': {
            'metrics': ['connections', 'response_time'],
            'thresholds': {
                'connections': {'max': 800},
                'response_time': {'max': 100}
            }
        }
    }

    # IaC生成
    generator = IaCGenerator()
    iac_code = generator.generate_iac(solution)
    
    # 結果の出力
    for filename, content in iac_code.items():
        print(f"\n=== {filename} ===")
        print(content)

if __name__ == "__main__":
    main()