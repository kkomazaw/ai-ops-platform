import yaml
from typing import List, Dict, Optional, Union
import logging
from dataclasses import dataclass
from datetime import datetime
import os
import jinja2
from pathlib import Path
import json

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class AnsibleTask:
    """Ansibleタスクの定義"""
    name: str
    module: str
    args: Dict
    register: Optional[str] = None
    when: Optional[str] = None
    tags: Optional[List[str]] = None
    ignore_errors: bool = False

@dataclass
class AnsiblePlaybook:
    """Ansibleプレイブックの定義"""
    name: str
    hosts: str
    tasks: List[AnsibleTask]
    vars: Optional[Dict] = None
    become: bool = True
    gather_facts: bool = True
    handlers: Optional[List[Dict]] = None

class AnsibleGenerator:
    def __init__(self, template_dir: Optional[str] = None):
        """
        Ansible Playbook生成クラス

        Args:
            template_dir (Optional[str]): テンプレートディレクトリのパス
        """
        self.template_dir = template_dir or "templates"
        self.jinja_env = self._setup_jinja_env()
        self.task_templates = self._load_task_templates()

    def _setup_jinja_env(self) -> jinja2.Environment:
        """Jinja2環境のセットアップ"""
        return jinja2.Environment(
            loader=jinja2.FileSystemLoader(self.template_dir),
            trim_blocks=True,
            lstrip_blocks=True
        )

    def _load_task_templates(self) -> Dict:
        """タスクテンプレートの読み込み"""
        return {
            "service_restart": {
                "name": "Restart service",
                "module": "systemd",
                "args": {
                    "name": "{{ service_name }}",
                    "state": "restarted"
                }
            },
            "file_backup": {
                "name": "Backup file",
                "module": "copy",
                "args": {
                    "src": "{{ src_path }}",
                    "dest": "{{ dest_path }}.{{ timestamp }}",
                    "remote_src": True
                }
            },
            "package_install": {
                "name": "Install package",
                "module": "package",
                "args": {
                    "name": "{{ package_name }}",
                    "state": "present"
                }
            },
            "config_update": {
                "name": "Update configuration",
                "module": "template",
                "args": {
                    "src": "{{ template_path }}",
                    "dest": "{{ config_path }}",
                    "backup": True
                }
            }
        }

    def generate_playbook(self, 
                         solution_data: Dict,
                         output_dir: str) -> str:
        """
        解決策からAnsible Playbookを生成

        Args:
            solution_data (Dict): 解決策の情報
            output_dir (str): 出力ディレクトリ

        Returns:
            str: 生成されたPlaybookのパス
        """
        try:
            # プレイブックの基本情報を設定
            playbook = AnsiblePlaybook(
                name=f"Execute solution: {solution_data['title']}",
                hosts=solution_data.get('target_hosts', 'all'),
                tasks=[],
                vars=self._prepare_variables(solution_data)
            )

            # タスクの生成
            for step in solution_data['steps']:
                task = self._generate_task(step)
                if task:
                    playbook.tasks.append(task)

            # ハンドラーの追加
            playbook.handlers = self._generate_handlers(solution_data)

            # プレイブックの保存
            output_path = self._save_playbook(playbook, output_dir)
            
            # インベントリファイルの生成
            self._generate_inventory(solution_data, output_dir)
            
            return output_path

        except Exception as e:
            logger.error(f"Error generating playbook: {e}")
            raise

    def _generate_task(self, step: Dict) -> Optional[AnsibleTask]:
        """個別タスクの生成"""
        try:
            task_type = step.get('type', '').lower()
            
            if task_type == 'service':
                return self._generate_service_task(step)
            elif task_type == 'file':
                return self._generate_file_task(step)
            elif task_type == 'package':
                return self._generate_package_task(step)
            elif task_type == 'config':
                return self._generate_config_task(step)
            elif task_type == 'command':
                return self._generate_command_task(step)
            else:
                logger.warning(f"Unknown task type: {task_type}")
                return None

        except Exception as e:
            logger.error(f"Error generating task: {e}")
            return None

    def _generate_service_task(self, step: Dict) -> AnsibleTask:
        """サービス関連タスクの生成"""
        template = self.task_templates['service_restart']
        return AnsibleTask(
            name=template['name'].replace('{{ service_name }}', step['service_name']),
            module=template['module'],
            args={
                'name': step['service_name'],
                'state': step.get('state', 'restarted')
            },
            tags=['service']
        )

    def _generate_file_task(self, step: Dict) -> AnsibleTask:
        """ファイル操作タスクの生成"""
        if step.get('operation') == 'backup':
            template = self.task_templates['file_backup']
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            return AnsibleTask(
                name=f"Backup {step['path']}",
                module=template['module'],
                args={
                    'src': step['path'],
                    'dest': f"{step['path']}.bak.{timestamp}",
                    'remote_src': True
                },
                tags=['file', 'backup']
            )
        else:
            return AnsibleTask(
                name=f"Manage file {step['path']}",
                module='file',
                args={
                    'path': step['path'],
                    'state': step.get('state', 'present'),
                    'mode': step.get('mode', '0644')
                },
                tags=['file']
            )

    def _generate_package_task(self, step: Dict) -> AnsibleTask:
        """パッケージ管理タスクの生成"""
        template = self.task_templates['package_install']
        return AnsibleTask(
            name=f"Manage package {step['package_name']}",
            module=template['module'],
            args={
                'name': step['package_name'],
                'state': step.get('state', 'present')
            },
            tags=['package']
        )

    def _generate_config_task(self, step: Dict) -> AnsibleTask:
        """設定ファイル更新タスクの生成"""
        template = self.task_templates['config_update']
        return AnsibleTask(
            name=f"Update config {step['path']}",
            module=template['module'],
            args={
                'src': step.get('template', f"{step['path']}.j2"),
                'dest': step['path'],
                'backup': True,
                'validate': step.get('validate_cmd')
            },
            tags=['config']
        )

    def _generate_command_task(self, step: Dict) -> AnsibleTask:
        """コマンド実行タスクの生成"""
        return AnsibleTask(
            name=f"Execute command: {step['command']}",
            module='command',
            args={'cmd': step['command']},
            register='command_result',
            when=step.get('condition'),
            ignore_errors=step.get('ignore_errors', False),
            tags=['command']
        )

    def _generate_handlers(self, solution_data: Dict) -> List[Dict]:
        """ハンドラーの生成"""
        handlers = []
        
        # サービス再起動ハンドラー
        if 'services' in solution_data:
            for service in solution_data['services']:
                handlers.append({
                    'name': f"restart {service}",
                    'systemd': {
                        'name': service,
                        'state': 'restarted'
                    }
                })

        return handlers

    def _prepare_variables(self, solution_data: Dict) -> Dict:
        """変数の準備"""
        vars_dict = {
            'ansible_become': True,
            'ansible_become_method': 'sudo',
            'backup_dir': '/var/backup',
            'timestamp': "{{ ansible_date_time.iso8601 }}"
        }
        
        # ソリューション固有の変数を追加
        if 'variables' in solution_data:
            vars_dict.update(solution_data['variables'])

        return vars_dict

    def _save_playbook(self, 
                      playbook: AnsiblePlaybook,
                      output_dir: str) -> str:
        """プレイブックの保存"""
        try:
            # 出力ディレクトリの作成
            os.makedirs(output_dir, exist_ok=True)
            
            # プレイブックの構造を作成
            playbook_data = {
                'name': playbook.name,
                'hosts': playbook.hosts,
                'become': playbook.become,
                'gather_facts': playbook.gather_facts,
                'vars': playbook.vars,
                'tasks': [
                    {
                        'name': task.name,
                        'module': task.module,
                        **task.args,
                        **(
                            {'register': task.register} 
                            if task.register else {}
                        ),
                        **(
                            {'when': task.when} 
                            if task.when else {}
                        ),
                        **(
                            {'tags': task.tags} 
                            if task.tags else {}
                        ),
                        **(
                            {'ignore_errors': task.ignore_errors} 
                            if task.ignore_errors else {}
                        )
                    }
                    for task in playbook.tasks
                ],
                'handlers': playbook.handlers
            }
            
            # ファイルに保存
            output_path = os.path.join(output_dir, 'playbook.yml')
            with open(output_path, 'w') as f:
                yaml.dump([playbook_data], f, default_flow_style=False)
            
            logger.info(f"Playbook saved to {output_path}")
            return output_path

        except Exception as e:
            logger.error(f"Error saving playbook: {e}")
            raise

    def _generate_inventory(self, 
                          solution_data: Dict,
                          output_dir: str) -> None:
        """インベントリファイルの生成"""
        try:
            inventory = {
                'all': {
                    'hosts': {},
                    'children': {}
                }
            }
            
            # ホストグループの設定
            if 'host_groups' in solution_data:
                for group, hosts in solution_data['host_groups'].items():
                    inventory['all']['children'][group] = {
                        'hosts': {host: {} for host in hosts}
                    }
            
            # インベントリファイルの保存
            inventory_path = os.path.join(output_dir, 'inventory.yml')
            with open(inventory_path, 'w') as f:
                yaml.dump(inventory, f, default_flow_style=False)
            
            logger.info(f"Inventory saved to {inventory_path}")

        except Exception as e:
            logger.error(f"Error generating inventory: {e}")
            raise

    def validate_playbook(self, playbook_path: str) -> bool:
        """プレイブックの検証"""
        try:
            # ansible-playbook --syntax-check の実行
            result = os.system(f"ansible-playbook --syntax-check {playbook_path}")
            return result == 0
        except Exception as e:
            logger.error(f"Error validating playbook: {e}")
            return False

    def generate_documentation(self,
                             playbook_path: str,
                             output_dir: str) -> str:
        """プレイブックのドキュメント生成"""
        try:
            # プレイブックの読み込み
            with open(playbook_path, 'r') as f:
                playbook_data = yaml.safe_load(f)

            # ドキュメントの生成
            doc = {
                'Playbook Documentation': {
                    'Name': playbook_data[0]['name'],
                    'Target Hosts': playbook_data[0]['hosts'],
                    'Variables': playbook_data[0].get('vars', {}),
                    'Tasks': [
                        {
                            'Name': task['name'],
                            'Module': task.get('module', 'command'),
                            'Description': task.get('description', 'No description provided')
                        }
                        for task in playbook_data[0]['tasks']
                    ],
                    'Handlers': playbook_data[0].get('handlers', []),
                    'Tags': list(set(
                        tag
                        for task in playbook_data[0]['tasks']
                        if 'tags' in task
                        for tag in task['tags']
                    ))
                }
            }

            # ドキュメントの保存
            doc_path = os.path.join(output_dir, 'playbook_documentation.md')
            with open(doc_path, 'w') as f:
                f.write("# Ansible Playbook Documentation\n\n")
                json.dump(doc, f, indent=2)

            return doc_path

        except Exception as e:
            logger.error(f"Error generating documentation: {e}")
            raise
