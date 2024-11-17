from .generators import AnsibleGenerator, TerraformGenerator 
from .validators import AnsibleValidator, TerraformValidator

class IaCGeneratorService:
   def __init__(self):
       self.ansible_generator = AnsibleGenerator()
       self.terraform_generator = TerraformGenerator()
       self.ansible_validator = AnsibleValidator() 
       self.terraform_validator = TerraformValidator()

   async def generate_iac(self, solution: Dict) -> Dict:
       # インフラ定義の生成
       if solution['type'] == 'ansible':
           code = self.ansible_generator.generate_ansible_code(
               solution['infrastructure_data']
           )
           validation = self.ansible_validator.validate_playbook(code)
       else:
           code = self.terraform_generator.generate_terraform_code(
               solution['infrastructure_data']
           )
           validation = self.terraform_validator.validate_terraform_code(code)

       return {
           'code': code,
           'validation_result': validation,
           'metadata': {
               'generator_type': solution['type'],
               'template_used': solution.get('template'),
               'generated_at': datetime.utcnow()
           }
       }