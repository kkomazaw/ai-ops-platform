from .models import SolutionGenerator, IaCGenerator, ValidationService

class RemediationService:
   def __init__(self):
       self.solution_generator = SolutionGenerator()
       self.iac_generator = IaCGenerator()
       self.validator = ValidationService()
   
   async def generate_solution(self, 
                             incident_id: str,
                             analysis_result: Dict) -> Dict:
       # 解決策の生成
       solutions = self.solution_generator.generate_solutions(
           root_cause=analysis_result
       )
       
       # IaCコードの生成
       for solution in solutions:
           if solution.automation_possible:
               if solution.type == 'ansible':
                   solution.code = self.iac_generator.generate_ansible(solution)
               elif solution.type == 'terraform': 
                   solution.code = self.iac_generator.generate_terraform(solution)
                   
               # コードの検証
               solution.validation_result = self.validator.validate_code(
                   solution.code,
                   solution.type
               )
               
       return solutions