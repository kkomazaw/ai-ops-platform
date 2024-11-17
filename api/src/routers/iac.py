@router.post("/solutions/{solution_id}/generate-iac")
async def generate_iac_code(
   solution_id: str,
   iac_service: IaCGeneratorService = Depends(get_iac_service)
):
   solution = await get_solution(solution_id)
   iac_code = await iac_service.generate_iac(solution)
   return iac_code