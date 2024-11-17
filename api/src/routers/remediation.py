@router.post("/incidents/{incident_id}/remediate")
async def generate_remediation(
   incident_id: str,
   remediation_service: RemediationService = Depends(get_remediation_service)
):
   analysis = await get_incident_analysis(incident_id)
   
   solutions = await remediation_service.generate_solution(
       incident_id=incident_id,
       analysis_result=analysis
   )
   
   return solutions