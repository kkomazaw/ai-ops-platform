const RootCauseAnalysis: React.FC<{ incidentId: string }> = ({ incidentId }) => {
    const { data, isLoading } = useQuery(
      ['root-cause', incidentId],
      () => analyzeIncident(incidentId)
    );
  
    return (
      <Card>
        <CardHeader title="Root Cause Analysis" />
        <CardContent>
          {isLoading ? (
            <CircularProgress />
          ) : (
            <>
              <RootCauseDetails rootCause={data.rootCause} />
              <EvidenceList evidence={data.evidence} />
              <RecommendedActions actions={data.recommendedActions} />
            </>
          )}
        </CardContent>
      </Card>
    );
  };