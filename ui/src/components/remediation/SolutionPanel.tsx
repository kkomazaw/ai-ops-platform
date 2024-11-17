const SolutionPanel: React.FC<{ incidentId: string }> = ({ incidentId }) => {
    const { data } = useQuery(
      ['solutions', incidentId],
      () => generateRemediation(incidentId)
    );
   
    return (
      <Card>
        <CardHeader title="Recommended Solutions" />
        <CardContent>
          {data?.solutions.map(solution => (
            <SolutionCard
              key={solution.id}
              solution={solution}
              onApply={() => applySolution(solution)}
            />
          ))}
        </CardContent>
      </Card>
    );
   };