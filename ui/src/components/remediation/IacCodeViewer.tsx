const IaCCodeViewer: React.FC<{ solutionId: string }> = ({ solutionId }) => {
    const { data } = useQuery(
      ['iac-code', solutionId], 
      () => generateIaCCode(solutionId)
    );
   
    return (
      <Card>
        <CardHeader 
          title="Infrastructure as Code"
          action={
            <Button
              onClick={() => downloadCode(data.code)}
              startIcon={<DownloadIcon />}
            >
              Download
            </Button>
          }
        />
        <CardContent>
          <CodeEditor
            value={data?.code}
            language={data?.metadata.generator_type}
            validation={data?.validation_result}
          />
        </CardContent>
      </Card>
    );
   };