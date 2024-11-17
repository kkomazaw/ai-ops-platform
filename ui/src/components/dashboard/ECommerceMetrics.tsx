// ui/src/components/dashboard/ECommerceMetrics.tsx
const ECommerceMetrics: React.FC = () => {
    const { data } = useQuery('ecommerce-metrics', fetchECommerceMetrics);
    
    return (
      <MetricsPanel title="ECサイトメトリクス">
        <LineChart data={data.timeseriesData} />
        <AnomalyList anomalies={data.detectedAnomalies} />
      </MetricsPanel>
    );
  };